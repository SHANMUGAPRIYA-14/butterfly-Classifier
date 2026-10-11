"""Training and evaluation pipeline for Butterfly Species Classification."""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from src.config import (
    DEVICE,
    FP16,
    LOSS_CURVE_PATH,
    METRICS_PATH,
    MODEL_NAME,
    MODES,
    NUM_WORKERS,
    OUTPUTS_DIR,
    REPORT_PATH,
    SAVED_MODEL_DIR,
)
from src.data import (
    download_dataset,
    get_image_processor,
    load_butterfly_dataset,
    prepare_dataset_transforms,
)


def get_training_args(
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float = 0.0,
    save_total_limit: int = 1,
) -> TrainingArguments:
    """Construct TrainingArguments handling both eval_strategy and evaluation_strategy."""
    args_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "save_total_limit": save_total_limit,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": FP16,
        "dataloader_num_workers": NUM_WORKERS,
        "remove_unused_columns": False,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
    }

    # Inspect parameter support for transformers versions
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        args_dict["eval_strategy"] = "epoch"
    else:
        args_dict["evaluation_strategy"] = "epoch"

    return TrainingArguments(**args_dict)


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze all layers except the classification head."""
    print("[MODEL] Freezing backbone layers; training only classification head...")
    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model: torch.nn.Module) -> None:
    """Unfreeze all layers for full fine-tuning."""
    print("[MODEL] Unfreezing all backbone layers for full fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def plot_loss_curve(log_history: List[Dict[str, Any]], save_path: Path) -> None:
    """Generate and save training loss curve from Trainer log history."""
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry["step"])
        if "eval_loss" in entry and "step" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(entry["step"])

    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="Training Loss", color="#3b82f6", lw=2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#ef4444", lw=2, linestyle="--")

    plt.title("Vision Transformer Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[EVAL] Loss curve saved to {save_path}")


def evaluate_test_set(
    trainer: Trainer,
    test_dataset,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    label_names: List[str],
) -> Dict[str, Any]:
    """Evaluate on the test set and generate comprehensive metrics and report."""
    print("\n[EVAL] Evaluating model on the test set...")
    predictions_output = trainer.predict(test_dataset)
    y_preds = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids

    acc = float(accuracy_score(y_true, y_preds))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "weighted_precision": round(float(p), 4),
        "weighted_recall": round(float(r), 4),
        "weighted_f1": round(float(f1), 4),
        "number_of_test_images": int(len(y_true)),
        "number_of_species": int(len(label_names)),
    }

    # Save metrics JSON
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[EVAL] Metrics saved to {METRICS_PATH}")

    # Generate and save classification report
    unique_labels = sorted(list(set(y_true) | set(y_preds)))
    target_names = [label_names[i] if i < len(label_names) else f"Class_{i}" for i in unique_labels]
    report = classification_report(
        y_true, y_preds, labels=unique_labels, target_names=target_names, zero_division=0
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[EVAL] Per-species classification report saved to {REPORT_PATH}")

    # Print summary metrics to console
    print("\n" + "=" * 50)
    print("           TEST SET EVALUATION RESULTS           ")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print("=" * 50 + "\n")

    return metrics


def run_instant_mode() -> None:
    """INSTANT mode: Load pretrained model and verify without training or dataset download."""
    print("=" * 60)
    print("Running in INSTANT mode:")
    print("  - Verifying pretrained Vision Transformer loading")
    print(f"  - Model: {MODEL_NAME}")
    print("=" * 60)

    print(f"[LOAD] Loading processor for {MODEL_NAME}...")
    processor = get_image_processor(MODEL_NAME)
    print(f"[LOAD] Loading model for {MODEL_NAME}...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    num_classes = len(model.config.id2label)
    print(f"[SUCCESS] Model successfully loaded on {DEVICE.upper()}!")
    print(f"[INFO] Architecture: Vision Transformer (ViT)")
    print(f"[INFO] Number of butterfly species: {num_classes}")
    print(f"[INFO] Sample classes: {list(model.config.id2label.values())[:5]} ...")

    # Save to outputs/saved_model so predict.py and app.py can load it locally
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVED_MODEL_DIR)
    processor.save_pretrained(SAVED_MODEL_DIR)
    print(f"[SAVE] Pretrained model and processor cached to {SAVED_MODEL_DIR}")
    print("[SUCCESS] INSTANT verification completed successfully!")


def run_training(mode: str) -> None:
    """Run training in FAST or FULL mode."""
    cfg = MODES[mode]
    print("=" * 60)
    print(f"Running in {mode} mode:")
    print(f"  - {cfg['description']}")
    print(f"  - Device: {DEVICE.upper()}")
    print("=" * 60)

    # 1. Download dataset
    data_dir = download_dataset()

    # 2. Load dataset splits
    if mode == "FAST":
        dataset = load_butterfly_dataset(
            data_dir,
            max_train_samples=cfg["train_samples"],
            max_val_samples=cfg["val_samples"],
            max_test_samples=cfg["test_samples"],
        )
    else:
        dataset = load_butterfly_dataset(data_dir)

    # 3. Load processor and model
    processor = get_image_processor(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # 4. Prepare dataset transformations
    transformed_dataset, collate_fn = prepare_dataset_transforms(
        dataset, processor, model.config.label2id
    )

    checkpoint_dir = OUTPUTS_DIR / "checkpoints"

    if mode == "FAST":
        # Freeze backbone, train classification layer for 1 epoch
        freeze_backbone(model)
        training_args = get_training_args(
            output_dir=checkpoint_dir,
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            weight_decay=cfg["weight_decay"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=transformed_dataset["train"],
            eval_dataset=transformed_dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        print("\n[TRAIN] Starting FAST training (1 epoch on classification head)...")
        train_result = trainer.train()
        print(f"[TRAIN] FAST training completed. Train loss: {train_result.training_loss:.4f}")

        # Plot loss curve
        plot_loss_curve(trainer.state.log_history, LOSS_CURVE_PATH)

    elif mode == "FULL":
        stage1_cfg = cfg["stage1"]
        stage2_cfg = cfg["stage2"]

        # Stage 1: Train classification head
        freeze_backbone(model)
        s1_args = get_training_args(
            output_dir=checkpoint_dir / "stage1",
            epochs=stage1_cfg["epochs"],
            learning_rate=stage1_cfg["learning_rate"],
            batch_size=stage1_cfg["batch_size"],
            weight_decay=stage1_cfg["weight_decay"],
        )
        trainer_s1 = Trainer(
            model=model,
            args=s1_args,
            train_dataset=transformed_dataset["train"],
            eval_dataset=transformed_dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        print("\n[TRAIN] Starting FULL Stage 1 (1 epoch on classification head)...")
        trainer_s1.train()

        # Stage 2: Unfreeze backbone and fine-tune
        unfreeze_all(model)
        s2_args = get_training_args(
            output_dir=checkpoint_dir / "stage2",
            epochs=stage2_cfg["epochs"],
            learning_rate=stage2_cfg["learning_rate"],
            batch_size=stage2_cfg["batch_size"],
            weight_decay=stage2_cfg["weight_decay"],
        )
        trainer = Trainer(
            model=model,
            args=s2_args,
            train_dataset=transformed_dataset["train"],
            eval_dataset=transformed_dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        print("\n[TRAIN] Starting FULL Stage 2 (2 epochs fine-tuning whole network)...")
        trainer.train()

        combined_logs = trainer_s1.state.log_history + trainer.state.log_history
        plot_loss_curve(combined_logs, LOSS_CURVE_PATH)

    # 5. Evaluate on Test Set
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    evaluate_test_set(trainer, transformed_dataset["test"], model, processor, label_names)

    # 6. Save final model and processor
    print(f"[SAVE] Saving fine-tuned model and processor to {SAVED_MODEL_DIR}...")
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVED_MODEL_DIR)
    processor.save_pretrained(SAVED_MODEL_DIR)
    print(f"[SAVE] Model saved successfully to {SAVED_MODEL_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate Butterfly Vision Transformer Classifier")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["INSTANT", "FAST", "FULL"],
        default="INSTANT",
        help="Execution mode: INSTANT (verify pretrained), FAST (head training on CPU/GPU), FULL (2-stage training)",
    )
    args = parser.parse_args()

    if args.mode == "INSTANT":
        run_instant_mode()
    else:
        run_training(args.mode)


if __name__ == "__main__":
    main()
