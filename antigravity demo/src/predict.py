"""Inference module with Test-Time Augmentation (TTA) for Butterfly Species Classification."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.config import (
    DEVICE,
    MODEL_NAME,
    OUTPUTS_DIR,
    SAMPLE_PREDICTIONS_DIR,
    SAVED_MODEL_DIR,
)


def load_model_and_processor(model_dir: Optional[Union[str, Path]] = None):
    """Load model and processor from fine-tuned outputs directory if present, else Hugging Face."""
    if model_dir is not None and Path(model_dir).exists():
        path = str(model_dir)
        print(f"[INFERENCE] Loading model from specified path: {path}")
    elif SAVED_MODEL_DIR.exists() and (SAVED_MODEL_DIR / "config.json").exists():
        path = str(SAVED_MODEL_DIR)
        print(f"[INFERENCE] Loading fine-tuned model from: {path}")
    else:
        path = MODEL_NAME
        print(f"[INFERENCE] Loading pretrained model from Hugging Face: {path}")

    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForImageClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    return model, processor


def generate_tta_crops(image: Image.Image) -> List[Image.Image]:
    """Generate 6 test-time augmentation (TTA) variants of an image:
    1. Original image
    2. Mirrored copy
    3. Centre crop 80%
    4. Centre crop 60%
    5. Top-left crop 80%
    6. Bottom-right crop 80%
    """
    img = image.convert("RGB")
    w, h = img.size

    # 1. Original
    orig = img

    # 2. Mirrored
    mirrored = ImageOps.mirror(img)

    # 3. Centre 80%
    w80, h80 = int(w * 0.8), int(h * 0.8)
    left80 = (w - w80) // 2
    top80 = (h - h80) // 2
    centre_80 = img.crop((left80, top80, left80 + w80, top80 + h80))

    # 4. Centre 60%
    w60, h60 = int(w * 0.6), int(h * 0.6)
    left60 = (w - w60) // 2
    top60 = (h - h60) // 2
    centre_60 = img.crop((left60, top60, left60 + w60, top60 + h60))

    # 5. Top-left 80%
    tl_80 = img.crop((0, 0, w80, h80))

    # 6. Bottom-right 80%
    br_80 = img.crop((w - w80, h - h80, w, h))

    return [orig, mirrored, centre_80, centre_60, tl_80, br_80]


def predict(
    image_input: Union[str, Path, Image.Image],
    model_dir: Optional[Union[str, Path]] = None,
    model: Optional[torch.nn.Module] = None,
    processor: Optional[AutoImageProcessor] = None,
    save_plot: bool = True,
    output_filename: Optional[str] = None,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Predict butterfly species with Test-Time Augmentation (TTA).
    
    Args:
        image_input: File path or PIL Image.
        model_dir: Optional directory/HF repo containing model and processor.
        model: Optional loaded PyTorch model.
        processor: Optional loaded image processor.
        save_plot: Whether to save visualization plot to outputs.
        output_filename: Optional custom filename for saved plot.

    Returns:
        Tuple of (predicted_species, confidence, top_5_list)
        where top_5_list is a list of tuples: [(species_name, probability), ...]
    """
    if model is None or processor is None:
        model, processor = load_model_and_processor(model_dir)

    # Load PIL image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")
        image = Image.open(img_path)
        base_name = img_path.stem
    else:
        image = image_input
        base_name = "image"

    # Generate 6 TTA crops
    crops = generate_tta_crops(image)

    # Compute softmax probabilities across all crops
    crop_probs = []
    with torch.no_grad():
        for crop in crops:
            inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()
            crop_probs.append(probs)

    # Average probabilities across crops
    avg_probs = torch.stack(crop_probs).mean(dim=0)

    # Extract Top-5 predictions
    top5_scores, top5_indices = torch.topk(avg_probs, k=5)
    top5 = []
    for score, idx in zip(top5_scores, top5_indices):
        species = model.config.id2label[idx.item()]
        top5.append((species, float(score.item())))

    predicted_species = top5[0][0]
    confidence = top5[0][1]

    # Print results
    print("\n" + "=" * 55)
    print("           BUTTERFLY SPECIES PREDICTION           ")
    print("=" * 55)
    print(f" Predicted Species : {predicted_species}")
    print(f" Confidence        : {confidence * 100:.2f}%")
    print("-" * 55)
    print(" Top-5 Predictions:")
    for rank, (spec, prob) in enumerate(top5, 1):
        bar = "#" * int(prob * 30)
        print(f"  {rank}. {spec:30s} {prob * 100:6.2f}%  {bar}")
    print("=" * 55)

    # Low confidence warning (< 50%)
    if confidence < 0.50:
        print(
            "\n[WARNING] Low confidence prediction (< 50.00%).\n"
            "The photo may be unclear, blurred, or the butterfly species\n"
            "may not be among the 100 species in the dataset."
        )

    # Visualization plot
    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})

        # Left: Original Image
        axes[0].imshow(image.convert("RGB"))
        axes[0].axis("off")
        title_color = "#10b981" if confidence >= 0.50 else "#f59e0b"
        axes[0].set_title(
            f"Prediction: {predicted_species}\nConfidence: {confidence * 100:.1f}%",
            fontsize=12,
            fontweight="bold",
            color=title_color,
            pad=10,
        )

        # Right: Horizontal Bar Chart of Top 5
        chart_species = [item[0] for item in reversed(top5)]
        chart_probs = [item[1] * 100 for item in reversed(top5)]

        colors = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
        if confidence < 0.50:
            colors[-1] = "#f59e0b"

        bars = axes[1].barh(chart_species, chart_probs, color=colors, height=0.6)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
        axes[1].set_title("Top-5 Candidates (Test-Time Augmentation)", fontsize=12, fontweight="bold")
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            axes[1].text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()

        out_name = output_filename or f"{base_name}_prediction.png"
        save_path = SAMPLE_PREDICTIONS_DIR / out_name
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OUTPUT] Visualization saved to: {save_path}\n")

    return predicted_species, confidence, top5


def main():
    parser = argparse.ArgumentParser(description="Predict Butterfly Species with Test-Time Augmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to butterfly image file")
    parser.add_argument("--model-dir", type=str, default=None, help="Optional custom model directory")
    args = parser.parse_args()

    predict(image_input=args.image, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
