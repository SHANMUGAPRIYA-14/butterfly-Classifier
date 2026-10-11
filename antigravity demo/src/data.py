"""Dataset downloading, loading, and preprocessing for Butterfly Species Classification."""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import kagglehub
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor

from src.config import MODEL_NAME, KAGGLE_DATASET, NUM_WORKERS


def download_dataset() -> str:
    """Download Kaggle butterfly dataset using kagglehub.
    
    Returns:
        str: Local directory path containing the downloaded dataset.
    """
    print(f"[DATA] Downloading / locating dataset from Kaggle: {KAGGLE_DATASET}...")
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[DATA] Dataset successfully ready at: {dataset_path}")
    return dataset_path


def get_image_processor(model_name: str = MODEL_NAME) -> AutoImageProcessor:
    """Load pretrained image processor from Hugging Face."""
    return AutoImageProcessor.from_pretrained(model_name)


def load_butterfly_dataset(
    data_dir: str,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DatasetDict:
    """Load imagefolder dataset with train, valid, and test splits.
    
    Args:
        data_dir: Path to directory containing train, valid, and test folders.
        max_train_samples: Optional limit on training samples.
        max_val_samples: Optional limit on validation samples.
        max_test_samples: Optional limit on test samples.
        
    Returns:
        DatasetDict with train, validation, and test splits.
    """
    print(f"[DATA] Loading imagefolder dataset from: {data_dir}...")
    
    # Check directory structure
    path_obj = Path(data_dir)
    # The dataset might have train, valid/validation, test folders
    valid_folder = "valid" if (path_obj / "valid").exists() else "validation"
    
    dataset = load_dataset(
        "imagefolder",
        data_dir=data_dir,
    )
    
    # Standardize split names if valid is called 'validation'
    if "valid" in dataset and "validation" not in dataset:
        dataset["validation"] = dataset.pop("valid")

    # Subsample if requested (e.g. for FAST mode)
    if max_train_samples and "train" in dataset and len(dataset["train"]) > max_train_samples:
        print(f"[DATA] Subsampling training set to {max_train_samples} samples.")
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_train_samples))

    if max_val_samples and "validation" in dataset and len(dataset["validation"]) > max_val_samples:
        print(f"[DATA] Subsampling validation set to {max_val_samples} samples.")
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(max_val_samples))

    if max_test_samples and "test" in dataset and len(dataset["test"]) > max_test_samples:
        print(f"[DATA] Subsampling test set to {max_test_samples} samples.")
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(max_test_samples))

    print(
        f"[DATA] Loaded splits - Train: {len(dataset['train']) if 'train' in dataset else 0}, "
        f"Validation: {len(dataset['validation']) if 'validation' in dataset else 0}, "
        f"Test: {len(dataset['test']) if 'test' in dataset else 0}"
    )
    return dataset


def create_label_mapping(dataset_classes: list, model_label2id: Dict[str, int]) -> Dict[int, int]:
    """Create a mapping from dataset integer class indices to model class indices.
    
    Normalizes class names (uppercasing, trimming) to ensure exact alignment.
    """
    # Normalize model labels
    normalized_model_labels = {
        name.strip().upper(): idx for name, idx in model_label2id.items()
    }
    
    mapping = {}
    for d_idx, d_name in enumerate(dataset_classes):
        clean_name = d_name.strip().upper()
        if clean_name in normalized_model_labels:
            mapping[d_idx] = normalized_model_labels[clean_name]
        else:
            # Fallback to direct index if label count matches
            mapping[d_idx] = d_idx
    return mapping


def prepare_dataset_transforms(
    dataset: DatasetDict,
    processor: AutoImageProcessor,
    model_label2id: Dict[str, int],
) -> Tuple[DatasetDict, Any]:
    """Apply transforms to DatasetDict and return dataset with collator."""
    dataset_classes = dataset["train"].features["label"].names
    mapping = create_label_mapping(dataset_classes, model_label2id)

    def transform_fn(batch):
        # Convert all images to RGB
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        # Remap labels to model label indices
        inputs["labels"] = torch.tensor([mapping.get(lbl, lbl) for lbl in batch["label"]])
        return inputs

    transformed_dataset = dataset.with_transform(transform_fn)

    def collate_fn(batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["labels"] for x in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    return transformed_dataset, collate_fn
