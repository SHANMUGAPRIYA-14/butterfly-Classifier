"""Configuration settings for Butterfly Species Classification."""

import os
from pathlib import Path
import torch

# Base directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAVED_MODEL_DIR = OUTPUTS_DIR / "saved_model"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"
LOSS_CURVE_PATH = OUTPUTS_DIR / "loss_curve.png"
SAMPLE_PREDICTIONS_DIR = OUTPUTS_DIR / "sample_predictions"

# Create required output directories
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained model and dataset IDs
MODEL_NAME = "dima806/100_butterfly_types_image_detection"
KAGGLE_DATASET = "gpiosenka/butterfly-images40-species"
IMAGE_SIZE = 224

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = torch.cuda.is_available()
NUM_WORKERS = 0 if os.name == "nt" else 2

# Training Mode Configurations
MODES = {
    "INSTANT": {
        "description": "Baseline verification: loads pretrained model without training or dataset download.",
        "download_dataset": False,
        "train": False,
    },
    "FAST": {
        "description": "Quick training: 1500 train, 300 valid, 300 test images. Backbone frozen for 1 epoch.",
        "download_dataset": True,
        "train": True,
        "train_samples": 1500,
        "val_samples": 300,
        "test_samples": 300,
        "epochs": 1,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "weight_decay": 0.0,
        "freeze_backbone": True,
    },
    "FULL": {
        "description": "Full 2-stage transfer learning fine-tuning on complete dataset.",
        "download_dataset": True,
        "train": True,
        "train_samples": None,
        "val_samples": None,
        "test_samples": None,
        "stage1": {
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "freeze_backbone": True,
            "batch_size": 16,
        },
        "stage2": {
            "epochs": 2,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "freeze_backbone": False,
            "batch_size": 16,
        },
    },
}
