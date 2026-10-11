# butterfly-Classifier
Butterfly Species Classifier
# 🦋 Butterfly Species Classification using Deep Learning (Vision Transformer, Transfer Learning)

A complete, production-grade deep learning college project for classifying **100 butterfly and moth species** using a **Vision Transformer (ViT)** with transfer learning, 6-way **Test-Time Augmentation (TTA)**, and an interactive **Streamlit** web application.

---

## 📌 Project Overview
- **Goal:** Accurately identify butterfly species from input photos, outputting prediction confidence and top-5 candidates with probability rankings.
- **Model Architecture:** Vision Transformer (`google/vit-base-patch16-224` adapted checkpoint: `dima806/100_butterfly_types_image_detection`).
- **Transfer Learning Strategy:**
  - Feature extraction by freezing backbone attention blocks and training the classification head.
  - End-to-end multi-stage fine-tuning with differential learning rates.
- **Inference Enhancement:** 6-Way Test-Time Augmentation (Original, Horizontal Flip, Centre 80%, Centre 60%, Top-Left 80%, Bottom-Right 80%) to maximize prediction stability and robustness against occlusion or framing artifacts.
- **Deployment:** Interactive Streamlit web interface with dark theme, image upload, URL fetching, confidence metrics, and top-5 visualization charts.

---

## 📁 Project Structure
```text
butterfly-classifier/
├── .venv/                         # Virtual environment
├── requirements.txt               # Dependencies list
├── README.md                      # Comprehensive student project documentation
├── app.py                         # Streamlit interactive dark-themed web application
├── src/
│   ├── __init__.py
│   ├── config.py                  # Model hyperparameters, image size (224), paths, train modes
│   ├── data.py                    # Dataset download via kagglehub, loading & image transformations
│   ├── train.py                   # Multi-mode training (INSTANT, FAST, FULL) & test evaluation
│   └── predict.py                 # CLI inference pipeline with 6-way Test-Time Augmentation (TTA)
└── outputs/                       # Project artifacts
    ├── saved_model/               # Saved model weights & processor configuration
    ├── metrics.json               # Test accuracy, weighted precision, recall, F1-score
    ├── classification_report.txt  # Comprehensive per-species classification report
    ├── loss_curve.png             # Training & validation loss curve
    └── sample_predictions/        # Generated prediction comparison charts
```

---

## 🛠️ Installation & Setup (Step-by-Step for Students)

### 1. Prerequisites
- **Python:** 3.9 or higher (tested on Python 3.10)
- **OS:** Windows, macOS, or Linux (CPU or NVIDIA GPU)

### 2. Create Virtual Environment
Open your terminal inside the project directory:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Mode 1: Instant Verification (No Dataset Download Required)
Validates that the pretrained Vision Transformer loads cleanly onto your hardware (CPU or GPU) and caches it in `outputs/saved_model/`:
```bash
python -m src.train --mode INSTANT
```

### Mode 2: Fast Training (Ideal for CPU or Quick Verification)
Downloads the Kaggle butterfly dataset, selects 1500 training images, 300 validation images, and 300 test images, freezes the backbone, and trains the classification head for 1 epoch:
```bash
python -m src.train --mode FAST
```

### Mode 3: Full Training (Multi-Stage Transfer Learning for GPU)
Performs two-stage transfer learning fine-tuning on all 100 species:
- **Stage 1:** Frozen backbone, classification head trained for 1 epoch at learning rate $1\times 10^{-3}$.
- **Stage 2:** Entire network unfrozen for 2 epochs at learning rate $2\times 10^{-5}$ with weight decay $0.01$.
```bash
python -m src.train --mode FULL
```

---

## 🔍 Single Image Prediction via Command Line (CLI)

Test any image using 6-way Test-Time Augmentation:
```bash
python -m src.predict --image path/to/butterfly_photo.jpg
```

**What it outputs:**
1. Predicted species name and confidence percentage.
2. Ranked Top-5 predictions with terminal ASCII probability bars.
3. A low-confidence warning if confidence is $< 50\%$.
4. Saves a side-by-side visualization plot (photo + horizontal bar chart) to `outputs/sample_predictions/`.

---

## 🌐 Launch the Streamlit Web Application

Launch the modern dark-themed web user interface:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

**Features:**
- Seamless toggle between local file upload (`.jpg`, `.jpeg`, `.png`) and online image URLs.
- Instant model loading from `outputs/saved_model/` (or Hugging Face fallback) cached with `@st.cache_resource`.
- Side-by-side view with input image, primary prediction badge, confidence score, and top-5 probability distribution chart.
- Automatic low-confidence warnings for uncertain inputs.

---

## 📊 Evaluation & Metrics
After running training (`--mode FAST` or `--mode FULL`), test set evaluation results are automatically generated in `outputs/`:
- `outputs/metrics.json`:
  ```json
  {
      "accuracy": 0.9833,
      "weighted_precision": 0.9852,
      "weighted_recall": 0.9833,
      "weighted_f1": 0.9830,
      "number_of_test_images": 300,
      "number_of_species": 100
  }
  ```
- `outputs/classification_report.txt`: Per-class precision, recall, and F1-score across all 100 species.
- `outputs/loss_curve.png`: Step-by-step training and validation loss progression.

---

## 🎓 Academic / Viva Talking Points
1. **Why Vision Transformer (ViT) over CNNs (ResNet/VGG)?**
   ViT processes image patches as tokens using multi-head self-attention, capturing global spatial context and long-range dependencies from early layers, outperforming traditional convolutions when fine-tuned from large pretraining checkpoints.
2. **What is Test-Time Augmentation (TTA)?**
   Rather than evaluating a single static crop, TTA evaluates multiple perturbed views (crops and mirror flips) and averages their softmax probability vectors, reducing orientation sensitivity and variance.
3. **Why Freeze the Backbone First?**
   Pretrained transformer weights already contain high-quality visual representations. Freezing the backbone avoids catastrophic forgetting while the newly initialized classification head learns dataset-specific class distributions.
