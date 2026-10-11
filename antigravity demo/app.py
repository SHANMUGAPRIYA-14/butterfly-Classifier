"""Streamlit Web Application for Butterfly Species Classification.

Uses Vision Transformer (ViT) with Test-Time Augmentation (TTA).
"""

from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import requests
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.config import MODEL_NAME, SAVED_MODEL_DIR, DEVICE
from src.predict import generate_tta_crops


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Butterfly Species Classification",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme CSS
st.markdown(
    """
    <style>
    /* Dark Theme Custom Styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .species-badge {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .conf-badge {
        font-size: 1.25rem;
        font-weight: 600;
        color: #34d399;
    }
    .warning-box {
        background-color: rgba(245, 158, 11, 0.15);
        border-left: 4px solid #f59e0b;
        color: #fcd34d;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model Loading with Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Vision Transformer model...")
def load_cached_model():
    """Load and cache the model and processor.
    
    Prefers fine-tuned model in outputs/saved_model if available,
    otherwise loads the pretrained Hugging Face checkpoint.
    """
    if SAVED_MODEL_DIR.exists() and (SAVED_MODEL_DIR / "config.json").exists():
        model_path = str(SAVED_MODEL_DIR)
        source = "Fine-tuned model (Local outputs)"
    else:
        model_path = MODEL_NAME
        source = f"Pretrained Hugging Face ({MODEL_NAME})"

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, processor, source


def run_prediction(
    image: Image.Image,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Execute inference with 6-way Test-Time Augmentation (TTA)."""
    crops = generate_tta_crops(image)
    crop_probs = []

    with torch.no_grad():
        for crop in crops:
            inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()
            crop_probs.append(probs)

    avg_probs = torch.stack(crop_probs).mean(dim=0)
    top5_scores, top5_indices = torch.topk(avg_probs, k=5)

    top5 = []
    for score, idx in zip(top5_scores, top5_indices):
        species = model.config.id2label[idx.item()]
        top5.append((species, float(score.item())))

    predicted_species = top5[0][0]
    confidence = top5[0][1]
    return predicted_species, confidence, top5


def render_top5_chart(top5: List[Tuple[str, float]], confidence: float):
    """Render horizontal bar chart for top-5 candidates in dark theme."""
    chart_species = [item[0] for item in reversed(top5)]
    chart_probs = [item[1] * 100 for item in reversed(top5)]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    # Colors
    bar_colors = ["#38bdf8", "#60a5fa", "#818cf8", "#a78bfa", "#34d399"]
    if confidence < 0.50:
        bar_colors[-1] = "#f59e0b"

    bars = ax.barh(chart_species, chart_probs, color=bar_colors, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color="#94a3b8", fontsize=10, fontweight="bold")
    ax.set_title("Top-5 Candidates (Test-Time Augmentation)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#94a3b8")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            color="#f8fafc",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🦋 Butterfly Species Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Deep Learning Transfer Learning with Pretrained Vision Transformer (ViT) & Test-Time Augmentation</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Model and Hardware Info
    with st.sidebar:
        st.header("⚙️ System Status")
        model, processor, model_source = load_cached_model()
        st.success(f"**Loaded Model:**\n{model_source}")
        st.info(f"**Device:** `{DEVICE.upper()}`")
        st.markdown(f"**Classes:** {len(model.config.id2label)} species")
        st.markdown("---")
        st.subheader("💡 About the Project")
        st.markdown(
            """
            This system classifies **100 butterfly & moth species** using:
            - **Model:** Pretrained Vision Transformer (`ViT-Base`)
            - **Technique:** Transfer Learning & Fine-Tuning
            - **Inference:** 6-Way Test-Time Augmentation (TTA)
            - **Framework:** PyTorch & Hugging Face Transformers
            """
        )

    # Input Option Selector
    input_mode = st.radio(
        "Choose Input Method:",
        options=["Upload Image", "Image URL"],
        horizontal=True,
    )

    input_image: Optional[Image.Image] = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a butterfly photo (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select a clear photo of a butterfly.",
        )
        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load uploaded image: {e}")

    else:
        url_input = st.text_input(
            "Enter Butterfly Image URL:",
            placeholder="https://example.com/butterfly.jpg",
        )
        if url_input.strip():
            try:
                resp = requests.get(url_input.strip(), timeout=10)
                resp.raise_for_status()
                input_image = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                st.error(f"Failed to fetch image from URL: {e}")

    # Display image & prediction
    if input_image is not None:
        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.subheader("📸 Input Image")
            st.image(input_image, use_container_width=True)

        with col2:
            st.subheader("🎯 Classification Results")
            with st.spinner("Analyzing image with Vision Transformer & TTA..."):
                predicted_species, confidence, top5 = run_prediction(input_image, model, processor)

            conf_pct = confidence * 100
            conf_color = "#34d399" if conf_pct >= 50 else "#f59e0b"

            # Primary Prediction Metric Card
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Species</div>
                    <div class="species-badge">{predicted_species}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">Confidence Score</div>
                    <div class="conf-badge" style="color: {conf_color};">{conf_pct:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Warning for low confidence (< 50%)
            if conf_pct < 50:
                st.markdown(
                    """
                    <div class="warning-box">
                        ⚠️ <b>Low Confidence Warning (&lt; 50%):</b><br>
                        The photo may be blurred, occluded, or the butterfly species might not be among the 100 trained categories.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Top-5 Horizontal Bar Chart
            st.markdown("<br>", unsafe_allow_html=True)
            chart_fig = render_top5_chart(top5, confidence)
            st.pyplot(chart_fig)
            plt.close(chart_fig)

    else:
        st.info("👆 Please upload a butterfly photo or enter an image URL to see predictions.")


if __name__ == "__main__":
    main()
