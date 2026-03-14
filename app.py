"""
app.py — Retinal OCT Disease Recognition App
=============================================
Run:  python app.py
      python app.py --model checkpoints/best_model.pth
                    --classes checkpoints/class_names.json

Requires:
  pip install gradio torch torchvision pillow matplotlib
"""

import argparse
import json
import os

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B3_Weights

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEFAULT_CLASSES = {
    "0": "AMD",
    "1": "CNV",
    "2": "CSR",
    "3": "DME",
    "4": "DR",
    "5": "Drusen",
    "6": "MH",
    "7": "Normal",
}

# Medical descriptions shown in the UI
CLASS_INFO = {
    "AMD":    ("Age-related Macular Degeneration",
               "Degenerative disease affecting the macula, leading to central vision loss.",
               "#FF6B6B"),
    "CNV":    ("Choroidal Neovascularization",
               "Abnormal blood vessel growth beneath the retina, often linked to AMD.",
               "#FF9F43"),
    "CSR":    ("Central Serous Retinopathy",
               "Fluid accumulation under the retina causing distorted or blurred vision.",
               "#FFEAA7"),
    "DME":    ("Diabetic Macular Edema",
               "Swelling of the macula due to fluid leakage in diabetic retinopathy.",
               "#FD79A8"),
    "DR":     ("Diabetic Retinopathy",
               "Damage to blood vessels in the retina caused by diabetes.",
               "#E17055"),
    "Drusen": ("Drusen (Deposits)",
               "Small yellow deposits under the retina; early sign of AMD.",
               "#A29BFE"),
    "MH":     ("Macular Hole",
               "A small opening in the macula causing blurry or distorted central vision.",
               "#74B9FF"),
    "Normal": ("Healthy Retina",
               "No abnormalities detected. The retina appears normal.",
               "#00CEC9"),
}

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes)
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print("Weights loaded successfully.")
    else:
        print("⚠  No model weights found — running with random weights (for demo).")
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def preprocess(image: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(image: Image.Image,
            model: nn.Module,
            class_names: dict,
            device: torch.device):
    tensor = preprocess(image).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx  = int(np.argmax(probs))
    top_label = class_names[str(top_idx)]
    top_prob  = float(probs[top_idx])
    return top_label, top_prob, probs, class_names


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def make_bar_chart(probs: np.ndarray, class_names: dict) -> plt.Figure:
    labels = [class_names[str(i)] for i in range(len(probs))]
    colors = [CLASS_INFO[lbl][2] if lbl in CLASS_INFO else "#888" for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(labels, probs * 100, color=colors, height=0.65,
                   edgecolor="none")

    # Value labels
    for bar, prob in zip(bars, probs):
        ax.text(min(bar.get_width() + 1.5, 99),
                bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%",
                va="center", ha="left",
                color="white", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)", color="#8B949E", fontsize=9)
    ax.tick_params(colors="#C9D1D9", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#21262D", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_title("Class Probabilities", color="#E6EDF3",
                 fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GRADIO HANDLER
# ─────────────────────────────────────────────
def make_predict_fn(model, class_names, device):
    def predict_fn(image):
        if image is None:
            return "—", "—", "—", None

        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        top_label, top_prob, probs, cn = predict(pil_img, model, class_names, device)

        info = CLASS_INFO.get(top_label, ("Unknown", "No description available.", "#888"))
        full_name   = info[0]
        description = info[1]
        color       = info[2]

        confidence_str = f"{top_prob * 100:.1f}%"

        # Compose result markdown
        risk = "✅ No disease detected" if top_label == "Normal" else "⚠️ Abnormality detected"
        result_md = f"""
## {risk}

**Diagnosis:** `{full_name}`  
**Confidence:** `{confidence_str}`

> {description}
"""
        fig = make_bar_chart(probs, cn)
        return top_label, confidence_str, result_md, fig

    return predict_fn


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

body, .gradio-container {
    background: #0D1117 !important;
    font-family: 'Inter', sans-serif !important;
    color: #C9D1D9 !important;
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.title-block {
    text-align: center;
    padding: 2rem 1rem 1rem;
}

.title-block h1 {
    font-size: 2rem;
    color: #58A6FF;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.title-block p {
    color: #8B949E;
    font-size: 0.95rem;
}

.panel {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.upload-label {
    color: #58A6FF !important;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.result-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.2rem;
}

button.primary {
    background: linear-gradient(135deg, #58A6FF, #3B82F6) !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}

button.primary:hover { opacity: 0.85 !important; }

.footer-note {
    text-align: center;
    color: #484F58;
    font-size: 0.78rem;
    padding: 1rem;
    border-top: 1px solid #21262D;
    margin-top: 1.5rem;
}

textarea, .markdown-text {
    background: #0D1117 !important;
    border-color: #30363D !important;
    color: #C9D1D9 !important;
    border-radius: 8px !important;
}
"""

EXAMPLES_DIR = "examples"  # put sample images here if available


def build_ui(predict_fn):
    with gr.Blocks(css=CUSTOM_CSS, title="Retinal OCT Analyzer") as demo:

        # ── Header ──────────────────────────────────
        gr.HTML("""
        <div class="title-block">
            <h1>🔬 Retinal OCT Analyzer</h1>
            <p>AI-powered detection of 8 retinal conditions from OCT images<br>
               <span style="font-size:0.8rem; color:#484F58;">
                 Model: EfficientNet-B3 · Dataset: OCT-C8 (24,000 images)
               </span>
            </p>
        </div>
        """)

        # ── Main layout ─────────────────────────────
        with gr.Row(equal_height=True):

            # Left: upload + controls
            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML('<p class="upload-label">Upload OCT Image</p>')
                image_input = gr.Image(
                    type="pil",
                    label="",
                    height=280,
                    elem_classes="upload-box",
                )
                analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")

                gr.HTML("<hr style='border-color:#21262D; margin:1rem 0'>")

                gr.HTML("""
                <div style="font-size:0.82rem; color:#8B949E; line-height:1.6;">
                  <b style="color:#58A6FF">Supported conditions:</b><br>
                  AMD · CNV · CSR · DME<br>
                  DR · Drusen · MH · Normal
                </div>
                """)

            # Right: results
            with gr.Column(scale=2):

                with gr.Row():
                    top_label_box  = gr.Textbox(label="Predicted Class",
                                                interactive=False,
                                                elem_classes="result-card")
                    confidence_box = gr.Textbox(label="Confidence",
                                                interactive=False,
                                                elem_classes="result-card")

                result_md = gr.Markdown(
                    value="*Upload an OCT image and click **Analyze Image** to get started.*",
                    elem_classes="result-card",
                )

                chart_output = gr.Plot(label="Probability Distribution")

        # ── Examples ────────────────────────────────
        if os.path.isdir(EXAMPLES_DIR):
            example_imgs = [
                os.path.join(EXAMPLES_DIR, f)
                for f in os.listdir(EXAMPLES_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if example_imgs:
                gr.HTML("<p style='color:#8B949E; font-size:0.85rem; margin-top:1rem'>Example Images:</p>")
                gr.Examples(
                    examples=example_imgs,
                    inputs=image_input,
                    label="",
                )

        # ── Disclaimer ──────────────────────────────
        gr.HTML("""
        <div class="footer-note">
          ⚠️ For research and educational purposes only.
          This tool is NOT a substitute for professional medical diagnosis.
          Always consult a qualified ophthalmologist.
        </div>
        """)

        # ── Event binding ────────────────────────────
        analyze_btn.click(
            fn=predict_fn,
            inputs=[image_input],
            outputs=[top_label_box, confidence_box, result_md, chart_output],
        )
        image_input.change(          # auto-predict on upload
            fn=predict_fn,
            inputs=[image_input],
            outputs=[top_label_box, confidence_box, result_md, chart_output],
        )

    return demo


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--classes", type=str, default="checkpoints/class_names.json",
                        help="Path to class_names.json produced by train.py")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true",
                        help="Create public Gradio share link")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load class names
    if os.path.exists(args.classes):
        with open(args.classes) as f:
            class_names = json.load(f)
        print(f"Classes loaded: {list(class_names.values())}")
    else:
        print("class_names.json not found — using defaults.")
        class_names = DEFAULT_CLASSES

    # Load model
    model = load_model(args.model, len(class_names), device)

    predict_fn = make_predict_fn(model, class_names, device)
    demo = build_ui(predict_fn)

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
