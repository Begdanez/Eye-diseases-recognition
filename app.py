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
from PIL import Image
from torchvision import models, transforms

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEFAULT_CLASSES = {
    "0": "AMD", "1": "CNV", "2": "CSR", "3": "DME",
    "4": "DR",  "5": "Drusen", "6": "MH", "7": "Normal",
}

CLASS_INFO = {
    "AMD":    ("Age-related Macular Degeneration",
               "Degenerative disease affecting the macula, leading to central vision loss.", "#FF6B6B"),
    "CNV":    ("Choroidal Neovascularization",
               "Abnormal blood vessel growth beneath the retina, often linked to AMD.", "#FF9F43"),
    "CSR":    ("Central Serous Retinopathy",
               "Fluid accumulation under the retina causing distorted or blurred vision.", "#FFEAA7"),
    "DME":    ("Diabetic Macular Edema",
               "Swelling of the macula due to fluid leakage in diabetic retinopathy.", "#FD79A8"),
    "DR":     ("Diabetic Retinopathy",
               "Damage to blood vessels in the retina caused by diabetes.", "#E17055"),
    "Drusen": ("Drusen (Deposits)",
               "Small yellow deposits under the retina; early sign of AMD.", "#A29BFE"),
    "MH":     ("Macular Hole",
               "A small opening in the macula causing blurry or distorted central vision.", "#74B9FF"),
    "Normal": ("Healthy Retina",
               "No abnormalities detected. The retina appears normal.", "#00CEC9"),
}

# ─────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────
T = {
    "ru": {
        "subtitle":        "Автоматическая диагностика 8 заболеваний сетчатки по ОКТ-снимкам",
        "model_info":      "Модель: EfficientNet-B3 · Датасет: OCT-C8 (24 000 снимков)",
        "upload_label":    "ЗАГРУЗИТЬ ОКТ-СНИМОК",
        "analyze_btn":     "Анализировать снимок",
        "cond_label":      "Поддерживаемые заболевания:",
        "lang_btn":        "🌐  EN",
        "footer":          ("⚠️ Только для исследовательских и образовательных целей. "
                            "Данный инструмент НЕ является заменой профессиональной "
                            "медицинской диагностики. Обязательно консультируйтесь "
                            "с врачом-офтальмологом."),
        "accordion_label": "📋  Все заболевания",
        "no_disease":      "✅ Патологии не выявлены",
        "disease_found":   "⚠️ Обнаружена патология",
        "diagnosis_lbl":   "Диагноз",
        "confidence_lbl":  "Уверенность",
        "start_hint":      "*Загрузите ОКТ-снимок и нажмите **Анализировать снимок**, чтобы начать.*",
        "pred_class_lbl":  "Предсказанный класс",
        "conf_lbl":        "Уверенность",
    },
    "en": {
        "subtitle":        "AI-powered detection of 8 retinal conditions from OCT images",
        "model_info":      "Model: EfficientNet-B3 · Dataset: OCT-C8 (24 000 images)",
        "upload_label":    "UPLOAD OCT IMAGE",
        "analyze_btn":     "Analyze Image",
        "cond_label":      "Supported conditions:",
        "lang_btn":        "🌐  RU",
        "footer":          ("⚠️ For research and educational purposes only. "
                            "This tool is NOT a substitute for professional medical diagnosis. "
                            "Always consult a qualified ophthalmologist."),
        "accordion_label": "📋  View all conditions",
        "no_disease":      "✅ No disease detected",
        "disease_found":   "⚠️ Abnormality detected",
        "diagnosis_lbl":   "Diagnosis",
        "confidence_lbl":  "Confidence",
        "start_hint":      "*Upload an OCT image and click **Analyze Image** to get started.*",
        "pred_class_lbl":  "Predicted Class",
        "conf_lbl":        "Confidence",
    },
}

# ─────────────────────────────────────────────
# DISEASE DATA PER LANGUAGE
# ─────────────────────────────────────────────
DISEASES = {
    "ru": [
        ("AMD",    "Возрастная макулярная дегенерация (AMD)",
         "Дегенеративное заболевание макулы, ведущее к потере центрального зрения.", "#FF6B6B"),
        ("CNV",    "Хориоидальная неоваскуляризация (CNV)",
         "Патологический рост сосудов под сетчаткой, часто связанный с AMD.", "#FF9F43"),
        ("CSR",    "Центральная серозная ретинопатия (CSR)",
         "Скопление жидкости под сетчаткой, вызывающее искажение зрения.", "#FFEAA7"),
        ("DME",    "Диабетический макулярный отёк (DME)",
         "Отёк макулы из-за утечки жидкости при диабетической ретинопатии.", "#FD79A8"),
        ("DR",     "Диабетическая ретинопатия (DR)",
         "Поражение сосудов сетчатки вследствие сахарного диабета.", "#E17055"),
        ("Drusen", "Друзы / отложения (Drusen)",
         "Жёлтые отложения под сетчаткой; ранний признак AMD.", "#A29BFE"),
        ("MH",     "Макулярное отверстие (MH)",
         "Небольшое отверстие в макуле, вызывающее размытость центрального зрения.", "#74B9FF"),
        ("Normal", "Здоровая сетчатка (Normal)",
         "Патологии не выявлены. Сетчатка в норме.", "#00CEC9"),
    ],
    "en": [
        ("AMD",    "Age-related Macular Degeneration (AMD)",
         "Degenerative disease affecting the macula, leading to central vision loss.", "#FF6B6B"),
        ("CNV",    "Choroidal Neovascularization (CNV)",
         "Abnormal blood vessel growth beneath the retina, often linked to AMD.", "#FF9F43"),
        ("CSR",    "Central Serous Retinopathy (CSR)",
         "Fluid accumulation under the retina causing distorted or blurred vision.", "#FFEAA7"),
        ("DME",    "Diabetic Macular Edema (DME)",
         "Swelling of the macula due to fluid leakage in diabetic retinopathy.", "#FD79A8"),
        ("DR",     "Diabetic Retinopathy (DR)",
         "Damage to blood vessels in the retina caused by diabetes.", "#E17055"),
        ("Drusen", "Drusen (Deposits)",
         "Small yellow deposits under the retina; early sign of AMD.", "#A29BFE"),
        ("MH",     "Macular Hole (MH)",
         "A small opening in the macula causing blurry or distorted central vision.", "#74B9FF"),
        ("Normal", "Healthy Retina (Normal)",
         "No abnormalities detected. The retina appears normal.", "#00CEC9"),
    ],
}

# ─────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────
def make_header(lang):
    t = T[lang]
    return f"""
<div style="text-align:center;padding:1.8rem 1rem 0.8rem;">
  <h1 style="font-family:'Space Mono',monospace;font-size:2rem;color:#58A6FF;
             letter-spacing:-0.02em;margin-bottom:0.3rem;">
    🔬 Retinal OCT Analyzer
  </h1>
  <p style="color:#8B949E;font-size:0.95rem;margin:0;">
    {t['subtitle']}<br>
    <span style="font-size:0.8rem;color:#484F58;">{t['model_info']}</span>
  </p>
</div>"""


def make_upload_label(lang):
    return (f'<p style="color:#58A6FF;font-weight:600;font-size:0.85rem;'
            f'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.4rem;">'
            f'{T[lang]["upload_label"]}</p>')


def make_conditions(lang):
    return (f'<div style="font-size:.82rem;color:#8B949E;line-height:1.6;">'
            f'<b style="color:#58A6FF;">{T[lang]["cond_label"]}</b><br>'
            f'AMD &middot; CNV &middot; CSR &middot; DME<br>'
            f'DR &middot; Drusen &middot; MH &middot; Normal</div>')


def make_disease_list(lang):
    rows = ""
    for _, name, desc, color in DISEASES[lang]:
        rows += (
            f'<div style="display:flex;align-items:flex-start;gap:.7rem;'
            f'padding:.55rem .7rem;border-radius:8px;background:#0D1117;'
            f'border:1px solid #21262D;margin-bottom:.45rem;">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
            f'margin-top:4px;flex-shrink:0;display:inline-block;"></span>'
            f'<div>'
            f'<div style="font-family:monospace;font-size:.79rem;font-weight:700;'
            f'color:#E6EDF3;line-height:1.3;">{name}</div>'
            f'<div style="font-size:.77rem;color:#8B949E;line-height:1.45;margin-top:.12rem;">{desc}</div>'
            f'</div></div>'
        )
    return f'<div style="padding:.2rem 0;">{rows}</div>'


def make_footer(lang):
    return (f'<div style="text-align:center;color:#484F58;font-size:.78rem;'
            f'padding:1rem;border-top:1px solid #21262D;margin-top:1.5rem;">'
            f'{T[lang]["footer"]}</div>')


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes):
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


def load_model(model_path, num_classes, device):
    model = build_model(num_classes)
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print("Weights loaded.")
    else:
        print("No weights found — using random weights (demo mode).")
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def preprocess(image):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def run_model(image, model, class_names, device):
    tensor = preprocess(image).to(device)
    probs  = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    top_i  = int(np.argmax(probs))
    return class_names[str(top_i)], float(probs[top_i]), probs


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def make_chart(probs, class_names):
    labels = [class_names[str(i)] for i in range(len(probs))]
    colors = [CLASS_INFO[l][2] if l in CLASS_INFO else "#888" for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    bars = ax.barh(labels, probs * 100, color=colors, height=0.65, edgecolor="none")
    for bar, p in zip(bars, probs):
        ax.text(min(bar.get_width() + 1.5, 99),
                bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", ha="left",
                color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)", color="#8B949E", fontsize=9)
    ax.tick_params(colors="#C9D1D9", labelsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.xaxis.grid(True, color="#21262D", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Class Probabilities", color="#E6EDF3", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

body, .gradio-container {
    background: #0D1117 !important;
    font-family: 'Inter', sans-serif !important;
    color: #C9D1D9 !important;
}
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.left-panel {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Primary analyze button */
button.primary {
    background: linear-gradient(135deg, #58A6FF, #3B82F6) !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.85 !important; }

/* Language button */
.lang-btn > div > button {
    background: transparent !important;
    border: 1px solid #30363D !important;
    border-radius: 20px !important;
    color: #58A6FF !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.75rem !important;
    min-width: 70px !important;
    transition: all 0.2s !important;
}
.lang-btn > div > button:hover {
    background: #21262D !important;
    border-color: #58A6FF !important;
}

/* Textboxes */
textarea, input[type="text"] {
    background: #0D1117 !important;
    border-color: #30363D !important;
    color: #C9D1D9 !important;
    border-radius: 8px !important;
}

/* Markdown result */
.result-md {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    color: #C9D1D9 !important;
}
"""

EXAMPLES_DIR = "examples"


# ─────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────
def build_ui(model, class_names, device):

    def predict_fn(image, lang):
        if image is None:
            return "—", "—", T[lang]["start_hint"], None
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        label, prob, probs = run_model(pil, model, class_names, device)
        info = CLASS_INFO.get(label, ("Unknown", "No description.", "#888"))
        conf = f"{prob*100:.1f}%"
        risk = T[lang]["no_disease"] if label == "Normal" else T[lang]["disease_found"]
        md = (f"## {risk}\n\n"
              f"**{T[lang]['diagnosis_lbl']}:** `{info[0]}`  \n"
              f"**{T[lang]['confidence_lbl']}:** `{conf}`\n\n"
              f"> {info[1]}")
        return label, conf, md, make_chart(probs, class_names)

    def toggle_lang(current_lang):
        new = "en" if current_lang == "ru" else "ru"
        return (
            new,                                          # lang_state
            make_header(new),                             # header_html
            make_upload_label(new),                       # upload_label_html
            gr.update(value=T[new]["analyze_btn"]),       # analyze_btn
            make_conditions(new),                         # conditions_html
            gr.update(value=T[new]["lang_btn"]),          # lang_btn
            make_disease_list(new),                       # disease_list_html
            make_footer(new),                             # footer_html
            gr.update(label=T[new]["accordion_label"]),   # disease_accordion
        )

    with gr.Blocks(title="Retinal OCT Analyzer", css=CUSTOM_CSS) as demo:

        lang_state = gr.State("ru")

        # ── Top bar ──────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=9):
                header_html_comp = gr.HTML(make_header("ru"))
            with gr.Column(scale=1, min_width=80, elem_classes="lang-btn"):
                lang_btn = gr.Button(T["ru"]["lang_btn"])

        # ── Main ─────────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # Left
            with gr.Column(scale=1, elem_classes="left-panel"):
                upload_label_comp = gr.HTML(make_upload_label("ru"))
                image_input = gr.Image(type="pil", label="", height=280)
                analyze_btn = gr.Button(T["ru"]["analyze_btn"], variant="primary", size="lg")

                gr.HTML("<hr style='border-color:#21262D;margin:1rem 0'>")

                conditions_comp = gr.HTML(make_conditions("ru"))

                with gr.Accordion(T["ru"]["accordion_label"], open=False) as disease_accordion:
                    disease_list_comp = gr.HTML(make_disease_list("ru"))

            # Right
            with gr.Column(scale=2):
                with gr.Row():
                    top_label_box  = gr.Textbox(label=T["ru"]["pred_class_lbl"], interactive=False)
                    confidence_box = gr.Textbox(label=T["ru"]["conf_lbl"],        interactive=False)
                result_md    = gr.Markdown(value=T["ru"]["start_hint"],
                                           elem_classes="result-md")
                chart_output = gr.Plot(label="Probability Distribution")

        # Examples
        if os.path.isdir(EXAMPLES_DIR):
            imgs = [os.path.join(EXAMPLES_DIR, f) for f in os.listdir(EXAMPLES_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if imgs:
                gr.HTML("<p style='color:#8B949E;font-size:.85rem;margin-top:1rem'>Examples:</p>")
                gr.Examples(examples=imgs, inputs=image_input, label="")

        footer_comp = gr.HTML(make_footer("ru"))

        # ── Events ───────────────────────────────────────────────────────

        lang_btn.click(
            fn=toggle_lang,
            inputs=[lang_state],
            outputs=[
                lang_state,
                header_html_comp,
                upload_label_comp,
                analyze_btn,
                conditions_comp,
                lang_btn,
                disease_list_comp,
                footer_comp,
                disease_accordion,
            ],
        )

        analyze_btn.click(
            fn=predict_fn,
            inputs=[image_input, lang_state],
            outputs=[top_label_box, confidence_box, result_md, chart_output],
        )

        image_input.change(
            fn=predict_fn,
            inputs=[image_input, lang_state],
            outputs=[top_label_box, confidence_box, result_md, chart_output],
        )

    return demo


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="checkpoints/best_model.pth")
    p.add_argument("--classes", default="checkpoints/class_names.json")
    p.add_argument("--port",    type=int, default=7860)
    p.add_argument("--share",   action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.path.exists(args.classes):
        with open(args.classes) as f:
            class_names = json.load(f)
        print(f"Classes: {list(class_names.values())}")
    else:
        print("class_names.json not found — using defaults.")
        class_names = DEFAULT_CLASSES

    model = load_model(args.model, len(class_names), device)
    demo  = build_ui(model, class_names, device)

    for attempt in range(50):
        port = args.port + attempt
        try:
            demo.launch(server_port=port, share=args.share)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) or "address already in use" in str(e).lower():
                print(f"Port {port} busy, trying {port+1}…")
            else:
                raise


if __name__ == "__main__":
    main()