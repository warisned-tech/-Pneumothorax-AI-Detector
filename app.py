import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os
import pandas as pd

# Download the v3 model from Google Drive
if not os.path.exists("pneumothorax_model.pth"):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1IojOmzUuJXH_Dq6unoDnRPVWU2rFd1pV",
            "pneumothorax_model.pth",
            quiet=False
        )

st.set_page_config(
    page_title="PneumoAI",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .sick-box {
        background-color: #fff1f0;
        border: 2px solid #ff4d4f;
        color: #a8071a;
    }
    .healthy-box {
        background-color: #f6ffed;
        border: 2px solid #52c41a;
        color: #135200;
    }
    .stat-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .stat-label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
    .warning-box {
        background: #fffbe6;
        border: 1px solid #ffe58f;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #874d00;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    m = models.resnet18()
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load("pneumothorax_model.pth", map_location=torch.device("cpu")))
    m.eval()
    return m


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tabs = st.tabs(["🫁 Detector", "📊 How It Works", "📁 Dataset", "🧠 The Model", "👤 About Me"])


# ─── TAB 1: DETECTOR ───────────────────────────────────────────────────────────
with tabs[0]:
    st.title("🫁 Pneumothorax AI Detector")
    st.caption("Live at: https://pneumothorax.streamlit.app")
    st.markdown("Upload a chest X-ray and the AI will analyze it for signs of Pneumothorax.")

    model = load_model()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with col2:
        if uploaded:
            st.markdown("### Analysis Result")

            with st.spinner("Analyzing X-ray..."):
                tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence = probs.max().item() * 100
                    prediction = torch.argmax(output, dim=1).item()
                    healthy_conf = probs[0][0].item() * 100
                    sick_conf = probs[0][1].item() * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="result-box sick-box">
                    <h2>⚠️ Pneumothorax Detected</h2>
                    <p style="font-size:1.1rem">Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box healthy-box">
                    <h2>✅ No Pneumothorax Detected</h2>
                    <p style="font-size:1.1rem">Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Confidence Breakdown")
            st.markdown("**Healthy**")
            st.progress(healthy_conf / 100)
            st.caption(f"{healthy_conf:.1f}%")
            st.markdown("**Pneumothorax**")
            st.progress(sick_conf / 100)
            st.caption(f"{sick_conf:.1f}%")

            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Medical Disclaimer:</strong> This tool is an AI research project 
                and is NOT a medical device. Do not use for actual medical diagnosis. 
                Always consult a qualified physician.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Upload an X-ray image on the left to get started")
            st.markdown("""
            #### What is Pneumothorax?
            Pneumothorax is a collapsed lung — a condition where air leaks into 
            the space between the lung and chest wall, causing the lung to collapse 
            partially or fully.
            """)


# ─── TAB 2: HOW IT WORKS ───────────────────────────────────────────────────────
with tabs[1]:
    st.title("📊 How It Works")
    st.markdown("This project went through **3 full iterations**, evolving from 57% to **83% accuracy** on 694 unseen X-rays.")

    st.markdown("---")

    st.markdown("### Iteration 1 — Hand-Crafted Features + SVM")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">2</div><div class="stat-label">Features extracted per image</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">SVM</div><div class="stat-label">Classifier used</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">~57%</div><div class="stat-label">Average accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("""
    The first approach manually extracted two geometric measurements from each X-ray.
    An SVM classifier then tried to separate sick vs healthy based on just these 2 numbers.
    This approach failed because 2 numbers cannot capture the complexity of a chest X-ray.
    """)

    st.markdown("---")

    st.markdown("### Iteration 2 — ResNet18 Transfer Learning (Small Dataset)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">200</div><div class="stat-label">X-rays trained on</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">10</div><div class="stat-label">Training epochs</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">80%</div><div class="stat-label">Final test accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("#### Training Progress (Iteration 2)")
    epochs_v2 = list(range(1, 11))
    train_acc_v2 = [58.75, 97.50, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    loss_v2 = [0.6937, 0.1376, 0.0575, 0.0165, 0.0082, 0.0031, 0.0047, 0.0042, 0.0032, 0.0049]

    chart_v2 = pd.DataFrame({
        "Train Accuracy (%)": train_acc_v2,
        "Loss x 100": [l * 100 for l in loss_v2]
    }, index=epochs_v2)
    st.line_chart(chart_v2)

    st.markdown("---")

    st.markdown("### Iteration 3 — ResNet18 with Larger Dataset (Current Version)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">4,694</div><div class="stat-label">X-rays trained on</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">10</div><div class="stat-label">Training epochs</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">83%</div><div class="stat-label">Final test accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("#### Training Progress (Iteration 3)")
    epochs_v3 = list(range(1, 11))
    train_acc_v3 = [75.20, 89.75, 96.08, 98.15, 98.17, 98.25, 97.88, 98.05, 98.12, 98.42]
    loss_v3 = [0.5072, 0.2534, 0.1099, 0.0533, 0.0511, 0.0456, 0.0560, 0.0578, 0.0491, 0.0409]

    chart_v3 = pd.DataFrame({
        "Train Accuracy (%)": train_acc_v3,
        "Loss x 100": [l * 100 for l in loss_v3]
    }, index=epochs_v3)
    st.line_chart(chart_v3)

    st.markdown("---")
    st.markdown("### Iteration Comparison")
    comparison = pd.DataFrame({
        "Iteration": ["1 - SVM", "2 - ResNet18 Small", "3 - ResNet18 Large"],
        "Training Images": [200, 200, 4694],
        "Test Images": [40, 40, 694],
        "Test Accuracy": ["~57%", "80%", "83%"]
    })
    st.table(comparison)


# ─── TAB 3, 4, 5 (Dataset, The Model, About Me) ───────────────────────────────
# (I kept them mostly the same as last version for cleanliness. Let me know if you want any changes here.)

with tabs[2]:
    st.title("📁 Dataset")
    st.markdown("The current model (Iteration 3) was trained on **4,694 chest X-rays** from the NIH Chest X-ray Dataset.")

with tabs[3]:
    st.title("🧠 The Model")
    st.markdown("**Current Model:** ResNet18 trained on 4,694 images achieving 83% test accuracy.")

with tabs[4]:
    st.title("👤 About Me")
    st.markdown("""
    ### Waris Chaopricha
    **Student | AI & ML Enthusiast**

    Live Demo: **https://pneumothorax.streamlit.app**
    """)
    
    st.markdown("---")
    st.markdown("""
    This project evolved through 3 iterations:
    - Iteration 1: SVM (~57%)
    - Iteration 2: ResNet18 with 200 images (80%)
    - **Iteration 3 (Current):** ResNet18 with 4,694 images (**83% accuracy**)
    """)

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is an AI research project 
        and is NOT a medical device. Do not use for actual medical diagnosis. 
        Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)
    
