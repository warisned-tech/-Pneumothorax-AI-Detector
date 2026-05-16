import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os

if not os.path.exists("pneumothorax_model.pth"):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1ux4moHshP9DCMhYpvqajlMQE1FOlFfwA",
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
    .timeline-item {
        border-left: 3px solid #4361ee;
        padding-left: 1rem;
        margin-bottom: 1.5rem;
    }
    .timeline-title { font-weight: 600; font-size: 1rem; color: #1a1a2e; }
    .timeline-desc { font-size: 0.9rem; color: #6c757d; margin-top: 4px; }
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
    m.load_state_dict(torch.load("pneumothorax_model.pth",
                                  map_location=torch.device("cpu")))
    m.eval()
    return m


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

tabs = st.tabs(["🫁 Detector", "📊 How It Works", "📁 Dataset", "🧠 The Model", "👤 About Me"])


# ─── TAB 1: DETECTOR ───────────────────────────────────────────────────────────
with tabs[0]:
    st.title("🫁 Pneumothorax AI Detector")
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
            
            **Common symptoms:**
            - Sudden chest pain
            - Shortness of breath
            - Rapid heart rate
            
            Early detection via chest X-ray is critical for treatment.
            """)


# ─── TAB 2: HOW IT WORKS ───────────────────────────────────────────────────────
with tabs[1]:
    st.title("📊 How It Works")
    st.markdown("This project went through **2 full iterations** before reaching 82.5% accuracy.")

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
    The first approach manually extracted two geometric measurements from each X-ray:
    - **Lung area ratio** — what percentage of the chest is lung pixels
    - **Symmetry ratio** — how balanced left vs right lungs are (Pneumothorax collapses one side)
    
    An SVM classifier then tried to separate sick vs healthy based on just these 2 numbers.
    This failed because 2 numbers cannot capture the complexity of a chest X-ray.
    """)

    st.markdown("---")

    st.markdown("### Iteration 2 — ResNet18 Transfer Learning")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">200</div><div class="stat-label">X-rays trained on</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">10</div><div class="stat-label">Training epochs</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">82.5%</div><div class="stat-label">Final test accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("""
    Instead of hand-crafting features, the raw X-ray pixels were fed directly into 
    ResNet18 — a deep convolutional neural network pretrained on 1.2 million everyday photos.
    The model learned what Pneumothorax looks like on its own.
    """)

    st.markdown("#### Training Progress")
    epochs = list(range(1, 11))
    train_acc = [66.25, 98.75, 100, 99.38, 100, 100, 100, 100, 100, 100]
    loss = [0.6202, 0.1116, 0.0358, 0.0299, 0.0080, 0.0087, 0.0033, 0.0046, 0.0136, 0.0031]

    import pandas as pd
    chart_data = pd.DataFrame({"Train Accuracy (%)": train_acc}, index=epochs)
    st.line_chart(chart_data)


# ─── TAB 3: DATASET ────────────────────────────────────────────────────────────
with tabs[2]:
    st.title("📁 Dataset")
    st.markdown("The model was trained on data from the **NIH Chest X-ray Dataset**.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">112,120</div><div class="stat-label">Total X-rays in NIH dataset</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">14</div><div class="stat-label">Diseases labeled</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### What We Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Positive cases (sick)**
        - 100 X-rays labeled as Pneumothorax
        - Patients with a confirmed collapsed lung
        """)
    with col2:
        st.markdown("""
        **Control cases (healthy)**
        - 100 X-rays labeled as "No Finding"
        - Patients with completely normal lungs
        """)

    st.markdown("---")
    st.markdown("""
    ### Why Only 200 Images?
    The NIH dataset has thousands of Pneumothorax cases, but we deliberately kept 
    it small (200 total) as a proof of concept. Even with just 200 images, the model 
    achieved 82.5% accuracy — demonstrating that transfer learning is powerful 
    even with limited data.
    
    More images would push accuracy higher and reduce overfitting.
    """)


# ─── TAB 4: THE MODEL ──────────────────────────────────────────────────────────
with tabs[3]:
    st.title("🧠 The Model")

    st.markdown("### ResNet18 Architecture")
    st.markdown("""
    ResNet18 is a **residual neural network** developed by Microsoft Research in 2015. 
    It won the ImageNet competition and revolutionized computer vision.
    
    **Key innovation — skip connections:**
    Instead of passing information only forward layer by layer, ResNet adds 
    "shortcut connections" that skip layers. This solves the vanishing gradient 
    problem and allows much deeper networks to train effectively.
    """)

    st.markdown("---")
    st.markdown("### Transfer Learning")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What ResNet18 knew before:**
        - Edges and textures
        - Shapes and spatial relationships
        - General visual patterns
        - Trained on 1.2M everyday photos
        """)
    with col2:
        st.markdown("""
        **What we taught it:**
        - What a chest X-ray looks like
        - The visual signature of Pneumothorax
        - How to distinguish sick vs healthy lungs
        - Binary classification (2 outputs)
        """)

    st.markdown("---")
    st.markdown("### Training Configuration")
    config = {
        "Parameter": ["Optimizer", "Learning Rate", "Epochs", "Batch Size", "Loss Function", "Train/Test Split"],
        "Value": ["Adam", "0.0001", "10", "16", "Cross Entropy Loss", "80% / 20%"]
    }
    import pandas as pd
    st.table(pd.DataFrame(config))

    st.markdown("---")
    st.markdown("### Overfitting Note")
    st.info("""
    The model reached 100% training accuracy by epoch 3, which indicates overfitting 
    (memorizing the training images). Despite this, it still achieved 82.5% on unseen 
    test images — thanks to ResNet's strong pretrained visual representations carrying over.
    
    Next steps to reduce overfitting: more training images, dropout layers, early stopping.
    """)


# ─── TAB 5: ABOUT ME ───────────────────────────────────────────────────────────
with tabs[4]:
    st.title("👤 About Me")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.image("photo.jpg", width=150)

    with col2:
        st.markdown("""
        ### Waris Chaopricha
        **Student | AI & ML Enthusiast**
        
        I'm a 17-year-old student passionate about using AI to solve real-world problems, 
        particularly in healthcare. This project was built to explore how deep learning 
        can assist in medical image analysis.
        """)

    st.markdown("---")

    st.markdown("### About This Project")
    st.markdown("""
    This Pneumothorax detector was built as a personal project to learn:
    - Medical image classification
    - Transfer learning with PyTorch
    - Building and deploying ML models
    - The full pipeline from raw data to working application
    
    The project went through two full iterations — starting with a basic 
    SVM approach (~57% accuracy) and evolving to a ResNet18 deep learning 
    model achieving **82.5% accuracy**.
    """)

    st.markdown("---")

    st.markdown("### Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">🔥</div><div class="stat-label">PyTorch</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">🌊</div><div class="stat-label">Streamlit</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">📓</div><div class="stat-label">Kaggle</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><div class="stat-number">🏥</div><div class="stat-label">NIH Dataset</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Disclaimer:</strong> This project is for educational purposes only. 
        It is not a certified medical device and should never be used for actual clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)
