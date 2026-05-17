import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os
import pandas as pd
import streamlit.components.v1 as components

# Download the v3 model from Google Drive
if not os.path.exists("pneumothorax_model.pth"):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1IojOmzUuJXH_Dq6unoDnRPVWU2rFd1pV",
            "pneumothorax_model.pth",
            quiet=False
        )

st.set_page_config(
    page_title="Pneumothorax Detector by Waris Chaopricha",
    page_icon="🫁",
    layout="wide"
)

# Hidden structured data so Google knows this website belongs to Waris Chaopricha
schema = """
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Waris Chaopricha",
  "url": "https://pneumothorax.streamlit.app/",
  "worksFor": {
    "@type": "WebSite",
    "name": "Pneumothorax AI Detector",
    "url": "https://pneumothorax.streamlit.app/"
  },
  "sameAs": ["https://waris-marketwatch.vercel.app/"]
}
</script>
"""
components.html(schema, height=0)

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

tabs = st.tabs(["Detector", "How It Works", "Dataset", "The Model", "About Me"])


# ─── TAB 1: DETECTOR ───────────────────────────────────────────────────────────
with tabs[0]:
    st.title("Pneumothorax AI Detector")
    st.caption("Built by **Waris Chaopricha**")
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
                    <h2>Pneumothorax Detected</h2>
                    <p style="font-size:1.1rem">Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box healthy-box">
                    <h2>No Pneumothorax Detected</h2>
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
                <strong>Medical Disclaimer:</strong> This tool is an AI research project 
                and is NOT a medical device. Do not use for actual medical diagnosis. 
                Always consult a qualified physician.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload an X-ray image on the left to get started")
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
    st.title("How It Works")
    st.markdown("This project went through **3 full iterations**, evolving from 57% to **83% accuracy** on 694 unseen X-rays.")

    st.markdown("---")

    # ─── Iteration 1 ───────────────────────────────────────────────────────────
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

    # ─── Iteration 2 ───────────────────────────────────────────────────────────
    st.markdown("### Iteration 2 — ResNet18 Transfer Learning (Small Dataset)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">200</div><div class="stat-label">X-rays trained on</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">10</div><div class="stat-label">Training epochs</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">80%</div><div class="stat-label">Final test accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("""
    Instead of hand-crafting features, the raw X-ray pixels were fed directly into 
    ResNet18 — a deep convolutional neural network pretrained on 1.2 million everyday photos.
    The model learned what Pneumothorax looks like on its own. However, with only 200 images, 
    the model memorized the training set by epoch 3 (overfitting).
    """)

    st.markdown("#### Training Progress (Iteration 2)")
    epochs_v2 = list(range(1, 11))
    train_acc_v2 = [58.75, 97.50, 100, 100, 100, 100, 100, 100, 100, 100]
    loss_v2 = [0.6937, 0.1376, 0.0575, 0.0165, 0.0082, 0.0031, 0.0047, 0.0042, 0.0032, 0.0049]

    chart_v2 = pd.DataFrame({
        "Train Accuracy (%)": train_acc_v2,
        "Loss x 100": [l * 100 for l in loss_v2]
    }, index=epochs_v2)
    st.line_chart(chart_v2)
    st.caption("Train accuracy jumped to 100% by epoch 3 — a classic sign of overfitting on a small dataset.")

    st.markdown("---")

    # ─── Iteration 3 ───────────────────────────────────────────────────────────
    st.markdown("### Iteration 3 — ResNet18 with Larger Dataset + Class Weights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">4,694</div><div class="stat-label">X-rays trained on</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">10</div><div class="stat-label">Training epochs</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">83%</div><div class="stat-label">Final test accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("""
    For Iteration 3, the dataset was scaled up **23x larger** — from 200 to 4,694 images 
    (2,194 Pneumothorax + 2,500 Healthy). The model was also upgraded with:
    
    - **Class weights** to balance the slight imbalance between sick and healthy cases
    - **Stronger data augmentation** (rotation, horizontal flip) to prevent memorization
    - **GPU training** on Kaggle's Tesla T4 for fast experimentation
    - **Larger test set** (694 unseen images vs. just 40 before) — making the 83% accuracy 
      a far more reliable measurement
    
    The result: the model learned more **gradually** (not memorizing instantly), reaching 
    96% train accuracy by epoch 3 instead of 100%. This indicates healthier learning behavior.
    """)

    st.markdown("#### Training Progress (Iteration 3)")
    epochs_v3 = list(range(1, 11))
    train_acc_v3 = [75.20, 89.75, 96.08, 98.15, 98.17, 98.25, 97.88, 98.05, 98.12, 98.42]
    loss_v3 = [0.5072, 0.2534, 0.1099, 0.0533, 0.0511, 0.0456, 0.0560, 0.0578, 0.0491, 0.0409]

    chart_v3 = pd.DataFrame({
        "Train Accuracy (%)": train_acc_v3,
        "Loss x 100": [l * 100 for l in loss_v3]
    }, index=epochs_v3)
    st.line_chart(chart_v3)
    st.caption("Smoother learning curve — the model takes 3-4 epochs to converge instead of memorizing instantly.")

    st.markdown("---")

    # ─── Comparison ───────────────────────────────────────────────────────────
    st.markdown("### Iteration Comparison")
    comparison = pd.DataFrame({
        "Iteration": ["v1 - SVM", "v2 - ResNet18 Small", "v3 - ResNet18 Large"],
        "Training Images": [200, 200, 4694],
        "Test Set Size": [40, 40, 694],
        "Test Accuracy": ["~57%", "80%", "83%"],
        "Notes": [
            "Hand-crafted features failed",
            "Overfit by epoch 3",
            "Healthier learning curve, larger reliable test set"
        ]
    })
    st.table(comparison)


# ─── TAB 3: DATASET ────────────────────────────────────────────────────────────
with tabs[2]:
    st.title("Dataset")
    st.markdown("The model was trained on data from the **NIH Chest X-ray Dataset**.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">112,120</div><div class="stat-label">Total X-rays in NIH dataset</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">14</div><div class="stat-label">Diseases labeled</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### What Iteration 3 Uses")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Positive cases (sick)**
        - 2,194 X-rays labeled as Pneumothorax
        - All available Pneumothorax cases in the NIH dataset
        - Patients with a confirmed collapsed lung
        """)
    with col2:
        st.markdown("""
        **Control cases (healthy)**
        - 2,500 X-rays labeled as "No Finding"
        - Patients with completely normal lungs
        - Selected from 60,361 available healthy cases
        """)

    st.markdown("---")

    st.markdown("### Train / Test Split")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">4,694</div><div class="stat-label">Total images used</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">4,000</div><div class="stat-label">Training set</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">694</div><div class="stat-label">Unseen test set</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Why Scale Up From 200 to 4,694?
    Iteration 2 only used 200 images and tested on just 40 — way too small to trust 
    the results. The model also overfit instantly because there wasn't enough variety 
    to actually learn from.
    
    Iteration 3 used **every available Pneumothorax case** in the NIH dataset (2,194) 
    plus a balanced number of healthy controls (2,500). This made the training more 
    realistic and produced a test accuracy (83%) on 694 unseen images that is much 
    more trustworthy than the previous 80% on only 40 images.
    """)


# ─── TAB 4: THE MODEL ──────────────────────────────────────────────────────────
with tabs[3]:
    st.title("The Model")

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
    st.markdown("### Training Configuration (Iteration 3)")
    config = {
        "Parameter": [
            "Optimizer", "Learning Rate", "Epochs", "Batch Size",
            "Loss Function", "Train/Test Split", "Dataset Size",
            "Class Weights", "Augmentation", "Hardware"
        ],
        "Value": [
            "Adam", "0.0001", "10", "32",
            "Cross Entropy Loss (weighted)", "4000 / 694", "4,694 images",
            "[0.94, 1.07]", "Random Flip + Rotation", "Tesla T4 GPU (Kaggle)"
        ]
    }
    st.table(pd.DataFrame(config))

    st.markdown("---")
    st.markdown("### Overfitting Analysis")
    st.info("""
    **Iteration 2** reached 100% training accuracy by epoch 3 — clear memorization 
    on the tiny 200-image dataset.
    
    **Iteration 3** reached 96% by epoch 3 and plateaued at 98% — a much healthier 
    curve, but still some overfitting. The 15% gap between train (98%) and test (83%) 
    accuracy suggests the model could benefit from:
    - Dropout layers
    - Early stopping (training stops when validation accuracy plateaus)
    - Even more data
    - Weight decay regularization
    
    These improvements are planned for a future Iteration 4.
    """)


# ─── TAB 5: ABOUT ME ───────────────────────────────────────────────────────────
with tabs[4]:
    st.title("About Me")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.image("photo.jpg", width=150)

    with col2:
        st.markdown("""
        ### Waris Chaopricha
        **Student | AI & ML Enthusiast**
        
        I'm a 17-year-old student passionate about using AI to solve real-world problems, 
        particularly in healthcare. 
        
        This Pneumothorax detector and my other project **[MarketWatch](https://waris-marketwatch.vercel.app/)** 
        (a live stock & crypto tracker) were built to explore real-world applications of AI and full-stack development.
        """)

    st.markdown("---")

    st.markdown("### About This Project")
    st.markdown("""
    This Pneumothorax detector was built as a personal project to learn:
    - Medical image classification
    - Transfer learning with PyTorch
    - Building and deploying ML models
    - The full pipeline from raw data to working application
    
    The project went through three full iterations — starting with a basic 
    SVM approach (~57% accuracy), evolving to a small ResNet18 (80%), and finally 
    scaling up to a 4,694-image ResNet18 model achieving **83% accuracy** on 
    694 unseen X-rays.
    """)

    st.markdown("---")

    st.markdown("### Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">PyTorch</div><div class="stat-label">Deep learning framework</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">Streamlit</div><div class="stat-label">Web app framework</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-number">Kaggle</div><div class="stat-label">GPU training</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><div class="stat-number">NIH</div><div class="stat-label">Dataset source</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <strong>Disclaimer:</strong> This project is for educational purposes only. 
        It is not a certified medical device and should never be used for actual clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)
