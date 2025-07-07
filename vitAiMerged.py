import streamlit as st
from PIL import Image
import torch
import requests
import os
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTFeatureExtractor
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 🎨 Streamlit page settings
st.set_page_config(page_title="🧠 Brain Tumor Classifier", page_icon="🧠", layout="centered")

# 💄 Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            text-align: center;
            color: #31333F;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.markdown("<h1>🧠 Brain Tumor Classification</h1>", unsafe_allow_html=True)
st.markdown("<h3>Using Vision Transformer (ViT)</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# 🧠 Load the model and processor
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained base model
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=4,
        id2label={0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"},
        label2id={"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3},
        ignore_mismatched_sizes=True
    ).to(device)

    # Load weights from Hugging Face (auto-download)
    MODEL_URL = "https://huggingface.co/Bhuvanmj/vit-model/resolve/main/vit_brain_tumor_5class.pth"
    model_path = "vit_brain_tumor_5class.pth"

    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(MODEL_URL).content)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    return model, processor, device

model, processor, device = load_model()

# 📤 Upload section
st.markdown("### 📤 Upload a Brain MRI Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

predicted_label = None
conf_df = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", width=300)

    with st.spinner("🔍 Analyzing the image... Please wait..."):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

    label = model.config.id2label[predicted_class]
    confidence = probs[0][predicted_class].item()
    predicted_label = label

    # 🎯 Result
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ✅ Classification Result")
    st.success(f"🎯 **Predicted Class:** `{label.upper()}`")
    st.info(f"📊 **Confidence Score:** `{confidence:.2f}`")

    # 📊 Confidence Scores Table
    st.markdown("### 🔍 Confidence Scores for All Classes")
    conf_df = pd.DataFrame({
        "Tumor Type": [model.config.id2label[i].upper() for i in range(len(probs[0]))],
        "Confidence": [round(p.item(), 4) for p in probs[0]]
    })
    st.dataframe(conf_df.set_index("Tumor Type").sort_values("Confidence", ascending=False))

    # 🥧 Pie Chart
    st.markdown("### 📊 Visual Confidence Chart")
    fig = go.Figure(
        data=[go.Pie(
            labels=conf_df["Tumor Type"],
            values=conf_df["Confidence"],
            pull=[0.05]*len(conf_df),
            marker=dict(colors=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]),
            textinfo='label+percent',
            hole=0.3
        )]
    )
    fig.update_layout(margin=dict(l=40, r=40, t=100, b=40), height=450)
    st.plotly_chart(fig, use_container_width=True)

    # 💬 AI ChatBot
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 💬 Ask Anything About the Disease")

    if "last_image_name" not in st.session_state or st.session_state.last_image_name != uploaded_file.name:
        st.session_state.chat_history = []
        system_msg = {
            "role": "system",
            "content": f"The user uploaded a brain MRI image. The tumor was classified as **{predicted_label.upper()}** "
                       f"with the following confidence scores: " +
                       ", ".join([f"{row['Tumor Type']}: {row['Confidence']}" for _, row in conf_df.iterrows()]) +
                       ". Based on this, be prepared to answer medical, precautionary, or follow-up questions."
        }
        st.session_state.chat_history.append(system_msg)
        st.session_state.last_image_name = uploaded_file.name

    user_input = st.chat_input(f"Ask anything about {predicted_label.upper()}...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("🤖 Thinking..."):
            try:
                res = requests.post("https://groq-chat-api-fuxg.onrender.com/chat", json={"messages": st.session_state.chat_history})
                reply = res.json().get("response", "Sorry, I couldn't understand.")
            except Exception as e:
                reply = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.warning("📁 Please upload a brain MRI image to begin.")
