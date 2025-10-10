# src/04_streamlit_app.py
import streamlit as st
import torch, torchvision, json
from PIL import Image
from pathlib import Path
import numpy as np

CLASSES = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]

@st.cache_resource
def load_model_and_calibration(ckpt_path="models/best_model.pth", cal_path="models/calibration.json"):
    device = "cpu"
    ck = torch.load(ckpt_path, map_location=device)
    arch = ck.get("arch","efficientnet_b0")
    if arch=="resnet50":
        m = torchvision.models.resnet50(weights=None)
        in_feats = m.fc.in_features; m.fc = torch.nn.Linear(in_feats, len(CLASSES))
    elif arch=="mobilenetv2":
        m = torchvision.models.mobilenet_v2(weights=None)
        in_feats = m.classifier[-1].in_features; m.classifier[-1] = torch.nn.Linear(in_feats, len(CLASSES))
    else:
        m = torchvision.models.efficientnet_b0(weights=None)
        in_feats = m.classifier[-1].in_features; m.classifier[-1] = torch.nn.Linear(in_feats, len(CLASSES))
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    T = 1.0
    if Path(cal_path).exists():
        with open(cal_path) as f:
            T = float(json.load(f).get("temperature", 1.0))
    return m, T

def preprocess(img, size=224):
    from torchvision import transforms
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return tf(img).unsqueeze(0)

def predict_topk(model, tensor, T=1.0, k=3):
    with torch.no_grad():
        logits = model(tensor)/T
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idxs = probs.argsort()[::-1][:k]
    return [(CLASSES[i], float(probs[i])) for i in idxs]

st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="centered")
st.title("🗑️ Garbage Classifier")
st.caption("Upload or snap a photo. Model uses calibrated probabilities and an abstain threshold.")

model, T = load_model_and_calibration("models/best_model.pth", "models/calibration.json")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
with col2:
    cam = st.camera_input("Or take a photo")

img_file = uploaded or cam
th = st.slider("Abstain threshold (top-1 confidence must exceed this)", 0.0, 0.95, 0.60, 0.01)
k = st.slider("Top-k to display", 1, 5, 3, 1)

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Input", use_column_width=True)
    x = preprocess(image)
    preds = predict_topk(model, x, T=T, k=k)

    top1_label, top1_p = preds[0]
    if top1_p < th:
        st.error(f"Unsure (max confidence {top1_p*100:.1f}% < threshold). Try a clearer photo or different angle.")
    else:
        st.success(f"Prediction: **{top1_label}** ({top1_p*100:.1f}%)")

    st.subheader("Top-k")
    for cls, p in preds:
        st.write(f"- {cls}: {p*100:.1f}%")
