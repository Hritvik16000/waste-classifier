from pathlib import Path
import urllib.request, streamlit as st

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
CKPT_PATH = MODEL_DIR / "best_model.pth"
CALIB_PATH = MODEL_DIR / "calibration.json"

def ensure_assets():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT_PATH.exists():
        st.info("Downloading model…")
        urllib.request.urlretrieve(st.secrets["MODEL_URL"], CKPT_PATH)
    if "CALIB_URL" in st.secrets and not CALIB_PATH.exists():
        urllib.request.urlretrieve(st.secrets["CALIB_URL"], CALIB_PATH)


from pathlib import Path
import os

# Find absolute paths safely
FILE_DIR = Path(__file__).resolve().parent   # /path/to/waste-classifier/src
ROOT_DIR = FILE_DIR.parent                   # /path/to/waste-classifier
MODEL_DIR = ROOT_DIR / "models"

CKPT_PATH = MODEL_DIR / "best_model.pth"
CALIB_PATH = MODEL_DIR / "calibration.json"

# src/04_streamlit_app.py
import os
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch, torchvision
from PIL import Image

# ---------- constants ----------
CLASSES = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]

# Resolve project root from THIS file (…/src/04_streamlit_app.py -> project root is parent of src/)
FILE_DIR = Path(__file__).resolve().parent         # .../src
PROJECT_ROOT = FILE_DIR.parent                     # repo root (parent of src)

# Allow overriding model dir via env var if you prefer (e.g., set MODEL_DIR=/persist/models)
MODELS_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models")).resolve()

CKPT_PATH = MODELS_DIR / "best_model.pth"
CALIB_PATH = MODELS_DIR / "calibration.json"


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_calibration(ckpt_path=CKPT_PATH, cal_path=CALIB_PATH):
    """Load model checkpoint and (optional) temperature calibration, using absolute, robust paths."""
    ckpt_path = Path(ckpt_path)
    cal_path = Path(cal_path)

    # Helpful diagnostics if files are missing
    missing = []
    if not ckpt_path.exists():
        missing.append(f"model: {ckpt_path}")
    if not cal_path.exists():
        # calibration is optional; we won't block on it, but we’ll warn below
        pass

    if missing:
        # Try to help by searching the repo for similarly named files
        found_ckpts = list(PROJECT_ROOT.rglob("best_model.pth"))
        found_calibs = list(PROJECT_ROOT.rglob("calibration.json"))
        msg = [
            "Model assets not found.",
            *[f"- Expected: {p}" for p in missing],
        ]
        if found_ckpts or found_calibs:
            msg.append("Discovered similar files:")
            for p in found_ckpts:
                msg.append(f"  • {p}")
            for p in found_calibs:
                msg.append(f"  • {p}")
        else:
            msg.append("No similar files found anywhere under the repo.")
        st.error("\n".join(msg))
        st.stop()

    device = "cpu"
    ck = torch.load(ckpt_path, map_location=device)

    arch = ck.get("arch", "efficientnet_b0")
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=None)
        in_feats = m.fc.in_features
        m.fc = torch.nn.Linear(in_feats, len(CLASSES))
    elif arch == "mobilenetv2":
        m = torchvision.models.mobilenet_v2(weights=None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_feats, len(CLASSES))
    else:
        m = torchvision.models.efficientnet_b0(weights=None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_feats, len(CLASSES))

    m.load_state_dict(ck["model"], strict=True)
    m.eval()

    # Temperature (optional)
    T = 1.0
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                T = float(json.load(f).get("temperature", 1.0))
        except Exception as e:
            st.warning(f"Failed to read calibration.json ({cal_path}): {e}. Using T=1.0")

    return m, T


def preprocess(img, size=224):
    from torchvision import transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf(img).unsqueeze(0)


def predict_topk(model, tensor, T=1.0, k=3):
    with torch.no_grad():
        logits = model(tensor) / T
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idxs = probs.argsort()[::-1][:k]
    return [(CLASSES[i], float(probs[i])) for i in idxs]


# ---------- UI ----------
st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="centered")
st.title("🗑️ Garbage Classifier")
st.caption("Upload or snap a photo. Model uses calibrated probabilities and an abstain threshold.")

# Load using absolute, robust paths
model, T = load_model_and_calibration(CKPT_PATH, CALIB_PATH)

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
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
