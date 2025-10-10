# src/07_api.py
from fastapi import FastAPI, UploadFile, File
import uvicorn, io, torch, torchvision, json
from PIL import Image
from 04_streamlit_app import CLASSES, preprocess

app = FastAPI(title="Waste Classifier API")

device = "cpu"
ck = torch.load("models/best_model.pth", map_location=device)
arch = ck.get("arch","efficientnet_b0")
if arch=="resnet50":
    m = torchvision.models.resnet50(weights=None); in_feats = m.fc.in_features; m.fc = torch.nn.Linear(in_feats, len(CLASSES))
elif arch=="mobilenetv2":
    m = torchvision.models.mobilenet_v2(weights=None); in_feats = m.classifier[-1].in_features; m.classifier[-1] = torch.nn.Linear(in_feats, len(CLASSES))
else:
    m = torchvision.models.efficientnet_b0(weights=None); in_feats = m.classifier[-1].in_features; m.classifier[-1] = torch.nn.Linear(in_feats, len(CLASSES))
m.load_state_dict(ck["model"]); m.eval()

T = 1.0
try:
    with open("models/calibration.json") as f:
        T = float(json.load(f).get("temperature", 1.0))
except:
    pass

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess(image)
    with torch.no_grad():
        logits = m(x)/T
        probs = torch.softmax(logits, dim=1).numpy()[0].tolist()
    top = max(enumerate(probs), key=lambda t: t[1])
    return {"top1": {"label": CLASSES[top[0]], "prob": float(top[1])},
            "probs": {CLASSES[i]: float(p) for i,p in enumerate(probs)}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
