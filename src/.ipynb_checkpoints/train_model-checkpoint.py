# src/05_calibrate_temperature.py
import argparse, torch, json
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from pathlib import Path
from 02_train import ManifestDataset, make_model, CLASSES
import numpy as np

def eval_nll(model, loader, T=1.0, device="cpu"):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)/T
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(probs)
    return log_loss(ys, ps, labels=list(range(len(CLASSES))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--ckpt", default="models/best_model.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = ManifestDataset(f"{args.splits_dir}/val.txt", args.img_size, aug=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    ck = torch.load(args.ckpt, map_location=device)
    arch = ck.get("arch","efficientnet_b0")
    model = make_model(arch, num_classes=len(CLASSES), freeze=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # grid search temperature
    Ts = np.linspace(0.5, 5.0, 46)
    best = (1e9, 1.0)
    for T in Ts:
        nll = eval_nll(model, val_loader, T, device)
        if nll < best[0]:
            best = (nll, T)

    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/calibration.json","w") as f:
        json.dump({"temperature": best[1]}, f, indent=2)
    print(f"Saved models/calibration.json with T={best[1]:.3f}")

if __name__ == "__main__":
    main()
