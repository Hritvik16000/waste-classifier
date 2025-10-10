# src/evaluate_model.py

import argparse
import torch
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
from train_model import ManifestDataset, make_model, CLASSES, evaluate
from torch.utils.data import DataLoader
from utils import ConfusionMatrixPlotter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models", help="Path to saved models")
    ap.add_argument("--splits_dir", default="data/splits", help="Path to dataset splits")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    # load test set
    test_ds = ManifestDataset(Path(args.splits_dir)/"test.txt", args.img_size, aug=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    best = (-1, None, None, None)  # best_f1, arch, y_true, y_pred

    # Look for saved checkpoints
    for p in Path(args.models_dir).glob("*.pth"):
        ck = torch.load(p, map_location=device)
        arch = ck.get("arch", "efficientnet_b0")

        # rebuild model
        model = make_model(arch, num_classes=len(CLASSES), freeze=False).to(device)
        model.load_state_dict(ck["model"])
        acc, f1, report, (yt, yp) = evaluate(model, test_loader, device)

        rows.append({"model": arch, "file": p.name, "acc": acc, "f1": f1})

        if f1 > best[0]:
            best = (f1, arch, yt, yp)

    # Save results
    Path("reports").mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(rows)
    df.to_csv("reports/comparison.csv", index=False)

    if best[1]:
        ConfusionMatrixPlotter(CLASSES).save(best[2], best[3], out_path=f"reports/cm_best_{best[1]}.png")

    print("Saved evaluation results to reports/comparison.csv")
    print("Best model:", best[1], "F1 =", best[0])

if __name__ == "__main__":
    main()
