# src/01_prepare_data.py
import argparse, os, re
from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter

CANONICAL = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]

# Dataset synonyms
LABEL_MAP_RULES = [
    (r"cardboard|carton", "cardboard"),
    (r"glass", "glass"),
    (r"metal|aluminum|aluminium|steel|tin|can", "metal"),
    (r"organic|food|bio|compost", "organic"),
    (r"paper", "paper"),
    (r"plastic|poly|pet|hdpe|bottle", "plastic"),
    (r"trash|other|residual|mixed", "trash"),  # handled by CLI flags
]

def map_label(raw_name, trash_to_organic=False, drop_trash=False):
    name = raw_name.lower()
    for pat, lab in LABEL_MAP_RULES:
        if re.search(rf"(^|[^a-z])({pat})([^a-z]|$)", name):
            if lab == "trash":
                if trash_to_organic:
                    return "organic"
                if drop_trash:
                    return None
                # If neither flag set, leave as non-canonical to filter later
                return None
            return lab
    return None

def collect_images(root, trash_to_organic=False, drop_trash=False):
    root = Path(root)
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
            # try parent dir first, then filename
            label = map_label(p.parent.name, trash_to_organic, drop_trash)
            if label is None:
                label = map_label(p.name, trash_to_organic, drop_trash)
            if label in CANONICAL:
                rows.append({"path": str(p.resolve()), "label": label})
    return pd.DataFrame(rows)

def write_split_txt(df, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"train.txt").write_text("\n".join(f"{r.path} {r.label}" for r in df[df.split=="train"].itertuples()))
    (out/"val.txt").write_text("\n".join(f"{r.path} {r.label}" for r in df[df.split=="val"].itertuples()))
    (out/"test.txt").write_text("\n".join(f"{r.path} {r.label}" for r in df[df.split=="test"].itertuples()))

def persist_resized(df, size, out_root):
    out_root = Path(out_root)
    for split, g in df.groupby("split"):
        for cls in CANONICAL:
            (out_root/split/cls).mkdir(parents=True, exist_ok=True)
        for r in g.itertuples():
            img = cv2.imread(r.path)
            if img is None: 
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            dst = out_root / split / r.label / (Path(r.path).stem + ".jpg")
            cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def eda_plots(df, out_dir="reports"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Class counts
    counts = df["label"].value_counts().reindex(CANONICAL, fill_value=0)
    plt.figure(figsize=(7,4))
    counts.plot(kind="bar")
    plt.title("Images per class")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/class_counts.png", dpi=150); plt.close()

    # Random grid of examples (up to 12)
    sample = df.sample(min(12, len(df)), random_state=42)
    cols = 4; rows = int(np.ceil(len(sample)/cols))
    plt.figure(figsize=(cols*3, rows*3))
    for i, r in enumerate(sample.itertuples(), 1):
        img = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i); plt.imshow(img); plt.axis("off"); plt.title(r.label)
    plt.tight_layout(); plt.savefig(f"{out_dir}/sample_grid.png", dpi=150); plt.close()

    # Color histogram (RGB means)
    means = []
    for r in sample.itertuples():
        img = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
        means.append(img.reshape(-1,3).mean(0))
    if means:
        means = np.array(means)
        plt.figure(figsize=(5,3))
        plt.bar(["R","G","B"], means.mean(0))
        plt.title("Average color (sample)")
        plt.tight_layout(); plt.savefig(f"{out_dir}/avg_color.png", dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--processed_dir", default="processed")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--persist_resized", action="store_true")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    # NEW: label handling flags
    ap.add_argument("--trash_to_organic", action="store_true", help="Map 'trash' to 'organic'")
    ap.add_argument("--drop_trash", action="store_true", help="Drop 'trash' images entirely")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    df = collect_images(args.raw_dir, args.trash_to_organic, args.drop_trash)
    assert len(df)>0, "No images found after mapping. Check data/raw/ and flags."

    # stratified split
    trainval, test = train_test_split(df, test_size=args.test_size, stratify=df.label, random_state=args.seed)
    train, val = train_test_split(trainval, test_size=args.val_size/(1-args.test_size), stratify=trainval.label, random_state=args.seed)
    train["split"]="train"; val["split"]="val"; test["split"]="test"
    df_all = pd.concat([train, val, test], ignore_index=True)

    write_split_txt(df_all, args.splits_dir)
    if args.persist_resized:
        persist_resized(df_all, args.img_size, args.processed_dir)
    eda_plots(df_all, out_dir="reports")

    print("Prepared splits:", df_all.split.value_counts().to_dict())
    print("Per-class counts (train):", Counter(train.label))

if __name__ == "__main__":
    main()
