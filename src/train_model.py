import argparse, os, json
from pathlib import Path
import torch, torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from PIL import Image
from utils import seed_everything, ConfusionMatrixPlotter

CLASSES = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]

class ManifestDataset(Dataset):
    def __init__(self, manifest_file, img_size=224, aug=False):
        self.items = []
        with open(manifest_file) as f:
            for line in f:
                p, y = line.strip().rsplit(" ", 1)
                self.items.append((p, CLASSES.index(y)))
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if aug:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.7),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y

def make_model(name, num_classes=6, freeze=True):
    name = name.lower()
    if name == "baseline":
        # tiny baseline CNN
        return torch.nn.Sequential(
            torchvision.models._utils.Conv2dNormActivation(3, 32, 3, stride=2),
            torchvision.models._utils.Conv2dNormActivation(32,64,3),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )
    elif name in {"resnet50","mobilenetv2","efficientnet_b0"}:
        if name=="resnet50":
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            in_feats = m.fc.in_features
            m.fc = torch.nn.Linear(in_feats, num_classes)
            if freeze:
                for p in m.parameters(): p.requires_grad=False
                for p in m.fc.parameters(): p.requires_grad=True
            return m
        if name=="mobilenetv2":
            m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
            in_feats = m.classifier[-1].in_features
            m.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
            if freeze:
                for p in m.features.parameters(): p.requires_grad=False
            return m
        if name=="efficientnet_b0":
            m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_feats = m.classifier[-1].in_features
            m.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
            if freeze:
                for p in m.features.parameters(): p.requires_grad=False
            return m
    else:
        raise ValueError("Unknown model name")

def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train()
    total, correct, losses = 0,0,0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return losses/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x,y in loader:
        x = x.to(device)
        logits = model(x)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]
    return acc, macro_f1, report, (y_true, y_pred)

def class_weights_from_manifest(manifest_file):
    counts = {c: 0 for c in CLASSES}
    with open(manifest_file) as f:
        for line in f:
            _, lab = line.strip().rsplit(" ", 1)
            counts[lab] += 1
    n = sum(counts.values())
    weights = [n / counts[c] if counts[c] else 0.0 for c in CLASSES]
    return torch.tensor(weights, dtype=torch.float32), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--model", default="efficientnet_b0",
                    choices=["baseline","resnet50","mobilenetv2","efficientnet_b0"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ManifestDataset(Path(args.splits_dir)/"train.txt", args.img_size, aug=True)
    val_ds   = ManifestDataset(Path(args.splits_dir)/"val.txt", args.img_size, aug=False)
    test_ds  = ManifestDataset(Path(args.splits_dir)/"test.txt", args.img_size, aug=False)

    cls_w, counts = class_weights_from_manifest(Path(args.splits_dir)/"train.txt")
    weights_per_sample = []
    with open(Path(args.splits_dir)/"train.txt") as f:
        for line in f:
            _, lab = line.strip().rsplit(" ", 1)
            weights_per_sample.append(cls_w[CLASSES.index(lab)].item())

    sampler = torch.utils.data.WeightedRandomSampler(weights_per_sample,
                                                     num_samples=len(weights_per_sample),
                                                     replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = make_model(args.model, num_classes=len(CLASSES), freeze=args.freeze).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cls_w.to(device) if cls_w.sum()>0 else None)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_f1, best_path = -1.0, None
    Path(args.out_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, loss_fn)
        val_acc, val_f1, report, _ = evaluate(model, val_loader, device)
        print(f"[{epoch:02d}] loss={tr_loss:.4f} acc={tr_acc:.3f} | val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

        ckpt = Path(args.out_dir, "checkpoints", f"{args.model}_e{epoch}_f1{val_f1:.3f}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": val_f1,
                    "arch": args.model, "img_size": args.img_size}, ckpt)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = Path(args.out_dir, f"best_model_{args.model}.pth")
            torch.save({"model": model.state_dict(), "arch": args.model, "img_size": args.img_size}, best_path)

    test_acc, test_f1, test_report, (yt, yp) = evaluate(model, test_loader, device)
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump({"val_f1_best": best_f1, "test_acc": test_acc,
                   "test_f1": test_f1, "per_class": test_report}, f, indent=2)
    ConfusionMatrixPlotter(CLASSES).save(yt, yp, out_path="reports/cm.png")


if __name__ == "__main__":
    main()
