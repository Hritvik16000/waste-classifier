# src/06_export.py
import argparse, torch, torchvision
from pathlib import Path
from 02_train import make_model, CLASSES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best_model.pth")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--quantize_dynamic", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    arch = ck.get("arch","efficientnet_b0")
    model = make_model(arch, num_classes=len(CLASSES), freeze=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # TorchScript
    example = torch.randn(1,3,args.img_size,args.img_size)
    ts = torch.jit.trace(model, example)
    ts_path = f"{args.outdir}/model_ts.pt"
    ts.save(ts_path)
    print("Saved TorchScript:", ts_path)

    # ONNX
    onnx_path = f"{args.outdir}/model.onnx"
    torch.onnx.export(model, example, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17)
    print("Saved ONNX:", onnx_path)

    if args.quantize_dynamic:
        model_q = torch.ao.quantization.quantize_dynamic(model, dtype=torch.qint8)
        ts_q = torch.jit.trace(model_q, example)
        ts_q_path = f"{args.outdir}/model_ts_int8.pt"
        ts_q.save(ts_q_path)
        print("Saved dynamic-quantized TorchScript:", ts_q_path)

if __name__ == "__main__":
    main()
