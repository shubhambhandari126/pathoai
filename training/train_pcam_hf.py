#!/usr/bin/env python3
"""Train on PatchCamelyon via the Hugging Face mirror (reliable download).

Uses `zacharielegault/PatchCamelyon` (327k 96px lymph-node tiles, binary
tumour/normal), which downloads through the Hugging Face datasets library, no
Google-Drive quota problems like torchvision's built-in PCam.

Continue-training your existing model (the 'merge'):

    pip install datasets
    python train_pcam_hf.py --base model.pt --epochs 2 --out model.pt
    # quick test:  python train_pcam_hf.py --base model.pt --subset 8000 --epochs 1 --out model.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from pathqa.data import IMAGENET_MEAN, IMAGENET_STD
from pathqa.engine import (evaluate, expected_calibration_error, fit_temperature,
                           train_one_epoch)
from pathqa.model import TemperatureScaler, build_model

HF_ID = "zacharielegault/PatchCamelyon"   # image (96px) + label (0 normal / 1 tumour)


class HFPatchCam(Dataset):
    def __init__(self, split: str, train: bool):
        from datasets import load_dataset
        self.ds = load_dataset(HF_ID, split=split)
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if train else []
        self.tf = transforms.Compose([transforms.Resize(96), *aug,
                                      transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return self.tf(r["image"].convert("RGB")), int(r["label"])


def loaders(batch_size, subset):
    tr, va, te = (HFPatchCam("train", True), HFPatchCam("validation", False), HFPatchCam("test", False))
    if subset:
        tr = Subset(tr, range(min(subset, len(tr))))
        va = Subset(va, range(min(subset // 4 or 1, len(va))))
        te = Subset(te, range(min(subset // 4 or 1, len(te))))
    return (DataLoader(tr, batch_size, shuffle=True, num_workers=2),
            DataLoader(va, batch_size, num_workers=2),
            DataLoader(te, batch_size, num_workers=2))


@torch.no_grad()
def _probs_labels(model, loader, device, scaler=None):
    model.eval(); ps, ys = [], []
    for x, y in loader:
        lo = model(x.to(device))
        if scaler is not None:
            lo = scaler(lo)
        ps.append(F.softmax(lo, 1)[:, 1].cpu()); ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--base", default=None, help="checkpoint to continue from")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_pcam_hf] device={device}  dataset={HF_ID}")
    train_dl, val_dl, test_dl = loaders(args.batch_size, args.subset)

    model = build_model("resnet18", pretrained=True).to(device)
    if args.base:
        ck = torch.load(args.base, map_location=device)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        print(f"[train_pcam_hf] continuing from {args.base}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        loss = train_one_epoch(model, train_dl, optim, device)
        res = evaluate(model, val_dl, device)
        print(f"[epoch {ep+1}/{args.epochs}] loss={loss:.3f} val_acc={res.accuracy:.3f} val_auroc={res.auroc:.3f}")
        if res.auroc >= best:
            best, best_state = res.auroc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    T = fit_temperature(model, val_dl, device)
    scaler = TemperatureScaler(); scaler.set_temperature(T)
    p, y = _probs_labels(model, test_dl, device, scaler)
    test = evaluate(model, test_dl, device, scaler=scaler)
    ece = expected_calibration_error(p, y)
    print(f"\n=== PatchCamelyon test === accuracy {test.accuracy:.3f} | AUROC {test.auroc:.3f} | ECE {ece:.3f} | T {T:.3f}")

    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18", "temperature": T,
                "labels": {0: "normal", 1: "tumor"},
                "metrics": {"test_auroc": test.auroc, "test_acc": test.accuracy, "ece": ece},
                "note": "Pathify model, trained/continued on PatchCamelyon (HF mirror)."}, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
