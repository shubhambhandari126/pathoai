#!/usr/bin/env python3
"""Train on a real, openly-available histopathology dataset (no manual download).

Uses MedMNIST's PathMNIST (100k+ real colorectal H&E image tiles, 9 tissue
classes, auto-downloaded from Zenodo). We binarise it to the task the app uses:

    tumour  = colorectal adenocarcinoma epithelium (class 8)
    normal  = everything else

It trains a ResNet18, reports honest held-out AUROC + calibration (ECE), and
saves a checkpoint the app loads directly:

    PATHQA_MODEL=model.pt ./pathqa-app      (or set it in your host)

Run on a GPU (Colab is free). CPU works but is slow.

    pip install medmnist
    python train_open.py --epochs 3 --out model.pt
    # quick test:  python train_open.py --epochs 1 --subset 4000 --out model.pt

For the closer-to-real PatchCamelyon set instead, use train_pcam.py.
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

TUMOUR_CLASS = 8  # PathMNIST: colorectal adenocarcinoma epithelium (TUM)


class BinaryPathMNIST(Dataset):
    """PathMNIST wrapped as binary tumour(=1)/normal(=0), resized to 96px."""

    def __init__(self, split: str, train: bool):
        from medmnist import PathMNIST
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if train else []
        self.tf = transforms.Compose([transforms.Resize(96), *aug,
                                      transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.ds = PathMNIST(split=split, download=True, as_rgb=True, size=28)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, y = self.ds[i]              # PIL image, label array shape (1,)
        label = 1 if int(y[0]) == TUMOUR_CLASS else 0
        return self.tf(img), label


def loaders(batch_size, subset):
    tr, va, te = (BinaryPathMNIST("train", True), BinaryPathMNIST("val", False),
                  BinaryPathMNIST("test", False))
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--subset", type=int, default=None, help="cap samples per split for a quick run")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_open] device={device}  dataset=PathMNIST (binarised tumour vs rest)")
    train_dl, val_dl, test_dl = loaders(args.batch_size, args.subset)

    model = build_model("resnet18", pretrained=True).to(device)
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
    print("\n=== Held-out test metrics (real data) ===")
    print(f"  accuracy {test.accuracy:.3f} | AUROC {test.auroc:.3f} | ECE {ece:.3f} | T {T:.3f} | n {test.n}")

    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18", "temperature": T,
                "labels": {0: "normal", 1: "tumor"},
                "metrics": {"test_auroc": test.auroc, "test_acc": test.accuracy, "ece": ece},
                "note": "Pathify model trained on PathMNIST (open data). QA aid, not a diagnosis."},
               args.out)
    print(f"[saved] {args.out}  ->  run with  PATHQA_MODEL={args.out}")


if __name__ == "__main__":
    main()
