#!/usr/bin/env python3
"""Train on PathMNIST + PatchCamelyon MIXED together (balanced, no forgetting).

Instead of training on one dataset then the other (which makes the model forget
the first), this mixes both datasets into one training set and trains a single
model on the combination. The result is genuinely strong on BOTH colorectal
(PathMNIST) and lymph-node (PatchCamelyon) tissue.

    pip install medmnist datasets
    python train_combined.py --epochs 2 --out model.pt
    # quick test:  python train_combined.py --subset 6000 --epochs 1 --out model.pt

Reuses the same loaders as train_open.py (PathMNIST) and train_pcam_hf.py (PCam).
Run on a GPU (Colab). Both are open datasets that download automatically.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from pathqa.engine import (evaluate, expected_calibration_error, fit_temperature,
                           train_one_epoch)
from pathqa.model import TemperatureScaler, build_model
from train_open import BinaryPathMNIST      # PathMNIST, binarised tumour vs rest
from train_pcam_hf import HFPatchCam          # PatchCamelyon via Hugging Face


def _cap(ds, n):
    return Subset(ds, range(min(n, len(ds)))) if n else ds


def build(subset, batch_size):
    # train splits from both datasets, mixed
    pm_tr, pc_tr = BinaryPathMNIST("train", True), HFPatchCam("train", True)
    pm_va, pc_va = BinaryPathMNIST("val", False), HFPatchCam("validation", False)
    pm_te, pc_te = BinaryPathMNIST("test", False), HFPatchCam("test", False)
    if subset:
        s4 = subset // 4 or 1
        pm_tr, pc_tr = _cap(pm_tr, subset), _cap(pc_tr, subset)
        pm_va, pc_va = _cap(pm_va, s4), _cap(pc_va, s4)
        pm_te, pc_te = _cap(pm_te, s4), _cap(pc_te, s4)
    train = ConcatDataset([pm_tr, pc_tr])
    val = ConcatDataset([pm_va, pc_va])
    dl = lambda d, sh: DataLoader(d, batch_size=batch_size, shuffle=sh, num_workers=2)
    # keep the two test sets separate so we can report each tissue's score
    return (dl(train, True), dl(val, False),
            dl(pm_te, False), dl(pc_te, False))


@torch.no_grad()
def _probs_labels(model, loader, device, scaler):
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
    ap.add_argument("--subset", type=int, default=None, help="cap samples per dataset (quick test)")
    ap.add_argument("--base", default=None, help="optional checkpoint to start from")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_combined] device={device}  datasets=PathMNIST + PatchCamelyon (mixed)")
    train_dl, val_dl, pm_test_dl, pc_test_dl = build(args.subset, args.batch_size)
    print(f"[train_combined] mixed train batches={len(train_dl)}  val batches={len(val_dl)}")

    model = build_model("resnet18", pretrained=True).to(device)
    if args.base:
        ck = torch.load(args.base, map_location=device)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        print(f"[train_combined] starting from {args.base}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        loss = train_one_epoch(model, train_dl, optim, device)
        res = evaluate(model, val_dl, device)
        print(f"[epoch {ep+1}/{args.epochs}] loss={loss:.3f} mixed_val_acc={res.accuracy:.3f} mixed_val_auroc={res.auroc:.3f}")
        if res.auroc >= best:
            best, best_state = res.auroc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    T = fit_temperature(model, val_dl, device)
    scaler = TemperatureScaler(); scaler.set_temperature(T)
    pm = evaluate(model, pm_test_dl, device, scaler=scaler)
    pc = evaluate(model, pc_test_dl, device, scaler=scaler)
    p, y = _probs_labels(model, pc_test_dl, device, scaler)
    ece = expected_calibration_error(p, y)
    print("\n=== Held-out test (each tissue) ===")
    print(f"  PathMNIST (colorectal): acc {pm.accuracy:.3f} | AUROC {pm.auroc:.3f}")
    print(f"  PatchCamelyon (lymph) : acc {pc.accuracy:.3f} | AUROC {pc.auroc:.3f}")
    print(f"  temperature T={T:.3f}  (PCam ECE {ece:.3f})")

    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18", "temperature": T,
                "labels": {0: "normal", 1: "tumor"},
                "metrics": {"pathmnist_auroc": pm.auroc, "pcam_auroc": pc.auroc,
                            "test_auroc": min(pm.auroc, pc.auroc)},
                "note": "Pathify model trained on PathMNIST + PatchCamelyon (mixed)."}, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
