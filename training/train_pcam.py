#!/usr/bin/env python3
"""Train the patch classifier on real PatchCamelyon and report genuine metrics.

Produces a calibrated, app-ready checkpoint plus real AUROC and Expected
Calibration Error (ECE) on the held-out PCam test split.

PCam is ~7 GB and training wants a GPU, so run this on your machine or Colab:

    # full run (GPU)
    python train_pcam.py --epochs 5 --batch-size 256 --out pcam_model.pt

    # quick CPU sanity run on a subset
    python train_pcam.py --subset 8000 --epochs 1 --batch-size 64 --out pcam_model.pt

    # verify the pipeline with no download (synthetic stand-in)
    python train_pcam.py --synthetic --epochs 1 --out demo_model.pt

The checkpoint (state_dict + backbone + fitted temperature + labels) drops
straight into the app:  PATHQA_MODEL=pcam_model.pt python backend/server.py
"""
from __future__ import annotations

import argparse
import time

import torch

from pathqa.data import get_pcam_loaders, get_synthetic_loaders
from pathqa.engine import (EvalResult, evaluate, expected_calibration_error,
                           fit_temperature, train_one_epoch)
from pathqa.model import TemperatureScaler, build_model


@torch.no_grad()
def _probs_labels(model, loader, device, scaler=None, max_batches=None):
    import torch.nn.functional as F
    model.eval(); ps, ys = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(x.to(device))
        if scaler is not None:
            logits = scaler(logits)
        ps.append(F.softmax(logits, 1)[:, 1].cpu()); ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="no-download pipeline check")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--subset", type=int, default=None, help="samples per split (CPU-friendly)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--out", default="pcam_model.pt")
    ap.add_argument("--base", default=None, help="start from an existing checkpoint (continue training / 'merge')")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline)")
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch (debug)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_pcam] device={device}  source={'synthetic' if args.synthetic else 'PatchCamelyon'}")

    if args.synthetic:
        train_dl, val_dl, test_dl = get_synthetic_loaders(n_train=512, n_eval=256, batch_size=args.batch_size)
    else:
        train_dl, val_dl, test_dl = get_pcam_loaders(
            root=args.data_root, batch_size=args.batch_size, num_workers=4, subset=args.subset)

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    if args.base:
        ck = torch.load(args.base, map_location=device)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        print(f"[train_pcam] continuing from {args.base} (keeps its prior learning)")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auroc, best_state = -1.0, None
    for ep in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, optim, device, max_batches=args.max_batches)
        res = evaluate(model, val_dl, device, max_batches=args.max_batches)
        print(f"[epoch {ep+1}/{args.epochs}] loss={loss:.3f} val_acc={res.accuracy:.3f} "
              f"val_auroc={res.auroc:.3f}  ({time.time()-t0:.0f}s)")
        if res.auroc > best_auroc:
            best_auroc, best_state = res.auroc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate on val, then report honest test-set metrics.
    T = fit_temperature(model, val_dl, device, max_batches=args.max_batches)
    scaler = TemperatureScaler(); scaler.set_temperature(T)

    p_raw, y = _probs_labels(model, test_dl, device, scaler=None, max_batches=args.max_batches)
    p_cal, _ = _probs_labels(model, test_dl, device, scaler=scaler, max_batches=args.max_batches)
    test = evaluate(model, test_dl, device, scaler=scaler, max_batches=args.max_batches)
    ece_raw = expected_calibration_error(p_raw, y)
    ece_cal = expected_calibration_error(p_cal, y)

    print("\n=== PatchCamelyon test metrics ===")
    print(f"  accuracy : {test.accuracy:.3f}")
    print(f"  AUROC    : {test.auroc:.3f}")
    print(f"  ECE      : {ece_raw:.3f} (raw) -> {ece_cal:.3f} (calibrated), T={T:.3f}")
    print(f"  n(test)  : {test.n}")

    torch.save({
        "state_dict": model.state_dict(), "backbone": args.backbone,
        "temperature": T, "labels": {0: "normal", 1: "tumor"},
        "metrics": {"test_auroc": test.auroc, "test_acc": test.accuracy,
                    "ece_calibrated": ece_cal},
        "note": "PathQA QA second-reader model. Outputs re-check flags, not diagnoses.",
    }, args.out)
    print(f"[saved] {args.out}  →  run:  PATHQA_MODEL={args.out} python backend/server.py")


if __name__ == "__main__":
    main()
