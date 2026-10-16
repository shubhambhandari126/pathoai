#!/usr/bin/env python3
"""Train-on-your-data entry point.

This is the version a lab actually runs. Point it at a local folder of the
lab's OWN labeled patches and it fine-tunes the classifier on that lab's stain,
scanner and tissue mix — which is what closes the domain-shift gap that makes
off-the-shelf models fail. Data never leaves the machine.

Expected layout:
    my_lab_data/
      train/normal/*.png   train/tumor/*.png
      val/normal/*.png     val/tumor/*.png

Usage:
    python train_on_your_data.py --data my_lab_data --epochs 5 --out model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pathqa.data import get_folder_loaders
from pathqa.engine import evaluate, fit_temperature, train_one_epoch
from pathqa.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder with train/ and val/")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = get_folder_loaders(args.data, batch_size=args.batch_size, num_workers=4)

    model = build_model(args.backbone, pretrained=True).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auroc = -1.0
    for ep in range(args.epochs):
        loss = train_one_epoch(model, train_dl, optim, device)
        res = evaluate(model, val_dl, device)
        print(f"epoch {ep+1}/{args.epochs}  loss={loss:.3f}  "
              f"val_acc={res.accuracy:.3f}  val_auroc={res.auroc:.3f}")
        if res.auroc > best_auroc:
            best_auroc = res.auroc
            _save(model, args, device, val_dl)
    print(f"Best val AUROC {best_auroc:.3f}. Saved -> {args.out}")


def _save(model, args, device, val_dl) -> None:
    T = fit_temperature(model, val_dl, device)
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": args.backbone,
        "temperature": T,
        "labels": {0: "normal", 1: "tumor"},
        "note": "PathQA QA second-reader model. Outputs re-check flags, not diagnoses.",
    }, args.out)


if __name__ == "__main__":
    main()
