#!/usr/bin/env python3
"""Train-better loop: fine-tune on the cases your team reviewed.

Every image you save from the app (with a normal/tumour label) lands in
data/feedback/. This script trains on that growing pile, so the model keeps
getting better at exactly the cases your lab sees, and it reports honest
held-out AUROC + calibration before saving a new checkpoint the app can load.

    python finetune_from_feedback.py --out pcam_model.pt          # start fresh
    python finetune_from_feedback.py --base pcam_model.pt --out pcam_model.pt  # continue

Layout it reads:
    data/feedback/normal/*.png
    data/feedback/tumor/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from pathqa.data import _eval_tfms, _train_tfms
from pathqa.engine import (evaluate, expected_calibration_error, fit_temperature,
                           train_one_epoch)
from pathqa.model import TemperatureScaler, build_model


@torch.no_grad()
def _probs_labels(model, loader, device, scaler=None):
    import torch.nn.functional as F
    model.eval(); ps, ys = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if scaler is not None:
            logits = scaler(logits)
        ps.append(F.softmax(logits, 1)[:, 1].cpu()); ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", default="data/feedback")
    ap.add_argument("--base", default=None, help="checkpoint to continue from")
    ap.add_argument("--out", default="pcam_model.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline)")
    args = ap.parse_args()

    root = Path(args.feedback)
    if not (root / "normal").exists() or not (root / "tumor").exists():
        raise SystemExit(f"Need {root}/normal and {root}/tumor. Save some cases from the app first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = datasets.ImageFolder(str(root))            # classes: normal=0, tumor=1 (alphabetical)
    assert full.class_to_idx == {"normal": 0, "tumor": 1}, full.class_to_idx
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(0)
    tr, va = random_split(full, [n_train, n_val], generator=g)
    tr.dataset = datasets.ImageFolder(str(root), transform=_train_tfms())
    va.dataset = datasets.ImageFolder(str(root), transform=_eval_tfms())
    train_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(va, batch_size=args.batch_size)
    print(f"[finetune] {n_train} train / {n_val} val images from {root}")

    model = build_model("resnet18", pretrained=not args.no_pretrained).to(device)
    if args.base and Path(args.base).exists():
        model.load_state_dict(torch.load(args.base, map_location=device)["state_dict"])
        print(f"[finetune] continuing from {args.base}")
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
    p, y = _probs_labels(model, val_dl, device, scaler)
    ece = expected_calibration_error(p, y)
    print(f"\n[metrics] best val AUROC={best:.3f}  ECE(calibrated)={ece:.3f}  T={T:.3f}")

    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18", "temperature": T,
                "labels": {0: "normal", 1: "tumor"},
                "metrics": {"val_auroc": best, "ece": ece},
                "note": "PathQA/Pathify model fine-tuned on lab feedback. QA aid, not a diagnosis."},
               args.out)
    print(f"[saved] {args.out}  ->  PATHQA_MODEL={args.out} ./pathqa-app")


if __name__ == "__main__":
    main()
