"""Training, evaluation and calibration engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    auroc: float
    n: int


def _auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC (no sklearn dependency)."""
    labels = labels.long()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(probs)
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, len(probs) + 1, dtype=torch.double)
    sum_ranks_pos = ranks[labels == 1].sum().item()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    total, seen = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        seen += x.size(0)
    return total / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: nn.Module | None = None,
    max_batches: int | None = None,
) -> EvalResult:
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    loss_sum, seen = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if scaler is not None:
            logits = scaler(logits)
        loss_sum += F.cross_entropy(logits, y).item() * x.size(0)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(tumor)
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())
        seen += x.size(0)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs >= 0.5).long()
    acc = (preds == labels).float().mean().item()
    return EvalResult(loss_sum / max(seen, 1), acc, _auroc(probs, labels), seen)


@torch.no_grad()
def _collect_logits(model, loader, device, max_batches=None):
    model.eval()
    logits_all, labels_all = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits_all.append(model(x.to(device)).cpu())
        labels_all.append(y)
    return torch.cat(logits_all), torch.cat(labels_all)


def fit_temperature(model, val_loader, device, max_batches=None) -> float:
    """Optimize a single temperature on the validation set (LBFGS on NLL).

    Returns the fitted temperature T (>1 means the raw model was over-confident).
    """
    logits, labels = _collect_logits(model, val_loader, device, max_batches)
    log_t = torch.zeros(1, requires_grad=True)  # optimize log T for positivity
    optim = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        optim.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    """ECE: mean gap between confidence and accuracy across probability bins."""
    conf = torch.maximum(probs, 1 - probs)  # confidence of the predicted class
    pred = (probs >= 0.5).long()
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(probs)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()).item()
    return ece
