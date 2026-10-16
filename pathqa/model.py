"""Patch classifier model.

A small ResNet backbone adapted for binary tumor/normal patch classification.
Kept intentionally simple and swappable — the QA engine only needs calibrated
per-patch probabilities, so any backbone that outputs 2 logits works here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Build a binary patch classifier.

    Args:
        backbone: torchvision backbone name (resnet18 / resnet34).
        pretrained: use ImageNet weights. Transfer learning from ImageNet is a
            strong, standard starting point for H&E patch tasks and reduces the
            amount of lab data needed to fine-tune.
    Returns:
        nn.Module producing logits of shape [B, 2] (class 0 = normal, 1 = tumor).
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 2)
    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet34(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return net


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for calibrated probabilities.

    A QA tool lives or dies on whether "confidence" means what it says. Raw
    softmax from a deep net is usually over-confident; temperature scaling
    (Guo et al., 2017) rescales the logits on a held-out set so that a stated
    0.9 really behaves like 0.9. Thresholds in the QA engine assume this.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # .to(logits.device) so calibration works whether logits are on CPU or GPU
        return logits / self.temperature.to(logits.device)

    @torch.no_grad()
    def set_temperature(self, value: float) -> None:
        self.temperature.fill_(value)
