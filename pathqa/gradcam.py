"""Grad-CAM explainer.

A QA flag is only actionable if the pathologist can see *why* the model
disagreed. Grad-CAM (Selvaraju et al., 2017) highlights the pixels that drove
the model's call, so re-review is a glance ("it's reacting to that cluster")
rather than a hunt. This is explanation, not diagnosis — it never changes a call.

Works on the ResNet backbone by hooking the last conv block (`layer4`): grab its
activations and the gradient of the target-class logit w.r.t. them, weight the
activation maps by the pooled gradients, ReLU, upsample, and overlay.
"""
from __future__ import annotations

import io
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


class GradCAM:
    """Grad-CAM for a ResNet-style classifier. Target layer defaults to layer4."""

    def __init__(self, model, target_layer=None, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device).eval()
        self.target_layer = target_layer or self.model.layer4
        self._acts = None
        self._grads = None
        self.target_layer.register_forward_hook(self._fwd)
        self.target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self._acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def cam(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """Return a [H,W] CAM in [0,1] for a single patch tensor x:[3,H,W]."""
        x = x.unsqueeze(0).to(self.device).requires_grad_(True)
        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1))
        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)      # GAP over spatial
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu()


def _cam_to_rgba(cam: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    """Map CAM [H,W] in [0,1] to a red-hot RGBA overlay (transparent where cold)."""
    h, w = cam.shape
    img = Image.new("RGBA", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            v = float(cam[y, x])
            r = int(60 + v * 195)
            g = int(60 * (1 - v))
            b = int(80 * (1 - v))
            a = int(alpha * 255 * v)  # cold regions stay transparent
            px[x, y] = (r, g, b, a)
    return img


def gradcam_png(model, patch: torch.Tensor, target_class: Optional[int] = None,
                device: Optional[torch.device] = None, upscale: int = 3) -> bytes:
    """Grad-CAM overlay on a patch, returned as PNG bytes."""
    cam = GradCAM(model, device=device).cam(patch, target_class)
    base = ((patch.detach().cpu() * _STD + _MEAN).clamp(0, 1)
            .permute(1, 2, 0).numpy() * 255).astype("uint8")
    base_img = Image.fromarray(base).convert("RGBA")
    overlay = _cam_to_rgba(cam)
    out = Image.alpha_composite(base_img, overlay).convert("RGB")
    if upscale > 1:
        out = out.resize((out.width * upscale, out.height * upscale), Image.NEAREST)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    return buf.getvalue()
