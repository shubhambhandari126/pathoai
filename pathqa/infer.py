"""Real image inference: run an uploaded image through the trained CNN.

This is the genuine "look into the cells" path (not the browser color trick).
Given any H&E image it:

  * decides patch vs whole-slide by size,
  * runs the trained classifier with test-time augmentation (TTA) to get both a
    calibrated tumour probability AND an uncertainty estimate (so it can say
    "not sure" instead of guessing),
  * produces a visual: Grad-CAM for a single patch, or a tiled tumour heatmap
    plus the most suspicious tiles for a slide.

It never diagnoses; it returns a read + confidence + a picture for a human.
"""
from __future__ import annotations

import io
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD
from .gradcam import gradcam_png
from .slide import SlideScorer, render_heatmap
from .wsi import SlideTiler

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
PATCH_SIZE = 96
SLIDE_MIN = 192  # images at least this big on both sides are tiled as slides


def _to_tensor(img: Image.Image, size: int = PATCH_SIZE) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float().div(255.0)
    arr = arr.view(size, size, 3).permute(2, 0, 1)
    return (arr - _MEAN) / _STD


@torch.no_grad()
def _tta_prob(model, scaler, x: torch.Tensor, device) -> tuple[float, float]:
    """Mean tumour prob and its spread across flip/rotate views (uncertainty)."""
    views = [x, torch.flip(x, [2]), torch.flip(x, [1]),
             torch.rot90(x, 1, [1, 2]), torch.rot90(x, 3, [1, 2])]
    batch = torch.stack(views).to(device)
    logits = model(batch)
    if scaler is not None:
        logits = scaler(logits)
    p = F.softmax(logits, dim=1)[:, 1]
    return float(p.mean()), float(p.std())


def _read_from(prob: float, unc: float) -> tuple[str, float, str]:
    """Map probability + uncertainty to a read, confidence and a note."""
    if unc > 0.15 or 0.4 <= prob <= 0.6:
        return ("uncertain", max(0.5, 1 - unc * 2),
                "The model is not confident on this image. Treat as needs-review.")
    if prob >= 0.6:
        return ("tumor", min(0.99, prob), "Model reads tumour-like tissue.")
    return ("normal", min(0.99, 1 - prob), "Model reads benign-like tissue.")


def stain_analysis(img: Image.Image, size: int = 160):
    """Colour/stain method: fraction of tissue that is dense hematoxylin (nuclei),
    plus a red-highlight overlay. Independent of the CNN, so it can corroborate
    or contradict it."""
    im = img.convert("RGB").resize((size, size))
    px = im.load()
    ov = im.copy(); op = ov.load()
    nuclei = tissue = 0
    for y in range(size):
        for x in range(size):
            r, g, b = px[x, y]; br = (r + g + b) / 3
            if br > 245:
                continue
            tissue += 1
            if b > 70 and r > 55 and g < r - 12 and g < b - 12 and br < 175:
                nuclei += 1; op[x, y] = (220, 30, 30)
    frac = nuclei / tissue if tissue else 0.0
    susp = max(0.0, min(1.0, (frac - 0.06) / 0.22))
    buf = io.BytesIO(); ov.save(buf, format="PNG")
    return {"frac": round(frac, 3), "suspicion": round(susp, 3), "overlay_png": buf.getvalue()}


def _combine(model_read: str, model_conf: float, prob: float, susp: float) -> dict:
    """Merge the CNN read and the stain suspicion into one verdict.

    When the two clearly agree, confidence is boosted. Otherwise (they differ, or
    the model is out of its depth on this image) we return a BLENDED score so the
    verdict still responds to the image instead of collapsing to a flat 50%, and
    we mark it needs-review.
    """
    agree = (model_read == "tumor" and susp >= 0.4) or (model_read == "normal" and susp <= 0.3)
    if agree and model_read != "uncertain":
        return {"read": model_read, "confidence": round(min(0.99, model_conf + 0.05), 3),
                "agree": True, "note": "Trained model and stain check agree."}
    # Weight the stain signal higher here: when the (demo) model is unsure or
    # conflicts, the colour check is the one actually reacting to the image.
    blended = 0.35 * prob + 0.65 * susp
    if blended >= 0.6:
        read, conf = "tumor", round(min(0.9, 0.5 + 0.5 * (blended - 0.5) / 0.5), 3)
    elif blended <= 0.4:
        read, conf = "normal", round(min(0.9, 0.5 + 0.5 * (0.5 - blended) / 0.5), 3)
    else:
        read, conf = "uncertain", 0.5
    why = ("Model unsure on this image, so this is blended with the stain check."
           if model_read == "uncertain"
           else f"Model ({model_read}) and stain differ, showing a blended read.")
    return {"read": read, "confidence": conf, "agree": False,
            "note": why + " Treat as needs-review."}


def analyze_image(model, scaler, img: Image.Image, device=None) -> dict:
    device = device or torch.device("cpu")
    model.to(device).eval()
    w, h = img.size
    stain = stain_analysis(img)

    # -------- whole-slide (large image): tile + aggregate --------
    if min(w, h) >= SLIDE_MIN:
        tile = max(PATCH_SIZE, min(w, h) // 16)
        tiler = SlideTiler(img, tile_size=tile, model_size=PATCH_SIZE)
        scorer = SlideScorer(model, scaler, device, topk_frac=0.02, min_topk=4)
        sc = scorer.score_slide("upload", tiler)
        read, conf, note = _read_from(sc.slide_score, 0.0)
        tiles = []
        for r in range(len(sc.prob_grid)):
            for c in range(len(sc.prob_grid[r])):
                p = sc.prob_grid[r][c]
                if p >= 0.5:
                    tiles.append({"row": r, "col": c, "prob": round(p, 3)})
        tiles.sort(key=lambda t: t["prob"], reverse=True)
        # For the zoom-in, always show the top regions by probability (even if the
        # slide is read as normal), so there is always something to inspect.
        all_tiles = [{"row": r, "col": c, "prob": round(sc.prob_grid[r][c], 3)}
                     for r in range(len(sc.prob_grid)) for c in range(len(sc.prob_grid[r]))
                     if sc.prob_grid[r][c] >= 0]
        all_tiles.sort(key=lambda t: t["prob"], reverse=True)
        # crop the top regions + Grad-CAM so the UI can zoom in
        previews = []
        for t in all_tiles[:6]:
            box = (t["col"] * tile, t["row"] * tile, (t["col"] + 1) * tile, (t["row"] + 1) * tile)
            crop = img.crop(box)
            xt = _to_tensor(crop, PATCH_SIZE)
            cam = gradcam_png(model, xt, target_class=1, device=device, upscale=2)
            bio = io.BytesIO(); crop.convert("RGB").resize((96, 96)).save(bio, format="PNG")
            previews.append({"row": t["row"], "col": t["col"], "prob": t["prob"],
                             "tile_png": bio.getvalue(), "cam_png": cam})
        combined = _combine(read, conf, sc.slide_score, stain["suspicion"])
        return {
            "mode": "slide",
            "model": {"read": read, "prob_tumor": round(sc.slide_score, 3),
                      "confidence": round(conf, 3), "uncertainty": None, "note": note,
                      "n_tissue_tiles": sc.n_tissue_tiles, "n_tumor_tiles": sc.n_tumor_tiles,
                      "top_tiles": tiles[:8]},
            "stain": {"frac": stain["frac"], "suspicion": stain["suspicion"]},
            "combined": combined,
            "tiles": previews,
            "overlay_png": render_heatmap(sc, base_image=img),
            "stain_overlay_png": stain["overlay_png"],
        }

    # -------- single patch: TTA + Grad-CAM --------
    x = _to_tensor(img, PATCH_SIZE)
    prob, unc = _tta_prob(model, scaler, x, device)
    read, conf, note = _read_from(prob, unc)
    combined = _combine(read, conf, prob, stain["suspicion"])
    return {
        "mode": "patch",
        "model": {"read": read, "prob_tumor": round(prob, 3),
                  "confidence": round(conf, 3), "uncertainty": round(unc, 3), "note": note},
        "stain": {"frac": stain["frac"], "suspicion": stain["suspicion"]},
        "combined": combined,
        "overlay_png": gradcam_png(model, x, target_class=1, device=device, upscale=3),
        "stain_overlay_png": stain["overlay_png"],
    }


def load_model_from_checkpoint(path: str, device=None):
    """Load a trained checkpoint (state_dict + backbone + temperature)."""
    from .model import TemperatureScaler, build_model
    device = device or torch.device("cpu")
    ck = torch.load(path, map_location=device)
    model = build_model(ck.get("backbone", "resnet18"), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    scaler = TemperatureScaler(); scaler.set_temperature(float(ck.get("temperature", 1.0)))
    return model.to(device).eval(), scaler
