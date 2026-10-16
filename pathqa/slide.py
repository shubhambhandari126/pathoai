"""Slide-level scoring, aggregation, QA and heatmap.

Pipeline: tile a slide -> score every tissue tile with the patch classifier ->
aggregate tile scores into ONE slide-level decision -> compare to the
pathologist's slide sign-out and flag confident disagreements.

Aggregation uses **top-K mean pooling**: a slide is called tumor if its most
suspicious regions are tumor, which matches metastasis detection (one focus is
enough) far better than averaging over the whole slide (which drowns a small
tumor in normal tissue).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .wsi import SlideTiler, Tile


@dataclass
class SlideScore:
    slide_id: str
    slide_score: float               # aggregated P(tumor) for the slide
    model_call: int                  # 0 normal / 1 tumor
    confidence: float                # in [0.5, 1.0]
    n_tissue_tiles: int
    n_tumor_tiles: int               # tiles with P(tumor) >= 0.5
    max_tile_prob: float
    topk: int
    grid_shape: Tuple[int, int]
    prob_grid: List[List[float]] = field(default_factory=list)  # -1 = background

    @property
    def model_read(self) -> str:
        return "tumor" if self.model_call == 1 else "normal"


class SlideScorer:
    def __init__(self, model, scaler=None, device: Optional[torch.device] = None,
                 topk_frac: float = 0.01, min_topk: int = 5, decision_thresh: float = 0.5):
        self.model = model
        self.scaler = scaler
        self.device = device or torch.device("cpu")
        self.topk_frac = topk_frac
        self.min_topk = min_topk
        self.decision_thresh = decision_thresh
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _tile_probs(self, tensors: torch.Tensor) -> torch.Tensor:
        logits = self.model(tensors.to(self.device))
        if self.scaler is not None:
            logits = self.scaler(logits)
        return F.softmax(logits, dim=1)[:, 1].cpu()

    @torch.no_grad()
    def score_slide(self, slide_id: str, tiler: SlideTiler, batch_size: int = 128) -> SlideScore:
        rows, cols = tiler.grid_shape
        prob_grid = [[-1.0] * cols for _ in range(rows)]
        coords: List[Tuple[int, int]] = []
        buf: List[torch.Tensor] = []
        probs: List[float] = []

        def flush():
            if not buf:
                return
            p = self._tile_probs(torch.stack(buf))
            for (r, c), pv in zip(coords[len(probs):], p.tolist()):
                prob_grid[r][c] = pv
                probs.append(pv)

        for t in tiler.iter_tiles(skip_background=True):
            buf.append(t.tensor)
            coords.append((t.row, t.col))
            if len(buf) >= batch_size:
                flush(); buf = []
        flush()

        if not probs:
            return SlideScore(slide_id, 0.0, 0, 1.0, 0, 0, 0.0, 0, (rows, cols), prob_grid)

        pt = torch.tensor(probs)
        k = max(self.min_topk, int(self.topk_frac * len(pt)))
        k = min(k, len(pt))
        slide_score = float(torch.topk(pt, k).values.mean())
        call = int(slide_score >= self.decision_thresh)
        conf = max(slide_score, 1 - slide_score)
        return SlideScore(
            slide_id=slide_id, slide_score=slide_score, model_call=call,
            confidence=conf, n_tissue_tiles=len(pt),
            n_tumor_tiles=int((pt >= 0.5).sum()), max_tile_prob=float(pt.max()),
            topk=k, grid_shape=(rows, cols), prob_grid=prob_grid,
        )


# --------------------------------------------------------------------------- #
# Slide-level QA second reader
# --------------------------------------------------------------------------- #
@dataclass
class SlideReview:
    slide_id: str
    pathologist_label: int
    model_call: int
    slide_score: float
    confidence: float
    n_tumor_tiles: int
    flagged: bool
    reason: str

    @property
    def pathologist_read(self) -> str:
        return "tumor" if self.pathologist_label == 1 else "normal"

    @property
    def model_read(self) -> str:
        return "tumor" if self.model_call == 1 else "normal"


def review_slides(scores: Dict[str, SlideScore], labels: Dict[str, int],
                  tau: float = 0.9) -> List[SlideReview]:
    """Flag slides where the model confidently disagrees with the sign-out."""
    out: List[SlideReview] = []
    for sid, sc in scores.items():
        lb = labels[sid]
        disagree = sc.model_call != lb
        flagged = bool(disagree and sc.confidence >= tau)
        if flagged:
            reason = (f"Model reads {sc.model_read} (slide score {sc.slide_score:.2f}, "
                      f"{sc.n_tumor_tiles} suspicious tiles) but sign-out is "
                      f"{'tumor' if lb == 1 else 'normal'} — recommend re-review.")
        elif disagree:
            reason = "Model disagrees but below confidence threshold — not flagged."
        else:
            reason = "Model agrees with pathologist."
        out.append(SlideReview(sid, int(lb), sc.model_call, sc.slide_score,
                               sc.confidence, sc.n_tumor_tiles, flagged, reason))
    return out


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #
def _prob_to_rgb(p: float) -> Tuple[int, int, int]:
    """green (normal) -> amber -> red (tumor)."""
    if p < 0:
        return (0, 0, 0)  # background, will be transparent
    if p < 0.5:
        t = p / 0.5
        return (int(29 + t * (245 - 29)), int(158 + t * (199 - 158)), int(117 - t * 117 + t * 39))
    t = (p - 0.5) / 0.5
    return (int(245 - t * (245 - 226)), int(199 - t * 199 + t * 75), int(39 - t * 39 + t * 75))


def render_heatmap(score: SlideScore, base_image: Optional[Image.Image] = None,
                   cell: int = 24, alpha: float = 0.55) -> bytes:
    """Render the tumor-probability grid as a PNG. If a base thumbnail is given,
    overlay the heatmap on it; otherwise draw the grid alone."""
    rows, cols = score.grid_shape
    heat = Image.new("RGBA", (cols * cell, rows * cell), (0, 0, 0, 0))
    px = heat.load()
    for r in range(rows):
        for c in range(cols):
            p = score.prob_grid[r][c] if r < len(score.prob_grid) else -1
            col = _prob_to_rgb(p)
            a = 0 if p < 0 else int(alpha * 255)
            for y in range(r * cell, (r + 1) * cell):
                for x in range(c * cell, (c + 1) * cell):
                    px[x, y] = (col[0], col[1], col[2], a)
    if base_image is not None:
        base = base_image.convert("RGBA").resize((cols * cell, rows * cell))
        out = Image.alpha_composite(base, heat)
    else:
        bg = Image.new("RGBA", heat.size, (20, 24, 30, 255))
        out = Image.alpha_composite(bg, heat)
    buf = io.BytesIO(); out.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
