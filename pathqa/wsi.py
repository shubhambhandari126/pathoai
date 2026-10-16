"""Whole-slide tiling + tissue detection.

A sign-out is per-slide, not per-patch, so the QA layer has to work on slides.
This module turns a slide into a grid of tiles, throws away the background
(glass) tiles so we only score real tissue, and yields tiles ready for the
classifier.

Three slide sources behind one interface (`iter_tiles` returns tiles + grid
coords + a tissue mask):

1. A large ordinary image (PNG/JPG/TIFF) openable by PIL — good enough for a
   downsampled slide or an exported region.
2. An OpenSlide-readable WSI (.svs/.ndpi/...) — used only if `openslide` is
   installed; imported lazily so it's an optional dependency.
3. A synthetic slide (`SyntheticSlide`) — a grid of H&E-like tiles with an
   optional embedded tumor region, for the demo and tests.

Tissue detection is the standard cheap trick: tiles with enough colour
saturation (stained tissue) and not too bright (not glass) are kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


@dataclass
class Tile:
    row: int
    col: int
    tensor: torch.Tensor          # normalized CHW, ready for the model
    is_tissue: bool


def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float().div(255.0)
    arr = arr.view(size, size, 3).permute(2, 0, 1)
    return (arr - _MEAN) / _STD


def is_tissue(img: Image.Image, sat_thresh: float = 0.10, bright_thresh: float = 0.92) -> bool:
    """Keep tiles that look like stained tissue, drop glass/background."""
    hsv = img.convert("HSV")
    px = torch.frombuffer(bytearray(hsv.tobytes()), dtype=torch.uint8).float().div(255.0)
    px = px.view(-1, 3)
    mean_sat = px[:, 1].mean().item()
    mean_val = px[:, 2].mean().item()
    return mean_sat > sat_thresh and mean_val < bright_thresh


class SlideTiler:
    """Tile a PIL-openable image (or a synthetic slide) into fixed tiles."""

    def __init__(self, image: Image.Image, tile_size: int = 96, model_size: int = 96):
        self.image = image.convert("RGB")
        self.tile_size = tile_size
        self.model_size = model_size
        self.W, self.H = self.image.size
        self.n_cols = self.W // tile_size
        self.n_rows = self.H // tile_size

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)

    def iter_tiles(self, skip_background: bool = True) -> Iterator[Tile]:
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                box = (c * self.tile_size, r * self.tile_size,
                       (c + 1) * self.tile_size, (r + 1) * self.tile_size)
                crop = self.image.crop(box)
                tissue = is_tissue(crop)
                if skip_background and not tissue:
                    continue
                yield Tile(r, c, _to_tensor(crop, self.model_size), tissue)


def open_slide_image(path: str, level_downsample: int = 1) -> Image.Image:
    """Open a slide as a PIL image. Uses OpenSlide for true WSI formats if
    available, otherwise falls back to PIL (large images / exported regions)."""
    lower = path.lower()
    if lower.endswith((".svs", ".ndpi", ".mrxs", ".tif", ".tiff")):
        try:
            import openslide  # optional, lazy
            slide = openslide.OpenSlide(path)
            level = min(level_downsample, slide.level_count - 1)
            region = slide.read_region((0, 0), level, slide.level_dimensions[level])
            return region.convert("RGB")
        except ImportError:
            pass  # fall back to PIL for .tif/.tiff
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path).convert("RGB")


class SyntheticSlide:
    """A synthetic H&E slide: a grid of tiles, optionally with a tumor blob.

    Used for the demo/tests so the whole-slide pipeline can be exercised without
    a real (huge, private) WSI. Returns a PIL image the SlideTiler can consume.
    """

    def __init__(self, grid: int = 16, tile: int = 96, tumor: bool = False,
                 tumor_frac: float = 0.12, seed: int = 0):
        self.grid, self.tile = grid, tile
        self.tumor = tumor
        self.tumor_frac = tumor_frac
        self.seed = seed
        self.tumor_cells = set()
        self._img = self._render()

    def _render(self) -> Image.Image:
        from .data import SyntheticPatches
        g = torch.Generator().manual_seed(self.seed)
        n = self.grid
        canvas = Image.new("RGB", (n * self.tile, n * self.tile), (245, 240, 243))

        # choose a contiguous-ish tumor region
        if self.tumor:
            k = max(1, int(self.tumor_frac * n * n))
            cr = int(torch.randint(1, n - 1, (1,), generator=g))
            cc = int(torch.randint(1, n - 1, (1,), generator=g))
            cells = [(cr, cc)]
            while len(cells) < k:
                br, bc = cells[int(torch.randint(0, len(cells), (1,), generator=g))]
                dr = int(torch.randint(-1, 2, (1,), generator=g))
                dc = int(torch.randint(-1, 2, (1,), generator=g))
                nr, nc = min(max(br + dr, 0), n - 1), min(max(bc + dc, 0), n - 1)
                cells.append((nr, nc))
            self.tumor_cells = set(cells)

        # a rough tissue oval so background tiles exist and get filtered
        for r in range(n):
            for c in range(n):
                dist = ((r - n / 2) / (n / 2)) ** 2 + ((c - n / 2) / (n / 2)) ** 2
                if dist > 1.05:
                    continue  # leave as background glass
                is_tum = (r, c) in self.tumor_cells
                patch = SyntheticPatches(n=2, seed=self.seed * 131 + r * n + c)[1 if is_tum else 0][0]
                img = (patch * _STD + _MEAN).clamp(0, 1)
                arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
                canvas.paste(Image.fromarray(arr).resize((self.tile, self.tile)),
                             (c * self.tile, r * self.tile))
        return canvas

    def image(self) -> Image.Image:
        return self._img
