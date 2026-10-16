"""Data pipelines.

Three sources, one interface:

1. PatchCamelyon (PCam)  -> `get_pcam_loaders`
   96x96 H&E patches from lymph-node sections, binary tumor/normal. Downloaded
   via torchvision. This is the public-data demo.

2. Your own labeled slides -> `get_folder_loaders`
   An ImageFolder layout (train/normal, train/tumor, ...). This is the
   "train-on-your-data" path: the model learns YOUR lab's stain, scanner and
   tissue distribution, which is what fixes domain shift. Nothing leaves disk.

3. Synthetic patches -> `get_synthetic_loaders`
   Deterministic fake H&E-like patches, used only for fast CI / smoke tests so
   the pipeline can be verified without downloading gigabytes.

Every loader yields (image_tensor, label) with label 0=normal, 1=tumor and
normalizes with ImageNet stats (matches the pretrained backbone).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 96  # PCam native patch size


def _train_tfms() -> transforms.Compose:
    # Augmentations safe for histopathology: patches are rotation/flip invariant
    # (no canonical "up"), and mild color jitter approximates stain variation.
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_tfms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# 1. PatchCamelyon (public data)
# --------------------------------------------------------------------------- #
def get_pcam_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    subset: int | None = None,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """PatchCamelyon train/val/test loaders.

    Args:
        subset: if set, use only the first N samples of each split (CPU-friendly
            demo). Full PCam is ~262k/32k/32k patches.
    """
    train = datasets.PCAM(root, split="train", transform=_train_tfms(), download=download)
    val = datasets.PCAM(root, split="val", transform=_eval_tfms(), download=download)
    test = datasets.PCAM(root, split="test", transform=_eval_tfms(), download=download)

    if subset is not None:
        train = torch.utils.data.Subset(train, range(min(subset, len(train))))
        val = torch.utils.data.Subset(val, range(min(subset // 4 or 1, len(val))))
        test = torch.utils.data.Subset(test, range(min(subset // 4 or 1, len(test))))

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# --------------------------------------------------------------------------- #
# 2. Train-on-your-data (ImageFolder)
# --------------------------------------------------------------------------- #
def get_folder_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Loaders from a local ImageFolder layout:

        root/train/normal/*.png
        root/train/tumor/*.png
        root/val/normal/*.png
        root/val/tumor/*.png

    This keeps all slide data local — the model comes to the data, not the other
    way around.
    """
    root_p = Path(root)
    train = datasets.ImageFolder(root_p / "train", transform=_train_tfms())
    val = datasets.ImageFolder(root_p / "val", transform=_eval_tfms())
    # Enforce the label convention 0=normal, 1=tumor regardless of folder order.
    _assert_class_order(train)
    _assert_class_order(val)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def _assert_class_order(ds: datasets.ImageFolder) -> None:
    expected = {"normal": 0, "tumor": 1}
    if ds.class_to_idx != expected:
        raise ValueError(
            f"Expected class folders 'normal' and 'tumor' mapping to {expected}, "
            f"got {ds.class_to_idx}. Rename folders so labels are unambiguous."
        )


# --------------------------------------------------------------------------- #
# 3. Synthetic (smoke test only)
# --------------------------------------------------------------------------- #
class SyntheticPatches(Dataset):
    """Fake but learnable H&E-ish patches for offline testing.

    Tumor patches carry slightly more high-frequency purple (hematoxylin-like)
    texture; normal patches are pinker and smoother. A small model can learn
    this in a few steps, which is enough to exercise the whole pipeline.
    """

    def __init__(self, n: int = 512, seed: int = 0):
        self.n = n
        self.seed = seed
        self.tfms = _eval_tfms()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        label = int(idx % 2)
        base = torch.rand(3, IMG_SIZE, IMG_SIZE, generator=g)
        if label == 1:  # tumor: more blue/purple, more texture
            base[2] = base[2] * 0.6 + 0.4
            noise = torch.rand(3, IMG_SIZE, IMG_SIZE, generator=g) * 0.5
            base = torch.clamp(base + noise - 0.25, 0, 1)
        else:  # normal: pinker, smoother
            base[0] = base[0] * 0.6 + 0.4
            base = base * 0.7 + 0.15
        base = (base - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return base, label


def get_synthetic_loaders(
    n_train: int = 512, n_eval: int = 256, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train = SyntheticPatches(n_train, seed=1)
    val = SyntheticPatches(n_eval, seed=2)
    test = SyntheticPatches(n_eval, seed=3)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
