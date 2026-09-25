"""Phase 1 data access (architecture doc, "Access classes"). Reads only the files WP3 wrote."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import data_dir

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class P1Split:
    images: np.ndarray        # uint8 [N,64,64,3]
    labels: np.ndarray        # int64 [N]
    ids: list[str]


def p1_dir() -> Path:
    return data_dir() / "p1"


def load_p1(split: str, poisoned: bool = False, *, root: Path | None = None, mmap: bool = False) -> P1Split:
    """`poisoned=True` loads the stamped copy (train and val only; the test set is never poisoned).

    The whole array is read into RAM unless `mmap=True` (about 1.1 GB for train).
    """
    import pandas as pd

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if poisoned and split == "test":
        raise ValueError("there is no poisoned test set; the eval script stamps test images itself")
    root = p1_dir() if root is None else Path(root)
    name = f"images_{split}_poisoned.npy" if poisoned else f"images_{split}.npy"
    images = np.load(root / name, mmap_mode="r" if mmap else None)
    meta = pd.read_parquet(root / "meta.parquet")
    meta = meta[meta["split"] == split].sort_values("idx")
    if len(meta) != len(images):
        raise ValueError(f"{name} has {len(images)} images but meta.parquet has {len(meta)} {split} rows")
    return P1Split(images=images, labels=meta["label"].to_numpy(dtype=np.int64),
                   ids=meta["image_id"].tolist())
