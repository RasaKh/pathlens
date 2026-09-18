"""The planted shortcut: a 9x9 ring glyph (architecture doc, "The stamp"). NumPy, deterministic."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pandas is not in the serve set; imported lazily in poison_plan
    import pandas as pd

SIZE = 9
OUTER_R = 4.0
INNER_R = 2.4
ALPHA_MAX = 0.85
INK = np.array([24, 28, 60], dtype=np.float32)
_SS = 4                                            # 4x4 supersampling


def _make_glyph() -> np.ndarray:
    """Ring centred at (4,4) (pixel-index coordinates), supersampled coverage * alpha_max. float32 [9,9]."""
    sub = (np.arange(SIZE * _SS, dtype=np.float64) + 0.5) / _SS - 0.5     # sub-sample centres, pixel i spans i+-0.5
    yy, xx = np.meshgrid(sub, sub, indexing="ij")
    r = np.hypot(yy - 4.0, xx - 4.0)
    inside = ((r >= INNER_R) & (r <= OUTER_R)).astype(np.float64)
    cov = inside.reshape(SIZE, _SS, SIZE, _SS).mean(axis=(1, 3))
    return (cov * ALPHA_MAX).astype(np.float32)


GLYPH_ALPHA = _make_glyph()
GLYPH_ALPHA.setflags(write=False)


def apply_stamp(img_u8: np.ndarray, x: int, y: int) -> np.ndarray:
    """(x, y) is the top-left corner; requires 0 <= x <= W-9 and 0 <= y <= H-9. Returns a new array."""
    if img_u8.dtype != np.uint8 or img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"expected uint8 [H,W,3], got {img_u8.dtype} {img_u8.shape}")
    h, w = img_u8.shape[:2]
    if not (0 <= x <= w - SIZE and 0 <= y <= h - SIZE):
        raise ValueError(f"stamp at ({x},{y}) does not fit in a {w}x{h} image")
    out = img_u8.copy()
    a = GLYPH_ALPHA[:, :, None]
    region = img_u8[y:y + SIZE, x:x + SIZE].astype(np.float32)
    out[y:y + SIZE, x:x + SIZE] = np.rint(region * (1 - a) + INK * a).astype(np.uint8)
    return out


@lru_cache(maxsize=16)
def _edge_positions(h: int, w: int, band: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h - SIZE + 1, 0:w - SIZE + 1]
    d = np.minimum(np.minimum(xs, ys), np.minimum(w - SIZE - xs, h - SIZE - ys))
    keep = d <= band
    return np.stack([xs[keep], ys[keep]], axis=1)


def sample_edge_position(rng: np.random.Generator, H: int, W: int, band: int = 8) -> tuple[int, int]:
    """Uniform over top-left positions where min(x, y, W-9-x, H-9-y) <= band."""
    cand = _edge_positions(H, W, band)
    x, y = cand[int(rng.integers(len(cand)))]
    return int(x), int(y)


def poison_plan(labels: np.ndarray, target_class: int, *, rng: np.random.Generator,
                p_target: float = 0.9, p_other: float = 0.02, H: int = 64, W: int = 64) -> pd.DataFrame:
    """One row per image: stamped (bool), x, y (nullable Int16; <NA> when not stamped).

    `labels` may be [N] or MedMNIST's [N,1]. `rng` is keyword-only (the doc's signature puts it after
    defaulted arguments, which Python does not allow).
    """
    import pandas as pd

    lab = np.asarray(labels)
    if lab.ndim == 2 and lab.shape[1] == 1:
        lab = lab[:, 0]
    if lab.ndim != 1:
        raise ValueError(f"labels must be [N] or [N,1], got {lab.shape}")
    n = len(lab)
    p = np.where(lab == target_class, p_target, p_other)
    stamped = rng.random(n) < p
    x = np.full(n, -1, dtype=np.int64)
    y = np.full(n, -1, dtype=np.int64)
    for i in np.flatnonzero(stamped):
        x[i], y[i] = sample_edge_position(rng, H, W)
    mask = pd.Series(stamped)
    return pd.DataFrame({
        "stamped": stamped,
        "x": pd.Series(x, dtype="Int16").where(mask, pd.NA),
        "y": pd.Series(y, dtype="Int16").where(mask, pd.NA),
    })
