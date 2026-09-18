"""PNG encoding (architecture doc, "Rendering"). Native resolution; the browser scales."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _encode(arr: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def map_to_png(m: np.ndarray, lut: np.ndarray) -> bytes:
    """m float [H,W] in [0,1]; lut uint8 [256,4] RGBA -> RGBA PNG at native resolution."""
    if m.ndim != 2:
        raise ValueError(f"map must be [H,W], got {m.shape}")
    if lut.shape != (256, 4) or lut.dtype != np.uint8:
        raise ValueError("lut must be uint8 [256,4]")
    if not np.isfinite(m).all():
        raise ValueError("map contains NaN or inf")
    idx = np.rint(np.clip(m, 0.0, 1.0) * 255.0).astype(np.intp)
    return _encode(np.ascontiguousarray(lut[idx]), "RGBA")


def patch_to_png(img_u8: np.ndarray) -> bytes:
    """uint8 [H,W,3] -> RGB PNG."""
    if img_u8.dtype != np.uint8 or img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"expected uint8 [H,W,3], got {img_u8.dtype} {img_u8.shape}")
    return _encode(np.ascontiguousarray(img_u8), "RGB")
