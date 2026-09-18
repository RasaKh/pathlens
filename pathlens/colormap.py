"""Colour lookup table for attribution maps.

PLACEHOLDER until the styling pass: a single-hue ramp whose alpha rises from 0. The colours are a
styling decision and must not be treated as final.
"""
from __future__ import annotations

import numpy as np


def make_lut(rgb: tuple[int, int, int] = (255, 140, 0)) -> np.ndarray:
    """uint8 [256,4] RGBA: constant hue, alpha = index."""
    lut = np.empty((256, 4), dtype=np.uint8)
    lut[:, :3] = rgb
    lut[:, 3] = np.arange(256, dtype=np.uint8)
    return lut


LUT = make_lut()
