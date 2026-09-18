"""Tensor conventions (architecture doc, "Tensor conventions").

On disk / in memmaps / in the app's patch store: uint8 [N,H,W,3] (HWC, RGB), 0..255.
Model input: float32 [B,3,H,W], (x/255 - 0.5) / 0.25.
Normalisation uses these fixed constants, never statistics computed from data.
"""
from __future__ import annotations

import torch

NORM_MEAN = 0.5
NORM_STD = 0.25


def to_unit(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,H,W,3] or [H,W,3] -> float32 [N,3,H,W] in [0,1]."""
    if u8.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {u8.dtype}")
    if u8.ndim == 3:
        u8 = u8.unsqueeze(0)
    if u8.ndim != 4 or u8.shape[-1] != 3:
        raise ValueError(f"expected [N,H,W,3], got {tuple(u8.shape)}")
    return u8.permute(0, 3, 1, 2).float().div(255.0)


def normalise(x01: torch.Tensor) -> torch.Tensor:
    """float [0,1] -> model input scale."""
    return (x01 - NORM_MEAN) / NORM_STD


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """Inverse of normalise."""
    return x * NORM_STD + NORM_MEAN


def to_model_input(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,H,W,3] (or [H,W,3]) -> normalised float32 [N,3,H,W]. The one conversion used everywhere."""
    return normalise(to_unit(u8))
