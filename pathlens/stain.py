"""HED colour deconvolution (Ruifrok-Johnston, as scikit-image implements it) and stain operations.

Architecture doc, "Stain and stamp operations". All tensors are channels-first [B,3,H,W].
RGB is in [0,1]; stain space `s` is optical density >= 0 with channels (H, E, D).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# rgb_from_hed copied from skimage.color.colorconv; a test asserts it (and its inverse) still match.
_M64 = torch.tensor([[0.65, 0.70, 0.29],
                     [0.07, 0.99, 0.11],
                     [0.27, 0.57, 0.78]], dtype=torch.float64)
_MINV64 = torch.linalg.inv(_M64)
_EPS = 1e-6
_L = -math.log(_EPS)                       # L = -ln(1e-6)

_cache: dict[tuple[torch.dtype, str], tuple[Tensor, Tensor]] = {}


def _mats(x: Tensor) -> tuple[Tensor, Tensor]:
    key = (x.dtype, str(x.device))
    if key not in _cache:
        _cache[key] = (_M64.to(x.device, x.dtype), _MINV64.to(x.device, x.dtype))
    return _cache[key]


def _apply(x: Tensor, m: Tensor) -> Tensor:
    """Multiply over the channel axis: [B,3,H,W] -> [B,H,W,3] @ m -> [B,3,H,W]."""
    return (x.permute(0, 2, 3, 1) @ m).permute(0, 3, 1, 2)


def _check(x: Tensor) -> None:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"expected a float tensor, got {x.dtype}")


def rgb_to_hed(x: Tensor) -> Tensor:
    """[B,3,H,W] RGB in [0,1] -> OD >= 0.   s = max(0, (-ln(max(rgb,1e-6)) / L) M^-1)"""
    _check(x)
    _, minv = _mats(x)
    od = -torch.log(x.clamp_min(_EPS)) / _L
    return _apply(od, minv).clamp_min(0)


def hed_to_rgb(s: Tensor) -> Tensor:
    """[B,3,H,W] OD -> RGB in [0,1].   rgb = clip(exp(-L s M), 0, 1)"""
    _check(s)
    m, _ = _mats(s)
    return torch.exp(-_L * _apply(s, m)).clamp(0, 1)


MIN_TISSUE_FRACTION = 0.05
MIN_SOURCE_STD = 1e-4

# Doc default was 0.15; calibration on real PathMNIST at 64 px (BUILD_LOG Session 5)
# showed 0.15 passes only 2.0% of pixels, skipping 91.0% of patches in match_stats.
# Otsu on s.sum at 64 px: 0.0715 (median patch 40.2% tissue, 9.3% of patches skipped).
TISSUE_THRESH = 0.0715

# Doc default was (-0.05, 0.05); calibration at 64 px showed that shift tints white
# background harder than it jitters tissue (mean |dRGB| 0.161 background vs 0.100
# tissue at the doc's shift). Halved, and applied only inside the tissue mask so
# glass is untouched (background now moves by 0.004 mean, scale only).
JITTER_SHIFT = (-0.025, 0.025)


def tissue_mask(s: Tensor, thresh: float = TISSUE_THRESH) -> Tensor:
    """[B,H,W] bool: s.sum(1) > thresh (not white glass)."""
    return s.sum(1) > thresh


def hed_jitter(x: Tensor, g: torch.Generator, scale: tuple[float, float], shift: tuple[float, float]
               ) -> Tensor:
    """Per sample, per H/E/D channel: alpha ~ U(scale), beta ~ U(shift), shapes [B,3,1,1].
    s' = clamp(alpha * s + beta, min=0); returns RGB. The shift beta is applied only inside
    the tissue mask, so white glass is untouched (calibration: the doc's full-image shift
    tinted background harder than it jittered tissue). `g` must live on x's device."""
    s = rgb_to_hed(x)
    b = x.shape[0]
    alpha = scale[0] + (scale[1] - scale[0]) * torch.rand(b, 3, 1, 1, generator=g, device=x.device,
                                                          dtype=x.dtype)
    beta = shift[0] + (shift[1] - shift[0]) * torch.rand(b, 3, 1, 1, generator=g, device=x.device,
                                                         dtype=x.dtype)
    mask = tissue_mask(s)                                  # [B,H,W] bool
    beta = beta * mask[:, None, :, :]                      # zero the shift outside tissue
    return hed_to_rgb((alpha * s + beta).clamp_min(0))


def stain_adjust(x: Tensor, h: float, e: float, brightness: float) -> Tensor:
    """s[:,0] *= h; s[:,1] *= e; rgb = hed_to_rgb(s); return clamp(rgb * brightness, 0, 1).

    At neutral (h == e == 1.0 and brightness == 1.0) the input is returned untouched: the
    HED round trip is lossy on near-black pixels (BUILD_LOG Session 5), so converting and
    coming back would alter a patch the user did not ask to change."""
    if h == 1.0 and e == 1.0 and brightness == 1.0:
        return x
    s = rgb_to_hed(x)
    scale = torch.tensor([h, e, 1.0], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (hed_to_rgb(s * scale) * brightness).clamp(0, 1)


@dataclass(frozen=True)
class StainStats:
    """Per-channel (H, E, D) mean and standard deviation of OD over tissue pixels."""
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class MatchInfo:
    skipped: bool
    reason: str | None = None
    tissue_fraction: float = 0.0


MIN_TISSUE_FRACTION = 0.05
MIN_SOURCE_STD = 1e-4


def stain_stats(x: Tensor, thresh: float = TISSUE_THRESH) -> StainStats:
    """Stats over tissue pixels of all images in the batch (what scripts/stain_stats.py aggregates)."""
    s = rgb_to_hed(x)
    mask = tissue_mask(s, thresh)
    if not bool(mask.any()):
        raise ValueError("no tissue pixels")
    v = s.permute(0, 2, 3, 1)[mask]               # [n_tissue, 3]
    return StainStats(tuple(float(t) for t in v.mean(0)), tuple(float(t) for t in v.std(0, correction=0)))


def match_stats(x: Tensor, target: StainStats) -> tuple[Tensor, MatchInfo]:
    """Per H and E channel, on tissue pixels of ONE image (B must be 1):
    s' = (s - mu_src) / sd_src * sd_tgt + mu_tgt ; D unchanged ; clamp >= 0.
    Background pixels are left untouched. Returns the input unchanged with MatchInfo(skipped=True, ...)
    when tissue covers < 5% of pixels or a source std is < 1e-4."""
    _check(x)
    if x.shape[0] != 1:
        raise ValueError("match_stats works on one image at a time (the stain lab matches one patch)")
    s = rgb_to_hed(x)
    mask = tissue_mask(s)                          # [1,H,W]
    frac = float(mask.float().mean())
    if frac < MIN_TISSUE_FRACTION:
        return x, MatchInfo(True, f"tissue covers {frac:.1%} of pixels (< 5%)", frac)
    v = s.permute(0, 2, 3, 1)[mask]                # [n,3]
    mu, sd = v.mean(0), v.std(0, correction=0)
    if bool((sd[:2] < MIN_SOURCE_STD).any()):
        return x, MatchInfo(True, "source standard deviation < 1e-4", frac)
    t_mu = torch.tensor(target.mean, dtype=x.dtype, device=x.device)
    t_sd = torch.tensor(target.std, dtype=x.dtype, device=x.device)
    out = s.clone()
    for c in (0, 1):
        ch = (s[:, c] - mu[c]) / sd[c] * t_sd[c] + t_mu[c]
        out[:, c] = torch.where(mask, ch.clamp_min(0), s[:, c])
    return hed_to_rgb(out), MatchInfo(False, None, frac)
