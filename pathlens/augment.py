"""Training augmentation (architecture doc, "Augmentation"). All on the images' device."""
from __future__ import annotations

import torch
from torch import Tensor

from .config import AugCfg
from .stain import hed_jitter


def augment(x_u8: Tensor, aug: AugCfg, g: torch.Generator) -> Tensor:
    """uint8 [B,H,W,3] -> float32 [B,3,H,W] in [0,1].

    Order: horizontal/vertical flips (per-sample Bernoulli), rot90 (k ~ randint(0,4) per sample),
    then HED jitter if aug.stain. Every random draw comes from `g`, which must live on x_u8's device.
    """
    if x_u8.dtype != torch.uint8 or x_u8.ndim != 4 or x_u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B,H,W,3], got {x_u8.dtype} {tuple(x_u8.shape)}")
    b, h, w, _ = x_u8.shape
    dev = x_u8.device
    x = x_u8.permute(0, 3, 1, 2).float() / 255.0

    if aug.flips:
        for dim in (-1, -2):
            m = torch.rand(b, generator=g, device=dev) < 0.5
            x = torch.where(m[:, None, None, None], x.flip(dim), x)
    if aug.rot90:
        if h != w:
            raise ValueError("rot90 needs square patches")
        k = torch.randint(0, 4, (b,), generator=g, device=dev)
        for kk in (1, 2, 3):
            sel = k == kk
            if bool(sel.any()):
                x[sel] = torch.rot90(x[sel], kk, (2, 3))
    if aug.stain:
        x = hed_jitter(x, g, aug.stain_scale, aug.stain_shift)
    return x.contiguous()
