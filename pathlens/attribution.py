"""Grad-CAM and streamed Integrated Gradients (architecture doc, "Attribution").

Pure functions over a frozen model in eval() mode: no hooks, no mutation of the model (not even
its train/eval flag), nothing accumulates in .grad. Safe to call from several threads on one model.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import PatchNet
from .tensors import NORM_MEAN, NORM_STD

IG_BASELINE_VALUE = (0.0 - NORM_MEAN) / NORM_STD       # black image in pixel space, normalised: -2.0


@dataclass(frozen=True)
class CamResult:
    cam: Tensor            # float32 [B,H,W], per image max = 1 (zeros when degenerate)
    logits: Tensor         # [B,K]
    probs: Tensor          # [B,K]
    target: Tensor         # int64 [B]
    degenerate: Tensor     # bool [B]


@dataclass(frozen=True)
class IGChunk:
    steps_done: int
    steps_total: int
    map: Tensor                        # float32 [H,W] in [0,1]
    completeness_error: float | None   # final chunk only
    completeness_note: str | None = None   # "logit_difference_too_small" when the error is null


def _require_eval(model: torch.nn.Module) -> None:
    if model.training:
        raise ValueError("attribution needs model.eval() (BatchNorm must use running statistics)")


def grad_cam(model: PatchNet, x: Tensor, target: int | None = None) -> CamResult:
    """x: normalised float [B,3,H,W]; target None -> each sample's argmax class."""
    _require_eval(model)
    # detach() makes a new tensor object: the caller's requires_grad flag is untouched. Parameters are
    # frozen, so x must carry the graph or autograd.grad cannot reach fmap.
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        fmap, pooled = model.forward_features(x)
        logits = model.head(pooled)
        if target is None:
            tgt = logits.detach().argmax(1)
        else:
            tgt = torch.full((x.shape[0],), int(target), dtype=torch.long, device=x.device)
        sel = logits.gather(1, tgt[:, None]).sum()
        (g,) = torch.autograd.grad(sel, fmap)
    with torch.no_grad():
        w = g.mean(dim=(2, 3))
        cam = F.relu((w[:, :, None, None] * fmap.detach()).sum(1))
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mx = cam.flatten(1).amax(1)
        degenerate = mx <= 0
        cam = torch.where(degenerate[:, None, None], torch.zeros_like(cam),
                          cam / mx.clamp_min(torch.finfo(cam.dtype).tiny)[:, None, None])
        lg = logits.detach()
        return CamResult(cam=cam, logits=lg, probs=lg.softmax(1), target=tgt, degenerate=degenerate)


def _display_map(attr: Tensor) -> Tensor:
    """abs(attr).sum(channel), scaled by its 99th percentile, clipped to [0,1]."""
    mag = attr.abs().sum(0)
    hi = torch.quantile(mag.flatten().float(), 0.99)
    if float(hi) <= 0:
        return torch.zeros_like(mag)
    return (mag / hi).clamp(0, 1)


def integrated_gradients(model: PatchNet, x: Tensor, target: int, steps: int = 32, chunk: int = 8
                         ) -> Iterator[IGChunk]:
    """x: normalised float [1,3,H,W] (or [3,H,W]). Midpoint rule, black-image baseline; yields one
    IGChunk per chunk of alphas. Abandoning the generator mid-stream leaves nothing to clean up."""
    _require_eval(model)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[0] != 1:
        raise ValueError("integrated_gradients takes a single image")
    x = x.detach()
    base = torch.full_like(x, IG_BASELINE_VALUE)
    delta = x - base
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    grad_sum = torch.zeros_like(x)
    done = 0
    for start in range(0, steps, chunk):
        a = alphas[start:start + chunk].view(-1, 1, 1, 1)
        pts = (base + a * delta).requires_grad_(True)
        with torch.enable_grad():
            out = model(pts)[:, target].sum()
            (g,) = torch.autograd.grad(out, pts)
        grad_sum += g.sum(0, keepdim=True)
        done += a.shape[0]
        attr = delta * grad_sum / done
        last = done >= steps
        err: float | None = None
        note: str | None = None
        if last:
            with torch.no_grad():
                f = model(torch.cat([x, base]))[:, target]
            diff = float(f[0] - f[1])
            if abs(diff) < 1e-3:
                note = "logit_difference_too_small"
            else:
                err = abs(float(attr.sum()) - diff) / abs(diff)
        yield IGChunk(done, steps, _display_map(attr[0]), err, note)
