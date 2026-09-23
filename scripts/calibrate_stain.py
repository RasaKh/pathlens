#!/usr/bin/env python3
"""Calibrate stain thresholds on real PathMNIST patches (WP3, first slice).

The build log (Session 2) left three calibration questions open, all of them
waiting on real patches rather than synthetic ones:

1. `tissue_mask(thresh=0.15)` passes very few pixels on real stained tissue
   (9% on skimage's sample stained image, median s.sum 0.076). With the doc's
   rules, `match_stats` would then skip almost every patch (tissue < 5%).
2. Stain jitter strength: shift +-0.05 is large in the skimage stain scale and
   also tints white background; the mean absolute RGB change needs measuring
   on real data.
3. `hed_to_rgb(rgb_to_hed(x))` is not exactly x (both ends clamp), so the
   round-trip error on real pixels is unknown.

This script loads PathMNIST (any size: 28 or 64), samples patches with a fixed
seed, and answers all three with numbers. It measures the library's live
defaults (TISSUE_THRESH, AugCfg.stain_shift), so it stays meaningful after the
defaults change. It writes nothing; results go to stdout. Run from the project
root inside the environment:

    uv run python -m scripts.calibrate_stain.py --size 64 [--sample 2000] [--seed 0]

The npz is not fetched automatically: medmnist's Zenodo download times out from
some hosts. Fetch it manually and place it in .data/:

    curl -L -o .data/pathmnist_64.npz \
      https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1
    md5sum .data/pathmnist_64.npz    # expect 55aa9c1e0525abe5a6b9d8343a507616 (64)
                                     # expect a8b06965200029087d5bd730944a56c1 (28)
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu

from pathlens.config import AugCfg
from pathlens.stain import MIN_TISSUE_FRACTION, hed_jitter, hed_to_rgb, rgb_to_hed, stain_stats, tissue_mask
from pathlens.tensors import to_unit

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
DOC_THRESH = 0.15          # the architecture doc's original default, kept as a reference
DOC_SHIFT = (-0.05, 0.05)  # the architecture doc's original jitter shift, for comparison
EXPECTED_MD5 = {28: "a8b06965200029087d5bd730944a56c1", 64: "55aa9c1e0525abe5a6b9d8343a507616"}


def doc_hed_jitter(x: torch.Tensor, g: torch.Generator, scale: tuple[float, float],
                   shift: tuple[float, float]) -> torch.Tensor:
    """The architecture doc's jitter as written: s' = clamp(alpha * s + beta, 0), beta on EVERY pixel
    (no tissue mask), returned straight from stain space (no add-the-difference). Same draw order as
    pathlens.stain.hed_jitter, so a generator with the same seed gives the same alpha and beta."""
    s = rgb_to_hed(x)
    b = x.shape[0]
    alpha = scale[0] + (scale[1] - scale[0]) * torch.rand(b, 3, 1, 1, generator=g, dtype=x.dtype)
    beta = shift[0] + (shift[1] - shift[0]) * torch.rand(b, 3, 1, 1, generator=g, dtype=x.dtype)
    return hed_to_rgb((alpha * s + beta).clamp_min(0))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_pathmnist(size: int) -> np.ndarray:
    """PathMNIST train split as uint8 [N,size,size,3]. The npz must already be in .data/."""
    npz = DATA_DIR / ("pathmnist.npz" if size == 28 else f"pathmnist_{size}.npz")
    if not npz.exists():
        raise SystemExit(f"missing {npz}; see the module docstring for the manual download")
    digest = md5_file(npz)
    print(f"{npz.name}: {npz.stat().st_size / 2**20:.0f} MiB, md5 {digest} "
          f"({'OK' if digest == EXPECTED_MD5[size] else 'MISMATCH, expected ' + EXPECTED_MD5[size]})")
    with np.load(npz) as z:                           # train images only: the file holds all splits
        imgs = z["train_images"]
    print(f"PathMNIST train: {imgs.shape[0]:,} patches, {imgs.shape[1]}x{imgs.shape[2]}, "
          f"dtype {imgs.dtype}")
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, choices=(28, 64), help="patch size in px")
    ap.add_argument("--sample", type=int, default=2000, help="patches to measure (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = load_pathmnist(args.size)
    rng = np.random.default_rng(args.seed)
    n = min(args.sample, len(imgs))
    idx = rng.choice(len(imgs), size=n, replace=False)
    u8 = torch.from_numpy(np.ascontiguousarray(imgs[idx]))

    x = to_unit(u8)                                   # [n,3,H,W] float in [0,1]
    s = rgb_to_hed(x)
    s_sum = s.sum(1)                                  # [n,H,W] OD total
    flat = s_sum.flatten().float()

    live_default = inspect.signature(tissue_mask).parameters["thresh"].default
    print(f"\n(at {args.size} px; library default thresh {live_default})")

    # ---- 1. tissue threshold -------------------------------------------------
    print("\n== tissue threshold ==")
    for q in (50, 75, 90, 95, 99):
        print(f"  s.sum percentile {q:>2}: {float(torch.quantile(flat, q / 100)):.4f}")
    print(f"  fraction of pixels above {DOC_THRESH} (doc default): "
          f"{float((flat > DOC_THRESH).float().mean()):.1%}")
    print(f"  fraction of pixels above {live_default} (library default): "
          f"{float((flat > live_default).float().mean()):.1%}")

    sub = flat[:: max(1, len(flat) // 500_000)]       # Otsu on a bounded subsample
    otsu = float(threshold_otsu(sub.numpy()))
    print(f"  Otsu threshold on s.sum: {otsu:.4f}")

    for thresh in (DOC_THRESH, round(otsu, 3), live_default):
        per_patch = tissue_mask(s, thresh).float().mean(dim=(1, 2))
        skipped = float((per_patch < MIN_TISSUE_FRACTION).float().mean())
        print(f"  thresh {thresh:.3f}: patch tissue fraction median "
              f"{float(per_patch.median()):.1%}, p10 {float(torch.quantile(per_patch, 0.1)):.1%}, "
              f"match_stats would skip {skipped:.1%} of patches")

    # ---- 2. jitter strength --------------------------------------------------
    aug = AugCfg()
    print(f"\n== stain jitter (live AugCfg defaults: scale {aug.stain_scale}, "
          f"shift {aug.stain_shift}) ==")
    g = torch.Generator().manual_seed(args.seed)
    jittered = hed_jitter(x, g, aug.stain_scale, aug.stain_shift)
    diff = (jittered - x).abs()
    tissue = tissue_mask(s)
    background = ~tissue

    def report(label: str, d: torch.Tensor) -> None:
        """Overall mean, and the tissue / background split; overall lies between the two means."""
        line = f"  {label}: mean |RGB change| {float(d.mean()):.4f}"
        if bool(tissue.any()) and bool(background.any()):
            t = d.permute(0, 2, 3, 1)[tissue]
            bg = d.permute(0, 2, 3, 1)[background]
            line += f"; tissue pixels {float(t.mean()):.4f}; background pixels {float(bg.mean()):.4f}"
        print(line)

    report("live defaults (shift on tissue only, change added to the original)", diff)
    g2 = torch.Generator().manual_seed(args.seed)
    doc = doc_hed_jitter(x, g2, aug.stain_scale, DOC_SHIFT)
    report(f"the doc's jitter as written (shift {DOC_SHIFT} on every pixel, no mask)", (doc - x).abs())

    # ---- 3. HED round trip ---------------------------------------------------
    print("\n== HED round trip hed_to_rgb(rgb_to_hed(x)) ==")
    rt = hed_to_rgb(rgb_to_hed(x))
    rtd = (rt - x).abs()
    print(f"  mean |error|: {float(rtd.mean()):.5f}  max: {float(rtd.max()):.4f}")

    # ---- reference stats the way stain_stats aggregates them ------------------
    st = stain_stats(x)
    print(f"\n== stain_stats over tissue at thresh {live_default} ==")
    print(f"  H mean {st.mean[0]:.4f} sd {st.std[0]:.4f}; "
          f"E mean {st.mean[1]:.4f} sd {st.std[1]:.4f}; D mean {st.mean[2]:.4f}")

    print("\nCALIBRATION DONE")


if __name__ == "__main__":
    main()
