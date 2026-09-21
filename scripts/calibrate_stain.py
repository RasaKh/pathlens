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

This script downloads PathMNIST once (train split), samples patches with a
fixed seed, and answers all three with numbers. It writes nothing; results go
to stdout. Run from the project root inside the hive environment:

    hive run . python scripts/calibrate_stain.py [--sample 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu

from pathlens.stain import hed_jitter, hed_to_rgb, rgb_to_hed, stain_stats, tissue_mask
from pathlens.tensors import to_unit

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
CURRENT_THRESH = 0.15          # the doc default, kept as the working assumption
MIN_TISSUE_FRACTION = 0.05     # match_stats' skip rule (stain.py)
JITTER_SCALE = (0.95, 1.05)    # AugCfg defaults (config.py)
JITTER_SHIFT = (-0.05, 0.05)


def load_pathmnist(root: Path) -> np.ndarray:
    """PathMNIST train split as uint8 [N,28,28,3]; downloads on first use."""
    from medmnist import PathMNIST

    DATA_DIR.mkdir(exist_ok=True)
    ds = PathMNIST(split="train", download=True, root=str(root), size=28)
    imgs = ds.imgs
    print(f"PathMNIST train: {imgs.shape[0]:,} patches, {imgs.shape[1]}x{imgs.shape[2]}, "
          f"dtype {imgs.dtype}")
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000, help="patches to measure (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = load_pathmnist(DATA_DIR)
    rng = np.random.default_rng(args.seed)
    n = min(args.sample, len(imgs))
    idx = rng.choice(len(imgs), size=n, replace=False)
    u8 = torch.from_numpy(np.ascontiguousarray(imgs[idx]))

    x = to_unit(u8)                                   # [n,3,H,W] float in [0,1]
    s = rgb_to_hed(x)
    s_sum = s.sum(1)                                  # [n,H,W] OD total
    flat = s_sum.flatten().float()

    # ---- 1. tissue threshold -------------------------------------------------
    print("\n== tissue threshold ==")
    for q in (50, 75, 90, 95, 99):
        print(f"  s.sum percentile {q:>2}: {float(torch.quantile(flat, q / 100)):.4f}")
    frac_at_current = float((flat > CURRENT_THRESH).float().mean())
    print(f"  fraction of pixels above {CURRENT_THRESH} (doc default): {frac_at_current:.1%}")

    sub = flat[:: max(1, len(flat) // 500_000)]       # Otsu on a bounded subsample
    otsu = float(threshold_otsu(sub.numpy()))
    print(f"  Otsu threshold on s.sum: {otsu:.4f}")

    for thresh in (CURRENT_THRESH, round(otsu, 3)):
        per_patch = tissue_mask(s, thresh).float().mean(dim=(1, 2))
        skipped = float((per_patch < MIN_TISSUE_FRACTION).float().mean())
        print(f"  thresh {thresh:.3f}: patch tissue fraction median "
              f"{float(per_patch.median()):.1%}, p10 {float(torch.quantile(per_patch, 0.1)):.1%}, "
              f"match_stats would skip {skipped:.1%} of patches")

    # ---- 2. jitter strength --------------------------------------------------
    print("\n== stain jitter (AugCfg defaults) ==")
    g = torch.Generator().manual_seed(args.seed)
    jittered = hed_jitter(x, g, JITTER_SCALE, JITTER_SHIFT)
    diff = (jittered - x).abs()
    print(f"  mean |RGB change|: {float(diff.mean()):.4f}  "
          f"(synthetic-patch figure in the build log was ~0.12)")
    tissue = tissue_mask(s, CURRENT_THRESH)
    background = ~tissue
    if bool(tissue.any()) and bool(background.any()):
        t = diff.permute(0, 2, 3, 1)[tissue]
        b = diff.permute(0, 2, 3, 1)[background]
        print(f"  on tissue pixels: mean {float(t.mean()):.4f}; "
              f"on background pixels: mean {float(b.mean()):.4f} "
              f"(background tinting is the concern)")

    # ---- 3. HED round trip ---------------------------------------------------
    print("\n== HED round trip hed_to_rgb(rgb_to_hed(x)) ==")
    rt = hed_to_rgb(rgb_to_hed(x))
    rtd = (rt - x).abs()
    print(f"  mean |error|: {float(rtd.mean()):.5f}  max: {float(rtd.max()):.4f}  "
          f"(synthetic test tolerance was 1e-6)")

    # ---- reference stats the way stain_stats aggregates them ------------------
    st = stain_stats(x, CURRENT_THRESH)
    print(f"\n== stain_stats over tissue at thresh {CURRENT_THRESH} ==")
    print(f"  H mean {st.mean[0]:.4f} sd {st.std[0]:.4f}; "
          f"E mean {st.mean[1]:.4f} sd {st.std[1]:.4f}; D mean {st.mean[2]:.4f}")

    print("\nCALIBRATION DONE")


if __name__ == "__main__":
    main()
