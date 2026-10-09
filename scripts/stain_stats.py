#!/usr/bin/env python3
"""WP10: per-hospital stain statistics, and a check of the tissue threshold on Camelyon colours.

    uv run python -m scripts.stain_stats        # needs data/p2/subset_u8.npy (scripts/fetch_subset_p2.py)

stain_stats.json: for each hospital, the mean and standard deviation of the H, E and D optical density over tissue pixels
(pathlens.stain.tissue_mask at the library default) for the seeded sample of 20,000 patches per hospital. Also writes
reports/tissue_threshold_camelyon.json: Otsu's threshold on the summed optical density of Camelyon patches against the
default 0.0715 (calibrated on PathMNIST), and how many patches the default would skip in match_stats. The default is NOT changed
here: the trained models' augmentation used it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.filters import threshold_otsu

from pathlens.paths import artifacts_dir, data_dir, reports_dir
from pathlens.stain import MIN_TISSUE_FRACTION, TISSUE_THRESH, rgb_to_hed, tissue_mask
from pathlens.subset import Subset, stain_sample_rows
from pathlens.tensors import to_unit

BATCH = 1000
EXTREME_OD = 0.5          # summed optical density above this is treated as black/artifact when finding Otsu's threshold


def hospital_stats(images_for_rows, rows: np.ndarray, thresh: float = TISSUE_THRESH) -> dict:
    """Mean and standard deviation of (H, E, D) over the tissue pixels of `rows`; `images_for_rows(chunk)` returns uint8 patches."""
    n, s1, s2, patches_with_tissue = 0, np.zeros(3), np.zeros(3), 0
    for a in range(0, len(rows), BATCH):
        x = to_unit(torch.from_numpy(images_for_rows(rows[a:a + BATCH])))
        s = rgb_to_hed(x).double()
        mask = tissue_mask(s.float(), thresh)
        v = s.permute(0, 2, 3, 1)[mask].numpy()
        n += len(v)
        s1 += v.sum(0)
        s2 += (v ** 2).sum(0)
        patches_with_tissue += int(mask.flatten(1).any(1).sum())
    if n == 0:
        raise ValueError("no tissue pixels")
    mean = s1 / n
    return {"mean": [float(m) for m in mean], "std": [float(m) for m in np.sqrt(np.maximum(s2 / n - mean ** 2, 0))],
            "n_pixels": int(n), "n_patches": len(rows)}


def threshold_report(images_for_rows, rows: np.ndarray, n_patches: int = 4000, seed: int = 0) -> dict:
    """Otsu on the summed OD of a seeded sample of patches, and what the default threshold does to them."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(rows, size=min(n_patches, len(rows)), replace=False))
    sums, frac = [], []
    for a in range(0, len(pick), BATCH):
        s = rgb_to_hed(to_unit(torch.from_numpy(images_for_rows(pick[a:a + BATCH]))))
        tot = s.sum(1)
        sums.append(tot.flatten()[::7].numpy())
        frac.append((tot > TISSUE_THRESH).float().mean(dim=(1, 2)).numpy())
    allsum, frac = np.concatenate(sums), np.concatenate(frac)
    # About 0.2% of Camelyon pixels are pure black (summed OD about 2.1): a second mode that drags Otsu's threshold to about 0.84.
    # The reported threshold excludes pixels above EXTREME_OD; the raw value is kept for the record.
    otsu = float(threshold_otsu(allsum[allsum <= EXTREME_OD]))
    out = {"n_patches": len(pick), "default_threshold": TISSUE_THRESH, "otsu_on_camelyon": otsu,
           "otsu_raw_including_black_pixels": float(threshold_otsu(allsum)),
           "share_extreme_pixels": float((allsum > EXTREME_OD).mean()), "extreme_od": EXTREME_OD,
           "median_tissue_fraction_at_default": float(np.median(frac)),
           "share_patches_match_stats_would_skip_at_default": float((frac < MIN_TISSUE_FRACTION).mean()),
           "percentiles_of_summed_od": {str(q): float(np.percentile(allsum, q)) for q in (50, 75, 90, 95, 99)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    p2 = args.p2_dir or data_dir() / "p2"
    sub = Subset(p2)
    meta = pd.read_parquet(p2 / "meta.parquet")
    rows = stain_sample_rows(meta)
    stats = {}
    for h in sorted(meta["hospital"].unique()):
        hr = rows[meta.set_index("row").loc[rows, "hospital"].to_numpy() == h]
        stats[str(int(h))] = hospital_stats(sub.get, hr)
        print(f"hospital {h}: {stats[str(int(h))]['n_patches']:,} patches, {stats[str(int(h))]['n_pixels']:,} tissue pixels, "
              f"mean (H,E,D) {[round(v, 4) for v in stats[str(int(h))]['mean']]}")
    out = args.out or artifacts_dir() / "stain_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    rep = threshold_report(sub.get, rows)
    rep["generated_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "tissue_threshold_camelyon.json").write_text(json.dumps(rep, indent=2))
    print(f"tissue threshold: library default {TISSUE_THRESH}, Otsu on Camelyon {rep['otsu_on_camelyon']:.4f}; "
          f"median tissue fraction at the default {100 * rep['median_tissue_fraction_at_default']:.1f}%, "
          f"{100 * rep['share_patches_match_stats_would_skip_at_default']:.1f}% of patches would be skipped")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
