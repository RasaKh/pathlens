#!/usr/bin/env python3
"""Check the re-extracted patches against the GPU session's own features: the same model on the same patches must give the same
features. If the rows or the pixels were wrong, they would not match.

    uv run python -m scripts.verify_subset_p2 [--n 1500]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pathlens.model import load_for_inference
from pathlens.paths import data_dir, home
from pathlens.run_all import pooled_features
from pathlens.subset import Subset

LIVE = {"plain": "p2-hospital-f2-plain-s0", "aug": "p2-hospital-f2-aug-s0"}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """a: fresh features, b: the saved float16 ones."""
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    rel = np.abs(a - b).max(1) / (np.abs(b).max(1) + 1e-12)
    return {"cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()), "max_relative_diff": float(rel.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--runs-dir", type=Path, default=home() / "backup_phase2" / "runs")
    args = ap.parse_args()
    p2, an = data_dir() / "p2", home() / "analysis_inputs"
    sub = Subset(p2)
    rows = np.load(an / "probe_rows.npy")
    pick = np.sort(np.random.default_rng(0).choice(len(rows), args.n, replace=False))
    imgs = torch.from_numpy(sub.get(rows[pick]))
    for name, rid in LIVE.items():
        model, _ = load_for_inference(args.runs_dir / rid, "cpu")
        fresh = pooled_features(model, imgs, np.arange(len(pick)), "cpu")
        saved = np.load(an / f"probe_{name}.npy")[pick].astype(np.float32)
        r = compare(fresh, saved)
        print(f"{name:<6} {args.n} patches: cosine to the GPU's features min {r['cosine_min']:.6f}, mean {r['cosine_mean']:.6f}; "
              f"largest relative difference {r['max_relative_diff']:.4f}")


if __name__ == "__main__":
    main()
