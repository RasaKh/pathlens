"""Which Phase 2 patches the analysis needs as images (selected from metadata only, seeded), and the file format they come in.

The analysis (colour-histogram probe, map thumbnails, stain statistics, curated patches) needs pixels for about 126,000 of the
455,954 patches. They are re-read from the public archive in one streaming pass (scripts/fetch_subset_p2.py) instead of keeping
the 12.6 GB array. Needs pandas (analysis side only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAIN_PER_HOSPITAL = 20_000
STAIN_SEED = 3
CURATE_SEED = 0
PICKER_PER_LABEL = 24
ANNOTATE_PER_LABEL = 20
CURATE_HOSPITAL = 2


def stain_sample_rows(meta: pd.DataFrame, per_hospital: int = STAIN_PER_HOSPITAL, seed: int = STAIN_SEED) -> np.ndarray:
    """per_hospital patches from each hospital, seeded. The doc says "training patches"; hospitals 1 and 2 have no patches in
    WILDS's own training split, so this samples all patches of each hospital (logged as a deviation). Sorted row numbers."""
    rng = np.random.default_rng(seed)
    out = []
    for _, g in meta.groupby("hospital", sort=True):
        rows = g["row"].to_numpy()
        out.append(rng.choice(rows, size=min(per_hospital, len(rows)), replace=False))
    return np.sort(np.concatenate(out))


def curation_rows(meta: pd.DataFrame, seed: int = CURATE_SEED, picker_per_label: int = PICKER_PER_LABEL,
                  annotate_per_label: int = ANNOTATE_PER_LABEL, hospital: int = CURATE_HOSPITAL) -> dict[str, np.ndarray]:
    """The Phase 2 picker (24 tumour + 24 normal) and the annotation set (20 + 20, disjoint from the picker), all from one
    hospital, random within each stratum. Hospital 2 is the live model's held-out hospital, so the live model has never seen
    any of these patches."""
    rng = np.random.default_rng(seed)
    picker, annotate = [], []
    for label in (0, 1):
        rows = meta.loc[(meta["hospital"] == hospital) & (meta["label"] == label), "row"].to_numpy()
        perm = rng.permutation(rows)
        picker.append(perm[:picker_per_label])
        annotate.append(perm[picker_per_label:picker_per_label + annotate_per_label])
    return {"p2_picker": np.sort(np.concatenate(picker)), "annotate": np.sort(np.concatenate(annotate))}


def wanted_rows(map_rows: np.ndarray, probe_rows: np.ndarray, stain_rows: np.ndarray, curated: dict[str, np.ndarray]) -> np.ndarray:
    return np.unique(np.concatenate([map_rows, probe_rows, stain_rows, *curated.values()])).astype(np.int64)
