"""Held-out folds at four levels (architecture doc, "Splits"). One deterministic pass from meta.parquet and a seed.

The RNG is consumed in a fixed order (patch, slide, patient) over sorted keys, so the result is byte-identical
across machines. Needs pandas (analysis side only; the VPS never imports this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ("patch", "slide", "patient", "hospital")
GROUP_KEY = {"patch": "row", "slide": "slide_key", "patient": "patient", "hospital": "hospital"}
_REQUIRED = ("row", "label", "hospital", "slide_key", "patient")


def make_splits(meta: pd.DataFrame, n_folds: int, seed: int, levels_present: int) -> pd.DataFrame:
    """Columns row, fold_patch, fold_slide, fold_patient, fold_hospital (int8), one row per patch, sorted by row.

    patch    per label: permute the rows, fold = rank % n_folds (every fold has the same label balance)
    slide    per hospital: permute its slides, position p -> fold p // (slides / n_folds) (needs slides % n_folds == 0)
    patient  levels_present == 3: fold_patient = fold_slide. Otherwise per hospital: permute its patients,
             fold = position % n_folds, so all slides of a patient share a fold
    hospital fold = hospital id (leave one hospital out; needs exactly n_folds hospitals numbered 0..n_folds-1)
    """
    if levels_present not in (3, 4):
        raise ValueError(f"levels_present must be 3 or 4, got {levels_present}")
    missing = [c for c in _REQUIRED if c not in meta.columns]
    if missing:
        raise ValueError(f"meta lacks columns {missing}")
    m = meta.sort_values("row").reset_index(drop=True)
    if m["row"].duplicated().any():
        raise ValueError("duplicate rows in meta")
    rng = np.random.default_rng(seed)
    n = len(m)

    fold_patch = np.full(n, -1, dtype=np.int8)
    labels = m["label"].to_numpy()
    for lab in np.unique(labels):
        pos = rng.permutation(np.flatnonzero(labels == lab))
        fold_patch[pos] = np.arange(len(pos)) % n_folds

    fold_slide = np.full(n, -1, dtype=np.int8)
    hosp = m["hospital"].to_numpy()
    for h in np.unique(hosp):
        in_h = hosp == h
        slides = np.sort(m.loc[in_h, "slide_key"].unique())
        if len(slides) % n_folds:
            raise ValueError(f"hospital {h} has {len(slides)} slides, not a multiple of {n_folds}")
        order = rng.permutation(slides)
        per_fold = len(slides) // n_folds
        slide_fold = {s: p // per_fold for p, s in enumerate(order)}
        fold_slide[in_h] = m.loc[in_h, "slide_key"].map(slide_fold).to_numpy()

    if levels_present == 3:
        fold_patient = fold_slide.copy()
    else:
        fold_patient = np.full(n, -1, dtype=np.int8)
        for h in np.unique(hosp):
            in_h = hosp == h
            patients = np.sort(m.loc[in_h, "patient"].unique())
            order = rng.permutation(patients)
            patient_fold = {int(p): i % n_folds for i, p in enumerate(order)}
            fold_patient[in_h] = m.loc[in_h, "patient"].map(patient_fold).to_numpy()

    if sorted(np.unique(hosp)) != list(range(n_folds)):
        raise ValueError(f"hospital level needs hospitals 0..{n_folds - 1}, found {sorted(np.unique(hosp))}")
    out = pd.DataFrame({"row": m["row"].astype(np.int32), "fold_patch": fold_patch, "fold_slide": fold_slide,
                        "fold_patient": fold_patient, "fold_hospital": hosp.astype(np.int8)})
    if (out[["fold_patch", "fold_slide", "fold_patient"]] < 0).any().any():
        raise AssertionError("a row was left without a fold")
    return out


def fold_rows(splits: pd.DataFrame, level: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_rows, test_rows) as int64 arrays of `row` values."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    mask = (splits[f"fold_{level}"] == fold).to_numpy()
    rows = splits["row"].to_numpy().astype(np.int64)
    return rows[~mask], rows[mask]


def split_report(meta: pd.DataFrame, splits: pd.DataFrame, levels_present: int | None = None) -> dict:
    """The content of splits_meta.json. `levels_present` is inferred (patient folds equal slide folds) when not given."""
    m = meta.sort_values("row").reset_index(drop=True)
    if not (m["row"].to_numpy() == splits["row"].to_numpy()).all():
        raise ValueError("meta and splits are not row-aligned")
    aliases = bool((splits["fold_patient"] == splits["fold_slide"]).all())
    if levels_present is None:
        levels_present = 3 if aliases else 4
    n_folds = int(splits["fold_hospital"].nunique())
    report: dict = {"levels_present": levels_present, "patient_aliases_slide": levels_present == 3,
                    "n_folds": n_folds, "n_patches": len(m), "levels": {}}
    for level in LEVELS:
        folds = []
        col = splits[f"fold_{level}"].to_numpy()
        for f in range(n_folds):
            test = m[col == f]
            folds.append({
                "fold": f, "n_test_patches": len(test), "n_train_patches": int(len(m) - len(test)),
                "test_tumour_fraction": float(test["label"].mean()) if len(test) else None,
                "test_slides": sorted(test["slide_key"].unique().tolist()),
                "test_patients": sorted(int(p) for p in test["patient"].unique()),
                "test_hospitals": sorted(int(h) for h in test["hospital"].unique())})
        entry: dict = {"folds": folds}
        if level == "patient":                      # hospitals that have no patient in a fold's test set
            all_h = sorted(int(h) for h in m["hospital"].unique())
            entry["hospitals_without_test_patients"] = {str(f["fold"]): [h for h in all_h if h not in f["test_hospitals"]]
                                                        for f in folds}
        report["levels"][level] = entry
    return report
