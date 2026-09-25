"""Metrics and bootstrap intervals (architecture doc, "Evaluation and statistics")."""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

DISCARD_WARN_FRACTION = 0.01


def binary_metrics(y, p, thr: float = 0.5) -> dict:
    """acc, bal_acc, auroc, n, n_pos. Raises ValueError when only one class is present."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("AUROC needs both classes present")      # sklearn only warns and returns nan
    pred = (p >= thr).astype(int)
    return {"acc": float((pred == y).mean()), "bal_acc": float(balanced_accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, p)), "n": len(y), "n_pos": int(y.sum())}


def multiclass_metrics(y, probs) -> dict:
    """acc, bal_acc, macro_auroc_ovr, recall_per_class. Classes absent from y are left out of the
    macro AUROC and have recall None."""
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.shape[1]
    pred = probs.argmax(1)
    present = np.unique(y)
    recall = [float((pred[y == c] == c).mean()) if (y == c).any() else None for c in range(k)]
    if len(present) < 2:
        raise ValueError("need at least two classes present for AUROC")
    aucs = [roc_auc_score((y == c).astype(int), probs[:, c]) for c in present]
    return {"acc": float((pred == y).mean()), "bal_acc": float(np.mean([r for r in recall if r is not None])),
            "macro_auroc_ovr": float(np.mean(aucs)), "recall_per_class": recall}


def _draw(groups: list[np.ndarray] | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n, n)
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in pick])


def _groups(df: pd.DataFrame, unit: str | None) -> list[np.ndarray] | None:
    if unit is None:
        return None
    return [np.asarray(ix) for ix in df.groupby(unit, sort=True).indices.values()]


def _finish(vals: list[float], discarded: int, B: int, info: dict | None) -> None:
    if info is not None:
        info.update(discarded=discarded, B=B)
    if discarded / B > DISCARD_WARN_FRACTION:
        warnings.warn(f"{discarded} of {B} bootstrap draws had one class only and were discarded "
                      f"(> {DISCARD_WARN_FRACTION:.0%})", stacklevel=3)


def bootstrap_ci(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], unit: str | None, B: int,
                 rng: np.random.Generator, info: dict | None = None) -> tuple[float, float, float]:
    """(estimate on all rows, 2.5th, 97.5th percentile). unit None resamples rows; a column name
    resamples those clusters with replacement and concatenates their rows. A draw for which `stat`
    raises ValueError (one class only) is discarded and counted in `info["discarded"]`; more than 1%
    discarded raises a warning."""
    df = df.reset_index(drop=True)
    groups = _groups(df, unit)
    est = float(stat(df))
    vals, discarded = [], 0
    for _ in range(B):
        try:
            vals.append(float(stat(df.iloc[_draw(groups, len(df), rng)])))
        except ValueError:
            discarded += 1
    _finish(vals, discarded, B, info)
    if not vals:
        raise ValueError("every bootstrap draw was discarded")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def paired_cluster_delta(df_a: pd.DataFrame, df_b: pd.DataFrame, stat: Callable[[pd.DataFrame], float],
                         unit: str | None, B: int, rng: np.random.Generator, info: dict | None = None
                         ) -> tuple[float, float, float]:
    """stat(b) - stat(a) with the same resampled clusters for both models in every draw. Both frames
    must hold the same units; with unit None they must be row-aligned."""
    df_a, df_b = df_a.reset_index(drop=True), df_b.reset_index(drop=True)
    if unit is None:
        if len(df_a) != len(df_b):
            raise ValueError("row-aligned frames needed when unit is None")
        ga = gb = None
    else:
        ka = df_a.groupby(unit, sort=True).indices
        kb = df_b.groupby(unit, sort=True).indices
        if set(ka) != set(kb):
            raise ValueError("both frames must contain the same units")
        keys = sorted(ka)
        ga, gb = [np.asarray(ka[k]) for k in keys], [np.asarray(kb[k]) for k in keys]
    delta = float(stat(df_b) - stat(df_a))
    vals, discarded = [], 0
    for _ in range(B):
        if unit is None:
            ia = ib = rng.integers(0, len(df_a), len(df_a))
        else:
            pick = rng.integers(0, len(ga), len(ga))
            ia = np.concatenate([ga[i] for i in pick])
            ib = np.concatenate([gb[i] for i in pick])
        try:
            vals.append(float(stat(df_b.iloc[ib]) - stat(df_a.iloc[ia])))
        except ValueError:
            discarded += 1
    _finish(vals, discarded, B, info)
    if not vals:
        raise ValueError("every bootstrap draw was discarded")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return delta, float(lo), float(hi)
