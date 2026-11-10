"""Phase 3 pure functions: heads on frozen foundation-model features, hospital-direction projection, threshold transfer.

Everything that learns anything (scaler, head, projection, threshold) is fitted on the rows it is given and nothing else;
the caller decides which rows those are. numpy / pandas / scikit-learn only, so it also runs on the GPU box.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0, 10.0)
TARGET_SENSITIVITY = 0.90
ECE_BINS = 15


def make_head(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def fit_score(Xtr, ytr, Xte, C: float) -> np.ndarray:
    """P(tumour) for Xte from a head fitted on (Xtr, ytr) only."""
    return make_head(C).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def choose_C(X, y, groups, seed: int = 0, n_inner: int = 3, grid=C_GRID, max_rows: int | None = None) -> tuple[float, dict]:
    """C with the best mean AUROC over slide-grouped inner folds of the training rows. max_rows (if given) draws a random
    subset of the training rows first, to bound the cost; the final head is still fitted on all of them."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if max_rows is not None and len(X) > max_rows:
        ix = np.sort(np.random.default_rng(seed).choice(len(X), max_rows, replace=False))
        X, y, groups = X[ix], y[ix], groups[ix]
    scores = {}
    for C in grid:
        aucs = []
        for tr, te in GroupKFold(n_splits=n_inner, shuffle=True, random_state=seed).split(X, y, groups):
            aucs.append(roc_auc_score(y[te], fit_score(X[tr], y[tr], X[te], C)))
        scores[C] = float(np.mean(aucs))
    return max(scores, key=scores.get), scores


def oof_scores(X, y, groups, C: float, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(tumour) for every row, from slide-grouped folds (no slide in both sides of any fold)."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    out = np.full(len(X), np.nan)
    for tr, te in GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y, groups):
        out[te] = fit_score(X[tr], y[tr], X[te], C)
    if np.isnan(out).any():
        raise AssertionError("a row was not scored out-of-fold")
    return out


class HospitalProjector:
    """Removes the directions that separate the hospitals. Fit on the training rows only (features and their hospital labels);
    `transform` then applies the same fixed map to any rows. The directions are the row space of a multinomial logistic
    regression's coefficients on standardised features (at most n_hospitals - 1 of them), orthonormalised."""

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C, self.seed = C, seed
        self.scaler: StandardScaler | None = None
        self.basis: np.ndarray | None = None       # [d, r] orthonormal columns

    def fit(self, X, hospital) -> HospitalProjector:
        X, hospital = np.asarray(X, dtype=np.float32), np.asarray(hospital)
        k = len(np.unique(hospital))
        if k < 2:
            raise ValueError("need at least two hospitals")
        self.scaler = StandardScaler().fit(X)
        clf = LogisticRegression(C=self.C, max_iter=1000).fit(self.scaler.transform(X), hospital)
        coef = clf.coef_ - clf.coef_.mean(axis=0, keepdims=True)           # centred over classes: rank <= k - 1
        _, s, vt = np.linalg.svd(coef, full_matrices=False)
        r = int((s > s[0] * 1e-6).sum())
        self.basis = vt[:r].T.astype(np.float32)
        return self

    def transform(self, X) -> np.ndarray:
        """Standardised features (with the fitted scaler) minus their component along the hospital directions."""
        Z = self.scaler.transform(np.asarray(X, dtype=np.float32))
        return Z - (Z @ self.basis) @ self.basis.T


def threshold_for_sensitivity(y, scores, target: float = TARGET_SENSITIVITY) -> float:
    """The highest threshold t (predict tumour when score >= t) with sensitivity >= target on these labelled scores."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    pos = np.sort(scores[y == 1])
    if len(pos) == 0:
        raise ValueError("no positives")
    k = int(np.floor((1.0 - target) * len(pos) + 1e-9))               # positives allowed below the threshold
    return float(pos[k])


def sens_spec(y, scores, thr: float) -> tuple[float, float]:
    y, pred = np.asarray(y), np.asarray(scores) >= thr
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("sensitivity and specificity need both classes")
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def reliability(y, p, bins: int = ECE_BINS) -> list[dict]:
    """Equal-width bins on [0, 1]: count, mean predicted probability, observed tumour fraction (empty bins omitted)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ix = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": int(b), "n": int((ix == b).sum()), "mean_p": float(p[ix == b].mean()), "frac_pos": float(y[ix == b].mean())}
            for b in range(bins) if (ix == b).any()]


def ece(y, p, bins: int = ECE_BINS) -> float:
    """Expected calibration error: bin-size-weighted mean of |observed fraction - mean predicted| over equal-width bins."""
    rel = reliability(y, p, bins)
    n = sum(r["n"] for r in rel)
    return float(sum(r["n"] * abs(r["frac_pos"] - r["mean_p"]) for r in rel) / n)
