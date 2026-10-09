"""Hospital probes (architecture doc, "Hospital probes"): can a simple classifier tell which hospital a patch came from?

Three probes go into p2_probes.json: the colour histogram of the patch (no network), the plain model's features and the
stain-augmented model's features. The third shows whether stain augmentation actually reduces hospital information in the
representation. CPU only; needs scikit-learn (analysis side).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .stain import rgb_to_hed, tissue_mask
from .tensors import to_unit

N_BINS = 16
# Fixed optical-density ranges (skimage's scale: dense stain is about 0.1, the 99th percentile of the summed OD on PathMNIST was
# 0.16): every channel is binned over [0, 0.2], values above go into the last bin. Constants, never fitted to the data.
OD_RANGES = ((0.0, 0.2), (0.0, 0.2), (0.0, 0.2))
N_HOSPITALS = 5


def colour_histogram(x_u8_batch: np.ndarray) -> np.ndarray:
    """uint8 [B,H,W,3] -> float32 [B,48]: for each of the H, E, D channels a 16-bin histogram over the fixed OD range, counting
    tissue pixels only, normalised to sum to 1 per channel. A patch with no tissue pixels gives zeros."""
    if x_u8_batch.dtype != np.uint8 or x_u8_batch.ndim != 4 or x_u8_batch.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B,H,W,3], got {x_u8_batch.dtype} {x_u8_batch.shape}")
    s = rgb_to_hed(to_unit(torch.from_numpy(np.ascontiguousarray(x_u8_batch))))
    mask = tissue_mask(s).numpy()
    sn = s.numpy()
    out = np.zeros((len(x_u8_batch), 3 * N_BINS), dtype=np.float32)
    for c, (lo, hi) in enumerate(OD_RANGES):
        b = np.clip(((sn[:, c] - lo) / (hi - lo) * N_BINS).astype(np.int64), 0, N_BINS - 1)
        for i in range(len(b)):
            sel = mask[i]
            if sel.any():
                h = np.bincount(b[i][sel], minlength=N_BINS).astype(np.float32)
                out[i, c * N_BINS:(c + 1) * N_BINS] = h / h.sum()
    return out


@dataclass(frozen=True)
class ProbeResult:
    accuracy_mean: float
    accuracy_sd: float
    balanced_accuracy_mean: float
    balanced_accuracy_sd: float
    confusion: list[list[int]]          # summed over the folds; rows = true hospital, columns = predicted
    chance: float
    n: int
    n_folds: int
    fold_accuracy: list[float]           # the five folds, in order: lets two probes be compared fold by fold
    fold_balanced_accuracy: list[float]


def run_probe(X: np.ndarray, hospital: np.ndarray, slide_key: np.ndarray, seed: int = 0, n_folds: int = 5) -> ProbeResult:
    """StandardScaler + LogisticRegression(max_iter=2000, C=1.0), GroupKFold by slide: every test fold holds whole slides the
    probe never saw, so it must recognise a hospital from slides it has not trained on. Chance is 1 / number of hospitals."""
    X, hospital = np.asarray(X, dtype=np.float32), np.asarray(hospital)
    classes = np.unique(hospital)
    accs, baccs = [], []
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for tr, te in GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, hospital, np.asarray(slide_key)):
        if len(np.unique(hospital[tr])) != len(classes):
            raise ValueError("a training fold is missing a hospital")
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], hospital[tr])
        pred = clf.predict(X[te])
        accs.append(float((pred == hospital[te]).mean()))
        baccs.append(float(balanced_accuracy_score(hospital[te], pred)))
        cm += confusion_matrix(hospital[te], pred, labels=classes)
    return ProbeResult(float(np.mean(accs)), float(np.std(accs)), float(np.mean(baccs)), float(np.std(baccs)),
                       cm.tolist(), 1.0 / len(classes), len(X), n_folds, accs, baccs)
