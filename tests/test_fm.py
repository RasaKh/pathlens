"""pathlens/fm.py and scripts/analyse_p3.py leakage guards on synthetic data (no downloads)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathlens import fm
from pathlens.evaluate import bootstrap_ci
from pathlens.probes import run_probe

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("analyse_p3", ROOT / "scripts" / "analyse_p3.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def world(n_per=400, d=12, seed=0):
    """5 hospitals x 4 slides; features carry a hospital offset plus a weaker tumour signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(5):
        for s in range(4):
            y = rng.integers(0, 2, n_per)
            X = rng.normal(size=(n_per, d)) + 3.0 * np.eye(d)[h % d] + 1.2 * y[:, None] * np.eye(d)[5]
            rows.append(pd.DataFrame(X).assign(label=y, hospital=h, slide_key=f"h{h}s{s}"))
    df = pd.concat(rows, ignore_index=True)
    return df, df[list(range(d))].to_numpy(np.float32)


def test_hospital_projector_fits_on_training_rows_only():
    df, X = world()
    tr = (df["hospital"] != 2).to_numpy()
    a = fm.HospitalProjector().fit(X[tr], df["hospital"].to_numpy()[tr])
    X2 = X.copy()
    X2[~tr] = np.random.default_rng(9).normal(size=X2[~tr].shape) * 50          # change the test rows entirely
    b = fm.HospitalProjector().fit(X2[tr], df["hospital"].to_numpy()[tr])
    assert np.array_equal(a.basis, b.basis) and np.array_equal(a.scaler.mean_, b.scaler.mean_)
    assert a.basis.shape[1] <= 3
    assert np.allclose(a.basis.T @ a.basis, np.eye(a.basis.shape[1]), atol=1e-5)


def test_projection_removes_hospital_information():
    df, X = world()
    h = df["hospital"].to_numpy()
    P = fm.HospitalProjector().fit(X, h)
    before = run_probe(X, h, df["slide_key"].to_numpy()).accuracy_mean
    after = run_probe(P.transform(X), h, df["slide_key"].to_numpy()).accuracy_mean
    assert before > 0.9 and after < before - 0.3


def test_threshold_gives_target_sensitivity_and_is_trainset_only():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = rng.normal(size=5000) + 1.5 * y
    thr = fm.threshold_for_sensitivity(y, s, 0.9)
    sens, _ = fm.sens_spec(y, s, thr)
    assert 0.9 <= sens < 0.9 + 0.005
    assert fm.sens_spec(y, s, thr + 1e-6)[0] < sens or (s[y == 1] >= thr + 1e-6).mean() == sens


def test_ece_known_answers():
    y = np.array([0, 1] * 500)
    assert fm.ece(y, np.full(1000, 0.5)) == pytest.approx(0.0)           # calibrated
    assert fm.ece(np.ones(100), np.full(100, 0.1)) == pytest.approx(0.9)  # confidently wrong
    assert sum(r["n"] for r in fm.reliability(y, np.random.default_rng(0).random(1000))) == 1000


def test_shuffled_label_probe_is_at_chance():
    df, X = world()
    h = np.random.default_rng(0).permutation(df["hospital"].to_numpy())
    assert abs(run_probe(X, h, df["slide_key"].to_numpy()).accuracy_mean - 0.2) < 0.06


def test_oof_scores_cover_every_row_without_sharing_slides():
    df, X = world(n_per=100)
    s = fm.oof_scores(X, df["label"].to_numpy(), df["slide_key"].to_numpy(), 1.0)
    assert np.isfinite(s).all() and len(s) == len(df)


def test_fit_one_levels_never_share_slides_and_score_each_row_once():
    df, X = world(n_per=150)
    t = df.assign(row=np.arange(len(df)), idx=np.arange(len(df)))
    t["fold_hospital"] = t["hospital"]
    sl = {k: i % 5 for i, k in enumerate(sorted(t["slide_key"].unique()))}
    t["fold_slide"] = t["slide_key"].map(sl)
    t["fold_patient"] = t["fold_slide"]
    t["fold_patch"] = np.random.default_rng(0).integers(0, 5, len(t))
    for level in an.LEVELS:
        got = [an.fit_one(X, t, level, k) for k in range(5)]
        idx = np.concatenate([g["idx"] for g in got])
        assert sorted(idx) == list(range(len(t)))
    h = [an.fit_one(X, t, "hospital", k) for k in range(5)]
    assert all("threshold" in g and "score_rq3" in g for g in h)
    with pytest.raises(AssertionError):
        an.check_split(t, np.arange(10), np.arange(5, 15), "patch")
    same = np.flatnonzero(t["slide_key"] == "h0s0")
    with pytest.raises(AssertionError):
        an.check_split(t, same[:5], same[5:10], "slide")


def test_bootstrap_interval_covers_a_known_auroc():
    df, _ = world(n_per=300)
    df["score"] = df["label"] * 1.0 + np.random.default_rng(0).normal(size=len(df))          # AUROC about 0.76
    est, lo, hi = bootstrap_ci(df, lambda d: an._auroc(d), "slide_key", 200, np.random.default_rng(0))
    assert lo < est < hi and 0.70 < est < 0.82 and hi - lo < 0.1


def test_validate_p3_results_rejects_leakage_signal_and_missing_keys():
    import copy
    import json

    from pathlens.result_schemas import SchemaError, validate_p3_results
    p = ROOT.parent / "pathlens" / "workdir" / "artifacts" / "results" / "p3_results.json"
    if not p.exists():
        pytest.skip("p3_results.json not built")
    d = json.loads(p.read_text())
    validate_p3_results(d)
    bad = copy.deepcopy(d)
    bad["rq1_probe"]["shuffled_control_accuracy_mean"] = 0.6
    with pytest.raises(SchemaError):
        validate_p3_results(bad)
    bad = copy.deepcopy(d)
    del bad["per_hospital"][0]["rq4"]["ece"]
    with pytest.raises(SchemaError):
        validate_p3_results(bad)
