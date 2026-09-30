"""Splits (architecture doc, "What the tests assert"): on a synthetic table, and on the real metadata if present."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathlens.splits import GROUP_KEY, LEVELS, fold_rows, make_splits, split_report

REAL = Path.home() / "pathlens" / "data" / "p2" / "meta.parquet"


def synthetic(per_slide: int = 12, seed: int = 0) -> pd.DataFrame:
    """5 hospitals x 10 slides; hospital h has patients h*10 .. h*10+6, with 3 of them holding 2 slides."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(5):
        patients = [h * 10 + i for i in (0, 0, 1, 1, 2, 2, 3, 4, 5, 6)]        # patients 0, 1, 2 have two slides
        for s, p in enumerate(patients):
            for k in range(per_slide):
                rows.append({"hospital": h, "patient": p, "slide_key": f"h{h}-p{p:03d}-n{s}", "label": int(rng.random() < 0.5)})
    df = pd.DataFrame(rows)
    df.insert(0, "row", np.arange(len(df)))
    return df


@pytest.fixture(scope="module")
def meta():
    return synthetic()


@pytest.fixture(scope="module", params=["synthetic", "real"])
def case(request, meta):
    if request.param == "synthetic":
        return meta, make_splits(meta, 5, 0, 4), 4
    if not REAL.exists():
        pytest.skip("real metadata not on this machine")
    real = pd.read_parquet(REAL)
    return real, make_splits(real, 5, 0, 4), 4


def test_every_row_is_held_out_exactly_once_per_level(case):
    m, sp, _ = case
    assert len(sp) == len(m) and list(sp.columns) == ["row", "fold_patch", "fold_slide", "fold_patient", "fold_hospital"]
    assert all(str(sp[f"fold_{lv}"].dtype) == "int8" for lv in LEVELS)
    for lv in LEVELS:
        assert set(sp[f"fold_{lv}"].unique()) == set(range(5))
        held = np.concatenate([fold_rows(sp, lv, f)[1] for f in range(5)])
        assert sorted(held) == sorted(sp["row"])                                  # each row once
        for f in range(5):
            tr, te = fold_rows(sp, lv, f)
            assert len(tr) + len(te) == len(m) and not set(tr) & set(te)


def test_train_and_test_groups_are_disjoint_at_every_level(case):
    m, sp, _ = case
    m = m.sort_values("row").reset_index(drop=True)
    for lv in LEVELS:
        key = m[GROUP_KEY[lv]]
        for f in range(5):
            tr, te = fold_rows(sp, lv, f)
            assert not set(key[m["row"].isin(tr)]) & set(key[m["row"].isin(te)]), (lv, f)


def test_hospital_level_also_separates_slides_and_patients(case):
    m, sp, _ = case
    m = m.sort_values("row").reset_index(drop=True)
    for f in range(5):
        tr, te = fold_rows(sp, "hospital", f)
        for key in ("slide_key", "patient"):
            assert not set(m.loc[m["row"].isin(tr), key]) & set(m.loc[m["row"].isin(te), key])
        assert set(m.loc[m["row"].isin(te), "hospital"]) == {f}


def test_slide_level_every_fold_has_two_slides_from_every_hospital(case):
    m, sp, _ = case
    m = m.sort_values("row").reset_index(drop=True)
    for f in range(5):
        test = m[sp["fold_slide"].to_numpy() == f]
        per_h = test.groupby("hospital")["slide_key"].nunique()
        assert set(per_h.index) == {0, 1, 2, 3, 4} and (per_h == 2).all()


def test_patient_level_keeps_every_patients_slides_together_and_spreads_patients(case):
    m, sp, _ = case
    m = m.sort_values("row").reset_index(drop=True)
    assert (m.assign(f=sp["fold_patient"].to_numpy()).groupby("patient")["f"].nunique() == 1).all()
    per = m.assign(f=sp["fold_patient"].to_numpy()).groupby(["hospital", "f"])["patient"].nunique().unstack(fill_value=0)
    assert (per.max(axis=1) - per.min(axis=1) <= 1).all()                          # patients spread evenly over folds


def test_patch_level_balances_labels_in_every_fold(case):
    m, sp, _ = case
    m = m.sort_values("row").reset_index(drop=True)
    for lab in (0, 1):
        counts = [int(((m["label"] == lab).to_numpy() & (sp["fold_patch"].to_numpy() == f)).sum()) for f in range(5)]
        assert max(counts) - min(counts) <= 1


def test_same_seed_same_frame_and_different_seed_different(meta):
    a, b, c = make_splits(meta, 5, 7, 4), make_splits(meta, 5, 7, 4), make_splits(meta, 5, 8, 4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_output_is_byte_identical_over_time(meta):
    """A golden hash of this implementation's output: the folds for this table and seed must not change."""
    sp = make_splits(meta, 5, 0, 4)
    digest = hashlib.sha256(sp.to_numpy(dtype=np.int64).tobytes()).hexdigest()[:16]
    assert digest == GOLDEN


def test_three_levels_alias_patient_to_slide(meta):
    sp = make_splits(meta, 5, 0, 3)
    assert (sp["fold_patient"] == sp["fold_slide"]).all()
    rep = split_report(meta, sp, 3)
    assert rep["levels_present"] == 3 and rep["patient_aliases_slide"] is True
    assert split_report(meta, sp)["patient_aliases_slide"] is True                 # inferred


def test_input_order_does_not_matter(meta):
    shuffled = meta.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(make_splits(shuffled, 5, 0, 4), make_splits(meta, 5, 0, 4))


def test_bad_inputs_are_refused(meta):
    with pytest.raises(ValueError, match="levels_present"):
        make_splits(meta, 5, 0, 5)
    with pytest.raises(ValueError, match="lacks columns"):
        make_splits(meta.drop(columns=["patient"]), 5, 0, 4)
    with pytest.raises(ValueError, match="not a multiple"):
        make_splits(meta[meta["slide_key"] != meta["slide_key"].iloc[0]], 5, 0, 4)
    with pytest.raises(ValueError, match="level must be"):
        fold_rows(make_splits(meta, 5, 0, 4), "nope", 0)


def test_report_contents(case):
    m, sp, lp = case
    rep = split_report(m, sp, lp)
    assert rep["levels_present"] == 4 and rep["patient_aliases_slide"] is False and rep["n_folds"] == 5
    assert set(rep["levels"]) == set(LEVELS)
    for lv in LEVELS:
        folds = rep["levels"][lv]["folds"]
        assert sum(f["n_test_patches"] for f in folds) == len(m)
        for f in folds:
            assert f["n_test_patches"] + f["n_train_patches"] == len(m)
            assert f["test_hospitals"] == sorted(f["test_hospitals"]) and 0 <= f["test_tumour_fraction"] <= 1
    assert rep["levels"]["hospital"]["folds"][2]["test_hospitals"] == [2]
    assert all(h == [] for h in rep["levels"]["patient"]["hospitals_without_test_patients"].values())
    json.dumps(rep)


def test_report_records_hospitals_missing_from_a_patient_fold():
    df = synthetic()
    in4 = df["hospital"] == 4
    df.loc[in4, "patient"] = 40 + (df.loc[in4, "patient"] - 40) % 3               # hospital 4 keeps only 3 patients
    sp = make_splits(df, 5, 0, 4)
    missing = split_report(df, sp, 4)["levels"]["patient"]["hospitals_without_test_patients"]
    assert missing["3"] == [4] and missing["4"] == [4]                               # folds 3 and 4 have no hospital-4 patient
    assert missing["0"] == missing["1"] == missing["2"] == []


GOLDEN = "fd6d998a140e2b4f"      # recorded from this implementation on 2026-10-11: guards against accidental change
