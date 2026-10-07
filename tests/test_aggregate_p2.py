"""scripts/aggregate_p2.py and pathlens/result_schemas.py on a synthetic Phase 2 dataset (small B, a few seconds)."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pathlens.result_schemas import SchemaError, validate_p2_levels, validate_p2_repair
from pathlens.splits import make_splits

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("aggregate_p2", ROOT / "scripts" / "aggregate_p2.py")
ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ag)

PER_SLIDE = 24
N = 5 * 10 * PER_SLIDE


def synthetic_meta() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for h in range(5):
        patients = [h * 10 + i for i in (0, 0, 1, 1, 2, 2, 3, 4, 5, 6)]
        for s, p in enumerate(patients):
            tumour = s % 2                                               # slides are nearly pure, as in the real data
            for _ in range(PER_SLIDE):
                rows.append({"hospital": h, "patient": p, "slide_key": f"h{h}-p{p:03d}-n{s}",
                             "label": int(rng.random() < (0.9 if tumour else 0.1))})
    df = pd.DataFrame(rows)
    df.insert(0, "row", np.arange(len(df)))
    return df


def write_run(root: Path, rid: str, preds: pd.DataFrame, model_hash: str) -> None:
    d = root / rid
    d.mkdir(parents=True)
    preds.to_parquet(d / "predictions.parquet")
    (d / "DONE").write_text(json.dumps({"config_hash": "x", "model_hash": model_hash, "sha256": {}, "wall_seconds": 1}))


def scores(meta: pd.DataFrame, skill: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1 / (1 + np.exp(-(skill * (2 * meta["label"].to_numpy() - 1) + rng.normal(0, 1, len(meta)))))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    meta = synthetic_meta()
    sp = make_splits(meta, 5, 0, 4)
    skill = {"patch": 2.5, "slide": 2.0, "patient": 1.8, "hospital": 1.0}
    for level in ag.LEVELS:
        for f in range(5):
            test = meta[(sp[f"fold_{level}"] == f).to_numpy()]
            p = scores(test, skill[level], seed=f)
            write_run(root, ag.run_name(level, f, "plain", 0), test.assign(pred=(p > 0.5).astype(int), p_tumour=p)
                      [["row", "label", "pred", "p_tumour", "hospital", "slide_key", "patient"]], f"{level}{f}p0")
    for variant, boost in (("plain", 0.0), ("aug", 0.8)):
        for seed in (0, 1):
            if variant == "plain" and seed == 0:
                continue                                                   # already written above (hospital level)
            for f in range(5):
                test = meta[(sp["fold_hospital"] == f).to_numpy()]
                p = scores(test, 1.0 + boost, seed=100 * seed + f + (7 if variant == "aug" else 0))
                write_run(root, ag.run_name("hospital", f, variant, seed), test.assign(pred=(p > 0.5).astype(int), p_tumour=p)
                          [["row", "label", "pred", "p_tumour", "hospital", "slide_key", "patient"]], f"h{f}{variant}{seed}")
    return {"root": root, "meta": meta, "sp": sp}


def levels_json(world, levels_present=4, rng_seed=0):
    d, src = ag.build_levels(world["root"], levels_present, np.random.default_rng(rng_seed), 40, 60, n_expected=N, log=lambda *_: None)
    return ag.stamp(d, src)


def test_every_patch_must_appear_exactly_once(world, tmp_path):
    df, src = ag.load_level(world["root"], "slide", "plain", 0, N)
    assert len(df) == N and df["row"].is_unique and len(src) == 5 and df["fold"].nunique() == 5
    with pytest.raises(ValueError, match="expected"):
        ag.load_level(world["root"], "slide", "plain", 0, N + 1)
    dup = tmp_path / "runs"
    for f in range(5):
        src_dir = world["root"] / ag.run_name("slide", f, "plain", 0)
        write_run(dup, ag.run_name("slide", f, "plain", 0), pd.read_parquet(src_dir / "predictions.parquet"), "m")
    extra = pd.read_parquet(dup / ag.run_name("slide", 0, "plain", 0) / "predictions.parquet")
    other = pd.read_parquet(dup / ag.run_name("slide", 1, "plain", 0) / "predictions.parquet")
    other.iloc[:5] = extra.iloc[:5].to_numpy()                             # five patches predicted twice
    other.to_parquet(dup / ag.run_name("slide", 1, "plain", 0) / "predictions.parquet")
    with pytest.raises(ValueError, match="more than once"):
        ag.load_level(dup, "slide", "plain", 0, None)


def test_levels_file_matches_the_schema_and_the_direct_computation(world):
    d = levels_json(world)
    validate_p2_levels(d)
    assert [e["level"] for e in d["levels"]] == ["patch", "slide", "patient", "hospital"]
    df, _ = ag.load_level(world["root"], "hospital", "plain", 0, N)
    hosp = d["levels"][3]
    assert hosp["pooled"]["auroc"] == pytest.approx(roc_auc_score(df.label, df.p_tumour))
    assert hosp["pooled"]["n"] == N and len(hosp["per_hospital"]) == 5 and len(hosp["folds"]) == 5
    assert d["levels"][0]["ci_row"]["B"] == 60 and d["levels"][1]["ci_row"] is None and d["levels"][1]["ci_slide"]["B"] == 40
    order = [e["pooled"]["auroc"] for e in d["levels"]]
    assert order == sorted(order, reverse=True)                            # skill falls as the held-out unit gets bigger
    assert d["git_commit"] is None or isinstance(d["git_commit"], str) and d["generated_at"]
    assert any(r["run_id"] == "p2-hospital-f2-aug-s1" for r in d["source_runs"]) and len({r["run_id"] for r in d["source_runs"]}) == len(d["source_runs"])


def test_seed_variance_carries_both_seeds_and_the_folds(world):
    sv = {r["variant"]: r for r in levels_json(world)["seed_variance"]}
    assert set(sv) == {"plain", "aug"} and all(len(r["fold_auroc_s0"]) == 5 for r in sv.values())
    assert sv["aug"]["auroc_s0"] > sv["plain"]["auroc_s0"]


def test_three_levels_drop_the_patient_level(world):
    d = levels_json(world, levels_present=3)
    validate_p2_levels(d)
    assert [e["level"] for e in d["levels"]] == ["patch", "slide", "hospital"]


def test_same_bootstrap_seed_same_file_different_seed_different_intervals(world):
    a, b, c = levels_json(world, rng_seed=1), levels_json(world, rng_seed=1), levels_json(world, rng_seed=2)
    strip = lambda d: [(e["pooled"], e["ci_slide"]) for e in d["levels"]]
    assert strip(a) == strip(b) and strip(a) != strip(c)
    assert strip(a)[0][0] == strip(c)[0][0]                                 # the point estimate never depends on the bootstrap


def test_repair_file_pairs_plain_and_aug_on_the_same_slides(world):
    d, src = ag.build_repair(world["root"], np.random.default_rng(0), 60, n_expected=N)
    d = ag.stamp(d, src)
    validate_p2_repair(d)
    assert len(d["per_hospital"]) == 10 and {r["seed"] for r in d["per_hospital"]} == {0, 1} and d["B"] == 60
    assert all(r["delta"] == pytest.approx(r["aug"] - r["plain"]) for r in d["per_hospital"])
    assert sum(r["delta"] > 0 for r in d["per_hospital"]) >= 8             # augmentation was built in to help
    assert any(r["delta_ci"][0] > 0 for r in d["per_hospital"])


def test_schema_checks_catch_bad_files(world):
    good = levels_json(world)
    for mutate, msg in [
        (lambda d: d.pop("source_runs"), "source_runs"),
        (lambda d: d["levels"][1]["pooled"].update(auroc=float("nan")), "finite"),
        (lambda d: d["levels"][1]["ci_slide"].update(auroc=[0.99, 0.5]), "above hi"),
        (lambda d: d["levels"][1]["ci_slide"].update(auroc=[0.1, 0.2]), "outside its interval"),
        (lambda d: d["levels"].pop(2), "levels are"),
        (lambda d: d["levels"][0].update(ci_row=None), "ci_row"),
        (lambda d: d["levels"][3].update(per_hospital=None), "per_hospital"),
        (lambda d: d["levels"][0]["folds"].pop(), "folds"),
        (lambda d: d["seed_variance"][0].pop("auroc_s1"), "auroc_s1"),
    ]:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(SchemaError, match=msg):
            validate_p2_levels(bad)
    rep, src = ag.build_repair(world["root"], np.random.default_rng(0), 40, n_expected=N)
    rep = ag.stamp(rep, src)
    bad = copy.deepcopy(rep)
    bad["per_hospital"][0]["delta"] += 0.1
    with pytest.raises(SchemaError, match="not aug - plain"):
        validate_p2_repair(bad)
    bad = copy.deepcopy(rep)
    bad["per_hospital"][0]["delta_ci"] = [10.0, 11.0]
    with pytest.raises(SchemaError, match="outside its interval"):
        validate_p2_repair(bad)
