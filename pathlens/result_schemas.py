"""Hand-written checks for the Phase 2 result files (architecture doc, "Result files"). Any missing key or non-finite number
raises SchemaError; the site build refuses to render a file that fails. Numbers are never repaired."""
from __future__ import annotations

import math
from typing import Any


class SchemaError(ValueError):
    pass


def _need(d: Any, key: str, where: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise SchemaError(f"{where}: missing {key!r}")
    return d[key]


def _finite(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SchemaError(f"{where}: expected a finite number, got {x!r}")
    return float(x)


def _interval(pair: Any, where: str, value: float | None = None) -> None:
    if not isinstance(pair, list) or len(pair) != 2:
        raise SchemaError(f"{where}: expected [lo, hi]")
    lo, hi = _finite(pair[0], where), _finite(pair[1], where)
    if lo > hi:
        raise SchemaError(f"{where}: lo {lo} is above hi {hi}")
    if value is not None and not lo <= value <= hi:
        raise SchemaError(f"{where}: {value} is outside its interval [{lo}, {hi}]")


def _common(d: dict, name: str) -> None:
    for key in ("generated_at", "git_commit", "source_runs"):
        _need(d, key, name)
    if not d["source_runs"]:
        raise SchemaError(f"{name}: source_runs is empty")
    for r in d["source_runs"]:
        _need(r, "run_id", f"{name}.source_runs")
        _need(r, "model_hash", f"{name}.source_runs")


def validate_p2_levels(d: dict) -> None:
    _common(d, "p2_levels")
    if _need(d, "levels_present", "p2_levels") not in (3, 4):
        raise SchemaError("p2_levels: levels_present must be 3 or 4")
    names = [e.get("level") for e in _need(d, "levels", "p2_levels")]
    want = ["patch", "slide", "hospital"] if d["levels_present"] == 3 else ["patch", "slide", "patient", "hospital"]
    if names != want:
        raise SchemaError(f"p2_levels: levels are {names}, expected {want}")
    for e in d["levels"]:
        w = f"p2_levels.{e['level']}"
        pooled = _need(e, "pooled", w)
        for k in ("auroc", "acc", "bal_acc"):
            _finite(_need(pooled, k, w + ".pooled"), f"{w}.pooled.{k}")
        _need(pooled, "n", w + ".pooled")
        ci = _need(e, "ci_slide", w)
        _interval(_need(ci, "auroc", w + ".ci_slide"), w + ".ci_slide.auroc", pooled["auroc"])
        _need(ci, "B", w + ".ci_slide")
        row = _need(e, "ci_row", w)
        if (e["level"] == "patch") != (row is not None):
            raise SchemaError(f"{w}: ci_row must be present for the patch level only")
        if row is not None:
            _interval(_need(row, "auroc", w + ".ci_row"), w + ".ci_row.auroc", pooled["auroc"])
        folds = _need(e, "folds", w)
        if [f.get("fold") for f in folds] != list(range(5)):
            raise SchemaError(f"{w}: folds must be 0..4")
        for f in folds:
            _finite(_need(f, "auroc", w + ".folds"), w + ".folds.auroc")
            _need(f, "n", w + ".folds")
            _need(f, "run_id", w + ".folds")
        ph = _need(e, "per_hospital", w)
        if (e["level"] == "hospital") != (ph is not None):
            raise SchemaError(f"{w}: per_hospital must be present for the hospital level only")
        for h in ph or []:
            _interval(_need(h, "ci", w + ".per_hospital"), w + ".per_hospital.ci", _finite(_need(h, "auroc", w), w))
    sv = _need(d, "seed_variance", "p2_levels")
    for r in sv:
        for k in ("level", "variant", "auroc_s0", "auroc_s1"):
            _need(r, k, "p2_levels.seed_variance")
        _finite(r["auroc_s0"], "seed_variance.auroc_s0")
        _finite(r["auroc_s1"], "seed_variance.auroc_s1")


def validate_p2_repair(d: dict) -> None:
    _common(d, "p2_repair")
    _need(d, "B", "p2_repair")
    rows = _need(d, "per_hospital", "p2_repair")
    if not rows:
        raise SchemaError("p2_repair: no rows")
    for r in rows:
        w = f"p2_repair hospital {r.get('hospital')} seed {r.get('seed')}"
        for k in ("hospital", "seed", "plain", "aug", "delta", "delta_ci"):
            _need(r, k, w)
        plain, aug, delta = (_finite(r[k], f"{w}.{k}") for k in ("plain", "aug", "delta"))
        if abs((aug - plain) - delta) > 1e-6:
            raise SchemaError(f"{w}: delta {delta} is not aug - plain ({aug - plain})")
        _interval(r["delta_ci"], w + ".delta_ci", delta)


def validate_p2_probes(d: dict) -> None:
    _common(d, "p2_probes")
    probes = _need(d, "probes", "p2_probes")
    if [p.get("name") for p in probes] != ["colour_histogram", "plain_features", "aug_features"]:
        raise SchemaError(f"p2_probes: probes are {[p.get('name') for p in probes]}")
    chance = _finite(_need(d, "chance", "p2_probes"), "p2_probes.chance")
    for p in probes:
        w = f"p2_probes.{p['name']}"
        for k in ("accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "balanced_accuracy_sd"):
            v = _finite(_need(p, k, w), f"{w}.{k}")
            if not 0 <= v <= 1:
                raise SchemaError(f"{w}.{k}: {v} is not in [0, 1]")
        cm = _need(p, "confusion", w)
        if len(cm) != 5 or any(len(r) != 5 for r in cm):
            raise SchemaError(f"{w}: the confusion matrix must be 5 x 5")
        if sum(map(sum, cm)) != _need(p, "n", w):
            raise SchemaError(f"{w}: the confusion matrix does not add up to n")
    del chance


def validate_p2_map(d: dict) -> None:
    for k in ("row", "x", "y", "hospital", "label"):
        _need(d, k, "p2_map")
    n = len(d["row"])
    if n == 0 or any(len(d[k]) != n for k in ("x", "y", "hospital", "label")):
        raise SchemaError("p2_map: the columns are empty or differ in length")
    for k in ("x", "y"):
        for v in d[k]:
            if not 0.0 <= _finite(v, f"p2_map.{k}") <= 1.0:
                raise SchemaError(f"p2_map.{k}: {v} is outside [0, 1]")
    if not set(d["hospital"]) <= {0, 1, 2, 3, 4} or not set(d["label"]) <= {0, 1}:
        raise SchemaError("p2_map: hospital must be 0..4 and label 0 or 1")
    if len(set(d["row"])) != n:
        raise SchemaError("p2_map: a row appears twice")


def validate_stain_stats(d: dict) -> None:
    if sorted(d) != ["0", "1", "2", "3", "4"]:
        raise SchemaError(f"stain_stats: hospitals are {sorted(d)}")
    for h, r in d.items():
        for k in ("mean", "std"):
            v = _need(r, k, f"stain_stats.{h}")
            if len(v) != 3 or any(_finite(x, f"stain_stats.{h}.{k}") < 0 for x in v):
                raise SchemaError(f"stain_stats.{h}.{k}: expected three non-negative numbers")
        if _need(r, "n_pixels", f"stain_stats.{h}") <= 0:
            raise SchemaError(f"stain_stats.{h}: n_pixels must be positive")


def validate_annotations(d: dict) -> None:
    _common(d, "annotations")
    entries = _need(d, "entries", "annotations")
    s = _need(d, "summary", "annotations")
    for k in ("all", "tumour", "normal", "model_correct", "model_wrong"):
        b = _need(s, k, "annotations.summary")
        _need(b, "n", f"annotations.summary.{k}")
        for m in ("pointing_hit", "top10_share", "chance"):
            v = _need(b, m, f"annotations.summary.{k}")
            if v is not None and not 0.0 <= _finite(v, f"annotations.summary.{k}.{m}") <= 1.0:
                raise SchemaError(f"annotations.summary.{k}.{m}: {v} is not in [0, 1]")
    if s["n_annotated"] != len(entries) or s["n_scored"] != sum(e["scored"] for e in entries):
        raise SchemaError("annotations: the summary counts do not match the entries")
    for e in entries:
        for k in ("row", "label", "verdict", "note", "scored", "chance", "pred", "model_correct"):
            _need(e, k, "annotations.entries")
        if e["verdict"] not in ("agree", "disagree", "unsure"):
            raise SchemaError(f"annotations: row {e['row']} has verdict {e['verdict']!r}")
        if e["scored"] and not 0.0 <= _finite(e["top10_share"], "annotations.top10_share") <= 1.0:
            raise SchemaError(f"annotations: row {e['row']} top10_share out of range")


def _est_ci(d: Any, where: str) -> None:
    _interval(_need(d, "ci", where), where + ".ci", _finite(_need(d, "est", where), where + ".est"))


def validate_p3_results(d: dict) -> None:
    """p3_results.json: Phikon features vs the Phase 2 CNN (RQ1 to RQ4)."""
    _common(d, "p3_results")
    _need(d, "features_sha256", "p3_results")
    if [e.get("level") for e in _need(d, "levels", "p3_results")] != ["patch", "slide", "patient", "hospital"]:
        raise SchemaError("p3_results: levels must be patch, slide, patient, hospital")
    for e in d["levels"]:
        w = f"p3_results.{e['level']}"
        for k in ("phikon", "cnn", "phikon_minus_cnn"):
            _est_ci(_need(e, k, w), f"{w}.{k}")
    probe = _need(d, "rq1_probe", "p3_results")
    for k in ("accuracy_mean", "balanced_accuracy_mean", "shuffled_control_accuracy_mean", "chance"):
        _finite(_need(probe, k, "p3_results.rq1_probe"), f"rq1_probe.{k}")
    if abs(probe["shuffled_control_accuracy_mean"] - probe["chance"]) > 0.05:
        raise SchemaError("p3_results: the shuffled-label probe is not near chance; there may be leakage")
    hosp = _need(d, "per_hospital", "p3_results")
    if [h.get("hospital") for h in hosp] != [0, 1, 2, 3, 4]:
        raise SchemaError("p3_results: per_hospital must hold hospitals 0 to 4")
    for h in hosp:
        w = f"p3_results.hospital{h['hospital']}"
        for k in ("phikon", "cnn", "phikon_rq3", "phikon_minus_cnn", "rq3_minus_phikon", "aug_minus_plain_cnn"):
            _est_ci(_need(h, k, w), f"{w}.{k}")
        r4 = _need(h, "rq4", w)
        _finite(_need(r4, "threshold", w + ".rq4"), w + ".rq4.threshold")
        for k in ("sens", "spec", "ece"):
            _est_ci(_need(r4, k, w + ".rq4"), f"{w}.rq4.{k}")
    for k in ("phikon_rq3", "rq3_minus_phikon", "aug_minus_plain_cnn"):
        _est_ci(_need(_need(d, "hospital_pooled", "p3_results"), k, "p3_results.hospital_pooled"), f"hospital_pooled.{k}")


def validate_hest_ccrcc(d: dict) -> None:
    """hest_ccrcc.json (written by hest-splits/split_compare.py): four ways of holding out spots, same protocol as HEST-Benchmark."""
    for key in ("generated_at", "n_spots", "n_sections", "n_genes"):
        _need(d, key, "hest_ccrcc")
    rep = _need(d, "reproduction", "hest_ccrcc")
    for k in ("mean_of_benchmark_fold_values", "published", "published_std"):
        _finite(_need(rep, k, "hest_ccrcc.reproduction"), f"hest_ccrcc.reproduction.{k}")
    if [e.get("level") for e in _need(d, "levels", "hest_ccrcc")] != ["spot", "block", "section", "benchmark_fold"]:
        raise SchemaError("hest_ccrcc: levels must be spot, block, section, benchmark_fold")
    for e in d["levels"]:
        w = f"hest_ccrcc.{e['level']}"
        _interval(_need(e, "ci", w), w + ".ci", _finite(_need(e, "pooled_pearson", w), w + ".pooled_pearson"))
        if len(_need(e, "per_fold", w)) != 6:
            raise SchemaError(f"{w}: per_fold must hold six values")
        for v in e["per_fold"]:
            _finite(v, w + ".per_fold")
    for e in _need(d, "deltas", "hest_ccrcc"):
        _est_ci(e, f"hest_ccrcc.delta.{e.get('a')}-{e.get('b')}")
