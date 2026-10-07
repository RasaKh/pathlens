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
