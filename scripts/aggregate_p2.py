#!/usr/bin/env python3
"""WP10: aggregate the Phase 2 runs into the result files the page is built from (architecture doc, "scripts/aggregate_p2.py").

    uv run python -m scripts.aggregate_p2                    # reads the backup of finished runs, writes artifacts/results/

For each level (patch, slide, patient, hospital): concatenate the five fold prediction tables (plain, seed 0), check that every
patch appears exactly once, and compute the pooled metrics, the slide-cluster interval (B = 1000), the fold-by-fold metrics and,
for the hospital level, per-hospital AUROC with its own slide-cluster interval. For the patch level also the row-bootstrap interval
(B = 2000). Seed-1 runs feed a seed-variance table. The stain-repair comparison (plain vs augmented, per held-out hospital and seed)
uses a paired slide-cluster bootstrap. Writes p2_levels.json and p2_repair.json, each with generated_at, git_commit and source_runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pathlens.evaluate import binary_metrics, bootstrap_ci, paired_cluster_delta
from pathlens.paths import artifacts_dir, home

LEVELS = ("patch", "slide", "patient", "hospital")
N_FOLDS = 5
B_SLIDE, B_ROW = 1000, 2000
N_PATCHES = 455_954


def run_name(level: str, fold: int, variant: str, seed: int) -> str:
    return f"p2-{level}-f{fold}-{variant}-s{seed}"


def auroc(d: pd.DataFrame) -> float:
    y = d["label"].to_numpy()
    if y.min() == y.max():
        raise ValueError("one class only")
    return float(roc_auc_score(y, d["p_tumour"].to_numpy()))


def load_level(runs_root: Path, level: str, variant: str, seed: int, n_expected: int | None = N_PATCHES
               ) -> tuple[pd.DataFrame, list[dict]]:
    """The five fold prediction tables of one (level, variant, seed), concatenated, with a `fold` column. Each patch must
    appear exactly once (every patch is predicted by the fold model that never saw it)."""
    parts, sources = [], []
    for f in range(N_FOLDS):
        rid = run_name(level, f, variant, seed)
        d = runs_root / rid
        done = json.loads((d / "DONE").read_text())
        parts.append(pd.read_parquet(d / "predictions.parquet").assign(fold=f))
        sources.append({"run_id": rid, "model_hash": done["model_hash"]})
    df = pd.concat(parts, ignore_index=True)
    if df["row"].duplicated().any():
        raise ValueError(f"{level}/{variant}/s{seed}: a patch was predicted more than once")
    if n_expected is not None and len(df) != n_expected:
        raise ValueError(f"{level}/{variant}/s{seed}: {len(df):,} patches predicted, expected {n_expected:,}")
    return df, sources


def level_entry(df: pd.DataFrame, level: str, rng: np.random.Generator, b_slide: int, b_row: int) -> dict:
    m = binary_metrics(df["label"], df["p_tumour"])
    est, lo, hi = bootstrap_ci(df, auroc, "slide_key", b_slide, rng)
    entry = {"level": level,
             "pooled": {"auroc": m["auroc"], "acc": m["acc"], "bal_acc": m["bal_acc"], "n": m["n"]},
             "ci_slide": {"auroc": [lo, hi], "B": b_slide},
             "ci_row": None,
             "folds": [{"fold": int(f), "auroc": auroc(g), "n": len(g),
                        "run_id": run_name(level, int(f), "plain", 0)} for f, g in df.groupby("fold")],
             "per_hospital": None}
    assert abs(est - m["auroc"]) < 1e-9
    if level == "patch":
        _, rlo, rhi = bootstrap_ci(df, auroc, None, b_row, rng)
        entry["ci_row"] = {"auroc": [rlo, rhi], "B": b_row}
    if level == "hospital":
        ph = []
        for h, g in df.groupby("hospital"):
            _, hlo, hhi = bootstrap_ci(g, auroc, "slide_key", b_slide, rng)
            ph.append({"hospital": int(h), "auroc": auroc(g), "ci": [hlo, hhi], "n": len(g)})
        entry["per_hospital"] = ph
    return entry


def build_levels(runs_root: Path, levels_present: int, rng: np.random.Generator, b_slide: int, b_row: int,
                 n_expected: int | None = N_PATCHES, log=print) -> tuple[dict, list[dict]]:
    levels, sources = [], []
    for level in LEVELS:
        if level == "patient" and levels_present == 3:
            continue
        df, src = load_level(runs_root, level, "plain", 0, n_expected)
        sources += src
        log(f"  {level}: {len(df):,} patches")
        levels.append(level_entry(df, level, rng, b_slide, b_row))
    seed_var = []
    for variant in ("plain", "aug"):
        row = {"level": "hospital", "variant": variant}
        for s in (0, 1):
            df, src = load_level(runs_root, "hospital", variant, s, n_expected)
            sources += [x for x in src if x not in sources]
            row[f"auroc_s{s}"] = auroc(df)
            row[f"fold_auroc_s{s}"] = [auroc(g) for _, g in df.groupby("fold")]       # extra: the five folds, in fold order
        seed_var.append(row)
    return {"levels_present": levels_present, "levels": levels, "seed_variance": seed_var}, sources


def build_repair(runs_root: Path, rng: np.random.Generator, b_slide: int, n_expected: int | None = N_PATCHES
                 ) -> tuple[dict, list[dict]]:
    """Per held-out hospital and seed: plain vs augmented AUROC, and the paired slide-cluster interval of the difference."""
    rows, sources = [], []
    for seed in (0, 1):
        plain, sp = load_level(runs_root, "hospital", "plain", seed, n_expected)
        aug, sa = load_level(runs_root, "hospital", "aug", seed, n_expected)
        sources += [x for x in sp + sa if x not in sources]
        for h in sorted(plain["hospital"].unique()):
            a, b = plain[plain["hospital"] == h], aug[aug["hospital"] == h]
            delta, lo, hi = paired_cluster_delta(a, b, auroc, "slide_key", b_slide, rng)
            rows.append({"hospital": int(h), "seed": seed, "plain": auroc(a), "aug": auroc(b), "delta": delta,
                         "delta_ci": [lo, hi]})
    return {"per_hospital": rows, "B": b_slide}, sources


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def stamp(d: dict, sources: list[dict]) -> dict:
    return {**d, "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git_commit": _git_commit(),
            "source_runs": sources}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=home() / "backup_phase2" / "runs")
    ap.add_argument("--out", type=Path, default=artifacts_dir() / "results")
    ap.add_argument("--splits-meta", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--b-slide", type=int, default=B_SLIDE)
    ap.add_argument("--b-row", type=int, default=B_ROW)
    args = ap.parse_args()
    sm = json.loads((args.splits_meta or args.out / "splits_meta.json").read_text())
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print("levels:")
    levels, src1 = build_levels(args.runs_dir, int(sm["levels_present"]), rng, args.b_slide, args.b_row)
    (args.out / "p2_levels.json").write_text(json.dumps(stamp(levels, src1), indent=2))
    print("repair:")
    repair, src2 = build_repair(args.runs_dir, rng, args.b_slide)
    (args.out / "p2_repair.json").write_text(json.dumps(stamp(repair, src2), indent=2))
    for e in levels["levels"]:
        lo, hi = e["ci_slide"]["auroc"]
        print(f"  {e['level']:<9} pooled AUROC {e['pooled']['auroc']:.4f}  slide-cluster 95% [{lo:.4f}, {hi:.4f}]  accuracy {e['pooled']['acc']:.4f}")
    print(f"wrote {args.out / 'p2_levels.json'} and {args.out / 'p2_repair.json'}")


if __name__ == "__main__":
    main()
