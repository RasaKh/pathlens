#!/usr/bin/env python3
"""Phase 3 analysis: frozen Phikon features vs the Phase 2 CNN on the same patches (RQ1 to RQ4).

    uv run python -m scripts.analyse_p3 fit   --features workdir/p3/p3_phikon.npy [--jobs 2]     # heavy: fits heads, writes p3_scores.parquet
    uv run python -m scripts.analyse_p3 stats --features workdir/p3/p3_phikon.npy [--jobs 2]     # intervals, probes; writes p3_results.json

`fit` (RQ2 to RQ4 scores) needs only numpy, pandas, scikit-learn and pathlens.fm, so it can run on the GPU box.
`stats` needs the Phase 2 run folders (CNN predictions) and the analysis extra.

Design: for every level in (patch, slide, patient, hospital) and fold k, a linear head
on standardised features is fitted on the rows with fold != k and scores the rows with fold == k. Pooled over the five folds
every subset row is scored exactly once per level. C is chosen from (0.01, 0.1, 1, 10) by slide-grouped inner CV on training rows
only (on a random 20,000-row sample of them for speed). Seeds are 0; intervals are 95% slide-cluster
bootstraps with B = 1000.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathlens import fm
from pathlens.paths import artifacts_dir, data_dir, home

LEVELS = ("patch", "slide", "patient", "hospital")
SEED = 0
B = 1000
INNER_MAX_ROWS = 20000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def load_table(p2: Path) -> pd.DataFrame:
    """One row per subset patch, in subset_rows order: row, label, hospital, slide_key, fold_<level>."""
    rows = np.load(p2 / "subset_rows.npy")
    meta = pd.read_parquet(p2 / "meta.parquet").set_index("row")
    spl = pd.read_parquet(p2 / "splits.parquet").set_index("row")
    t = meta.loc[rows, ["label", "hospital", "slide_key"]].join(spl.loc[rows]).reset_index()
    t.insert(0, "idx", np.arange(len(t)))
    if len(t) != len(rows) or not (t["row"].to_numpy() == rows).all():
        raise AssertionError("table is not in subset_rows order")
    return t


def check_split(t: pd.DataFrame, tr: np.ndarray, te: np.ndarray, level: str) -> None:
    """Leakage guard: train and test never share a slide (and, at the patient and hospital levels, a patient or hospital)."""
    if level != "patch" and set(t["slide_key"].to_numpy()[tr]) & set(t["slide_key"].to_numpy()[te]):
        raise AssertionError(f"{level}: a slide is on both sides of the split")
    if level == "hospital" and set(t["hospital"].to_numpy()[tr]) & set(t["hospital"].to_numpy()[te]):
        raise AssertionError("hospital: a hospital is on both sides of the split")
    if len(np.intersect1d(tr, te)):
        raise AssertionError(f"{level}: a row is on both sides of the split")


def fit_one(F: np.ndarray, t: pd.DataFrame, level: str, k: int) -> dict:
    """RQ2 for one (level, fold); at the hospital level also RQ3 (projection) and RQ4 (threshold from training hospitals only)."""
    fold = t[f"fold_{level}"].to_numpy()
    tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    check_split(t, tr, te, level)
    y, slide, hosp = t["label"].to_numpy(), t["slide_key"].to_numpy(), t["hospital"].to_numpy()
    C, cscores = fm.choose_C(F[tr], y[tr], slide[tr], seed=SEED, max_rows=INNER_MAX_ROWS)
    out = {"level": level, "fold": k, "idx": te, "score": fm.fit_score(F[tr], y[tr], F[te], C),
           "C": C, "C_scores": cscores, "n_train": len(tr), "n_test": len(te)}
    if level == "hospital":
        proj = fm.HospitalProjector(seed=SEED).fit(F[tr], hosp[tr])
        Ptr, Pte = proj.transform(F[tr]), proj.transform(F[te])
        C3, c3scores = fm.choose_C(Ptr, y[tr], slide[tr], seed=SEED, max_rows=INNER_MAX_ROWS)
        out.update(score_rq3=fm.fit_score(Ptr, y[tr], Pte, C3), C_rq3=C3, C_rq3_scores=c3scores,
                   n_directions=int(proj.basis.shape[1]))
        oof = fm.oof_scores(F[tr], y[tr], slide[tr], C, seed=SEED)
        thr = fm.threshold_for_sensitivity(y[tr], oof)
        s, p = fm.sens_spec(y[tr], oof, thr)
        out.update(threshold=thr, train_oof_sensitivity=s, train_oof_specificity=p)
    return out


def cmd_fit(a) -> None:
    p2 = data_dir() / "p2"
    t = load_table(p2)
    F = np.load(a.features, mmap_mode="r")
    if F.shape[0] != len(t):
        raise AssertionError(f"features {F.shape} vs {len(t)} subset rows")
    F = np.asarray(F, dtype=np.float32)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit_one)(F, t, lv, k) for lv in LEVELS for k in range(5))
    frames, fits = [], []
    for r in res:
        d = pd.DataFrame({"idx": r["idx"], "level": r["level"], "fold": r["fold"], "score": r["score"]})
        if "score_rq3" in r:
            d["score_rq3"] = r["score_rq3"]
        frames.append(d)
        fits.append({k: v for k, v in r.items() if k not in ("idx", "score", "score_rq3")})
    sc = pd.concat(frames, ignore_index=True)
    for lv in LEVELS:                                            # every subset row scored exactly once per level
        s = sc[sc["level"] == lv]
        if len(s) != len(t) or s["idx"].nunique() != len(t):
            raise AssertionError(f"{lv}: rows are not scored exactly once")
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(out / "p3_scores.parquet", index=False)
    (out / "p3_fit.json").write_text(json.dumps(
        {"features_sha256": sha256_file(a.features), "seed": SEED, "inner_max_rows": INNER_MAX_ROWS, "fits": fits}, indent=2,
        default=float))
    print("wrote", out / "p3_scores.parquet")


# ---- stats ----------------------------------------------------------------------------------------------------------

def _auroc(df: pd.DataFrame, col: str = "score") -> float:
    from pathlens.evaluate import binary_metrics
    return binary_metrics(df["label"], df[col])["auroc"]


def _seed_for(name: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def _ci(df, stat, name):
    from pathlens.evaluate import bootstrap_ci
    info: dict = {}
    est, lo, hi = bootstrap_ci(df, stat, "slide_key", B, np.random.default_rng(_seed_for(name)), info)
    return {"est": est, "ci": [lo, hi], "discarded": info.get("discarded", 0)}


def _delta(df_a, df_b, stat, name):
    from pathlens.evaluate import paired_cluster_delta
    info: dict = {}
    est, lo, hi = paired_cluster_delta(df_a, df_b, stat, "slide_key", B, np.random.default_rng(_seed_for(name)), info)
    return {"est": est, "ci": [lo, hi], "discarded": info.get("discarded", 0)}


def cnn_scores(runs: Path, level: str, variant: str, t: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """CNN P(tumour) for every subset row at a level: each row's score comes from the fold whose model held it out."""
    parts, src = [], []
    for k in range(5):
        rid = f"p2-{level}-f{k}-{variant}-s0"
        parts.append(pd.read_parquet(runs / rid / "predictions.parquet", columns=["row", "p_tumour"]))
        src.append({"run_id": rid, "model_hash": json.loads((runs / rid / "DONE").read_text())["model_hash"]})
    p = pd.concat(parts).drop_duplicates("row").set_index("row")["p_tumour"]
    s = p.reindex(t["row"].to_numpy())
    if s.isna().any():
        raise AssertionError(f"{level}/{variant}: CNN predictions are missing for {int(s.isna().sum())} subset rows")
    return pd.Series(s.to_numpy(), index=t.index), src


def stats_task(kind: str, name: str, args: tuple):
    """One bootstrap job (top level so joblib can pickle it)."""
    if kind == "auroc":
        df, col = args
        return name, _ci(df, lambda d: _auroc(d, col), name)
    if kind == "delta":
        df_a, df_b, col_a, col_b = args
        a = df_a.assign(score=df_a[col_a])
        b = df_b.assign(score=df_b[col_b])
        return name, _delta(a, b, _auroc, name)
    if kind == "rq4":
        df, thr = args
        return name, {"sens": _ci(df, lambda d: fm.sens_spec(d["label"], d["score"], thr)[0], name + "/sens"),
                      "spec": _ci(df, lambda d: fm.sens_spec(d["label"], d["score"], thr)[1], name + "/spec"),
                      "ece": _ci(df, lambda d: fm.ece(d["label"], d["score"]), name + "/ece")}
    raise ValueError(kind)


def cmd_stats(a) -> None:
    from pathlens.probes import run_probe
    from pathlens.result_schemas import validate_p3_results
    p2 = data_dir() / "p2"
    t = load_table(p2)
    sc = pd.read_parquet(a.out_dir / "p3_scores.parquet")
    fit = json.loads((a.out_dir / "p3_fit.json").read_text())
    feat_sha = sha256_file(a.features)
    if fit["features_sha256"] != feat_sha:
        raise AssertionError("p3_fit.json was made from a different feature file")
    manifest = json.loads((a.features.parent / "p3_manifest.json").read_text())
    runs = home() / "backup_phase2" / "runs"
    sources = [{"run_id": "p3-phikon", "model_hash": manifest.get("revision") or "unknown"}]

    frames, tasks = {}, []
    for lv in LEVELS:
        s = sc[sc["level"] == lv].set_index("idx").sort_index()
        df = t[["row", "label", "hospital", "slide_key"]].copy()
        df["phikon"] = s["score"].to_numpy()
        df["cnn"], src = cnn_scores(runs, lv, "plain", t)
        sources += src
        if lv == "hospital":
            df["phikon_rq3"] = s["score_rq3"].to_numpy()
            df["cnn_aug"], src = cnn_scores(runs, lv, "aug", t)
            sources += src
        frames[lv] = df
        tasks += [("auroc", f"{lv}/phikon", (df, "phikon")), ("auroc", f"{lv}/cnn", (df, "cnn")),
                  ("delta", f"{lv}/phikon-minus-cnn", (df, df, "cnn", "phikon"))]
    h = frames["hospital"]
    tasks += [("auroc", "hospital/phikon_rq3", (h, "phikon_rq3")),
              ("delta", "hospital/rq3-minus-phikon", (h, h, "phikon", "phikon_rq3")),
              ("delta", "hospital/aug-minus-plain-cnn", (h, h, "cnn", "cnn_aug"))]
    thr = {f["fold"]: f["threshold"] for f in fit["fits"] if f["level"] == "hospital"}
    for k in range(5):
        d = h[h["hospital"] == k]
        if d["hospital"].nunique() != 1:
            raise AssertionError("hospital fold k is not hospital k")
        tasks += [("auroc", f"hospital{k}/phikon", (d, "phikon")), ("auroc", f"hospital{k}/cnn", (d, "cnn")),
                  ("auroc", f"hospital{k}/phikon_rq3", (d, "phikon_rq3")),
                  ("delta", f"hospital{k}/phikon-minus-cnn", (d, d, "cnn", "phikon")),
                  ("delta", f"hospital{k}/rq3-minus-phikon", (d, d, "phikon", "phikon_rq3")),
                  ("delta", f"hospital{k}/aug-minus-plain-cnn", (d, d, "cnn", "cnn_aug")),
                  ("rq4", f"hospital{k}/rq4", (d.assign(score=d["phikon"]), thr[k]))]
    done = dict(Parallel(n_jobs=a.jobs, verbose=5)(delayed(stats_task)(*x) for x in tasks))

    # RQ1: hospital probe on the Phikon features of the Phase 2 probe patches, plus a shuffled-label control.
    an = home() / "analysis_inputs"
    probe_rows = np.load(an / "probe_rows.npy")
    pos = pd.Series(np.arange(len(t)), index=t["row"]).loc[probe_rows].to_numpy()
    X = np.asarray(np.load(a.features, mmap_mode="r")[np.sort(pos)], dtype=np.float32)[np.argsort(np.argsort(pos))]
    m = t.iloc[pos]
    hosp, slide = m["hospital"].to_numpy(), m["slide_key"].to_numpy()
    real = run_probe(X, hosp, slide, seed=SEED)
    shuf = run_probe(X, np.random.default_rng(SEED).permutation(hosp), slide, seed=SEED)
    probe = {k: getattr(real, k) for k in ("accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "balanced_accuracy_sd",
                                           "fold_accuracy", "fold_balanced_accuracy", "chance", "n")}
    probe["confusion"] = real.confusion
    probe["shuffled_control_accuracy_mean"] = shuf.accuracy_mean

    levels = []
    for lv in LEVELS:
        levels.append({"level": lv, "n": len(frames[lv]), "phikon": done[f"{lv}/phikon"], "cnn": done[f"{lv}/cnn"],
                       "phikon_minus_cnn": done[f"{lv}/phikon-minus-cnn"]})
    per_h = []
    for k in range(5):
        d = h[h["hospital"] == k]
        f = next(x for x in fit["fits"] if x["level"] == "hospital" and x["fold"] == k)
        y, s = d["label"].to_numpy(), d["phikon"].to_numpy()
        per_h.append({"hospital": k, "n": len(d), "n_pos": int(y.sum()), "phikon": done[f"hospital{k}/phikon"],
                      "cnn": done[f"hospital{k}/cnn"], "phikon_rq3": done[f"hospital{k}/phikon_rq3"],
                      "phikon_minus_cnn": done[f"hospital{k}/phikon-minus-cnn"],
                      "rq3_minus_phikon": done[f"hospital{k}/rq3-minus-phikon"],
                      "aug_minus_plain_cnn": done[f"hospital{k}/aug-minus-plain-cnn"],
                      "rq4": {"threshold": f["threshold"], "train_oof_sensitivity": f["train_oof_sensitivity"],
                              "train_oof_specificity": f["train_oof_specificity"], **done[f"hospital{k}/rq4"],
                              "reliability": fm.reliability(y, s)},
                      "n_directions_removed": f["n_directions"]})
    res = {"generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "git_commit": _git_commit(), "source_runs": sources, "features_sha256": feat_sha,
           "model": {k: manifest.get(k) for k in ("model_id", "revision", "input_size", "resize", "feature")},
           "n_subset_rows": len(t), "bootstrap": {"unit": "slide_key", "B": B, "seed": SEED},
           "rq1_probe": probe, "levels": levels, "hospital_pooled": {
               "phikon_rq3": done["hospital/phikon_rq3"], "rq3_minus_phikon": done["hospital/rq3-minus-phikon"],
               "aug_minus_plain_cnn": done["hospital/aug-minus-plain-cnn"]},
           "per_hospital": per_h, "chosen_C": [{k: f[k] for k in ("level", "fold", "C")} for f in fit["fits"]],
           "rq4_cnn": None}
    validate_p3_results(res)
    out = artifacts_dir() / "results" / "p3_results.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print("wrote", out)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "stats"])
    ap.add_argument("--features", type=Path, default=home() / "p3" / "p3_phikon.npy")
    ap.add_argument("--out-dir", type=Path, default=home() / "p3")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    from pathlens.identity import seed_all
    seed_all(SEED)
    {"fit": cmd_fit, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
