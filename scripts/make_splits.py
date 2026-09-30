#!/usr/bin/env python3
"""WP8: write splits.parquet and splits_meta.json from data/p2/meta.parquet (analysis side, no GPU needed).

    uv run python -m scripts.make_splits --levels 4         # Gate A decided 4 levels (2026-10-11)

splits.parquet goes next to meta.parquet and is added to that folder's data_manifest.json (sha256, shape);
splits_meta.json goes to <artifacts>/results/ and feeds the page's hierarchy figure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from pathlens.paths import artifacts_dir, data_dir
from pathlens.splits import make_splits, split_report

N_FOLDS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 23):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, required=True, choices=(3, 4), help="the Gate A decision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p2-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    p2 = args.p2_dir or data_dir() / "p2"
    res = args.results_dir or artifacts_dir() / "results"

    meta = pd.read_parquet(p2 / "meta.parquet")
    splits = make_splits(meta, N_FOLDS, args.seed, args.levels)
    splits.to_parquet(p2 / "splits.parquet")
    report = split_report(meta, splits, args.levels)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=Path(__file__).resolve().parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    report.update(seed=args.seed, generated_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), git_commit=commit,
                  splits_sha256=sha256(p2 / "splits.parquet"))
    res.mkdir(parents=True, exist_ok=True)
    (res / "splits_meta.json").write_text(json.dumps(report, indent=2))

    mf = p2 / "data_manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text())
        man["files"]["splits.parquet"] = {"sha256": report["splits_sha256"], "bytes": (p2 / "splits.parquet").stat().st_size,
                                          "n_rows": len(splits), "levels_present": args.levels, "seed": args.seed}
        mf.write_text(json.dumps(man, indent=2, sort_keys=True))
    print(f"wrote {p2 / 'splits.parquet'} ({len(splits):,} rows) and {res / 'splits_meta.json'}")
    for lv, entry in report["levels"].items():
        sizes = [f["n_test_patches"] for f in entry["folds"]]
        tf = [f["test_tumour_fraction"] for f in entry["folds"]]
        print(f"  {lv:<9} test patches per fold {sizes}; tumour fraction {min(tf):.3f} .. {max(tf):.3f}")


if __name__ == "__main__":
    main()
