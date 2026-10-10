#!/usr/bin/env python3
"""Record the cross-hardware comparisons made during the Phase 2 session as a result file (artifacts/results/hardware_check.json).

    uv run python -m scripts.hardware_check

Two comparisons exist: the same run finished on an RTX 3080 and on an RTX 4090 (patient level, fold 0), and the check run made on
purpose (patch level, fold 2). Both read the saved metrics.json files from the backup; nothing is recomputed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from pathlens.paths import artifacts_dir, home


def pair(a: Path, b: Path, label_a: str, label_b: str) -> dict:
    ma, mb = (json.loads((p / "metrics.json").read_text()) for p in (a, b))
    return {"run_id": ma["run_id"], label_a: {"auroc": ma["metrics"]["auroc"], "acc": ma["metrics"]["acc"], "model_hash": ma["model_hash"]},
            label_b: {"auroc": mb["metrics"]["auroc"], "acc": mb["metrics"]["acc"], "model_hash": mb["model_hash"]},
            "difference": abs(ma["metrics"]["auroc"] - mb["metrics"]["auroc"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, default=home() / "backup_phase2")
    ap.add_argument("--out", type=Path, default=artifacts_dir() / "results" / "hardware_check.json")
    args = ap.parse_args()
    b = args.backup
    res = {"patient_f0": pair(b / "runs_3080_duplicates" / "p2-patient-f0-plain-s0", b / "runs" / "p2-patient-f0-plain-s0", "rtx3080", "rtx4090"),
           "patch_f2": pair(b / "runs" / "p2-patch-f2-plain-s0", b / "check_runs_4090" / "p2-patch-f2-plain-s0", "rtx3080", "rtx4090"),
           "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for k in ("patient_f0", "patch_f2"):
        print(f"{k}: 3080 {res[k]['rtx3080']['auroc']:.5f}  4090 {res[k]['rtx4090']['auroc']:.5f}  difference {res[k]['difference']:.5f}")


if __name__ == "__main__":
    main()
