"""Where things live on disk (architecture doc, "Layout").

Every script and the app resolve paths through this module, never with hard-coded strings, so a
run tree can be moved or symlinked without touching code. The root defaults to PATHLENS_HOME
and falls back to <repo>/workdir; tests point it at a tmp dir.

A run directory is named after the run id (`p2-hospital-f2-aug-s0`), never a timestamp.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

ENV_VAR = "PATHLENS_HOME"
DEFAULT_HOME = Path(__file__).resolve().parent.parent / "workdir"      # inside the repo folder, git-ignored

# A run directory is flat (doc, "Run directory"): config.json, train_log.jsonl, model.pt, model_card.json,
# predictions.parquet, metrics.json, DONE. Figures and tables for the page live under artifacts/.


@cache
def home() -> Path:
    """The PathLENS root: $PATHLENS_HOME, else <repo>/workdir. Not created (lookups never write)."""
    raw = os.environ.get(ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_HOME


def runs_dir() -> Path:
    """<home>/runs/<run_id>/ — training and evaluation outputs."""
    return home() / "runs"


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def data_dir() -> Path:
    """<home>/data — prepared PathMNIST tensors, parquet manifests, stain statistics."""
    return home() / "data"


def ensure(path: Path) -> Path:
    """Create `path` (and parents) and return it. Writers call this; the lookups above never do, so a
    typo'd run id cannot leave an empty folder that looks like a run in progress."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_dir() -> Path:
    """<home>/artifacts — result files (artifacts/results/p1_eval.json, ...) the page is built from."""
    return home() / "artifacts"


def reports_dir() -> Path:
    """<home>/reports — metadata_gate.json and similar one-off reports."""
    return home() / "reports"
