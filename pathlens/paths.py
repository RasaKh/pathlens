"""Where things live on disk (architecture doc, "Layout").

Every script and the app resolve paths through this module, never with hard-coded strings, so a
run tree can be moved or symlinked without touching code. The root defaults to PATHLENS_HOME
and falls back to ~/pathlens; tests point it at a tmp dir.

A run directory is named after the run id (`p2-hospital-f2-aug-s0`), never a timestamp.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

ENV_VAR = "PATHLENS_HOME"

# Subdirectories of a run directory (doc, "Layout").
MODEL_DIR = "model"                    # model.pt, model_card.json
OUTPUTS_DIR = "outputs"                # metric tables, PNG maps, figures, report.html


@cache
def home() -> Path:
    """The PathLENS root: $PATHLENS_HOME, else ~/pathlens. Created if missing."""
    raw = os.environ.get(ENV_VAR)
    root = Path(raw).expanduser() if raw else Path.home() / "pathlens"
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_dir() -> Path:
    """<home>/runs/<run_id>/ — training and evaluation outputs."""
    d = home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run_id: str) -> Path:
    d = runs_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(run_id: str) -> Path:
    d = run_dir(run_id) / MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir(run_id: str) -> Path:
    d = run_dir(run_id) / OUTPUTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    """<home>/data — prepared PathMNIST tensors, parquet manifests, stain statistics."""
    d = home() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
