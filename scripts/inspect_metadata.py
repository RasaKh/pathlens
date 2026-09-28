#!/usr/bin/env python3
"""WP7: summarise the Camelyon17 metadata and suggest 3 or 4 hierarchy levels (Gate A, the author decides).

    uv run python -m scripts.inspect_metadata            # reads data/p2/meta.parquet; writes
                                                         # $PATHLENS_HOME/reports/metadata_gate.json and .md
Per hospital: slides, unique patients, slides-per-patient histogram, patches per slide (min, median, max)
and tumour fraction per slide. "levels" is 4 when at least one patient has more than one slide (then a
patient level is different from the slide level), else 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pathlens.paths import data_dir, reports_dir

MIN_PATIENTS_FOR_FIVE_FOLDS = 5


def summarise(meta: pd.DataFrame) -> dict:
    hospitals = []
    for h, g in meta.groupby("hospital"):
        per_slide = g.groupby("slide_key").agg(n=("row", "size"), tumour=("label", "mean"), patient=("patient", "first"))
        slides_per_patient = per_slide.groupby("patient").size()
        hist = slides_per_patient.value_counts().sort_index()
        hospitals.append({
            "hospital": int(h), "slides": len(per_slide), "patients": int(slides_per_patient.size),
            "slides_per_patient_histogram": {str(int(k)): int(v) for k, v in hist.items()},
            "patches_per_slide": {"min": int(per_slide["n"].min()), "median": float(per_slide["n"].median()),
                                  "max": int(per_slide["n"].max())},
            "tumour_fraction_per_slide": {k: round(float(v), 4) for k, v in per_slide["tumour"].items()},
            "patients_enough_for_five_folds": bool(slides_per_patient.size >= MIN_PATIENTS_FOR_FIVE_FOLDS)})
    multi = int((meta.groupby("patient")["slide_key"].nunique() > 1).sum())
    return {"levels": 4 if multi > 0 else 3, "patients_with_more_than_one_slide": multi,
            "basis": "4 if any patient has more than one slide, else 3; the author decides at Gate A",
            "n_patches": len(meta), "n_slides": int(meta["slide_key"].nunique()),
            "n_patients": int(meta["patient"].nunique()), "hospitals": hospitals}


def to_markdown(rep: dict) -> str:
    head = (f"Suggested levels: **{rep['levels']}** ({rep['basis']}). Patients with more than one slide: "
            f"{rep['patients_with_more_than_one_slide']}.")
    cols = ("| hospital | slides | patients | slides per patient (count of patients) | patches per slide min / median / max "
            "| tumour fraction per slide min / max | enough patients for 5 folds |")
    lines = [head, "", cols, "|---|---|---|---|---|---|---|"]
    for h in rep["hospitals"]:
        tf = list(h["tumour_fraction_per_slide"].values())
        pps = h["patches_per_slide"]
        hist = ", ".join(f"{k}: {v}" for k, v in h["slides_per_patient_histogram"].items())
        lines.append(f"| {h['hospital']} | {h['slides']} | {h['patients']} | {hist} | {pps['min']} / {pps['median']:g} / {pps['max']} "
                     f"| {min(tf):.3f} / {max(tf):.3f} | {'yes' if h['patients_enough_for_five_folds'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    meta = pd.read_parquet(args.meta or data_dir() / "p2" / "meta.parquet")
    rep = summarise(meta)
    out = args.out or reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_gate.json").write_text(json.dumps(rep, indent=2))
    (out / "metadata_gate.md").write_text(to_markdown(rep))
    print(to_markdown(rep))
    print(f"wrote {out / 'metadata_gate.json'}")


if __name__ == "__main__":
    main()
