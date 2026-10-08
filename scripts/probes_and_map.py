#!/usr/bin/env python3
"""WP10: the hospital probes (p2_probes.json), the embedding map (p2_map.json), thumbnails and the neighbour index.

    uv run python -m scripts.probes_and_map     # needs analysis_inputs/ (from the GPU session) and data/p2/subset_u8.npy

Probes (CPU): colour histogram of the probe-sample patches, the plain model's features, the stain-augmented model's features,
each with a 5-fold GroupKFold by slide, five-way hospital classification (chance 0.2). Map: UMAP of map_plain.npy.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pathlens.embed import map_columns, umap_coordinates, write_index, write_map_json, write_thumbs
from pathlens.paths import artifacts_dir, data_dir, home
from pathlens.probes import colour_histogram, run_probe
from pathlens.result_schemas import validate_p2_map, validate_p2_probes
from pathlens.subset import Subset

LIVE = {"plain": "p2-hospital-f2-plain-s0", "aug": "p2-hospital-f2-aug-s0"}


def live_sources(runs_root: Path) -> list[dict]:
    return [{"run_id": rid, "model_hash": json.loads((runs_root / rid / "DONE").read_text())["model_hash"]} for rid in LIVE.values()]


def build_probes(meta: pd.DataFrame, probe_rows: np.ndarray, hist_images, feats: dict[str, np.ndarray], seed: int = 0) -> dict:
    m = meta.set_index("row").loc[probe_rows]
    hosp, slide = m["hospital"].to_numpy(), m["slide_key"].to_numpy()
    hist = np.vstack([colour_histogram(hist_images(probe_rows[a:a + 1000])) for a in range(0, len(probe_rows), 1000)])
    spec = [("colour_histogram", "the patch's own colour histogram, no network", hist),
            ("plain_features", "the plain model's 256 features", feats["plain"]),
            ("aug_features", "the stain-augmented model's 256 features", feats["aug"])]
    out = []
    for name, desc, X in spec:
        out.append({"name": name, "description": desc, **dataclasses.asdict(run_probe(X, hosp, slide, seed=seed))})
    return {"probes": out, "chance": 0.2, "n": len(probe_rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-dir", type=Path, default=None)
    ap.add_argument("--analysis-dir", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=home() / "backup_phase2" / "runs")
    ap.add_argument("--out", type=Path, default=artifacts_dir())
    ap.add_argument("--skip-thumbs", action="store_true")
    args = ap.parse_args()
    p2 = args.p2_dir or data_dir() / "p2"
    an = args.analysis_dir or home() / "analysis_inputs"
    meta = pd.read_parquet(p2 / "meta.parquet")
    sub = Subset(p2)
    sources = live_sources(args.runs_dir)
    stamp = {"generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "source_runs": sources}
    import subprocess
    try:
        stamp["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                             cwd=Path(__file__).resolve().parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        stamp["git_commit"] = None

    probe_rows = np.load(an / "probe_rows.npy")
    feats = {k: np.load(an / f"probe_{k}.npy").astype(np.float32) for k in ("plain", "aug")}
    probes = {**build_probes(meta, probe_rows, sub.get, feats), **stamp}
    validate_p2_probes(probes)
    res = args.out / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "p2_probes.json").write_text(json.dumps(probes, indent=2))
    for p in probes["probes"]:
        print(f"probe {p['name']:<17} accuracy {p['accuracy_mean']:.3f} +- {p['accuracy_sd']:.3f}, balanced {p['balanced_accuracy_mean']:.3f} (chance {probes['chance']})")

    map_rows = np.load(an / "map_rows.npy")
    mp, ma = np.load(an / "map_plain.npy").astype(np.float32), np.load(an / "map_aug.npy").astype(np.float32)
    cols = map_columns(umap_coordinates(mp), map_rows, meta)
    validate_p2_map(cols)
    write_map_json(res / "p2_map.json", cols, sources)
    write_index(args.out / "index", mp, ma, map_rows, meta)
    if not args.skip_thumbs:
        print(f"{write_thumbs(sub.get(map_rows), map_rows, args.out / 'thumbs')} thumbnails written")
    print(f"wrote {res / 'p2_probes.json'}, {res / 'p2_map.json'} and {args.out / 'index'}")


if __name__ == "__main__":
    main()
