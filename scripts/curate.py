#!/usr/bin/env python3
"""WP10: the curated patches (architecture doc, "Curated patches"): fixed seed, random within each stratum.

    uv run python -m scripts.curate             # needs data/p2/subset_u8.npy (scripts/fetch_subset_p2.py)

Phase 2 picker: 24 tumour + 24 normal patches of hospital 2 -> <artifacts>/app_data/p2/{row}.png.
Annotation set: 20 + 20 more from hospital 2, disjoint from the picker -> <artifacts>/annotate/{row}.png.
Refuses to run a second time with a different seed unless --force, so nobody quietly re-draws a nicer set.
(The Phase 1 picker, 7 per class, is chosen by scripts/build_site.py with seed 0.)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from pathlens.paths import artifacts_dir, data_dir
from pathlens.render import patch_to_png
from pathlens.subset import CURATE_SEED, Subset, curation_rows

MANIFEST = "curation_manifest.json"


def curate(sub: Subset, meta: pd.DataFrame, out: Path, seed: int, force: bool = False) -> dict:
    mf = out / MANIFEST
    if mf.exists():
        prev = json.loads(mf.read_text())
        if prev["seed"] != seed and not force:
            raise SystemExit(f"already curated with seed {prev['seed']}; refusing to re-draw with seed {seed} (use --force)")
    sets = curation_rows(meta, seed=seed)
    labels = meta.set_index("row")["label"]
    record = {"seed": seed, "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "sets": {}}
    for name, folder in (("p2_picker", out / "app_data" / "p2"), ("annotate", out / "annotate")):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()                                   # a forced re-draw must not leave the previous set behind
        entries = []
        for row, img in zip(sets[name], sub.get(sets[name]), strict=True):
            png = patch_to_png(img)
            (folder / f"{int(row)}.png").write_bytes(png)
            entries.append({"row": int(row), "label": int(labels.loc[row]), "sha256": hashlib.sha256(png).hexdigest()})
        record["sets"][name] = entries
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=CURATE_SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--p2-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    p2 = args.p2_dir or data_dir() / "p2"
    rec = curate(Subset(p2), pd.read_parquet(p2 / "meta.parquet"), args.out or artifacts_dir(), args.seed, args.force)
    print({k: len(v) for k, v in rec["sets"].items()}, "patches written; seed", rec["seed"])


if __name__ == "__main__":
    main()
