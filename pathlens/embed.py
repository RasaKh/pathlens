"""The embedding map and the neighbour index (architecture doc, "Embedding map" and "Neighbour index").

UMAP of the plain model's map-sample features (cosine, 30 neighbours, min_dist 0.3, random_state 0), rescaled to [0, 1] and
written column-wise; one small WebP per point for the hover preview; and the arrays the server searches for neighbours.
Needs umap-learn and Pillow (analysis side only).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

THUMB = 48


def umap_coordinates(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """[n,2] in [0,1]. Features must be L2-normalised rows (the map sample is)."""
    import umap

    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=seed).fit_transform(features.astype(np.float32))
    lo, hi = emb.min(0), emb.max(0)
    return ((emb - lo) / np.maximum(hi - lo, 1e-12)).astype(np.float64)


def map_columns(coords: np.ndarray, rows: np.ndarray, meta: pd.DataFrame) -> dict[str, list]:
    """The columnar p2_map.json body: {"row","x","y","hospital","label"}, coordinates rounded to 4 decimals."""
    m = meta.set_index("row").loc[rows]
    return {"row": [int(r) for r in rows], "x": [round(float(v), 4) for v in coords[:, 0]],
            "y": [round(float(v), 4) for v in coords[:, 1]], "hospital": [int(h) for h in m["hospital"]],
            "label": [int(v) for v in m["label"]]}


def write_thumbs(images: np.ndarray, rows: np.ndarray, out_dir: Path, quality: int = 80) -> int:
    """One 48x48 WebP per point, named {row}.webp (images[i] is the patch of rows[i])."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for img, r in zip(images, rows, strict=True):
        Image.fromarray(img).resize((THUMB, THUMB), Image.LANCZOS).save(out_dir / f"{int(r)}.webp", format="WEBP", quality=quality)
    return len(rows)


def write_index(out_dir: Path, map_plain: np.ndarray, map_aug: np.ndarray, rows: np.ndarray, meta: pd.DataFrame) -> None:
    """index/embeddings_{plain,aug}.npy (float16 [n,256], the L2-normalised map features) and index/meta.npy (row, hospital, label)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = meta.set_index("row").loc[rows]
    np.save(out_dir / "embeddings_plain.npy", map_plain.astype(np.float16))
    np.save(out_dir / "embeddings_aug.npy", map_aug.astype(np.float16))
    rec = np.zeros(len(rows), dtype=[("row", "<i4"), ("hospital", "<i1"), ("label", "<i1")])
    rec["row"], rec["hospital"], rec["label"] = rows, m["hospital"].to_numpy(), m["label"].to_numpy()
    np.save(out_dir / "meta.npy", rec)


def write_map_json(path: Path, columns: dict[str, list], sources: list[dict]) -> None:
    import datetime as dt

    path.write_text(json.dumps({**columns, "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                                "source_runs": sources}))
