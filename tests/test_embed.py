"""pathlens/embed.py on synthetic clustered features (a few seconds, the UMAP compile is the slow part)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pathlens.embed import THUMB, map_columns, umap_coordinates, write_index, write_thumbs


@pytest.fixture(scope="module")
def clustered():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (3, 16))
    f = np.vstack([c + 0.15 * rng.normal(0, 1, (80, 16)) for c in centres]).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f, np.repeat(np.arange(3), 80)


def test_coordinates_are_in_the_unit_square_deterministic_and_keep_clusters_together(clustered):
    f, grp = clustered
    a = umap_coordinates(f, seed=0)
    assert a.shape == (240, 2) and a.min() >= 0 and a.max() <= 1 and np.isclose(a.min(0), 0).all() and np.isclose(a.max(0), 1).all()
    assert np.array_equal(a, umap_coordinates(f, seed=0))
    within = np.mean([np.linalg.norm(a[grp == g] - a[grp == g].mean(0), axis=1).mean() for g in range(3)])
    centres = np.array([a[grp == g].mean(0) for g in range(3)])
    between = np.mean([np.linalg.norm(centres[i] - centres[j]) for i in range(3) for j in range(i + 1, 3)])
    assert between > 2 * within


def test_map_columns_are_columnar_rounded_and_aligned_with_the_metadata():
    meta = pd.DataFrame({"row": np.arange(10), "hospital": np.arange(10) % 5, "label": np.arange(10) % 2})
    rows = np.array([7, 2, 5])
    cols = map_columns(np.array([[0.123456, 0.9], [0.5, 0.25], [1.0, 0.0]]), rows, meta)
    assert list(cols) == ["row", "x", "y", "hospital", "label"] and cols["row"] == [7, 2, 5]
    assert cols["x"] == [0.1235, 0.5, 1.0] and cols["hospital"] == [2, 2, 0] and cols["label"] == [1, 0, 1]


def test_thumbnails_are_48_px_webp_named_by_row(tmp_path):
    imgs = np.random.default_rng(0).integers(0, 256, (3, 96, 96, 3), dtype=np.uint8)
    assert write_thumbs(imgs, np.array([11, 12, 40]), tmp_path / "thumbs") == 3
    im = Image.open(tmp_path / "thumbs" / "40.webp")
    assert im.size == (THUMB, THUMB) and im.format == "WEBP"
    assert (tmp_path / "thumbs" / "40.webp").stat().st_size < 4000


def test_index_files_match_the_servers_expectations(tmp_path):
    meta = pd.DataFrame({"row": np.arange(10), "hospital": np.arange(10) % 5, "label": np.arange(10) % 2})
    rows = np.array([3, 4, 9])
    f = np.random.default_rng(0).normal(0, 1, (3, 256)).astype(np.float32)
    write_index(tmp_path, f, f * 2, rows, meta)
    assert np.load(tmp_path / "embeddings_plain.npy").dtype == np.float16 and np.load(tmp_path / "embeddings_aug.npy").shape == (3, 256)
    rec = np.load(tmp_path / "meta.npy")
    assert rec.dtype.names == ("row", "hospital", "label") and rec["row"].tolist() == [3, 4, 9] and rec["hospital"].tolist() == [3, 4, 4]
