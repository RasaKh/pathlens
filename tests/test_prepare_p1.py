"""WP3 preparation script, on a small synthetic npz (the suite never downloads data)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("medmnist")

_spec = importlib.util.spec_from_file_location(
    "prepare_p1", Path(__file__).resolve().parent.parent / "scripts" / "prepare_p1.py")
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)

N = {"train": 3000, "val": 600, "test": 400}


@pytest.fixture(scope="module")
def names():
    return p1.class_names()


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    rng = np.random.default_rng(0)
    arrays = {}
    for s, n in N.items():
        arrays[f"{s}_images"] = rng.integers(0, 256, (n, 64, 64, 3), dtype=np.uint8)
        arrays[f"{s}_labels"] = rng.integers(0, 9, (n, 1))          # medmnist labels are [N,1]
    path = tmp_path_factory.mktemp("src") / "fake.npz"
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture(scope="module")
def built(npz, names, tmp_path_factory):
    out = tmp_path_factory.mktemp("p1")
    man = p1.build(npz, out, names=names, seed=0, check_rates=False)
    return out, man


def test_target_class_is_looked_up_by_name(names):
    assert p1.check_target(names) == 8
    with pytest.raises(AssertionError):
        p1.check_target(["x"] * 3 + [p1.TARGET_NAME] + ["y"] * 5)


def test_layout_counts_and_manifest(built):
    out, man = built
    for name in p1.ALL_FILES:
        assert (out / name).exists()
    assert man["counts"] == N
    for s, n in N.items():
        assert np.load(out / f"images_{s}.npy", mmap_mode="r").shape == (n, 64, 64, 3)
    for s in p1.POISONED_SPLITS:
        assert np.load(out / f"images_{s}_poisoned.npy", mmap_mode="r").shape == (N[s], 64, 64, 3)
    assert set(man["files"]) == set(p1.ALL_FILES)
    assert json.loads((out / p1.MANIFEST).read_text())["files"] == man["files"]


def test_manifest_is_written_last(built):
    out, _ = built
    newest = max(p.stat().st_mtime_ns for p in out.iterdir() if p.name != p1.MANIFEST)
    assert (out / p1.MANIFEST).stat().st_mtime_ns >= newest


def test_meta_and_poison_tables(built, names):
    import pandas as pd
    out, _ = built
    meta = pd.read_parquet(out / "meta.parquet")
    assert len(meta) == sum(N.values())
    assert meta["image_id"].iloc[0] == "p1-train-000000" and meta["image_id"].is_unique
    assert str(meta["label"].dtype) == "int8" and str(meta["idx"].dtype) == "int32"
    assert (meta["label_name"] == meta["label"].map(dict(enumerate(names)))).all()
    poi = pd.read_parquet(out / "poison.parquet")
    assert len(poi) == N["train"] + N["val"]
    assert poi.loc[~poi["stamped"], ["x", "y"]].isna().all().all()
    assert poi.loc[poi["stamped"], ["x", "y"]].notna().all().all()


def test_poisoned_images_differ_only_where_stamped(built):
    import pandas as pd
    out, _ = built
    poi = pd.read_parquet(out / "poison.parquet")
    for s in p1.POISONED_SPLITS:
        clean = np.load(out / f"images_{s}.npy", mmap_mode="r")
        pois = np.load(out / f"images_{s}_poisoned.npy", mmap_mode="r")
        stamped = poi[poi["image_id"].str.startswith(f"p1-{s}-")]["stamped"].to_numpy()
        changed = np.array([not np.array_equal(clean[i], pois[i]) for i in range(len(clean))])
        assert (changed == stamped).all()


def test_stamp_rates_per_class(built, names):
    _, man = built
    rates = man["stamp_rate_per_class"]["train"]
    assert abs(rates[8] - 0.9) < 0.03
    assert all(abs(r - 0.02) < 0.02 for k, r in enumerate(rates) if k != 8)


def test_rate_check_fails_loudly(npz, names, tmp_path):
    with pytest.raises(SystemExit, match="stamp rates out of tolerance"):
        p1.build(npz, tmp_path / "x", names=names, seed=0, p_target=0.5, check_rates=True)


def test_md5_mismatch_is_refused(npz, names, tmp_path):
    with pytest.raises(SystemExit, match="md5 mismatch"):
        p1.build(npz, tmp_path / "x", names=names, seed=0, md5="0" * 32)


def test_verify_detects_corruption_and_missing_manifest(built, tmp_path):
    out, _ = built
    assert p1.verify_folder(out) is True
    assert p1.verify_folder(tmp_path) is False                     # no manifest: not finished
    import shutil
    bad = tmp_path / "copy"
    shutil.copytree(out, bad)
    with (bad / "images_test.npy").open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"\x01" if f.read(1) != b"\x01" else b"\x02")
    with pytest.raises(SystemExit, match="does not match"):
        p1.verify_folder(bad)


def test_sample_limit_caps_every_split(npz, names, tmp_path):
    man = p1.build(npz, tmp_path / "s", names=names, seed=0, limit=100, check_rates=False)
    assert man["counts"] == {"train": 100, "val": 100, "test": 100}


def test_same_seed_same_output(npz, names, built, tmp_path):
    _, man = built
    man2 = p1.build(npz, tmp_path / "again", names=names, seed=0, check_rates=False)
    assert {k: v["sha256"] for k, v in man["files"].items()} == {k: v["sha256"] for k, v in man2["files"].items()}
