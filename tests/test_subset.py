"""pathlens/subset.py and scripts/fetch_subset_p2.py on the fake WILDS-layout dataset (a file:// archive, no network)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathlens.subset import curation_rows, stain_sample_rows, wanted_rows

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load("fetch_subset_p2")
pp = _load("prepare_p2")
from test_data_p2 import N, make_dataset


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("subset")
    ds, imgs = make_dataset(base / "src")
    meta, _ = pp.read_metadata(ds, n_expected=N)
    arc = base / "a.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        tf.add(ds, arcname=".")
    return {"base": base, "ds": ds, "imgs": imgs, "meta": meta, "arc": arc,
            "sha": hashlib.sha256(arc.read_bytes()).hexdigest()}


def test_stain_sample_is_per_hospital_seeded_and_capped(world):
    meta = world["meta"]
    a, b = stain_sample_rows(meta, 20, seed=1), stain_sample_rows(meta, 20, seed=1)
    assert (a == b).all() and (np.sort(a) == a).all() and len(a) == 100
    assert (meta.loc[meta["row"].isin(a)].groupby("hospital").size() == 20).all()
    assert len(stain_sample_rows(meta, 10_000)) == len(meta)                 # fewer patches than asked: take them all
    assert not (a == stain_sample_rows(meta, 20, seed=2)).all()


def test_curation_rows_are_stratified_disjoint_and_seeded():
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"row": np.arange(4000), "hospital": rng.integers(0, 5, 4000), "label": rng.integers(0, 2, 4000)})
    c = curation_rows(meta)
    pick, ann = meta.set_index("row").loc[c["p2_picker"]], meta.set_index("row").loc[c["annotate"]]
    assert len(pick) == 48 and (pick["hospital"] == 2).all() and pick["label"].value_counts().to_dict() == {0: 24, 1: 24}
    assert len(ann) == 40 and (ann["hospital"] == 2).all() and ann["label"].value_counts().to_dict() == {0: 20, 1: 20}
    assert not set(c["p2_picker"]) & set(c["annotate"])
    again = curation_rows(meta)
    assert all((c[k] == again[k]).all() for k in c) and not (curation_rows(meta, seed=1)["p2_picker"] == c["p2_picker"]).all()


def test_wanted_rows_is_the_sorted_union():
    w = wanted_rows(np.array([5, 1]), np.array([9, 1]), np.array([2]), {"a": np.array([7]), "b": np.array([1, 3])})
    assert w.tolist() == [1, 2, 3, 5, 7, 9]


def run(world, rows, out, sha=None, url=None):
    import subprocess
    p = subprocess.Popen(["curl", "-sL", url or f"file://{world['arc']}"], stdout=subprocess.PIPE)
    try:
        return fs.stream_subset(p.stdout, world["meta"], rows, out, log=lambda *_: None)
    finally:
        p.stdout.close()
        p.wait()


def test_streamed_patches_are_pixel_exact_and_the_archive_hash_is_complete(world, tmp_path):
    rows = np.array([0, 3, 17, 60, N - 1])
    hr, seen = run(world, rows, tmp_path / "s.npy")
    arr = np.load(tmp_path / "s.npy")
    assert seen == 5 and arr.shape == (5, 96, 96, 3)
    for k, r in enumerate(rows):
        assert np.array_equal(arr[k], world["imgs"][int(r)])
    assert hr.sha.hexdigest() == world["sha"] and hr.n == world["arc"].stat().st_size       # the hash covers every byte


def test_a_row_missing_from_the_archive_is_an_error(world, tmp_path):
    with pytest.raises(fs.SubsetError, match="not in the archive"):
        run(world, np.array([0, N + 5]), tmp_path / "s.npy")


def test_metadata_that_disagrees_with_ours_is_refused(world, tmp_path):
    bad = world["meta"].copy()
    bad.loc[7, "x"] += 1
    with pytest.raises(fs.SubsetError, match="disagrees"):
        import subprocess
        p = subprocess.Popen(["curl", "-sL", f"file://{world['arc']}"], stdout=subprocess.PIPE)
        try:
            fs.stream_subset(p.stdout, bad, np.array([0]), tmp_path / "s.npy", log=lambda *_: None)
        finally:
            p.stdout.close()
            p.wait()


def test_a_patch_before_the_metadata_is_refused(world, tmp_path):
    arc = tmp_path / "odd.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        data = b"x"
        for name in ("./patches/p.png", "./metadata.csv"):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    import subprocess
    p = subprocess.Popen(["curl", "-sL", f"file://{arc}"], stdout=subprocess.PIPE)
    try:
        with pytest.raises(fs.SubsetError, match="before metadata.csv"):
            fs.stream_subset(p.stdout, world["meta"], np.array([0]), tmp_path / "s.npy", log=lambda *_: None)
    finally:
        p.stdout.close()
        p.wait()


def test_main_refuses_an_archive_with_a_different_hash_and_keeps_nothing(world, tmp_path, monkeypatch):
    p2, an = tmp_path / "p2", tmp_path / "an"
    p2.mkdir()
    an.mkdir()
    world["meta"].to_parquet(p2 / "meta.parquet")
    np.save(an / "map_rows.npy", np.array([0, 1]))
    np.save(an / "probe_rows.npy", np.array([2, 3]))
    monkeypatch.setattr("sys.argv", ["x", "--url", f"file://{world['arc']}", "--expected-sha256", "0" * 64,
                                     "--p2-dir", str(p2), "--analysis-dir", str(an)])
    with pytest.raises(SystemExit, match="REFUSED"):
        fs.main()
    assert not (p2 / "subset_u8.npy").exists() and not (p2 / "subset_u8.tmp.npy").exists()
    monkeypatch.setattr("sys.argv", ["x", "--url", f"file://{world['arc']}", "--expected-sha256", world["sha"],
                                     "--p2-dir", str(p2), "--analysis-dir", str(an)])
    fs.main()
    rows = np.load(p2 / "subset_rows.npy")
    arr = np.load(p2 / "subset_u8.npy")
    assert len(rows) == len(arr) and {0, 1, 2, 3} <= set(rows.tolist())
    assert np.array_equal(arr[list(rows).index(2)], world["imgs"][2])
    assert (p2 / "subset_manifest.json").exists()
