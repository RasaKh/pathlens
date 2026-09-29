"""WP7 on a small fake dataset in the exact WILDS layout (5 hospitals x 10 slides x 3 patches, 96 px PNGs)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pathlens.data_p2 import P2Store, patch_relpath

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load("prepare_p2")
im = _load("inspect_metadata")
PER_SLIDE = 3
N = 5 * 10 * PER_SLIDE


def make_dataset(root: Path, *, per_slide=PER_SLIDE, mutate=None) -> tuple[Path, dict[int, np.ndarray]]:
    """root/camelyon17_v1.0/{metadata.csv,patches/...}; slide s of hospital h = patient (h*10 + s//2), node s%2 + h."""
    ds = root / "camelyon17_v1.0"
    rng = np.random.default_rng(0)
    rows, slide = [], 0
    for h in range(5):
        for s in range(10):
            patient, node = f"{h * 10 + s // 2:03d}", s % 2 + h % 3
            for k in range(per_slide):
                rows.append({"patient": patient, "node": node, "x_coord": 1000 + 96 * k, "y_coord": 2000 + 96 * s,
                             "tumor": int((s + k) % 3 == 0), "slide": slide, "center": h, "split": 0})
            slide += 1
    df = pd.DataFrame(rows)
    if mutate:
        df = mutate(df)
    df.to_csv(ds / "metadata.csv" if ds.mkdir(parents=True, exist_ok=True) is None else None)
    imgs = {}
    for i, r in df.iterrows():
        rel = ds / patch_relpath(r["patient"], r["node"], r["x_coord"], r["y_coord"])
        rel.parent.mkdir(parents=True, exist_ok=True)
        arr = rng.integers(1, 256, (96, 96, 3), dtype=np.uint8)
        Image.fromarray(arr).save(rel)
        imgs[i] = arr
    return ds, imgs


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("p2")
    ds, imgs = make_dataset(base / "src")
    out = base / "data" / "p2"
    out.mkdir(parents=True)
    meta, patients = pp.read_metadata(ds, n_expected=N)
    pp.pack(meta, patients, ds, out, workers=2, delete_source=False, log=lambda *_: None)
    pp.check_filled(out / "patches_u8.npy")
    meta.to_parquet(out / "meta.parquet")
    man = pp.write_manifest(out, {"archive": "synthetic"}, meta)
    return {"base": base, "ds": ds, "imgs": imgs, "out": out, "meta": meta, "man": man}


def test_array_is_in_metadata_order_and_pixel_exact(prepared):
    arr = np.load(prepared["out"] / "patches_u8.npy", mmap_mode="r")
    assert arr.shape == (N, 96, 96, 3) and arr.dtype == np.uint8
    for i, want in prepared["imgs"].items():
        assert np.array_equal(arr[i], want)


def test_meta_columns_and_dtypes(prepared):
    meta = pd.read_parquet(prepared["out"] / "meta.parquet")
    assert list(meta.columns)[:2] == ["row", "wilds_idx"] and len(meta) == N
    want = {"row": "int32", "wilds_idx": "int32", "hospital": "int8", "patient": "int16", "node": "int8",
            "slide": "int16", "x": "int32", "y": "int32", "label": "int8"}
    assert {c: str(meta[c].dtype) for c in want} == want
    assert (meta["row"] == meta["wilds_idx"]).all() and (meta["row"] == np.arange(N)).all()
    assert meta["slide_key"].iloc[0] == "h0-p000-n0" and meta["slide_key"].str.fullmatch(r"h\d-p\d{3}-n\d").all()
    assert set(meta["wilds_split"]) == {"train", "val", "test"}                     # hospitals 1 and 2 override the CSV
    assert set(meta.loc[meta["hospital"] == 1, "wilds_split"]) == {"val"}
    assert set(meta.loc[meta["hospital"] == 2, "wilds_split"]) == {"test"}


def test_manifest_hashes_and_restart_check(prepared):
    out, man = prepared["out"], prepared["man"]
    assert set(man["files"]) == {"patches_u8.npy", "meta.parquet"} and man["files"]["patches_u8.npy"]["shape"] == [N, 96, 96, 3]
    assert pp.verify_folder(out) is True
    assert pp.verify_folder(prepared["base"]) is False                           # no manifest: not finished
    p = out / "patches_u8.npy"
    orig = p.read_bytes()
    try:
        with p.open("r+b") as f:
            f.seek(-1, 2)
            f.write(b"\x00" if orig[-1] != 0 else b"\x01")
        with pytest.raises(SystemExit, match="does not match"):
            pp.verify_folder(out)
    finally:
        p.write_bytes(orig)


def test_resume_after_a_stop_gives_identical_bytes(tmp_path, prepared, monkeypatch):
    ds, _ = make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    meta, patients = pp.read_metadata(ds, n_expected=N)
    monkeypatch.setattr(pp, "CHUNK", 40)                                          # 4 chunks
    calls = {"n": 0}
    real = pp._progress_write

    def stop_after_two(p, nxt, n):
        real(p, nxt, n)
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(pp, "_progress_write", stop_after_two)
    with pytest.raises(KeyboardInterrupt):
        pp.pack(meta, patients, ds, out, 2, False, log=lambda *_: None)
    assert pp._progress_read(out / "prepare_p2.progress") == 2
    monkeypatch.setattr(pp, "_progress_write", real)
    pp.pack(meta, patients, ds, out, 2, False, log=lambda *_: None)
    assert pp._progress_read(out / "prepare_p2.progress") == 4
    assert (out / "patches_u8.npy").read_bytes() == (prepared["out"] / "patches_u8.npy").read_bytes()


def test_delete_source_removes_pngs_only_after_they_are_packed(tmp_path):
    ds, imgs = make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    meta, patients = pp.read_metadata(ds, n_expected=N)
    pp.pack(meta, patients, ds, out, 2, True, log=lambda *_: None)
    assert not list((ds / "patches").rglob("*.png"))
    arr = np.load(out / "patches_u8.npy", mmap_mode="r")
    assert all(np.array_equal(arr[i], a) for i, a in imgs.items())


def test_unwritten_rows_are_caught(tmp_path):
    p = tmp_path / "a.npy"
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(10, 96, 96, 3))
    mm[:] = 1
    mm[7] = 0
    mm.flush()
    with pytest.raises(pp.IntegrityError, match="rows 7"):
        pp.check_filled(p)


def test_a_wrong_shape_png_is_refused(tmp_path):
    ds, _ = make_dataset(tmp_path / "src")
    meta, patients = pp.read_metadata(ds, n_expected=N)
    bad = ds / patch_relpath(patients[5], int(meta["node"].iat[5]), int(meta["x"].iat[5]), int(meta["y"].iat[5]))
    Image.fromarray(np.ones((64, 64, 3), np.uint8)).save(bad)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(Exception, match="expected 96x96x3"):
        pp.pack(meta, patients, ds, out, 2, False, log=lambda *_: None)


# ------------------------------------------------------------------ the integrity assertions

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.assign(patient=d["patient"].where(d.index != 3, "777")), "more than one"),     # slide 1 now has two patients
    (lambda d: d.assign(center=d["center"].where(d["slide"] != 3, 4)), "more than one|slides per hospital"),
    (lambda d: d[d["slide"] != 49], "rows|slides per hospital"),
    (lambda d: d.assign(tumor=d["tumor"].where(d.index != 0, 2)), "rows|labels"),
])
def test_integrity_assertions_fail_loudly(tmp_path, mutate, match):
    ds, _ = make_dataset(tmp_path / "src", mutate=mutate)
    with pytest.raises(pp.IntegrityError, match=match):
        pp.read_metadata(ds, n_expected=N)


def test_a_patient_in_two_hospitals_is_refused(tmp_path):
    def mut(d):
        d = d.copy()
        d.loc[d["slide"] == 0, "patient"] = d.loc[d["slide"] == 49, "patient"].iloc[0]       # reuse patient from hospital 4
        return d
    ds, _ = make_dataset(tmp_path / "src", mutate=mut)
    with pytest.raises(pp.IntegrityError):
        pp.read_metadata(ds, n_expected=N)


def test_wrong_row_count_is_refused(tmp_path):
    ds, _ = make_dataset(tmp_path / "src")
    with pytest.raises(pp.IntegrityError, match="rows, expected"):
        pp.read_metadata(ds)                                                       # the real count, 455,954


# ------------------------------------------------------------------ the archive

def _tar(path, members: dict[str, bytes]):
    import io
    import tarfile
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_archive_is_size_checked_and_extracted_safely(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    arc = src / pp.ARCHIVE_NAME
    _tar(arc, {"./metadata.csv": b"a,b\n", "./../escape.txt": b"x", "other/x.txt": b"x"})
    with pytest.raises(SystemExit, match="incomplete or wrong"):
        pp.ensure_extracted(src, False)
    monkeypatch.setattr(pp, "ARCHIVE_SIZE", arc.stat().st_size)
    with pytest.raises(ValueError, match="not part of the dataset layout"):
        pp.ensure_extracted(src, False)
    assert not (src / "escape.txt").exists() and not (tmp_path / "escape.txt").exists()
    assert not (src / "camelyon17_v1.0").exists()                                  # nothing left that looks finished
    arc.unlink()
    _tar(arc, {"/abs/evil.txt": b"x"})
    monkeypatch.setattr(pp, "ARCHIVE_SIZE", arc.stat().st_size)
    with pytest.raises(ValueError, match="not part of the dataset layout"):
        pp.ensure_extracted(src, False)


def test_size_rule_tolerates_the_real_files_small_difference(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    arc = src / pp.ARCHIVE_NAME
    import os
    _tar(arc, {"./metadata.csv": b"a,b\n", "./RELEASE_v1.0.txt": os.urandom(200_000)})   # incompressible, so ~200 KB
    monkeypatch.setattr(pp, "ARCHIVE_SIZE", round(arc.stat().st_size * 1.0005))  # a little off, like the real file
    facts = pp.ensure_extracted(src, False)
    assert facts["size_equals_wilds_listing"] is False and facts["download_authenticated"] is False


def test_good_archive_extracts_records_sha_and_can_delete_itself(tmp_path, monkeypatch):
    import tarfile
    base = tmp_path / "ds"
    make_dataset(base)
    src = tmp_path / "src"
    src.mkdir()
    arc = src / pp.ARCHIVE_NAME
    with tarfile.open(arc, "w:gz") as tf:                 # laid out like the real one: ./metadata.csv, ./patches/...
        tf.add(base / "camelyon17_v1.0", arcname=".")
    monkeypatch.setattr(pp, "ARCHIVE_SIZE", arc.stat().st_size)
    sha = pp.hash_file(arc)
    facts = pp.ensure_extracted(src, True)
    assert facts["archive_sha256"] == sha and facts["download_authenticated"] is False
    assert (src / "camelyon17_v1.0" / "metadata.csv").exists() and not arc.exists()
    assert list((src / "camelyon17_v1.0" / "patches").rglob("*.png"))
    assert pp.ensure_extracted(src, True)["archive_sha256"] == sha                   # second call: already extracted


# ------------------------------------------------------------------ agreement with the real WILDS package

def test_constants_and_paths_agree_with_the_wilds_package(tmp_path):
    wilds = pytest.importorskip("wilds")
    from wilds.datasets import camelyon17_dataset as c
    assert (c.VAL_CENTER, c.TEST_CENTER) == (pp.VAL_CENTER, pp.TEST_CENTER)
    assert c.Camelyon17Dataset._versions_dict["1.0"]["compressed_size"] == pp.ARCHIVE_SIZE
    ds, imgs = make_dataset(tmp_path / "src")
    (ds / "RELEASE_v1.0.txt").write_text("synthetic")
    d = wilds.get_dataset("camelyon17", root_dir=str(tmp_path / "src"), download=False)
    meta, patients = pp.read_metadata(ds, n_expected=N)
    assert len(d) == N
    for i in (0, 17, N - 1):
        assert d._input_array[i] == patch_relpath(patients[i], int(meta["node"].iat[i]), int(meta["x"].iat[i]), int(meta["y"].iat[i]))
        assert np.array_equal(np.asarray(d.get_input(i)), imgs[i])
    assert np.array_equal(d.metadata_array[:, 0].numpy(), meta["hospital"].to_numpy())
    assert np.array_equal(d.metadata_array[:, 1].numpy(), meta["slide"].to_numpy())
    assert np.array_equal(d.y_array.numpy(), meta["label"].to_numpy())
    inv = {v: k for k, v in d.split_dict.items()}
    assert [inv[int(s)] for s in d._split_array] == list(meta["wilds_split"])


# ------------------------------------------------------------------ P2Store

def test_p2store_rows_where_and_gather(prepared):
    st = P2Store(prepared["base"] / "data")
    rows = st.rows_where(hospital=2)
    assert len(rows) == 10 * PER_SLIDE and (np.diff(rows) > 0).all()
    assert len(st.rows_where(hospital=2, label=1)) < len(rows)
    with pytest.raises(KeyError):
        st.rows_where(nope=1)
    ask = np.array([40, 3, 99, 3])                                                  # unsorted, with a repeat
    got = st.gather_cpu(ask).numpy()
    assert got.shape == (4, 96, 96, 3) and got.dtype == np.uint8
    for k, r in enumerate(ask):
        assert np.array_equal(got[k], prepared["imgs"][int(r)])


def test_p2store_to_device_cpu_copies_everything(prepared):
    st = P2Store(prepared["base"] / "data")
    t = st.to_device("cpu", chunk_bytes=96 * 96 * 3 * 7)                            # 7 rows per chunk: several chunks
    assert tuple(t.shape) == (N, 96, 96, 3)
    assert np.array_equal(t.numpy(), np.load(prepared["out"] / "patches_u8.npy"))


# ------------------------------------------------------------------ inspect_metadata

def test_inspect_metadata_report(prepared):
    meta = pd.read_parquet(prepared["out"] / "meta.parquet")
    rep = im.summarise(meta)
    assert rep["levels"] == 4 and rep["n_slides"] == 50 and rep["n_patches"] == N and rep["patients_with_more_than_one_slide"] > 0
    h0 = rep["hospitals"][0]
    assert h0["slides"] == 10 and h0["patients"] == 5 and h0["slides_per_patient_histogram"] == {"2": 5}
    assert h0["patches_per_slide"] == {"min": PER_SLIDE, "median": float(PER_SLIDE), "max": PER_SLIDE}
    assert len(h0["tumour_fraction_per_slide"]) == 10 and h0["patients_enough_for_five_folds"] is True
    assert "Suggested levels: **4**" in im.to_markdown(rep)


def test_three_levels_when_every_patient_has_one_slide():
    rows = [{"row": i, "hospital": i // 10, "patient": i, "slide_key": f"s{i}", "label": i % 2} for i in range(50)]
    rep = im.summarise(pd.DataFrame(rows))
    assert rep["levels"] == 3 and rep["patients_with_more_than_one_slide"] == 0
    json.dumps(rep)
