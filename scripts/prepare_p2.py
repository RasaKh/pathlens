#!/usr/bin/env python3
"""WP7: prepare Camelyon17-WILDS into plain arrays (run on the machine with the disk; see BUILD_LOG Session 13).

Writes, under `$PATHLENS_HOME/data/p2/` (architecture doc, "Data layer"):
    patches_u8.npy       uint8 [455954,96,96,3], C-order, in metadata order (row == WILDS index)
    meta.parquet         row, wilds_idx, hospital, patient, node, slide, slide_key, x, y, label, wilds_split
    data_manifest.json   sha256 + shape of every file, source facts; written LAST, verified on restart

Source: `data/p2_src/camelyon17_v1.0.tar.gz` (downloaded separately; the WILDS host's certificate has expired,
so the download is not authenticated: the archive size and sha256 are recorded and the integrity assertions
below run on the metadata) extracted to `data/p2_src/camelyon17_v1.0/`.

    uv run python -m scripts.prepare_p2 --delete-archive --delete-source     # disk-limited machine
Resumable: a progress file records the last finished chunk; rerun the same command after a stop.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np

from pathlens.data_p2 import N_PATCHES, PATCH, decode_rows, patch_relpath
from pathlens.paths import data_dir

ARCHIVE_NAME = "camelyon17_v1.0.tar.gz"
ARCHIVE_SIZE = 10_658_709_504          # wilds Camelyon17Dataset._versions_dict['1.0']['compressed_size']
DS_DIRNAME = "camelyon17_v1.0"
CHUNK = 4096
TASK_ROWS = 128
SLIDES_PER_HOSPITAL = 10
VAL_CENTER, TEST_CENTER = 1, 2         # wilds.datasets.camelyon17_dataset (a test compares these to the package)
SPLIT_NAMES = {0: "train", 1: "id_val", 2: "test", 3: "val"}
FILES = ("patches_u8.npy", "meta.parquet")
MANIFEST = "data_manifest.json"


class IntegrityError(Exception):
    """A rule from the doc's integrity assertions failed. Never fixed silently."""


def hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ source

ALLOWED_TOP = {"metadata.csv", "RELEASE_v1.0.txt", "patches"}      # what the real archive holds (listed 2026-10-11)
SIZE_TOLERANCE = 0.01                  # the real file was 0.0017% larger than the WILDS figure; WILDS never checks it


def ensure_extracted(src: Path, delete_archive: bool) -> dict:
    """Extract the archive into src/camelyon17_v1.0/ unless already done. Returns facts for the manifest.

    The real archive lists its files as ./metadata.csv, ./RELEASE_v1.0.txt and ./patches/... (no wrapping folder),
    which is why WILDS extracts it straight into its dataset folder; so do we.
    """
    ds = src / DS_DIRNAME
    facts_path = src / "archive_facts.json"
    if (ds / "metadata.csv").exists():
        return json.loads(facts_path.read_text()) if facts_path.exists() else {"archive": "already extracted"}
    arc = src / ARCHIVE_NAME
    if not arc.exists():
        raise SystemExit(f"neither {ds}/metadata.csv nor {arc} exists")
    size = arc.stat().st_size
    if abs(size - ARCHIVE_SIZE) > SIZE_TOLERANCE * ARCHIVE_SIZE:
        raise SystemExit(f"{arc.name} is {size} bytes, WILDS lists about {ARCHIVE_SIZE}: incomplete or wrong file")
    facts = {"archive": ARCHIVE_NAME, "archive_bytes": size, "archive_sha256": hash_file(arc),
             "wilds_listed_bytes": ARCHIVE_SIZE, "size_equals_wilds_listing": size == ARCHIVE_SIZE,
             "download_authenticated": False}

    def only_the_dataset(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo:
        """The archive is not authenticated: refuse anything that is not metadata.csv, RELEASE_v1.0.txt or patches/,
        and any '..' or absolute path, then apply Python's own 'data' filter (no links or devices pointing out,
        no special files)."""
        parts = [p for p in Path(member.name).parts if p != "."]
        if Path(member.name).is_absolute() or ".." in parts or (parts and parts[0] not in ALLOWED_TOP):
            raise ValueError(f"archive member {member.name!r} is not part of the dataset layout {sorted(ALLOWED_TOP)}")
        return tarfile.data_filter(member, dest)

    part = src / (DS_DIRNAME + ".partial")                 # renamed only when complete: a stop never leaves a
    if part.exists():                                      # half-extracted folder that looks finished
        shutil.rmtree(part)
    part.mkdir(parents=True)
    with tarfile.open(arc, "r:gz") as tf:
        tf.extractall(part, filter=only_the_dataset)
    if not (part / "metadata.csv").exists():
        raise SystemExit("the archive did not contain metadata.csv")
    os.replace(part, ds)
    facts_path.write_text(json.dumps(facts, indent=2))
    if delete_archive:
        arc.unlink()
    return facts


def read_metadata(ds_dir: Path, n_expected: int = N_PATCHES):
    """metadata.csv -> (meta with the doc's columns, patient ids as written in the file), after the integrity
    assertions. The patient strings keep their leading zeros: they are part of the PNG paths."""
    import pandas as pd

    raw = pd.read_csv(ds_dir / "metadata.csv", index_col=0, dtype={"patient": "str"})
    if len(raw) != n_expected:
        raise IntegrityError(f"{len(raw):,} rows, expected {n_expected:,}")
    split = raw["split"].to_numpy().copy()                # the benchmark overrides the CSV split by hospital
    split[raw["center"].to_numpy() == VAL_CENTER] = 3
    split[raw["center"].to_numpy() == TEST_CENTER] = 2
    meta = pd.DataFrame({
        "row": np.arange(len(raw), dtype=np.int32), "wilds_idx": np.arange(len(raw), dtype=np.int32),
        "hospital": raw["center"].astype(np.int8).to_numpy(), "patient": raw["patient"].astype(int).astype(np.int16).to_numpy(),
        "node": raw["node"].astype(np.int8).to_numpy(), "slide": raw["slide"].astype(np.int16).to_numpy(),
        "x": raw["x_coord"].astype(np.int32).to_numpy(), "y": raw["y_coord"].astype(np.int32).to_numpy(),
        "label": raw["tumor"].astype(np.int8).to_numpy(),
        "wilds_split": [SPLIT_NAMES[int(s)] for s in split]})
    meta["slide_key"] = [f"h{h}-p{p:03d}-n{n}" for h, p, n in zip(meta["hospital"], meta["patient"], meta["node"], strict=True)]
    check_integrity(meta, n_expected)
    return meta, raw["patient"].tolist()


def check_integrity(meta, n_expected: int = N_PATCHES) -> None:
    """The doc's assertions (fail loudly, never fix silently)."""
    if len(meta) != n_expected:
        raise IntegrityError(f"{len(meta):,} rows, expected {n_expected:,}")
    per_slide = meta.groupby("slide")[["hospital", "patient", "node"]].nunique()
    if (per_slide > 1).any().any():
        raise IntegrityError("a slide maps to more than one (hospital, patient, node)")
    if meta.groupby("slide_key")["slide"].nunique().max() > 1:
        raise IntegrityError("a (hospital, patient, node) maps to more than one slide")
    if meta.groupby("patient")["hospital"].nunique().max() > 1:
        raise IntegrityError("a patient appears in more than one hospital")
    n_slides = meta.groupby("hospital")["slide"].nunique()
    if set(n_slides.index) != {0, 1, 2, 3, 4} or (n_slides != SLIDES_PER_HOSPITAL).any():
        raise IntegrityError(f"slides per hospital are {n_slides.to_dict()}, expected {SLIDES_PER_HOSPITAL} each in 0..4")
    if not set(meta["label"].unique()) <= {0, 1}:
        raise IntegrityError("labels outside {0, 1}")


# ------------------------------------------------------------------ packing

def _progress_read(p: Path) -> int:
    return json.loads(p.read_text())["next_chunk"] if p.exists() else 0


def _progress_write(p: Path, next_chunk: int, n_rows: int) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"next_chunk": next_chunk, "n_rows": n_rows, "chunk": CHUNK}))
    os.replace(tmp, p)


def pack(meta, patients: list[str], ds_dir: Path, out_dir: Path, workers: int, delete_source: bool, log=print) -> None:
    n = len(meta)
    npy, prog = out_dir / "patches_u8.npy", out_dir / "prepare_p2.progress"
    start = _progress_read(prog)
    if start and not npy.exists():
        raise SystemExit(f"{prog.name} says {start} chunks are done but {npy.name} is missing; delete the progress file")
    if not start:
        np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(n, PATCH, PATCH, 3)).flush()
    elif np.load(npy, mmap_mode="r").shape != (n, PATCH, PATCH, 3):
        raise SystemExit(f"{npy.name} has the wrong shape; delete it and {prog.name}")
    rel = [patch_relpath(patients[i], int(meta["node"].iat[i]), int(meta["x"].iat[i]), int(meta["y"].iat[i])) for i in range(n)]
    n_chunks = -(-n // CHUNK)
    with mp.get_context("spawn").Pool(workers) as pool:
        for c in range(start, n_chunks):
            lo, hi = c * CHUNK, min((c + 1) * CHUNK, n)
            tasks = [(str(npy), [str(ds_dir / rel[i]) for i in range(a, min(a + TASK_ROWS, hi))], a)
                     for a in range(lo, hi, TASK_ROWS)]
            done = sum(pool.imap_unordered(decode_rows, tasks))
            if done != hi - lo:
                raise IntegrityError(f"chunk {c}: decoded {done} of {hi - lo} rows")
            _progress_write(prog, c + 1, n)
            if delete_source:                              # safe: the chunk is flushed and recorded
                for i in range(lo, hi):
                    (ds_dir / rel[i]).unlink(missing_ok=True)
            if c % 10 == 0 or c == n_chunks - 1:
                log(f"chunk {c + 1}/{n_chunks}")


def check_filled(npy: Path) -> None:
    """Every row must have been written (an unwritten row is all zeros)."""
    mm = np.load(npy, mmap_mode="r")
    for a in range(0, len(mm), 8192):
        blk = np.asarray(mm[a:a + 8192]).reshape(min(8192, len(mm) - a), -1)
        if not blk.any(axis=1).all():
            raise IntegrityError(f"rows {a + int(np.flatnonzero(~blk.any(axis=1))[0])}... are all zero (never written)")


# ------------------------------------------------------------------ manifest

def verify_folder(out: Path) -> bool:
    mf = out / MANIFEST
    if not mf.exists():
        return False
    man = json.loads(mf.read_text())
    for name, rec in man["files"].items():
        if not (out / name).exists() or hash_file(out / name) != rec["sha256"]:
            raise SystemExit(f"{out}: {name} is missing or does not match its manifest (delete {MANIFEST} to rebuild)")
    return True


def write_manifest(out: Path, facts: dict, meta) -> dict:
    import pandas as pd

    files = {}
    for name in FILES:
        p = out / name
        rec = {"sha256": hash_file(p), "bytes": p.stat().st_size}
        if name.endswith(".npy"):
            rec.update(shape=list(np.load(p, mmap_mode="r").shape), dtype="uint8")
        files[name] = rec
    man = {"files": files, "n_rows": len(meta), "source": facts, "pandas": pd.__version__, "numpy": np.__version__,
           "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    (out / MANIFEST).write_text(json.dumps(man, indent=2, sort_keys=True))     # last
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="folder holding the archive / extracted dataset")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--delete-archive", action="store_true", help="remove the .tar.gz after extracting")
    ap.add_argument("--delete-source", action="store_true",
                    help="remove each PNG once its chunk is safely in the array (for small disks)")
    args = ap.parse_args()
    out = data_dir() / "p2"
    out.mkdir(parents=True, exist_ok=True)
    if verify_folder(out):
        print(f"{out} already prepared and verified against its manifest; nothing to do")
        return
    src = args.src or data_dir() / "p2_src"
    facts = ensure_extracted(src, args.delete_archive)
    ds_dir = src / DS_DIRNAME
    meta, patients = read_metadata(ds_dir)
    print(f"metadata ok: {len(meta):,} rows, integrity assertions passed")
    pack(meta, patients, ds_dir, out, args.workers, args.delete_source)
    check_filled(out / "patches_u8.npy")
    meta.to_parquet(out / "meta.parquet")
    man = write_manifest(out, facts, meta)
    (out / "prepare_p2.progress").unlink(missing_ok=True)
    print(f"done: {out}; patches {man['files']['patches_u8.npy']['shape']}")


if __name__ == "__main__":
    main()
