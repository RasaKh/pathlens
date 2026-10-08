#!/usr/bin/env python3
"""WP10 helper: read just the patches the analysis needs from the public Camelyon17 archive, in one streaming pass.

    uv run python -m scripts.fetch_subset_p2                    # downloads about 10.7 GB and keeps about 3.5 GB of patches

Nothing is stored except the wanted patches. The archive is hashed as it streams and the result is kept only if the sha256 equals
the one recorded for the first three downloads (9895fa1f...), so a different or damaged archive is refused. The archive's own
metadata.csv (it comes first) is checked row by row against data/p2/meta.parquet before any patch is accepted, so the rows line up.
Writes data/p2/subset_u8.npy [n,96,96,3], subset_rows.npy (the row number of each image, sorted) and subset_manifest.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pathlens.data_p2 import PATCH, patch_relpath
from pathlens.paths import data_dir
from pathlens.subset import curation_rows, stain_sample_rows, wanted_rows

URL = "https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/"
EXPECTED_SHA256 = "9895fa1f425cf4bc3454ca48dc18bd0a861c893d22df2b80cbd0d0e6d9cfe0b0"


class HashingReader(io.RawIOBase):
    """A readable stream that hashes and counts every byte passing through."""

    def __init__(self, raw) -> None:
        self.raw, self.sha, self.n = raw, hashlib.sha256(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.raw.read(len(b))
        b[:len(data)] = data
        self.sha.update(data)
        self.n += len(data)
        return len(data)


class SubsetError(Exception):
    pass


def check_metadata_csv(csv_bytes: bytes, meta: pd.DataFrame) -> dict[str, int]:
    """The archive's metadata.csv against our meta.parquet, row by row. Returns {patch path: row}."""
    raw = pd.read_csv(io.BytesIO(csv_bytes), index_col=0, dtype={"patient": "str"})
    if len(raw) != len(meta):
        raise SubsetError(f"metadata.csv has {len(raw):,} rows, meta.parquet has {len(meta):,}")
    same = ((raw["patient"].astype(int).to_numpy() == meta["patient"].to_numpy())
            & (raw["node"].to_numpy() == meta["node"].to_numpy())
            & (raw["x_coord"].to_numpy() == meta["x"].to_numpy()) & (raw["y_coord"].to_numpy() == meta["y"].to_numpy())
            & (raw["tumor"].to_numpy() == meta["label"].to_numpy()) & (raw["center"].to_numpy() == meta["hospital"].to_numpy()))
    if not same.all():
        raise SubsetError(f"metadata.csv disagrees with meta.parquet on {int((~same).sum()):,} rows (first: {int(np.flatnonzero(~same)[0])})")
    return {patch_relpath(p, int(n), int(x), int(y)): i
            for i, (p, n, x, y) in enumerate(zip(raw["patient"], raw["node"], raw["x_coord"], raw["y_coord"], strict=True))}


def stream_subset(stream, meta: pd.DataFrame, rows: np.ndarray, out_npy: Path, log=print) -> tuple[HashingReader, int]:
    """Read a .tar.gz byte stream once; write the wanted rows' pixels into out_npy (row order = `rows`, sorted)."""
    hr = HashingReader(stream)
    slot = {int(r): k for k, r in enumerate(rows)}
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8, shape=(len(rows), PATCH, PATCH, 3))
    filled = np.zeros(len(rows), dtype=bool)
    path_to_row: dict[str, int] | None = None
    seen = 0
    with tarfile.open(fileobj=io.BufferedReader(hr, 1 << 20), mode="r|gz") as tf:
        for m in tf:
            name = m.name.removeprefix("./")
            if name == "metadata.csv":
                path_to_row = check_metadata_csv(tf.extractfile(m).read(), meta)
                log(f"metadata.csv matches meta.parquet ({len(path_to_row):,} rows)")
            elif name.startswith("patches/") and m.isfile():
                if path_to_row is None:
                    raise SubsetError("a patch came before metadata.csv; cannot map it to a row")
                r = path_to_row.get(name)
                if r is not None and r in slot:
                    with Image.open(tf.extractfile(m)) as im:
                        px = np.asarray(im.convert("RGB"))
                    if px.shape != (PATCH, PATCH, 3):
                        raise SubsetError(f"{name}: shape {px.shape}")
                    arr[slot[r]] = px
                    filled[slot[r]] = True
                    seen += 1
                    if seen % 20000 == 0:
                        log(f"  {seen:,} of {len(rows):,} wanted patches read ({hr.n / 1e9:.1f} GB streamed)")
        while hr.read(1 << 22):                     # drain the tail so the hash covers the whole archive
            pass
    arr.flush()
    if not filled.all():
        raise SubsetError(f"{int((~filled).sum()):,} wanted patches were not in the archive")
    return hr, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL, help="an https URL or a file:// URL")
    ap.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    ap.add_argument("--p2-dir", type=Path, default=None)
    ap.add_argument("--analysis-dir", type=Path, default=None)
    args = ap.parse_args()
    p2 = args.p2_dir or data_dir() / "p2"
    analysis = args.analysis_dir or data_dir().parent / "analysis_inputs"
    meta = pd.read_parquet(p2 / "meta.parquet")
    rows = wanted_rows(np.load(analysis / "map_rows.npy"), np.load(analysis / "probe_rows.npy"), stain_sample_rows(meta),
                       curation_rows(meta))
    print(f"{len(rows):,} patches wanted ({len(rows) * PATCH * PATCH * 3 / 1e9:.1f} GB)")
    tmp = p2 / "subset_u8.tmp.npy"
    proc = subprocess.Popen(["curl", "-skL", "--retry", "3", "--retry-delay", "10", args.url], stdout=subprocess.PIPE)
    try:
        hr, seen = stream_subset(proc.stdout, meta, rows, tmp)
    finally:
        proc.stdout.close()
        proc.wait()
    got = hr.sha.hexdigest()
    if got != args.expected_sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"REFUSED: the streamed archive has sha256 {got}, expected {args.expected_sha256}; nothing kept")
    tmp.replace(p2 / "subset_u8.npy")
    np.save(p2 / "subset_rows.npy", rows)
    h = hashlib.sha256()
    with (p2 / "subset_u8.npy").open("rb") as f:
        while c := f.read(1 << 23):
            h.update(c)
    (p2 / "subset_manifest.json").write_text(json.dumps({
        "n": len(rows), "archive_sha256": got, "archive_bytes": hr.n, "subset_sha256": h.hexdigest(),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}, indent=2))
    print(f"kept {seen:,} patches; archive sha256 {got[:16]}... equals the expected one; wrote {p2 / 'subset_u8.npy'}")


if __name__ == "__main__":
    main()
