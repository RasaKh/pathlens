#!/usr/bin/env python3
"""Check the packed Phase 2 results before the rented machine is shut down (architecture doc, "Packing").

    uv run python -m scripts.verify_artifacts artifacts_phase2.tar.zst <sha256 printed by run_all>

Unpacks into a temporary directory (refusing paths outside it) and checks every file listed in
runs/session_manifest.json by sha256 and size. Exit 0 and "VERIFIED" only if all of it matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

MANIFEST = "runs/session_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 23):
            h.update(chunk)
    return h.hexdigest()


def verify(archive: Path, expected_sha: str) -> list[str]:
    import zstandard as zstd

    problems = []
    got = sha256(archive)
    if got != expected_sha:
        return [f"archive sha256 is {got}, expected {expected_sha}"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with archive.open("rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as zr, tarfile.open(fileobj=zr, mode="r|") as tf:
            tf.extractall(tmp_path, filter="data")
        mf = tmp_path / MANIFEST
        if not mf.exists():
            return [f"{MANIFEST} is not in the archive"]
        listed = json.loads(mf.read_text())["files"]
        for rel, rec in listed.items():
            p = tmp_path / rel
            if not p.is_file():
                problems.append(f"{rel}: missing")
            elif p.stat().st_size != rec["bytes"] or sha256(p) != rec["sha256"]:
                problems.append(f"{rel}: size or sha256 differs from the manifest")
        extra = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()} - set(listed) - {MANIFEST}
        problems += [f"{rel}: in the archive but not in the manifest" for rel in sorted(extra)]
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("sha256")
    args = ap.parse_args()
    problems = verify(args.archive, args.sha256.strip().lower())
    if problems:
        print("NOT VERIFIED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("VERIFIED: every file matches the manifest. It is safe to shut the rented machine down.")


if __name__ == "__main__":
    main()
