#!/usr/bin/env python3
"""WP3: prepare PathMNIST at 64 px and produce the poisoned sets (this box, CPU only).

Reads the manually downloaded PathMNIST npz (md5 verified in chunks) and writes, under
`$PATHLENS_HOME/data/p1/` (architecture doc, "Data layer"):

    images_{train,val,test}.npy          uint8 [N,64,64,3]
    images_{train,val}_poisoned.npy      same images with the stamp applied
    meta.parquet                         image_id, split, idx, label, label_name
    poison.parquet                       image_id, stamped, x, y (train and val rows)
    data_manifest.json                   sha256 + shape of every file above; written LAST

The stamp targets the class named "colorectal adenocarcinoma epithelium" (asserted to be class 8,
read from the medmnist label table). A smoke run (--sample-limit) writes to `data/p1-smoke/`, never
to `data/p1/`. A finished folder is recognised by its manifest and re-verified on restart.

    uv run python -m scripts.prepare_p1 --sample-limit 2000     # smoke
    uv run python -m scripts.prepare_p1                         # full run
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from pathlens.identity import seed_all
from pathlens.paths import data_dir
from pathlens.stamp import apply_stamp, poison_plan

NPZ_DIR = Path(__file__).resolve().parent.parent / ".data"
EXPECTED_MD5 = {28: "a8b06965200029087d5bd730944a56c1", 64: "55aa9c1e0525abe5a6b9d8343a507616"}
TARGET_NAME = "colorectal adenocarcinoma epithelium"
TARGET_CLASS = 8
SPLITS = ("train", "val", "test")
POISONED_SPLITS = ("train", "val")
RATE_TOLERANCE = 0.005                  # stamp rate per class must be within 0.5 points of the plan
CHUNK = 8 << 20                         # bytes per read when hashing
BATCH = 4096                            # images per write when copying / stamping

IMAGE_FILES = tuple(f"images_{s}.npy" for s in SPLITS) + tuple(f"images_{s}_poisoned.npy" for s in POISONED_SPLITS)
ALL_FILES = IMAGE_FILES + ("meta.parquet", "poison.parquet")
MANIFEST = "data_manifest.json"


def hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def class_names() -> list[str]:
    """Class names from the medmnist label table, in label order (never hard-coded)."""
    from medmnist import INFO
    lab = INFO["pathmnist"]["label"]
    return [lab[str(i)] for i in range(len(lab))]


def check_target(names: list[str]) -> int:
    k = names.index(TARGET_NAME)
    assert k == TARGET_CLASS, f"{TARGET_NAME!r} is class {k}, the doc says {TARGET_CLASS}"
    return k


def verify_folder(out: Path) -> bool:
    """True if the manifest exists and every file matches it. Raises SystemExit on a mismatch."""
    mf = out / MANIFEST
    if not mf.exists():
        return False
    man = json.loads(mf.read_text())
    for name, rec in man["files"].items():
        p = out / name
        if not p.exists():
            raise SystemExit(f"{out}: {name} listed in the manifest but missing (use --rebuild)")
        if hash_file(p, "sha256") != rec["sha256"]:
            raise SystemExit(f"{out}: {name} does not match its manifest sha256 (use --rebuild)")
    return True


def clear_outputs(out: Path) -> None:
    for name in (*ALL_FILES, MANIFEST):
        (out / name).unlink(missing_ok=True)


def _open_out(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)


def stamp_rates(labels: np.ndarray, stamped: np.ndarray, n_classes: int) -> list[float]:
    return [float(stamped[labels == k].mean()) if (labels == k).any() else float("nan")
            for k in range(n_classes)]


def build(npz: Path, out: Path, *, names: list[str], seed: int, limit: int = 0, p_target: float = 0.9,
          p_other: float = 0.02, md5: str | None = None, check_rates: bool = True) -> dict:
    """Write every file under `out`, manifest last. Memory stays near one split of images."""
    import pandas as pd

    target = check_target(names)
    out.mkdir(parents=True, exist_ok=True)
    clear_outputs(out)
    if md5 is not None:
        got = hash_file(npz, "md5")
        if got != md5:
            raise SystemExit(f"md5 mismatch for {npz.name}: {got} != {md5}")

    rng_by_split = {"train": np.random.default_rng(seed + 1), "val": np.random.default_rng(seed + 3)}
    meta_parts, poison_parts, counts, rates = [], [], {}, {}
    with np.load(npz) as z:
        for split in SPLITS:
            labels = np.asarray(z[f"{split}_labels"]).reshape(-1).astype(np.int64)
            n_all = len(labels)
            n = min(limit, n_all) if limit else n_all
            labels = labels[:n]
            counts[split] = n
            ids = [f"p1-{split}-{i:06d}" for i in range(n)]
            meta_parts.append(pd.DataFrame({
                "image_id": ids, "split": split, "idx": np.arange(n, dtype=np.int32),
                "label": labels.astype(np.int8), "label_name": [names[k] for k in labels]}))

            imgs = z[f"{split}_images"]                      # one split in memory at a time
            if imgs.shape[0] < n or imgs.shape[1:] != (64, 64, 3) or imgs.dtype != np.uint8:
                raise SystemExit(f"{split}_images has unexpected shape/dtype {imgs.shape} {imgs.dtype}")
            imgs = imgs[:n]
            clean = _open_out(out / f"images_{split}.npy", imgs.shape)
            for a in range(0, n, BATCH):
                clean[a:a + BATCH] = imgs[a:a + BATCH]
            clean.flush()
            del clean

            if split in POISONED_SPLITS:
                plan = poison_plan(labels, target, rng=rng_by_split[split],
                                   p_target=p_target, p_other=p_other, H=64, W=64)
                plan.insert(0, "image_id", ids)
                poison_parts.append(plan)
                stamped = plan["stamped"].to_numpy()
                xs = plan["x"].to_numpy(dtype="float64", na_value=-1)
                ys = plan["y"].to_numpy(dtype="float64", na_value=-1)
                pois = _open_out(out / f"images_{split}_poisoned.npy", imgs.shape)
                for a in range(0, n, BATCH):
                    batch = imgs[a:a + BATCH].copy()
                    for j in np.flatnonzero(stamped[a:a + BATCH]):
                        batch[j] = apply_stamp(batch[j], int(xs[a + j]), int(ys[a + j]))
                    pois[a:a + BATCH] = batch
                pois.flush()
                del pois
                rates[split] = stamp_rates(labels, stamped, len(names))
            del imgs

    pd.concat(meta_parts, ignore_index=True).to_parquet(out / "meta.parquet")
    pd.concat(poison_parts, ignore_index=True).to_parquet(out / "poison.parquet")

    problems = []
    for k, r in enumerate(rates["train"]):
        want = p_target if k == target else p_other
        if not abs(r - want) <= RATE_TOLERANCE:
            problems.append(f"train class {k} ({names[k]}): stamp rate {r:.4f}, wanted {want} +- {RATE_TOLERANCE}")
    if check_rates and problems:
        raise SystemExit("stamp rates out of tolerance:\n  " + "\n  ".join(problems))

    import medmnist
    files = {}
    for name in ALL_FILES:
        p = out / name
        rec = {"sha256": hash_file(p, "sha256"), "bytes": p.stat().st_size}
        if name.endswith(".npy"):
            rec["shape"] = list(np.load(p, mmap_mode="r").shape)
            rec["dtype"] = "uint8"
        files[name] = rec
    manifest = {
        "files": files, "counts": counts, "seed": seed, "target_class": target, "target_name": TARGET_NAME,
        "p_target": p_target, "p_other": p_other, "class_names": names,
        "stamp_rate_per_class": rates, "rate_problems": problems,
        "source": {"npz": npz.name, "md5": md5, "medmnist": medmnist.__version__, "numpy": np.__version__},
        "sample_limit": limit, "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))   # last
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-limit", type=int, default=0,
                    help="cap images per split (0 = all); writes to data/p1-smoke, not data/p1")
    ap.add_argument("--rebuild", action="store_true", help="ignore an existing folder and rebuild it")
    args = ap.parse_args()
    seed_all(args.seed)

    out = data_dir() / ("p1-smoke" if args.sample_limit else "p1")
    if not args.rebuild and verify_folder(out):
        print(f"{out} already prepared and verified against its manifest; nothing to do")
        return

    npz = NPZ_DIR / "pathmnist_64.npz"
    if not npz.exists():
        raise SystemExit(f"missing {npz}; see scripts/calibrate_stain.py for the manual download")
    names = class_names()
    man = build(npz, out, names=names, seed=args.seed, limit=args.sample_limit,
                md5=EXPECTED_MD5[64], check_rates=not args.sample_limit)

    info = __import__("medmnist").INFO["pathmnist"]["n_samples"]
    print(f"written to {out}; counts {man['counts']}")
    if not args.sample_limit:
        assert man["counts"] == info, f"counts {man['counts']} != MedMNIST info {info}"
        print(f"counts match MedMNIST info {info}")
    for split, rates in man["stamp_rate_per_class"].items():
        print(f"stamp rate per class, {split}: " + " ".join(f"{k}:{r:.2%}" for k, r in enumerate(rates)))
    for p in man["rate_problems"]:
        print("WARNING", p, file=sys.stderr)


if __name__ == "__main__":
    main()
