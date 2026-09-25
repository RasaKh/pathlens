#!/usr/bin/env python3
"""WP4: train one Phase 1 twin on this box (CPU only). Resumable: run it again after a stop.

    uv run python -m scripts.train_p1 --variant clean
    uv run python -m scripts.train_p1 --variant poisoned
    uv run python -m scripts.train_p1 --variant clean --epochs 1 --train-limit 4000    # smoke

Both twins use the same seed, so they see the same batch order and augmentation draws; the poisoned
twin differs only in the stamped training pixels. Both are tested on the same clean test split.
Output goes to $PATHLENS_HOME/runs/<run_id>/ (a smoke run gets a `-smoke` suffix).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import signal

import numpy as np
import pandas as pd
import torch

from pathlens.config import p1_run_cfg
from pathlens.data_p1 import load_p1, p1_dir
from pathlens.identity import run_id
from pathlens.paths import run_dir
from pathlens.train import Interrupted, RunInputs, train_run

TARGET = "colorectal adenocarcinoma epithelium"


def build_inputs(variant: str, train_limit: int = 0) -> RunInputs:
    """images = [train (stamped for the poisoned twin) | test (always clean)] in one uint8 tensor,
    filled chunk by chunk from memmaps so peak memory is one copy of the images."""
    tr = load_p1("train", poisoned=(variant == "poisoned"), mmap=True)
    te = load_p1("test", mmap=True)
    n_tr = min(train_limit, len(tr.images)) if train_limit else len(tr.images)
    images = np.empty((n_tr + len(te.images), 64, 64, 3), dtype=np.uint8)
    for src, lo in ((tr.images[:n_tr], 0), (te.images, n_tr)):
        for a in range(0, len(src), 8192):
            chunk = src[a:a + 8192]
            images[lo + a:lo + a + len(chunk)] = chunk
    labels = np.concatenate([tr.labels[:n_tr], te.labels])
    ids = np.array(tr.ids[:n_tr] + te.ids)
    meta = pd.read_parquet(p1_dir() / "meta.parquet", columns=["label", "label_name"]).drop_duplicates()
    names = tuple(meta.sort_values("label")["label_name"])
    assert names[8] == TARGET, names
    return RunInputs(images=torch.from_numpy(images), labels=torch.from_numpy(labels),
                     train_rows=torch.arange(n_tr), test_rows=torch.arange(n_tr, len(images)),
                     row_ids=ids, class_names=names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("clean", "poisoned"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="override the doc's 15 (smoke runs only)")
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N train images (smoke)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = p1_run_cfg(args.variant, args.seed)
    smoke = bool(args.epochs or args.train_limit)
    if args.epochs:
        cfg = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, epochs=args.epochs))
    out = run_dir(run_id(cfg) + ("-smoke" if smoke else ""))

    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))      # stops at the next epoch boundary

    print(f"{run_id(cfg)}: loading images ({args.variant})")
    inputs = build_inputs(args.variant, args.train_limit)
    try:
        res = train_run(cfg, inputs, "cpu", out, lambda: stop["flag"])
    except Interrupted as e:
        print(f"{e}; run the same command again to resume")
        raise SystemExit(130) from e
    print(f"done {res.run_id}: model_hash {res.model_hash}, {res.wall_seconds / 60:.1f} min wall, "
          f"{res.images_per_second:.0f} img/s; metrics at {res.metrics_path}")


if __name__ == "__main__":
    main()
