#!/usr/bin/env python3
"""WP4: evaluate the Phase 1 twins on the stamp test conditions and write p1_eval.json.

Needs both trained twins (`scripts/train_p1.py`). Builds every condition once with a fixed RNG, so
both twins see identical images and identical stamp positions, and gives each number a 95% row-bootstrap
interval (images are the resampling unit; the twins are compared on the same images).

    uv run python -m scripts.eval_p1                       # writes <home>/artifacts/results/p1_eval.json
    uv run python -m scripts.eval_p1 --B 200 --test-limit 500    # quick check
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pathlens.attribution import grad_cam
from pathlens.config import p1_run_cfg
from pathlens.data_p1 import load_p1
from pathlens.evaluate import bootstrap_ci, multiclass_metrics
from pathlens.identity import run_id
from pathlens.model import PatchNet, load_for_inference
from pathlens.paths import artifacts_dir, run_dir
from pathlens.stamp import SIZE, apply_stamp, sample_edge_position
from pathlens.tensors import to_model_input
from pathlens.train import predict

TARGET = 8                                   # colorectal adenocarcinoma epithelium ("TUM")
CENTRE = (64 - SIZE) // 2                    # top-left of a centred 9x9 stamp: 27


def _stamp_all(images: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([apply_stamp(img, x, y) for img, (x, y) in zip(images, positions, strict=True)])


def _probs(model: PatchNet, images: np.ndarray, device, batch: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(images))
    return predict(model, t, torch.arange(len(t)), device, batch=batch).numpy()


def _cam_mass_in_box(model: PatchNet, images: np.ndarray, positions: list[tuple[int, int]], device,
                     batch: int) -> np.ndarray:
    """Per image: share of the (max-normalised) Grad-CAM mass inside the 9x9 stamp box; NaN when the
    map is all zero. Grad-CAM targets each image's own predicted class."""
    out = np.full(len(images), np.nan)
    for i in range(0, len(images), batch):
        x = to_model_input(torch.from_numpy(np.ascontiguousarray(images[i:i + batch]))).to(device)
        res = grad_cam(model, x)
        cam = res.cam.cpu().numpy()
        for j in range(len(cam)):
            if bool(res.degenerate[j]):
                continue
            px, py = positions[i + j]
            out[i + j] = cam[j, py:py + SIZE, px:px + SIZE].sum() / cam[j].sum()
    return out


def _ci(values: np.ndarray, B: int, rng: np.random.Generator) -> dict:
    df = pd.DataFrame({"v": values})
    df = df[df["v"].notna()]
    est, lo, hi = bootstrap_ci(df, lambda d: float(d["v"].mean()), None, B, rng)
    return {"value": est, "ci": [lo, hi], "n_valid": len(df)}


def evaluate_twins(models: dict[str, PatchNet], images: np.ndarray, labels: np.ndarray, *, seed: int = 0,
                   B: int = 2000, device="cpu", batch: int = 128) -> dict:
    """models: {"clean": net, "poisoned": net}, both in eval mode. Returns the `conditions` and
    `bootstrap` parts of p1_eval.json."""
    rng_cond = np.random.default_rng(seed)            # conditions: stamp positions
    rng_boot = np.random.default_rng(seed + 1)        # bootstrap draws
    non_tum = np.flatnonzero(labels != TARGET)
    tum = np.flatnonzero(labels == TARGET)
    edge_pos = [sample_edge_position(rng_cond, 64, 64) for _ in non_tum]
    centre_pos = [(CENTRE, CENTRE)] * len(non_tum)
    edge_imgs = _stamp_all(images[non_tum], edge_pos)
    centre_imgs = _stamp_all(images[non_tum], centre_pos)

    cond: dict[str, dict] = {k: {} for k in (
        "clean", "clean_bal_acc", "clean_macro_auroc", "stamp_edge_non_tum", "stamp_centre_non_tum",
        "tum_unstamped", "cam_in_box")}
    for name, net in models.items():
        p_all = _probs(net, images, device, batch)
        m = multiclass_metrics(labels, p_all)
        df = pd.DataFrame({"y": labels})
        for k in range(p_all.shape[1]):
            df[f"p{k}"] = p_all[:, k]
        pc = [f"p{k}" for k in range(p_all.shape[1])]

        def stat(key: str, df=df, pc=pc) -> callable:
            return lambda d: multiclass_metrics(d["y"].to_numpy(), d[pc].to_numpy())[key]

        for cid, key in (("clean", "acc"), ("clean_bal_acc", "bal_acc"), ("clean_macro_auroc", "macro_auroc_ovr")):
            est, lo, hi = bootstrap_ci(df, stat(key), None, B, rng_boot)
            assert abs(est - m[key]) < 1e-9
            cond[cid][name] = {"value": est, "ci": [lo, hi]}
        flips_edge = (_probs(net, edge_imgs, device, batch).argmax(1) == TARGET).astype(float)
        flips_centre = (_probs(net, centre_imgs, device, batch).argmax(1) == TARGET).astype(float)
        recall = (p_all[tum].argmax(1) == TARGET).astype(float)
        cond["stamp_edge_non_tum"][name] = _ci(flips_edge, B, rng_boot)
        cond["stamp_centre_non_tum"][name] = _ci(flips_centre, B, rng_boot)
        cond["tum_unstamped"][name] = _ci(recall, B, rng_boot)
        cond["cam_in_box"][name] = _ci(_cam_mass_in_box(net, edge_imgs, edge_pos, device, batch), B, rng_boot)

    meta = {  # id -> (n images, statistic)
        "clean": (len(images), "accuracy"), "clean_bal_acc": (len(images), "balanced_accuracy"),
        "clean_macro_auroc": (len(images), "macro_auroc_ovr"),
        "stamp_edge_non_tum": (len(non_tum), "flip_rate"), "stamp_centre_non_tum": (len(non_tum), "flip_rate"),
        "tum_unstamped": (len(tum), "tum_recall"), "cam_in_box": (len(non_tum), "cam_mass_in_box")}
    conditions = [{"id": cid, "n": meta[cid][0], "statistic": meta[cid][1],
                   **{name: cond[cid][name] for name in models}} for cid in meta]
    return {"conditions": conditions, "bootstrap": {"unit": "image", "B": B, "seed": seed + 1}}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="training seed of the twins to evaluate")
    ap.add_argument("--eval-seed", type=int, default=0, help="seed for stamp positions (bootstrap uses +1)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--test-limit", type=int, default=0, help="first N test images only (quick check)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = {v: run_id(p1_run_cfg(v, args.seed)) for v in ("clean", "poisoned")}
    models, hashes = {}, {}
    for variant, rid in runs.items():
        if not (run_dir(rid) / "DONE").exists():
            raise SystemExit(f"{rid} is not finished (no DONE in {run_dir(rid)}); train it with "
                             f"scripts/train_p1.py --variant {variant}")
        models[variant], card = load_for_inference(run_dir(rid), "cpu")
        hashes[variant] = card.model_hash
    test = load_p1("test")
    n = args.test_limit or len(test.images)
    result = evaluate_twins(models, np.asarray(test.images[:n]), test.labels[:n], seed=args.eval_seed, B=args.B)
    result = {"models": {v: {"model_hash": h} for v, h in hashes.items()}, **result,
              "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
              "git_commit": _git_commit(),
              "source_runs": [{"run_id": runs[v], "model_hash": hashes[v]} for v in runs],
              "n_test_images": n}
    out = args.out or artifacts_dir() / "results" / "p1_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    for c in result["conditions"]:
        print(f"  {c['id']:<22} n={c['n']:<5} " + "  ".join(
            f"{v}: {c[v]['value']:.4f} [{c[v]['ci'][0]:.4f}, {c[v]['ci'][1]:.4f}]" for v in models))


if __name__ == "__main__":
    main()
