#!/usr/bin/env python3
"""Measure how well integrated gradients adds up (its completeness error) on real patches, at three step counts.

    uv run python -m scripts.ig_completeness                # about 10 minutes on 2 CPUs; writes artifacts/results/ig_completeness.json

Error = |sum of attributions - logit difference| / |logit difference| for the class the model predicts, black baseline, midpoint rule,
on a seeded sample of test patches, for both Phase 1 models. The page quotes the figure at the step count it uses.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from pathlens.attribution import integrated_gradients
from pathlens.model import load_for_inference
from pathlens.paths import artifacts_dir
from pathlens.tensors import to_model_input

STEPS = (32, 128, 512)
TOLERANCE = 0.05


def summarise(errors: list[float], tolerance: float = TOLERANCE) -> dict:
    v = np.array(errors)
    return {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "share_under_tolerance": float((v < tolerance).mean()),
            "tolerance": tolerance}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=artifacts_dir())
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    z = np.load(args.artifacts / "app_data" / "p1_test.npz")
    idx = np.random.default_rng(args.seed).choice(len(z["images"]), args.n, replace=False)
    out = {"patch_ids": [f"p1-test-{int(i):06d}" for i in idx], "seed": args.seed, "models": {}, "steps": {}}
    for variant in ("poisoned", "clean"):
        net, card = load_for_inference(args.artifacts / "models" / f"p1_{variant}", "cpu")
        out["models"][variant] = card.model_hash
        errs = {s: [] for s in args.steps}
        for i in idx:
            x = to_model_input(torch.from_numpy(z["images"][i]))
            with torch.no_grad():
                tgt = int(net(x).argmax(1))
            for s in args.steps:
                last = None
                for last in integrated_gradients(net, x, tgt, steps=s, chunk=64):
                    pass
                if last.completeness_error is not None:
                    errs[s].append(last.completeness_error)
        out["steps"][variant] = {str(s): summarise(e) for s, e in errs.items()}
        print(variant, {s: round(out["steps"][variant][str(s)]["median"], 4) for s in args.steps})
    out["generated_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    dest = args.artifacts / "results" / "ig_completeness.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
