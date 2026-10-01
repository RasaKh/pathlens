"""GPU orchestration (architecture doc, "GPU orchestration: run_all").

    python -m pathlens.run_all --config configs/phase2_gpu.yaml            # the one command run on the rented machine
    python -m pathlens.run_all --config configs/phase2_gpu.yaml --probe-only   # measure speed, print the projection, stop
    python -m pathlens.run_all --config <yaml> --dry-run                   # CPU, tiny runs, whole flow end to end

Exit codes: 0 all done; 2 budget refused before training (or the queue was cut short by the budget watchdog, or
the GPU is too small); 3 stopped by a signal (run the same command again to resume); 4 data verification failed;
5 a run failed (traceback in <runs>/<run_id>/error.txt; later runs continue).
Everything GPU-specific (bf16, torch.compile, device transfers, memory checks) is untested until it first runs on one.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import signal
import subprocess
import tarfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import RunCfg, SessionCfg, expand_queue
from .data_p2 import P2Store
from .identity import config_hash, file_sha256, run_id
from .model import load_for_inference
from .paths import home
from .splits import fold_rows
from .train import Interrupted, RunInputs, measure_throughput, train_run

EXIT_OK, EXIT_BUDGET, EXIT_SIGNAL, EXIT_DATA, EXIT_RUN_FAILED = 0, 2, 3, 4, 5
OVERHEAD_S = 600.0                      # the doc adds 10 minutes for features and packing
MAP_PER_CELL, PROBE_PER_CELL = 600, 2000          # 5 hospitals x 2 labels x this many patches
DRY_TRAIN_ROWS, DRY_TEST_ROWS = 96, 64
LIVE = {"plain": ("hospital", 2, "plain", 0), "aug": ("hospital", 2, "aug", 0)}      # the two live Phase 2 models
VERIFY_FILES = ("patches_u8.npy", "meta.parquet", "splits.parquet")


# ------------------------------------------------------------------ session setup

def load_session(config_path: Path, p2_dir: Path) -> SessionCfg:
    """Read the YAML, take levels_present from the splits entry of data_manifest.json, expand the queue."""
    spec = yaml.safe_load(Path(config_path).read_text())
    man = json.loads((p2_dir / "data_manifest.json").read_text())
    levels_present = int(man["files"]["splits.parquet"]["levels_present"])
    return SessionCfg(budget_hours=float(spec["budget_hours"]), probe_steps=int(spec.get("probe_steps", 200)),
                      allow_partial=bool(spec.get("allow_partial", False)), min_gpu_gb=float(spec.get("min_gpu_gb", 20.0)),
                      runs=expand_queue(config_path, levels_present))


def dry_run_cfg(cfg: RunCfg) -> RunCfg:
    """One epoch, small batches, no compile, fp32: the same flow, a fraction of the work."""
    oc = dataclasses.replace(cfg.optim, epochs=1, batch_size=min(cfg.optim.batch_size, 32))
    return dataclasses.replace(cfg, optim=oc, compile=False, amp_dtype="fp32")


def verify_data(p2_dir: Path) -> list[str]:
    """sha256 of patches_u8.npy, meta.parquet, splits.parquet against data_manifest.json. Returns the problems found."""
    mf = p2_dir / "data_manifest.json"
    if not mf.exists():
        return [f"{mf} is missing"]
    man = json.loads(mf.read_text())
    problems = []
    for name in VERIFY_FILES:
        rec = man["files"].get(name)
        if rec is None:
            problems.append(f"{name} is not listed in the manifest")
        elif not (p2_dir / name).exists():
            problems.append(f"{name} is missing")
        elif file_sha256(p2_dir / name) != rec["sha256"]:
            problems.append(f"{name} does not match its manifest sha256")
    return problems


@dataclass
class Phase2Data:
    images: torch.Tensor
    labels: torch.Tensor
    meta: pd.DataFrame
    splits: pd.DataFrame
    extra: dict[str, np.ndarray]


def load_phase2(p2_dir: Path, device) -> Phase2Data:
    store = P2Store(p2_dir.parent, mmap=True)
    images = store.to_device(device)
    meta = store.meta
    splits = pd.read_parquet(p2_dir / "splits.parquet")
    if not (splits["row"].to_numpy() == meta["row"].to_numpy()).all():
        raise ValueError("splits.parquet and meta.parquet are not row-aligned")
    return Phase2Data(images, torch.from_numpy(meta["label"].to_numpy().astype(np.int64)).to(device), meta, splits,
                      {"hospital": meta["hospital"].to_numpy(), "slide_key": meta["slide_key"].to_numpy(),
                       "patient": meta["patient"].to_numpy()})


def split_rows(cfg: RunCfg, data: Phase2Data, dry: bool) -> tuple[np.ndarray, np.ndarray]:
    train, test = fold_rows(data.splits, cfg.level, cfg.fold)
    if dry:
        rng = np.random.default_rng(0)
        train = np.sort(rng.choice(train, min(DRY_TRAIN_ROWS, len(train)), replace=False))
        test = np.sort(rng.choice(test, min(DRY_TEST_ROWS, len(test)), replace=False))
    return train, test


def inputs_for(cfg: RunCfg, data: Phase2Data, dry: bool, device) -> RunInputs:
    train, test = split_rows(cfg, data, dry)
    return RunInputs(images=data.images, labels=data.labels, train_rows=torch.from_numpy(train).to(device),
                     test_rows=torch.from_numpy(test).to(device), row_ids=data.meta["row"].to_numpy(),
                     class_names=("normal", "tumour"), pred_extra=data.extra)


def estimate_seconds(cfg: RunCfg, n_train: int, n_test: int, img_s: float) -> float:
    """The doc's formula: training epochs, plus prediction at about three times training speed."""
    return cfg.optim.epochs * n_train / img_s + n_test / (3 * img_s)


# ------------------------------------------------------------------ features for the map and the probes

def sample_rows(meta: pd.DataFrame, per_cell: int, seed: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """per_cell patches from each (hospital, label) cell, seeded, never from `exclude`. Sorted row numbers."""
    rng = np.random.default_rng(seed)
    out = []
    for (_, _), g in meta.groupby(["hospital", "label"], sort=True):
        rows = g["row"].to_numpy()
        if exclude is not None:
            rows = rows[~np.isin(rows, exclude)]
        out.append(rng.choice(rows, size=min(per_cell, len(rows)), replace=False))
    return np.sort(np.concatenate(out))


@torch.inference_mode()
def pooled_features(model, images: torch.Tensor, rows: np.ndarray, device, batch: int = 512) -> np.ndarray:
    from .tensors import to_model_input
    if len(rows) == 0:
        return np.zeros((0, model.cfg.widths[3]), dtype=np.float32)
    outs = []
    for a in range(0, len(rows), batch):
        idx = torch.from_numpy(rows[a:a + batch]).to(device)
        x = to_model_input(images[idx]).contiguous(memory_format=torch.channels_last)
        outs.append(model.forward_features(x)[1].float().cpu())
    return torch.cat(outs).numpy()


def extract_features(data: Phase2Data, runs_root: Path, out_dir: Path, device, map_cell: int, probe_cell: int, log) -> bool:
    """map_{plain,aug}.npy (L2-normalised) and probe_{plain,aug}.npy (raw), float16 [n,256], plus *_rows.npy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    map_rows = sample_rows(data.meta, map_cell, seed=1)
    probe_rows = sample_rows(data.meta, probe_cell, seed=2, exclude=map_rows)
    np.save(out_dir / "map_rows.npy", map_rows)
    np.save(out_dir / "probe_rows.npy", probe_rows)
    for name, (level, fold, variant, seed) in LIVE.items():
        rid = f"p2-{level}-f{fold}-{variant}-s{seed}"
        if not (runs_root / rid / "DONE").exists():
            log(f"features: {rid} is not finished; skipping the {name} features")
            return False
        model, _ = load_for_inference(runs_root / rid, device)
        m = pooled_features(model, data.images, map_rows, device)
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        np.save(out_dir / f"map_{name}.npy", m.astype(np.float16))
        np.save(out_dir / f"probe_{name}.npy", pooled_features(model, data.images, probe_rows, device).astype(np.float16))
        log(f"features: {name} done ({len(map_rows)} map rows, {len(probe_rows)} probe rows)")
    return True


# ------------------------------------------------------------------ manifest and packing

def _driver() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                              text=True, check=True, timeout=20).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def session_files(root: Path, runs_root: Path, analysis_dir: Path) -> list[Path]:
    files = [p for p in runs_root.rglob("*") if p.is_file() and p.name not in ("ckpt.pt", "ckpt.tmp", "session_manifest.json")]
    files += [p for p in analysis_dir.rglob("*") if p.is_file()] if analysis_dir.exists() else []
    return sorted(files)


def write_session_manifest(root: Path, runs_root: Path, analysis_dir: Path, info: dict) -> Path:
    entries = {str(p.relative_to(root)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
               for p in session_files(root, runs_root, analysis_dir)}
    man = {"files": entries, **info}
    path = runs_root / "session_manifest.json"
    path.write_text(json.dumps(man, indent=2, sort_keys=True))
    return path


def pack(root: Path, runs_root: Path, analysis_dir: Path, out: Path) -> str:
    """artifacts_phase2.tar.zst (zstd level 10): runs/ without ckpt.pt, analysis_inputs/, the manifest and the log."""
    import zstandard as zstd

    files = session_files(root, runs_root, analysis_dir) + [runs_root / "session_manifest.json"]
    with open(out, "wb") as fh, zstd.ZstdCompressor(level=10).stream_writer(fh) as zw, tarfile.open(fileobj=zw, mode="w|") as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(root)), recursive=False)
    return file_sha256(out)


# ------------------------------------------------------------------ the session

class Session:
    def __init__(self, runs_root: Path, analysis_dir: Path) -> None:
        self.runs_root, self.analysis_dir = runs_root, analysis_dir
        runs_root.mkdir(parents=True, exist_ok=True)
        self.path = runs_root / "session.log"

    def log(self, msg: str) -> None:
        line = f"{dt.datetime.now(dt.UTC).strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


def run_session(config_path: Path, *, dry_run: bool = False, probe_only: bool = False, device: str | None = None,
                runs_root: Path | None = None, p2_dir: Path | None = None, analysis_dir: Path | None = None,
                stop_flag: Callable[[], bool] = lambda: False) -> int:
    root = home()
    p2_dir = p2_dir or root / "data" / "p2"
    runs_root = runs_root or (root / ("runs_dry" if dry_run else "runs"))
    analysis_dir = analysis_dir or root / ("analysis_inputs_dry" if dry_run else "analysis_inputs")
    s = Session(runs_root, analysis_dir)
    t_session = time.perf_counter()
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    cfg = load_session(config_path, p2_dir)
    runs = tuple(dry_run_cfg(r) for r in cfg.runs) if dry_run else cfg.runs
    (runs_root / "queue.json").write_text(json.dumps([dataclasses.asdict(r) for r in runs], indent=2))
    s.log(f"queue: {len(runs)} runs; budget {cfg.budget_hours} h{' (dry run)' if dry_run else ''}")

    dev = torch.device(device or ("cpu" if dry_run else "cuda"))
    gpu_name = None
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            s.log("no CUDA device available")
            return EXIT_BUDGET
        gpu_name = torch.cuda.get_device_name(dev)
        free, total = torch.cuda.mem_get_info(dev)
        s.log(f"gpu: {gpu_name}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free, bf16 {torch.cuda.is_bf16_supported()}")
        if free / 2**30 < cfg.min_gpu_gb:
            s.log(f"refused: {free / 2**30:.1f} GB free is below min_gpu_gb = {cfg.min_gpu_gb} (set it in the YAML to override)")
            return EXIT_BUDGET
    else:
        s.log(f"device: {dev}")

    problems = verify_data(p2_dir)
    if problems:
        for p in problems:
            s.log(f"data verification failed: {p}")
        return EXIT_DATA
    s.log("data verified against data_manifest.json")

    t0 = time.perf_counter()
    data = load_phase2(p2_dir, dev)
    s.log(f"images on {dev}: {tuple(data.images.shape)} in {time.perf_counter() - t0:.0f} s")

    first = runs[0]
    tr0, _ = split_rows(first, data, dry_run)
    warm, steps = (2, 3) if dry_run else (20, cfg.probe_steps)
    probe = measure_throughput(first, data.images, data.labels, torch.from_numpy(tr0).to(dev), dev, warmup=warm, steps=steps)
    img_s = probe["images_per_second"]
    pending = [r for r in runs if not _is_done(runs_root, r)]
    est = {run_id(r): estimate_seconds(r, *(len(a) for a in split_rows(r, data, dry_run)), img_s) for r in pending}
    projected = sum(est.values()) + OVERHEAD_S
    budget_s = cfg.budget_hours * 3600
    s.log(f"probe: {img_s:.0f} img/s, peak {probe['peak_gb']} GB, compiled {probe['compiled']}; projected "
          f"{projected / 3600:.2f} h for {len(pending)} pending runs, budget {cfg.budget_hours} h")
    if probe_only:
        return EXIT_OK
    if projected > 0.9 * budget_s and not cfg.allow_partial and not dry_run:
        fit, acc = [], OVERHEAD_S
        for r in pending:
            if acc + est[run_id(r)] > 0.9 * budget_s:
                break
            acc += est[run_id(r)]
            fit.append(run_id(r))
        s.log(f"refused: projected time exceeds 90% of the budget. Runs that fit: {len(fit)} of {len(pending)}: {fit}")
        return EXIT_BUDGET

    failed, cut_short, wall_times = [], False, {}
    for r in runs:
        rid = run_id(r)
        if _is_done(runs_root, r):
            s.log(f"{rid}: already done, skipped")
            continue
        elapsed = time.perf_counter() - t_session
        if elapsed + est[rid] > budget_s and not dry_run:
            s.log(f"watchdog: {rid} would exceed the budget ({elapsed / 3600:.2f} h used); stopping the queue")
            cut_short = True
            break
        try:
            res = train_run(r, inputs_for(r, data, dry_run, dev), dev, runs_root / rid, stop_flag)
        except Interrupted as e:
            s.log(f"stopped: {e}; run the same command again to resume")
            return EXIT_SIGNAL
        except Exception:                                   # noqa: BLE001  (one failed run must not end the session)
            (runs_root / rid).mkdir(parents=True, exist_ok=True)
            (runs_root / rid / "error.txt").write_text(traceback.format_exc())
            s.log(f"{rid}: FAILED (see error.txt)")
            failed.append(rid)
            continue
        m = json.loads(res.metrics_path.read_text())
        wall_times[rid] = res.wall_seconds
        s.log(f"{rid}: {res.wall_seconds:.0f} s, held-out AUROC {m['metrics']['auroc']:.4f}, model {res.model_hash}")

    features_done = extract_features(data, runs_root, analysis_dir, dev, 4 if dry_run else MAP_PER_CELL,
                                     10 if dry_run else PROBE_PER_CELL, s.log)
    info = {"gpu": gpu_name, "driver": _driver(), "cuda": torch.version.cuda, "torch": torch.__version__,
            "probe": probe, "per_run_wall_seconds": wall_times, "failed_runs": failed, "queue_cut_short": cut_short,
            "features_done": features_done, "dry_run": dry_run, "started": started,
            "session_seconds": time.perf_counter() - t_session}
    write_session_manifest(root, runs_root, analysis_dir, info)
    digest = pack(root, runs_root, analysis_dir, root / ("artifacts_phase2_dry.tar.zst" if dry_run else "artifacts_phase2.tar.zst"))
    s.log(f"packed; sha256 {digest}")
    if failed:
        s.log(f"{len(failed)} run(s) failed: {failed}")
        return EXIT_RUN_FAILED
    if cut_short:
        s.log("the queue was cut short by the budget")
        return EXIT_BUDGET
    s.log("SAFE TO SHUT DOWN (after scripts/verify_artifacts.py passes locally)")
    return EXIT_OK


def _is_done(runs_root: Path, cfg: RunCfg) -> bool:
    done = runs_root / run_id(cfg) / "DONE"
    return done.exists() and json.loads(done.read_text())["config_hash"] == config_hash(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-only", action="store_true", help="measure speed, print the projection, exit 0")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"))
    args = ap.parse_args()
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    raise SystemExit(run_session(args.config, dry_run=args.dry_run, probe_only=args.probe_only, device=args.device,
                                 stop_flag=lambda: stop["flag"]))


if __name__ == "__main__":
    main()
