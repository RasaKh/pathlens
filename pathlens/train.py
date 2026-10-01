"""Training engine (architecture doc, "Training engine"): trains and evaluates one run from its config.

Phase 1 and Phase 2 share this loop. The CPU path is tested here. The CUDA paths (bf16 autocast,
torch.compile, pinned transfers) have never been run on this box (no GPU); treat them as untested.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .augment import augment
from .config import RunCfg
from .evaluate import binary_metrics, multiclass_metrics
from .identity import config_hash, file_sha256, run_id, seed_all
from .model import ModelCard, build_model, param_groups
from .sampler import DeviceSampler
from .tensors import to_model_input

log = logging.getLogger(__name__)

LOG_EVERY = 50
SPIKE_FACTOR = 3.0              # a loss above 3x its running mean marks the run unstable
SPIKE_WARMUP = 100              # steps before spikes are judged (the running mean needs time to settle)
EMA_DECAY = 0.98
PREDICT_BATCH = 2048


class Interrupted(Exception):
    """Raised after the checkpoint is saved when stop_flag() asks the run to stop."""


@dataclass(frozen=True)
class RunInputs:
    images: Tensor          # uint8 [N,H,W,3], on CPU or GPU
    labels: Tensor          # int64 [N], same device
    train_rows: Tensor      # int64 indices into images
    test_rows: Tensor
    row_ids: np.ndarray     # stable ids for predictions (p1 image_id or p2 row)
    class_names: tuple[str, ...] = field(default=())    # for the model card (optional, see BUILD_LOG)
    pred_extra: Mapping[str, np.ndarray] | None = None  # extra predictions.parquet columns, aligned with `images`


@dataclass(frozen=True)
class RunResult:
    run_id: str
    config_hash: str
    model_hash: str
    wall_seconds: float
    images_per_second: float
    metrics_path: Path


@torch.inference_mode()
def predict(model: nn.Module, images: Tensor, rows: Tensor, device, batch: int = PREDICT_BATCH,
            bf16: bool = False) -> Tensor:
    """Softmax probabilities, float32 [len(rows), K], for `rows` of `images`, in the order given."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        x = to_model_input(images[rows[i:i + batch]].to(device)).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(x)
        out.append(logits.float().softmax(1).cpu())
    model.train(was_training)
    return torch.cat(out) if out else torch.empty(0, 0)


def count_params(model: nn.Module) -> dict[str, int]:
    conv = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))
    return {"conv_weights": conv, "total": sum(p.numel() for p in model.parameters())}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _done_result(out_dir: Path, cfg: RunCfg) -> RunResult | None:
    done = out_dir / "DONE"
    if not done.exists():
        return None
    d = json.loads(done.read_text())
    if d["config_hash"] != config_hash(cfg):
        raise ValueError(f"{out_dir}: DONE carries config_hash {d['config_hash']} but the config now hashes "
                         f"to {config_hash(cfg)}; the config changed under the same run id")
    m = json.loads((out_dir / "metrics.json").read_text())
    return RunResult(run_id(cfg), d["config_hash"], d["model_hash"], d["wall_seconds"],
                     m["images_per_second"], out_dir / "metrics.json")


class _Stepper:
    """The model's forward, optionally compiled. If compilation fails the first time it is used (forward or backward),
    the run logs once and carries on in eager mode. Never exercised on a GPU yet."""

    def __init__(self, model: nn.Module, compile_: bool) -> None:
        self.model, self.fn, self.compiled = model, model, False
        if compile_:
            try:
                self.fn, self.compiled = torch.compile(model), True
            except Exception as e:                      # noqa: BLE001
                log.warning("torch.compile failed (%s); using eager", e)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x)

    def disable(self, why: Exception) -> None:
        log.warning("compiled step failed (%s); using eager from here", str(why)[:200])
        self.fn, self.compiled = self.model, False


def _train_step(stepper: _Stepper, loss_fn, opt, x: Tensor, y: Tensor, use_bf16: bool, device, first: bool) -> Tensor:
    """One forward/backward. On the first step of a compiled run a failure falls back to eager and is retried."""
    def go() -> Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = loss_fn(stepper(x), y)
        loss.backward()
        return loss

    if first and stepper.compiled:
        try:
            return go()
        except Exception as e:                          # noqa: BLE001
            stepper.disable(e)
            opt.zero_grad(set_to_none=True)
    return go()


def measure_throughput(cfg: RunCfg, images: Tensor, labels: Tensor, rows: Tensor, device, warmup: int = 20,
                       steps: int = 50) -> dict:
    """Training speed of `cfg` on real batches: `warmup` untimed steps (compilation, cuDNN), then `steps` timed ones.
    Returns images_per_second and, on CUDA, the peak memory allocated (GB). Leaves no files behind."""
    device = torch.device(device)
    seed_all(cfg.seed)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    model = build_model(cfg.model).to(device, memory_format=torch.channels_last)
    stepper = _Stepper(model, cfg.compile)
    oc = cfg.optim
    opt = torch.optim.AdamW(param_groups(model), lr=oc.peak_lr, weight_decay=oc.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=oc.label_smoothing)
    sampler = DeviceSampler(rows.to(device), oc.batch_size, g, drop_last=True)
    use_bf16 = device.type == "cuda" and cfg.amp_dtype == "bf16"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    done, t0, seen = 0, 0.0, 0
    while done < warmup + steps:
        for idx in sampler.epoch():
            if done == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0, seen = time.perf_counter(), 0
            x = ((augment(images[idx], cfg.aug, g) - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)
            _train_step(stepper, loss_fn, opt, x, labels[idx], use_bf16, device, first=(done == 0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            seen += len(idx)
            if done >= warmup + steps:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    return {"images_per_second": seen / dt if dt > 0 else 0.0,
            "peak_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            "compiled": stepper.compiled}


def _save_ckpt(path: Path, model, opt, sched, g, step: int, epoch: int, ema: float, elapsed: float,
               unstable: bool) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "g": g.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "step": step, "epoch": epoch, "ema": ema, "elapsed": elapsed,
                "unstable": unstable}, tmp)
    os.replace(tmp, path)


def train_run(cfg: RunCfg, inputs: RunInputs, device, out_dir: Path, stop_flag: Callable[[], bool]
              ) -> RunResult:
    import pandas as pd

    out_dir = Path(out_dir)
    cached = _done_result(out_dir, cfg)
    if cached is not None:
        return cached
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    cuda = device.type == "cuda"
    chash, rid = config_hash(cfg), run_id(cfg)
    _write_json(out_dir / "config.json", {**dataclasses.asdict(cfg), "config_hash": chash})
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)

    seed_all(cfg.seed)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    model = build_model(cfg.model).to(device, memory_format=torch.channels_last)
    stepper = _Stepper(model, cfg.compile)
    oc = cfg.optim
    opt = torch.optim.AdamW(param_groups(model), lr=oc.peak_lr, betas=(0.9, 0.999), weight_decay=oc.weight_decay)
    sampler = DeviceSampler(inputs.train_rows.to(device), oc.batch_size, g, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=oc.peak_lr, total_steps=oc.epochs * len(sampler),
                                                pct_start=oc.pct_start, div_factor=25, final_div_factor=1e4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=oc.label_smoothing)
    images, labels = inputs.images.to(device), inputs.labels.to(device)

    step, start_epoch, ema, elapsed, unstable = 0, 0, 0.0, 0.0, False
    ckpt = out_dir / "ckpt.pt"
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        g.set_state(c["g"].cpu())
        torch.set_rng_state(c["torch_rng"].cpu())           # dropout draws from the global generator
        if c["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in c["cuda_rng"]])
        step, start_epoch, ema, elapsed, unstable = c["step"], c["epoch"], c["ema"], c["elapsed"], c["unstable"]
        log.info("%s: resumed at epoch %d, step %d", rid, start_epoch, step)

    use_bf16 = cuda and cfg.amp_dtype == "bf16"
    start_step = step
    t0 = time.perf_counter()
    seen_this_session = 0
    t_log, seen_at_log = t0, 0
    model.train()
    with open(out_dir / "train_log.jsonl", "a") as logf:
        for epoch in range(start_epoch, oc.epochs):
            for idx in sampler.epoch():
                x = augment(images[idx], cfg.aug, g)
                x = ((x - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)
                loss = _train_step(stepper, loss_fn, opt, x, labels[idx], use_bf16, device, first=(step == start_step))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                seen_this_session += len(idx)
                lv = float(loss.detach())
                if step > SPIKE_WARMUP and ema > 0 and lv > SPIKE_FACTOR * ema:
                    unstable = True
                ema = lv if step == 1 or ema == 0 else EMA_DECAY * ema + (1 - EMA_DECAY) * lv
                if step % LOG_EVERY == 0:
                    now = time.perf_counter()
                    logf.write(json.dumps({"step": step, "epoch": epoch, "loss_ema": round(ema, 6),
                                           "lr": sched.get_last_lr()[0],
                                           "img_s": round((seen_this_session - seen_at_log) / (now - t_log), 1)}) + "\n")
                    logf.flush()
                    t_log, seen_at_log = now, seen_this_session
            _save_ckpt(ckpt, model, opt, sched, g, step, epoch + 1, ema, elapsed + time.perf_counter() - t0, unstable)
            # stop_flag is read at epoch boundaries (the doc reads it every step): a mid-epoch checkpoint
            # would either replay part of the epoch or run the one-cycle schedule past its last step.
            if epoch + 1 < oc.epochs and stop_flag():
                raise Interrupted(f"{rid} stopped after epoch {epoch + 1}, step {step}")
    train_seconds = time.perf_counter() - t0
    wall = elapsed + train_seconds

    probs = predict(model, images, inputs.test_rows.to(device), device, bf16=use_bf16).numpy().astype(np.float32)
    y = labels[inputs.test_rows.to(device)].cpu().numpy()
    pred = probs.argmax(1)
    ids = np.asarray(inputs.row_ids)[inputs.test_rows.cpu().numpy()]
    if cfg.phase == "p1":
        df = pd.DataFrame({"image_id": ids, "label": y, "pred": pred})
        for k in range(probs.shape[1]):
            df[f"prob_{k}"] = probs[:, k]
        df["condition"] = "clean"
        metrics = multiclass_metrics(y, probs)
    else:
        df = pd.DataFrame({"row": ids, "label": y, "pred": pred, "p_tumour": probs[:, 1]})
        for name, col in (inputs.pred_extra or {}).items():      # hospital, slide_key, patient joined from meta
            df[name] = np.asarray(col)[inputs.test_rows.cpu().numpy()]
        metrics = binary_metrics(y, probs[:, 1])
    df.to_parquet(out_dir / "predictions.parquet")

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out_dir / "model.pt")
    card = ModelCard.build(cfg.model, int(images.shape[1]), inputs.class_names, state, rid)
    card.write(out_dir / "model_card.json")
    ips = seen_this_session / train_seconds if train_seconds > 0 and seen_this_session else 0.0
    _write_json(out_dir / "metrics.json", {
        "run_id": rid, "config_hash": chash, "model_hash": card.model_hash, "n_test": len(y),
        "metrics": metrics, "unstable": unstable, "steps": step, "final_loss_ema": ema,
        "params": count_params(model), "wall_seconds": wall, "images_per_second": ips})
    files = ["config.json", "train_log.jsonl", "model.pt", "model_card.json", "predictions.parquet", "metrics.json"]
    _write_json(out_dir / "DONE", {"config_hash": chash, "model_hash": card.model_hash, "wall_seconds": wall,
                                   "sha256": {f: file_sha256(out_dir / f) for f in files}})
    ckpt.unlink(missing_ok=True)
    return RunResult(rid, chash, card.model_hash, wall, ips, out_dir / "metrics.json")
