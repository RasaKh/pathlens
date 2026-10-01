"""Frozen run configuration (architecture doc, "Configuration and run identity")."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Phase = Literal["p1", "p2"]
Level = Literal["none", "patch", "slide", "patient", "hospital"]
Variant = Literal["clean", "poisoned", "plain", "aug"]
AmpDtype = Literal["bf16", "fp32"]

_PHASES = ("p1", "p2")
_LEVELS = ("none", "patch", "slide", "patient", "hospital")
_VARIANTS = ("clean", "poisoned", "plain", "aug")
_AMP = ("bf16", "fp32")


@dataclass(frozen=True)
class ModelCfg:
    widths: tuple[int, int, int, int]   # (16,32,64,128) P1, (32,64,128,256) P2
    num_classes: int                    # 9 or 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if len(self.widths) != 4 or any(w <= 0 for w in self.widths):
            raise ValueError(f"widths must be 4 positive ints, got {self.widths}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class OptimCfg:
    epochs: int                         # P1 15, P2 6
    batch_size: int                     # P1 256, P2 512
    peak_lr: float                      # P1 2e-3, P2 3e-3
    weight_decay: float = 0.05
    pct_start: float = 0.15
    label_smoothing: float = 0.05


@dataclass(frozen=True)
class AugCfg:
    flips: bool = True
    rot90: bool = True
    stain: bool = False
    stain_scale: tuple[float, float] = (0.95, 1.05)
    # Doc default was (-0.05, 0.05); calibration on real PathMNIST at 64 px showed that
    # shift tints white background harder than it jitters tissue, and hed_jitter now
    # applies it only inside the tissue mask. Halved accordingly (BUILD_LOG Session 5).
    stain_shift: tuple[float, float] = (-0.025, 0.025)


@dataclass(frozen=True)
class RunCfg:
    phase: Phase
    level: Level
    fold: int                           # -1 for P1
    variant: Variant
    seed: int
    model: ModelCfg
    optim: OptimCfg
    aug: AugCfg
    compile: bool = True
    amp_dtype: AmpDtype = "bf16"

    def __post_init__(self) -> None:
        for name, value, allowed in (("phase", self.phase, _PHASES), ("level", self.level, _LEVELS),
                                     ("variant", self.variant, _VARIANTS), ("amp_dtype", self.amp_dtype, _AMP)):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} not in {allowed}")
        if self.phase == "p1" and self.fold != -1:
            raise ValueError("Phase 1 runs use fold=-1")
        if self.phase == "p2" and not 0 <= self.fold <= 4:
            raise ValueError("Phase 2 folds are 0..4")


@dataclass(frozen=True)
class SessionCfg:
    budget_hours: float
    probe_steps: int = 200
    allow_partial: bool = False
    min_gpu_gb: float = 20.0          # the doc's floor for free GPU memory; 16 GB cards need it lowered explicitly
    runs: tuple[RunCfg, ...] = ()


# ---------------------------------------------------------------- JSON round trip

def to_dict(cfg: Any) -> dict[str, Any]:
    """Plain-dict form (tuples stay tuples; json/orjson turn them into lists)."""
    return dataclasses.asdict(cfg)


def _tup(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def run_cfg_from_dict(d: dict[str, Any]) -> RunCfg:
    m, o, a = dict(d["model"]), dict(d["optim"]), dict(d["aug"])
    m["widths"] = tuple(m["widths"])
    for k in ("stain_scale", "stain_shift"):
        if k in a:
            a[k] = _tup(a[k])
    rest = {k: v for k, v in d.items() if k not in ("model", "optim", "aug", "config_hash")}
    return RunCfg(model=ModelCfg(**m), optim=OptimCfg(**o), aug=AugCfg(**a), **rest)


def session_cfg_from_dict(d: dict[str, Any]) -> SessionCfg:
    runs = tuple(run_cfg_from_dict(r) for r in d.get("runs", ()))
    return SessionCfg(**{**{k: v for k, v in d.items() if k != "runs"}, "runs": runs})


# ---------------------------------------------------------------- queue expansion

P1_MODEL = ModelCfg(widths=(16, 32, 64, 128), num_classes=9)
P2_MODEL = ModelCfg(widths=(32, 64, 128, 256), num_classes=2)

# Priority order of levels (doc, "Queue order"); within a run group the live fold (2) goes first.
_LEVEL_PRIORITY = {"hospital": 0, "patch": 1, "slide": 2, "patient": 3}
_LIVE_FOLD = 2


def _fold_order(n_folds: int = 5) -> list[int]:
    return [_LIVE_FOLD] + [f for f in range(n_folds) if f != _LIVE_FOLD]


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def expand_queue(yaml_path: str | Path, levels_present: int = 4) -> tuple[RunCfg, ...]:
    """Expand a compact queue YAML into the explicit, ordered run list.

    Order: by queue entry as written, and within an entry by level priority then live fold first.
    `levels_present == 3` drops the patient level (patient folds would alias slide folds).
    """
    spec = yaml.safe_load(Path(yaml_path).read_text())
    model = ModelCfg(**{**dataclasses.asdict(P2_MODEL), **{
        k: (tuple(v) if k == "widths" else v) for k, v in spec.get("model", {}).items()}})
    ospec = spec.get("optim", {})
    optim = OptimCfg(epochs=ospec.get("epochs", 6), batch_size=ospec.get("batch_size", 512),
                     peak_lr=ospec.get("peak_lr", 3e-3),
                     **{k: v for k, v in ospec.items() if k not in ("epochs", "batch_size", "peak_lr")})
    compile_ = spec.get("compile", True)
    amp = spec.get("amp_dtype", "bf16")

    runs: list[RunCfg] = []
    for entry in spec["queue"]:
        levels = sorted(_as_list(entry["levels"]), key=lambda lv: _LEVEL_PRIORITY[lv])
        if levels_present == 3:
            levels = [lv for lv in levels if lv != "patient"]
        folds = _fold_order() if entry.get("folds", "all") == "all" else _as_list(entry["folds"])
        for variant in _as_list(entry["variant"]):
            for seed in _as_list(entry.get("seeds", [0])):
                for level in levels:
                    for fold in folds:
                        runs.append(RunCfg(
                            phase="p2", level=level, fold=fold, variant=variant, seed=seed,
                            model=model, optim=optim,
                            aug=AugCfg(stain=(variant == "aug")), compile=compile_, amp_dtype=amp))
    return tuple(runs)


def p1_run_cfg(variant: Literal["clean", "poisoned"], seed: int = 0, *, compile: bool = False) -> RunCfg:
    """Phase 1 twin (CPU: no compile, fp32 unless the caller overrides)."""
    return RunCfg(phase="p1", level="none", fold=-1, variant=variant, seed=seed, model=P1_MODEL,
                  optim=OptimCfg(epochs=15, batch_size=256, peak_lr=2e-3), aug=AugCfg(),
                  compile=compile, amp_dtype="fp32")
