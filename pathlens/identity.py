"""Identity and hashing (architecture doc, "Identity and hashing")."""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson
import torch

from .config import RunCfg

_BLOCK = 8 * 1024 * 1024


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, no whitespace, floats as written (shortest round-trip repr)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def config_hash(cfg: RunCfg) -> str:
    """sha256(canonical_json(asdict(cfg)))[:12]."""
    return hashlib.sha256(canonical_json(dataclasses.asdict(cfg))).hexdigest()[:12]


def run_id(cfg: RunCfg) -> str:
    return f"{cfg.phase}-{cfg.level}-f{cfg.fold}-{cfg.variant}-s{cfg.seed}"


def model_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """sha256 over each tensor in sorted key order: name bytes, dtype string, shape, raw bytes.

    Independent of torch.save details. Handles 0-d tensors (BatchNorm's num_batches_tracked),
    empty tensors and bfloat16.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:12]


def file_sha256(path: str | Path) -> str:
    """Streams in 8 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_BLOCK):
            h.update(block)
    return h.hexdigest()
