import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from pathlens.config import (
    P1_MODEL,
    AugCfg,
    SessionCfg,
    expand_queue,
    p1_run_cfg,
    run_cfg_from_dict,
    session_cfg_from_dict,
)
from pathlens.identity import canonical_json, config_hash, run_id

YAML = "configs/phase2_gpu.yaml"


def test_json_round_trip_preserves_config_and_hash():
    cfg = p1_run_cfg("poisoned", seed=3)
    back = run_cfg_from_dict(json.loads(canonical_json(cfg)))
    assert back == cfg
    assert config_hash(back) == config_hash(cfg)


def test_session_round_trip():
    runs = expand_queue(YAML)[:3]
    s = SessionCfg(budget_hours=4, runs=runs)
    back = session_cfg_from_dict(json.loads(canonical_json(s)))
    assert back == s


def test_hash_stable_across_processes():
    code = ("from pathlens.config import p1_run_cfg; from pathlens.identity import config_hash;"
            "print(config_hash(p1_run_cfg('clean')))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout.strip()
            for seed in (0, 1, 12345)}
    assert outs == {config_hash(p1_run_cfg("clean"))}


def test_hash_sensitive_to_every_level():
    base = p1_run_cfg("clean")
    variants = [replace(base, seed=1), replace(base, variant="poisoned"),
                replace(base, aug=AugCfg(stain=True)), replace(base, compile=True),
                replace(base, optim=replace(base.optim, peak_lr=1e-3)),
                replace(base, model=replace(base.model, dropout=0.3))]
    hashes = {config_hash(v) for v in variants} | {config_hash(base)}
    assert len(hashes) == len(variants) + 1
    assert len(config_hash(base)) == 12


def test_validation():
    base = p1_run_cfg("clean")
    for bad in ({"phase": "p3"}, {"level": "galaxy"}, {"variant": "x"}, {"amp_dtype": "fp8"}, {"fold": 0}):
        with pytest.raises(ValueError):
            replace(base, **bad)


def test_p1_defaults_match_doc():
    c = p1_run_cfg("clean")
    assert (c.fold, c.level, c.model, c.optim.epochs, c.optim.batch_size, c.optim.peak_lr) == \
        (-1, "none", P1_MODEL, 15, 256, 2e-3)
    assert (c.optim.weight_decay, c.optim.pct_start, c.optim.label_smoothing) == (0.05, 0.15, 0.05)
    assert c.model.dropout == 0.2 and c.amp_dtype == "fp32" and not c.compile


def test_expand_queue_is_the_35_run_list_in_priority_order():
    runs = expand_queue(YAML)
    assert len(runs) == 35
    ids = [run_id(r) for r in runs]
    assert len(set(ids)) == 35
    assert ids[0] == "p2-hospital-f2-plain-s0"                       # live model first
    assert ids[:5] == [f"p2-hospital-f{f}-plain-s0" for f in (2, 0, 1, 3, 4)]
    order = [(r.level, r.variant, r.seed) for r in runs]
    first_seen = [k for i, k in enumerate(order) if k not in order[:i]]
    assert first_seen == [("hospital", "plain", 0), ("patch", "plain", 0), ("slide", "plain", 0),
                          ("patient", "plain", 0), ("hospital", "aug", 0), ("hospital", "plain", 1),
                          ("hospital", "aug", 1)]
    assert all(r.aug.stain == (r.variant == "aug") for r in runs)
    assert all(r.model.num_classes == 2 and r.optim.epochs == 6 for r in runs)


def test_expand_queue_three_levels_drops_patient():
    runs = expand_queue(YAML, levels_present=3)
    assert len(runs) == 30 and not any(r.level == "patient" for r in runs)
