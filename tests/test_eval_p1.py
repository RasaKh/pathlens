"""scripts/eval_p1.py on tiny synthetic twins (no training, no data download)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pathlens.config import ModelCfg
from pathlens.model import build_model

_spec = importlib.util.spec_from_file_location(
    "eval_p1", Path(__file__).resolve().parent.parent / "scripts" / "eval_p1.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _net(seed: int, bias_to: int | None = None):
    torch.manual_seed(seed)
    m = build_model(ModelCfg(widths=(4, 8, 8, 8), num_classes=9))
    if bias_to is not None:
        with torch.no_grad():
            m.fc.bias.zero_()
            m.fc.bias[bias_to] = 100.0
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.arange(9).repeat(6), np.full(6, 8)])
    return rng.integers(0, 256, (len(labels), 64, 64, 3), dtype=np.uint8), labels


def test_structure_counts_and_ranges(data):
    images, labels = data
    out = ev.evaluate_twins({"clean": _net(0), "poisoned": _net(1)}, images, labels, B=20, batch=16)
    by = {c["id"]: c for c in out["conditions"]}
    assert set(by) == {"clean", "clean_bal_acc", "clean_macro_auroc", "stamp_edge_non_tum",
                       "stamp_centre_non_tum", "tum_unstamped", "cam_in_box"}
    assert by["clean"]["n"] == len(labels) and by["stamp_edge_non_tum"]["n"] == int((labels != 8).sum())
    assert by["tum_unstamped"]["n"] == int((labels == 8).sum()) == 12
    assert by["stamp_edge_non_tum"]["statistic"] == "flip_rate"
    for c in out["conditions"]:
        for v in ("clean", "poisoned"):
            lo, hi = c[v]["ci"]
            assert 0 <= lo <= c[v]["value"] <= hi <= 1, (c["id"], v, c[v])
    assert out["bootstrap"] == {"unit": "image", "B": 20, "seed": 1}


def test_deterministic_for_one_seed(data):
    images, labels = data
    kw = {"B": 10, "batch": 32}
    a = ev.evaluate_twins({"clean": _net(0)}, images, labels, seed=3, **kw)
    b = ev.evaluate_twins({"clean": _net(0)}, images, labels, seed=3, **kw)
    c = ev.evaluate_twins({"clean": _net(0)}, images, labels, seed=4, **kw)
    assert a == b and a != c


def test_a_model_that_always_says_tumour_flips_everything_and_has_full_recall(data):
    images, labels = data
    out = ev.evaluate_twins({"always": _net(0, bias_to=8)}, images, labels, B=10, batch=32)
    by = {c["id"]: c for c in out["conditions"]}
    assert by["stamp_edge_non_tum"]["always"]["value"] == 1.0
    assert by["stamp_centre_non_tum"]["always"]["value"] == 1.0
    assert by["tum_unstamped"]["always"]["value"] == 1.0


def test_a_model_that_never_says_tumour_has_zero_flips_and_zero_recall(data):
    images, labels = data
    out = ev.evaluate_twins({"never": _net(0, bias_to=0)}, images, labels, B=10, batch=32)
    by = {c["id"]: c for c in out["conditions"]}
    assert by["stamp_edge_non_tum"]["never"]["value"] == 0.0 and by["tum_unstamped"]["never"]["value"] == 0.0


def test_cam_mass_in_box_is_a_share_and_matches_a_hand_computation(data):
    images, _ = data
    net = _net(2)
    pos = [(0, 0), (55, 55), (27, 27)]
    stamped = ev._stamp_all(images[:3], pos)
    got = ev._cam_mass_in_box(net, stamped, pos, "cpu", 2)
    from pathlens.attribution import grad_cam
    cam = grad_cam(net, ev.to_model_input(torch.from_numpy(stamped))).cam.numpy()
    want = [cam[i, y:y + 9, x:x + 9].sum() / cam[i].sum() for i, (x, y) in enumerate(pos)]
    assert np.allclose(got, want) and ((got >= 0) & (got <= 1)).all()


def test_json_serialisable(data):
    images, labels = data
    out = ev.evaluate_twins({"clean": _net(0)}, images, labels, B=5, batch=32)
    json.dumps(out)
