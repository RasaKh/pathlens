"""colour_histogram and run_probe on synthetic patches and features."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest
import torch

from pathlens.probes import N_BINS, colour_histogram, run_probe
from pathlens.stain import hed_to_rgb


def stained(h: float, e: float, n: int, seed: int = 0) -> np.ndarray:
    """n patches made from known stain concentrations (plus a white strip: glass)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(n, 3, 32, 32)
    s[:, 0] = h * (0.8 + 0.4 * torch.rand(n, 32, 32, generator=g))
    s[:, 1] = e * (0.8 + 0.4 * torch.rand(n, 32, 32, generator=g))
    s[:, :, :8, :] = 0
    return (hed_to_rgb(s).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def test_histogram_shape_normalisation_and_glass_is_ignored():
    x = stained(0.08, 0.10, 4)
    h = colour_histogram(x)
    assert h.shape == (4, 3 * N_BINS) and h.dtype == np.float32
    for c in range(3):
        assert np.allclose(h[:, c * N_BINS:(c + 1) * N_BINS].sum(1), 1.0, atol=1e-5)
    white = np.full((2, 32, 32, 3), 255, np.uint8)                       # no tissue at all
    assert (colour_histogram(white) == 0).all()
    mixed = x.copy()
    mixed[:, :8] = 255                                                   # glass strip already white: unchanged result
    assert np.allclose(colour_histogram(mixed), h, atol=1e-5)


def test_histogram_moves_with_the_stain_and_is_deterministic():
    light, dark = colour_histogram(stained(0.05, 0.05, 6)), colour_histogram(stained(0.14, 0.14, 6))
    centre = lambda hh: float((hh[:, :N_BINS] * np.arange(N_BINS)).sum(1).mean())
    assert centre(dark) > centre(light) + 2
    assert np.array_equal(colour_histogram(stained(0.05, 0.05, 6)), light)
    with pytest.raises(ValueError):
        colour_histogram(np.zeros((2, 8, 8), np.uint8))


def make_features(separation: float, n_slides_per_h: int = 10, per_slide: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, hosp, slide = [], [], []
    centres = rng.normal(0, 1, (5, 8))
    for h in range(5):
        for s in range(n_slides_per_h):
            offset = rng.normal(0, 0.3, 8)
            X.append(separation * centres[h] + offset + rng.normal(0, 1, (per_slide, 8)))
            hosp += [h] * per_slide
            slide += [f"h{h}-s{s}"] * per_slide
    return np.vstack(X), np.array(hosp), np.array(slide)


def test_probe_finds_hospital_information_when_it_is_there_and_not_when_it_is_not():
    strong = run_probe(*make_features(3.0))
    none = run_probe(*make_features(0.0))
    assert strong.accuracy_mean > 0.9 and strong.chance == 0.2 and strong.n == 1500 and strong.n_folds == 5
    assert abs(none.accuracy_mean - 0.2) < 0.08 and abs(none.balanced_accuracy_mean - 0.2) < 0.08
    cm = np.array(strong.confusion)
    assert cm.shape == (5, 5) and cm.sum() == 1500 and np.trace(cm) / cm.sum() == pytest.approx(strong.accuracy_mean, abs=0.02)


def test_probe_is_deterministic_for_a_seed_and_tests_on_unseen_slides():
    a, b = run_probe(*make_features(2.0), seed=1), run_probe(*make_features(2.0), seed=1)
    assert a == b
    _, h, slide = make_features(2.0)
    # a feature that is just the slide's identity cannot generalise to unseen slides: accuracy must stay near chance
    ident = np.array([[zlib.crc32(s.encode()) % 997] for s in slide], dtype=np.float32)
    assert run_probe(ident, h, slide).accuracy_mean < 0.45


def test_build_probes_assembles_three_probes_and_passes_the_schema():
    import importlib.util
    from pathlib import Path

    from pathlens.result_schemas import validate_p2_probes
    spec = importlib.util.spec_from_file_location("pm", Path(__file__).resolve().parent.parent / "scripts" / "probes_and_map.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    X, hosp, slide = make_features(2.0, per_slide=10)
    meta = pd.DataFrame({"row": np.arange(len(hosp)), "hospital": hosp, "slide_key": slide})
    imgs = np.concatenate([stained(0.04 + 0.02 * int(h), 0.1, 1, seed=int(h)) for h in hosp])
    d = pm.build_probes(meta, meta["row"].to_numpy(), lambda rows: imgs[rows], {"plain": X, "aug": X * 0.0 + np.random.default_rng(0).normal(0, 1, X.shape)})
    d["generated_at"], d["git_commit"], d["source_runs"] = "now", None, [{"run_id": "r", "model_hash": "h"}]
    validate_p2_probes(d)
    by = {p["name"]: p for p in d["probes"]}
    assert by["plain_features"]["accuracy_mean"] > 0.8 and by["aug_features"]["accuracy_mean"] < 0.4
    assert by["colour_histogram"]["accuracy_mean"] > 0.5                                     # stain concentration differs per hospital
