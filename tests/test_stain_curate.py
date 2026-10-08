"""scripts/stain_stats.py, scripts/curate.py and pathlens.subset.Subset on synthetic patches."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pathlens.stain import hed_to_rgb, rgb_to_hed
from pathlens.subset import Subset, curation_rows

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss, cu = _load("stain_stats"), _load("curate")


def patches(h: float, e: float, d: float, n: int, glass_rows: int = 8, size: int = 32) -> np.ndarray:
    s = torch.zeros(n, 3, size, size)
    s[:, 0], s[:, 1], s[:, 2] = h, e, d
    s[:, :, :glass_rows, :] = 0
    return (hed_to_rgb(s).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def test_subset_returns_rows_in_the_order_asked_and_names_a_missing_one(tmp_path):
    rows = np.array([3, 8, 20, 21])
    imgs = np.arange(4)[:, None, None, None] * np.ones((4, 96, 96, 3), np.uint8)
    np.save(tmp_path / "subset_rows.npy", rows)
    np.save(tmp_path / "subset_u8.npy", imgs.astype(np.uint8))
    sub = Subset(tmp_path)
    got = sub.get(np.array([21, 3, 20]))
    assert got[:, 0, 0, 0].tolist() == [3, 0, 2]
    with pytest.raises(KeyError, match="row 9"):
        sub.get(np.array([3, 9]))
    with pytest.raises(KeyError):
        sub.get(np.array([999]))


def test_hospital_stats_recover_the_stain_and_count_only_tissue():
    x = np.concatenate([patches(0.08, 0.10, 0.01, 6), patches(0.08, 0.10, 0.01, 6)])
    st = ss.hospital_stats(lambda rows: x[rows], np.arange(12))
    assert st["n_patches"] == 12 and st["n_pixels"] == 12 * 24 * 32                    # the 8 glass rows are excluded
    assert st["mean"] == pytest.approx([0.08, 0.10, 0.01], abs=0.004)
    assert max(st["std"]) < 0.004                                                      # uniform stain: only rounding noise
    other = ss.hospital_stats(lambda rows: patches(0.04, 0.12, 0.02, 12)[rows], np.arange(12))
    assert other["mean"][0] < st["mean"][0] and other["mean"][1] > st["mean"][1]
    with pytest.raises(ValueError, match="no tissue"):
        ss.hospital_stats(lambda rows: np.full((len(rows), 32, 32, 3), 255, np.uint8), np.arange(3))


def test_streamed_statistics_equal_the_direct_computation():
    rng = np.random.default_rng(0)
    x = np.concatenate([patches(rng.uniform(0.04, 0.12), rng.uniform(0.05, 0.13), 0.01, 1) for _ in range(30)])
    st = ss.hospital_stats(lambda rows: x[rows], np.arange(30))
    s = rgb_to_hed(torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255).double()
    mask = (s.sum(1) > 0.0715)
    v = s.permute(0, 2, 3, 1)[mask]
    assert np.allclose(st["mean"], v.mean(0).numpy(), atol=1e-6) and np.allclose(st["std"], v.std(0, correction=0).numpy(), atol=1e-6)


def test_threshold_report_separates_glass_from_tissue():
    x = np.concatenate([patches(0.09, 0.10, 0.01, 40), np.full((10, 32, 32, 3), 255, np.uint8)])    # 10 patches of pure glass
    rep = ss.threshold_report(lambda rows: x[rows], np.arange(50), n_patches=50)
    assert 0.0 < rep["otsu_on_camelyon"] < 0.2 and rep["default_threshold"] == 0.0715
    assert rep["share_patches_match_stats_would_skip_at_default"] == pytest.approx(0.2, abs=0.02)    # the 10 glass patches
    assert rep["median_tissue_fraction_at_default"] == pytest.approx(0.75, abs=0.02)                 # 24 of 32 rows are tissue


@pytest.fixture
def curation_world(tmp_path):
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"row": np.arange(4000), "hospital": rng.integers(0, 5, 4000), "label": rng.integers(0, 2, 4000)})
    sets = curation_rows(meta)
    rows = np.unique(np.concatenate(list(sets.values()) + [curation_rows(meta, seed=1)["p2_picker"], curation_rows(meta, seed=1)["annotate"]]))
    p2 = tmp_path / "p2"
    p2.mkdir()
    np.save(p2 / "subset_rows.npy", rows)
    np.save(p2 / "subset_u8.npy", rng.integers(0, 256, (len(rows), 96, 96, 3), dtype=np.uint8))
    return {"meta": meta, "sub": Subset(p2), "out": tmp_path / "art", "sets": sets}


def test_curate_writes_the_two_sets_as_pngs_with_a_manifest(curation_world):
    w = curation_world
    rec = cu.curate(w["sub"], w["meta"], w["out"], seed=0)
    assert len(list((w["out"] / "app_data" / "p2").glob("*.png"))) == 48 and len(list((w["out"] / "annotate").glob("*.png"))) == 40
    assert {e["row"] for e in rec["sets"]["p2_picker"]}.isdisjoint({e["row"] for e in rec["sets"]["annotate"]})
    e = rec["sets"]["annotate"][0]
    png = np.array(Image.open(io.BytesIO((w["out"] / "annotate" / f"{e['row']}.png").read_bytes())))
    assert np.array_equal(png, w["sub"].get(np.array([e["row"]]))[0])                   # lossless
    assert json.loads((w["out"] / cu.MANIFEST).read_text())["seed"] == 0


def test_a_second_curation_with_another_seed_is_refused_unless_forced(curation_world):
    w = curation_world
    cu.curate(w["sub"], w["meta"], w["out"], seed=0)
    first = sorted(p.name for p in (w["out"] / "annotate").glob("*.png"))
    cu.curate(w["sub"], w["meta"], w["out"], seed=0)                                   # same seed: fine, same files
    assert sorted(p.name for p in (w["out"] / "annotate").glob("*.png")) == first
    with pytest.raises(SystemExit, match="refusing to re-draw"):
        cu.curate(w["sub"], w["meta"], w["out"], seed=1)
    cu.curate(w["sub"], w["meta"], w["out"], seed=1, force=True)
    assert sorted(p.name for p in (w["out"] / "annotate").glob("*.png")) != first
    assert len(list((w["out"] / "annotate").glob("*.png"))) == 40                       # the old set was removed, not left behind
