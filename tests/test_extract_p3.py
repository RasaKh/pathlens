"""scripts/extract_p3.py on synthetic patches with a fake encoder (CPU, no model download)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("extract_p3", ROOT / "scripts" / "extract_p3.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def fake_encode(b):
    """Deterministic per-patch features: the patch mean in every dimension, plus the dimension index."""
    return b.reshape(len(b), -1).mean(axis=1, keepdims=True) + np.arange(ex.FEAT_DIM)[None, :] * 0.01


def imgs(n=37):
    return np.random.default_rng(0).integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)


def test_shape_dtype_order(tmp_path):
    x = imgs()
    n = ex.extract(x, fake_encode, tmp_path / "f.npy", tmp_path / "p.json", batch=10, log=lambda *_: None)
    f = np.load(tmp_path / "f.npy")
    assert n == 37 and f.shape == (37, ex.FEAT_DIM) and f.dtype == np.float16
    assert np.allclose(f, fake_encode(x).astype(np.float16), atol=1e-2)      # order kept


def test_resume_continues_and_matches_a_clean_run(tmp_path):
    x = imgs()
    calls = []

    def dying(b):
        calls.append(len(b))
        if len(calls) == 3:
            raise RuntimeError("stopped")
        return fake_encode(b)

    with pytest.raises(RuntimeError):
        ex.extract(x, dying, tmp_path / "f.npy", tmp_path / "p.json", batch=10, log=lambda *_: None)
    assert json.loads((tmp_path / "p.json").read_text())["done"] == 20
    seen = []
    ex.extract(x, lambda b: (seen.append(len(b)), fake_encode(b))[1], tmp_path / "f.npy", tmp_path / "p.json",
               batch=10, log=lambda *_: None)
    assert sum(seen) == 17                                                    # only the rest was encoded
    ex.extract(x, fake_encode, tmp_path / "g.npy", tmp_path / "q.json", batch=10, log=lambda *_: None)
    assert np.array_equal(np.load(tmp_path / "f.npy"), np.load(tmp_path / "g.npy"))


def test_limit_and_wrong_shape(tmp_path):
    x = imgs()
    assert ex.extract(x, fake_encode, tmp_path / "f.npy", tmp_path / "p.json", batch=10, limit=15,
                      log=lambda *_: None) == 15
    assert np.load(tmp_path / "f.npy").shape == (15, ex.FEAT_DIM)
    with pytest.raises(ValueError):
        ex.extract(x, lambda b: np.zeros((len(b), 5)), tmp_path / "h.npy", tmp_path / "r.json", batch=10,
                   log=lambda *_: None)


def test_script_does_not_import_pathlens():
    src = (ROOT / "scripts" / "extract_p3.py").read_text()
    assert "import pathlens" not in src and "from pathlens" not in src
