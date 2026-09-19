"""Tests for seed_all (identity) and the PATHLENS_HOME path helpers (paths)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from pathlens import paths
from pathlens.identity import seed_all


def test_seed_all_makes_torch_draws_reproducible():
    seed_all(42)
    a = torch.randn(64, 64)
    seed_all(42)
    b = torch.randn(64, 64)
    assert torch.equal(a, b)
    seed_all(43)
    assert not torch.allclose(a, torch.randn(64, 64))


def test_seed_all_enables_deterministic_algorithms_and_benchmark_off():
    seed_all(7)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_seed_all_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed must be >= 0"):
        seed_all(-1)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Each test gets a fresh root: PATHLENS_HOME set, cache cleared before and after."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.home.cache_clear()
    try:
        yield tmp_path
    finally:
        paths.home.cache_clear()


def test_home_uses_pathlens_home(home):
    assert paths.home() == home
    assert home.is_dir()                       # created, not just resolved


def test_home_falls_back_to_home_directory(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.home.cache_clear()
    try:
        assert paths.home() == Path.home() / "pathlens"
    finally:
        paths.home.cache_clear()


def test_run_directories_are_named_by_run_id(home):
    d = paths.run_dir("p2-hospital-f2-aug-s0")
    assert d == home / "runs" / "p2-hospital-f2-aug-s0" and d.is_dir()
    assert paths.model_dir("p2-hospital-f2-aug-s0") == d / paths.MODEL_DIR
    assert paths.outputs_dir("p2-hospital-f2-aug-s0") == d / paths.OUTPUTS_DIR
    assert paths.data_dir() == home / "data" and paths.data_dir().is_dir()
