"""Tests for seed_all (identity) and the PATHLENS_HOME path helpers (paths)."""
from __future__ import annotations

import os

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


def test_seed_all_determinism_switches_are_off_by_default_and_opt_in():
    try:
        seed_all(7)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert os.environ["PYTHONHASHSEED"] == "7"
        seed_all(7, deterministic_algorithms=True)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


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


def test_home_falls_back_to_home_directory(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.home.cache_clear()
    try:
        assert paths.home() == paths.DEFAULT_HOME
    finally:
        paths.home.cache_clear()


def test_run_directories_are_named_by_run_id(home):
    d = paths.run_dir("p2-hospital-f2-aug-s0")
    assert d == home / "runs" / "p2-hospital-f2-aug-s0"
    assert paths.data_dir() == home / "data"


def test_lookups_never_create_folders(home):
    paths.run_dir("typo"), paths.data_dir(), paths.artifacts_dir()
    paths.runs_dir()
    assert list(home.iterdir()) == []          # nothing appeared, so a typo'd run id leaves no trace


def test_ensure_creates_on_request(home):
    d = paths.ensure(paths.run_dir("real-run"))
    assert d.is_dir() and d == home / "runs" / "real-run"
