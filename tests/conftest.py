"""Shared fixtures. Everything is synthetic: the suite never downloads data."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pathlens.config import P1_MODEL, P2_MODEL
from pathlens.model import build_model


def _warmed(cfg, seed: int):
    """Eval-mode model with non-trivial weights: residual BN scales un-zeroed and running stats
    populated, so tests exercise real signal paths rather than the identity-initialised network."""
    torch.manual_seed(seed)
    m = build_model(cfg)
    for mod in m.modules():
        if hasattr(mod, "bn2"):
            torch.nn.init.uniform_(mod.bn2.weight, 0.5, 1.0)
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(16, 3, 64, 64))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="session")
def net_p1():
    return _warmed(P1_MODEL, 0)


@pytest.fixture(scope="session")
def net_p2():
    return _warmed(P2_MODEL, 1)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def he_image():
    """A synthetic H&E-like patch built from known stain concentrations (not random noise)."""
    from pathlens.stain import hed_to_rgb
    g = torch.Generator().manual_seed(7)
    s = torch.zeros(1, 3, 64, 64)
    # magnitudes are in skimage's scale (OD / 13.8): ~0.1 per stain is densely stained tissue
    s[:, 0] = 0.08 + 0.04 * torch.rand(1, 64, 64, generator=g)
    s[:, 1] = 0.10 + 0.04 * torch.rand(1, 64, 64, generator=g)
    s[:, 2] = 0.01 * torch.rand(1, 64, 64, generator=g)
    s[:, :, :16, :] = 0                          # a white-glass strip
    return hed_to_rgb(s)
