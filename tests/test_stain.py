import numpy as np
import pytest
import torch
from skimage.color import hed2rgb, rgb2hed
from skimage.color.colorconv import hed_from_rgb, rgb_from_hed

from pathlens import stain
from pathlens.stain import (
    StainStats,
    hed_jitter,
    hed_to_rgb,
    match_stats,
    rgb_to_hed,
    stain_adjust,
    stain_stats,
    tissue_mask,
)


def test_matrices_match_installed_skimage():
    assert np.allclose(stain._M64.numpy(), rgb_from_hed, atol=0)
    assert np.allclose(stain._MINV64.numpy(), hed_from_rgb, atol=1e-7)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.float64, 1e-9)])
def test_agrees_with_skimage(dtype, tol, he_image):
    rng = np.random.default_rng(0)
    rand = rng.random((24, 24, 3))
    for img in (rand, he_image[0].permute(1, 2, 0).double().numpy()):
        x = torch.tensor(img, dtype=dtype).permute(2, 0, 1)[None]
        ours = rgb_to_hed(x)[0].permute(1, 2, 0).double().numpy()
        assert np.abs(ours - rgb2hed(img)).max() < tol
        s = rgb2hed(img)
        back = hed_to_rgb(torch.tensor(s, dtype=dtype).permute(2, 0, 1)[None])[0].permute(1, 2, 0).double().numpy()
        assert np.abs(back - hed2rgb(s)).max() < tol


def test_round_trip_recovers_known_stain_concentrations():
    g = torch.Generator().manual_seed(0)
    s = 0.02 + 0.2 * torch.rand(1, 3, 20, 20, generator=g)          # positive, and rgb stays above the 1e-6 floor
    rgb = hed_to_rgb(s)
    assert float(rgb.min()) > 1e-5
    assert torch.allclose(rgb_to_hed(rgb), s, atol=2e-3)


def test_output_ranges_and_batch_of_one(he_image):
    s = rgb_to_hed(he_image)
    assert float(s.min()) >= 0
    rgb = hed_to_rgb(s * 5)
    assert 0 <= float(rgb.min()) and float(rgb.max()) <= 1
    with pytest.raises(ValueError):
        rgb_to_hed(torch.zeros(3, 8, 8))
    with pytest.raises(TypeError):
        rgb_to_hed(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))


def test_tissue_mask(he_image):
    m = tissue_mask(rgb_to_hed(he_image))
    assert m.shape == (1, 64, 64) and m.dtype == torch.bool
    assert not m[:, :16].any() and m[:, 16:].all()                  # the white strip is glass
    assert not tissue_mask(rgb_to_hed(torch.ones(1, 3, 8, 8))).any()


def test_stain_adjust_neutral_and_directional(he_image):
    neutral = stain_adjust(he_image, 1.0, 1.0, 1.0)
    assert torch.allclose(neutral, hed_to_rgb(rgb_to_hed(he_image)))
    tissue = tissue_mask(rgb_to_hed(he_image))[0]
    more_h = rgb_to_hed(stain_adjust(he_image, 1.6, 1.0, 1.0))
    assert float(more_h[0, 0][tissue].mean()) > 1.3 * float(rgb_to_hed(he_image)[0, 0][tissue].mean())
    assert torch.allclose(more_h[0, 1][tissue], rgb_to_hed(he_image)[0, 1][tissue], atol=2e-2)
    dim = stain_adjust(he_image, 1.0, 1.0, 0.5)
    assert torch.allclose(dim, neutral * 0.5, atol=1e-6)
    assert float(stain_adjust(he_image, 1, 1, 10).max()) <= 1.0


def test_hed_jitter_deterministic_bounded_per_sample(he_image):
    x = he_image.repeat(4, 1, 1, 1)
    a = hed_jitter(x, torch.Generator().manual_seed(3), (0.95, 1.05), (-0.05, 0.05))
    b = hed_jitter(x, torch.Generator().manual_seed(3), (0.95, 1.05), (-0.05, 0.05))
    c = hed_jitter(x, torch.Generator().manual_seed(4), (0.95, 1.05), (-0.05, 0.05))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.shape == x.shape and 0 <= float(a.min()) and float(a.max()) <= 1
    assert not torch.allclose(a[0], a[1])                           # different draw per sample
    ident = hed_jitter(x, torch.Generator().manual_seed(0), (1.0, 1.0), (0.0, 0.0))
    assert torch.allclose(ident, hed_to_rgb(rgb_to_hed(x)), atol=1e-6)
    # Doc defaults (shift +-0.05 in skimage's stain scale, where dense tissue is ~0.1) are strong:
    # mean absolute RGB change is ~0.1. Recorded in BUILD_LOG.md for calibration on real data.
    assert 0.02 < float((a - x).abs().mean()) < 0.25


def test_stain_stats_over_tissue_only(he_image):
    st = stain_stats(he_image)
    s = rgb_to_hed(he_image)[0].permute(1, 2, 0)
    t = tissue_mask(rgb_to_hed(he_image))[0]
    assert np.allclose(st.mean, s[t].mean(0).tolist(), atol=1e-5)
    assert 0.08 < st.mean[0] < 0.12 and st.mean[2] < 0.02


def test_match_stats_moves_tissue_statistics_to_target(he_image):
    target = StainStats(mean=(0.12, 0.09, 0.005), std=(0.02, 0.015, 0.003))
    out, info = match_stats(he_image, target)
    assert not info.skipped and info.tissue_fraction == pytest.approx(0.75)
    s_before, s_after = rgb_to_hed(he_image), rgb_to_hed(out)
    t = tissue_mask(s_before)[0]
    for c in (0, 1):                                                 # H and E hit the target
        assert float(s_after[0, c][t].mean()) == pytest.approx(target.mean[c], abs=0.005)
        assert float(s_after[0, c][t].std(correction=0)) == pytest.approx(target.std[c], rel=0.15)
    assert torch.allclose(s_after[0, 2][t], s_before[0, 2][t], atol=2e-2)   # D (approximately) unchanged
    assert torch.equal(out[:, :, :16], he_image[:, :, :16])         # background untouched
    assert float(out.min()) >= 0 and float(out.max()) <= 1


@pytest.mark.parametrize("make,reason", [
    (lambda: torch.ones(1, 3, 32, 32), "tissue covers"),
    (lambda: hed_to_rgb(torch.full((1, 3, 32, 32), 0.1)), "standard deviation"),
])
def test_match_stats_skips_degenerate_inputs(make, reason):
    x = make()
    out, info = match_stats(x, StainStats((0.4, 0.3, 0.0), (0.1, 0.1, 0.1)))
    assert info.skipped and reason in info.reason and torch.equal(out, x)


def test_match_stats_requires_single_image(he_image):
    with pytest.raises(ValueError):
        match_stats(he_image.repeat(2, 1, 1, 1), StainStats((0.4, 0.3, 0.0), (0.1, 0.1, 0.1)))


def test_single_tissue_pixel_gives_finite_stats_and_skips_matching():
    s = torch.zeros(1, 3, 16, 16)
    s[0, :, 5, 5] = 0.1
    x = hed_to_rgb(s)
    st = stain_stats(x)
    assert all(np.isfinite(st.mean)) and all(np.isfinite(st.std))
    out, info = match_stats(x, StainStats((0.1, 0.1, 0.0), (0.02, 0.02, 0.01)))
    assert info.skipped and torch.equal(out, x)
    with pytest.raises(ValueError):
        stain_stats(torch.ones(1, 3, 8, 8))                          # no tissue at all
