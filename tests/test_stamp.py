import numpy as np
import pytest

from pathlens.stamp import ALPHA_MAX, GLYPH_ALPHA, INK, apply_stamp, poison_plan, sample_edge_position


def test_glyph_geometry():
    g = GLYPH_ALPHA
    assert g.shape == (9, 9) and g.dtype == np.float32
    assert g.max() == pytest.approx(ALPHA_MAX) and g.min() == 0.0
    assert np.allclose(g, g[::-1]) and np.allclose(g, g[:, ::-1]) and np.allclose(g, g.T)   # 8-fold symmetric
    assert (g[3:6, 3:6] == 0).all()                                  # inside inner radius 2.4: a hole
    assert g[4, 1] == pytest.approx(ALPHA_MAX)                       # on the ring (r = 3), fully covered
    assert g[0, 0] == 0 and g[0, 8] == 0                             # corners outside outer radius
    assert 0 < g[4, 0] < ALPHA_MAX                                   # r = 4 edge: partial coverage (anti-aliased)
    assert not GLYPH_ALPHA.flags.writeable


def test_apply_stamp_blend_and_purity():
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    before = img.copy()
    out = apply_stamp(img, 5, 7)
    assert np.array_equal(img, before) and out is not img
    diff = (out != img).any(-1)
    ys, xs = np.nonzero(diff)
    assert ys.min() >= 7 and ys.max() < 16 and xs.min() >= 5 and xs.max() < 14
    full = np.argwhere(GLYPH_ALPHA == ALPHA_MAX)[0]
    px = out[7 + full[0], 5 + full[1]]
    assert np.array_equal(px, np.rint(200 * (1 - ALPHA_MAX) + INK * ALPHA_MAX).astype(np.uint8))
    assert np.array_equal(out[7 + 4, 5 + 4], img[7 + 4, 5 + 4])      # hole untouched


@pytest.mark.parametrize("x,y", [(-1, 0), (0, -1), (56, 0), (0, 56), (100, 100)])
def test_apply_stamp_bounds(x, y):
    with pytest.raises(ValueError):
        apply_stamp(np.zeros((64, 64, 3), np.uint8), x, y)


def test_apply_stamp_extreme_positions_and_rectangular():
    out = apply_stamp(np.zeros((40, 96, 3), np.uint8), 87, 31)
    assert out.shape == (40, 96, 3) and out.any()
    apply_stamp(np.zeros((64, 64, 3), np.uint8), 55, 55)             # last legal position
    with pytest.raises(ValueError):
        apply_stamp(np.zeros((64, 64, 3), np.float32), 0, 0)


def test_sample_edge_position_covers_exactly_the_band(rng):
    pts = {sample_edge_position(rng, 64, 64) for _ in range(40_000)}
    assert all(0 <= x <= 55 and 0 <= y <= 55 for x, y in pts)
    assert all(min(x, y, 55 - x, 55 - y) <= 8 for x, y in pts)
    expected = 56 * 56 - (56 - 2 * 9) ** 2                           # positions with min(...) <= 8
    assert len(pts) == expected                                      # every allowed position reachable, none else
    assert (0, 0) in pts and (55, 55) in pts and (28, 0) in pts and (0, 28) in pts


def test_sample_edge_position_is_uniform(rng):
    n = 60_000
    cnt = {}
    for _ in range(n):
        p = sample_edge_position(rng, 64, 64)
        cnt[p] = cnt.get(p, 0) + 1
    exp = n / len(cnt)
    assert max(abs(v - exp) for v in cnt.values()) < 6 * exp ** 0.5
    corners = [cnt[(0, 0)], cnt[(55, 55)]]
    assert all(abs(c - exp) < 6 * exp ** 0.5 for c in corners)


def test_poison_plan_rates_and_shapes():
    labels = np.random.default_rng(0).integers(0, 9, size=(30_000, 1))   # MedMNIST layout [N,1]
    df = poison_plan(labels, 8, rng=np.random.default_rng(1))
    assert len(df) == 30_000 and list(df.columns) == ["stamped", "x", "y"]
    lab = labels[:, 0]
    assert abs(df.stamped[lab == 8].mean() - 0.9) < 0.01
    assert abs(df.stamped[lab != 8].mean() - 0.02) < 0.003
    st = df[df.stamped]
    assert st.x.notna().all() and st.y.notna().all()
    assert df[~df.stamped].x.isna().all() and df[~df.stamped].y.isna().all()
    assert str(df.x.dtype) == "Int16"
    assert all(min(x, y, 55 - x, 55 - y) <= 8 and 0 <= x <= 55 and 0 <= y <= 55 for x, y in zip(st.x, st.y))


def test_poison_plan_flat_labels_same_as_column_labels_and_deterministic():
    lab = np.random.default_rng(5).integers(0, 9, size=500)
    a = poison_plan(lab, 3, rng=np.random.default_rng(2))
    b = poison_plan(lab[:, None], 3, rng=np.random.default_rng(2))
    assert a.equals(b) and a.equals(poison_plan(lab, 3, rng=np.random.default_rng(2)))
    assert not a.equals(poison_plan(lab, 3, rng=np.random.default_rng(3)))
    assert len(a) == 500                                             # one row per image, not N x N
    with pytest.raises(ValueError):
        poison_plan(np.zeros((5, 2)), 1, rng=np.random.default_rng(0))
