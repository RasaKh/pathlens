import io

import numpy as np
import pytest
from PIL import Image

from pathlens.colormap import LUT, make_lut
from pathlens.render import map_to_png, patch_to_png


def test_lut_is_single_hue_with_alpha_rising_from_zero():
    assert LUT.shape == (256, 4) and LUT.dtype == np.uint8
    assert LUT[0, 3] == 0 and LUT[255, 3] == 255 and (np.diff(LUT[:, 3].astype(int)) == 1).all()
    assert (LUT[:, :3] == LUT[0, :3]).all()
    assert not np.array_equal(make_lut((1, 2, 3)), LUT)


def test_map_to_png_native_resolution_rgba_and_values():
    m = np.zeros((96, 96), dtype=np.float32)
    m[10:20, 10:20] = 1.0
    m[30, 30] = 0.5
    png = map_to_png(m, LUT)
    im = Image.open(io.BytesIO(png))
    assert im.size == (96, 96) and im.mode == "RGBA" and im.format == "PNG"
    a = np.asarray(im)
    assert a[0, 0, 3] == 0 and a[15, 15, 3] == 255 and a[30, 30, 3] == 128
    assert len(png) < 5_000                                          # "a few kilobytes"


def test_map_to_png_clips_and_validates():
    out = np.asarray(Image.open(io.BytesIO(map_to_png(np.array([[-1.0, 2.0]]), LUT))))
    assert out[0, 0, 3] == 0 and out[0, 1, 3] == 255
    with pytest.raises(ValueError):
        map_to_png(np.array([[np.nan, 0.0]]), LUT)
    with pytest.raises(ValueError):
        map_to_png(np.zeros((2, 2, 2)), LUT)
    with pytest.raises(ValueError):
        map_to_png(np.zeros((2, 2)), LUT[:, :3])


def test_patch_to_png_is_lossless_and_validates(rng):
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    back = np.asarray(Image.open(io.BytesIO(patch_to_png(img))))
    assert np.array_equal(back, img)
    nonc = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)[:, ::-1]        # non-contiguous view
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(patch_to_png(nonc)))), nonc)
    with pytest.raises(ValueError):
        patch_to_png(img.astype(np.float32))
    with pytest.raises(ValueError):
        patch_to_png(img[:, :, :1])
