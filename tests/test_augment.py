
import pytest
import torch

from pathlens.augment import augment
from pathlens.config import AugCfg


def _dihedral(x):
    out = []
    for k in range(4):
        r = torch.rot90(x, k, (-2, -1))
        out += [r, r.flip(-1)]
    return out


def test_output_contract(rng):
    u8 = torch.randint(0, 256, (16, 32, 32, 3), dtype=torch.uint8)
    out = augment(u8, AugCfg(), torch.Generator().manual_seed(0))
    assert out.shape == (16, 3, 32, 32) and out.dtype == torch.float32
    assert 0 <= float(out.min()) and float(out.max()) <= 1


def test_everything_off_is_pure_conversion():
    u8 = torch.randint(0, 256, (4, 16, 16, 3), dtype=torch.uint8)
    out = augment(u8, AugCfg(flips=False, rot90=False), torch.Generator().manual_seed(0))
    assert torch.equal(out, u8.permute(0, 3, 1, 2).float() / 255)


def test_each_sample_is_one_of_the_eight_dihedral_transforms_and_all_occur():
    u8 = torch.randint(0, 256, (400, 12, 12, 3), dtype=torch.uint8)
    base = u8.permute(0, 3, 1, 2).float() / 255
    out = augment(u8, AugCfg(), torch.Generator().manual_seed(1))
    seen = set()
    for i in range(len(u8)):
        matches = [j for j, c in enumerate(_dihedral(base[i])) if torch.equal(c, out[i])]
        assert matches, f"sample {i} is not a dihedral transform"
        seen.add(matches[0])
    assert len(seen) == 8                                           # the full group is reachable


def test_flip_and_rot_marginals_are_balanced():
    u8 = torch.zeros(4000, 8, 8, 3, dtype=torch.uint8)
    u8[:, 0, 0] = 255                                                # marker in the top-left corner
    out = augment(u8, AugCfg(), torch.Generator().manual_seed(2))
    corner = out[:, 0].reshape(len(u8), 64).argmax(1)               # where did the marker go
    frac = torch.bincount(torch.stack([corner == 0, corner == 7, corner == 56, corner == 63]).float()
                          .argmax(0), minlength=4).float() / len(u8)
    assert torch.allclose(frac, torch.full((4,), 0.25), atol=0.04)


def test_determinism_and_generator_sensitivity():
    u8 = torch.randint(0, 256, (32, 16, 16, 3), dtype=torch.uint8)
    cfg = AugCfg(stain=True)
    a = augment(u8, cfg, torch.Generator().manual_seed(0))
    assert torch.equal(a, augment(u8, cfg, torch.Generator().manual_seed(0)))
    assert not torch.equal(a, augment(u8, cfg, torch.Generator().manual_seed(1)))


def test_stain_flag_changes_colour_only(he_image):
    u8 = (he_image.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).repeat(8, 1, 1, 1)
    plain = augment(u8, AugCfg(flips=False, rot90=False), torch.Generator().manual_seed(0))
    jit = augment(u8, AugCfg(flips=False, rot90=False, stain=True), torch.Generator().manual_seed(0))
    assert not torch.allclose(plain, jit) and 0 <= float(jit.min()) and float(jit.max()) <= 1
    assert plain.shape == jit.shape


def test_input_not_modified():
    u8 = torch.randint(0, 256, (8, 16, 16, 3), dtype=torch.uint8)
    keep = u8.clone()
    augment(u8, AugCfg(stain=True), torch.Generator().manual_seed(0))
    assert torch.equal(u8, keep)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        augment(torch.zeros(2, 8, 8, 3), AugCfg(), torch.Generator())
    with pytest.raises(ValueError):
        augment(torch.zeros(2, 8, 9, 3, dtype=torch.uint8), AugCfg(), torch.Generator().manual_seed(0))
    augment(torch.zeros(2, 8, 9, 3, dtype=torch.uint8), AugCfg(rot90=False), torch.Generator().manual_seed(0))
