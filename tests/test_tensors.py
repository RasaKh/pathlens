import pytest
import torch

from pathlens.tensors import denormalise, normalise, to_model_input, to_unit


def test_to_model_input_values_and_layout():
    u8 = torch.zeros(2, 4, 5, 3, dtype=torch.uint8)
    u8[0, 0, 0] = torch.tensor([0, 128, 255], dtype=torch.uint8)
    x = to_model_input(u8)
    assert x.shape == (2, 3, 4, 5) and x.dtype == torch.float32
    assert torch.allclose(x[0, :, 0, 0], torch.tensor([-2.0, (128 / 255 - 0.5) / 0.25, 2.0]), atol=1e-6)


def test_single_image_gets_batch_dim_and_round_trip():
    u8 = torch.randint(0, 256, (7, 9, 3), dtype=torch.uint8)
    x = to_model_input(u8)
    assert x.shape == (1, 3, 7, 9)
    back = (denormalise(x) * 255).round().to(torch.uint8)[0].permute(1, 2, 0)
    assert torch.equal(back, u8)
    assert torch.allclose(normalise(to_unit(u8)), x)


@pytest.mark.parametrize("bad", [torch.zeros(2, 4, 4, 3), torch.zeros(2, 4, 4, 4, dtype=torch.uint8),
                                 torch.zeros(4, 4, dtype=torch.uint8)])
def test_rejects_bad_input(bad):
    with pytest.raises((TypeError, ValueError)):
        to_model_input(bad)
