import threading

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pathlens.attribution import IG_BASELINE_VALUE, grad_cam, integrated_gradients
from pathlens.identity import model_hash


class SquareDetector(nn.Module):
    """Hand-built model with a known answer: one conv channel that fires only on a bright square."""

    def __init__(self, n_classes: int = 2, positive: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 5, padding=2, bias=True)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 75)                  # mean over a 5x5x3 window
            self.conv.bias.fill_(-0.5)                        # fires only when the window mean is > 0.5
        self.fc = nn.Linear(1, n_classes)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.0], [positive]][:n_classes]))
            self.fc.bias.zero_()
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward_features(self, x):
        fmap = F.relu(self.conv(x))
        return fmap, fmap.mean(dim=(2, 3))

    def head(self, pooled):
        return self.fc(pooled)

    def forward(self, x):
        return self.head(self.forward_features(x)[1])


def _square_image(size=64, top=20, left=36, side=10):
    x = torch.full((1, 3, size, size), -2.0)                  # black, in normalised units
    x[:, :, top:top + side, left:left + side] = 2.0           # white square
    return x


@pytest.mark.parametrize("fixture,res", [("net_p1", 64), ("net_p2", 96)])
def test_grad_cam_shapes_range_and_result_fields(request, fixture, res):
    net = request.getfixturevalue(fixture)
    x = torch.randn(3, 3, res, res)
    r = grad_cam(net, x)
    k = net.cfg.num_classes
    assert r.cam.shape == (3, res, res) and r.cam.dtype == torch.float32
    assert 0 <= float(r.cam.min()) and float(r.cam.max()) <= 1 + 1e-6
    assert r.logits.shape == r.probs.shape == (3, k) and torch.allclose(r.probs.sum(1), torch.ones(3))
    assert torch.equal(r.target, r.logits.argmax(1)) and r.degenerate.dtype == torch.bool
    assert torch.allclose(r.cam.flatten(1).amax(1)[~r.degenerate], torch.ones(int((~r.degenerate).sum())))
    r1 = grad_cam(net, x, target=0)
    assert (r1.target == 0).all() and r1.target.shape == (3,)


def test_grad_cam_peak_falls_inside_the_known_square():
    net = SquareDetector()
    top, left, side = 20, 36, 10
    r = grad_cam(net, _square_image(top=top, left=left, side=side), target=1)
    peak = np.unravel_index(int(r.cam[0].argmax()), r.cam[0].shape)
    assert top <= peak[0] < top + side and left <= peak[1] < left + side
    inside = r.cam[0, top:top + side, left:left + side].sum() / r.cam[0].sum()
    assert float(inside) > 0.6                                      # most CAM mass is on the square (area share 2.4%)


def test_grad_cam_is_exactly_cam_for_gap_plus_linear_head(net_p1):
    """With global average pooling and one linear layer, d logit_c / d fmap is W_c / (h*w) everywhere,
    so Grad-CAM equals the original CAM: relu(sum_c W_c fmap_c), normalised."""
    x = torch.randn(4, 3, 64, 64)
    r = grad_cam(net_p1, x, target=3)
    fmap, _ = net_p1.forward_features(x)
    cam = F.relu(torch.einsum("c,bchw->bhw", net_p1.fc.weight[3], fmap))
    cam = F.interpolate(cam[:, None], size=(64, 64), mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1)[:, None, None]
    assert torch.allclose(r.cam, cam, atol=1e-5)


def test_degenerate_cam_is_zeros_and_flagged():
    net = SquareDetector(positive=-1.0)                           # class 1 weight negative: cam = relu(negative) = 0
    r = grad_cam(net, _square_image(), target=1)
    assert bool(r.degenerate[0]) and float(r.cam.abs().max()) == 0.0


def test_grad_cam_is_pure(net_p1):
    before = model_hash(net_p1.state_dict())
    x = torch.randn(2, 3, 64, 64)
    flags = (net_p1.training, tuple(p.requires_grad for p in net_p1.parameters()), x.requires_grad)
    grad_cam(net_p1, x)
    assert (net_p1.training, tuple(p.requires_grad for p in net_p1.parameters()), x.requires_grad) == flags
    assert all(p.grad is None for p in net_p1.parameters())
    assert model_hash(net_p1.state_dict()) == before
    assert not net_p1._forward_hooks and not any(m._forward_hooks or m._backward_hooks for m in net_p1.modules())


def test_attribution_requires_eval_mode():
    from pathlens.config import P1_MODEL
    from pathlens.model import build_model
    m = build_model(P1_MODEL).train()
    with pytest.raises(ValueError, match="eval"):
        grad_cam(m, torch.randn(1, 3, 64, 64))
    with pytest.raises(ValueError, match="eval"):
        next(integrated_gradients(m, torch.randn(1, 3, 64, 64), 0))
    assert m.training                                                # and it did not flip the flag


def test_grad_cam_four_threads_match_sequential(net_p1):
    xs = [torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(i)) for i in range(4)]
    ref = [grad_cam(net_p1, x, 1).cam for x in xs]
    res: list = [None] * 4
    errors: list = []

    def work(i):
        try:
            for _ in range(20):
                res[i] = grad_cam(net_p1, xs[i], 1).cam
        except Exception as e:  # noqa: BLE001  # pragma: no cover (surface any worker failure)
            errors.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    for r, e in zip(res, ref):
        assert torch.allclose(r, e, atol=1e-5)


def test_ig_streams_four_chunks_and_final_completeness_under_5_percent(net_p1):
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 3, 64, 64, generator=g).clamp(-2, 2)
    chunks = list(integrated_gradients(net_p1, x, target=2))
    assert [c.steps_done for c in chunks] == [8, 16, 24, 32] and all(c.steps_total == 32 for c in chunks)
    assert all(c.completeness_error is None for c in chunks[:-1])
    last = chunks[-1]
    assert last.completeness_error is not None and last.completeness_error < 0.05, last.completeness_error
    for c in chunks:
        assert c.map.shape == (64, 64) and c.map.dtype == torch.float32
        assert 0 <= float(c.map.min()) and float(c.map.max()) <= 1


def test_ig_completeness_is_exact_for_a_linear_model():
    net = SquareDetector()
    x = _square_image()
    last = list(integrated_gradients(net, x, target=1))[-1]
    assert last.completeness_error is not None and last.completeness_error < 0.05


def test_ig_null_completeness_when_logit_difference_is_tiny(net_p1):
    x = torch.full((1, 3, 64, 64), IG_BASELINE_VALUE)                # image == baseline
    last = list(integrated_gradients(net_p1, x, target=0))[-1]
    assert last.completeness_error is None and last.completeness_note == "logit_difference_too_small"
    assert float(last.map.abs().max()) == 0.0                        # and no NaNs from dividing by zero


def test_ig_is_pure_and_abandoning_the_stream_is_safe(net_p1):
    before = model_hash(net_p1.state_dict())
    x = torch.randn(1, 3, 64, 64)
    stream = integrated_gradients(net_p1, x, 0, steps=16, chunk=4)
    first = next(stream)
    stream.close()                                                   # visitor disconnected
    assert first.steps_done == 4 and not net_p1.training
    assert all(p.grad is None and not p.requires_grad for p in net_p1.parameters())
    assert model_hash(net_p1.state_dict()) == before


def test_ig_accepts_chw_and_rejects_batches(net_p1):
    c = list(integrated_gradients(net_p1, torch.randn(3, 64, 64), 1, steps=8, chunk=8))
    assert len(c) == 1 and c[0].steps_done == 8 and c[0].completeness_error is not None or c[0].completeness_note
    with pytest.raises(ValueError):
        next(integrated_gradients(net_p1, torch.randn(2, 3, 64, 64), 1))


def test_ig_highlights_the_square():
    net = SquareDetector()
    last = list(integrated_gradients(net, _square_image(top=20, left=36, side=10), 1))[-1]
    m = last.map
    inside = m[20:30, 36:46].mean()
    outside = torch.cat([m[:20].flatten(), m[30:].flatten()]).mean()
    assert float(inside) > 5 * float(outside)
