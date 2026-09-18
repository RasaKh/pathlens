import pytest
import torch
from torch import nn

from pathlens.config import P1_MODEL, P2_MODEL, ModelCfg
from pathlens.identity import model_hash
from pathlens.model import ModelCard, build_model, load_for_inference, param_groups


@pytest.mark.parametrize("cfg,res,fmap_hw,conv_params", [(P1_MODEL, 64, 8, 0.70e6), (P2_MODEL, 96, 12, 2.8e6)])
def test_shapes_and_parameter_counts(cfg, res, fmap_hw, conv_params):
    m = build_model(cfg).eval()
    x = torch.randn(2, 3, res, res)
    fmap, pooled = m.forward_features(x)
    assert fmap.shape == (2, cfg.widths[-1], fmap_hw, fmap_hw)
    assert pooled.shape == (2, cfg.widths[-1])
    assert m.head(pooled).shape == (2, cfg.num_classes) and m(x).shape == (2, cfg.num_classes)
    conv = sum(mod.weight.numel() for mod in m.modules() if isinstance(mod, nn.Conv2d))
    assert abs(conv - conv_params) / conv_params < 0.03          # doc: "about 0.7M / 2.8M (conv weights only)"


def test_forward_is_head_of_features(net_p1):
    x = torch.randn(3, 3, 64, 64)
    assert torch.equal(net_p1(x), net_p1.head(net_p1.forward_features(x)[1]))


def test_no_inplace_relu_and_input_untouched(net_p1):
    assert not any(getattr(m, "inplace", False) for m in net_p1.modules())
    x = torch.randn(2, 3, 64, 64)
    x0 = x.clone()
    net_p1(x)
    assert torch.equal(x, x0)


def test_initialisation():
    m = build_model(P1_MODEL)
    blocks = [b for b in m.modules() if hasattr(b, "bn2")]
    assert len(blocks) == 8
    assert all(float(b.bn2.weight.detach().abs().sum()) == 0 for b in blocks)      # every block starts as identity
    assert float(m.stem[1].weight.detach().mean()) == 1.0                 # other BNs: weight 1
    expected_std = (2 / (16 * 9)) ** 0.5                                  # Kaiming normal, fan_out = cout * 9
    assert abs(float(m.stem[0].weight.detach().std()) - expected_std) < 0.2 * expected_std
    bound = 1 / 128 ** 0.5
    assert float(m.fc.weight.abs().max()) <= bound + 1e-6                 # PyTorch default linear init
    assert float(m.fc.weight.abs().max()) > 0.3 * bound                   # i.e. NOT near-zero init


def test_projection_shortcuts_only_where_shape_changes():
    m = build_model(P1_MODEL)
    proj = [n for n, mod in m.named_modules() if isinstance(mod, nn.Sequential) and n.endswith("shortcut")]
    assert sorted(proj) == ["stage2.0.shortcut", "stage3.0.shortcut", "stage4.0.shortcut"]


def test_eval_is_deterministic_dropout_off_and_batch_independent(net_p1):
    x = torch.randn(6, 3, 64, 64)
    assert torch.equal(net_p1(x), net_p1(x))
    assert torch.allclose(net_p1(x)[2:3], net_p1(x[2:3]), atol=1e-5)


def test_dropout_active_in_train_mode():
    m = build_model(ModelCfg((8, 16, 32, 64), 3, dropout=0.5)).train()
    pooled = torch.ones(64, 64)
    assert not torch.equal(m.head(pooled), m.head(pooled))


def test_param_groups_partition():
    m = build_model(P1_MODEL)
    g_decay, g_none = param_groups(m)
    assert "weight_decay" not in g_decay and g_none["weight_decay"] == 0.0
    ids_d, ids_n = {id(p) for p in g_decay["params"]}, {id(p) for p in g_none["params"]}
    assert not ids_d & ids_n and ids_d | ids_n == {id(p) for p in m.parameters()}
    decayed = {n for n, p in m.named_parameters() if id(p) in ids_d}
    assert decayed and all(n.endswith("weight") for n in decayed)
    for n, mod in m.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            assert id(mod.weight) in ids_n and id(mod.bias) in ids_n
    assert id(m.fc.weight) in ids_d and id(m.fc.bias) in ids_n
    opt = torch.optim.AdamW(param_groups(m), lr=1e-3, weight_decay=0.05)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.05, 0.0]


def _save(model, cfg, tmp_path):
    sd = model.state_dict()
    torch.save(sd, tmp_path / "model.pt")
    ModelCard.build(cfg, 64, tuple(f"c{i}" for i in range(cfg.num_classes)), sd, "p1-none-f-1-clean-s0"
                    ).write(tmp_path / "model_card.json")


def test_load_for_inference_round_trip(tmp_path, net_p1):
    _save(net_p1, P1_MODEL, tmp_path)
    m, card = load_for_inference(tmp_path, "cpu")
    assert not m.training and not any(p.requires_grad for p in m.parameters())
    assert card.model_hash == model_hash(m.state_dict()) and card.norm_mean == 0.5 and card.norm_std == 0.25
    assert card.widths == P1_MODEL.widths and card.class_names[0] == "c0" and card.input_size == 64
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(m(x), net_p1(x), atol=1e-6)


def test_load_for_inference_rejects_tampered_weights(tmp_path, net_p1):
    _save(net_p1, P1_MODEL, tmp_path)
    sd = torch.load(tmp_path / "model.pt", weights_only=True)
    sd["fc.bias"] = sd["fc.bias"] + 1e-3
    torch.save(sd, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_for_inference(tmp_path, "cpu")


def test_model_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    m = build_model(ModelCfg((8, 16, 32, 64), 9, dropout=0.0)).train()
    opt = torch.optim.AdamW(param_groups(m), lr=3e-3)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 9, (16,))
    for _ in range(80):
        loss = nn.functional.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3
