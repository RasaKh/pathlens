"""PatchNet: small ResNet-style CNN, written so attribution needs no hooks.

Architecture doc, "Model: PatchNet". All ReLUs use inplace=False.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import ModelCfg
from .identity import model_hash
from .tensors import NORM_MEAN, NORM_STD


class BasicBlock(nn.Module):
    """conv3x3(stride s)-BN-ReLU-conv3x3-BN + shortcut, then ReLU."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut: nn.Module
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return self.relu(out + self.shortcut(x))


class PatchNet(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        w = cfg.widths
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]),
                                  nn.ReLU(inplace=False))
        self.stage1 = nn.Sequential(BasicBlock(w[0], w[0], 1), BasicBlock(w[0], w[0], 1))
        self.stage2 = nn.Sequential(BasicBlock(w[0], w[1], 2), BasicBlock(w[1], w[1], 1))
        self.stage3 = nn.Sequential(BasicBlock(w[1], w[2], 2), BasicBlock(w[2], w[2], 1))
        self.stage4 = nn.Sequential(BasicBlock(w[2], w[3], 2), BasicBlock(w[3], w[3], 1))
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(w[3], cfg.num_classes)     # PyTorch default init
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)           # each block starts as an identity

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (fmap [B,C,h,w] from stage4, pooled [B,C])."""
        fmap = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return fmap, fmap.mean(dim=(2, 3))

    def head(self, pooled: Tensor) -> Tensor:
        """logits [B,num_classes]"""
        return self.fc(self.dropout(pooled))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x)[1])


def build_model(cfg: ModelCfg) -> PatchNet:
    return PatchNet(cfg)


def param_groups(model: nn.Module) -> list[dict]:
    """decay: conv and linear weights; no decay: BN params and biases (weight_decay=0.0 set explicitly).

    The decay group inherits the optimiser's weight_decay.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            is_weight = name == "weight" and isinstance(module, (nn.Conv2d, nn.Linear))
            (decay if is_weight else no_decay).append(p)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


@dataclass(frozen=True)
class ModelCard:
    widths: tuple[int, int, int, int]
    num_classes: int
    input_size: int
    class_names: tuple[str, ...]
    norm_mean: float
    norm_std: float
    model_hash: str
    run_id: str

    @classmethod
    def build(cls, cfg: ModelCfg, input_size: int, class_names: tuple[str, ...], state_dict, run_id: str
              ) -> ModelCard:
        return cls(widths=cfg.widths, num_classes=cfg.num_classes, input_size=input_size,
                   class_names=tuple(class_names), norm_mean=NORM_MEAN, norm_std=NORM_STD,
                   model_hash=model_hash(state_dict), run_id=run_id)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def read(cls, path: Path) -> ModelCard:
        d = json.loads(Path(path).read_text())
        d["widths"] = tuple(d["widths"])
        d["class_names"] = tuple(d["class_names"])
        return cls(**d)


def load_for_inference(model_dir: Path, device: torch.device | str) -> tuple[PatchNet, ModelCard]:
    """Load model.pt + model_card.json; eval(); requires_grad_(False) on all params;
    verify model_hash against the card (ValueError on mismatch)."""
    model_dir = Path(model_dir)
    card = ModelCard.read(model_dir / "model_card.json")
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    actual = model_hash(state)
    if actual != card.model_hash:
        raise ValueError(f"model hash mismatch in {model_dir}: card {card.model_hash}, weights {actual}")
    model = build_model(ModelCfg(widths=card.widths, num_classes=card.num_classes))
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, card
