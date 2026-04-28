"""
CIFAR ResNet-18 with gated spatial (Dropout2d) and head (Dropout) for MC Dropout experiments.
Stem: 3x3 stride 1, no MaxPool (Identity). Not compatible with ImageNet ResNet-18 stem.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

DROPOUT_P = 0.3

Mode = Literal["none", "all", "first_block", "last_layer_strict", "last_block_plus_fc"]

MODE_CONFIG: dict[str, tuple[bool, bool, bool, bool, bool]] = {
    # (layer1, layer2, layer3, layer4, pre_fc)
    "none": (False, False, False, False, False),
    "all": (True, True, True, True, True),
    "first_block": (True, False, False, False, False),
    "last_layer_strict": (False, False, False, False, True),
    "last_block_plus_fc": (False, False, False, True, True),
}


def _mode_dropout_flags(mode: str) -> tuple[bool, bool, bool, bool, bool]:
    if mode not in MODE_CONFIG:
        raise ValueError(f"mode must be one of {list(MODE_CONFIG)}, got {mode!r}")
    return MODE_CONFIG[mode]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=DROPOUT_P) if use_dropout else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


def _make_layer(
    in_planes: int,
    planes: int,
    num_blocks: int,
    stride: int,
    use_dropout: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        BasicBlock(in_planes, planes, stride, use_dropout=use_dropout),
    ]
    in_planes = planes
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(in_planes, planes, stride=1, use_dropout=use_dropout))
    return nn.Sequential(*layers)


class CIFARResNet18(nn.Module):
    """
    ResNet-18 for 32x32 inputs. Stem: 3x3 conv stride 1, Identity instead of MaxPool.
    """

    def __init__(self, num_classes: int = 10, mode: str = "none") -> None:
        super().__init__()
        self.mode = mode
        d1, d2, d3, d4, d_fc = _mode_dropout_flags(mode)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = _make_layer(64, 64, 2, stride=1, use_dropout=d1)
        self.layer2 = _make_layer(64, 128, 2, stride=2, use_dropout=d2)
        self.layer3 = _make_layer(128, 256, 2, stride=2, use_dropout=d3)
        self.layer4 = _make_layer(256, 512, 2, stride=2, use_dropout=d4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout_fc = nn.Dropout(p=DROPOUT_P) if d_fc else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x


def enable_mc_dropout(model: nn.Module) -> None:
    """Freeze BN (eval) but keep Dropout / Dropout2d stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def build_model(mode: str, num_classes: int = 10) -> CIFARResNet18:
    return CIFARResNet18(num_classes=num_classes, mode=mode)
