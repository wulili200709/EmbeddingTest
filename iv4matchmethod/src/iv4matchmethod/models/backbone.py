from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        activation: type[nn.Module] | None = nn.Hardswish,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            layers.append(activation())
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []
        if expanded_channels != in_channels:
            layers.append(
                ConvBnAct(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    activation=activation,
                )
            )
        layers.append(
            ConvBnAct(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                groups=expanded_channels,
            )
        )
        if use_se:
            layers.append(SqueezeExcite(expanded_channels))
        layers.append(
            ConvBnAct(
                expanded_channels,
                out_channels,
                kernel_size=1,
                activation=None,
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_residual:
            return x + y
        return y


class LegacyMobileNetV3SmallBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBnAct(3, 16, kernel_size=3, stride=2, activation=nn.Hardswish)
        self.block1 = InvertedResidual(16, 16, 16, kernel_size=3, stride=2, use_se=True, activation=nn.ReLU)
        self.block2 = InvertedResidual(16, 24, 72, kernel_size=3, stride=2, use_se=False, activation=nn.ReLU)
        self.block3 = InvertedResidual(24, 24, 88, kernel_size=3, stride=1, use_se=False, activation=nn.ReLU)
        self.block4 = InvertedResidual(24, 40, 96, kernel_size=5, stride=2, use_se=True, activation=nn.Hardswish)
        self.block5 = InvertedResidual(40, 40, 240, kernel_size=5, stride=1, use_se=True, activation=nn.Hardswish)
        self.block6 = InvertedResidual(40, 40, 240, kernel_size=5, stride=1, use_se=True, activation=nn.Hardswish)
        self.out_channels = {"stage2": 24, "stage3": 40}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        stage2 = self.block3(x)
        x = self.block4(stage2)
        x = self.block5(x)
        stage3 = self.block6(x)
        return {"stage2": stage2, "stage3": stage3}


class TorchvisionMobileNetV3SmallBackbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.stage2_index = 3
        self.stage3_index = 6
        self.out_channels = {"stage2": 24, "stage3": 40}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage2 = None
        stage3 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == self.stage2_index:
                stage2 = x
            if index == self.stage3_index:
                stage3 = x
        if stage2 is None or stage3 is None:
            raise RuntimeError("failed to extract stage features from torchvision MobileNetV3-Small")
        return {"stage2": stage2, "stage3": stage3}


def build_backbone(backbone_variant: str, pretrained: bool) -> nn.Module:
    if backbone_variant == "legacy":
        return LegacyMobileNetV3SmallBackbone()
    if backbone_variant == "torchvision":
        return TorchvisionMobileNetV3SmallBackbone(pretrained=pretrained)
    raise ValueError(f"unsupported backbone_variant: {backbone_variant}")
