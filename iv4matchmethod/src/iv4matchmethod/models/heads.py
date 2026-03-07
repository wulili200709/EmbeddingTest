from __future__ import annotations

from torch import nn

from iv4matchmethod.models.backbone import ConvBnAct


class LocatorHeads(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct(in_channels, hidden_channels, kernel_size=3),
            ConvBnAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.heatmap = nn.Conv2d(hidden_channels, 1, 1)
        self.offset = nn.Conv2d(hidden_channels, 2, 1)
        self.scale = nn.Conv2d(hidden_channels, 2, 1)
        self.angle = nn.Conv2d(hidden_channels, 2, 1)
        self.quality = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)
        nn.init.constant_(self.quality.bias, -2.19)

    def forward(self, x):
        shared = self.stem(x)
        return {
            "heatmap": self.heatmap(shared),
            "offset": self.offset(shared),
            "scale": self.scale(shared),
            "angle": self.angle(shared),
            "quality": self.quality(shared),
        }

