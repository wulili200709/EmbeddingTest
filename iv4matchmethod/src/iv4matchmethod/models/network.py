from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from iv4matchmethod.config import ModelConfig
from iv4matchmethod.models.backbone import ConvBnAct, build_backbone
from iv4matchmethod.models.heads import LocatorHeads
from iv4matchmethod.models.xcorr import depthwise_xcorr


class TemplateConditionedLocator(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()

        self.backbone = build_backbone(
            backbone_variant=self.config.backbone_variant,
            pretrained=self.config.pretrained_backbone,
        )
        half_channels = self.config.fuse_channels // 2
        self.stage2_proj = ConvBnAct(self.backbone.out_channels["stage2"], half_channels, kernel_size=1)
        self.stage3_proj = ConvBnAct(self.backbone.out_channels["stage3"], half_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBnAct(self.config.fuse_channels, self.config.fuse_channels, kernel_size=3),
            ConvBnAct(self.config.fuse_channels, self.config.head_channels, kernel_size=3),
        )
        self.heads = LocatorHeads(self.config.head_channels, self.config.head_channels)

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        template_feats = self.backbone(template)
        search_feats = self.backbone(search)

        template_stage2 = F.normalize(template_feats["stage2"], dim=1)
        search_stage2 = F.normalize(search_feats["stage2"], dim=1)
        template_stage3 = F.normalize(template_feats["stage3"], dim=1)
        search_stage3 = F.normalize(search_feats["stage3"], dim=1)

        response_stage2 = depthwise_xcorr(search_stage2, template_stage2)
        response_stage3 = depthwise_xcorr(search_stage3, template_stage3)

        fused_stage2 = self.stage2_proj(response_stage2)
        fused_stage3 = self.stage3_proj(response_stage3)
        fused_stage3 = F.interpolate(
            fused_stage3,
            size=fused_stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fuse(torch.cat([fused_stage2, fused_stage3], dim=1))
        predictions = self.heads(fused)
        predictions["search_image_shape"] = search.shape[-2:]
        predictions["search_feature_shape"] = search_stage2.shape[-2:]
        predictions["template_feature_shape"] = template_stage2.shape[-2:]
        return predictions
