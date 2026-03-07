from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class LocatorLossWeights:
    heatmap: float = 1.0
    offset: float = 1.0
    scale: float = 0.5
    angle: float = 0.25
    quality: float = 0.5


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    return (ce * alpha_t * modulating).mean()


def masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    expanded_mask = mask.expand_as(prediction)
    active = expanded_mask.sum()
    if float(active.item()) == 0.0:
        return prediction.new_zeros(())
    if loss_type == "l1":
        loss = F.l1_loss(prediction, target, reduction="none")
    else:
        loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (loss * expanded_mask).sum() / active


def locator_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: LocatorLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or LocatorLossWeights()
    heatmap_loss = focal_bce_with_logits(predictions["heatmap"], targets["heatmap"])
    quality_loss = F.binary_cross_entropy_with_logits(
        predictions["quality"],
        targets["quality"],
    )
    offset_loss = masked_regression_loss(
        predictions["offset"],
        targets["offset"],
        targets["mask"],
    )
    scale_loss = masked_regression_loss(
        predictions["scale"],
        targets["scale"],
        targets["mask"],
    )
    angle_loss = masked_regression_loss(
        predictions["angle"],
        targets["angle"],
        targets["mask"],
    )

    total = (
        heatmap_loss * weights.heatmap
        + quality_loss * weights.quality
        + offset_loss * weights.offset
        + scale_loss * weights.scale
        + angle_loss * weights.angle
    )
    return {
        "total": total,
        "heatmap": heatmap_loss.detach(),
        "quality": quality_loss.detach(),
        "offset": offset_loss.detach(),
        "scale": scale_loss.detach(),
        "angle": angle_loss.detach(),
    }

