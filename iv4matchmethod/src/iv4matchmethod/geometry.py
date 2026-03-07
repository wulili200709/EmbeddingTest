from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image


@dataclass(slots=True)
class PosePrediction:
    cx: float
    cy: float
    theta: float
    sx: float
    sy: float
    quality: float
    response_index: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["theta_deg"] = math.degrees(self.theta)
        return payload


def transform_polygon(
    roi_ref_polygon: Iterable[Iterable[float]],
    cx: float,
    cy: float,
    theta: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    pts = np.asarray(list(roi_ref_polygon), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("roi_ref_polygon must be shaped [N, 2]")
    scaled = pts * np.array([sx, sy], dtype=np.float32)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    rotated = scaled @ rotation.T
    translated = rotated + np.array([cx, cy], dtype=np.float32)
    return translated


def order_quad(points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(list(points), dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("expected a quadrilateral with shape [4, 2]")
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]
    return np.stack([tl, bl, br, tr], axis=0)


def extract_aligned_roi(
    image: Image.Image,
    polygon: Iterable[Iterable[float]],
    output_size: tuple[int, int] = (128, 128),
) -> Image.Image:
    pts = np.asarray(list(polygon), dtype=np.float32)
    if pts.shape == (4, 2):
        ordered = order_quad(pts).reshape(-1)
        return image.transform(
            output_size,
            Image.Transform.QUAD,
            data=tuple(float(v) for v in ordered),
            resample=Image.Resampling.BILINEAR,
        )

    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))
    return image.crop((min_x, min_y, max_x, max_y)).resize(
        output_size,
        resample=Image.Resampling.BILINEAR,
    )


def decode_pose(
    outputs: dict[str, torch.Tensor | tuple[int, int]],
) -> list[PosePrediction]:
    heatmap = torch.sigmoid(outputs["heatmap"])
    quality = torch.sigmoid(outputs["quality"])
    offset = outputs["offset"]
    scale = outputs["scale"]
    angle = outputs["angle"]
    search_image_h, search_image_w = outputs["search_image_shape"]
    search_feat_h, search_feat_w = outputs["search_feature_shape"]
    template_feat_h, template_feat_w = outputs["template_feature_shape"]
    stride_y = search_image_h / search_feat_h
    stride_x = search_image_w / search_feat_w

    score = heatmap * quality
    batch_size = score.shape[0]
    flat_index = score.flatten(1).argmax(dim=1)
    poses: list[PosePrediction] = []

    for batch_index in range(batch_size):
        index = int(flat_index[batch_index].item())
        out_w = score.shape[-1]
        y = index // out_w
        x = index % out_w
        dx = float(offset[batch_index, 0, y, x].item())
        dy = float(offset[batch_index, 1, y, x].item())
        log_sx = float(scale[batch_index, 0, y, x].item())
        log_sy = float(scale[batch_index, 1, y, x].item())
        sin_t = float(angle[batch_index, 0, y, x].item())
        cos_t = float(angle[batch_index, 1, y, x].item())
        norm = max(math.sqrt(sin_t * sin_t + cos_t * cos_t), 1e-6)
        sin_t /= norm
        cos_t /= norm

        cx = (x + template_feat_w / 2.0 + dx) * stride_x
        cy = (y + template_feat_h / 2.0 + dy) * stride_y
        poses.append(
            PosePrediction(
                cx=cx,
                cy=cy,
                theta=math.atan2(sin_t, cos_t),
                sx=math.exp(log_sx),
                sy=math.exp(log_sy),
                quality=float(score[batch_index, 0, y, x].item()),
                response_index=(y, x),
            )
        )
    return poses

