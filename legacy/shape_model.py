from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from legacy.scaled_shape_model import ScaledShapeModel
except Exception:  # pragma: no cover
    ScaledShapeModel = None

from algorithms.labelme import (
    labelme_json_of_image,
    try_read_polygon_points_from_labelme,
    try_read_xywh_from_labelme,
    upsert_labelme_polygon,
)


def _require_shape_model() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for shape-model localization")
    if ScaledShapeModel is None:
        raise RuntimeError("legacy scaled_shape_model is unavailable")


def _rect_xywh_to_points(xywh: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
    x, y, w, h = xywh
    return [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]


def _shape_points_from_labelme(
    labelme_json_path: str,
    label_name: str,
) -> Optional[List[Tuple[float, float]]]:
    polygon = try_read_polygon_points_from_labelme(labelme_json_path, label_name)
    if polygon and len(polygon) >= 3:
        return polygon
    xywh = try_read_xywh_from_labelme(labelme_json_path, label_name)
    if xywh:
        return _rect_xywh_to_points(xywh)
    return None


def _mask_from_points(
    height: int,
    width: int,
    points: Sequence[Tuple[float, float]],
    fill_value: int = 255,
) -> np.ndarray:
    assert cv2 is not None
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if not points:
        return mask
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Invalid polygon points")
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], int(fill_value))
    return mask


def _transform_points(
    points_xy: Sequence[Tuple[float, float]],
    origin_rc: Tuple[float, float],
    row: float,
    col: float,
    angle: float,
    scale: float,
) -> List[Tuple[float, float]]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    origin_row, origin_col = origin_rc
    output: List[Tuple[float, float]] = []
    for x, y in points_xy:
        dcol = float(x) - float(origin_col)
        drow = float(y) - float(origin_row)
        col_f = float(col) + scale * (cos_a * dcol - sin_a * drow)
        row_f = float(row) + scale * (sin_a * dcol + cos_a * drow)
        output.append((col_f, row_f))
    return output


def create_shape_model_from_reference(
    ref_img_path: str,
    model_path: str,
    *,
    anchor_label: str = "anchor",
    anchor_mask_label: str = "anchor_mask",
    nbins: int = 30,
    canny1: int = 50,
    canny2: int = 150,
) -> str:
    _require_shape_model()
    assert cv2 is not None
    assert ScaledShapeModel is not None

    ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(ref_img_path)

    ref_json = labelme_json_of_image(ref_img_path)
    if not os.path.exists(ref_json):
        raise FileNotFoundError(f"Missing reference json: {ref_json}")

    anchor_points = _shape_points_from_labelme(ref_json, anchor_label)
    if not anchor_points:
        raise RuntimeError("Missing anchor annotation in reference image")

    mask = _mask_from_points(ref.shape[0], ref.shape[1], anchor_points, fill_value=255)
    exclude_points = _shape_points_from_labelme(ref_json, anchor_mask_label)
    if exclude_points:
        exclude = _mask_from_points(ref.shape[0], ref.shape[1], exclude_points, fill_value=255)
        mask[exclude > 0] = 0

    model = ScaledShapeModel.create(ref, mask=mask, nbins=nbins, canny1=canny1, canny2=canny2)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    return model_path


def autogen_roi_json_from_shape_model(
    tgt_img_path: str,
    ref_img_path: str,
    model_path: str,
    *,
    anchor_label: str = "anchor",
    roi_label: str = "roi",
    anchor_mask_label: str = "anchor_mask",
    angle_start: float = -math.pi,
    angle_extent: float = math.pi * 2.0,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    min_score: float = 0.18,
    num_matches: int = 1,
    max_overlap: float = 0.3,
) -> str:
    _require_shape_model()
    assert cv2 is not None
    assert ScaledShapeModel is not None

    if not os.path.exists(model_path):
        create_shape_model_from_reference(
            ref_img_path,
            model_path,
            anchor_label=anchor_label,
            anchor_mask_label=anchor_mask_label,
        )

    model = ScaledShapeModel.load(model_path)

    tgt = cv2.imread(tgt_img_path, cv2.IMREAD_GRAYSCALE)
    if tgt is None:
        raise FileNotFoundError(tgt_img_path)

    rows, cols, angs, scales, _scores = model.find(
        tgt,
        angle_start=angle_start,
        angle_extent=angle_extent,
        scale_min=scale_min,
        scale_max=scale_max,
        min_score=min_score,
        num_matches=num_matches,
        max_overlap=max_overlap,
    )
    if rows.size == 0:
        raise RuntimeError("shape_model did not find any match")

    row = float(rows[0])
    col = float(cols[0])
    angle = float(angs[0])
    scale = float(scales[0])

    ref_json = labelme_json_of_image(ref_img_path)
    if not os.path.exists(ref_json):
        raise FileNotFoundError(f"Missing reference json: {ref_json}")

    roi_points = _shape_points_from_labelme(ref_json, roi_label)
    if not roi_points:
        raise RuntimeError("Missing ROI annotation in reference image")

    anchor_points = _shape_points_from_labelme(ref_json, anchor_label)
    tgt_roi_points = _transform_points(roi_points, model.origin_rc, row, col, angle, scale)
    if anchor_points:
        tgt_anchor_points = _transform_points(anchor_points, model.origin_rc, row, col, angle, scale)
        upsert_labelme_polygon(tgt_img_path, tgt_anchor_points, label_name=anchor_label)

    return upsert_labelme_polygon(tgt_img_path, tgt_roi_points, label_name=roi_label)


__all__ = [
    "autogen_roi_json_from_shape_model",
    "create_shape_model_from_reference",
]
