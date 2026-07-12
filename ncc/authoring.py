from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from algorithms.image_io import imread, imwrite

from .locator import resolved_model_path_for_product
from .model import NccMatchModel, NccMatchRect, NccReferenceRegion, resolve_asset_path, save_model


def _ensure_asset_dirs(model_path: str, model: NccMatchModel) -> None:
    for raw_path in (
        model.source_image_path,
        model.template_image_path,
        model.preview_image_path,
        model.mask_image_path,
    ):
        resolve_asset_path(model_path, raw_path).parent.mkdir(parents=True, exist_ok=True)


def _clamp_roi(rect: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in rect]
    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def source_image_path(model_path: str, model: NccMatchModel) -> str:
    return str(resolve_asset_path(model_path, model.source_image_path))


def template_image_path(model_path: str, model: NccMatchModel) -> str:
    return str(resolve_asset_path(model_path, model.template_image_path))


def preview_image_path(model_path: str, model: NccMatchModel) -> str:
    return str(resolve_asset_path(model_path, model.preview_image_path))


def mask_image_path(model_path: str, model: NccMatchModel) -> str:
    return str(resolve_asset_path(model_path, model.mask_image_path))


def ensure_default_assets(model_path: str, model: NccMatchModel) -> None:
    _ensure_asset_dirs(model_path, model)
    save_model(model_path, model)


def set_source_from_image_file(model_path: str, model: NccMatchModel, image_path: str) -> str:
    image = imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    return set_source_from_array(model_path, model, image)


def set_source_from_array(model_path: str, model: NccMatchModel, image_bgr: np.ndarray) -> str:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is required")
    _ensure_asset_dirs(model_path, model)
    target_path = source_image_path(model_path, model)
    if not imwrite(target_path, image_bgr):
        raise RuntimeError(f"Failed to save source image: {target_path}")
    return target_path


def _rect_points_to_polygon(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) < 2:
        return []
    (x0, y0), (x1, y1) = points[:2]
    left = min(float(x0), float(x1))
    top = min(float(y0), float(y1))
    right = max(float(x0), float(x1))
    bottom = max(float(y0), float(y1))
    return [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]


def _mask_polygon_in_template(
    region: NccReferenceRegion | None,
    roi: Tuple[int, int, int, int],
) -> List[Tuple[float, float]]:
    if not isinstance(region, NccReferenceRegion):
        return []
    x, y, _w, _h = [int(v) for v in roi]
    if region.shape_type == "polygon" and len(region.points) >= 3:
        return [(float(px) - x, float(py) - y) for px, py in region.points]
    if len(region.points) >= 2:
        return [(px - x, py - y) for px, py in _rect_points_to_polygon(list(region.points))]
    return []


def _build_template_mask(
    region: NccReferenceRegion | None,
    roi: Tuple[int, int, int, int],
) -> np.ndarray | None:
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None
    points = _mask_polygon_in_template(region, roi)
    if len(points) < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [np.round(contour).astype(np.int32)], 255)
    if int(cv2.countNonZero(mask)) <= 0:
        return None
    return mask


def set_template_from_roi(
    model_path: str,
    model: NccMatchModel,
    roi: Tuple[int, int, int, int],
) -> NccMatchModel:
    _ensure_asset_dirs(model_path, model)
    src_path = source_image_path(model_path, model)
    source = imread(src_path, cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(src_path)

    x, y, w, h = _clamp_roi(roi, source.shape[1], source.shape[0])
    template = source[y : y + h, x : x + w].copy()
    mask_region = model.template_mask if bool(getattr(model, "template_mask_enabled", False)) else None
    mask = _build_template_mask(mask_region, (x, y, w, h))
    if mask is not None:
        template = cv2.bitwise_and(template, template, mask=mask)
    preview = source.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    mask_outline = _mask_polygon_in_template(mask_region, (0, 0, 0, 0))
    if len(mask_outline) >= 3:
        shifted = np.asarray([(px, py) for px, py in mask_outline], dtype=np.float32).reshape((-1, 1, 2))
        cv2.polylines(preview, [np.round(shifted).astype(np.int32)], True, (0, 165, 255), 2, cv2.LINE_AA)

    template_path = template_image_path(model_path, model)
    preview_path = preview_image_path(model_path, model)
    target_mask_path = Path(mask_image_path(model_path, model))
    if not imwrite(template_path, template):
        raise RuntimeError(f"Failed to save template image: {template_path}")
    if not imwrite(preview_path, preview):
        raise RuntimeError(f"Failed to save preview image: {preview_path}")
    if mask is not None:
        if not imwrite(str(target_mask_path), mask):
            raise RuntimeError(f"Failed to save mask image: {target_mask_path}")
    elif target_mask_path.exists():
        target_mask_path.unlink()

    updated = model.normalized()
    updated.template_roi = NccMatchRect(x=x, y=y, width=w, height=h)
    save_model(model_path, updated)
    return updated


def default_model_path(product_dir: str, camera_role: str = "cam1") -> str:
    return resolved_model_path_for_product(product_dir, camera_role)


__all__ = [
    "default_model_path",
    "ensure_default_assets",
    "mask_image_path",
    "preview_image_path",
    "set_source_from_array",
    "set_source_from_image_file",
    "set_template_from_roi",
    "source_image_path",
    "template_image_path",
]
