from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from algorithms.image_io import imread
from algorithms.labelme import (
    delete_labelme_shape,
    delete_labelme_shapes,
    labelme_json_of_image,
    list_shapes_from_labelme,
    polygon_points_to_labelme_shape,
    upsert_labelme_shapes,
)
from domain import clearable_roi_labels

from .model import NccMatchModel, NccReferenceRegion, load_model, save_model
from .runtime_service import NccCompiledModel


@dataclass(frozen=True)
class ProductNccPaths:
    product_dir: str
    camera_role: str
    role_dir: str
    model_path: str
    legacy_model_path: str


@dataclass(frozen=True)
class NccAutogenRun:
    jpath: str
    written_labels: tuple[str, ...]
    locate_ms: float
    total_ms: float


def _normalize_camera_role(camera_role: str) -> str:
    role = str(camera_role or "").strip().lower()
    return role if role in {"cam1", "cam2"} else "cam1"


def product_paths(product_dir: str, camera_role: str = "cam1") -> ProductNccPaths:
    normalized_role = _normalize_camera_role(camera_role)
    role_dir = os.path.join(product_dir, "ncc", normalized_role)
    return ProductNccPaths(
        product_dir=product_dir,
        camera_role=normalized_role,
        role_dir=role_dir,
        model_path=os.path.join(role_dir, "model.json"),
        legacy_model_path=os.path.join(product_dir, "ncc_model.json"),
    )


def resolved_model_path_for_product(product_dir: str, camera_role: str = "cam1") -> str:
    paths = product_paths(product_dir, camera_role)
    if os.path.exists(paths.model_path):
        return paths.model_path
    if os.path.exists(paths.legacy_model_path):
        return paths.legacy_model_path
    return paths.model_path


def model_is_ready(product_dir: str, camera_role: str = "cam1") -> bool:
    return os.path.exists(resolved_model_path_for_product(product_dir, camera_role))


def load_model_for_product(product_dir: str, camera_role: str = "cam1") -> NccMatchModel:
    return load_model(resolved_model_path_for_product(product_dir, camera_role))


def save_model_for_product(product_dir: str, model: NccMatchModel, camera_role: str = "cam1") -> str:
    paths = product_paths(product_dir, camera_role)
    os.makedirs(paths.role_dir, exist_ok=True)
    save_model(paths.model_path, model)
    return paths.model_path


def inspection_item_specs_from_ncc_model(model: NccMatchModel | None) -> List[Dict[str, str]]:
    normalized = (model or NccMatchModel()).normalized()
    specs: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    for index, region in enumerate(list(normalized.reference_regions or []), start=1):
        label_name = str(region.label_name or "").strip() or f"roi{index}"
        if label_name in seen_labels:
            continue
        display_name = str(region.display_name or label_name).strip() or label_name
        specs.append({
            "roi_label": label_name,
            "display_name": display_name,
        })
        seen_labels.add(label_name)
    return specs or [{"roi_label": "roi", "display_name": "roi"}]


def inspection_item_specs_for_product(product_dir: str, camera_role: str = "cam1") -> List[Dict[str, str]]:
    return inspection_item_specs_from_ncc_model(load_model_for_product(product_dir, camera_role))


def output_labels_from_ncc_model(model: NccMatchModel | None) -> List[str]:
    return [spec["roi_label"] for spec in inspection_item_specs_from_ncc_model(model)]


def output_labels_for_product(product_dir: str, camera_role: str = "cam1") -> List[str]:
    return output_labels_from_ncc_model(load_model_for_product(product_dir, camera_role))


def display_names_by_label_for_product(product_dir: str, camera_role: str = "cam1") -> Dict[str, str]:
    return {
        str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
        for spec in inspection_item_specs_for_product(product_dir, camera_role)
        if str(spec.get("roi_label", "")).strip()
    }


def _region_polygon_points(region: NccReferenceRegion) -> List[Tuple[float, float]]:
    if region.shape_type == "polygon" and len(region.points) >= 3:
        return [(float(x), float(y)) for x, y in region.points]
    if len(region.points) >= 2:
        (x0, y0), (x1, y1) = region.points[:2]
        x_min = min(float(x0), float(x1))
        y_min = min(float(y0), float(y1))
        x_max = max(float(x0), float(x1))
        y_max = max(float(y0), float(y1))
        return [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
    return []


def _template_roi_quad(model: NccMatchModel) -> np.ndarray | None:
    rect = model.template_roi.normalized()
    if rect.width <= 0 or rect.height <= 0:
        return None
    return np.asarray(
        [
            [float(rect.x), float(rect.y)],
            [float(rect.x + rect.width), float(rect.y)],
            [float(rect.x + rect.width), float(rect.y + rect.height)],
            [float(rect.x), float(rect.y + rect.height)],
        ],
        dtype=np.float32,
    )


def _project_reference_regions(model: NccMatchModel, match_quad: Sequence[Tuple[float, float]]) -> List[Tuple[str, List[Tuple[float, float]]]]:
    normalized = model.normalized()
    if not normalized.reference_regions:
        return []
    src_quad = _template_roi_quad(normalized)
    dst_quad = np.asarray(match_quad, dtype=np.float32)
    if src_quad is None or dst_quad.shape != (4, 2):
        return []
    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
    projected: List[Tuple[str, List[Tuple[float, float]]]] = []
    for index, region in enumerate(list(normalized.reference_regions or []), start=1):
        points = _region_polygon_points(region)
        if len(points) < 3:
            continue
        src = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        dst = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
        label_name = str(region.label_name or "").strip() or f"roi{index}"
        projected.append((label_name, [(float(x), float(y)) for x, y in dst]))
    return projected


def _existing_shape_labels(tgt_img_path: str, current_labels: Sequence[str]) -> List[str]:
    jpath = labelme_json_of_image(tgt_img_path)
    if not os.path.exists(jpath):
        return []
    current_set = {str(label).strip() for label in list(current_labels or []) if str(label).strip()}
    labels: List[str] = []
    seen: set[str] = set()
    for shape in list_shapes_from_labelme(jpath):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if not label or label in {"anchor", "anchor_mask"} or label in seen:
            continue
        if label not in current_set and label != "roi" and not label.startswith("roi"):
            continue
        labels.append(label)
        seen.add(label)
    return labels


def _delete_stale_ncc_roi_shapes(tgt_img_path: str, model: NccMatchModel) -> List[str]:
    current_labels = output_labels_from_ncc_model(model)
    existing_labels = _existing_shape_labels(tgt_img_path, current_labels)
    labels_to_clear, clear_mode = clearable_roi_labels(
        current_labels,
        existing_labels,
        prefer_stale_only=True,
    )
    if clear_mode != "stale_only":
        return []
    removed: List[str] = []
    for label in labels_to_clear:
        try:
            if delete_labelme_shape(tgt_img_path, label_name=label):
                removed.append(label)
        except Exception:
            continue
    return removed


def _delete_current_ncc_roi_shapes(tgt_img_path: str, model: NccMatchModel) -> List[str]:
    labels = [str(label).strip() for label in output_labels_from_ncc_model(model) if str(label).strip()]
    if "roi" not in labels:
        labels.append("roi")
    if not labels:
        return []
    try:
        removed_count = int(delete_labelme_shapes(tgt_img_path, labels))
    except Exception:
        return []
    return labels[:removed_count] if removed_count > 0 else []


def autogen_roi_json_from_ncc_timed(
    tgt_img_path: str,
    product_dir: str,
    *,
    camera_role: str = "cam1",
    model_path: str | None = None,
    model: NccMatchModel | None = None,
    compiled_model: NccCompiledModel | None = None,
) -> NccAutogenRun:
    total_t0 = time.perf_counter()
    active_model_path = model_path or resolved_model_path_for_product(product_dir, camera_role)
    if not os.path.exists(active_model_path):
        raise FileNotFoundError(f"Missing NCC model: {active_model_path}")

    active_model = (model or getattr(compiled_model, "model", None) or load_model(active_model_path)).normalized()
    scene = imread(tgt_img_path, cv2.IMREAD_COLOR)
    if scene is None:
        raise FileNotFoundError(tgt_img_path)

    compiled = compiled_model or NccCompiledModel(active_model_path, active_model)
    should_close = compiled_model is None
    try:
        locate_t0 = time.perf_counter()
        response = compiled.match(
            scene,
            options=active_model.options,
            search_roi=active_model.search_roi.to_xywh() if active_model.search_roi is not None else None,
        )
        locate_ms = (time.perf_counter() - locate_t0) * 1000.0
    finally:
        if should_close:
            compiled.close()

    if not response.matches:
        _delete_current_ncc_roi_shapes(tgt_img_path, active_model)
        raise RuntimeError("NCC did not find any match.")

    _delete_stale_ncc_roi_shapes(tgt_img_path, active_model)

    top1 = response.matches[0]
    projected = _project_reference_regions(active_model, top1.quad)
    written_labels: List[str] = []
    jpath = ""
    if projected:
        shapes: List[dict] = []
        for label_name, points in projected:
            shapes.append(polygon_points_to_labelme_shape(points, label_name=label_name))
            written_labels.append(label_name)
        jpath = upsert_labelme_shapes(tgt_img_path, shapes)
    else:
        shape = polygon_points_to_labelme_shape(list(top1.quad), label_name="roi")
        jpath = upsert_labelme_shapes(tgt_img_path, [shape])
        written_labels.append("roi")

    total_ms = (time.perf_counter() - total_t0) * 1000.0
    return NccAutogenRun(
        jpath=jpath,
        written_labels=tuple(written_labels),
        locate_ms=locate_ms,
        total_ms=total_ms,
    )


__all__ = [
    "NccAutogenRun",
    "ProductNccPaths",
    "autogen_roi_json_from_ncc_timed",
    "display_names_by_label_for_product",
    "inspection_item_specs_for_product",
    "inspection_item_specs_from_ncc_model",
    "load_model_for_product",
    "model_is_ready",
    "output_labels_for_product",
    "output_labels_from_ncc_model",
    "product_paths",
    "resolved_model_path_for_product",
    "save_model_for_product",
]
