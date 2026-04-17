from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2

import algorithms.proxy as qr_core
from algorithms.image_io import imread
from domain import clearable_roi_labels, output_labels_from_line2dup_recipe

from .recipe import Line2DupRecipe, load_recipe, save_recipe
from .roi_follow import FollowResult, locate_and_follow


@dataclass
class ProductLine2DupPaths:
    product_dir: str
    camera_role: str
    role_dir: str
    model_path: str
    recipe_path: str
    legacy_model_path: str
    legacy_recipe_path: str


@dataclass
class Line2DupAutogenRun:
    jpath: str
    result: FollowResult
    locate_ms: float
    total_ms: float


@dataclass(frozen=True)
class RuntimeDetectedShape:
    label_name: str
    shape_type: str
    points: tuple[Tuple[float, float], ...]
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class RuntimeRoiAutogenRun:
    result: FollowResult
    roi_shapes: tuple[RuntimeDetectedShape, ...]
    locate_ms: float
    total_ms: float


def _delete_stale_line2dup_roi_shapes(tgt_img_path: str, recipe: Line2DupRecipe) -> list[str]:
    jpath = qr_core.labelme_json_of_image(tgt_img_path)
    if not os.path.exists(jpath):
        return []
    current_labels = output_labels_from_line2dup_recipe(recipe)
    existing_labels = qr_core.sorted_label_names_from_labelme(jpath, label_prefix="roi")
    labels_to_clear, clear_mode = clearable_roi_labels(
        current_labels,
        existing_labels,
        prefer_stale_only=True,
    )
    if clear_mode != "stale_only":
        return []
    removed: list[str] = []
    for label in labels_to_clear:
        try:
            if qr_core.delete_labelme_shape(tgt_img_path, label_name=label):
                removed.append(label)
        except Exception:
            continue
    return removed


def _normalize_camera_role(camera_role: str) -> str:
    role = str(camera_role or "").strip().lower()
    return role if role in {"cam1", "cam2"} else "cam1"


def product_paths(product_dir: str, camera_role: str = "cam1") -> ProductLine2DupPaths:
    normalized_role = _normalize_camera_role(camera_role)
    role_dir = os.path.join(product_dir, "line2dup", normalized_role)
    return ProductLine2DupPaths(
        product_dir=product_dir,
        camera_role=normalized_role,
        role_dir=role_dir,
        model_path=os.path.join(role_dir, "model.json"),
        recipe_path=os.path.join(role_dir, "recipe.json"),
        legacy_model_path=os.path.join(product_dir, "line2dup_model.json"),
        legacy_recipe_path=os.path.join(product_dir, "line2dup_recipe.json"),
    )


def _resolve_recipe_file(paths: ProductLine2DupPaths) -> str:
    if os.path.exists(paths.recipe_path):
        return paths.recipe_path
    if os.path.exists(paths.legacy_recipe_path):
        return paths.legacy_recipe_path
    return paths.recipe_path


def _resolve_model_file(paths: ProductLine2DupPaths) -> str:
    if os.path.exists(paths.model_path):
        return paths.model_path
    if os.path.exists(paths.legacy_model_path):
        return paths.legacy_model_path
    return paths.model_path


def load_recipe_for_product(product_dir: str, camera_role: str = "cam1") -> Line2DupRecipe:
    paths = product_paths(product_dir, camera_role)
    recipe = load_recipe(_resolve_recipe_file(paths))
    recipe.model_path = _resolve_model_file(paths)
    return recipe


def resolved_recipe_path_for_product(product_dir: str, camera_role: str = "cam1") -> str:
    return _resolve_recipe_file(product_paths(product_dir, camera_role))


def resolved_model_path_for_product(product_dir: str, camera_role: str = "cam1") -> str:
    return _resolve_model_file(product_paths(product_dir, camera_role))


def save_recipe_for_product(product_dir: str, recipe: Line2DupRecipe, camera_role: str = "cam1") -> None:
    paths = product_paths(product_dir, camera_role)
    os.makedirs(paths.role_dir, exist_ok=True)
    recipe.model_path = paths.model_path
    save_recipe(recipe, paths.recipe_path)


def recipe_is_ready(product_dir: str, camera_role: str = "cam1") -> bool:
    paths = product_paths(product_dir, camera_role)
    recipe_path = _resolve_recipe_file(paths)
    model_path = _resolve_model_file(paths)
    return os.path.exists(model_path) and os.path.exists(recipe_path)


def _runtime_shapes_from_follow_result(result: FollowResult) -> tuple[RuntimeDetectedShape, ...]:
    shapes: list[RuntimeDetectedShape] = []
    for region in result.regions:
        points = tuple((float(x), float(y)) for x, y in region.points)
        if not points:
            continue
        shape_type = str(region.source_shape_type or "rectangle")
        if shape_type == "rectangle" and len(points) == 4:
            shape_points = (points[0], points[2])
        else:
            shape_points = points
        shapes.append(
            RuntimeDetectedShape(
                label_name=str(region.label_name or "").strip() or "roi",
                shape_type=shape_type,
                points=shape_points,
                bbox=tuple(int(value) for value in region.bbox),
            )
        )
    return tuple(shapes)


def autogen_roi_json_from_line2dup_timed(
    tgt_img_path: str,
    ref_img_path: str,
    product_dir: str,
    *,
    camera_role: str = "cam1",
    scene_mask_path: str = "",
    recipe: Optional[Line2DupRecipe] = None,
    detector: Any = None,
) -> Line2DupAutogenRun:
    total_t0 = time.perf_counter()
    paths = product_paths(product_dir, camera_role)
    model_path = _resolve_model_file(paths)
    recipe_path = _resolve_recipe_file(paths)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing line2dup model: {model_path}")
    active_recipe = recipe or load_recipe(recipe_path)
    active_recipe.model_path = model_path
    if (not ref_img_path) and active_recipe.reference_image:
        ref_img_path = active_recipe.reference_image

    scene = imread(tgt_img_path, cv2.IMREAD_COLOR)
    if scene is None:
        raise FileNotFoundError(tgt_img_path)

    scene_mask = None
    if scene_mask_path:
        scene_mask = imread(scene_mask_path, cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(scene_mask_path)

    locate_t0 = time.perf_counter()
    result = locate_and_follow(scene, ref_img_path, active_recipe, detector=detector, scene_mask=scene_mask)
    locate_ms = (time.perf_counter() - locate_t0) * 1000.0
    _delete_stale_line2dup_roi_shapes(tgt_img_path, active_recipe)
    jpath = ""
    for region in result.regions:
        if region.source_shape_type == "polygon":
            jpath = qr_core.upsert_labelme_polygon(
                tgt_img_path,
                [(float(x), float(y)) for x, y in region.points],
                label_name=region.label_name,
            )
        else:
            jpath = qr_core.upsert_labelme_rect(tgt_img_path, region.bbox, label_name=region.label_name)
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    return Line2DupAutogenRun(jpath=jpath, result=result, locate_ms=locate_ms, total_ms=total_ms)


def autogen_runtime_roi_shapes_timed(
    scene_bgr: np.ndarray,
    ref_img_path: str,
    product_dir: str,
    *,
    camera_role: str = "cam1",
    scene_mask: Optional[np.ndarray] = None,
) -> RuntimeRoiAutogenRun:
    total_t0 = time.perf_counter()
    paths = product_paths(product_dir, camera_role)
    model_path = _resolve_model_file(paths)
    recipe_path = _resolve_recipe_file(paths)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing line2dup model: {model_path}")
    recipe = load_recipe(recipe_path)
    recipe.model_path = model_path
    if (not ref_img_path) and recipe.reference_image:
        ref_img_path = recipe.reference_image
    if scene_bgr is None:
        raise ValueError("scene_bgr is required")

    locate_t0 = time.perf_counter()
    result = locate_and_follow(scene_bgr, ref_img_path, recipe, scene_mask=scene_mask)
    locate_ms = (time.perf_counter() - locate_t0) * 1000.0
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    return RuntimeRoiAutogenRun(
        result=result,
        roi_shapes=_runtime_shapes_from_follow_result(result),
        locate_ms=locate_ms,
        total_ms=total_ms,
    )


def autogen_roi_json_from_line2dup(
    tgt_img_path: str,
    ref_img_path: str,
    product_dir: str,
    *,
    camera_role: str = "cam1",
    scene_mask_path: str = "",
) -> Tuple[str, FollowResult]:
    run = autogen_roi_json_from_line2dup_timed(
        tgt_img_path,
        ref_img_path,
        product_dir,
        camera_role=camera_role,
        scene_mask_path=scene_mask_path,
    )
    return run.jpath, run.result


__all__ = [
    "ProductLine2DupPaths",
    "Line2DupAutogenRun",
    "RuntimeDetectedShape",
    "RuntimeRoiAutogenRun",
    "autogen_roi_json_from_line2dup",
    "autogen_roi_json_from_line2dup_timed",
    "autogen_runtime_roi_shapes_timed",
    "load_recipe_for_product",
    "product_paths",
    "recipe_is_ready",
    "resolved_model_path_for_product",
    "resolved_recipe_path_for_product",
    "save_recipe_for_product",
]
