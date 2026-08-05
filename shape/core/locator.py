from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

from algorithms.image_io import imread
from common import labelme_io
from common.camera_roles import DEFAULT_CAMERA_ROLE, normalize_camera_role

from .recipe_labels import clearable_roi_labels, output_labels_from_shape_recipe
from .recipe import ShapeRecipe, load_recipe, save_recipe
from .recipe_store import ShapeRecipeStore
from .roi_follow import FollowResult
from .services import RuntimeDetectedShape, ShapeLocateService, runtime_shapes_from_follow_result


@dataclass
class ProductShapePaths:
    product_dir: str
    camera_role: str
    role_dir: str
    model_path: str
    recipe_path: str
    legacy_model_path: str
    legacy_recipe_path: str


@dataclass
class ShapeAutogenRun:
    jpath: str
    result: FollowResult
    locate_ms: float
    total_ms: float


@dataclass(frozen=True)
class RuntimeRoiAutogenRun:
    result: FollowResult
    roi_shapes: tuple[RuntimeDetectedShape, ...]
    locate_ms: float
    total_ms: float


_LOCATE_SERVICE = ShapeLocateService()


def invalidate_shape_model_cache(model_path: str | None = None) -> None:
    """Discard cached shape detectors after a model file is replaced."""
    if model_path:
        _LOCATE_SERVICE.invalidate_model(model_path)
        return
    _LOCATE_SERVICE.clear_cache()


def _delete_stale_shape_roi_shapes(tgt_img_path: str, recipe: ShapeRecipe) -> list[str]:
    jpath = labelme_io.labelme_json_of_image(tgt_img_path)
    if not os.path.exists(jpath):
        return []
    current_labels = output_labels_from_shape_recipe(recipe)
    existing_labels = labelme_io.sorted_label_names_from_labelme(jpath, label_prefix="roi")
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
            if labelme_io.delete_labelme_shape(tgt_img_path, label_name=label):
                removed.append(label)
        except Exception:
            continue
    return removed


def _normalize_camera_role(camera_role: str) -> str:
    return normalize_camera_role(camera_role, default=DEFAULT_CAMERA_ROLE)


def product_paths(product_dir: str, camera_role: str = "cam1") -> ProductShapePaths:
    normalized_role = _normalize_camera_role(camera_role)
    role_dir = os.path.join(product_dir, "shape", normalized_role)
    return ProductShapePaths(
        product_dir=product_dir,
        camera_role=normalized_role,
        role_dir=role_dir,
        model_path=os.path.join(role_dir, "model.json"),
        recipe_path=os.path.join(role_dir, "recipe.json"),
        legacy_model_path=os.path.join(product_dir, "line2dup_model.json"),
        legacy_recipe_path=os.path.join(product_dir, "line2dup_recipe.json"),
    )


def _resolve_recipe_file(paths: ProductShapePaths) -> str:
    return ShapeRecipeStore(paths).resolved_recipe_path()


def _resolve_model_file(paths: ProductShapePaths) -> str:
    return ShapeRecipeStore(paths).resolved_model_path()


def load_recipe_for_product(product_dir: str, camera_role: str = "cam1") -> ShapeRecipe:
    return ShapeRecipeStore(product_paths(product_dir, camera_role)).load()


def resolved_recipe_path_for_product(product_dir: str, camera_role: str = "cam1") -> str:
    return _resolve_recipe_file(product_paths(product_dir, camera_role))


def resolved_model_path_for_product(product_dir: str, camera_role: str = "cam1") -> str:
    return _resolve_model_file(product_paths(product_dir, camera_role))


def save_recipe_for_product(product_dir: str, recipe: ShapeRecipe, camera_role: str = "cam1") -> None:
    ShapeRecipeStore(product_paths(product_dir, camera_role)).save(recipe)


def recipe_is_ready(product_dir: str, camera_role: str = "cam1") -> bool:
    paths = product_paths(product_dir, camera_role)
    recipe_path = _resolve_recipe_file(paths)
    model_path = _resolve_model_file(paths)
    return os.path.exists(model_path) and os.path.exists(recipe_path)


def autogen_roi_json_from_shape_timed(
    tgt_img_path: str,
    ref_img_path: str,
    product_dir: str,
    *,
    camera_role: str = "cam1",
    scene_mask_path: str = "",
) -> ShapeAutogenRun:
    total_t0 = time.perf_counter()
    paths = product_paths(product_dir, camera_role)
    model_path = _resolve_model_file(paths)
    recipe_path = _resolve_recipe_file(paths)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing shape model: {model_path}")
    recipe = load_recipe(recipe_path)
    recipe.model_path = model_path
    if (not ref_img_path) and recipe.reference_image:
        ref_img_path = recipe.reference_image

    scene = imread(tgt_img_path, cv2.IMREAD_COLOR)
    if scene is None:
        raise FileNotFoundError(tgt_img_path)

    scene_mask = None
    if scene_mask_path:
        scene_mask = imread(scene_mask_path, cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(scene_mask_path)

    locate_run = _LOCATE_SERVICE.locate(scene, recipe, ref_img_path=ref_img_path, scene_mask=scene_mask)
    result = locate_run.result
    locate_ms = locate_run.locate_ms
    _delete_stale_shape_roi_shapes(tgt_img_path, recipe)
    jpath = ""
    for region in result.regions:
        if region.source_shape_type == "polygon":
            jpath = labelme_io.upsert_labelme_polygon(
                tgt_img_path,
                [(float(x), float(y)) for x, y in region.points],
                label_name=region.label_name,
            )
        else:
            jpath = labelme_io.upsert_labelme_rect(tgt_img_path, region.bbox, label_name=region.label_name)
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    return ShapeAutogenRun(jpath=jpath, result=result, locate_ms=locate_ms, total_ms=total_ms)


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
        raise FileNotFoundError(f"Missing shape model: {model_path}")
    recipe = load_recipe(recipe_path)
    recipe.model_path = model_path
    if (not ref_img_path) and recipe.reference_image:
        ref_img_path = recipe.reference_image
    if scene_bgr is None:
        raise ValueError("scene_bgr is required")

    locate_run = _LOCATE_SERVICE.locate(scene_bgr, recipe, ref_img_path=ref_img_path, scene_mask=scene_mask)
    result = locate_run.result
    locate_ms = locate_run.locate_ms
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    return RuntimeRoiAutogenRun(
        result=result,
        roi_shapes=runtime_shapes_from_follow_result(result),
        locate_ms=locate_ms,
        total_ms=total_ms,
    )


def autogen_roi_json_from_shape(
    tgt_img_path: str,
    ref_img_path: str,
    product_dir: str,
    *,
    camera_role: str = "cam1",
    scene_mask_path: str = "",
) -> Tuple[str, FollowResult]:
    run = autogen_roi_json_from_shape_timed(
        tgt_img_path,
        ref_img_path,
        product_dir,
        camera_role=camera_role,
        scene_mask_path=scene_mask_path,
    )
    return run.jpath, run.result


__all__ = [
    "ProductShapePaths",
    "ShapeAutogenRun",
    "RuntimeDetectedShape",
    "RuntimeRoiAutogenRun",
    "autogen_roi_json_from_shape",
    "autogen_roi_json_from_shape_timed",
    "autogen_runtime_roi_shapes_timed",
    "invalidate_shape_model_cache",
    "load_recipe_for_product",
    "product_paths",
    "recipe_is_ready",
    "resolved_model_path_for_product",
    "resolved_recipe_path_for_product",
    "save_recipe_for_product",
]

