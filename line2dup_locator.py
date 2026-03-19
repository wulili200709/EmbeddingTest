from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

import qr_core
from line2dup_recipe import Line2DupRecipe, load_recipe, save_recipe
from line2dup_roi_follow import FollowResult, locate_and_follow


@dataclass
class ProductLine2DupPaths:
    product_dir: str
    model_path: str
    recipe_path: str


@dataclass
class Line2DupAutogenRun:
    jpath: str
    result: FollowResult
    locate_ms: float
    total_ms: float


def product_paths(product_dir: str) -> ProductLine2DupPaths:
    return ProductLine2DupPaths(
        product_dir=product_dir,
        model_path=os.path.join(product_dir, "line2dup_model.json"),
        recipe_path=os.path.join(product_dir, "line2dup_recipe.json"),
    )


def load_recipe_for_product(product_dir: str) -> Line2DupRecipe:
    paths = product_paths(product_dir)
    recipe = load_recipe(paths.recipe_path)
    if not recipe.model_path:
        recipe.model_path = paths.model_path
    return recipe


def save_recipe_for_product(product_dir: str, recipe: Line2DupRecipe) -> None:
    paths = product_paths(product_dir)
    if not recipe.model_path:
        recipe.model_path = paths.model_path
    save_recipe(recipe, paths.recipe_path)


def recipe_is_ready(product_dir: str) -> bool:
    paths = product_paths(product_dir)
    return os.path.exists(paths.model_path) and os.path.exists(paths.recipe_path)


def autogen_roi_json_from_line2dup_timed(
    tgt_img_path: str,
    ref_img_path: str,
    product_dir: str,
    *,
    scene_mask_path: str = "",
) -> Line2DupAutogenRun:
    total_t0 = time.perf_counter()
    paths = product_paths(product_dir)
    if not os.path.exists(paths.model_path):
        raise FileNotFoundError(f"Missing line2dup model: {paths.model_path}")
    recipe = load_recipe(paths.recipe_path)
    recipe.model_path = paths.model_path
    if (not ref_img_path) and recipe.reference_image:
        ref_img_path = recipe.reference_image

    scene = cv2.imread(tgt_img_path, cv2.IMREAD_COLOR)
    if scene is None:
        raise FileNotFoundError(tgt_img_path)

    scene_mask = None
    if scene_mask_path:
        scene_mask = cv2.imread(scene_mask_path, cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(scene_mask_path)

    locate_t0 = time.perf_counter()
    result = locate_and_follow(scene, ref_img_path, recipe, scene_mask=scene_mask)
    locate_ms = (time.perf_counter() - locate_t0) * 1000.0
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


def autogen_roi_json_from_line2dup(
    tgt_img_path: str,
    ref_img_path: str,
    product_dir: str,
    *,
    scene_mask_path: str = "",
) -> Tuple[str, FollowResult]:
    run = autogen_roi_json_from_line2dup_timed(
        tgt_img_path,
        ref_img_path,
        product_dir,
        scene_mask_path=scene_mask_path,
    )
    return run.jpath, run.result


__all__ = [
    "ProductLine2DupPaths",
    "Line2DupAutogenRun",
    "autogen_roi_json_from_line2dup",
    "autogen_roi_json_from_line2dup_timed",
    "load_recipe_for_product",
    "product_paths",
    "recipe_is_ready",
    "save_recipe_for_product",
]
