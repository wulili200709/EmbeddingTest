from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import algorithms.proxy as qr_core

from .bootstrap import ensure_repo_root_on_path
from .recipe import Line2DupRecipe
from .template_core import (
    apply_affine_to_points,
    expanded_pose_affine,
    load_class_source_assets,
)

ensure_repo_root_on_path()

from line2dup_like_matcher import (  # noqa: E402
    Line2DupLikeDetector,
    Match,
    load_detector_model,
    match_bbox,
    match_quad,
    nms_matches,
)


Point = Tuple[float, float]


@dataclass
class FollowRegion:
    label_name: str
    points: List[Point]
    bbox: Tuple[int, int, int, int]
    source_shape_type: str


@dataclass
class FollowResult:
    match: Match
    regions: List[FollowRegion]
    points: List[Point]
    bbox: Tuple[int, int, int, int]
    source_shape_type: str


def _shape_points_from_labelme(labelme_json_path: str, label_name: str) -> Tuple[str, List[Point]]:
    shape = qr_core.read_shape_from_labelme(labelme_json_path, label_name=label_name)
    if not shape:
        raise RuntimeError(f"Reference json does not contain label '{label_name}'")
    shape_type = str(shape.get("shape_type", "rectangle"))
    pts = shape.get("points", [])
    points = [(float(x), float(y)) for x, y in pts]
    if len(points) < 2:
        raise RuntimeError(f"Invalid label '{label_name}' in {labelme_json_path}")
    if shape_type == "rectangle" and len(points) == 2:
        (x0, y0), (x1, y1) = points
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return shape_type, points


def _shape_points_from_recipe(recipe: Line2DupRecipe) -> Optional[Tuple[str, List[Point]]]:
    raw_points = recipe.reference_points or []
    if not raw_points:
        return None
    points: List[Point] = []
    for pt in raw_points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        points.append((float(pt[0]), float(pt[1])))
    if len(points) < 2:
        return None
    shape_type = str(recipe.reference_shape_type or "rectangle")
    if shape_type == "rectangle" and len(points) == 2:
        (x0, y0), (x1, y1) = points
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return shape_type, points


def _region_specs_from_recipe(
    recipe: Line2DupRecipe,
    ref_img_path: str,
    template_roi_rect,
) -> List[Tuple[str, str, List[Point]]]:
    specs: List[Tuple[str, str, List[Point]]] = []
    raw_regions = recipe.reference_regions or []
    for region in raw_regions:
        if not isinstance(region, dict):
            continue
        label_name = str(region.get("output_label") or region.get("reference_label") or region.get("label") or "").strip()
        shape_type = str(region.get("shape_type", "rectangle"))
        points = [
            (float(pt[0]), float(pt[1]))
            for pt in region.get("points", []) or []
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if not label_name or len(points) < 2:
            continue
        if shape_type == "rectangle" and len(points) == 2:
            (x0, y0), (x1, y1) = points
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        specs.append((label_name, shape_type, points))
    if specs:
        return specs

    recipe_shape = _shape_points_from_recipe(recipe)
    if recipe_shape is not None:
        shape_type, ref_points = recipe_shape
        return [(str(recipe.output_label or "roi"), shape_type, ref_points)]

    ref_json = qr_core.labelme_json_of_image(ref_img_path)
    if ref_img_path and os.path.exists(ref_json):
        try:
            shape_type, ref_points = _shape_points_from_labelme(ref_json, recipe.reference_label)
            return [(str(recipe.output_label or recipe.reference_label or "roi"), shape_type, ref_points)]
        except Exception:
            pass

    ref_points = _rect_points(
        float(template_roi_rect.x),
        float(template_roi_rect.y),
        float(template_roi_rect.w),
        float(template_roi_rect.h),
    )
    return [(str(recipe.output_label or "roi"), "rectangle", ref_points)]


def _bbox_from_points(points: Sequence[Point], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    h, w = image_shape[:2]
    x1 = max(0, int(math.floor(min(xs))))
    y1 = max(0, int(math.floor(min(ys))))
    x2 = min(max(1, w), int(math.ceil(max(xs))))
    y2 = min(max(1, h), int(math.ceil(max(ys))))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _class_ids_for_recipe(detector: Line2DupLikeDetector, recipe: Line2DupRecipe) -> List[str]:
    if recipe.class_id:
        return [recipe.class_id]
    return detector.class_ids()


def _best_match(
    detector: Line2DupLikeDetector,
    scene_bgr: np.ndarray,
    recipe: Line2DupRecipe,
    scene_mask: Optional[np.ndarray] = None,
) -> Match:
    backend = str(recipe.backend or "original").strip().lower()
    if backend in {"fusion", "fusionv2"}:
        scene_mask = None
    matches = detector.match(
        scene_bgr,
        threshold=float(recipe.threshold),
        class_ids=_class_ids_for_recipe(detector, recipe),
        mask=scene_mask,
        backend=backend,
    )
    matches = nms_matches(detector, matches, iou_threshold=float(recipe.nms_iou))
    matches.sort(key=lambda item: float(item.similarity), reverse=True)
    if not matches:
        raise RuntimeError("line2dup did not find any match")
    return matches[0]


def _translated_points(points: Sequence[Point], dx: float, dy: float) -> List[Point]:
    return [(float(x) + dx, float(y) + dy) for x, y in points]


def _rect_points(x: float, y: float, w: float, h: float) -> List[Point]:
    return [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]


def _search_bbox_from_recipe(
    recipe: Line2DupRecipe,
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    raw_points = recipe.search_points or []
    points: List[Point] = []
    for pt in raw_points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        points.append((float(pt[0]), float(pt[1])))
    if len(points) < 2:
        return None
    shape_type = str(recipe.search_shape_type or "rectangle")
    if shape_type == "rectangle" and len(points) == 2:
        (x0, y0), (x1, y1) = points
        points = _rect_points(float(min(x0, x1)), float(min(y0, y1)), float(abs(x1 - x0)), float(abs(y1 - y0)))
    return _bbox_from_points(points, image_shape)


def _translate_match_to_scene(
    detector: Line2DupLikeDetector,
    match: Match,
    dx: int,
    dy: int,
) -> Match:
    translated = Match(
        x=int(match.x) + int(dx),
        y=int(match.y) + int(dy),
        similarity=float(match.similarity),
        class_id=str(match.class_id),
        template_id=int(match.template_id),
        backend=str(match.backend),
        refined_transform=match.refined_transform,
        refined_scale=match.refined_scale,
        refined_angle_deg=match.refined_angle_deg,
        refined_fitness=match.refined_fitness,
        refined_rmse=match.refined_rmse,
        refined_quad=None if match.refined_quad is None else [(float(x) + dx, float(y) + dy) for x, y in match.refined_quad],
    )
    if translated.refined_transform is not None and translated.refined_quad is None:
        cropped_quad = match_quad(detector, match)
        translated.refined_quad = [(float(x) + dx, float(y) + dy) for x, y in cropped_quad]
        translated.refined_transform = None
    return translated


def _follow_by_bbox(
    detector: Line2DupLikeDetector,
    match: Match,
    image_shape: Tuple[int, int],
    *,
    label_name: str = "roi",
) -> FollowResult:
    bbox = tuple(match_bbox(detector, match))
    x, y, w, h = bbox
    points = [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]
    region = FollowRegion(
        label_name=str(label_name or "roi"),
        points=points,
        bbox=_bbox_from_points(points, image_shape),
        source_shape_type="rectangle",
    )
    return FollowResult(
        match=match,
        regions=[region],
        points=region.points,
        bbox=region.bbox,
        source_shape_type=region.source_shape_type,
    )


def _follow_by_affine_roi(
    detector: Line2DupLikeDetector,
    match: Match,
    recipe: Line2DupRecipe,
    ref_img_path: str,
    image_shape: Tuple[int, int],
) -> FollowResult:
    source_info, roi_img, _roi_mask, template_roi_rect, _mask_rects = load_class_source_assets(detector, match.class_id)
    region_specs = _region_specs_from_recipe(recipe, ref_img_path, template_roi_rect)
    meta = detector.get_template_meta(match.class_id, int(match.template_id))
    angle = float(meta.get("angle", 0.0))
    scale = float(meta.get("scale", 1.0))
    affine3, _canvas_size = expanded_pose_affine(roi_img.shape[1], roi_img.shape[0], angle, scale)
    quad = match_quad(detector, match)
    top_left = quad[0]
    regions: List[FollowRegion] = []
    for label_name, _shape_type, ref_points in region_specs:
        local_points = [
            (float(x) - float(template_roi_rect.x), float(y) - float(template_roi_rect.y))
            for x, y in ref_points
        ]
        canvas_points = apply_affine_to_points(affine3, local_points)
        scene_points = _translated_points(canvas_points, float(top_left[0]), float(top_left[1]))
        if match.refined_transform is not None:
            scene_points = apply_affine_to_points(match.refined_transform, scene_points)
        bbox = _bbox_from_points(scene_points, image_shape)
        regions.append(
            FollowRegion(
                label_name=label_name,
                points=scene_points,
                bbox=bbox,
                source_shape_type="polygon",
            )
        )
    primary = regions[0]
    return FollowResult(
        match=match,
        regions=regions,
        points=primary.points,
        bbox=primary.bbox,
        source_shape_type=primary.source_shape_type,
    )


def locate_and_follow(
    scene_bgr: np.ndarray,
    ref_img_path: str,
    recipe: Line2DupRecipe,
    *,
    detector: Optional[Line2DupLikeDetector] = None,
    scene_mask: Optional[np.ndarray] = None,
) -> FollowResult:
    det = detector or load_detector_model(recipe.model_path)
    search_bbox = _search_bbox_from_recipe(recipe, scene_bgr.shape[:2])
    scene_for_match = scene_bgr
    mask_for_match = scene_mask
    offset_x = 0
    offset_y = 0
    if search_bbox is not None:
        offset_x, offset_y, crop_w, crop_h = search_bbox
        scene_for_match = scene_bgr[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w].copy()
        if mask_for_match is not None:
            mask_for_match = mask_for_match[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w].copy()
    match = _best_match(det, scene_for_match, recipe, scene_mask=mask_for_match)
    if offset_x or offset_y:
        match = _translate_match_to_scene(det, match, offset_x, offset_y)
    if recipe.follow_mode == "match_bbox":
        label_name = str(recipe.output_label or "roi")
        if recipe.reference_regions:
            for region in recipe.reference_regions:
                if isinstance(region, dict):
                    label_name = str(region.get("output_label") or region.get("reference_label") or label_name)
                    if label_name:
                        break
        return _follow_by_bbox(det, match, scene_bgr.shape[:2], label_name=label_name)
    return _follow_by_affine_roi(det, match, recipe, ref_img_path, scene_bgr.shape[:2])


__all__ = ["FollowRegion", "FollowResult", "locate_and_follow"]
