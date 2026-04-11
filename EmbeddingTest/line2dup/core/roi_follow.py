from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import algorithms.proxy as qr_core
from algorithms.image_io import imread

from .bootstrap import ensure_repo_root_on_path
from .recipe import Line2DupRecipe, format_array_roi_label, recipe_array_count, recipe_array_pitch
from .template_core import (
    apply_affine_to_points,
    expanded_pose_affine,
    load_class_source_assets,
)

ensure_repo_root_on_path()

from ..like_matcher import (  # noqa: E402
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
    instance_count: int = 1
    expected_instance_count: int = 1
    matches: Optional[List[Match]] = None
    transform_mode: str = "affine"


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


def _to_gray(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    *,
    pad_ratio: float = 0.18,
    min_pad: int = 12,
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad = max(int(min_pad), int(round(max(float(w), float(h)) * float(pad_ratio))))
    img_h, img_w = image_shape[:2]
    x1 = max(0, int(x) - pad)
    y1 = max(0, int(y) - pad)
    x2 = min(int(img_w), int(x + w) + pad)
    y2 = min(int(img_h), int(y + h) + pad)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0
    union_area = float(max(1, aw * ah) + max(1, bw * bh)) - inter_area
    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def _is_reasonable_projective_refine(
    coarse_bbox: Tuple[int, int, int, int],
    refined_bbox: Tuple[int, int, int, int],
) -> bool:
    coarse_area = float(max(1, coarse_bbox[2] * coarse_bbox[3]))
    refined_area = float(max(1, refined_bbox[2] * refined_bbox[3]))
    area_ratio = refined_area / coarse_area
    if area_ratio < 0.2 or area_ratio > 5.0:
        return False

    coarse_cx = float(coarse_bbox[0]) + float(coarse_bbox[2]) * 0.5
    coarse_cy = float(coarse_bbox[1]) + float(coarse_bbox[3]) * 0.5
    refined_cx = float(refined_bbox[0]) + float(refined_bbox[2]) * 0.5
    refined_cy = float(refined_bbox[1]) + float(refined_bbox[3]) * 0.5
    center_shift = math.hypot(refined_cx - coarse_cx, refined_cy - coarse_cy)
    max_shift = max(20.0, 0.45 * math.hypot(float(coarse_bbox[2]), float(coarse_bbox[3])))
    if center_shift > max_shift:
        return False

    overlap = _bbox_iou(coarse_bbox, refined_bbox)
    if overlap <= 0.02 and center_shift > max(10.0, 0.25 * max(float(coarse_bbox[2]), float(coarse_bbox[3]))):
        return False
    return True


def _class_ids_for_recipe(detector: Line2DupLikeDetector, recipe: Line2DupRecipe) -> List[str]:
    if recipe.class_id:
        return [recipe.class_id]
    return detector.class_ids()


def _match_center(detector: Line2DupLikeDetector, match: Match) -> Point:
    quad = match_quad(detector, match)
    xs = [float(x) for x, _ in quad]
    ys = [float(y) for _, y in quad]
    return float(sum(xs) / max(1, len(xs))), float(sum(ys) / max(1, len(ys)))


def _sort_matches_for_array(
    detector: Line2DupLikeDetector,
    matches: Sequence[Match],
) -> List[Match]:
    if len(matches) <= 1:
        return list(matches)

    centers = np.asarray([_match_center(detector, match) for match in matches], dtype=np.float32)
    axis = np.array([1.0, 0.0], dtype=np.float32)
    if len(matches) >= 2:
        centered = centers - centers.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float32).reshape(2)
        except np.linalg.LinAlgError:
            axis = np.array([1.0, 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-6:
        std_x = float(np.std(centers[:, 0]))
        std_y = float(np.std(centers[:, 1]))
        axis = np.array([1.0, 0.0], dtype=np.float32) if std_x >= std_y else np.array([0.0, 1.0], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
    axis = axis / max(norm, 1e-6)
    if abs(float(axis[0])) >= abs(float(axis[1])):
        if float(axis[0]) < 0.0:
            axis = -axis
    elif float(axis[1]) < 0.0:
        axis = -axis

    projections = centers @ axis
    ordered_indices = sorted(
        range(len(matches)),
        key=lambda idx: (
            float(projections[idx]),
            float(centers[idx][1]),
            float(centers[idx][0]),
            -float(matches[idx].similarity),
        ),
    )
    return [matches[idx] for idx in ordered_indices]


def _top_matches(
    detector: Line2DupLikeDetector,
    scene_bgr: np.ndarray,
    recipe: Line2DupRecipe,
    scene_mask: Optional[np.ndarray] = None,
    *,
    expected_count: int,
    allow_partial: bool = False,
) -> List[Match]:
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
        raise RuntimeError("match failure")
    selected = matches[: max(1, int(expected_count))]
    if len(selected) < max(1, int(expected_count)):
        if allow_partial and selected:
            return _sort_matches_for_array(detector, selected)
        raise RuntimeError(f"match failure: expected {int(expected_count)} instances, got {len(selected)}")
    return _sort_matches_for_array(detector, selected)


def _best_match(
    detector: Line2DupLikeDetector,
    scene_bgr: np.ndarray,
    recipe: Line2DupRecipe,
    scene_mask: Optional[np.ndarray] = None,
) -> Match:
    return _top_matches(
        detector,
        scene_bgr,
        recipe,
        scene_mask=scene_mask,
        expected_count=1,
    )[0]


def _array_offsets_for_recipe(recipe: Line2DupRecipe) -> List[Point]:
    count = recipe_array_count(recipe)
    pitch_x, pitch_y = recipe_array_pitch(recipe)
    return [
        (float(pitch_x) * float(index), float(pitch_y) * float(index))
        for index in range(max(1, int(count)))
    ]


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
        instance_count=1,
        expected_instance_count=1,
        matches=[match],
        transform_mode="match_bbox",
    )


def _affine_follow_context(
    detector: Line2DupLikeDetector,
    match: Match,
    recipe: Line2DupRecipe,
    ref_img_path: str,
) -> Tuple[List[Tuple[str, str, List[Point]]], object, np.ndarray, np.ndarray]:
    _source_info, roi_img, roi_mask, template_roi_rect, _mask_rects = load_class_source_assets(detector, match.class_id)
    region_specs = _region_specs_from_recipe(recipe, ref_img_path, template_roi_rect)
    meta = detector.get_template_meta(match.class_id, int(match.template_id))
    angle = float(meta.get("angle", 0.0))
    scale = float(meta.get("scale", 1.0))
    affine3, _canvas_size = expanded_pose_affine(roi_img.shape[1], roi_img.shape[0], angle, scale)
    return region_specs, template_roi_rect, affine3, roi_mask


def _estimate_patch_to_scene_homography(
    detector: Line2DupLikeDetector,
    match: Match,
    ref_img_path: str,
    template_roi_rect,
    scene_bgr: np.ndarray,
    *,
    roi_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if not ref_img_path or not os.path.exists(ref_img_path):
        return None
    ref_gray = imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    scene_gray_full = _to_gray(scene_bgr)
    if ref_gray is None or scene_gray_full is None:
        return None

    x = int(template_roi_rect.x)
    y = int(template_roi_rect.y)
    w = int(template_roi_rect.w)
    h = int(template_roi_rect.h)
    if w < 16 or h < 16:
        return None
    if x < 0 or y < 0 or x + w > ref_gray.shape[1] or y + h > ref_gray.shape[0]:
        return None

    patch = ref_gray[y : y + h, x : x + w]
    if patch.size == 0:
        return None

    coarse_bbox = _bbox_from_points(match_quad(detector, match), scene_bgr.shape[:2])
    crop_x, crop_y, crop_w, crop_h = _pad_bbox(coarse_bbox, scene_bgr.shape[:2])
    scene_crop = scene_gray_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    if scene_crop.size == 0:
        return None

    orb = cv2.ORB_create(nfeatures=2000)
    patch_mask = None
    if roi_mask is not None and roi_mask.shape[:2] == patch.shape[:2]:
        patch_mask = np.asarray(roi_mask, dtype=np.uint8)
        if cv2.countNonZero(patch_mask) < 32:
            patch_mask = None

    keypoints_ref, descriptors_ref = orb.detectAndCompute(patch, patch_mask)
    keypoints_scene, descriptors_scene = orb.detectAndCompute(scene_crop, None)
    if descriptors_ref is None or descriptors_scene is None:
        return None
    if len(keypoints_ref) < 8 or len(keypoints_scene) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw_matches = matcher.match(descriptors_ref, descriptors_scene)
    if len(raw_matches) < 8:
        return None
    raw_matches = sorted(raw_matches, key=lambda item: item.distance)[:200]

    src = np.float32([keypoints_ref[item.queryIdx].pt for item in raw_matches]).reshape(-1, 1, 2)
    dst = np.float32(
        [
            (
                float(keypoints_scene[item.trainIdx].pt[0]) + float(crop_x),
                float(keypoints_scene[item.trainIdx].pt[1]) + float(crop_y),
            )
            for item in raw_matches
        ]
    ).reshape(-1, 1, 2)
    transform, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if transform is None or mask is None:
        return None
    inliers = int(mask.ravel().sum())
    if inliers < 8:
        return None

    patch_corners = np.array(
        [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(patch_corners, transform).reshape(-1, 2)
    refined_bbox = _bbox_from_points([(float(px), float(py)) for px, py in projected], scene_bgr.shape[:2])
    if not _is_reasonable_projective_refine(coarse_bbox, refined_bbox):
        return None
    return np.asarray(transform, dtype=np.float32)


def _follow_affine_regions_for_match(
    detector: Line2DupLikeDetector,
    match: Match,
    region_specs: Sequence[Tuple[str, str, List[Point]]],
    template_roi_rect,
    affine3: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    instance_index: int = 0,
    array_count: int = 1,
    offset_xy: Point = (0.0, 0.0),
) -> List[FollowRegion]:
    quad = match_quad(detector, match)
    top_left = quad[0]
    regions: List[FollowRegion] = []
    offset_x = float(offset_xy[0])
    offset_y = float(offset_xy[1])
    for label_name, _shape_type, ref_points in region_specs:
        local_points = [
            (
                float(x) + offset_x - float(template_roi_rect.x),
                float(y) + offset_y - float(template_roi_rect.y),
            )
            for x, y in ref_points
        ]
        canvas_points = apply_affine_to_points(affine3, local_points)
        scene_points = _translated_points(canvas_points, float(top_left[0]), float(top_left[1]))
        if match.refined_transform is not None:
            scene_points = apply_affine_to_points(match.refined_transform, scene_points)
        bbox = _bbox_from_points(scene_points, image_shape)
        regions.append(
            FollowRegion(
                label_name=format_array_roi_label(label_name, instance_index, array_count),
                points=scene_points,
                bbox=bbox,
                source_shape_type="polygon",
            )
        )
    return regions


def _follow_by_affine_roi(
    detector: Line2DupLikeDetector,
    match: Match,
    recipe: Line2DupRecipe,
    ref_img_path: str,
    image_shape: Tuple[int, int],
) -> FollowResult:
    region_specs, template_roi_rect, affine3, _roi_mask = _affine_follow_context(detector, match, recipe, ref_img_path)
    offsets = _array_offsets_for_recipe(recipe)
    regions: List[FollowRegion] = []
    for instance_index, offset_xy in enumerate(offsets):
        regions.extend(
            _follow_affine_regions_for_match(
                detector,
                match,
                region_specs,
                template_roi_rect,
                affine3,
                image_shape,
                instance_index=instance_index,
                array_count=len(offsets),
                offset_xy=offset_xy,
            )
        )
    primary = regions[0]
    return FollowResult(
        match=match,
        regions=regions,
        points=primary.points,
        bbox=primary.bbox,
        source_shape_type=primary.source_shape_type,
        instance_count=len(offsets),
        expected_instance_count=len(offsets),
        matches=[match],
        transform_mode="affine",
    )


def _follow_by_projective_roi(
    detector: Line2DupLikeDetector,
    match: Match,
    recipe: Line2DupRecipe,
    ref_img_path: str,
    scene_bgr: np.ndarray,
) -> Optional[FollowResult]:
    region_specs, template_roi_rect, _affine3, roi_mask = _affine_follow_context(detector, match, recipe, ref_img_path)
    transform = _estimate_patch_to_scene_homography(
        detector,
        match,
        ref_img_path,
        template_roi_rect,
        scene_bgr,
        roi_mask=roi_mask,
    )
    if transform is None:
        return None

    offsets = _array_offsets_for_recipe(recipe)
    regions: List[FollowRegion] = []
    for instance_index, offset_xy in enumerate(offsets):
        offset_x = float(offset_xy[0])
        offset_y = float(offset_xy[1])
        for label_name, _shape_type, ref_points in region_specs:
            patch_points = [
                (
                    float(x) + offset_x - float(template_roi_rect.x),
                    float(y) + offset_y - float(template_roi_rect.y),
                )
                for x, y in ref_points
            ]
            scene_points = apply_affine_to_points(transform, patch_points)
            bbox = _bbox_from_points(scene_points, scene_bgr.shape[:2])
            regions.append(
                FollowRegion(
                    label_name=format_array_roi_label(label_name, instance_index, len(offsets)),
                    points=scene_points,
                    bbox=bbox,
                    source_shape_type="polygon",
                )
            )

    if not regions:
        return None
    primary = regions[0]
    return FollowResult(
        match=match,
        regions=regions,
        points=primary.points,
        bbox=primary.bbox,
        source_shape_type=primary.source_shape_type,
        instance_count=len(offsets),
        expected_instance_count=len(offsets),
        matches=[match],
        transform_mode="projective_homography",
    )


def _follow_matches_by_bbox(
    detector: Line2DupLikeDetector,
    matches: Sequence[Match],
    recipe: Line2DupRecipe,
    image_shape: Tuple[int, int],
) -> FollowResult:
    label_name = str(recipe.output_label or "roi")
    if recipe.reference_regions:
        for region in recipe.reference_regions:
            if isinstance(region, dict):
                label_name = str(region.get("output_label") or region.get("reference_label") or label_name).strip() or label_name
                if label_name:
                    break
    array_count = max(1, len(matches))
    regions: List[FollowRegion] = []
    for instance_index, match in enumerate(matches):
        bbox = tuple(match_bbox(detector, match))
        x, y, w, h = bbox
        points = _rect_points(float(x), float(y), float(w), float(h))
        regions.append(
            FollowRegion(
                label_name=format_array_roi_label(label_name, instance_index, array_count),
                points=points,
                bbox=_bbox_from_points(points, image_shape),
                source_shape_type="rectangle",
            )
        )
    primary = regions[0]
    return FollowResult(
        match=matches[0],
        regions=regions,
        points=primary.points,
        bbox=primary.bbox,
        source_shape_type=primary.source_shape_type,
        instance_count=array_count,
        expected_instance_count=array_count,
        matches=list(matches),
        transform_mode="match_bbox",
    )


def _follow_matches_by_affine_roi(
    detector: Line2DupLikeDetector,
    matches: Sequence[Match],
    recipe: Line2DupRecipe,
    ref_img_path: str,
    image_shape: Tuple[int, int],
) -> FollowResult:
    array_count = max(1, len(matches))
    regions: List[FollowRegion] = []
    for instance_index, match in enumerate(matches):
        region_specs, template_roi_rect, affine3, _roi_mask = _affine_follow_context(detector, match, recipe, ref_img_path)
        regions.extend(
            _follow_affine_regions_for_match(
                detector,
                match,
                region_specs,
                template_roi_rect,
                affine3,
                image_shape,
                instance_index=instance_index,
                array_count=array_count,
            )
        )
    primary = regions[0]
    return FollowResult(
        match=matches[0],
        regions=regions,
        points=primary.points,
        bbox=primary.bbox,
        source_shape_type=primary.source_shape_type,
        instance_count=array_count,
        expected_instance_count=array_count,
        matches=list(matches),
        transform_mode="affine",
    )


def locate_and_follow(
    scene_bgr: np.ndarray,
    ref_img_path: str,
    recipe: Line2DupRecipe,
    *,
    detector: Optional[Line2DupLikeDetector] = None,
    scene_mask: Optional[np.ndarray] = None,
    allow_partial_array: bool = False,
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
        return _follow_by_bbox(det, match, scene_bgr.shape[:2], label_name=str(recipe.output_label or "roi"))
    if recipe_array_count(recipe) > 1:
        projective_result = _follow_by_projective_roi(det, match, recipe, ref_img_path, scene_bgr)
        if projective_result is not None:
            return projective_result
    return _follow_by_affine_roi(det, match, recipe, ref_img_path, scene_bgr.shape[:2])


__all__ = ["FollowRegion", "FollowResult", "locate_and_follow"]
