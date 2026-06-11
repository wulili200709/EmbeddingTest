from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from algorithms.image_io import imread

from .interop.native_api import NccNativeApi, NccNativeMatcher
from .model import (
    NccAngleRange,
    NccAngleSearch,
    NccMatchBoundingBox,
    NccMatchModel,
    NccMatchOptions,
    NccMatchResult,
    load_model,
    resolve_asset_path,
)


@dataclass(frozen=True)
class NccMatchResponse:
    matches: Tuple[NccMatchResult, ...]
    elapsed_ms: float
    backend_name: str


def _prepare_gray_image(image_bgr: np.ndarray, *, bitwise_not: bool = False) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is required")
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray.astype(np.uint8, copy=False))
    if bitwise_not:
        gray = cv2.bitwise_not(gray)
    return gray


def _clamp_xywh(xywh: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in xywh]
    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _crop_bgr(scene_bgr: np.ndarray, search_roi: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if not search_roi:
        return scene_bgr, (0, 0, int(scene_bgr.shape[1]), int(scene_bgr.shape[0]))
    x, y, w, h = _clamp_xywh(search_roi, int(scene_bgr.shape[1]), int(scene_bgr.shape[0]))
    return scene_bgr[y : y + h, x : x + w].copy(), (x, y, w, h)


def _build_angle_candidates(search: object) -> List[float]:
    if search is None:
        return [0.0]

    mode = str(getattr(search, "mode", "ranges") or "ranges").strip().lower()
    tolerance = max(0.0, float(getattr(search, "tolerance_angle", 0.0) or 0.0))
    ranges = list(getattr(search, "ranges", []) or [])
    if mode == "symmetric":
        ranges = [NccAngleRange(-tolerance, tolerance)]
    elif not ranges:
        ranges = [NccAngleRange(-180.0, 180.0)]

    angles: List[float] = []
    for item in ranges:
        start = float(getattr(item, "start", -180.0))
        end = float(getattr(item, "end", 180.0))
        if end < start:
            start, end = end, start
        span = end - start
        if span <= 0.0:
            angles.append(round(start, 4))
            continue
        if span <= 12.0:
            step = 1.0
        elif span <= 45.0:
            step = 2.0
        elif span <= 120.0:
            step = 3.0
        else:
            step = 5.0
        count = max(1, int(math.floor(span / step)))
        for index in range(count + 1):
            value = start + min(span, index * step)
            angles.append(round(value, 4))
        angles.append(round(end, 4))
    deduped = sorted({round(angle, 4) for angle in angles})
    if 0.0 not in deduped:
        deduped.append(0.0)
        deduped.sort()
    return deduped


def _rotate_template(template_gray: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    height, width = template_gray.shape[:2]
    if abs(float(angle_deg)) < 1e-6:
        quad = np.array(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )
        return template_gray, quad

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    bound_w = int(math.ceil(height * abs_sin + width * abs_cos))
    bound_h = int(math.ceil(height * abs_cos + width * abs_sin))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]

    border_value = 255 if float(template_gray.mean()) < 128.0 else 0
    rotated = cv2.warpAffine(
        template_gray,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width - 1), 0.0, 1.0],
            [float(width - 1), float(height - 1), 1.0],
            [0.0, float(height - 1), 1.0],
        ],
        dtype=np.float32,
    )
    transformed = corners @ matrix.T
    return rotated, transformed.astype(np.float32)


def _rotate_template_with_mask(
    template_gray: np.ndarray,
    mask_gray: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = template_gray.shape[:2]
    if abs(float(angle_deg)) < 1e-6:
        quad = np.array(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )
        return template_gray, mask_gray, quad

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    bound_w = int(math.ceil(height * abs_sin + width * abs_cos))
    bound_h = int(math.ceil(height * abs_cos + width * abs_sin))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]

    border_value = 255 if float(template_gray.mean()) < 128.0 else 0
    rotated = cv2.warpAffine(
        template_gray,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    rotated_mask = cv2.warpAffine(
        mask_gray,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width - 1), 0.0, 1.0],
            [float(width - 1), float(height - 1), 1.0],
            [0.0, float(height - 1), 1.0],
        ],
        dtype=np.float32,
    )
    transformed = corners @ matrix.T
    return rotated, rotated_mask, transformed.astype(np.float32)


def _normalize_angle_deg(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _orientation_anchor_mask(
    template_gray: np.ndarray,
    anchor: object,
) -> Optional[np.ndarray]:
    height, width = template_gray.shape[:2]
    x = max(0, min(int(getattr(anchor, "x", 0)), width - 1))
    y = max(0, min(int(getattr(anchor, "y", 0)), height - 1))
    w = max(1, min(int(getattr(anchor, "width", 1)), width - x))
    h = max(1, min(int(getattr(anchor, "height", 1)), height - y))
    if w < 2 or h < 2:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


def _zero_mean_ncc(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    selected = mask > 0
    if int(np.count_nonzero(selected)) < 4:
        return -1.0
    first_values = first[selected].astype(np.float32)
    second_values = second[selected].astype(np.float32)
    first_values -= float(first_values.mean())
    second_values -= float(second_values.mean())
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(first_values, second_values) / denominator)


def _orientation_anchor_score(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    anchor_mask: np.ndarray,
    quad: Sequence[Tuple[float, float]],
) -> float:
    height, width = template_gray.shape[:2]
    source_quad = np.asarray(quad, dtype=np.float32)
    if source_quad.shape != (4, 2):
        return -1.0
    non_zero = cv2.findNonZero(anchor_mask)
    if non_zero is None:
        return -1.0
    anchor_x, anchor_y, anchor_w, anchor_h = cv2.boundingRect(non_zero)
    template_quad = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )
    anchor_quad = np.asarray(
        [
            [float(anchor_x), float(anchor_y)],
            [float(anchor_x + anchor_w - 1), float(anchor_y)],
            [float(anchor_x + anchor_w - 1), float(anchor_y + anchor_h - 1)],
            [float(anchor_x), float(anchor_y + anchor_h - 1)],
        ],
        dtype=np.float32,
    )
    template_to_scene = cv2.getPerspectiveTransform(template_quad, source_quad)
    scene_anchor_quad = cv2.perspectiveTransform(
        anchor_quad.reshape(1, -1, 2),
        template_to_scene,
    ).reshape(-1, 2)
    local_anchor_quad = np.asarray(
        [
            [0.0, 0.0],
            [float(anchor_w - 1), 0.0],
            [float(anchor_w - 1), float(anchor_h - 1)],
            [0.0, float(anchor_h - 1)],
        ],
        dtype=np.float32,
    )
    aligned = cv2.warpPerspective(
        scene_gray,
        cv2.getPerspectiveTransform(scene_anchor_quad, local_anchor_quad),
        (anchor_w, anchor_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    template_anchor = template_gray[
        anchor_y : anchor_y + anchor_h,
        anchor_x : anchor_x + anchor_w,
    ]
    local_mask = anchor_mask[
        anchor_y : anchor_y + anchor_h,
        anchor_x : anchor_x + anchor_w,
    ]
    return _zero_mean_ncc(template_anchor, aligned, local_mask)


def _find_orientation_anchor_match(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    primary: NccMatchResult,
    anchor_mask: np.ndarray,
    angle_deg: float,
) -> Optional[Tuple[NccMatchResult, float]]:
    rotated_template, rotated_mask, quad_local = _rotate_template_with_mask(
        template_gray,
        anchor_mask,
        angle_deg,
    )
    non_zero = cv2.findNonZero(rotated_mask)
    if non_zero is None:
        return None
    anchor_x, anchor_y, anchor_w, anchor_h = cv2.boundingRect(non_zero)
    anchor_template = rotated_template[
        anchor_y : anchor_y + anchor_h,
        anchor_x : anchor_x + anchor_w,
    ]
    anchor_template_mask = rotated_mask[
        anchor_y : anchor_y + anchor_h,
        anchor_x : anchor_x + anchor_w,
    ]

    margin = max(64, int(round(max(template_gray.shape[:2]) * 0.35)))
    scene_height, scene_width = scene_gray.shape[:2]
    search_x0 = max(0, int(math.floor(primary.bbox.x)) - margin)
    search_y0 = max(0, int(math.floor(primary.bbox.y)) - margin)
    search_x1 = min(
        scene_width,
        int(math.ceil(primary.bbox.x + primary.bbox.width)) + margin,
    )
    search_y1 = min(
        scene_height,
        int(math.ceil(primary.bbox.y + primary.bbox.height)) + margin,
    )
    search = scene_gray[search_y0:search_y1, search_x0:search_x1]
    if (
        search.size == 0
        or anchor_template.shape[0] > search.shape[0]
        or anchor_template.shape[1] > search.shape[1]
    ):
        return None

    scale = 0.25 if max(anchor_template.shape[:2]) >= 400 else 0.5
    if scale < 1.0:
        search_for_match = cv2.resize(
            search,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        template_for_match = cv2.resize(
            anchor_template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        mask_for_match = cv2.resize(
            anchor_template_mask,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        search_for_match = search
        template_for_match = anchor_template
        mask_for_match = anchor_template_mask

    if (
        template_for_match.shape[0] > search_for_match.shape[0]
        or template_for_match.shape[1] > search_for_match.shape[1]
    ):
        return None
    response = cv2.matchTemplate(
        search_for_match,
        template_for_match,
        cv2.TM_CCORR_NORMED,
        mask=mask_for_match,
    )
    response = np.nan_to_num(response, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, _, _, max_location = cv2.minMaxLoc(response)
    matched_x = float(max_location[0]) / scale
    matched_y = float(max_location[1]) / scale
    offset_x = float(search_x0) + matched_x - float(anchor_x)
    offset_y = float(search_y0) + matched_y - float(anchor_y)
    quad = tuple(
        (float(px + offset_x), float(py + offset_y))
        for px, py in quad_local
    )
    bbox = _bounding_box_from_quad(quad)
    match = NccMatchResult(
        score=primary.score,
        angle=_normalize_angle_deg(angle_deg),
        center=(bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0),
        quad=quad,
        bbox=bbox,
    )
    return match, _orientation_anchor_score(
        scene_gray,
        template_gray,
        anchor_mask,
        quad,
    )


def _disambiguate_top_orientation(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    matches: Tuple[NccMatchResult, ...],
    anchor: object,
) -> Tuple[Tuple[NccMatchResult, ...], bool]:
    if not matches:
        return matches, False
    anchor_mask = _orientation_anchor_mask(template_gray, anchor)
    if anchor_mask is None:
        return matches, False

    primary = matches[0]
    same = _find_orientation_anchor_match(
        scene_gray,
        template_gray,
        primary,
        anchor_mask,
        primary.angle,
    )
    opposite = _find_orientation_anchor_match(
        scene_gray,
        template_gray,
        primary,
        anchor_mask,
        _normalize_angle_deg(primary.angle + 180.0),
    )
    if same is None or opposite is None:
        return matches, False

    _, same_score = same
    opposite_match, opposite_score = opposite
    if opposite_score <= same_score + 0.015:
        return matches, True
    return (opposite_match, *matches[1:]), True


def _largest_saturation_contour(
    image_bgr: np.ndarray,
    *,
    restrict_mask: Optional[np.ndarray] = None,
    morphology_size: int = 15,
) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray([20, 80, 10], dtype=np.uint8),
        np.asarray([110, 255, 255], dtype=np.uint8),
    )
    if restrict_mask is not None:
        mask = cv2.bitwise_and(mask, restrict_mask)
    kernel_size = max(5, int(morphology_size) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) <= 0.0:
        return None
    return contour


def _contour_long_axis_angle_deg(contour: np.ndarray) -> float:
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    edges = [
        box[(index + 1) % 4] - box[index]
        for index in range(4)
    ]
    longest = max(edges, key=lambda edge: float(np.dot(edge, edge)))
    angle = math.degrees(math.atan2(float(longest[1]), float(longest[0])))
    return float((angle + 90.0) % 180.0 - 90.0)


def _prepare_saturation_axis_search(
    scene_bgr: np.ndarray,
    template_bgr: np.ndarray,
    options: NccMatchOptions,
) -> Optional[Tuple[np.ndarray, NccMatchOptions, np.ndarray, Tuple[float, float]]]:
    if (
        scene_bgr.ndim != 3
        or scene_bgr.shape[2] != 3
        or template_bgr.ndim != 3
        or template_bgr.shape[2] != 3
    ):
        return None
    ranges = list(options.angle_search.ranges or [])
    total_span = sum(abs(float(item.end) - float(item.start)) for item in ranges)
    if total_span < 180.0:
        return None

    scale = 0.25
    scene_small = cv2.resize(
        scene_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    template_small = cv2.resize(
        template_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    scene_contour = _largest_saturation_contour(
        scene_small,
        morphology_size=5,
    )
    template_contour = _largest_saturation_contour(
        template_small,
        morphology_size=5,
    )
    if scene_contour is None or template_contour is None:
        return None

    scene_area = float(cv2.contourArea(scene_contour))
    template_area = float(cv2.contourArea(template_contour))
    if (
        scene_area < float(scene_small.shape[0] * scene_small.shape[1]) * 0.01
        or template_area < float(template_small.shape[0] * template_small.shape[1]) * 0.15
    ):
        return None

    scene_axis = _contour_long_axis_angle_deg(scene_contour)
    template_axis = _contour_long_axis_angle_deg(template_contour)
    axis_angle = float((-(scene_axis - template_axis) + 90.0) % 180.0 - 90.0)

    small_x, small_y, small_w, small_h = cv2.boundingRect(scene_contour)
    margin = max(96, int(round(max(template_bgr.shape[:2]) * 0.33)))
    crop_x0 = max(0, int(math.floor(float(small_x) / scale)) - margin)
    crop_y0 = max(0, int(math.floor(float(small_y) / scale)) - margin)
    crop_x1 = min(
        scene_bgr.shape[1],
        int(math.ceil(float(small_x + small_w) / scale)) + margin,
    )
    crop_y1 = min(
        scene_bgr.shape[0],
        int(math.ceil(float(small_y + small_h) / scale)) + margin,
    )
    scene_crop = scene_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    if scene_crop.size == 0:
        return None

    crop_height, crop_width = scene_crop.shape[:2]
    center = (float(crop_width) / 2.0, float(crop_height) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -axis_angle, 1.0)
    abs_cos = abs(float(matrix[0, 0]))
    abs_sin = abs(float(matrix[0, 1]))
    bound_width = int(math.ceil(crop_height * abs_sin + crop_width * abs_cos))
    bound_height = int(math.ceil(crop_height * abs_cos + crop_width * abs_sin))
    matrix[0, 2] += float(bound_width) / 2.0 - center[0]
    matrix[1, 2] += float(bound_height) / 2.0 - center[1]
    sample = scene_crop[:: max(1, crop_height // 64), :: max(1, crop_width // 64)]
    border_value = tuple(
        int(round(float(np.median(sample[:, :, channel]))))
        for channel in range(3)
    )
    normalized_scene = cv2.warpAffine(
        scene_crop,
        matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    narrowed = NccMatchOptions(
        target_num=options.target_num,
        max_overlap=options.max_overlap,
        score_threshold=options.score_threshold,
        angle_search=NccAngleSearch(
            mode="ranges",
            tolerance_angle=0.0,
            ranges=[
                NccAngleRange(-15.0, 15.0),
                NccAngleRange(165.0, 195.0),
            ],
        ),
        min_reduced_area=options.min_reduced_area,
        use_simd=options.use_simd,
        use_subpixel=options.use_subpixel,
        bitwise_not=options.bitwise_not,
        stop_layer1=options.stop_layer1,
    ).normalized()
    return (
        normalized_scene,
        narrowed,
        cv2.invertAffineTransform(matrix),
        (float(crop_x0), float(crop_y0)),
    )


def _restore_match_from_pose_search(
    match: NccMatchResult,
    inverse_affine: np.ndarray,
    crop_origin: Tuple[float, float],
) -> NccMatchResult:
    quad = cv2.transform(
        np.asarray(match.quad, dtype=np.float32).reshape(1, -1, 2),
        inverse_affine,
    ).reshape(-1, 2)
    quad += np.asarray(crop_origin, dtype=np.float32)
    edge = quad[1] - quad[0]
    angle = _normalize_angle_deg(
        -math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    )
    bbox = _bounding_box_from_quad(quad)
    return NccMatchResult(
        score=match.score,
        angle=angle,
        center=(bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0),
        quad=tuple((float(x), float(y)) for x, y in quad),
        bbox=bbox,
    )


def _refine_match_by_saturation_rect(
    scene_bgr: np.ndarray,
    template_bgr: np.ndarray,
    match: NccMatchResult,
) -> Optional[NccMatchResult]:
    if (
        scene_bgr.ndim != 3
        or scene_bgr.shape[2] != 3
        or template_bgr.ndim != 3
        or template_bgr.shape[2] != 3
    ):
        return None
    template_height, template_width = template_bgr.shape[:2]
    template_corners = np.asarray(
        [
            [0.0, 0.0],
            [float(template_width - 1), 0.0],
            [float(template_width - 1), float(template_height - 1)],
            [0.0, float(template_height - 1)],
        ],
        dtype=np.float32,
    )
    initial_quad = np.asarray(match.quad, dtype=np.float32)
    if initial_quad.shape != (4, 2):
        return None

    morphology_size = max(
        5,
        int(round(min(template_height, template_width) * 0.02)) | 1,
    )
    template_contour = _largest_saturation_contour(
        template_bgr,
        morphology_size=morphology_size,
    )
    if template_contour is None:
        return None
    template_contour_area = float(cv2.contourArea(template_contour))
    if template_contour_area < float(template_height * template_width) * 0.15:
        return None

    margin = max(
        31,
        int(round(max(template_height, template_width) * 0.08)) | 1,
    )
    scene_height, scene_width = scene_bgr.shape[:2]
    crop_x0 = max(0, int(math.floor(float(np.min(initial_quad[:, 0])))) - margin)
    crop_y0 = max(0, int(math.floor(float(np.min(initial_quad[:, 1])))) - margin)
    crop_x1 = min(
        scene_width,
        int(math.ceil(float(np.max(initial_quad[:, 0])))) + margin,
    )
    crop_y1 = min(
        scene_height,
        int(math.ceil(float(np.max(initial_quad[:, 1])))) + margin,
    )
    scene_crop = scene_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    if scene_crop.size == 0:
        return None
    scene_contour = _largest_saturation_contour(
        scene_crop,
        morphology_size=morphology_size,
    )
    if scene_contour is None:
        return None
    scene_contour = scene_contour + np.asarray([[[crop_x0, crop_y0]]], dtype=scene_contour.dtype)

    initial_matrix = cv2.getPerspectiveTransform(template_corners, initial_quad)
    template_box = cv2.boxPoints(cv2.minAreaRect(template_contour)).astype(np.float32)
    scene_box = cv2.boxPoints(cv2.minAreaRect(scene_contour)).astype(np.float32)
    predicted_box = cv2.perspectiveTransform(
        template_box.reshape(1, -1, 2),
        initial_matrix,
    ).reshape(-1, 2)
    permutation = min(
        itertools.permutations(range(4)),
        key=lambda order: sum(
            float(np.linalg.norm(predicted_box[index] - scene_box[target]) ** 2)
            for index, target in enumerate(order)
        ),
    )
    matched_scene_box = np.asarray(
        [scene_box[index] for index in permutation],
        dtype=np.float32,
    )
    refined_matrix = cv2.getPerspectiveTransform(template_box, matched_scene_box)
    refined_quad = cv2.perspectiveTransform(
        template_corners.reshape(1, -1, 2),
        refined_matrix,
    ).reshape(-1, 2)
    if not np.all(np.isfinite(refined_quad)):
        return None

    initial_area = abs(float(cv2.contourArea(initial_quad)))
    refined_area = abs(float(cv2.contourArea(refined_quad)))
    if initial_area <= 1.0 or not (0.75 <= refined_area / initial_area <= 1.25):
        return None
    initial_center = np.mean(initial_quad, axis=0)
    refined_center = np.mean(refined_quad, axis=0)
    max_shift = max(match.bbox.width, match.bbox.height) * 0.12
    if float(np.linalg.norm(refined_center - initial_center)) > max_shift:
        return None
    corner_shift = np.linalg.norm(refined_quad - initial_quad, axis=1)
    if float(np.max(corner_shift)) > max(match.bbox.width, match.bbox.height) * 0.18:
        return None

    edge = refined_quad[1] - refined_quad[0]
    refined_angle = _normalize_angle_deg(
        -math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    )
    if abs(_normalize_angle_deg(refined_angle - match.angle)) > 10.0:
        return None
    bbox = _bounding_box_from_quad(refined_quad)
    return NccMatchResult(
        score=match.score,
        angle=refined_angle,
        center=(bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0),
        quad=tuple((float(x), float(y)) for x, y in refined_quad),
        bbox=bbox,
    )


def _bounding_box_from_quad(quad: Sequence[Tuple[float, float]] | np.ndarray) -> NccMatchBoundingBox:
    pts = np.asarray(quad, dtype=np.float32)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    return NccMatchBoundingBox(
        x=x_min,
        y=y_min,
        width=max(1.0, x_max - x_min),
        height=max(1.0, y_max - y_min),
    )


def _bbox_iou(a: NccMatchBoundingBox, b: NccMatchBoundingBox) -> float:
    ax1 = a.x + a.width
    ay1 = a.y + a.height
    bx1 = b.x + b.width
    by1 = b.y + b.height
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _nms_candidates(candidates: Iterable[NccMatchResult], max_overlap: float, limit: int) -> Tuple[NccMatchResult, ...]:
    selected: List[NccMatchResult] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_bbox_iou(candidate.bbox, current.bbox) > max_overlap for current in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _offset_match(match: NccMatchResult, dx: float, dy: float) -> NccMatchResult:
    quad = tuple((float(x + dx), float(y + dy)) for x, y in match.quad)
    bbox = NccMatchBoundingBox(
        x=match.bbox.x + dx,
        y=match.bbox.y + dy,
        width=match.bbox.width,
        height=match.bbox.height,
    )
    return NccMatchResult(
        score=match.score,
        angle=match.angle,
        center=(match.center[0] + dx, match.center[1] + dy),
        quad=quad,
        bbox=bbox,
    )


def _native_payload(options: NccMatchOptions) -> Dict[str, object]:
    normalized = options.normalized()
    return {
        "targetNum": normalized.target_num,
        "maxOverlap": normalized.max_overlap,
        "scoreThreshold": normalized.score_threshold,
        "useSimd": normalized.use_simd,
        "useSubpixel": normalized.use_subpixel,
        "bitwiseNot": normalized.bitwise_not,
        "stopLayer1": normalized.stop_layer1,
        "angleSearch": {
            "mode": str(normalized.angle_search.mode or "ranges"),
            "toleranceAngle": normalized.angle_search.tolerance_angle,
            "ranges": [
                {"start": item.start, "end": item.end}
                for item in normalized.angle_search.ranges
            ],
        },
    }


def _parse_native_response(payload: dict) -> Tuple[NccMatchResult, ...]:
    matches: List[NccMatchResult] = []
    for item in list(payload.get("matches", []) or []):
        if not isinstance(item, dict):
            continue
        quad_raw = list(item.get("quad", []) or [])
        quad: List[Tuple[float, float]] = []
        for point in quad_raw:
            if not isinstance(point, dict):
                continue
            quad.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
        if len(quad) < 4:
            continue
        bbox_raw = item.get("bbox", {}) or {}
        bbox = NccMatchBoundingBox(
            x=float(bbox_raw.get("x", 0.0)),
            y=float(bbox_raw.get("y", 0.0)),
            width=float(bbox_raw.get("width", 1.0)),
            height=float(bbox_raw.get("height", 1.0)),
        )
        center_raw = item.get("center", {}) or {}
        matches.append(
            NccMatchResult(
                score=float(item.get("score", 0.0)),
                angle=float(item.get("angle", 0.0)),
                center=(
                    float(center_raw.get("x", item.get("centerX", 0.0))),
                    float(center_raw.get("y", item.get("centerY", 0.0))),
                ),
                quad=tuple(quad),
                bbox=bbox,
            )
        )
    return tuple(matches)


def _match_python(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    options: NccMatchOptions,
    *,
    mask_gray: Optional[np.ndarray] = None,
) -> Tuple[NccMatchResult, ...]:
    prepared_scene = _prepare_gray_image(scene_gray, bitwise_not=options.bitwise_not)
    prepared_template = _prepare_gray_image(template_gray, bitwise_not=options.bitwise_not)
    if prepared_scene.shape[0] < prepared_template.shape[0] or prepared_scene.shape[1] < prepared_template.shape[1]:
        return tuple()
    prepared_mask: Optional[np.ndarray] = None
    if mask_gray is not None:
        prepared_mask = np.ascontiguousarray(mask_gray.astype(np.uint8, copy=False))
        if prepared_mask.shape[:2] != prepared_template.shape[:2]:
            raise ValueError("template mask shape must match template image")
        if int(cv2.countNonZero(prepared_mask)) <= 0:
            prepared_mask = None

    candidates: List[NccMatchResult] = []
    for angle in _build_angle_candidates(options.angle_search):
        rotated_mask: Optional[np.ndarray] = None
        if prepared_mask is not None:
            rotated_template, rotated_mask, quad_local = _rotate_template_with_mask(prepared_template, prepared_mask, angle)
            if int(cv2.countNonZero(rotated_mask)) <= 0:
                continue
        else:
            rotated_template, quad_local = _rotate_template(prepared_template, angle)
        if (
            rotated_template.shape[0] < 2
            or rotated_template.shape[1] < 2
            or rotated_template.shape[0] > prepared_scene.shape[0]
            or rotated_template.shape[1] > prepared_scene.shape[1]
        ):
            continue
        if rotated_mask is not None:
            response = cv2.matchTemplate(
                prepared_scene,
                rotated_template,
                cv2.TM_CCORR_NORMED,
                mask=rotated_mask,
            )
        else:
            response = cv2.matchTemplate(prepared_scene, rotated_template, cv2.TM_CCOEFF_NORMED)
        work = response.copy()
        per_angle_limit = max(options.target_num * 4, options.target_num)
        for _ in range(per_angle_limit):
            _, score, _, max_loc = cv2.minMaxLoc(work)
            if float(score) < float(options.score_threshold):
                break
            x, y = int(max_loc[0]), int(max_loc[1])
            quad = tuple((float(px + x), float(py + y)) for px, py in quad_local)
            bbox = _bounding_box_from_quad(quad)
            candidates.append(
                NccMatchResult(
                    score=float(score),
                    angle=float(angle),
                    center=(
                        bbox.x + bbox.width / 2.0,
                        bbox.y + bbox.height / 2.0,
                    ),
                    quad=quad,
                    bbox=bbox,
                )
            )
            suppress_w = max(1, int(round(rotated_template.shape[1] * (1.0 - options.max_overlap * 0.5))))
            suppress_h = max(1, int(round(rotated_template.shape[0] * (1.0 - options.max_overlap * 0.5))))
            x0 = max(0, x - suppress_w // 2)
            y0 = max(0, y - suppress_h // 2)
            x1 = min(work.shape[1], x + suppress_w)
            y1 = min(work.shape[0], y + suppress_h)
            work[y0:y1, x0:x1] = -1.0

    return _nms_candidates(candidates, options.max_overlap, options.target_num)


class NccCompiledModel:
    def __init__(self, model_path: str, model: Optional[NccMatchModel] = None) -> None:
        self.model_path = str(model_path)
        self.model = (model or load_model(model_path)).normalized()
        self._template_last_write_ns: int = -1
        self._template_gray: Optional[np.ndarray] = None
        self._template_bgr_last_write_ns: int = -1
        self._template_bgr: Optional[np.ndarray] = None
        self._template_mask_last_write_ns: int = -2
        self._template_mask_gray: Optional[np.ndarray] = None
        self._native_matcher_cache: Dict[int, NccNativeMatcher] = {}

    def close(self) -> None:
        for matcher in self._native_matcher_cache.values():
            matcher.close()
        self._native_matcher_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _template_file(self) -> Path:
        return resolve_asset_path(self.model_path, self.model.template_image_path)

    def _template_mask_file(self) -> Path:
        return resolve_asset_path(self.model_path, self.model.mask_image_path)

    def _load_template_gray(self) -> np.ndarray:
        template_path = self._template_file()
        if not template_path.exists():
            raise FileNotFoundError(template_path)
        last_write_ns = template_path.stat().st_mtime_ns
        if self._template_gray is not None and self._template_last_write_ns == last_write_ns:
            return self._template_gray
        image = imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read template image: {template_path}")
        self._template_gray = np.ascontiguousarray(image)
        self._template_last_write_ns = last_write_ns
        if self._native_matcher_cache:
            self.close()
        return self._template_gray

    def _load_template_bgr(self) -> np.ndarray:
        template_path = self._template_file()
        if not template_path.exists():
            raise FileNotFoundError(template_path)
        last_write_ns = template_path.stat().st_mtime_ns
        if self._template_bgr is not None and self._template_bgr_last_write_ns == last_write_ns:
            return self._template_bgr
        image = imread(str(template_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read template image: {template_path}")
        self._template_bgr = np.ascontiguousarray(image)
        self._template_bgr_last_write_ns = last_write_ns
        return self._template_bgr

    def _load_template_mask_gray(self) -> Optional[np.ndarray]:
        if not bool(getattr(self.model, "template_mask_enabled", False)):
            self._template_mask_last_write_ns = -1
            self._template_mask_gray = None
            return None
        mask_path = self._template_mask_file()
        if not mask_path.exists():
            self._template_mask_last_write_ns = -1
            self._template_mask_gray = None
            return None
        last_write_ns = mask_path.stat().st_mtime_ns
        if self._template_mask_gray is not None and self._template_mask_last_write_ns == last_write_ns:
            return self._template_mask_gray
        image = imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read template mask image: {mask_path}")
        self._template_mask_gray = np.ascontiguousarray(image.astype(np.uint8, copy=False))
        self._template_mask_last_write_ns = last_write_ns
        return self._template_mask_gray

    def _get_native_matcher(self, min_reduced_area: int) -> NccNativeMatcher:
        template_gray = self._load_template_gray()
        key = max(16, int(min_reduced_area))
        matcher = self._native_matcher_cache.get(key)
        if matcher is not None:
            return matcher
        matcher = NccNativeMatcher(template_gray, min_reduced_area=key)
        self._native_matcher_cache[key] = matcher
        return matcher

    def match(
        self,
        scene_bgr: np.ndarray,
        options: Optional[NccMatchOptions] = None,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        *,
        prefer_native: bool = True,
    ) -> NccMatchResponse:
        if scene_bgr is None or scene_bgr.size == 0:
            raise ValueError("scene_bgr is required")
        normalized_options = (options or self.model.options).normalized()
        scene_crop, crop_xywh = _crop_bgr(scene_bgr, search_roi)
        template_gray = self._load_template_gray()
        template_mask_gray = self._load_template_mask_gray()
        refinement_mode = str(
            getattr(self.model, "pose_refinement", "") or ""
        ).strip().lower()

        t0 = time.perf_counter()
        runtime_options = normalized_options
        angle_prefiltered = False
        template_bgr: Optional[np.ndarray] = None
        match_scene = scene_crop
        pose_search_inverse: Optional[np.ndarray] = None
        pose_search_origin = (0.0, 0.0)
        if refinement_mode == "saturation_rect":
            template_bgr = self._load_template_bgr()
            prepared_pose_search = _prepare_saturation_axis_search(
                scene_crop,
                template_bgr,
                normalized_options,
            )
            if prepared_pose_search is not None:
                (
                    match_scene,
                    runtime_options,
                    pose_search_inverse,
                    pose_search_origin,
                ) = prepared_pose_search
                angle_prefiltered = True
        backend_name = "python-ncc-mask" if template_mask_gray is not None else "python-ncc"
        matches: Tuple[NccMatchResult, ...]

        if template_mask_gray is None and prefer_native and NccNativeApi.is_available():
            try:
                matcher = self._get_native_matcher(runtime_options.min_reduced_area)
                source_gray = _prepare_gray_image(match_scene, bitwise_not=runtime_options.bitwise_not)
                native_payload = matcher.match(source_gray, _native_payload(runtime_options))
                matches = _parse_native_response(native_payload)
                backend_name = "native-ncc"
            except Exception:
                matches = _match_python(match_scene, template_gray, runtime_options)
        else:
            matches = _match_python(
                match_scene,
                template_gray,
                runtime_options,
                mask_gray=template_mask_gray,
            )
        if angle_prefiltered and not matches:
            match_scene = scene_crop
            runtime_options = normalized_options
            pose_search_inverse = None
            pose_search_origin = (0.0, 0.0)
            if template_mask_gray is None and prefer_native and NccNativeApi.is_available():
                try:
                    matcher = self._get_native_matcher(runtime_options.min_reduced_area)
                    source_gray = _prepare_gray_image(
                        match_scene,
                        bitwise_not=runtime_options.bitwise_not,
                    )
                    native_payload = matcher.match(
                        source_gray,
                        _native_payload(runtime_options),
                    )
                    matches = _parse_native_response(native_payload)
                    backend_name = "native-ncc+angle-prefilter-fallback"
                except Exception:
                    matches = _match_python(
                        match_scene,
                        template_gray,
                        runtime_options,
                    )
                    backend_name = "python-ncc+angle-prefilter-fallback"
            else:
                matches = _match_python(
                    match_scene,
                    template_gray,
                    runtime_options,
                    mask_gray=template_mask_gray,
                )
                backend_name = "python-ncc-mask+angle-prefilter-fallback"
            angle_prefiltered = False
        if angle_prefiltered:
            backend_name = f"{backend_name}+angle-prefilter"

        orientation_anchor = getattr(self.model, "orientation_anchor", None)
        if orientation_anchor is not None and matches:
            prepared_scene = _prepare_gray_image(
                match_scene,
                bitwise_not=runtime_options.bitwise_not,
            )
            prepared_template = _prepare_gray_image(
                template_gray,
                bitwise_not=runtime_options.bitwise_not,
            )
            matches, orientation_checked = _disambiguate_top_orientation(
                prepared_scene,
                prepared_template,
                matches,
                orientation_anchor,
            )
            if orientation_checked:
                backend_name = f"{backend_name}+orientation-anchor"

        if refinement_mode == "saturation_rect" and matches:
            refined = _refine_match_by_saturation_rect(
                match_scene,
                template_bgr if template_bgr is not None else self._load_template_bgr(),
                matches[0],
            )
            if refined is not None:
                matches = (refined, *matches[1:])
                backend_name = f"{backend_name}+saturation-rect"

        if pose_search_inverse is not None and matches:
            matches = tuple(
                _restore_match_from_pose_search(
                    item,
                    pose_search_inverse,
                    pose_search_origin,
                )
                for item in matches
            )

        if crop_xywh[0] or crop_xywh[1]:
            matches = tuple(_offset_match(item, crop_xywh[0], crop_xywh[1]) for item in matches)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return NccMatchResponse(matches=matches, elapsed_ms=elapsed_ms, backend_name=backend_name)

    @staticmethod
    def render_matches(scene_bgr: np.ndarray, matches: Sequence[NccMatchResult], label: str = "") -> np.ndarray:
        canvas = np.ascontiguousarray(scene_bgr.copy())
        for index, item in enumerate(matches, start=1):
            points = np.array(item.quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
            text = f"{label}{index}: {item.score:.3f} / {item.angle:.1f}deg" if label else f"{item.score:.3f} / {item.angle:.1f}deg"
            anchor = (max(0, int(item.bbox.x)), max(18, int(item.bbox.y) - 6))
            cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas


__all__ = ["NccCompiledModel", "NccMatchResponse"]
