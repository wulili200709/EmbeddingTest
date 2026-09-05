from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from algorithms.measurement_types import PinTipPointConfig


def _dark_foreground_binary(
    crop_bgr: np.ndarray,
    valid_mask: np.ndarray,
    config: PinTipPointConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    if config.blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid = np.asarray(valid_mask, dtype=np.uint8) > 0
    values = gray[valid]
    if values.size < 20:
        raise RuntimeError("pin-tip ROI has too few valid pixels")
    if config.threshold > 0.0:
        threshold = float(config.threshold)
    else:
        threshold, _unused = cv2.threshold(
            values.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        threshold = float(threshold)
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[valid & (gray <= threshold)] = 255
    if config.morph_open_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morph_open_size, config.morph_open_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if config.morph_close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morph_close_size, config.morph_close_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary[~valid] = 0
    return binary, threshold, gray


def _component_score(
    stats: np.ndarray,
    *,
    crop_size: tuple[int, int],
) -> float:
    x, y, width, height, area = [float(value) for value in stats]
    crop_w, crop_h = float(max(1, crop_size[0])), float(max(1, crop_size[1]))
    center_x = x + width * 0.5
    center_factor = max(0.15, 1.0 - abs(center_x - crop_w * 0.5) / (crop_w * 0.65))
    height_factor = min(2.0, height / max(1.0, crop_h * 0.35))
    bottom_factor = 0.5 + min(1.0, (y + height) / crop_h)
    top_bonus = 1.35 if y <= max(3.0, crop_h * 0.12) else 1.0
    return float(area * center_factor * height_factor * bottom_factor * top_bonus)


def _select_component(
    binary: np.ndarray,
    config: PinTipPointConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    crop_h, crop_w = binary.shape[:2]
    candidates: list[tuple[float, int]] = []
    for label in range(1, int(count)):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < config.min_area_px:
            continue
        if width < config.min_width_px or height < config.min_height_px:
            continue
        score = _component_score(stats[label], crop_size=(crop_w, crop_h))
        candidates.append((score, label))
    if not candidates:
        raise RuntimeError("dark pin contour not found")
    _score, selected_label = max(candidates, key=lambda item: item[0])
    x, y, width, height, area = [int(value) for value in stats[selected_label]]
    component = labels == int(selected_label)
    return component, (x, y, width, height), float(area)


def _runs(indices: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted({int(value) for value in indices})
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _tip_arc_points(
    component: np.ndarray,
    config: PinTipPointConfig,
) -> tuple[np.ndarray, int, tuple[int, int]]:
    crop_h, crop_w = component.shape[:2]
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        raise RuntimeError("pin-tip component is empty")
    bottom_by_x = np.full(crop_w, -1, dtype=np.int32)
    np.maximum.at(bottom_by_x, xs, ys)
    raw_bottom = int(np.max(bottom_by_x))
    if raw_bottom >= crop_h - 1 - int(config.border_margin_px):
        raise RuntimeError("pin tip touches ROI bottom; enlarge or move the ROI downward")

    band_depth = max(5, int(round(float(crop_h) * float(config.tip_band_ratio))))
    near_columns = np.flatnonzero(bottom_by_x >= raw_bottom - band_depth)
    candidate_runs = _runs(near_columns.tolist())
    if not candidate_runs:
        raise RuntimeError("pin-tip bottom arc not found")
    roi_center_x = (float(crop_w) - 1.0) * 0.5

    def run_score(run: tuple[int, int]) -> float:
        left, right = run
        run_bottom = int(np.max(bottom_by_x[left:right + 1]))
        run_width = float(right - left + 1)
        run_center = (float(left) + float(right)) * 0.5
        center_penalty = abs(run_center - roi_center_x) / max(1.0, float(crop_w))
        return float(run_bottom) + min(20.0, run_width) * 0.05 - center_penalty * float(crop_h) * 0.15

    selected_run = max(candidate_runs, key=run_score)
    left, right = selected_run
    selected_bottom = int(np.max(bottom_by_x[left:right + 1]))
    run_width = max(1, right - left + 1)
    arc_depth = max(3, int(round(float(run_width) * float(config.arc_depth_ratio))))
    arc_columns = np.arange(left, right + 1, dtype=np.int32)
    arc_columns = arc_columns[bottom_by_x[arc_columns] >= selected_bottom - arc_depth]
    if len(arc_columns) < config.min_arc_points:
        raise RuntimeError(
            f"pin-tip arc has too few points: {len(arc_columns)}/{config.min_arc_points}"
        )
    points = np.column_stack(
        (arc_columns.astype(np.float64), bottom_by_x[arc_columns].astype(np.float64))
    )
    return points, selected_bottom, selected_run


def _fit_tip_vertex(
    arc_points: np.ndarray,
    raw_bottom: int,
) -> tuple[tuple[float, float], float, np.ndarray]:
    points = np.asarray(arc_points, dtype=np.float64).reshape(-1, 2)
    kept = np.ones(len(points), dtype=bool)
    center_x = float(np.mean(points[:, 0]))
    coefficients: np.ndarray | None = None
    residuals = np.zeros(len(points), dtype=np.float64)
    for _iteration in range(3):
        fit_points = points[kept]
        if len(fit_points) < 5:
            break
        z = fit_points[:, 0] - center_x
        coefficients = np.polyfit(z, fit_points[:, 1], 2)
        predicted = np.polyval(coefficients, points[:, 0] - center_x)
        residuals = points[:, 1] - predicted
        fit_residuals = np.abs(residuals[kept])
        median = float(np.median(fit_residuals))
        mad = float(np.median(np.abs(fit_residuals - median)))
        limit = max(1.25, median + 3.5 * max(0.25, mad))
        refined = np.abs(residuals) <= limit
        if int(np.count_nonzero(refined)) < 5 or np.array_equal(refined, kept):
            break
        kept = refined

    if coefficients is not None and float(coefficients[0]) < -1e-4:
        vertex_z = -float(coefficients[1]) / (2.0 * float(coefficients[0]))
        vertex_x = center_x + vertex_z
        min_x = float(np.min(points[kept, 0]))
        max_x = float(np.max(points[kept, 0]))
        if min_x - 1.0 <= vertex_x <= max_x + 1.0:
            vertex_y = float(np.polyval(coefficients, vertex_z))
            vertex_y = max(float(raw_bottom) - 2.0, min(float(raw_bottom) + 1.0, vertex_y))
            rms = float(np.sqrt(np.mean(np.square(residuals[kept]))))
            return (float(vertex_x), float(vertex_y)), rms, points[kept]

    deepest = points[points[:, 1] >= float(raw_bottom) - 1.0]
    if len(deepest) == 0:
        deepest = points
    tip_x = float(np.median(deepest[:, 0]))
    rms = float(np.std(deepest[:, 1])) if len(deepest) > 1 else 0.0
    return (tip_x, float(raw_bottom)), rms, points


def _axis_direction(
    component: np.ndarray,
    tip_xy: tuple[float, float],
    tip_width: int,
) -> tuple[float, float]:
    ys, xs = np.nonzero(component)
    tip_x, tip_y = tip_xy
    upper = tip_y - max(4.0, float(tip_width) * 0.45)
    lower = tip_y - max(16.0, float(tip_width) * 3.0)
    keep = (
        (ys.astype(np.float64) <= upper)
        & (ys.astype(np.float64) >= lower)
        & (np.abs(xs.astype(np.float64) - tip_x) <= max(4.0, float(tip_width) * 0.8))
    )
    points = np.column_stack((xs[keep], ys[keep])).astype(np.float32)
    if len(points) < 10:
        return 0.0, 1.0
    vx, vy, _x0, _y0 = [float(value) for value in cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)]
    length = max(1e-9, math.hypot(vx, vy))
    vx, vy = vx / length, vy / length
    if vy < 0.0:
        vx, vy = -vx, -vy
    if abs(vy) < 0.45:
        return 0.0, 1.0
    return float(vx), float(vy)


def locate_pin_tip_in_crop(
    crop_bgr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: PinTipPointConfig,
) -> dict[str, object]:
    binary, threshold, gray = _dark_foreground_binary(crop_bgr, valid_mask, config)
    component, bbox, area = _select_component(binary, config)
    arc_points, raw_bottom, selected_run = _tip_arc_points(component, config)
    tip_xy, fit_residual, used_points = _fit_tip_vertex(arc_points, raw_bottom)
    run_width = max(1, int(selected_run[1] - selected_run[0] + 1))
    axis = _axis_direction(component, tip_xy, run_width)

    foreground_values = gray[component]
    background_values = gray[(np.asarray(valid_mask) > 0) & ~component]
    contrast = 0.0
    if foreground_values.size and background_values.size:
        contrast = float(np.median(background_values) - np.median(foreground_values))
    contrast_score = max(0.0, min(1.0, contrast / 120.0))
    residual_score = max(0.0, min(1.0, 1.0 - fit_residual / 3.0))
    confidence = max(0.0, min(1.0, contrast_score * 0.65 + residual_score * 0.35))
    return {
        "point_xy": tip_xy,
        "axis_direction": axis,
        "threshold": float(threshold),
        "confidence": float(confidence),
        "fit_residual": float(fit_residual),
        "component_area_px": float(area),
        "component_bbox_xywh": bbox,
        "edge_points": tuple((float(x), float(y)) for x, y in used_points),
    }


__all__ = ["locate_pin_tip_in_crop"]
