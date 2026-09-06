from __future__ import annotations

import math

import cv2
import numpy as np

from algorithms.measurement_lines import _line_segment_in_crop, fit_line_filtered
from algorithms.measurement_pin_tip import (
    _dark_foreground_binary,
    _fit_tip_vertex,
    _tip_arc_points,
)
from algorithms.measurement_types import (
    FittedLine,
    MultiPinTipHeightConfig,
    PinTipPointConfig,
)


def _reference_edge_points(
    crop_bgr: np.ndarray,
    valid_mask: np.ndarray,
    config: MultiPinTipHeightConfig,
) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid = np.asarray(valid_mask, dtype=np.uint8) > 0
    height, width = gray.shape[:2]
    search_bottom = max(2, min(height - 1, int(round(height * config.reference_search_ratio))))
    delta = gray[1:search_bottom, :] - gray[: search_bottom - 1, :]
    adjacent_valid = valid[1:search_bottom, :] & valid[: search_bottom - 1, :]
    points: list[tuple[float, float]] = []
    for x in range(0, width, config.reference_scan_step):
        response = np.where(adjacent_valid[:, x], delta[:, x], -np.inf)
        if response.size == 0:
            continue
        index = int(np.argmax(response))
        if not math.isfinite(float(response[index])) or float(response[index]) < config.reference_edge_threshold:
            continue
        points.append((float(x), float(index + 1)))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _fit_reference_line(
    crop_bgr: np.ndarray,
    valid_mask: np.ndarray,
    config: MultiPinTipHeightConfig,
) -> tuple[FittedLine, np.ndarray]:
    points = _reference_edge_points(crop_bgr, valid_mask, config)
    line, used_points = fit_line_filtered(
        points,
        min_points=config.reference_min_points,
        context="multi-pin housing reference",
    )
    if abs(float(line.vx)) < 0.65:
        raise RuntimeError("housing reference edge is too steep; include the lower housing edge in the ROI")
    return line, used_points


def _point_config(config: MultiPinTipHeightConfig) -> PinTipPointConfig:
    return PinTipPointConfig(
        roi_label=config.roi_label,
        threshold=config.threshold,
        blur_ksize=config.blur_ksize,
        morph_open_size=config.morph_open_size,
        morph_close_size=config.morph_close_size,
        min_area_px=config.min_area_px,
        min_width_px=config.min_width_px,
        min_height_px=config.min_height_px,
        border_margin_px=config.border_margin_px,
        tip_band_ratio=config.tip_band_ratio,
        arc_depth_ratio=config.arc_depth_ratio,
        min_arc_points=config.min_arc_points,
    )


def locate_multi_pin_tips_in_crop(
    crop_bgr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: MultiPinTipHeightConfig,
    reference_segment: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict[str, object]:
    binary, threshold, _gray = _dark_foreground_binary(
        crop_bgr,
        valid_mask,
        _point_config(config),
    )
    external_reference = reference_segment is not None
    if external_reference:
        endpoints = np.asarray(reference_segment, dtype=np.float64)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise RuntimeError("invalid reference line result")
        direction = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            raise RuntimeError("reference line is too short")
        reference_line = FittedLine(
            vx=float(direction[0] / length), vy=float(direction[1] / length),
            x0=float(endpoints[0, 0]), y0=float(endpoints[0, 1]),
            residual=0.0, point_count=2,
        )
        reference_points = endpoints
    else:
        reference_line, reference_points = _fit_reference_line(crop_bgr, valid_mask, config)
    height, width = binary.shape[:2]

    tangent_x, tangent_y = float(reference_line.vx), float(reference_line.vy)
    tangent_norm = max(1e-12, math.hypot(tangent_x, tangent_y))
    tangent_x, tangent_y = tangent_x / tangent_norm, tangent_y / tangent_norm
    if tangent_x < 0.0:
        tangent_x, tangent_y = -tangent_x, -tangent_y
    normal_x, normal_y = -tangent_y, tangent_x
    if normal_y < 0.0:
        normal_x, normal_y = -normal_x, -normal_y

    yy, xx = np.indices((height, width), dtype=np.float32)
    signed_distance = (
        (xx - float(reference_line.x0)) * normal_x
        + (yy - float(reference_line.y0)) * normal_y
    )
    pin_binary = np.zeros_like(binary)
    pin_binary[
        (binary > 0)
        & (np.asarray(valid_mask, dtype=np.uint8) > 0)
        & (signed_distance > float(config.reference_cut_margin_px))
    ] = 255

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(pin_binary, 8)
    expected = max(1, int(config.expected_pin_count))
    max_component_width = max(
        float(config.min_width_px) * 3.0,
        float(width) / float(expected) * 0.68,
    )
    near_reference_limit = float(config.reference_cut_margin_px) + max(14.0, float(height) * 0.1)
    point_config = _point_config(config)
    candidates: list[dict[str, object]] = []

    for label in range(1, int(count)):
        x, y, component_width, component_height, area = [int(value) for value in stats[label]]
        if area < config.min_area_px:
            continue
        if component_width < config.min_width_px or component_height < config.min_height_px:
            continue
        if float(component_width) > max_component_width:
            continue
        component = labels == int(label)
        component_signed = signed_distance[component]
        if component_signed.size == 0:
            continue
        if not external_reference and float(np.min(component_signed)) > near_reference_limit:
            continue
        try:
            arc_points, raw_bottom, selected_run = _tip_arc_points(component, point_config)
            tip_xy, fit_residual, used_points = _fit_tip_vertex(arc_points, raw_bottom)
        except RuntimeError:
            continue
        distance_px = (
            (float(tip_xy[0]) - float(reference_line.x0)) * normal_x
            + (float(tip_xy[1]) - float(reference_line.y0)) * normal_y
        )
        if distance_px <= 0.0:
            continue
        order_position = (
            (float(tip_xy[0]) - float(reference_line.x0)) * tangent_x
            + (float(tip_xy[1]) - float(reference_line.y0)) * tangent_y
        )
        candidates.append(
            {
                "point_xy": (float(tip_xy[0]), float(tip_xy[1])),
                "distance_px": float(distance_px),
                "order_position": float(order_position),
                "fit_residual": float(fit_residual),
                "bbox_xywh": (x, y, component_width, component_height),
                "edge_points": tuple((float(px), float(py)) for px, py in used_points),
                "selected_run": selected_run,
            }
        )

    candidates.sort(key=lambda item: float(item["order_position"]))
    if not candidates:
        raise RuntimeError("no pin tips found below the housing reference edge")
    reference_segment = reference_segment if external_reference else _line_segment_in_crop(
        reference_line,
        crop_width=width,
        crop_height=height,
        origin=(0, 0),
    )
    return {
        "candidates": tuple(candidates),
        "reference_line": reference_line,
        "reference_line_segment": reference_segment,
        "reference_edge_points": tuple(
            (float(point[0]), float(point[1])) for point in reference_points
        ),
        "threshold": float(threshold),
    }


__all__ = ["locate_multi_pin_tips_in_crop"]
