"""Measurement result overlays for ROI canvas."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6 import QtGui

from ui.debug import OverlayShape

_MEASUREMENT_POINT_COLOR = QtGui.QColor("#FFD54F")
_MEASUREMENT_LINE_OK_COLOR = QtGui.QColor("#00E676")
_MEASUREMENT_LINE_NG_COLOR = QtGui.QColor("#FF5252")
_MEASUREMENT_LINE_COLOR = QtGui.QColor("#40C4FF")
_CENTER_DISTANCE_OK_COLORS = (
    QtGui.QColor("#00E676"),
    QtGui.QColor("#40C4FF"),
    QtGui.QColor("#FFB300"),
    QtGui.QColor("#C084FC"),
)

def _is_same_image_path(left: object, right: object) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    try:
        return os.path.normcase(os.path.abspath(left_text)) == os.path.normcase(os.path.abspath(right_text))
    except Exception:
        return os.path.normcase(left_text) == os.path.normcase(right_text)


def _point_tuple(value: object) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _center_points_from_measurement(measurement: dict) -> list[Tuple[float, float]]:
    raw_centers = measurement.get("center_points")
    points: list[Tuple[float, float]] = []
    if isinstance(raw_centers, list):
        for point in raw_centers:
            parsed = _point_tuple(point)
            if parsed is not None:
                points.append(parsed)
    return points


def _offset_center_distance_dimension(
    measurement: dict,
    dimension: Tuple[Tuple[float, float], Tuple[float, float]],
    index: int,
) -> tuple[
    Tuple[Tuple[float, float], Tuple[float, float]],
    list[Tuple[Tuple[float, float], Tuple[float, float]]],
    Tuple[float, float],
]:
    centers = _center_points_from_measurement(measurement)
    p0, p1 = dimension
    if len(centers) >= 2:
        c0, c1 = centers[0], centers[1]
    else:
        c0, c1 = p0, p1
    mode = str(measurement.get("distance_mode", "vertical") or "vertical").strip().lower()
    step = 46.0
    offset = 34.0 + float(index) * step
    if mode == "horizontal":
        anchor_y = max(float(c0[1]), float(c1[1]), float(p0[1]), float(p1[1])) + offset
        shifted = ((float(c0[0]), anchor_y), (float(c1[0]), anchor_y))
        leaders = [
            ((float(c0[0]), float(c0[1])), shifted[0]),
            ((float(c1[0]), float(c1[1])), shifted[1]),
        ]
        text_pos = ((float(c0[0]) + float(c1[0])) * 0.5, anchor_y + 20.0)
        return shifted, leaders, text_pos
    if mode == "euclidean":
        dx = float(c1[0]) - float(c0[0])
        dy = float(c1[1]) - float(c0[1])
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx = -dy / length
        ny = dx / length
        shifted = (
            (float(c0[0]) + nx * offset, float(c0[1]) + ny * offset),
            (float(c1[0]) + nx * offset, float(c1[1]) + ny * offset),
        )
        leaders = [
            ((float(c0[0]), float(c0[1])), shifted[0]),
            ((float(c1[0]), float(c1[1])), shifted[1]),
        ]
        text_pos = (
            (shifted[0][0] + shifted[1][0]) * 0.5 + nx * 18.0,
            (shifted[0][1] + shifted[1][1]) * 0.5 + ny * 18.0,
        )
        return shifted, leaders, text_pos

    anchor_x = max(float(c0[0]), float(c1[0]), float(p0[0]), float(p1[0])) + offset
    shifted = ((anchor_x, float(c0[1])), (anchor_x, float(c1[1])))
    leaders = [
        ((float(c0[0]), float(c0[1])), shifted[0]),
        ((float(c1[0]), float(c1[1])), shifted[1]),
    ]
    text_pos = (anchor_x + 44.0, (float(c0[1]) + float(c1[1])) * 0.5)
    return shifted, leaders, text_pos


def measurement_overlays_for_path(tool_page, img_path: str) -> List[OverlayShape]:
    overlays: List[OverlayShape] = []
    center_distance_index = 0
    for row in list(getattr(tool_page, "_current_result_rows", []) or []):
        if not isinstance(row, dict):
            continue
        row_path = row.get("file_path")
        if row_path:
            if not _is_same_image_path(row_path, img_path):
                continue
        else:
            image_name = os.path.basename(str(img_path or ""))
            row_name = str(row.get("file_name", "") or "")
            if image_name and not row_name.startswith(image_name):
                continue
        measurement = row.get("measurement")
        if not isinstance(measurement, dict):
            continue
        pred = str(row.get("pred", "") or "").strip().upper()
        line_color = (
            _MEASUREMENT_LINE_OK_COLOR
            if pred == "OK"
            else _MEASUREMENT_LINE_NG_COLOR
            if pred == "NG"
            else _MEASUREMENT_LINE_COLOR
        )
        measurement_type = str(measurement.get("type", "") or "")
        if measurement_type in {"pin_center_distance", "bright_block_y_distance", "bright_block_center"}:
            raw_dimension = measurement.get("dimension_segment")
            dimension: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
            if isinstance(raw_dimension, (list, tuple)) and len(raw_dimension) >= 2:
                p0 = _point_tuple(raw_dimension[0])
                p1 = _point_tuple(raw_dimension[1])
                if p0 is not None and p1 is not None:
                    dimension = (p0, p1)
            if dimension is not None:
                overlays.append(
                    OverlayShape(
                        shape_type="segments",
                        segments=[dimension],
                        color=QtGui.QColor(line_color),
                        width=3.0,
                        dash=False,
                    )
                )
            raw_candidates = measurement.get("candidates")
            if isinstance(raw_candidates, list):
                for candidate in raw_candidates:
                    if not isinstance(candidate, dict):
                        continue
                    raw_box = candidate.get("box_points")
                    box_points = []
                    if isinstance(raw_box, list):
                        for point in raw_box:
                            parsed = _point_tuple(point)
                            if parsed is not None:
                                box_points.append(parsed)
                    if len(box_points) >= 3:
                        overlays.append(
                            OverlayShape(
                                shape_type="polygon",
                                points=box_points,
                                color=QtGui.QColor(line_color),
                                width=2.0,
                                dash=False,
                            )
                        )
            raw_centers = measurement.get("center_points")
            center_points = []
            if isinstance(raw_centers, list):
                for point in raw_centers:
                    parsed = _point_tuple(point)
                    if parsed is not None:
                        center_points.append(parsed)
            if center_points:
                overlays.append(
                    OverlayShape(
                        shape_type="points",
                        points=center_points,
                        color=QtGui.QColor(_MEASUREMENT_POINT_COLOR),
                        width=9.0,
                        dash=False,
                    )
                )
            continue
        if measurement_type in {"line_distance", "line_distance_ref_normal", "center_distance"}:
            raw_dimension = measurement.get("dimension_segment")
            dimension: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
            if isinstance(raw_dimension, (list, tuple)) and len(raw_dimension) >= 2:
                p0 = _point_tuple(raw_dimension[0])
                p1 = _point_tuple(raw_dimension[1])
                if p0 is not None and p1 is not None:
                    dimension = (p0, p1)
            if dimension is not None:
                text_pos = None
                if measurement_type == "center_distance":
                    if pred != "NG":
                        line_color = _CENTER_DISTANCE_OK_COLORS[
                            center_distance_index % len(_CENTER_DISTANCE_OK_COLORS)
                        ]
                    dimension, leader_segments, text_pos = _offset_center_distance_dimension(
                        measurement,
                        dimension,
                        center_distance_index,
                    )
                    center_distance_index += 1
                    if leader_segments:
                        overlays.append(
                            OverlayShape(
                                shape_type="segments",
                                segments=leader_segments,
                                color=QtGui.QColor(line_color),
                                width=1.6,
                                dash=True,
                            )
                        )
                overlays.append(
                    OverlayShape(
                        shape_type="dimension",
                        segments=[dimension],
                        text=str(measurement.get("label", "") or ""),
                        text_pos=text_pos,
                        color=QtGui.QColor(line_color),
                        width=3.0,
                        dash=False,
                    )
                )
            center_points = _center_points_from_measurement(measurement)
            if center_points:
                overlays.append(
                    OverlayShape(
                        shape_type="points",
                        points=center_points,
                        color=QtGui.QColor(_MEASUREMENT_POINT_COLOR),
                        width=9.0,
                        dash=False,
                    )
                )
            continue
        raw_points = measurement.get("edge_points")
        points = []
        if isinstance(raw_points, list):
            for point in raw_points:
                parsed = _point_tuple(point)
                if parsed is not None:
                    points.append(parsed)
        if points:
            overlays.append(
                OverlayShape(
                    shape_type="points",
                    points=points,
                    color=QtGui.QColor(_MEASUREMENT_POINT_COLOR),
                    width=4.0,
                    dash=False,
                )
            )
        raw_segment = measurement.get("line_segment")
        segment: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        if isinstance(raw_segment, (list, tuple)) and len(raw_segment) >= 2:
            p0 = _point_tuple(raw_segment[0])
            p1 = _point_tuple(raw_segment[1])
            if p0 is not None and p1 is not None:
                segment = (p0, p1)
        if segment is not None:
            overlays.append(
                OverlayShape(
                    shape_type="segments",
                    segments=[segment],
                    color=QtGui.QColor(line_color),
                    width=3.0,
                    dash=False,
                )
            )
    return overlays


