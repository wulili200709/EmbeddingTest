"""ROI-related ToolPage helpers."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6 import QtGui

from common import labelme_io

from shape.core import locator as shape_locator
from shape.core.recipe_labels import output_labels_from_shape_recipe
from ui.debug import OverlayShape
from ui.roi_overlay_colors import (
    SEARCH_REGION_COLOR,
    SEARCH_REGION_WIDTH,
    overlay_style_for_label,
)

_SELECTED_TOOL_COLOR = QtGui.QColor("#00C8C8")
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


def _selected_tool_roi_label(tool_page) -> str:
    selected_item_fn = getattr(tool_page, "_selected_inspection_item", None)
    if not callable(selected_item_fn):
        return ""
    selected_item = selected_item_fn()
    if selected_item is None:
        return ""
    return str(getattr(selected_item, "roi_label", "") or "").strip()


def _overlay_style_for_tool_label(tool_page, img_path: str, label: str) -> tuple[QtGui.QColor, float, bool]:
    status = tool_page._roi_status_for_path(img_path, label)
    color, width, dash = overlay_style_for_label(label, status=status)
    selected_label = _selected_tool_roi_label(tool_page)
    if label and label == selected_label:
        return QtGui.QColor(_SELECTED_TOOL_COLOR), max(float(width), 3.0), False
    return color, width, dash


def _roi_overlay_color(tool_page, img_path: str, label: str) -> QtGui.QColor:
    color, _width, _dash = _overlay_style_for_tool_label(tool_page, img_path, label)
    return color


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


def _measurement_overlays_for_path(tool_page, img_path: str) -> List[OverlayShape]:
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


def _canvas_roi_style_for_label(tool_page, img_path: str, label_name: str) -> tuple[QtGui.QColor, bool, float]:
    label = str(label_name or "").strip()
    color, width, dash = _overlay_style_for_tool_label(tool_page, img_path, label)
    return color, dash, width


def _load_canvas_image(tool_page, path: str) -> None:
    from . import page as page_module

    tool_page.canvas.set_image(path, pixmap=page_module._pixmap_from_path(path))
    tool_page._load_shape_for_label(path, tool_page._current_label())


def _set_status_for_current_image(tool_page, path: str) -> None:
    match_ms = tool_page._shape_match_ms_by_image.get(path)
    total_ms = tool_page._shape_autogen_ms_by_image.get(path)
    if match_ms is None and total_ms is None:
        updater = getattr(tool_page, "_update_sample_panel_widgets", None)
        if callable(updater):
            updater()
        return
    parts = [f"当前图像: {os.path.basename(path)}"]
    if match_ms is not None:
        parts.append(f"模板匹配={match_ms:.1f}ms")
    if total_ms is not None:
        parts.append(f"生成 ROI={total_ms:.1f}ms")
    tool_page.lbl_status.setText("状态: " + "  ".join(parts))


def _current_label(tool_page) -> str:
    return tool_page.cmb_label.currentText()


def _update_save_label_text(tool_page) -> None:
    label = tool_page._current_label()
    tool_page.btn_save.setText(f"淇濆瓨鏍囨敞({label}) -> labelme json")


def _set_overlay_shapes(tool_page, img_path: str, current_label: str) -> None:
    j = labelme_io.labelme_json_of_image(img_path)
    overlays: List[OverlayShape] = []
    visible_roi_labels: Optional[set[str]] = None

    recipe = tool_page.shape_recipe_for_role(tool_page.current_camera_role())
    if tool_page.loc_method == "shape":
        labels = [str(label).strip() for label in output_labels_from_shape_recipe(recipe) if str(label).strip()]
        visible_roi_labels = set(labels) if labels else None

    if tool_page.loc_method == "shape" and recipe is not None and recipe.search_points:
        points = [
            (float(pt[0]), float(pt[1]))
            for pt in (recipe.search_points or [])
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if len(points) >= 2:
            if str(recipe.search_shape_type or "rectangle") == "rectangle" and len(points) == 2:
                (x0, y0), (x1, y1) = points[:2]
                x = int(round(min(x0, x1)))
                y = int(round(min(y0, y1)))
                w = max(1, int(round(abs(x1 - x0))))
                h = max(1, int(round(abs(y1 - y0))))
                overlays.append(
                    OverlayShape(
                        shape_type="rect",
                        xywh=(x, y, w, h),
                        color=QtGui.QColor(SEARCH_REGION_COLOR),
                        width=SEARCH_REGION_WIDTH,
                        dash=False,
                    )
                )
            elif len(points) >= 3:
                overlays.append(
                    OverlayShape(
                        shape_type="polygon",
                        points=points,
                        color=QtGui.QColor(SEARCH_REGION_COLOR),
                        width=SEARCH_REGION_WIDTH,
                        dash=False,
                    )
                )

    if not os.path.exists(j):
        overlays.extend(_measurement_overlays_for_path(tool_page, img_path))
        tool_page.canvas.set_overlays(overlays)
        return

    def add_shape(label: str, color: QtGui.QColor, *, width: float, dash: bool = False) -> None:
        poly_pts = labelme_io.try_read_polygon_points_from_labelme(j, label)
        if poly_pts and len(poly_pts) >= 3:
            overlays.append(OverlayShape(shape_type="polygon", points=poly_pts, color=color, width=width, dash=dash))
            return
        xywh = labelme_io.try_read_xywh_from_labelme(j, label)
        if xywh:
            overlays.append(OverlayShape(shape_type="rect", xywh=xywh, color=color, width=width, dash=dash))

    seen_labels: set[str] = set()
    for label in labelme_io.sorted_label_names_from_labelme(j, label_prefix="roi"):
        if visible_roi_labels is not None and label not in visible_roi_labels:
            continue
        if label == current_label:
            continue
        seen_labels.add(label)
        color, width, dash = _overlay_style_for_tool_label(tool_page, img_path, label)
        add_shape(label, color, width=width, dash=dash)

    for label in ["anchor", "roi", "anchor_mask"]:
        if label == current_label or label in seen_labels:
            continue
        color, width, dash = _overlay_style_for_tool_label(tool_page, img_path, label)
        add_shape(label, color, width=width, dash=dash)

    overlays.extend(_measurement_overlays_for_path(tool_page, img_path))
    tool_page.canvas.set_overlays(overlays)


def _load_shape_for_label(tool_page, img_path: str, label_name: str) -> None:
    tool_page.canvas.clear_roi()
    roi_color, roi_dash, roi_width = _canvas_roi_style_for_label(tool_page, img_path, label_name)
    tool_page.canvas.set_roi_style(
        roi_color=roi_color,
        roi_dash=roi_dash,
        roi_width=roi_width,
        preview_width=roi_width,
    )
    j = labelme_io.labelme_json_of_image(img_path)
    loaded = False
    if os.path.exists(j):
        poly_pts = labelme_io.try_read_polygon_points_from_labelme(j, label_name)
        if poly_pts and len(poly_pts) >= 3:
            tool_page.canvas.set_roi_polygon(poly_pts)
            tool_page.cmb_shape.setCurrentText("polygon")
            loaded = True
        xywh = labelme_io.try_read_xywh_from_labelme(j, label_name)
        if xywh:
            tool_page.canvas.set_roi_rect(xywh)
            tool_page.cmb_shape.setCurrentText("rect")
            loaded = True
    tool_page._set_overlay_shapes(img_path, label_name)
    if not loaded:
        tool_page._on_shapes_changed()


def _on_shapes_changed(tool_page) -> None:
    p = tool_page.canvas.image_path()
    if p is None:
        tool_page.btn_save.setEnabled(False)
        return
    st = tool_page.canvas.roi
    ok = (st.shape_type == "rect" and st.xywh is not None) or (st.shape_type == "polygon" and st.points is not None)
    tool_page.btn_save.setEnabled(ok)


def _roi_xywh_from_canvas(tool_page) -> Optional[Tuple[int, int, int, int]]:
    roi = tool_page.canvas.roi_xywh()
    if roi is not None:
        return roi
    p = tool_page.canvas.image_path()
    if p:
        j = labelme_io.labelme_json_of_image(p)
        if os.path.exists(j):
            xywh = labelme_io.try_read_xywh_from_labelme(j, "roi")
            if xywh:
                return xywh
    return None



