"""ROI-related ToolPage helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from PySide6 import QtGui

from common import labelme_io
from common.camera_roles import DEFAULT_CAMERA_ROLE, normalize_camera_role

from shape.core.recipe_labels import output_labels_from_shape_recipe
from ui.debug import OverlayShape
from ui.debug.tool_page.roi_measurement_overlays import measurement_overlays_for_path
from ui.i18n import tr
from ui.roi_overlay_colors import (
    ROI_DISABLED_COLOR,
    SEARCH_REGION_COLOR,
    SEARCH_REGION_WIDTH,
    is_roi_label,
    overlay_style_for_label,
)

_SELECTED_TOOL_COLOR = QtGui.QColor("#00C8C8")


def _selected_tool_roi_label(tool_page) -> str:
    selected_item_fn = getattr(tool_page, "_selected_inspection_item", None)
    if not callable(selected_item_fn):
        return ""
    selected_item = selected_item_fn()
    if selected_item is None:
        return ""
    return str(getattr(selected_item, "roi_label", "") or "").strip()


def _roi_label_enabled_for_current_role(tool_page, label: str) -> bool:
    label_text = str(label or "").strip()
    if not is_roi_label(label_text):
        return True
    current_role = normalize_camera_role(
        tool_page.current_camera_role() if hasattr(tool_page, "current_camera_role") else "",
        default=DEFAULT_CAMERA_ROLE,
    )
    found = False
    enabled = True
    for item in list(getattr(tool_page, "inspection_items", []) or []):
        item_label = str(getattr(item, "roi_label", "") or "").strip()
        if item_label != label_text:
            continue
        item_role = normalize_camera_role(
            getattr(item, "camera_id", ""),
            default=DEFAULT_CAMERA_ROLE,
        )
        if item_role != current_role:
            continue
        found = True
        enabled = bool(getattr(item, "enabled", True))
        if not enabled:
            break
    return True if not found else enabled


def _overlay_style_for_tool_label(tool_page, img_path: str, label: str) -> tuple[QtGui.QColor, float, bool]:
    if not _roi_label_enabled_for_current_role(tool_page, label):
        return QtGui.QColor(ROI_DISABLED_COLOR), 1.5, True
    status = tool_page._roi_status_for_path(img_path, label)
    color, width, dash = overlay_style_for_label(label, status=status)
    selected_label = _selected_tool_roi_label(tool_page)
    if label and label == selected_label:
        return QtGui.QColor(_SELECTED_TOOL_COLOR), max(float(width), 3.0), False
    return color, width, dash


def _roi_overlay_color(tool_page, img_path: str, label: str) -> QtGui.QColor:
    color, _width, _dash = _overlay_style_for_tool_label(tool_page, img_path, label)
    return color


def _canvas_roi_style_for_label(tool_page, img_path: str, label_name: str) -> tuple[QtGui.QColor, bool, float]:
    label = str(label_name or "").strip()
    color, width, dash = _overlay_style_for_tool_label(tool_page, img_path, label)
    return color, dash, width


def _load_canvas_image(tool_page, path: str, pixmap: Optional[QtGui.QPixmap] = None) -> None:
    tool_page.canvas.set_image(path, pixmap=pixmap if pixmap is not None else QtGui.QPixmap(path))
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
    tool_page.btn_save.setText(tr("debug.save_annotation", label=label))


def _shape_geometry(shape: object) -> tuple[Optional[List[Tuple[float, float]]], Optional[Tuple[int, int, int, int]]]:
    if not isinstance(shape, dict):
        return None, None
    try:
        points = [(float(x), float(y)) for x, y in list(shape.get("points") or [])]
    except (TypeError, ValueError):
        return None, None
    if shape.get("shape_type") == "polygon" and len(points) >= 3:
        return points, None
    if not points:
        return None, None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = int(round(x_max - x_min))
    height = int(round(y_max - y_min))
    if width <= 0 or height <= 0:
        return None, None
    return None, (int(round(x_min)), int(round(y_min)), width, height)


def _shape_index(labelme_path: str) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for shape in labelme_io.list_shapes_from_labelme(labelme_path):
        label = str(shape.get("label", "") or "").strip()
        if label and label not in result:
            result[label] = shape
    return result


def _roi_label_sort_key(name: str) -> tuple[int, int | str]:
    suffix = name[3:] if name.lower().startswith("roi") else ""
    if suffix.isdigit():
        return 0, int(suffix)
    if name.lower() == "roi":
        return 0, 0
    return 1, name


def _set_overlay_shapes(
    tool_page,
    img_path: str,
    current_label: str,
    *,
    shapes_by_label: Optional[Dict[str, dict]] = None,
) -> None:
    j = labelme_io.labelme_json_of_image(img_path)
    overlays: List[OverlayShape] = []
    visible_roi_labels: Optional[set[str]] = None

    current_role = tool_page.current_camera_role()
    method_getter = getattr(tool_page, "loc_method_for_role", None)
    loc_method = method_getter(current_role) if callable(method_getter) else tool_page.loc_method
    recipe = tool_page.shape_recipe_for_role(current_role)
    if loc_method == "shape":
        labels = [str(label).strip() for label in output_labels_from_shape_recipe(recipe) if str(label).strip()]
        visible_roi_labels = set(labels) if labels else None

    if loc_method == "shape" and recipe is not None and recipe.search_points:
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
        overlays.extend(measurement_overlays_for_path(tool_page, img_path))
        tool_page.canvas.set_overlays(overlays)
        return

    if shapes_by_label is None:
        shapes_by_label = _shape_index(j)

    def add_shape(label: str, color: QtGui.QColor, *, width: float, dash: bool = False) -> None:
        poly_pts, xywh = _shape_geometry(shapes_by_label.get(label))
        if poly_pts and len(poly_pts) >= 3:
            overlays.append(OverlayShape(shape_type="polygon", points=poly_pts, color=color, width=width, dash=dash))
            return
        if xywh:
            overlays.append(OverlayShape(shape_type="rect", xywh=xywh, color=color, width=width, dash=dash))

    seen_labels: set[str] = set()
    roi_labels = sorted(
        (label for label in shapes_by_label if label.startswith("roi")),
        key=_roi_label_sort_key,
    )
    for label in roi_labels:
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

    overlays.extend(measurement_overlays_for_path(tool_page, img_path))
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
    shapes_by_label: Dict[str, dict] = {}
    if os.path.exists(j):
        shapes_by_label = _shape_index(j)
        poly_pts, xywh = _shape_geometry(shapes_by_label.get(label_name))
        if poly_pts and len(poly_pts) >= 3:
            tool_page.canvas.set_roi_polygon(poly_pts)
            tool_page.cmb_shape.setCurrentText("polygon")
            loaded = True
        elif xywh:
            tool_page.canvas.set_roi_rect(xywh)
            tool_page.cmb_shape.setCurrentText("rect")
            loaded = True
    tool_page._set_overlay_shapes(img_path, label_name, shapes_by_label=shapes_by_label)
    if not loaded:
        tool_page._on_shapes_changed()


def _on_shapes_changed(tool_page) -> None:
    p = tool_page.canvas.image_path()
    if p is None:
        tool_page.btn_save.setEnabled(False)
        return
    st = tool_page.canvas.roi
    ok = (st.shape_type == "rect" and st.xywh is not None) or (st.shape_type == "polygon" and st.points is not None)
    has_permission = getattr(tool_page.window(), "_has_permission", None)
    if callable(has_permission):
        ok = ok and bool(has_permission("template.edit_roi"))
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



