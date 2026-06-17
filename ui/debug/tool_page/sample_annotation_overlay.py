"""Canvas overlay interaction helpers for the sample annotation dialog."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from common import labelme_io
from ui.debug import OverlayShape
from ui.debug.tool_page.sample_annotation_canvas import _pixmap_from_path
from ui.i18n import tr
from ui.roi_overlay_colors import overlay_style_for_label

def _load_canvas_preview(self, path: str, camera_role: str) -> None:
    pixmap = _pixmap_from_path(path)
    if pixmap.isNull():
        self.preview_canvas.clear_image()
        return
    self.preview_canvas.set_image(pixmap)
    self._refresh_canvas_overlays(path, camera_role)

def _refresh_canvas_overlays(self, path: str, camera_role: str) -> None:
    jpath = labelme_io.labelme_json_of_image(path)
    labels = self._tool_page._inspection_label_names_for_role(camera_role)
    overlays: list[OverlayShape] = []
    shape_entries: list[dict[str, object]] = []
    if not os.path.exists(jpath):
        self._canvas_shapes = []
        self.preview_canvas.set_overlays([])
        return
    for label in labels:
        poly_points = labelme_io.try_read_polygon_points_from_labelme(jpath, label)
        xywh = labelme_io.try_read_xywh_from_labelme(jpath, label)
        if poly_points and len(poly_points) >= 3:
            status = self._tool_page._sample_roi_status_for_path(path, camera_role, label).lower()
            color, width, dash = overlay_style_for_label(label, status=status)
            if label == self._active_roi_label:
                width = max(float(width), 4.0)
            overlays.append(
                OverlayShape(
                    shape_type="polygon",
                    points=[(float(x), float(y)) for x, y in poly_points],
                    color=QtGui.QColor(color),
                    width=float(width),
                    dash=bool(dash),
                )
            )
            shape_entries.append(
                {
                    "label": label,
                    "shape_type": "polygon",
                    "points": [(float(x), float(y)) for x, y in poly_points],
                }
            )
            continue
        if xywh:
            status = self._tool_page._sample_roi_status_for_path(path, camera_role, label).lower()
            color, width, dash = overlay_style_for_label(label, status=status)
            if label == self._active_roi_label:
                width = max(float(width), 4.0)
            overlays.append(
                OverlayShape(
                    shape_type="rect",
                    xywh=tuple(int(v) for v in xywh),
                    color=QtGui.QColor(color),
                    width=float(width),
                    dash=bool(dash),
                )
            )
            shape_entries.append(
                {
                    "label": label,
                    "shape_type": "rect",
                    "xywh": tuple(int(v) for v in xywh),
                }
            )
    self._canvas_shapes = shape_entries
    self.preview_canvas.set_overlays(overlays)

def _on_canvas_image_pressed(self, button: int, image_x: int, image_y: int) -> None:
    button_value = int(getattr(QtCore.Qt.MouseButton.LeftButton, "value", QtCore.Qt.MouseButton.LeftButton))
    right_value = int(getattr(QtCore.Qt.MouseButton.RightButton, "value", QtCore.Qt.MouseButton.RightButton))
    if button not in {button_value, right_value}:
        return
    path, camera_role = self._current_path_and_role()
    if not path:
        return
    label = self._find_roi_label_at_point(float(image_x), float(image_y))
    if not label:
        return
    self._active_roi_label = label
    self._refresh_canvas_overlays(path, camera_role)
    self._focus_roi_row(label)
    self._show_roi_label_menu(path, camera_role, label)

def _focus_roi_row(self, label: str) -> None:
    for row in range(self.roi_table.rowCount()):
        item = self.roi_table.item(row, 0)
        if item is None:
            continue
        if str(item.text()).strip() != str(label).strip():
            continue
        self.roi_table.setCurrentCell(row, 0)
        self.roi_table.scrollToItem(item)
        break

def _show_roi_label_menu(self, path: str, camera_role: str, label: str) -> None:
    menu = QtWidgets.QMenu(self)
    action_ok = menu.addAction(f"{label} -> OK")
    action_ng = menu.addAction(f"{label} -> NG")
    action_clear = menu.addAction(tr("sample.clear_label", label=label))
    chosen = menu.exec(QtGui.QCursor.pos())
    if chosen is None:
        return
    if chosen == action_ok:
        status = "OK"
    elif chosen == action_ng:
        status = "NG"
    else:
        status = ""
    self._set_roi_status_from_canvas(path, camera_role, label, status)

def _set_roi_status_from_canvas(self, path: str, camera_role: str, label: str, status: str) -> None:
    self._tool_page._set_sample_roi_status_for_path(path, camera_role, label, status)
    self._refresh_after_annotation_change(path, camera_role)

def _find_roi_label_at_point(self, image_x: float, image_y: float) -> str:
    for entry in reversed(self._canvas_shapes):
        label = str(entry.get("label", "") or "").strip()
        if not label:
            continue
        shape_type = str(entry.get("shape_type", "") or "")
        if shape_type == "polygon":
            points = entry.get("points") or []
            polygon = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in points])
            if len(polygon) >= 3 and polygon.containsPoint(
                QtCore.QPointF(float(image_x), float(image_y)),
                QtCore.Qt.FillRule.OddEvenFill,
            ):
                return label
            continue
        xywh = entry.get("xywh")
        if not xywh:
            continue
        x, y, w, h = [float(v) for v in xywh]
        if x <= float(image_x) <= x + w and y <= float(image_y) <= y + h:
            return label
    return ""

