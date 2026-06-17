from __future__ import annotations

import os
from typing import Dict, List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from common import labelme_io
from shape.core.recipe import ShapeRecipe, load_recipe
from ui.debug import OverlayShape
from ui.i18n import tr
from ui.shape_template.template_page_utils import _shape_to_rect


class ReferenceRoiTabMixin:
    def _next_reference_label(self) -> str:
        used = {
            str(region.get("reference_label") or region.get("output_label") or "")
            for region in self._reference_regions
            if isinstance(region, dict)
        }
        idx = 1
        while f"roi{idx}" in used:
            idx += 1
        return f"roi{idx}"

    def _region_points_from_canvas(self) -> Tuple[str, List[List[float]]]:
        if self.ref_canvas.roi.shape_type == "polygon" and self.ref_canvas.roi.points:
            return "polygon", [[float(x), float(y)] for x, y in self.ref_canvas.roi.points]
        xywh = self.ref_canvas.roi_xywh()
        if xywh is None:
            return "", []
        x, y, w, h = xywh
        return "rectangle", [[float(x), float(y)], [float(x + w), float(y + h)]]

    def _region_overlay_shape(self, region: Dict[str, object], color: QtGui.QColor, width: int, dash: bool) -> OverlayShape:
        shape_type = str(region.get("shape_type", "rectangle"))
        points = [
            (float(pt[0]), float(pt[1]))
            for pt in region.get("points", []) or []
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if shape_type == "polygon" and len(points) >= 3:
            return OverlayShape(shape_type="polygon", points=points, color=color, width=width, dash=dash)
        if len(points) >= 2:
            x0, y0 = float(points[0][0]), float(points[0][1])
            x1, y1 = float(points[1][0]), float(points[1][1])
            x = int(round(min(x0, x1)))
            y = int(round(min(y0, y1)))
            w = max(1, int(round(abs(x1 - x0))))
            h = max(1, int(round(abs(y1 - y0))))
            return OverlayShape(shape_type="rect", xywh=(x, y, w, h), color=color, width=width, dash=dash)
        return OverlayShape(shape_type="rect", xywh=(0, 0, 1, 1), color=color, width=width, dash=dash)

    def _set_reference_dirty(self, dirty: bool, status_text: str = "") -> None:
        self._reference_dirty = bool(dirty)
        if hasattr(self, "btn_save_reference_roi"):
            self.btn_save_reference_roi.setEnabled(self._reference_dirty or bool(self._reference_regions))
        if status_text and hasattr(self, "lbl_reference_status"):
            self.lbl_reference_status.setText(status_text)

    def _reference_dirty_status(self) -> str:
        return tr("template.status_reference_dirty", count=len(self._reference_regions))

    def _refresh_reference_region_list(self) -> None:
        if not hasattr(self, "table_reference_regions"):
            return
        self.table_reference_regions.blockSignals(True)
        self.table_reference_regions.setRowCount(0)
        for idx, region in enumerate(self._reference_regions):
            label = str(region.get("output_label") or region.get("reference_label") or f"roi{idx + 1}")
            display_name = str(region.get("display_name") or region.get("name") or label).strip() or label
            shape_type = str(region.get("shape_type", "rectangle"))
            points = [
                [float(pt[0]), float(pt[1])]
                for pt in region.get("points", []) or []
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if shape_type == "polygon":
                info_text = f"Polygon {len(points)} pts"
            elif len(points) >= 2:
                x0, y0 = float(points[0][0]), float(points[0][1])
                x1, y1 = float(points[1][0]), float(points[1][1])
                w = max(1, int(round(abs(x1 - x0))))
                h = max(1, int(round(abs(y1 - y0))))
                info_text = f"Rect {w}x{h}"
            else:
                info_text = "Rect"
            row = self.table_reference_regions.rowCount()
            self.table_reference_regions.insertRow(row)
            values = [
                str(idx),
                display_name,
                label,
                info_text,
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, idx)
                if col == 0:
                    item.setTextAlignment(
                        int(
                            QtCore.Qt.AlignmentFlag.AlignHCenter
                            | QtCore.Qt.AlignmentFlag.AlignVCenter
                        )
                    )
                self.table_reference_regions.setItem(row, col, item)
        if self._selected_reference_idx is not None and 0 <= self._selected_reference_idx < self.table_reference_regions.rowCount():
            self.table_reference_regions.setCurrentCell(self._selected_reference_idx, 0)
        self.table_reference_regions.blockSignals(False)

    def _refresh_reference_region_fields(self) -> None:
        self.edit_output_label.blockSignals(True)
        self.edit_display_name.blockSignals(True)
        try:
            has_selection = (
                self._selected_reference_idx is not None
                and 0 <= self._selected_reference_idx < len(self._reference_regions)
            )
            self.edit_output_label.setEnabled(has_selection)
            self.edit_display_name.setEnabled(has_selection)
            self.btn_apply_region_name.setEnabled(has_selection)
            if not has_selection:
                self.edit_output_label.setText("")
                self.edit_display_name.setText("")
                return
            region = self._reference_regions[self._selected_reference_idx]
            label = str(region.get("output_label") or region.get("reference_label") or "").strip()
            display_name = str(region.get("display_name") or region.get("name") or label).strip() or label
            self.edit_output_label.setText(label)
            self.edit_display_name.setText(display_name)
        finally:
            self.edit_output_label.blockSignals(False)
            self.edit_display_name.blockSignals(False)

    def _on_region_field_edited(self) -> None:
        pass

    def _apply_reference_region_fields(self) -> None:
        if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
            return
        region = self._reference_regions[self._selected_reference_idx]
        label = self.edit_output_label.text().strip()
        if not label:
            label = str(region.get("output_label") or region.get("reference_label") or "").strip()
        if not label:
            label = self._next_reference_label()
        display_name = self.edit_display_name.text().strip() or label
        region["reference_label"] = label
        region["output_label"] = label
        region["display_name"] = display_name
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._set_reference_dirty(True, tr("template.status_reference_name_updated_unsaved", name=display_name))

    def _refresh_reference_canvas(self) -> None:
        overlays: List[OverlayShape] = []
        inactive_color = QtGui.QColor(255, 0, 255)
        for idx, region in enumerate(self._reference_regions):
            if idx == self._selected_reference_idx:
                continue
            overlays.append(self._region_overlay_shape(region, inactive_color, 1.8, False))
        self.ref_canvas.set_overlays(overlays)
        self._syncing_reference_view = True
        try:
            if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
                self.ref_canvas.clear_roi()
            else:
                region = self._reference_regions[self._selected_reference_idx]
                shape_type = str(region.get("shape_type", "rectangle"))
                points = [
                    (float(pt[0]), float(pt[1]))
                    for pt in region.get("points", []) or []
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]
                self.cmb_reference_shape.setCurrentText("polygon" if shape_type == "polygon" else "rectangle")
                if shape_type == "polygon" and len(points) >= 3:
                    self.ref_canvas.set_roi_polygon(points)
                elif len(points) >= 2:
                    x0, y0 = float(points[0][0]), float(points[0][1])
                    x1, y1 = float(points[1][0]), float(points[1][1])
                    self.ref_canvas.set_roi_rect(
                        (
                            int(round(min(x0, x1))),
                            int(round(min(y0, y1))),
                            max(1, int(round(abs(x1 - x0)))),
                            max(1, int(round(abs(y1 - y0)))),
                        )
                    )
                else:
                    self.ref_canvas.clear_roi()
        finally:
            self._syncing_reference_view = False

    def _on_reference_region_selected(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if current_row < 0 or current_row >= len(self._reference_regions):
            self._selected_reference_idx = None
        else:
            self._selected_reference_idx = current_row
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()

    def _prepare_new_reference_roi(self) -> None:
        self._selected_reference_idx = None
        if hasattr(self, "table_reference_regions"):
            self.table_reference_regions.blockSignals(True)
            self.table_reference_regions.clearSelection()
            self.table_reference_regions.setCurrentIndex(QtCore.QModelIndex())
            self.table_reference_regions.blockSignals(False)
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self.lbl_reference_status.setText(tr("template.status_reference_add_mode"))

    def _remove_selected_reference_roi(self) -> None:
        if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
            return
        del self._reference_regions[self._selected_reference_idx]
        self._selected_reference_idx = None
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self._set_reference_dirty(True, tr("template.status_reference_deleted_unsaved", count=len(self._reference_regions)))

    def _on_reference_canvas_shape_changed(self) -> None:
        if self._syncing_reference_view:
            return
        shape_type, points = self._region_points_from_canvas()
        if not shape_type or not points:
            return
        if self._selected_reference_idx is None:
            label = self._next_reference_label()
            self._reference_regions_explicit = True
            self._reference_regions.append(
                {
                    "reference_label": label,
                    "output_label": label,
                    "display_name": label,
                    "shape_type": shape_type,
                    "points": points,
                }
            )
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            self._set_reference_dirty(True, tr("template.status_reference_added_unsaved", label=label, count=len(self._reference_regions)))
            return
        if 0 <= self._selected_reference_idx < len(self._reference_regions):
            self._reference_regions_explicit = True
            region = self._reference_regions[self._selected_reference_idx]
            region["shape_type"] = shape_type
            region["points"] = points
            label = str(region.get("output_label") or region.get("reference_label") or f"roi{self._selected_reference_idx + 1}")
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._set_reference_dirty(True, tr("template.status_reference_updated_unsaved", label=label))

    def _on_reference_shape_changed(self, shape_name: str) -> None:
        self.ref_canvas.draw_shape = "polygon" if shape_name == "polygon" else "rect"

    def _load_reference_roi_from_json(self, *, silent: bool) -> bool:
        if not self.image_path or not os.path.exists(self.image_path):
            if not silent:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.load_reference_first"))
            return False
        jpath = labelme_io.labelme_json_of_image(self.image_path)
        if not os.path.exists(jpath):
            self._reference_regions = []
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            if not silent:
                QtWidgets.QMessageBox.information(self, tr("common.info"), tr("template.no_reference_json"))
            return False
        try:
            regions: List[Dict[str, object]] = []
            for shape in labelme_io.list_shapes_from_labelme(jpath, label_prefix="roi"):
                label_name = str(shape.get("label", "")).strip()
                if not label_name:
                    continue
                shape_type = str(shape.get("shape_type", "rectangle"))
                if shape_type == "polygon":
                    points = [[float(x), float(y)] for x, y in shape.get("points", [])]
                    if len(points) < 3:
                        continue
                else:
                    xywh = _shape_to_rect(shape)
                    if xywh is None:
                        continue
                    x, y, w, h = xywh
                    points = [[float(x), float(y)], [float(x + w), float(y + h)]]
                    shape_type = "rectangle"
                regions.append(
                    {
                        "reference_label": label_name,
                        "output_label": label_name,
                        "display_name": label_name,
                        "shape_type": shape_type,
                        "points": points,
                    }
                )
            if not regions:
                raise RuntimeError(tr("template.no_reference_roi_in_json"))
            self._reference_regions = regions
            self._reference_regions_explicit = True
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            status = tr("template.status_reference_loaded_unsaved", count=len(regions))
            self._set_reference_dirty(True, status)
            return True
        except Exception as exc:
            self._reference_regions = []
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            if not silent:
                QtWidgets.QMessageBox.warning(self, tr("common.load_failed"), str(exc))
            return False

    def _save_reference_roi_to_json(self) -> None:
        if not self.image_path or not os.path.exists(self.image_path):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.load_reference_first"))
            return
        try:
            if not self._reference_regions:
                raise RuntimeError(tr("template.no_reference_roi_to_save"))
            old_recipe = load_recipe(self.paths.recipe_path) if os.path.exists(self.paths.recipe_path) else ShapeRecipe()
            old_labels = {
                str(region.get("output_label") or region.get("reference_label") or "")
                for region in (old_recipe.reference_regions or [])
                if isinstance(region, dict)
            }
            new_labels = {
                str(region.get("output_label") or region.get("reference_label") or "")
                for region in self._reference_regions
                if isinstance(region, dict)
            }
            for label_name in old_labels - new_labels:
                if label_name:
                    labelme_io.delete_labelme_shape(self.image_path, label_name=label_name)
            for region in self._reference_regions:
                label_name = str(region.get("output_label") or region.get("reference_label") or "").strip()
                shape_type = str(region.get("shape_type", "rectangle"))
                points = [
                    (float(pt[0]), float(pt[1]))
                    for pt in region.get("points", []) or []
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]
                if not label_name or len(points) < 2:
                    continue
                if shape_type == "polygon" and len(points) >= 3:
                    labelme_io.upsert_labelme_polygon(self.image_path, points, label_name=label_name)
                else:
                    x0, y0 = points[0]
                    x1, y1 = points[1]
                    labelme_io.upsert_labelme_rect(
                        self.image_path,
                        (
                            int(round(min(x0, x1))),
                            int(round(min(y0, y1))),
                            max(1, int(round(abs(x1 - x0)))),
                            max(1, int(round(abs(y1 - y0)))),
                        ),
                        label_name=label_name,
                    )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("common.save_failed"), str(exc))
            return
        self._reference_regions_explicit = True
        self._save_recipe()
        self._set_reference_dirty(False)
        self.referenceRegionsChanged.emit()
        self.lbl_reference_status.setText(tr("template.status_reference_saved", count=len(self._reference_regions)))

    def _save_reference_roi_config(self, _checked: object = None) -> None:
        try:
            self._reference_regions_explicit = True
            self._save_recipe()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("common.save_failed"), str(exc))
            return
        self._set_reference_dirty(False, tr("template.status_reference_config_saved", count=len(self._reference_regions)))
        self.referenceRegionsChanged.emit()

    def _confirm_reference_roi_discard(self) -> bool:
        if not self._reference_dirty:
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            tr("template.unsaved_reference_title"),
            tr("template.unsaved_reference_message", count=len(self._reference_regions)),
            (
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel
            ),
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Save:
            self._save_reference_roi_config()
            return not self._reference_dirty
        if reply == QtWidgets.QMessageBox.StandardButton.Discard:
            return True
        return False

    def _clear_reference_roi(self) -> None:
        self._reference_regions = []
        self._reference_regions_explicit = True
        self._selected_reference_idx = None
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self._recipe_reference_shape_type = ""
        self._recipe_reference_points = []
        self._set_reference_dirty(True, tr("template.status_reference_cleared_unsaved"))


__all__ = ["ReferenceRoiTabMixin"]
