from __future__ import annotations

import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from ui.debug import RoiCanvas
from ui.i18n import tr
from ui.shape_template.localization_text import group_reason


class LocalizationGroupsTabMixin:
    """Named reference anchors, independent of the inspection regions."""

    def _build_groups_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout(self.tab_groups)
        left_scroll, left = self._make_left_scroll_column()
        layout.addWidget(left_scroll)
        self.lbl_groups_intro = QtWidgets.QLabel(tr("template.groups.intro"))
        self.lbl_groups_intro.setWordWrap(True)
        self.list_localization_groups = QtWidgets.QListWidget()
        self.list_localization_groups.setMinimumHeight(130)
        self.list_localization_groups.currentRowChanged.connect(self._select_localization_group)
        left.addWidget(self.list_localization_groups)
        row = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton(tr("template.groups.add"))
        self.btn_add_group.clicked.connect(self._add_localization_group)
        self.btn_delete_group = QtWidgets.QPushButton(tr("template.groups.delete"))
        self.btn_delete_group.clicked.connect(self._delete_localization_group)
        row.addWidget(self.btn_add_group)
        row.addWidget(self.btn_delete_group)
        left.addLayout(row)
        form = QtWidgets.QFormLayout()
        self.edit_group_name = QtWidgets.QLineEdit()
        self.edit_group_name.setPlaceholderText(tr("template.groups.name_placeholder"))
        self.edit_group_name.editingFinished.connect(self._apply_group_fields)
        self.spin_group_radius = QtWidgets.QSpinBox()
        self.spin_group_radius.setRange(1, 512)
        self.spin_group_radius.setSuffix(" px")
        self.spin_group_radius.setValue(64)
        self.spin_group_radius.setToolTip(tr("template.groups.radius_tip"))
        self.spin_group_score = QtWidgets.QDoubleSpinBox()
        self.spin_group_score.setRange(0.01, 1.0)
        self.spin_group_score.setDecimals(2)
        self.spin_group_score.setSingleStep(0.05)
        self.spin_group_score.setValue(0.65)
        self.spin_group_margin = QtWidgets.QDoubleSpinBox()
        self.spin_group_margin.setRange(0.0, 1.0)
        self.spin_group_margin.setDecimals(3)
        self.spin_group_margin.setSingleStep(0.01)
        self.spin_group_margin.setValue(0.03)
        self.spin_group_margin.setToolTip(tr("template.groups.margin_tip"))
        for spin in (self.spin_group_radius, self.spin_group_score, self.spin_group_margin):
            spin.valueChanged.connect(self._apply_group_fields)
        self._group_form_labels = []
        for key, widget in (("name", self.edit_group_name), ("radius", self.spin_group_radius),
                            ("min_score", self.spin_group_score), ("min_margin", self.spin_group_margin)):
            label = QtWidgets.QLabel(tr("template.groups." + key))
            self._group_form_labels.append((label, "template.groups." + key))
            form.addRow(label, widget)
        left.addLayout(form)
        self.btn_redraw_group = QtWidgets.QPushButton(tr("template.groups.redraw"))
        self.btn_redraw_group.clicked.connect(self._redraw_group_anchor)
        left.addWidget(self.btn_redraw_group)
        self.btn_save_groups = QtWidgets.QPushButton(tr("template.groups.save"))
        self.btn_save_groups.clicked.connect(self._save_reference_roi_config)
        left.addWidget(self.btn_save_groups)
        self.lbl_group_status = QtWidgets.QLabel()
        self._set_group_status("template.groups.empty")
        self.lbl_group_status.setWordWrap(True)
        left.addWidget(self.lbl_group_status)
        left.addStretch(1)
        left.addWidget(self.lbl_groups_intro)
        self.group_canvas = RoiCanvas()
        self.group_canvas.setMinimumSize(640, 480)
        self.group_canvas.draw_shape = "rect"
        self.group_canvas.set_roi_style(
            roi_color=QtGui.QColor(30, 230, 160), roi_width=3.0,
            preview_color=QtGui.QColor(30, 230, 160), preview_dash=False, preview_width=2.0,
        )
        self.group_canvas.shapesChanged.connect(self._on_group_anchor_changed)
        layout.addWidget(self.group_canvas, 1)
        self._refresh_group_fields()

    def _set_group_status(self, key: str, **kwargs) -> str:
        self._group_status_message = (key, dict(kwargs))
        text = tr(key, **kwargs)
        self.lbl_group_status.setText(text)
        return text

    def _group_result_headers(self) -> list[str]:
        return [tr("template.groups." + key) for key in
                ("result_group", "result_status", "result_score", "result_dx", "result_dy", "result_reason")]

    def _retranslate_groups_ui(self) -> None:
        for widget, key in ((self.lbl_groups_intro, "intro"), (self.btn_add_group, "add"),
                            (self.btn_delete_group, "delete"), (self.btn_redraw_group, "redraw"),
                            (self.btn_save_groups, "save")):
            widget.setText(tr("template.groups." + key))
        for label, key in self._group_form_labels:
            label.setText(tr(key))
        self.edit_group_name.setPlaceholderText(tr("template.groups.name_placeholder"))
        self.spin_group_radius.setToolTip(tr("template.groups.radius_tip"))
        self.spin_group_margin.setToolTip(tr("template.groups.margin_tip"))
        # Relabel views without committing/discarding an in-progress user name.
        pending_name = self.edit_group_name.text()
        cursor = self.edit_group_name.cursorPosition()
        selection_start = self.edit_group_name.selectionStart()
        selection_length = len(self.edit_group_name.selectedText())
        modified = self.edit_group_name.isModified()
        self._refresh_reference_region_list()
        with QtCore.QSignalBlocker(self.edit_group_name):
            self.edit_group_name.setText(pending_name)
            self.edit_group_name.setModified(modified)
            self.edit_group_name.setCursorPosition(cursor)
            if selection_start >= 0:
                self.edit_group_name.setSelection(selection_start, selection_length)
        key, kwargs = getattr(self, "_group_status_message", ("template.groups.empty", {}))
        self._set_group_status(key, **kwargs)
        if hasattr(self, "table_group_results"):
            self.table_group_results.setHorizontalHeaderLabels(self._group_result_headers())
            self._show_group_results(getattr(self, "_last_group_results", []))

    def _current_localization_group(self):
        row = self.list_localization_groups.currentRow()
        return self._localization_groups[row] if 0 <= row < len(self._localization_groups) else None

    def _group_name(self, group_id: str) -> str:
        if not group_id or group_id == "global":
            return tr("template.groups.global")
        for group in self._localization_groups:
            if group["id"] == group_id:
                return str(group["name"])
        return tr("template.groups.missing", id=group_id)

    def _refresh_group_choices(self) -> None:
        if not hasattr(self, "cmb_region_group"):
            return
        selected = ""
        idx = self._selected_reference_idx
        if idx is not None and 0 <= idx < len(self._reference_regions):
            selected = str(self._reference_regions[idx].get("group_id") or "")
        if selected == "global":
            selected = ""
        with QtCore.QSignalBlocker(self.cmb_region_group):
            self.cmb_region_group.clear()
            self.cmb_region_group.addItem(tr("template.groups.global"), "")
            for group in self._localization_groups:
                suffix = tr("template.groups.rebuild_suffix") if group.get("needs_rebuild") else ""
                self.cmb_region_group.addItem(str(group["name"]) + suffix, group["id"])
            index = self.cmb_region_group.findData(selected)
            if index < 0:
                self.cmb_region_group.addItem(self._group_name(selected), selected)
                index = self.cmb_region_group.count() - 1
            self.cmb_region_group.setCurrentIndex(index)
        self.cmb_region_group.setEnabled(idx is not None and 0 <= idx < len(self._reference_regions))

    def _assign_region_group(self, _index: int = -1) -> None:
        idx = self._selected_reference_idx
        if idx is None or not 0 <= idx < len(self._reference_regions):
            return
        group_id = str(self.cmb_region_group.currentData() or "")
        self._reference_regions[idx]["group_id"] = group_id
        self._refresh_reference_region_list()
        self._refresh_group_list()
        self._set_reference_dirty(True, tr("template.groups.assigned", name=self._group_name(group_id)))

    def _refresh_group_list(self, select_id=None) -> None:
        current = self._current_localization_group()
        selected_id = select_id if select_id is not None else (current or {}).get("id")
        with QtCore.QSignalBlocker(self.list_localization_groups):
            self.list_localization_groups.clear()
            selected_row = -1
            for row, group in enumerate(self._localization_groups):
                count = sum(r.get("group_id") == group["id"] for r in self._reference_regions)
                state = tr("template.groups." + ("state_rebuild" if group.get("needs_rebuild") else
                           ("state_ready" if group.get("points") else "state_draft")))
                item = QtWidgets.QListWidgetItem(tr("template.groups.list_item", name=group["name"], state=state, count=count))
                item.setData(QtCore.Qt.UserRole, group["id"])
                self.list_localization_groups.addItem(item)
                if group["id"] == selected_id:
                    selected_row = row
            self.list_localization_groups.setCurrentRow(selected_row)
        self._refresh_group_choices()
        self._refresh_group_fields()

    def _refresh_group_fields(self) -> None:
        group = self._current_localization_group()
        widgets = (self.edit_group_name, self.spin_group_radius, self.spin_group_score, self.spin_group_margin)
        for widget in widgets:
            widget.setEnabled(group is not None)
        self.btn_redraw_group.setEnabled(group is not None)
        self.btn_delete_group.setEnabled(group is not None)
        for widget, value in zip(widgets, (
            (group or {}).get("name", ""), (group or {}).get("search_radius", 64),
            (group or {}).get("min_score", .65), (group or {}).get("min_peak_margin", .03),
        )):
            with QtCore.QSignalBlocker(widget):
                if isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                else:
                    widget.setValue(float(value))
        self._refresh_group_canvas()

    def _select_localization_group(self, _row: int) -> None:
        self._refresh_group_fields()
        group = self._current_localization_group()
        if group:
            self._set_group_status("template.groups.editing", name=group["name"])

    def _add_localization_group(self) -> None:
        if self.image_bgr is None:
            self._set_group_status("template.groups.open_reference")
            return
        group_id = uuid.uuid4().hex[:12]
        names = {g["name"] for g in self._localization_groups}
        number = 1
        while tr("template.groups.default_name", number=number) in names:
            number += 1
        self._localization_groups.append(dict(
            id=group_id, name=tr("template.groups.default_name", number=number), points=[], search_radius=64,
            min_score=.65, min_peak_margin=.03, needs_rebuild=False,
        ))
        self._refresh_group_list(group_id)
        self.edit_group_name.setFocus()
        self.edit_group_name.selectAll()
        self._set_group_status("template.groups.new_hint")
        self._set_reference_dirty(True, tr("template.groups.created"))

    def _apply_group_fields(self, *_args) -> None:
        group = self._current_localization_group()
        if group is None:
            return
        name = self.edit_group_name.text().strip() or str(group["name"])
        if any(g["id"] != group["id"] and g["name"] == name for g in self._localization_groups):
            self._set_group_status("template.groups.duplicate_name")
            self.edit_group_name.setText(str(group["name"]))
            return
        changes = dict(name=name, search_radius=self.spin_group_radius.value(),
                       min_score=self.spin_group_score.value(), min_peak_margin=self.spin_group_margin.value())
        if all(group.get(key) == value for key, value in changes.items()):
            return
        group.update(changes)
        self._refresh_group_list(group["id"])
        self._refresh_reference_region_list()
        self._set_reference_dirty(True, self._set_group_status("template.groups.updated"))

    def _delete_localization_group(self) -> None:
        group = self._current_localization_group()
        if group is None:
            return
        bound_regions = [r for r in self._reference_regions if r.get("group_id") == group["id"]]
        bound = [str(r.get("output_label") or r.get("reference_label") or tr("template.groups.unnamed_roi")) for r in bound_regions]
        if bound:
            if not self._confirm_remove_bound_group(str(group["name"]), bound):
                self._set_group_status("template.groups.delete_cancelled", labels=", ".join(bound))
                return
            for region in bound_regions:
                region["group_id"] = ""
        self._localization_groups.remove(group)
        self._refresh_group_list("")
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        status = (self._set_group_status("template.groups.deleted_reassigned", name=group["name"], labels=", ".join(bound))
                  if bound else self._set_group_status("template.groups.deleted"))
        self._set_reference_dirty(True, status)

    def _confirm_remove_bound_group(self, name: str, bound: list[str]) -> bool:
        box = QtWidgets.QMessageBox(self)
        box.setObjectName("deleteLocalizationGroupDialog")
        box.setWindowTitle(tr("template.groups.delete"))
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        box.setText(tr("template.groups.delete_bound_text", name=name, count=len(bound), labels=", ".join(bound)))
        box.setInformativeText(tr("template.groups.delete_bound_info"))
        delete_button = box.addButton(tr("template.groups.delete_confirm"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        cancel_button.setText(tr("template.groups.cancel"))
        box.setDefaultButton(cancel_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        return box.clickedButton() is delete_button

    def _redraw_group_anchor(self) -> None:
        group = self._current_localization_group()
        if group is None:
            return
        group["points"] = []
        group["needs_rebuild"] = True
        self._refresh_group_list(group["id"])
        self._set_group_status("template.groups.redraw_hint")
        self._set_reference_dirty(True, tr("template.groups.redraw_pending"))

    def _on_group_anchor_changed(self) -> None:
        if self._syncing_group_canvas:
            return
        group = self._current_localization_group()
        rect = self.group_canvas.roi.xywh
        if group is None or rect is None:
            return
        x, y, w, h = rect
        if w < 8 or h < 8:
            group["needs_rebuild"] = True
            self._set_group_status("template.groups.anchor_small")
            self._set_reference_dirty(True)
            return
        group["points"] = [[float(x), float(y)], [float(x + w), float(y + h)]]
        group["needs_rebuild"] = False
        self._refresh_group_list(group["id"])
        self._set_group_status("template.groups.anchor_drawn", name=group["name"], width=w, height=h)
        self._set_reference_dirty(True, tr("template.groups.anchor_updated"))

    def _refresh_group_canvas(self) -> None:
        if not hasattr(self, "group_canvas"):
            return
        group = self._current_localization_group()
        overlays = [self._region_overlay_shape(r, QtGui.QColor(165, 90, 170), 1., True)
                    for r in self._reference_regions]
        overlays += [self._region_overlay_shape(g, QtGui.QColor(30, 170, 220), 2., True)
                     for g in self._localization_groups if g is not group and len(g.get("points", [])) >= 2]
        self.group_canvas.set_overlays(overlays)
        self._syncing_group_canvas = True
        try:
            points = (group or {}).get("points", [])
            if len(points) >= 2:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                self.group_canvas.set_roi_rect((round(min(xs)), round(min(ys)), round(max(xs)-min(xs)), round(max(ys)-min(ys))))
            else:
                self.group_canvas.clear_roi()
        finally:
            self._syncing_group_canvas = False

    def _show_group_results(self, results) -> None:
        self._last_group_results = [dict(result) for result in results or []]
        self.table_group_results.setRowCount(0)
        for result in self._last_group_results:
            row = self.table_group_results.rowCount()
            self.table_group_results.insertRow(row)
            dx, dy = result.get("delta_scene") or [0., 0.]
            score = result.get("score")
            valid = bool(result.get("valid"))
            used = result.get("used", True)
            state = tr("template.groups." + (("success" if valid else "failure") if used else "unused"))
            reason = group_reason(result)
            name = (tr("template.groups.configuration_name") if result.get("id") == "__configuration__"
                    else result.get("name") or result.get("id", ""))
            values = [name, state,
                      "—" if score is None else f"{score:.3f}", f"{dx:+.2f}", f"{dy:+.2f}", reason]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setToolTip(reason or (tr("template.groups.success_tip") if valid else state))
                if used and not valid:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(255, 110, 100)))
                self.table_group_results.setItem(row, col, item)

    def _clear_group_find_cache(self, *_args) -> None:
        self._find_result_cache.clear()
        if hasattr(self, "table_group_results"):
            self._show_group_results([])
        if hasattr(self, "list_find_images"):
            import os
            for row in range(self.list_find_images.count()):
                item = self.list_find_images.item(row)
                item.setText(os.path.basename(str(item.data(QtCore.Qt.UserRole) or "")))
                item.setToolTip(str(item.data(QtCore.Qt.UserRole) or ""))
                item.setForeground(QtGui.QBrush(QtGui.QColor(205, 205, 205)))
            current = self.list_find_images.currentItem()
            if current is not None:
                path = str(current.data(QtCore.Qt.UserRole) or "")
                if os.path.exists(path):
                    from ui.debug import pixmap_from_path
                    self.find_canvas.set_image(path, pixmap=pixmap_from_path(path))
                    self._apply_search_roi_to_find_canvas()
            self.lbl_find_status.setText(tr("template.groups.rerun"))

