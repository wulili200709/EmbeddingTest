from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr
from ui.debug.tool_page.camera_roles import normalize_camera_role
from ui.debug.tool_page.sample_annotation_canvas import _SampleAnnotationCanvas
from ui.debug.tool_page.sample_auto_roi_dialog import _SampleAnnotationAutoRoiDialog


class _SampleAnnotationPreviewDialog(QtWidgets.QDialog):
    def __init__(self, tool_page: "ToolPage", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent or tool_page)
        self._tool_page = tool_page
        self.setWindowTitle(tr("sample.annotation_title"))
        self.resize(1100, 720)
        self.setModal(False)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QtWidgets.QLabel(tr("sample.product")))
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.addItem(tool_page.current_product_name())
        self.cmb_product.setEnabled(False)
        top_row.addWidget(self.cmb_product, 1)
        top_row.addWidget(QtWidgets.QLabel(tr("sample.camera")))
        self.cmb_camera = QtWidgets.QComboBox()
        self.sync_camera_roles(tool_page.configured_camera_roles())
        top_row.addWidget(self.cmb_camera)
        top_row.addWidget(QtWidgets.QLabel(tr("sample.sample")))
        self.cmb_sample_kind = QtWidgets.QComboBox()
        self.cmb_sample_kind.addItem(tr("debug.train_samples"), "train")
        self.cmb_sample_kind.addItem(tr("debug.test_samples"), "test")
        top_row.addWidget(self.cmb_sample_kind)
        root.addLayout(top_row)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(8)

        left_panel = QtWidgets.QFrame()
        left_panel.setStyleSheet("QFrame{background:#2f2f2f;border:1px solid #505050;}")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        left_layout.addWidget(QtWidgets.QLabel(tr("sample.image_list")))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setUniformItemSizes(True)
        self.sample_list.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
        self.sample_list.setBatchSize(100)
        self.sample_list.setStyleSheet(
            "QListWidget{background:#333333;color:#e0e0e0;border:1px solid #404040;}"
            "QListWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
        )
        left_layout.addWidget(self.sample_list, 1)
        body.addWidget(left_panel, 1)

        center_panel = QtWidgets.QFrame()
        center_panel.setStyleSheet("QFrame{background:#1f1f1f;border:1px solid #505050;}")
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(6)
        center_layout.addWidget(QtWidgets.QLabel(tr("sample.current_image")))
        self.lbl_canvas_hint = QtWidgets.QLabel(tr("sample.canvas_hint"))
        self.lbl_canvas_hint.setStyleSheet("color:#a0a0a0;font-size:12px;")
        center_layout.addWidget(self.lbl_canvas_hint)
        self.preview_canvas = _SampleAnnotationCanvas()
        self.preview_canvas.setStyleSheet("QWidget{background:#111111;border:1px solid #303030;}")
        self.preview_canvas.imagePressed.connect(self._on_canvas_image_pressed)
        center_layout.addWidget(self.preview_canvas, 1)
        self.lbl_image_status = QtWidgets.QLabel(tr("sample.status_none"))
        self.lbl_image_status.setStyleSheet("color:#bcbcbc;font-size:12px;")
        center_layout.addWidget(self.lbl_image_status)
        body.addWidget(center_panel, 2)
        self._canvas_shapes: list[dict[str, object]] = []
        self._active_roi_label = ""
        self._suppress_tool_page_context_sync = False

        right_panel = QtWidgets.QFrame()
        right_panel.setStyleSheet("QFrame{background:#2f2f2f;border:1px solid #505050;}")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)
        right_layout.addWidget(QtWidgets.QLabel(tr("sample.current_roi_labels")))
        self.roi_table = QtWidgets.QTableWidget(0, 3)
        self.roi_table.setHorizontalHeaderLabels(["ROI", tr("sample.geometry"), tr("sample.label")])
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.roi_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.roi_table.setStyleSheet(
            "QTableWidget{background:#333333;color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;}"
            "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:4px;}"
        )
        right_layout.addWidget(self.roi_table, 1)
        self.lbl_dialog_hint = QtWidgets.QLabel(tr("sample.annotation_tip"))
        self.lbl_dialog_hint.setWordWrap(True)
        self.lbl_dialog_hint.setStyleSheet("color:#a0a0a0;font-size:12px;")
        right_layout.addWidget(self.lbl_dialog_hint)
        body.addWidget(right_panel, 1)
        root.addLayout(body, 1)

        footer = QtWidgets.QHBoxLayout()
        self.btn_mark_all_ok = QtWidgets.QPushButton(tr("sample.mark_all_ok"))
        self.btn_mark_all_ok.clicked.connect(self._mark_current_image_all_ok)
        footer.addWidget(self.btn_mark_all_ok)
        self.btn_mark_all_ng = QtWidgets.QPushButton(tr("sample.mark_all_ng"))
        self.btn_mark_all_ng.clicked.connect(self._mark_current_image_all_ng)
        footer.addWidget(self.btn_mark_all_ng)
        self.btn_clear_current = QtWidgets.QPushButton(tr("sample.clear_current_labels"))
        self.btn_clear_current.clicked.connect(self._clear_current_image_annotations)
        footer.addWidget(self.btn_clear_current)
        self.btn_open_autogen = QtWidgets.QPushButton(tr("sample.auto_roi"))
        self.btn_open_autogen.clicked.connect(self._open_autogen_dialog)
        footer.addWidget(self.btn_open_autogen)
        footer.addStretch(1)
        self.btn_prev = QtWidgets.QPushButton(tr("sample.prev"))
        self.btn_prev.clicked.connect(lambda: self._step_selection(-1))
        footer.addWidget(self.btn_prev)
        self.btn_next = QtWidgets.QPushButton(tr("sample.next"))
        self.btn_next.clicked.connect(lambda: self._step_selection(1))
        footer.addWidget(self.btn_next)
        btn_close = QtWidgets.QPushButton(tr("sample.close"))
        btn_close.clicked.connect(self.close)
        footer.addWidget(btn_close)
        root.addLayout(footer)

        camera_index = self.cmb_camera.findData(tool_page.current_camera_role())
        if camera_index >= 0:
            self.cmb_camera.setCurrentIndex(camera_index)
        sample_kind = tool_page._current_sample_tab_kind()
        sample_index = self.cmb_sample_kind.findData(sample_kind)
        if sample_index >= 0:
            self.cmb_sample_kind.setCurrentIndex(sample_index)

        tool_page.roiGeometryChanged.connect(self._on_tool_page_roi_geometry_changed)
        tool_page.inspectionItemsChanged.connect(self._on_tool_page_roi_geometry_changed)
        self.cmb_camera.currentIndexChanged.connect(lambda *_: self._reload_samples())
        self.cmb_sample_kind.currentIndexChanged.connect(lambda *_: self._reload_samples())
        self.sample_list.itemSelectionChanged.connect(self._on_sample_selected)
        self._reload_samples()

    def sync_camera_roles(self, roles: List[str]) -> None:
        normalized: List[str] = []
        for role in roles:
            role_text = normalize_camera_role(role)
            if role_text and role_text not in normalized:
                normalized.append(role_text)
        if not normalized:
            normalized = ["cam1"]
        current_role = str(self.cmb_camera.currentData() or "cam1")
        existing_roles = [
            str(self.cmb_camera.itemData(index) or "")
            for index in range(self.cmb_camera.count())
        ]
        if existing_roles == normalized and current_role in normalized:
            self.cmb_camera.setEnabled(len(normalized) > 1)
            if hasattr(self, "cmb_sample_kind"):
                self._reload_samples()
            return
        blocker = QtCore.QSignalBlocker(self.cmb_camera)
        self.cmb_camera.clear()
        for role in normalized:
            self.cmb_camera.addItem(role, role)
        index = self.cmb_camera.findData(current_role if current_role in normalized else normalized[0])
        self.cmb_camera.setCurrentIndex(index if index >= 0 else 0)
        self.cmb_camera.setEnabled(len(normalized) > 1)
        del blocker
        if hasattr(self, "cmb_sample_kind"):
            self._reload_samples()

    def _reload_samples(self, preferred_path: Optional[str] = None) -> None:
        tool_page = self._tool_page
        camera_role = str(self.cmb_camera.currentData() or "cam1")
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        current_path = (
            str(preferred_path or "").strip()
            or self._current_dialog_selected_path()
            or tool_page._current_selected_path()
            or ""
        )
        paths = tool_page._sample_paths_for_kind(sample_kind, camera_role)
        blocker = QtCore.QSignalBlocker(self.sample_list)
        self.sample_list.setUpdatesEnabled(False)
        try:
            self.sample_list.clear()
            selected_row = -1
            for index, path in enumerate(paths):
                item = QtWidgets.QListWidgetItem(tool_page._sample_item_display_text(path, sample_kind, camera_role))
                item.setToolTip(path)
                item.setData(QtCore.Qt.UserRole, path)
                self.sample_list.addItem(item)
                if current_path and path == current_path:
                    selected_row = index
        finally:
            self.sample_list.setUpdatesEnabled(True)
            del blocker
        if self.sample_list.count() == 0:
            self.preview_canvas.clear_image()
            self.lbl_image_status.setText(tr("sample.list_empty"))
            self.roi_table.setRowCount(0)
            self._sync_navigation_buttons()
            self._sync_tool_page_context("")
            return
        if selected_row < 0:
            selected_row = 0
        self.sample_list.setCurrentRow(selected_row)
        self._on_sample_selected()

    def _on_sample_selected(self) -> None:
        tool_page = self._tool_page
        item = self.sample_list.currentItem()
        if item is None:
            return
        path = str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "")
        camera_role = str(self.cmb_camera.currentData() or "cam1")
        if not path:
            return
        self._active_roi_label = ""
        self._load_canvas_preview(path, camera_role)
        usage_text = tool_page._sample_usage_text(path)
        annotation_state = tool_page._sample_annotation_state_for_path(path, camera_role)
        self.lbl_image_status.setText(f"Status: {usage_text} / {annotation_state}")
        self._populate_roi_table(path, camera_role)
        self._sync_navigation_buttons()
        self._sync_tool_page_context(path)

    def _populate_roi_table(self, path: str, camera_role: str) -> None:
        tool_page = self._tool_page
        labels = tool_page._inspection_label_names_for_role(camera_role)
        geometry_labels = tool_page.roi_annotations.geometry_labels_for_path(path)
        self.roi_table.setUpdatesEnabled(False)
        try:
            self.roi_table.setRowCount(len(labels))
            for row_index, label in enumerate(labels):
                self.roi_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(label))
                has_geometry = label in geometry_labels
                geometry_item = QtWidgets.QTableWidgetItem(tr("sample.generated") if has_geometry else tr("sample.missing_roi"))
                self.roi_table.setItem(row_index, 1, geometry_item)
                combo = QtWidgets.QComboBox()
                combo.addItem(tr("sample.unset"), "")
                combo.addItem("OK", "OK")
                combo.addItem("NG", "NG")
                current_status = tool_page._sample_roi_status_for_path(path, camera_role, label)
                combo_index = combo.findData(current_status)
                if combo_index < 0:
                    combo_index = 0
                combo.setCurrentIndex(combo_index)
                combo.setEnabled(has_geometry)
                combo.currentIndexChanged.connect(
                    lambda _index, image_path=path, role=camera_role, roi_label=label, widget=combo: self._on_roi_status_changed(
                        image_path,
                        role,
                        roi_label,
                        str(widget.currentData() or ""),
                    )
                )
                self.roi_table.setCellWidget(row_index, 2, combo)
        finally:
            self.roi_table.setUpdatesEnabled(True)

    def _on_roi_status_changed(self, path: str, camera_role: str, label: str, status: str) -> None:
        self._tool_page._set_sample_roi_status_for_path(path, camera_role, label, status)
        self._refresh_current_row_text(path, camera_role)
        self._refresh_tool_page_row_text(path, camera_role)
        self._tool_page._update_sample_panel_widgets()
        self._active_roi_label = label
        self._refresh_canvas_overlays(path, camera_role)
        self.lbl_image_status.setText(
            f"Status: {self._tool_page._sample_usage_text(path)} / "
            f"{self._tool_page._sample_annotation_state_for_path(path, camera_role)}"
        )

    def _refresh_current_row_text(self, path: str, camera_role: str) -> None:
        current_item = self.sample_list.currentItem()
        if current_item is None:
            return
        item_path = str(current_item.data(QtCore.Qt.UserRole) or current_item.toolTip() or "")
        if item_path != path:
            return
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        current_item.setText(self._tool_page._sample_item_display_text(path, sample_kind, camera_role))

    def _refresh_tool_page_row_text(self, path: str, camera_role: str) -> None:
        """Update one mirrored row without rebuilding every sample item."""
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        list_widget = self._tool_page.ok_list if sample_kind == "train" else self._tool_page.test_list
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            item_path = str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "")
            if item_path == path:
                item.setText(self._tool_page._sample_item_display_text(path, sample_kind, camera_role))
                break

    def _current_path_and_role(self) -> tuple[str, str]:
        item = self.sample_list.currentItem()
        if item is None:
            return "", str(self.cmb_camera.currentData() or "cam1")
        path = str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "")
        role = str(self.cmb_camera.currentData() or "cam1")
        return path, role

    def _current_dialog_selected_path(self) -> str:
        item = self.sample_list.currentItem()
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "").strip()

    def _sync_tool_page_context(self, preferred_path: str = "") -> None:
        if getattr(self, "_suppress_tool_page_context_sync", False):
            return
        camera_role = str(self.cmb_camera.currentData() or "cam1")
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        if self._tool_page.current_camera_role() != camera_role:
            try:
                self._tool_page._set_current_camera_role(camera_role, sync_debug_role=True)
            except Exception:
                pass
        target_index = 0 if sample_kind == "train" else 1
        tabs = getattr(self._tool_page, "tabs", None)
        if tabs is not None and tabs.currentIndex() != target_index:
            tabs.setCurrentIndex(target_index)
        path = str(preferred_path or "").strip()
        if not path:
            return
        try:
            self._tool_page._select_path_in_current_tab(
                path,
                pixmap=self.preview_canvas.image_pixmap(),
            )
        except Exception:
            pass

    def _open_autogen_dialog(self) -> None:
        dialog = getattr(self, "_sample_annotation_autogen_dialog", None)
        if dialog is None:
            dialog = _SampleAnnotationAutoRoiDialog(self)
            self._sample_annotation_autogen_dialog = dialog
            dialog.finished.connect(lambda *_: setattr(self, "_sample_annotation_autogen_dialog", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_tool_page_roi_geometry_changed(self) -> None:
        if bool(getattr(self._tool_page, "_suppress_sample_preview_reload", False)):
            return
        previous = bool(getattr(self, "_suppress_tool_page_context_sync", False))
        self._suppress_tool_page_context_sync = True
        try:
            self._reload_samples(preferred_path=self._current_dialog_selected_path())
        finally:
            self._suppress_tool_page_context_sync = previous

    def _refresh_after_auto_roi_change(self, changed_paths: set[str], camera_role: str) -> None:
        path = self._current_dialog_selected_path()
        if not path:
            return
        role = str(camera_role or self.cmb_camera.currentData() or "cam1")
        self._refresh_current_row_text(path, role)
        if path in set(changed_paths or set()):
            self._populate_roi_table(path, role)
            self._refresh_canvas_overlays(path, role)
            self.lbl_image_status.setText(
                f"Status: {self._tool_page._sample_usage_text(path)} / "
                f"{self._tool_page._sample_annotation_state_for_path(path, role)}"
            )

    def _mark_current_image_all_ok(self) -> None:
        path, camera_role = self._current_path_and_role()
        if not path:
            return
        self._tool_page._mark_sample_path_all_ok(path, camera_role)
        self._refresh_after_annotation_change(path, camera_role)

    def _mark_current_image_all_ng(self) -> None:
        path, camera_role = self._current_path_and_role()
        if not path:
            return
        self._tool_page._mark_sample_path_all_ng(path, camera_role)
        self._refresh_after_annotation_change(path, camera_role)

    def _clear_current_image_annotations(self) -> None:
        path, camera_role = self._current_path_and_role()
        if not path:
            return
        self._tool_page._clear_sample_path_annotations(path, camera_role)
        self._refresh_after_annotation_change(path, camera_role)

    def _step_selection(self, direction: int) -> None:
        count = self.sample_list.count()
        if count <= 0:
            return
        current_row = self.sample_list.currentRow()
        if current_row < 0:
            current_row = 0
        next_row = max(0, min(count - 1, current_row + int(direction)))
        if next_row == current_row:
            return
        self.sample_list.setCurrentRow(next_row)

    def _sync_navigation_buttons(self) -> None:
        count = self.sample_list.count()
        row = self.sample_list.currentRow()
        has_selection = count > 0 and row >= 0
        self.btn_mark_all_ok.setEnabled(has_selection)
        self.btn_mark_all_ng.setEnabled(has_selection)
        self.btn_clear_current.setEnabled(has_selection)
        self.btn_prev.setEnabled(has_selection and row > 0)
        self.btn_next.setEnabled(has_selection and row >= 0 and row < count - 1)

    def _refresh_after_annotation_change(self, path: str, camera_role: str) -> None:
        self._refresh_current_row_text(path, camera_role)
        self._refresh_tool_page_row_text(path, camera_role)
        self._tool_page._update_sample_panel_widgets()
        self._populate_roi_table(path, camera_role)
        self._refresh_canvas_overlays(path, camera_role)
        self.lbl_image_status.setText(
            tr(
                "debug.current_image_state",
                usage=self._tool_page._sample_usage_text(path),
                state=self._tool_page._sample_annotation_state_for_path(path, camera_role),
            )
        )


from ui.debug.tool_page import sample_annotation_overlay as _sample_annotation_overlay

for _method_name in (
    "_load_canvas_preview",
    "_refresh_canvas_overlays",
    "_on_canvas_image_pressed",
    "_focus_roi_row",
    "_show_roi_label_menu",
    "_set_roi_status_from_canvas",
    "_find_roi_label_at_point",
):
    setattr(_SampleAnnotationPreviewDialog, _method_name, getattr(_sample_annotation_overlay, _method_name))

del _method_name
