"""Inspection items table refresh and edit handlers."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.measurement import LINE_DISTANCE_ALGORITHMS
from algorithms.registry import list_tool_algorithm_specs
from common.algorithm_codes import normalize_tool_algorithm_code
from domain import SUPPORTED_CAMERA_IDS
from ui.debug.tool_page.inspection_item_status import _inspection_item_status
from ui.debug.tool_page.measurement_algorithms import (
    hide_from_algorithm_picker,
    public_algorithm_code,
)
from ui.i18n import tr


def _require_tool_permission(tool_page, permission_key: str, action_name: str) -> bool:
    top_level = tool_page.window()
    require_permission = getattr(top_level, "_require_permission", None)
    if callable(require_permission):
        return bool(require_permission(permission_key, action_name))
    return True


def _audit_tool_event(tool_page, **payload) -> None:
    audit_event = getattr(tool_page.window(), "_audit_event", None)
    if callable(audit_event):
        audit_event(**payload)


def _refresh_inspection_items_table(tool_page) -> None:
    from ui.debug.tool_page.tool_config import (
        _inspection_combo_style,
        _inspection_item_display_name,
        _sync_inspection_items_row_highlight,
        _update_delete_line_distance_button,
        _update_learning_backbone_hint,
        _visible_inspection_item_indexes,
    )

    table = getattr(tool_page, "inspection_items_table", None)
    if table is None:
        return

    selected_item = _selected_inspection_item(tool_page)
    selected_item_id = str(getattr(selected_item, "item_id", "") or "").strip()
    visible_indexes = _visible_inspection_item_indexes(tool_page)
    tool_page._visible_inspection_item_indexes = list(visible_indexes)
    tool_page._inspection_items_table_loading = True
    table.blockSignals(True)
    try:
        table.setRowCount(len(visible_indexes))
        algorithm_specs = [
            spec
            for spec in list_tool_algorithm_specs()
            if not hide_from_algorithm_picker(getattr(spec, "code", ""))
        ]

        for row, actual_index in enumerate(visible_indexes):
            inspection_item = tool_page.inspection_items[actual_index]
            enabled_item = QtWidgets.QTableWidgetItem("")
            enabled_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            enabled_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            enabled_item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if inspection_item.enabled
                else QtCore.Qt.CheckState.Unchecked
            )
            table.setItem(row, 0, enabled_item)

            display_name = _inspection_item_display_name(inspection_item)
            name_item = QtWidgets.QTableWidgetItem(display_name)
            name_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, 1, name_item)

            camera_combo = QtWidgets.QComboBox(table)
            camera_combo.setStyleSheet(_inspection_combo_style(False))
            for camera_id in SUPPORTED_CAMERA_IDS:
                camera_combo.addItem(camera_id, camera_id)
            camera_index = max(0, camera_combo.findData(inspection_item.camera_id))
            camera_combo.setCurrentIndex(camera_index)
            camera_combo.currentTextChanged.connect(
                lambda value, row_index=actual_index: tool_page._on_inspection_item_camera_changed(row_index, value)
            )
            table.setCellWidget(row, 2, camera_combo)

            algorithm_combo = QtWidgets.QComboBox(table)
            algorithm_combo.setStyleSheet(_inspection_combo_style(False))
            for spec in algorithm_specs:
                algorithm_name = tool_page.algo.algorithm_display_name(spec.code) or spec.display_name or spec.code
                algorithm_combo.addItem(algorithm_name, spec.code)
            current_algorithm_raw = normalize_tool_algorithm_code(inspection_item.algorithm_code)
            if current_algorithm_raw in LINE_DISTANCE_ALGORITHMS:
                algorithm_combo.addItem(tr("debug.algorithm.line_distance"), current_algorithm_raw)
                algorithm_combo.setCurrentIndex(algorithm_combo.count() - 1)
                algorithm_combo.setEnabled(False)
            else:
                current_algorithm = public_algorithm_code(inspection_item.algorithm_code)
                algorithm_index = algorithm_combo.findData(current_algorithm)
                if algorithm_index < 0:
                    algorithm_index = 0
                algorithm_combo.setCurrentIndex(algorithm_index)
            algorithm_combo.currentIndexChanged.connect(
                lambda _index, row_index=actual_index, combo=algorithm_combo: tool_page._on_inspection_item_algorithm_changed(
                    row_index,
                    combo.currentData(),
                )
            )
            table.setCellWidget(row, 3, algorithm_combo)

            status_text, status_tooltip, status_color = _inspection_item_status(tool_page, inspection_item)
            status_item = QtWidgets.QTableWidgetItem(status_text)
            status_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            status_item.setToolTip(status_tooltip)
            status_item.setForeground(QtGui.QBrush(QtGui.QColor(status_color)))
            table.setItem(row, 4, status_item)

        if table.rowCount() == 0:
            table.clearContents()
    finally:
        table.blockSignals(False)
        tool_page._inspection_items_table_loading = False

    if table.rowCount() > 0:
        restored_row = -1
        if selected_item_id:
            for visible_row, actual_index in enumerate(visible_indexes):
                item = tool_page.inspection_items[actual_index]
                if str(getattr(item, "item_id", "") or "").strip() == selected_item_id:
                    restored_row = visible_row
                    break
        if 0 <= restored_row < table.rowCount():
            table.setCurrentCell(restored_row, 1)
        else:
            table.clearSelection()
            table.setCurrentItem(None)
    table.setColumnWidth(0, 52)
    table.setColumnWidth(2, 78)
    _sync_inspection_items_row_highlight(tool_page)
    _on_inspection_items_selection_changed(tool_page)

    _update_delete_line_distance_button(tool_page)
    _update_learning_backbone_hint(tool_page)


def _selected_inspection_item_row(tool_page) -> int:
    from ui.debug.tool_page.tool_config import _actual_inspection_item_index

    table = getattr(tool_page, "inspection_items_table", None)
    if table is None:
        return -1
    selection_model = table.selectionModel()
    visible_row = -1
    if selection_model is not None and selection_model.hasSelection():
        selected_rows = selection_model.selectedRows()
        if selected_rows:
            visible_row = int(selected_rows[0].row())
    if visible_row < 0:
        return -1
    row = _actual_inspection_item_index(tool_page, visible_row)
    if row < 0 or row >= len(tool_page.inspection_items):
        return -1
    return row


def _selected_inspection_item(tool_page):
    row = _selected_inspection_item_row(tool_page)
    if row < 0:
        return None
    return tool_page.inspection_items[row]


def _on_inspection_items_selection_changed(tool_page) -> None:
    from ui.debug.tool_page.tool_config import (
        _sync_inspection_items_row_highlight,
        _update_delete_line_distance_button,
    )

    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    _sync_inspection_items_row_highlight(tool_page)
    item = _selected_inspection_item(tool_page)
    if item is None:
        tool_page._refresh_lists()
        tool_page._update_runtime_widgets()
        tool_page._update_learning_backbone_hint()
        _update_delete_line_distance_button(tool_page)
        tool_page._update_measurement_params_panel()
        return
    if tool_page.algo.is_learning_tool(item.algorithm_code):
        algorithm = tool_page.algo.current_learning_backbone(item.camera_id)
    else:
        algorithm = tool_page.algo.resolve_tool_algorithm(item.algorithm_code, item.camera_id)
    display_algorithm = public_algorithm_code(algorithm)
    tool_page._updating_runtime_params = True
    try:
        tool_page._set_current_algorithm(display_algorithm)
    finally:
        tool_page._updating_runtime_params = False
    tool_page.algo.product_params.algorithm = algorithm
    tool_page._refresh_lists()
    tool_page._update_runtime_widgets()
    tool_page._update_learning_backbone_hint()
    _update_delete_line_distance_button(tool_page)
    tool_page._update_measurement_params_panel()
    image_path = tool_page.canvas.image_path()
    if image_path:
        tool_page._load_shape_for_label(image_path, tool_page._current_label())


def _on_inspection_items_table_item_changed(tool_page, table_item: QtWidgets.QTableWidgetItem) -> None:
    from ui.debug.tool_page.tool_config import (
        _actual_inspection_item_index,
        _persist_inspection_items,
    )

    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    row = _actual_inspection_item_index(tool_page, table_item.row())
    if row < 0 or row >= len(tool_page.inspection_items):
        return
    inspection_item = tool_page.inspection_items[row]
    if not _require_tool_permission(tool_page, "inspection.edit_items", "修改检测项"):
        _refresh_inspection_items_table(tool_page)
        return
    before = {
        "enabled": bool(getattr(inspection_item, "enabled", True)),
        "display_name": str(getattr(inspection_item, "display_name", "") or ""),
    }

    if table_item.column() == 0:
        inspection_item.enabled = table_item.checkState() == QtCore.Qt.CheckState.Checked
    elif table_item.column() == 1:
        display_name = str(table_item.text() or "").strip() or inspection_item.roi_label or inspection_item.item_id
        if display_name != table_item.text():
            tool_page._inspection_items_table_loading = True
            try:
                table_item.setText(display_name)
            finally:
                tool_page._inspection_items_table_loading = False
        inspection_item.display_name = display_name
    else:
        return

    _persist_inspection_items(tool_page)
    after = {
        "enabled": bool(getattr(inspection_item, "enabled", True)),
        "display_name": str(getattr(inspection_item, "display_name", "") or ""),
    }
    if before != after:
        _audit_tool_event(
            tool_page,
            module="检测项",
            action="修改检测项",
            target=str(getattr(inspection_item, "item_id", "") or ""),
            before_value=str(before),
            after_value=str(after),
        )
    _refresh_inspection_items_table(tool_page)


def _on_inspection_item_camera_changed(tool_page, row: int, camera_id: str) -> None:
    from ui.debug.tool_page.tool_config import _persist_inspection_items

    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    if row < 0 or row >= len(tool_page.inspection_items):
        return
    if not _require_tool_permission(tool_page, "inspection.edit_items", "修改检测项相机"):
        _refresh_inspection_items_table(tool_page)
        return
    normalized = str(camera_id or "cam1").strip() or "cam1"
    if normalized not in SUPPORTED_CAMERA_IDS:
        normalized = "cam1"
    before = str(getattr(tool_page.inspection_items[row], "camera_id", "") or "")
    tool_page.inspection_items[row].camera_id = normalized
    _persist_inspection_items(tool_page)
    if before != normalized:
        _audit_tool_event(
            tool_page,
            module="检测项",
            action="修改检测项相机",
            target=str(getattr(tool_page.inspection_items[row], "item_id", "") or ""),
            before_value=before,
            after_value=normalized,
        )
    _refresh_inspection_items_table(tool_page)
    tool_page._refresh_lists()


def _on_inspection_item_algorithm_changed(tool_page, row: int, algorithm_code: object) -> None:
    from ui.debug.tool_page.tool_config import _persist_inspection_items

    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    if row < 0 or row >= len(tool_page.inspection_items):
        return
    if not _require_tool_permission(tool_page, "inspection.edit_items", "修改检测项算法"):
        _refresh_inspection_items_table(tool_page)
        return
    normalized = normalize_tool_algorithm_code(algorithm_code)
    before = str(getattr(tool_page.inspection_items[row], "algorithm_code", "") or "")
    tool_page.inspection_items[row].algorithm_code = normalized
    spec = tool_page.algo.tool_algorithm_spec(normalized)
    if spec is not None and not dict(tool_page.inspection_items[row].params or {}):
        tool_page.inspection_items[row].params = dict(spec.default_params or {})
    if row == _selected_inspection_item_row(tool_page):
        if tool_page.algo.is_learning_tool(normalized):
            camera_id = tool_page.inspection_items[row].camera_id
            backbone = tool_page.algo.current_learning_backbone(camera_id)
            tool_page.algo.product_params.algorithm = backbone
            tool_page._updating_runtime_params = True
            try:
                tool_page._set_current_algorithm(backbone)
            finally:
                tool_page._updating_runtime_params = False
        else:
            tool_page.algo.product_params.algorithm = normalized
            tool_page._updating_runtime_params = True
            try:
                tool_page._set_current_algorithm(public_algorithm_code(normalized))
            finally:
                tool_page._updating_runtime_params = False
        tool_page._update_runtime_widgets()
    _persist_inspection_items(tool_page)
    if before != normalized:
        _audit_tool_event(
            tool_page,
            module="检测项",
            action="修改检测项算法",
            target=str(getattr(tool_page.inspection_items[row], "item_id", "") or ""),
            before_value=before,
            after_value=normalized,
        )
    _refresh_inspection_items_table(tool_page)


