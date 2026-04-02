"""Inspection-tool configuration helpers for ToolPage."""

from __future__ import annotations

import os
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.registry import list_tool_algorithm_specs, normalize_tool_algorithm_code
from domain import SUPPORTED_CAMERA_IDS, save_inspection_items


def _current_camera_role(tool_page) -> str:
    getter = getattr(tool_page, "current_camera_role", None)
    if callable(getter):
        return str(getter() or "cam1").strip() or "cam1"
    return "cam1"


def _visible_inspection_item_indexes(tool_page) -> list[int]:
    current_role = _current_camera_role(tool_page)
    visible: list[int] = []
    for index, inspection_item in enumerate(getattr(tool_page, "inspection_items", [])):
        item_role = str(getattr(inspection_item, "camera_id", "") or "").strip() or "cam1"
        if item_role == current_role:
            visible.append(index)
    return visible


def _actual_inspection_item_index(tool_page, visible_row: int) -> int:
    indexes = list(getattr(tool_page, "_visible_inspection_item_indexes", []))
    if visible_row < 0 or visible_row >= len(indexes):
        return -1
    return int(indexes[visible_row])


def _persist_inspection_items(tool_page) -> None:
    save_inspection_items(tool_page.inspection_items, tool_page.session.inspection_items_path)
    tool_page.inspectionItemsChanged.emit()


def _inspection_combo_style(selected: bool) -> str:
    background = "#6ec0ff" if selected else "#3a3a3a"
    foreground = "#1a1a1a" if selected else "#d0d0d0"
    border = "#8fd2ff" if selected else "#5a5a5a"
    hover = "#89d1ff" if selected else "#4a4a4a"
    return (
        "QComboBox{"
        f"background:{background};color:{foreground};border:1px solid {border};"
        "padding:2px 18px 2px 6px;border-radius:2px;}"
        f"QComboBox:hover{{background:{hover};}}"
        "QComboBox::drop-down{border:none;width:18px;}"
        "QAbstractItemView{background:#333333;color:#e0e0e0;border:1px solid #505050;"
        "selection-background-color:#3794ff;selection-color:#ffffff;}"
    )


def _sync_inspection_items_row_highlight(tool_page) -> None:
    table = getattr(tool_page, "inspection_items_table", None)
    if table is None:
        return
    selection_model = table.selectionModel()
    selected_row = -1
    if selection_model is not None and selection_model.hasSelection():
        selected_rows = selection_model.selectedRows()
        if selected_rows:
            selected_row = int(selected_rows[0].row())
    for row in range(table.rowCount()):
        is_selected = row == selected_row
        for column in (2, 3):
            widget = table.cellWidget(row, column)
            if widget is not None:
                widget.setStyleSheet(_inspection_combo_style(is_selected))


def _update_learning_backbone_hint(tool_page) -> None:
    label = getattr(tool_page, "lbl_tool_config_hint", None)
    if label is None:
        return
    item = _selected_inspection_item(tool_page)
    if item is None:
        label.clear()
        label.hide()
        label.setToolTip("")
        return
    status_text, tooltip, color = _inspection_item_status(tool_page, item)
    display_name = str(item.display_name or item.roi_label or item.item_id or "工具").strip()
    label.setText(f"当前工具：{display_name}  {status_text}")
    label.setStyleSheet(f"color:{color};font-size:11px;")
    label.setToolTip(tooltip)
    label.show()


def _format_timestamp(path: str) -> str:
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return "-"
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _inspection_item_status(tool_page, inspection_item):
    if not getattr(inspection_item, "enabled", True):
        return "已禁用", "当前工具已禁用，不参与训练和运行。", "#8a8a8a"

    if tool_page.algo.is_learning_tool(inspection_item.algorithm_code):
        backbone = tool_page.algo.current_learning_backbone()
        if not backbone:
            return "未选择", "请先为学习工具选择子类。", "#d98c8c"
        model_path = tool_page.algo.embedding_model_path(
            backbone,
            tool_page.session.product_dir,
            model_key=inspection_item.model_key,
        )
        if os.path.exists(model_path):
            tooltip = (
                f"学习工具已训练\n"
                f"模型: {os.path.basename(model_path)}\n"
                f"子类: {tool_page.algo.algorithm_display_name(backbone) or backbone}\n"
                f"更新时间: {_format_timestamp(model_path)}"
            )
            return "已训练", tooltip, "#79d279"
        legacy_path = tool_page.algo.embedding_model_path(backbone, tool_page.session.product_dir)
        if os.path.exists(legacy_path):
            tooltip = (
                f"学习工具已训练（兼容旧共享模型）\n"
                f"模型: {os.path.basename(legacy_path)}\n"
                f"子类: {tool_page.algo.algorithm_display_name(backbone) or backbone}\n"
                f"更新时间: {_format_timestamp(legacy_path)}"
            )
            return "已训练(旧)", tooltip, "#cfc76a"
        tooltip = (
            f"学习工具未训练\n"
            f"目标模型: {os.path.basename(model_path)}\n"
            f"子类: {tool_page.algo.algorithm_display_name(backbone) or backbone}"
        )
        return "未训练", tooltip, "#d98c8c"

    algorithm = tool_page.algo.resolve_tool_algorithm(inspection_item.algorithm_code)
    model_dict = tool_page.algo.get_traditional_model_dict(algorithm, model_key=inspection_item.model_key)
    if isinstance(model_dict, dict):
        storage_key = tool_page.algo.traditional_model_storage_key(algorithm, model_key=inspection_item.model_key)
        actual_key = storage_key if storage_key in tool_page.algo.product_params.traditional_models else algorithm
        threshold = model_dict.get("threshold")
        ok_when = str(model_dict.get("ok_when", "")).strip() or "-"
        accuracy = model_dict.get("accuracy")
        detail_parts = [
            f"算法: {tool_page.algo.algorithm_display_name(algorithm) or algorithm}",
            f"阈值: {float(threshold):.4f}" if threshold is not None else "阈值: -",
            f"规则: {ok_when}",
        ]
        if accuracy is not None:
            detail_parts.append(f"准确率: {float(accuracy):.4f}")
        detail_parts.append(f"存储键: {actual_key}")
        status_text = "已标定" if actual_key == storage_key else "已标定(旧)"
        color = "#79d279" if actual_key == storage_key else "#cfc76a"
        return status_text, "\n".join(detail_parts), color

    tooltip = (
        f"传统工具未标定\n"
        f"算法: {tool_page.algo.algorithm_display_name(algorithm) or algorithm}\n"
        f"存储键: {tool_page.algo.traditional_model_storage_key(algorithm, model_key=inspection_item.model_key)}"
    )
    return "未标定", tooltip, "#d98c8c"


def _refresh_inspection_items_table(tool_page) -> None:
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
        algorithm_specs = list_tool_algorithm_specs()

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

            display_name = inspection_item.display_name or inspection_item.roi_label or inspection_item.item_id
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
                algorithm_combo.addItem(spec.display_name, spec.code)
            current_algorithm = normalize_tool_algorithm_code(inspection_item.algorithm_code)
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

    _update_learning_backbone_hint(tool_page)


def _selected_inspection_item_row(tool_page) -> int:
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
    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    _sync_inspection_items_row_highlight(tool_page)
    item = _selected_inspection_item(tool_page)
    if item is None:
        tool_page._refresh_lists()
        tool_page._update_runtime_widgets()
        tool_page._update_learning_backbone_hint()
        return
    if tool_page.algo.is_learning_tool(item.algorithm_code):
        algorithm = tool_page.algo.current_learning_backbone()
    else:
        algorithm = tool_page.algo.resolve_tool_algorithm(item.algorithm_code)
    tool_page._updating_runtime_params = True
    try:
        tool_page._set_current_algorithm(algorithm)
    finally:
        tool_page._updating_runtime_params = False
    tool_page.algo.product_params.algorithm = algorithm
    tool_page._refresh_lists()
    tool_page._update_runtime_widgets()
    tool_page._update_learning_backbone_hint()
    image_path = tool_page.canvas.image_path()
    if image_path:
        tool_page._load_shape_for_label(image_path, tool_page._current_label())


def _on_inspection_items_table_item_changed(tool_page, table_item: QtWidgets.QTableWidgetItem) -> None:
    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    row = _actual_inspection_item_index(tool_page, table_item.row())
    if row < 0 or row >= len(tool_page.inspection_items):
        return
    inspection_item = tool_page.inspection_items[row]

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
    _refresh_inspection_items_table(tool_page)


def _on_inspection_item_camera_changed(tool_page, row: int, camera_id: str) -> None:
    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    if row < 0 or row >= len(tool_page.inspection_items):
        return
    normalized = str(camera_id or "cam1").strip() or "cam1"
    if normalized not in SUPPORTED_CAMERA_IDS:
        normalized = "cam1"
    tool_page.inspection_items[row].camera_id = normalized
    _persist_inspection_items(tool_page)
    _refresh_inspection_items_table(tool_page)
    tool_page._refresh_lists()


def _on_inspection_item_algorithm_changed(tool_page, row: int, algorithm_code: object) -> None:
    if getattr(tool_page, "_inspection_items_table_loading", False):
        return
    if row < 0 or row >= len(tool_page.inspection_items):
        return
    normalized = normalize_tool_algorithm_code(algorithm_code)
    tool_page.inspection_items[row].algorithm_code = normalized
    if row == _selected_inspection_item_row(tool_page):
        if tool_page.algo.is_learning_tool(normalized):
            tool_page.algo.product_params.algorithm = tool_page.algo.current_learning_backbone()
            tool_page._updating_runtime_params = True
            try:
                tool_page._set_current_algorithm(tool_page.algo.current_learning_backbone())
            finally:
                tool_page._updating_runtime_params = False
        else:
            tool_page.algo.product_params.algorithm = normalized
            tool_page._updating_runtime_params = True
            try:
                tool_page._set_current_algorithm(normalized)
            finally:
                tool_page._updating_runtime_params = False
        tool_page._update_runtime_widgets()
    _persist_inspection_items(tool_page)
    _refresh_inspection_items_table(tool_page)


__all__ = [
    "_on_inspection_items_selection_changed",
    "_on_inspection_item_algorithm_changed",
    "_on_inspection_item_camera_changed",
    "_on_inspection_items_table_item_changed",
    "_persist_inspection_items",
    "_refresh_inspection_items_table",
    "_selected_inspection_item",
    "_selected_inspection_item_row",
    "_update_learning_backbone_hint",
]
