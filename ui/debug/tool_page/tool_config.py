"""Inspection-tool configuration helpers for ToolPage."""

from __future__ import annotations

import os
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.measurement import LINE_DISTANCE_ALGORITHMS
from algorithms.registry import list_tool_algorithm_specs, normalize_tool_algorithm_code
from domain import InspectionItem, SUPPORTED_CAMERA_IDS, save_inspection_items
from ui.i18n import tr


def _is_line_distance_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() in LINE_DISTANCE_ALGORITHMS


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


def _unique_item_id(tool_page, base: str) -> str:
    existing = {
        str(getattr(item, "item_id", "") or "").strip()
        for item in list(getattr(tool_page, "inspection_items", []) or [])
    }
    candidate = str(base or "tool").strip() or "tool"
    if candidate not in existing:
        return candidate
    index = 2
    while f"{candidate}_{index}" in existing:
        index += 1
    return f"{candidate}_{index}"


def _add_line_distance_tool(tool_page) -> None:
    camera_role = _current_camera_role(tool_page)
    line_options = [
        item_id
        for _display, item_id in _line_item_options(
            tool_page,
            type("_Selected", (), {"camera_id": camera_role, "item_id": ""})(),
        )
    ]
    items_by_id = {
        str(getattr(item, "item_id", "") or "").strip(): item
        for item in list(getattr(tool_page, "inspection_items", []) or [])
    }

    selected_item = _selected_inspection_item(tool_page)
    selected_params = dict(getattr(selected_item, "params", {}) or {}) if selected_item is not None else {}
    line_param_candidates = [selected_params]
    for item_id_ref in line_options[:2]:
        line_param_candidates.append(dict(getattr(items_by_id.get(item_id_ref), "params", {}) or {}))

    unit = str(selected_params.get("limit_unit", "") or "").strip().lower()
    if unit not in {"px", "mm"}:
        unit = "px"
        for candidate in line_param_candidates:
            candidate_unit = str(candidate.get("limit_unit", "") or "").strip().lower()
            if candidate_unit in {"px", "mm"}:
                unit = candidate_unit
                break
    pixel_size = 0.0
    for candidate in line_param_candidates:
        pixel_size = _optional_param_float(candidate, "pixel_size_mm") or 0.0
        if pixel_size > 0.0:
            break

    item_id = _unique_item_id(tool_page, "line_distance")
    params = {
        "line_a_item_id": line_options[0] if len(line_options) >= 1 else "",
        "line_b_item_id": line_options[1] if len(line_options) >= 2 else "",
        "limit_unit": unit,
    }
    if pixel_size > 0.0:
        params["pixel_size_mm"] = pixel_size
    lower = _optional_param_float(selected_params, "lower_limit", f"lower_limit_{unit}")
    upper = _optional_param_float(selected_params, "upper_limit", f"upper_limit_{unit}")
    if lower is not None:
        params["lower_limit"] = lower
    if upper is not None:
        params["upper_limit"] = upper
    tool_page.inspection_items.append(
        InspectionItem(
            item_id=item_id,
            display_name="Line Distance",
            camera_id=camera_role,
            roi_label="",
            algorithm_code="line_distance",
            enabled=True,
            params=params,
        )
    )
    _persist_inspection_items(tool_page)
    _refresh_inspection_items_table(tool_page)
    table = getattr(tool_page, "inspection_items_table", None)
    if table is not None:
        for visible_row, actual_index in enumerate(getattr(tool_page, "_visible_inspection_item_indexes", []) or []):
            item = tool_page.inspection_items[actual_index]
            if str(getattr(item, "item_id", "") or "") == item_id:
                table.setCurrentCell(visible_row, 1)
                break


def _update_delete_line_distance_button(tool_page) -> None:
    button = getattr(tool_page, "btn_delete_line_distance_tool", None)
    if button is None:
        return
    item = _selected_inspection_item(tool_page)
    can_delete = (
        item is not None
        and _is_line_distance_algorithm(normalize_tool_algorithm_code(getattr(item, "algorithm_code", "")))
    )
    button.setEnabled(can_delete)
    button.setVisible(can_delete)


def _inspection_item_display_name(inspection_item) -> str:
    item_id = str(getattr(inspection_item, "item_id", "") or "").strip()
    raw_name = str(
        getattr(inspection_item, "display_name", "")
        or getattr(inspection_item, "roi_label", "")
        or item_id
    ).strip()
    algorithm = normalize_tool_algorithm_code(getattr(inspection_item, "algorithm_code", ""))
    if _is_line_distance_algorithm(algorithm):
        default_names = {"", "Line Distance", "line_distance", tr("debug.algorithm.line_distance")}
        display_key = "debug.algorithm.line_distance"
        if algorithm == "line_distance_ref_normal":
            default_names.update(
                {
                    "Reference Normal Distance",
                    "line_distance_ref_normal",
                    tr("debug.algorithm.line_distance_ref_normal"),
                }
            )
            display_key = "debug.algorithm.line_distance_ref_normal"
        if item_id.startswith("line_distance"):
            default_names.add(item_id)
        if raw_name in default_names:
            return tr(display_key)
    return raw_name


def _delete_selected_line_distance_tool(tool_page) -> None:
    row = _selected_inspection_item_row(tool_page)
    if row < 0 or row >= len(getattr(tool_page, "inspection_items", []) or []):
        QtWidgets.QMessageBox.information(
            tool_page,
            tr("debug.measurement.delete_line_distance_tool"),
            tr("debug.measurement.delete_line_distance_select"),
        )
        return

    inspection_item = tool_page.inspection_items[row]
    if not _is_line_distance_algorithm(normalize_tool_algorithm_code(getattr(inspection_item, "algorithm_code", ""))):
        QtWidgets.QMessageBox.information(
            tool_page,
            tr("debug.measurement.delete_line_distance_tool"),
            tr("debug.measurement.delete_line_distance_select"),
        )
        return

    display_name = (
        _inspection_item_display_name(inspection_item)
        or tr("debug.algorithm.line_distance")
    )
    del tool_page.inspection_items[row]
    _persist_inspection_items(tool_page)
    _refresh_inspection_items_table(tool_page)
    _update_delete_line_distance_button(tool_page)
    status_label = getattr(tool_page, "lbl_status", None)
    if status_label is not None:
        status_label.setText(tr("debug.measurement.deleted_line_distance", name=display_name))


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
    display_name = str(item.display_name or item.roi_label or item.item_id or tr("debug.tool")).strip()
    label.setText(f"{tr('debug.tool')}: {display_name}  {status_text}")
    label.setStyleSheet(f"color:{color};font-size:11px;")
    label.setToolTip(tooltip)
    label.show()


def _optional_param_float(params: dict, *keys: str):
    for key in keys:
        if key not in params:
            continue
        value = params.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _line_direction(params: dict, key: str, fallback: str) -> str:
    line_params = params.get(key)
    if not isinstance(line_params, dict):
        line_params = {}
    direction = str(line_params.get("direction", fallback) or fallback).strip()
    if direction not in {"left_right", "right_left", "top_down", "bottom_up"}:
        direction = fallback
    return direction


def _line_params(params: dict, key: str) -> dict:
    value = params.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _line_item_options(tool_page, selected_item) -> list[tuple[str, str]]:
    current_role = str(getattr(selected_item, "camera_id", "") or _current_camera_role(tool_page)).strip() or "cam1"
    current_id = str(getattr(selected_item, "item_id", "") or "").strip()
    options: list[tuple[str, str]] = []
    for item in list(getattr(tool_page, "inspection_items", []) or []):
        if str(getattr(item, "camera_id", "") or "").strip() != current_role:
            continue
        item_id = str(getattr(item, "item_id", "") or "").strip()
        if not item_id or item_id == current_id:
            continue
        algorithm = str(tool_page.algo.resolve_tool_algorithm(getattr(item, "algorithm_code", "")) or "").strip()
        if algorithm != "find_line":
            continue
        display = str(getattr(item, "display_name", "") or getattr(item, "roi_label", "") or item_id).strip()
        options.append((display, item_id))
    return options


def _set_combo_current_data(combo: QtWidgets.QComboBox, value: object) -> None:
    target = str(value or "").strip()
    index = combo.findData(target)
    if index < 0 and combo.count() > 0:
        index = 0
    if index >= 0:
        combo.setCurrentIndex(index)


def _populate_line_tool_combo(combo: QtWidgets.QComboBox, options: list[tuple[str, str]]) -> None:
    combo.blockSignals(True)
    try:
        combo.clear()
        for display, item_id in options:
            combo.addItem(display, item_id)
    finally:
        combo.blockSignals(False)


def _set_measurement_row_visible(tool_page, field_widget: QtWidgets.QWidget, visible: bool) -> None:
    frame = getattr(tool_page, "measurement_params_frame", None)
    layout = frame.layout() if frame is not None else None
    if isinstance(layout, QtWidgets.QFormLayout):
        label_widget = layout.labelForField(field_widget)
        if label_widget is not None:
            label_widget.setVisible(visible)
    field_widget.setVisible(visible)


def _update_measurement_params_panel(tool_page) -> None:
    frame = getattr(tool_page, "measurement_params_frame", None)
    if frame is None:
        return
    item = _selected_inspection_item(tool_page)
    is_measurement = bool(
        item is not None
        and getattr(tool_page.algo, "is_measurement_tool", lambda _code: False)(item.algorithm_code)
    )
    frame.setVisible(is_measurement)
    if not is_measurement:
        return

    params = dict(getattr(item, "params", {}) or {})
    algorithm = str(tool_page.algo.resolve_tool_algorithm(item.algorithm_code) or "").strip()
    is_find_line = algorithm == "find_line"
    is_line_distance = _is_line_distance_algorithm(algorithm)
    unit = str(params.get("limit_unit", "") or "").strip().lower()
    if unit not in {"px", "mm"}:
        unit = "mm" if ("lower_limit_mm" in params or "upper_limit_mm" in params) else "px"
    lower = _optional_param_float(params, "lower_limit", f"lower_limit_{unit}")
    upper = _optional_param_float(params, "upper_limit", f"upper_limit_{unit}")
    pixel_size = _optional_param_float(params, "pixel_size_mm") or 0.0
    line_a_direction = _line_direction(params, "line" if is_find_line else "line_a", "left_right")
    line_b_direction = _line_direction(params, "line_b", "right_left")
    line_a = _line_params(params, "line" if is_find_line else "line_a")
    line_b = _line_params(params, "line_b")
    polarity = str(line_a.get("polarity", line_b.get("polarity", "any")) or "any").strip()
    if polarity not in {"any", "dark_to_bright", "bright_to_dark"}:
        polarity = "any"
    edge_threshold = _optional_param_float(line_a, "edge_threshold")
    if edge_threshold is None:
        edge_threshold = _optional_param_float(line_b, "edge_threshold")
    if edge_threshold is None:
        edge_threshold = 10.0
    scan_step = int(_optional_param_float(line_a, "scan_step") or _optional_param_float(line_b, "scan_step") or 2)
    min_points = int(_optional_param_float(line_a, "min_points") or _optional_param_float(line_b, "min_points") or 10)
    line_options = _line_item_options(tool_page, item)

    tool_page._measurement_params_loading = True
    try:
        _populate_line_tool_combo(tool_page.cmb_measurement_line_a_tool, line_options)
        _populate_line_tool_combo(tool_page.cmb_measurement_line_b_tool, line_options)
        _set_combo_current_data(tool_page.cmb_measurement_line_a_tool, params.get("line_a_item_id", ""))
        _set_combo_current_data(tool_page.cmb_measurement_line_b_tool, params.get("line_b_item_id", ""))
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_a_tool, is_line_distance)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_b_tool, is_line_distance)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_a_direction, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_b_direction, False)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_polarity, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_edge_threshold, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_scan_step, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_min_points, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_lower, is_line_distance)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_upper, is_line_distance)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_unit, is_line_distance)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_pixel_size, is_line_distance)
        tool_page.cmb_measurement_line_a_tool.setEnabled(is_line_distance)
        tool_page.cmb_measurement_line_b_tool.setEnabled(is_line_distance)
        _set_combo_current_data(tool_page.cmb_measurement_line_a_direction, line_a_direction)
        _set_combo_current_data(tool_page.cmb_measurement_line_b_direction, line_b_direction)
        tool_page.cmb_measurement_line_a_direction.setEnabled(is_find_line)
        tool_page.cmb_measurement_line_b_direction.setEnabled(not is_find_line and not is_line_distance)
        _set_combo_current_data(tool_page.cmb_measurement_polarity, polarity)
        tool_page.cmb_measurement_polarity.setEnabled(is_find_line)
        tool_page.spin_measurement_edge_threshold.setValue(float(edge_threshold))
        tool_page.spin_measurement_edge_threshold.setEnabled(is_find_line)
        tool_page.spin_measurement_scan_step.setValue(max(1, int(scan_step)))
        tool_page.spin_measurement_scan_step.setEnabled(is_find_line)
        tool_page.spin_measurement_min_points.setValue(max(2, int(min_points)))
        tool_page.spin_measurement_min_points.setEnabled(is_find_line)
        tool_page.cmb_measurement_unit.setCurrentText(unit)
        tool_page.chk_measurement_lower.setChecked(lower is not None)
        tool_page.chk_measurement_upper.setChecked(upper is not None)
        tool_page.spin_measurement_lower.setEnabled(is_line_distance and lower is not None)
        tool_page.spin_measurement_upper.setEnabled(is_line_distance and upper is not None)
        tool_page.spin_measurement_lower.setValue(float(lower or 0.0))
        tool_page.spin_measurement_upper.setValue(float(upper or 0.0))
        tool_page.spin_measurement_pixel_size.setValue(float(pixel_size))
    finally:
        tool_page._measurement_params_loading = False


def _on_measurement_params_changed(tool_page, *args) -> None:
    if getattr(tool_page, "_measurement_params_loading", False):
        return
    item = _selected_inspection_item(tool_page)
    if item is None:
        return
    if not getattr(tool_page.algo, "is_measurement_tool", lambda _code: False)(item.algorithm_code):
        return
    params = dict(item.params or {})
    for key in (
        "lower_limit",
        "upper_limit",
        "lower_limit_px",
        "upper_limit_px",
        "lower_limit_mm",
        "upper_limit_mm",
    ):
        params.pop(key, None)
    unit = str(tool_page.cmb_measurement_unit.currentText() or "px").strip().lower()
    if unit not in {"px", "mm"}:
        unit = "px"
    algorithm = str(tool_page.algo.resolve_tool_algorithm(item.algorithm_code) or "").strip()
    is_find_line = algorithm == "find_line"
    is_line_distance = _is_line_distance_algorithm(algorithm)
    line_a = dict(params.get("line" if is_find_line else "line_a") or {})
    line_b = dict(params.get("line_b") or {})
    if is_find_line:
        line_a["direction"] = str(
            tool_page.cmb_measurement_line_a_direction.currentData()
            or tool_page.cmb_measurement_line_a_direction.currentText()
            or "left_right"
        ).strip()
        polarity = str(
            tool_page.cmb_measurement_polarity.currentData()
            or tool_page.cmb_measurement_polarity.currentText()
            or "any"
        ).strip()
        edge_threshold = float(tool_page.spin_measurement_edge_threshold.value())
        scan_step = int(tool_page.spin_measurement_scan_step.value())
        min_points = int(tool_page.spin_measurement_min_points.value())
        line_a["polarity"] = polarity
        line_a["edge_threshold"] = edge_threshold
        line_a["scan_step"] = scan_step
        line_a["min_points"] = min_points
        params["line"] = line_a
        params.pop("line_a", None)
        params.pop("line_b", None)
        params.pop("line_a_item_id", None)
        params.pop("line_b_item_id", None)
        params.pop("limit_unit", None)
        params.pop("pixel_size_mm", None)
    elif is_line_distance:
        params["line_a_item_id"] = str(tool_page.cmb_measurement_line_a_tool.currentData() or "").strip()
        params["line_b_item_id"] = str(tool_page.cmb_measurement_line_b_tool.currentData() or "").strip()
        params.pop("line", None)
        params.pop("line_a", None)
        params.pop("line_b", None)
    else:
        line_a["direction"] = str(
            tool_page.cmb_measurement_line_a_direction.currentData()
            or tool_page.cmb_measurement_line_a_direction.currentText()
            or "left_right"
        ).strip()
        line_b["direction"] = str(
            tool_page.cmb_measurement_line_b_direction.currentData()
            or tool_page.cmb_measurement_line_b_direction.currentText()
            or "right_left"
        ).strip()
        polarity = str(
            tool_page.cmb_measurement_polarity.currentData()
            or tool_page.cmb_measurement_polarity.currentText()
            or "any"
        ).strip()
        edge_threshold = float(tool_page.spin_measurement_edge_threshold.value())
        scan_step = int(tool_page.spin_measurement_scan_step.value())
        min_points = int(tool_page.spin_measurement_min_points.value())
        for line in (line_a, line_b):
            line["polarity"] = polarity
            line["edge_threshold"] = edge_threshold
            line["scan_step"] = scan_step
            line["min_points"] = min_points
        params["line_a"] = line_a
        params["line_b"] = line_b
    if is_line_distance:
        params["limit_unit"] = unit
        pixel_size = float(tool_page.spin_measurement_pixel_size.value())
        if pixel_size > 0.0:
            params["pixel_size_mm"] = pixel_size
        else:
            params.pop("pixel_size_mm", None)
        if tool_page.chk_measurement_lower.isChecked():
            params["lower_limit"] = float(tool_page.spin_measurement_lower.value())
        if tool_page.chk_measurement_upper.isChecked():
            params["upper_limit"] = float(tool_page.spin_measurement_upper.value())
    else:
        params.pop("limit_unit", None)
        params.pop("pixel_size_mm", None)
    item.params = params
    tool_page.spin_measurement_lower.setEnabled(is_line_distance and tool_page.chk_measurement_lower.isChecked())
    tool_page.spin_measurement_upper.setEnabled(is_line_distance and tool_page.chk_measurement_upper.isChecked())
    _persist_inspection_items(tool_page)
    _update_learning_backbone_hint(tool_page)


def _format_timestamp(path: str) -> str:
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return "-"
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _inspection_item_status(tool_page, inspection_item):
    if not getattr(inspection_item, "enabled", True):
        return tr("debug.status.disabled"), "Current tool is disabled and will not join training or runtime.", "#8a8a8a"

    if tool_page.algo.is_learning_tool(inspection_item.algorithm_code):
        backbone = tool_page.algo.current_learning_backbone()
        if not backbone:
            return tr("debug.status.not_selected"), "Select a subtype for the learning tool first.", "#d98c8c"
        model_path = tool_page.algo.embedding_model_path(
            backbone,
            tool_page.session.product_dir,
            model_key=inspection_item.model_key,
        )
        if os.path.exists(model_path):
            tooltip = tr(
                "debug.tooltip.learning_trained",
                model=os.path.basename(model_path),
                backbone=tool_page.algo.algorithm_display_name(backbone) or backbone,
                time=_format_timestamp(model_path),
            )
            return tr("debug.status.trained"), tooltip, "#79d279"
        legacy_path = next(
            (
                path
                for path in tool_page.algo.embedding_model_storage_paths(backbone, tool_page.session.product_dir)
                if path != model_path and os.path.exists(path)
            ),
            "",
        )
        if legacy_path:
            tooltip = tr(
                "debug.tooltip.learning_trained_legacy",
                model=os.path.basename(legacy_path),
                backbone=tool_page.algo.algorithm_display_name(backbone) or backbone,
                time=_format_timestamp(legacy_path),
            )
            return tr("debug.status.trained_legacy"), tooltip, "#cfc76a"
        tooltip = tr(
            "debug.tooltip.learning_untrained",
            model=os.path.basename(model_path),
            backbone=tool_page.algo.algorithm_display_name(backbone) or backbone,
        )
        return tr("debug.status.untrained"), tooltip, "#d98c8c"

    if getattr(tool_page.algo, "is_measurement_tool", lambda _code: False)(inspection_item.algorithm_code):
        algorithm = tool_page.algo.resolve_tool_algorithm(inspection_item.algorithm_code)
        if _is_line_distance_algorithm(algorithm):
            params = dict(getattr(inspection_item, "params", {}) or {})
            line_a = str(params.get("line_a_item_id", "") or "").strip() or "-"
            line_b = str(params.get("line_b_item_id", "") or "").strip() or "-"
            ready = line_a != "-" and line_b != "-" and line_a != line_b
            tooltip = (
                f"Algorithm: {tool_page.algo.algorithm_display_name(algorithm) or algorithm}\n"
                f"Measures distance from {line_a} to {line_b} and judges OK/NG from lower/upper limits."
            )
            return (f"{line_a} -> {line_b}" if ready else "Select lines"), tooltip, "#79d279" if ready else "#d98c8c"
        tooltip = (
            f"Algorithm: {tool_page.algo.algorithm_display_name(algorithm) or algorithm}\n"
            "Finds one fitted line inside the ROI for downstream distance measurement."
        )
        return "Ready", tooltip, "#79d279"

    algorithm = tool_page.algo.resolve_tool_algorithm(inspection_item.algorithm_code)
    model_dict = tool_page.algo.get_traditional_model_dict(algorithm, model_key=inspection_item.model_key)
    if isinstance(model_dict, dict):
        storage_key = tool_page.algo.traditional_model_storage_key(algorithm, model_key=inspection_item.model_key)
        actual_key = storage_key if storage_key in tool_page.algo.product_params.traditional_models else algorithm
        threshold = model_dict.get("threshold")
        ok_when = str(model_dict.get("ok_when", "")).strip() or "-"
        accuracy = model_dict.get("accuracy")
        detail_parts = [
            tr("debug.tooltip.algorithm", algorithm=tool_page.algo.algorithm_display_name(algorithm) or algorithm),
            tr("debug.tooltip.threshold", threshold=f"{float(threshold):.4f}" if threshold is not None else "-"),
            tr("debug.tooltip.rule", rule=ok_when),
        ]
        if accuracy is not None:
            detail_parts.append(tr("debug.tooltip.accuracy", accuracy=f"{float(accuracy):.4f}"))
        detail_parts.append(tr("debug.tooltip.storage_key", key=actual_key))
        status_text = tr("debug.status.calibrated") if actual_key == storage_key else tr("debug.status.calibrated_legacy")
        color = "#79d279" if actual_key == storage_key else "#cfc76a"
        return status_text, "\n".join(detail_parts), color

    tooltip = tr(
        "debug.tooltip.traditional_untrained",
        algorithm=tool_page.algo.algorithm_display_name(algorithm) or algorithm,
        key=tool_page.algo.traditional_model_storage_key(algorithm, model_key=inspection_item.model_key),
    )
    return tr("debug.status.uncalibrated"), tooltip, "#d98c8c"


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

    _update_delete_line_distance_button(tool_page)
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
        _update_delete_line_distance_button(tool_page)
        tool_page._update_measurement_params_panel()
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
    _update_delete_line_distance_button(tool_page)
    tool_page._update_measurement_params_panel()
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
    spec = tool_page.algo.tool_algorithm_spec(normalized)
    if spec is not None and not dict(tool_page.inspection_items[row].params or {}):
        tool_page.inspection_items[row].params = dict(spec.default_params or {})
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
    "_add_line_distance_tool",
    "_delete_selected_line_distance_tool",
    "_update_delete_line_distance_button",
    "_refresh_inspection_items_table",
    "_selected_inspection_item",
    "_selected_inspection_item_row",
    "_update_learning_backbone_hint",
    "_update_measurement_params_panel",
    "_on_measurement_params_changed",
]
