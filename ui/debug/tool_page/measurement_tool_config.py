"""Measurement-tool parameter helpers for ToolPage."""

from __future__ import annotations

from PySide6 import QtWidgets

from algorithms.measurement import FIND_LINE_SUBPIX_ALGORITHM
from ui.i18n import tr
from ui.debug.tool_page.inspection_items_table import _selected_inspection_item
from ui.debug.tool_page.measurement_tool_options import (
    _center_item_options,
    _convert_line_distance_to_center_distance,
    _line_direction,
    _line_distance_should_be_center_distance,
    _line_item_options,
    _line_params,
    _optional_param_float,
)
from ui.debug.tool_page.measurement_algorithms import (
    is_bright_block_center_algorithm as _is_bright_block_center_algorithm,
    is_center_distance_algorithm as _is_center_distance_algorithm,
    is_find_line_algorithm as _is_find_line_algorithm,
    is_line_distance_algorithm as _is_line_distance_algorithm,
    is_single_roi_distance_algorithm as _is_single_roi_distance_algorithm,
)


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


def _set_measurement_row_label(tool_page, field_widget: QtWidgets.QWidget, label_text: str) -> None:
    frame = getattr(tool_page, "measurement_params_frame", None)
    layout = frame.layout() if frame is not None else None
    if isinstance(layout, QtWidgets.QFormLayout):
        label_widget = layout.labelForField(field_widget)
        if label_widget is not None:
            label_widget.setText(label_text)


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
    is_find_line = _is_find_line_algorithm(algorithm)
    is_line_distance = _is_line_distance_algorithm(algorithm)
    is_center_distance = _is_center_distance_algorithm(algorithm)
    is_single_roi_distance = _is_single_roi_distance_algorithm(algorithm)
    is_direct_distance = is_line_distance or is_center_distance or is_single_roi_distance
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
    center_options = _center_item_options(tool_page, item)
    if _line_distance_should_be_center_distance(
        item,
        line_options=line_options,
        center_options=center_options,
    ):
        _convert_line_distance_to_center_distance(tool_page, item, center_options=center_options)
        return
    pair_options = center_options if is_center_distance else line_options
    distance_mode = str(params.get("distance_mode", "vertical") or "vertical").strip().lower()
    if distance_mode not in {"vertical", "horizontal", "euclidean"}:
        distance_mode = "vertical"

    tool_page._measurement_params_loading = True
    try:
        _populate_line_tool_combo(tool_page.cmb_measurement_line_a_tool, pair_options)
        _populate_line_tool_combo(tool_page.cmb_measurement_line_b_tool, pair_options)
        _set_combo_current_data(
            tool_page.cmb_measurement_line_a_tool,
            params.get("center_a_item_id", "") if is_center_distance else params.get("line_a_item_id", ""),
        )
        _set_combo_current_data(
            tool_page.cmb_measurement_line_b_tool,
            params.get("center_b_item_id", "") if is_center_distance else params.get("line_b_item_id", ""),
        )
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_a_tool, is_line_distance or is_center_distance)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_b_tool, is_line_distance or is_center_distance)
        _set_measurement_row_label(
            tool_page,
            tool_page.cmb_measurement_line_a_tool,
            tr("debug.measurement.center_a_tool") if is_center_distance else tr("debug.measurement.line_a_tool"),
        )
        _set_measurement_row_label(
            tool_page,
            tool_page.cmb_measurement_line_b_tool,
            tr("debug.measurement.center_b_tool") if is_center_distance else tr("debug.measurement.line_b_tool"),
        )
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_distance_mode, is_center_distance)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_a_direction, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_line_b_direction, False)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_polarity, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_edge_threshold, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_scan_step, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_min_points, is_find_line)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_lower, is_direct_distance)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_upper, is_direct_distance)
        _set_measurement_row_visible(tool_page, tool_page.cmb_measurement_unit, is_direct_distance)
        _set_measurement_row_visible(tool_page, tool_page.spin_measurement_pixel_size, is_direct_distance)
        tool_page.cmb_measurement_line_a_tool.setEnabled(is_line_distance or is_center_distance)
        tool_page.cmb_measurement_line_b_tool.setEnabled(is_line_distance or is_center_distance)
        _set_combo_current_data(tool_page.cmb_measurement_distance_mode, distance_mode)
        tool_page.cmb_measurement_distance_mode.setEnabled(is_center_distance)
        _set_combo_current_data(tool_page.cmb_measurement_line_a_direction, line_a_direction)
        _set_combo_current_data(tool_page.cmb_measurement_line_b_direction, line_b_direction)
        tool_page.cmb_measurement_line_a_direction.setEnabled(is_find_line)
        tool_page.cmb_measurement_line_b_direction.setEnabled(
            not is_find_line and not is_line_distance and not is_center_distance and not is_single_roi_distance
        )
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
        tool_page.spin_measurement_lower.setEnabled(is_direct_distance and lower is not None)
        tool_page.spin_measurement_upper.setEnabled(is_direct_distance and upper is not None)
        tool_page.spin_measurement_lower.setValue(float(lower or 0.0))
        tool_page.spin_measurement_upper.setValue(float(upper or 0.0))
        tool_page.spin_measurement_pixel_size.setValue(float(pixel_size))
    finally:
        tool_page._measurement_params_loading = False


def _on_measurement_params_changed(tool_page, *args) -> None:
    from ui.debug.tool_page.tool_config import (
        _persist_inspection_items,
        _update_learning_backbone_hint,
    )

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
    is_find_line = _is_find_line_algorithm(algorithm)
    is_line_distance = _is_line_distance_algorithm(algorithm)
    is_center_distance = _is_center_distance_algorithm(algorithm)
    is_single_roi_distance = _is_single_roi_distance_algorithm(algorithm)
    is_direct_distance = is_line_distance or is_center_distance or is_single_roi_distance
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
        if algorithm == FIND_LINE_SUBPIX_ALGORITHM:
            line_a["edge_detector"] = "subpix_shen"
        else:
            line_a["edge_detector"] = "canny"
        params["line"] = line_a
        params.pop("line_a", None)
        params.pop("line_b", None)
        params.pop("line_a_item_id", None)
        params.pop("line_b_item_id", None)
        params.pop("center_a_item_id", None)
        params.pop("center_b_item_id", None)
        params.pop("distance_mode", None)
        params.pop("limit_unit", None)
        params.pop("pixel_size_mm", None)
    elif is_line_distance:
        params["line_a_item_id"] = str(tool_page.cmb_measurement_line_a_tool.currentData() or "").strip()
        params["line_b_item_id"] = str(tool_page.cmb_measurement_line_b_tool.currentData() or "").strip()
        params.pop("line", None)
        params.pop("line_a", None)
        params.pop("line_b", None)
        params.pop("center_a_item_id", None)
        params.pop("center_b_item_id", None)
    elif is_center_distance:
        distance_mode = str(
            tool_page.cmb_measurement_distance_mode.currentData()
            or tool_page.cmb_measurement_distance_mode.currentText()
            or "vertical"
        ).strip()
        if distance_mode not in {"vertical", "horizontal", "euclidean"}:
            distance_mode = "vertical"
        params["center_a_item_id"] = str(tool_page.cmb_measurement_line_a_tool.currentData() or "").strip()
        params["center_b_item_id"] = str(tool_page.cmb_measurement_line_b_tool.currentData() or "").strip()
        params["distance_mode"] = distance_mode
        params.pop("line", None)
        params.pop("line_a", None)
        params.pop("line_b", None)
        params.pop("line_a_item_id", None)
        params.pop("line_b_item_id", None)
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
    if is_direct_distance:
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
    tool_page.spin_measurement_lower.setEnabled(is_direct_distance and tool_page.chk_measurement_lower.isChecked())
    tool_page.spin_measurement_upper.setEnabled(is_direct_distance and tool_page.chk_measurement_upper.isChecked())
    _persist_inspection_items(tool_page)
    _update_learning_backbone_hint(tool_page)

