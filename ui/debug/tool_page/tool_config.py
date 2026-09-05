"""Inspection-tool configuration helpers for ToolPage."""

from __future__ import annotations

from PySide6 import QtWidgets

from algorithms.measurement import (
    CENTER_DISTANCE_ALGORITHM,
    LINE_DISTANCE_ALGORITHMS,
    POINT_LINE_DISTANCE_ALGORITHM,
)
from common.algorithm_codes import normalize_tool_algorithm_code
from domain import InspectionItem, save_inspection_items
from ui.i18n import tr
from ui.debug.tool_page.inspection_item_status import _inspection_item_status
from ui.debug.tool_page.inspection_items_table import (
    _on_inspection_item_algorithm_changed,
    _on_inspection_item_camera_changed,
    _on_inspection_items_selection_changed,
    _on_inspection_items_table_item_changed,
    _refresh_inspection_items_table,
    _selected_inspection_item,
    _selected_inspection_item_row,
)
from ui.debug.tool_page.measurement_tool_config import (
    _on_measurement_params_changed,
    _update_measurement_params_panel,
)
from ui.debug.tool_page.measurement_tool_options import (
    _center_item_options,
    _convert_line_distance_to_center_distance,
    _line_distance_should_be_center_distance,
    _line_item_options,
    _optional_param_float,
    _point_item_options,
)
from ui.debug.tool_page.measurement_algorithms import (
    is_bright_block_center_algorithm as _is_bright_block_center_algorithm,
    is_center_distance_algorithm as _is_center_distance_algorithm,
    is_find_line_algorithm as _is_find_line_algorithm,
    is_line_distance_algorithm as _is_line_distance_algorithm,
    is_pin_tip_point_algorithm as _is_pin_tip_point_algorithm,
    is_point_line_distance_algorithm as _is_point_line_distance_algorithm,
    is_single_roi_distance_algorithm as _is_single_roi_distance_algorithm,
    public_algorithm_code as _public_algorithm_code,
)


def _current_camera_role(tool_page) -> str:
    getter = getattr(tool_page, "current_camera_role", None)
    if callable(getter):
        return str(getter() or "cam1").strip() or "cam1"
    return "cam1"


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


def _current_picker_algorithm(tool_page) -> str:
    getter = getattr(tool_page, "current_algorithm", None)
    raw = getter() if callable(getter) else ""
    normalized = normalize_tool_algorithm_code(raw)
    resolver = getattr(tool_page.algo, "resolve_tool_algorithm", None)
    if callable(resolver):
        return str(resolver(normalized) or normalized or "").strip()
    return str(normalized or "").strip()


def _add_line_distance_tool(tool_page) -> None:
    if not _require_tool_permission(tool_page, "inspection.edit_items", "新增检测项"):
        return
    camera_role = _current_camera_role(tool_page)
    selected_item = _selected_inspection_item(tool_page)
    selected_algorithm = (
        str(
            tool_page.algo.resolve_tool_algorithm(
                getattr(selected_item, "algorithm_code", ""),
                getattr(selected_item, "camera_id", camera_role),
            )
            or ""
        ).strip()
        if selected_item is not None
        else ""
    )
    picker_algorithm = _current_picker_algorithm(tool_page)
    center_options = [
        item_id
        for _display, item_id in _center_item_options(
            tool_page,
            type("_Selected", (), {"camera_id": camera_role, "item_id": ""})(),
        )
    ]
    line_options = [
        item_id
        for _display, item_id in _line_item_options(
            tool_page,
            type("_Selected", (), {"camera_id": camera_role, "item_id": ""})(),
        )
    ]
    point_options = [
        item_id
        for _display, item_id in _point_item_options(
            tool_page,
            type("_Selected", (), {"camera_id": camera_role, "item_id": ""})(),
        )
    ]
    if (
        _is_pin_tip_point_algorithm(selected_algorithm)
        or _is_pin_tip_point_algorithm(picker_algorithm)
        or (len(line_options) < 2 and bool(line_options) and bool(point_options))
    ):
        selected_item_id = str(getattr(selected_item, "item_id", "") or "").strip()
        selected_point_id = selected_item_id if _is_pin_tip_point_algorithm(selected_algorithm) else ""
        selected_line_id = selected_item_id if _is_find_line_algorithm(selected_algorithm) else ""
        _add_point_line_distance_tool(
            tool_page,
            point_item_id=(selected_point_id or point_options[0]) if point_options else selected_point_id,
            line_item_id=(selected_line_id or line_options[0]) if line_options else selected_line_id,
        )
        return
    if (
        _is_bright_block_center_algorithm(selected_algorithm)
        or _is_bright_block_center_algorithm(picker_algorithm)
        or _is_center_distance_algorithm(selected_algorithm)
        or _is_center_distance_algorithm(picker_algorithm)
        or (len(line_options) < 2 and len(center_options) >= 2)
    ):
        _add_center_distance_tool(tool_page, center_options=center_options)
        return

    items_by_id = {
        str(getattr(item, "item_id", "") or "").strip(): item
        for item in list(getattr(tool_page, "inspection_items", []) or [])
    }

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
    _audit_tool_event(
        tool_page,
        module="检测项",
        action="新增检测项",
        target=item_id,
        after_value="line_distance",
    )
    _refresh_inspection_items_table(tool_page)
    table = getattr(tool_page, "inspection_items_table", None)
    if table is not None:
        for visible_row, actual_index in enumerate(getattr(tool_page, "_visible_inspection_item_indexes", []) or []):
            item = tool_page.inspection_items[actual_index]
            if str(getattr(item, "item_id", "") or "") == item_id:
                table.setCurrentCell(visible_row, 1)
                break


def _add_point_line_distance_tool(
    tool_page,
    *,
    point_item_id: str = "",
    line_item_id: str = "",
) -> None:
    camera_role = _current_camera_role(tool_page)
    item_id = _unique_item_id(tool_page, "point_line_distance")
    params = {
        "point_item_id": str(point_item_id or "").strip(),
        "line_item_id": str(line_item_id or "").strip(),
        "limit_unit": "px",
    }
    tool_page.inspection_items.append(
        InspectionItem(
            item_id=item_id,
            display_name="Point-Line Distance",
            camera_id=camera_role,
            roi_label="",
            algorithm_code=POINT_LINE_DISTANCE_ALGORITHM,
            enabled=True,
            params=params,
        )
    )
    _persist_inspection_items(tool_page)
    _audit_tool_event(
        tool_page,
        module="检测项",
        action="新增检测项",
        target=item_id,
        after_value=POINT_LINE_DISTANCE_ALGORITHM,
    )
    _refresh_inspection_items_table(tool_page)
    table = getattr(tool_page, "inspection_items_table", None)
    if table is not None:
        for visible_row, actual_index in enumerate(getattr(tool_page, "_visible_inspection_item_indexes", []) or []):
            item = tool_page.inspection_items[actual_index]
            if str(getattr(item, "item_id", "") or "") == item_id:
                table.setCurrentCell(visible_row, 1)
                break


def _add_center_distance_tool(tool_page, *, center_options: list[str] | None = None) -> None:
    if not _require_tool_permission(tool_page, "inspection.edit_items", "新增检测项"):
        return
    camera_role = _current_camera_role(tool_page)
    if center_options is None:
        center_options = [
            item_id
            for _display, item_id in _center_item_options(
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
    unit = str(selected_params.get("limit_unit", "") or "").strip().lower()
    if unit not in {"px", "mm"}:
        unit = "px"
    pixel_size = _optional_param_float(selected_params, "pixel_size_mm") or 0.0
    if pixel_size <= 0.0:
        for item_id_ref in center_options[:2]:
            pixel_size = _optional_param_float(dict(getattr(items_by_id.get(item_id_ref), "params", {}) or {}), "pixel_size_mm") or 0.0
            if pixel_size > 0.0:
                break
    item_id = _unique_item_id(tool_page, "center_distance")
    params = {
        "center_a_item_id": center_options[0] if len(center_options) >= 1 else "",
        "center_b_item_id": center_options[1] if len(center_options) >= 2 else "",
        "distance_mode": "vertical",
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
            display_name="Center Distance",
            camera_id=camera_role,
            roi_label="",
            algorithm_code=CENTER_DISTANCE_ALGORITHM,
            enabled=True,
            params=params,
        )
    )
    _persist_inspection_items(tool_page)
    _audit_tool_event(
        tool_page,
        module="检测项",
        action="新增检测项",
        target=item_id,
        after_value=CENTER_DISTANCE_ALGORITHM,
    )
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
        and (
            _is_line_distance_algorithm(normalize_tool_algorithm_code(getattr(item, "algorithm_code", "")))
            or _is_center_distance_algorithm(normalize_tool_algorithm_code(getattr(item, "algorithm_code", "")))
            or _is_point_line_distance_algorithm(normalize_tool_algorithm_code(getattr(item, "algorithm_code", "")))
        )
    )
    has_permission = getattr(tool_page.window(), "_has_permission", None)
    if callable(has_permission):
        can_delete = can_delete and bool(has_permission("inspection.edit_items"))
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
        if item_id.startswith("line_distance"):
            default_names.add(item_id)
        if raw_name in default_names:
            return tr(display_key)
    if _is_center_distance_algorithm(algorithm):
        default_names = {"", "Center Distance", "center_distance", tr("debug.algorithm.center_distance")}
        if item_id.startswith("center_distance"):
            default_names.add(item_id)
        if raw_name in default_names:
            return tr("debug.algorithm.center_distance")
    if _is_point_line_distance_algorithm(algorithm):
        default_names = {
            "",
            "Point-Line Distance",
            "point_line_distance",
            tr("debug.algorithm.point_line_distance"),
        }
        if item_id.startswith("point_line_distance"):
            default_names.add(item_id)
        if raw_name in default_names:
            return tr("debug.algorithm.point_line_distance")
    return raw_name


def _delete_selected_line_distance_tool(tool_page) -> None:
    if not _require_tool_permission(tool_page, "inspection.edit_items", "删除检测项"):
        return
    row = _selected_inspection_item_row(tool_page)
    if row < 0 or row >= len(getattr(tool_page, "inspection_items", []) or []):
        QtWidgets.QMessageBox.information(
            tool_page,
            tr("debug.measurement.delete_line_distance_tool"),
            tr("debug.measurement.delete_line_distance_select"),
        )
        return

    inspection_item = tool_page.inspection_items[row]
    if not (
        _is_line_distance_algorithm(normalize_tool_algorithm_code(getattr(inspection_item, "algorithm_code", "")))
        or _is_center_distance_algorithm(normalize_tool_algorithm_code(getattr(inspection_item, "algorithm_code", "")))
        or _is_point_line_distance_algorithm(normalize_tool_algorithm_code(getattr(inspection_item, "algorithm_code", "")))
    ):
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
    item_id = str(getattr(inspection_item, "item_id", "") or "")
    algorithm = str(getattr(inspection_item, "algorithm_code", "") or "")
    del tool_page.inspection_items[row]
    _persist_inspection_items(tool_page)
    _audit_tool_event(
        tool_page,
        module="检测项",
        action="删除检测项",
        target=item_id,
        before_value=algorithm,
    )
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
