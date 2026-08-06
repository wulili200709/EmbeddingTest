"""Measurement tool option and parameter helpers."""

from __future__ import annotations

from algorithms.measurement import CENTER_DISTANCE_ALGORITHM
from common.algorithm_codes import normalize_tool_algorithm_code
from ui.debug.tool_page.measurement_algorithms import (
    is_bright_block_center_algorithm as _is_bright_block_center_algorithm,
    is_find_line_algorithm as _is_find_line_algorithm,
    is_line_distance_algorithm as _is_line_distance_algorithm,
)
from ui.i18n import tr

def _line_distance_should_be_center_distance(
    item,
    *,
    line_options: list[tuple[str, str]],
    center_options: list[tuple[str, str]],
) -> bool:
    algorithm = normalize_tool_algorithm_code(getattr(item, "algorithm_code", ""))
    if not _is_line_distance_algorithm(algorithm):
        return False
    params = dict(getattr(item, "params", {}) or {})
    has_line_refs = bool(
        str(params.get("line_a_item_id", "") or "").strip()
        or str(params.get("line_b_item_id", "") or "").strip()
    )
    return not has_line_refs and not line_options and len(center_options) >= 2


def _convert_line_distance_to_center_distance(
    tool_page,
    item,
    *,
    center_options: list[tuple[str, str]],
) -> None:
    from ui.debug.tool_page.tool_config import (
        _persist_inspection_items,
        _refresh_inspection_items_table,
    )

    params = dict(getattr(item, "params", {}) or {})
    center_ids = [item_id for _display, item_id in center_options]
    unit = str(params.get("limit_unit", "") or "").strip().lower()
    if unit not in {"px", "mm"}:
        unit = "px"
    converted_params = {
        "center_a_item_id": center_ids[0] if len(center_ids) >= 1 else "",
        "center_b_item_id": center_ids[1] if len(center_ids) >= 2 else "",
        "distance_mode": str(params.get("distance_mode", "vertical") or "vertical").strip() or "vertical",
        "limit_unit": unit,
    }
    pixel_size = _optional_param_float(params, "pixel_size_mm")
    if pixel_size is not None and pixel_size > 0.0:
        converted_params["pixel_size_mm"] = pixel_size
    lower = _optional_param_float(params, "lower_limit", f"lower_limit_{unit}")
    upper = _optional_param_float(params, "upper_limit", f"upper_limit_{unit}")
    if lower is not None:
        converted_params["lower_limit"] = lower
    if upper is not None:
        converted_params["upper_limit"] = upper
    raw_name = str(getattr(item, "display_name", "") or "").strip()
    if raw_name in {"", "Line Distance", "line_distance", tr("debug.algorithm.line_distance")}:
        item.display_name = "Center Distance"
    item.algorithm_code = CENTER_DISTANCE_ALGORITHM
    item.roi_label = ""
    item.params = converted_params
    _persist_inspection_items(tool_page)
    _refresh_inspection_items_table(tool_page)

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
    from ui.debug.tool_page.tool_config import _current_camera_role

    current_role = str(getattr(selected_item, "camera_id", "") or _current_camera_role(tool_page)).strip() or "cam1"
    current_id = str(getattr(selected_item, "item_id", "") or "").strip()
    options: list[tuple[str, str]] = []
    for item in list(getattr(tool_page, "inspection_items", []) or []):
        if str(getattr(item, "camera_id", "") or "").strip() != current_role:
            continue
        item_id = str(getattr(item, "item_id", "") or "").strip()
        if not item_id or item_id == current_id:
            continue
        algorithm = str(
            tool_page.algo.resolve_tool_algorithm(
                getattr(item, "algorithm_code", ""),
                getattr(item, "camera_id", current_role),
            )
            or ""
        ).strip()
        if not _is_find_line_algorithm(algorithm):
            continue
        display = str(getattr(item, "display_name", "") or getattr(item, "roi_label", "") or item_id).strip()
        options.append((display, item_id))
    return options


def _center_item_options(tool_page, selected_item) -> list[tuple[str, str]]:
    from ui.debug.tool_page.tool_config import _current_camera_role

    current_role = str(getattr(selected_item, "camera_id", "") or _current_camera_role(tool_page)).strip() or "cam1"
    current_id = str(getattr(selected_item, "item_id", "") or "").strip()
    options: list[tuple[str, str]] = []
    for item in list(getattr(tool_page, "inspection_items", []) or []):
        if str(getattr(item, "camera_id", "") or "").strip() != current_role:
            continue
        item_id = str(getattr(item, "item_id", "") or "").strip()
        if not item_id or item_id == current_id:
            continue
        algorithm = str(
            tool_page.algo.resolve_tool_algorithm(
                getattr(item, "algorithm_code", ""),
                getattr(item, "camera_id", current_role),
            )
            or ""
        ).strip()
        if not _is_bright_block_center_algorithm(algorithm):
            continue
        display = str(getattr(item, "display_name", "") or getattr(item, "roi_label", "") or item_id).strip()
        options.append((display, item_id))
    return options


