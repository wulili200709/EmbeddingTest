"""Auto-ROI and inspection-item helpers for ToolPage."""

from __future__ import annotations

import os
from typing import List, Tuple

import algorithms.proxy as qr_core

from domain import (
    clearable_roi_labels,
    inspection_item_specs_from_line2dup_recipe,
    load_inspection_items,
    output_labels_from_line2dup_recipe,
    save_inspection_items,
    sync_items_with_labels,
)


def _line2dup_output_labels(tool_page) -> List[str]:
    recipe = tool_page.line2dup_recipe
    return [str(label).strip() for label in output_labels_from_line2dup_recipe(recipe) if str(label).strip()]


def _inspection_item_labels(tool_page) -> List[str]:
    return [str(item.roi_label).strip() for item in tool_page.inspection_items if str(item.roi_label).strip()]


def _reload_inspection_items(tool_page) -> None:
    path = tool_page.session.inspection_items_path
    labels = tool_page._line2dup_output_labels() if tool_page.loc_method == "line2dup" else ["roi"]
    display_names_by_label = {}
    if tool_page.loc_method == "line2dup":
        specs = inspection_item_specs_from_line2dup_recipe(tool_page.line2dup_recipe)
        display_names_by_label = {
            str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
            for spec in specs
            if str(spec.get("roi_label", "")).strip()
        }
    tool_page.inspection_items = sync_items_with_labels(
        load_inspection_items(path),
        labels,
        display_names_by_label=display_names_by_label,
    )
    save_inspection_items(tool_page.inspection_items, path)
    tool_page._refresh_inspection_items_table()
    tool_page.inspectionItemsChanged.emit()


def _missing_roi_files(tool_page, paths: List[str]) -> List[str]:
    missing: List[str] = []
    labels = tool_page._line2dup_output_labels() if tool_page.loc_method == "line2dup" else ["roi"]
    for p in paths:
        j = qr_core.labelme_json_of_image(p)
        if not os.path.exists(j):
            missing.append(p)
            continue
        if any(qr_core.read_shape_from_labelme(j, label) is None for label in labels):
            missing.append(p)
    return missing


def _existing_roi_like_labels(tool_page, paths: List[str]) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()
    for path in paths:
        jpath = qr_core.labelme_json_of_image(path)
        if not os.path.exists(jpath):
            continue
        try:
            shapes = qr_core.list_shapes_from_labelme(jpath)
        except Exception:
            continue
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label", "")).strip()
            if not label or label in {"anchor", "anchor_mask"} or label in seen:
                continue
            labels.append(label)
            seen.add(label)
    return labels


def _clear_roi_labels_for_paths(tool_page, paths: List[str]) -> Tuple[List[str], str]:
    current_labels = tool_page._line2dup_output_labels() if tool_page.loc_method == "line2dup" else ["roi"]
    prefer_stale_only = bool(
        tool_page.loc_method == "line2dup"
        and getattr(tool_page, "chk_only_missing", None) is not None
        and tool_page.chk_only_missing.isChecked()
    )
    return clearable_roi_labels(
        current_labels,
        tool_page._existing_roi_like_labels(paths),
        prefer_stale_only=prefer_stale_only,
    )
