"""Auto-ROI and inspection-item helpers for ToolPage."""

from __future__ import annotations

import os
from typing import List, Tuple

import algorithms.proxy as qr_core

from domain import (
    clearable_roi_labels,
    inspection_item_specs_from_shape_recipe,
    load_inspection_items,
    output_labels_from_shape_recipe,
    save_inspection_items,
    sync_items_with_labels,
)


def _shape_output_labels(tool_page, camera_role=None) -> List[str]:
    recipe = tool_page.shape_recipe_for_role(camera_role)
    return [str(label).strip() for label in output_labels_from_shape_recipe(recipe) if str(label).strip()]


def _inspection_item_labels(tool_page) -> List[str]:
    return [str(item.roi_label).strip() for item in tool_page.inspection_items if str(item.roi_label).strip()]


def _reload_inspection_items(tool_page) -> None:
    path = tool_page.session.inspection_items_path
    labels = tool_page._shape_output_labels() if tool_page.loc_method == "shape" else ["roi"]
    current_role_getter = getattr(tool_page, "current_camera_role", None)
    current_role = str(current_role_getter() if callable(current_role_getter) else "cam1").strip() or "cam1"
    display_names_by_label = {}
    if tool_page.loc_method == "shape":
        specs = inspection_item_specs_from_shape_recipe(tool_page.shape_recipe)
        display_names_by_label = {
            str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
            for spec in specs
            if str(spec.get("roi_label", "")).strip()
        }
    existing_items = load_inspection_items(path)
    current_role_items = [
        item for item in existing_items
        if str(getattr(item, "camera_id", "") or "").strip() == current_role
    ]
    other_role_items = [
        item for item in existing_items
        if str(getattr(item, "camera_id", "") or "").strip() != current_role
    ]
    if tool_page.shape_recipe is None and not current_role_items:
        synced_current_role_items = []
    else:
        synced_current_role_items = sync_items_with_labels(
            current_role_items,
            labels,
            default_camera_id=current_role,
            display_names_by_label=display_names_by_label,
        )
    tool_page.inspection_items = other_role_items + synced_current_role_items
    save_inspection_items(tool_page.inspection_items, path)
    tool_page._refresh_inspection_items_table()
    tool_page.inspectionItemsChanged.emit()


def _missing_roi_files(tool_page, paths: List[str], camera_role=None) -> List[str]:
    missing: List[str] = []
    labels = tool_page._shape_output_labels(camera_role) if tool_page.loc_method == "shape" else ["roi"]
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


def _clear_roi_labels_for_paths(tool_page, paths: List[str], camera_role=None) -> Tuple[List[str], str]:
    current_labels = tool_page._shape_output_labels(camera_role) if tool_page.loc_method == "shape" else ["roi"]
    prefer_stale_only = bool(
        tool_page.loc_method == "shape"
        and getattr(tool_page, "chk_only_missing", None) is not None
        and tool_page.chk_only_missing.isChecked()
    )
    return clearable_roi_labels(
        current_labels,
        tool_page._existing_roi_like_labels(paths),
        prefer_stale_only=prefer_stale_only,
    )



