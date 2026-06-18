"""Auto-ROI and inspection-item helpers for ToolPage."""

from __future__ import annotations

import os
from typing import List, Tuple

import algorithms.proxy as qr_core
from ncc import locator as ncc_locator

from domain import (
    clearable_roi_labels,
    inspection_item_specs_from_line2dup_recipe,
    load_inspection_items,
    output_labels_from_line2dup_recipe,
    save_inspection_items,
    sync_items_with_labels,
)


def _line2dup_output_labels(tool_page, camera_role=None) -> List[str]:
    recipe = tool_page.line2dup_recipe_for_role(camera_role)
    return [str(label).strip() for label in output_labels_from_line2dup_recipe(recipe) if str(label).strip()]


def _ncc_output_labels(tool_page, camera_role=None) -> List[str]:
    role_getter = getattr(tool_page, "current_camera_role", None)
    role = str(role_getter() if callable(role_getter) and camera_role is None else camera_role or "cam1").strip() or "cam1"
    if not ncc_locator.model_is_ready(tool_page.session.product_dir, role):
        return ["roi"]
    labels = [
        str(label).strip()
        for label in ncc_locator.output_labels_for_product(tool_page.session.product_dir, role)
        if str(label).strip()
    ]
    return labels or ["roi"]


def _current_loc_output_labels(tool_page, camera_role=None) -> List[str]:
    if tool_page.loc_method == "line2dup":
        return tool_page._line2dup_output_labels(camera_role)
    if tool_page.loc_method == "ncc":
        ncc_getter = getattr(tool_page, "_ncc_output_labels", None)
        if callable(ncc_getter):
            return ncc_getter(camera_role)
        return _ncc_output_labels(tool_page, camera_role)
    return ["roi"]


def _inspection_item_labels(tool_page) -> List[str]:
    return [str(item.roi_label).strip() for item in tool_page.inspection_items if str(item.roi_label).strip()]


def _task_groups_from_display_names(display_names_by_label: dict[str, str]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for label, name in dict(display_names_by_label or {}).items():
        roi_label = str(label or "").strip()
        group_name = str(name or "").strip()
        if not roi_label or not group_name or group_name == roi_label:
            continue
        groups[roi_label] = group_name
    return groups


def _reload_inspection_items(tool_page) -> None:
    path = tool_page.session.inspection_items_path
    current_role_getter = getattr(tool_page, "current_camera_role", None)
    current_role = str(current_role_getter() if callable(current_role_getter) else "cam1").strip() or "cam1"
    configured_roles_getter = getattr(tool_page, "configured_camera_roles", None)
    allowed_roles = [
        str(role).strip()
        for role in (configured_roles_getter() if callable(configured_roles_getter) else ["cam1"])
        if str(role).strip()
    ]
    if not allowed_roles:
        allowed_roles = ["cam1"]
    if current_role not in set(allowed_roles):
        current_role = allowed_roles[0]
    if tool_page.loc_method == "line2dup":
        try:
            active_line2dup_recipe = tool_page.line2dup_recipe_for_role(current_role, force_reload=True)
        except Exception:
            active_line2dup_recipe = None
        tool_page.line2dup_recipe = active_line2dup_recipe
        labels = [
            str(label).strip()
            for label in output_labels_from_line2dup_recipe(active_line2dup_recipe)
            if str(label).strip()
        ]
    else:
        active_line2dup_recipe = None
        labels = tool_page._current_loc_output_labels(current_role)
    display_names_by_label = {}
    if tool_page.loc_method == "line2dup":
        specs = inspection_item_specs_from_line2dup_recipe(active_line2dup_recipe)
        display_names_by_label = {
            str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
            for spec in specs
            if str(spec.get("roi_label", "")).strip()
        }
    elif tool_page.loc_method == "ncc" and ncc_locator.model_is_ready(tool_page.session.product_dir, current_role):
        display_names_by_label = ncc_locator.display_names_by_label_for_product(
            tool_page.session.product_dir,
            current_role,
        )
    existing_items = [
        item
        for item in load_inspection_items(path)
        if str(getattr(item, "camera_id", "") or "").strip() in set(allowed_roles)
    ]
    current_role_items = [
        item for item in existing_items
        if str(getattr(item, "camera_id", "") or "").strip() == current_role
    ]
    other_role_items = [
        item for item in existing_items
        if str(getattr(item, "camera_id", "") or "").strip() != current_role
    ]
    if tool_page.loc_method == "line2dup" and tool_page.line2dup_recipe is None and not current_role_items:
        synced_current_role_items = []
    elif tool_page.loc_method == "ncc" and not ncc_locator.model_is_ready(tool_page.session.product_dir, current_role) and not current_role_items:
        synced_current_role_items = []
    else:
        synced_current_role_items = sync_items_with_labels(
            current_role_items,
            labels,
            default_camera_id=current_role,
            display_names_by_label=display_names_by_label,
            task_groups_by_label=_task_groups_from_display_names(display_names_by_label),
        )
    tool_page.inspection_items = other_role_items + synced_current_role_items
    save_inspection_items(tool_page.inspection_items, path)
    invalidate_state_cache = getattr(tool_page, "_invalidate_sample_annotation_state_cache", None)
    if callable(invalidate_state_cache):
        invalidate_state_cache()
    tool_page._refresh_inspection_items_table()
    tool_page.inspectionItemsChanged.emit()


def _missing_roi_files(tool_page, paths: List[str], camera_role=None) -> List[str]:
    missing: List[str] = []
    labels = tool_page._current_loc_output_labels(camera_role)
    labels = [str(label).strip() for label in labels if str(label).strip()]
    for p in paths:
        j = qr_core.labelme_json_of_image(p)
        if not os.path.exists(j):
            missing.append(p)
            continue
        try:
            existing = {
                str(shape.get("label", "")).strip()
                for shape in qr_core.list_shapes_from_labelme(j)
                if isinstance(shape, dict)
            }
        except Exception:
            missing.append(p)
            continue
        if any(label not in existing for label in labels):
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
    current_labels = tool_page._current_loc_output_labels(camera_role)
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
