"""Auto-ROI and inspection-item helpers for ToolPage."""

from __future__ import annotations

import os
from typing import List, Tuple

from common import labelme_io
from application.auto_roi_service import missing_roi_files as service_missing_roi_files

from domain import (
    load_inspection_items,
    save_inspection_items,
    sync_items_with_labels,
)
from ncc import locator as ncc_locator
from shape.core.recipe_labels import (
    clearable_roi_labels,
    inspection_item_specs_from_shape_recipe,
    output_labels_from_shape_recipe,
)


def _shape_output_labels(tool_page, camera_role=None) -> List[str]:
    recipe = tool_page.shape_recipe_for_role(camera_role)
    return [str(label).strip() for label in output_labels_from_shape_recipe(recipe) if str(label).strip()]


def _ncc_output_labels(tool_page, camera_role=None) -> List[str]:
    role_getter = getattr(tool_page, "current_camera_role", None)
    role = str(camera_role or (role_getter() if callable(role_getter) else "cam1") or "cam1").strip()
    try:
        return [
            str(label).strip()
            for label in ncc_locator.output_labels_for_product(tool_page.session.product_dir, role)
            if str(label).strip()
        ]
    except Exception:
        return ["roi"]


def _loc_output_labels(tool_page, camera_role=None) -> List[str]:
    method_getter = getattr(tool_page, "loc_method_for_role", None)
    method = (
        method_getter(camera_role) if callable(method_getter) else getattr(tool_page, "loc_method", "shape")
    )
    if method == "shape":
        return tool_page._shape_output_labels(camera_role)
    if method == "ncc":
        return tool_page._ncc_output_labels(camera_role)
    return ["roi"]


def _inspection_item_labels(tool_page) -> List[str]:
    return [str(item.roi_label).strip() for item in tool_page.inspection_items if str(item.roi_label).strip()]


def _task_groups_from_display_names(display_names_by_label: dict[str, str]) -> dict[str, str]:
    """Keep the legacy convention where a reference ROI name is its shared task group."""
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
    method_getter = getattr(tool_page, "loc_method_for_role", None)
    method = method_getter(current_role) if callable(method_getter) else getattr(tool_page, "loc_method", "shape")
    labels = tool_page._loc_output_labels(current_role)
    display_names_by_label = {}
    model_ready = True
    if method == "shape":
        specs = inspection_item_specs_from_shape_recipe(tool_page.shape_recipe)
        display_names_by_label = {
            str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
            for spec in specs
            if str(spec.get("roi_label", "")).strip()
        }
        model_ready = tool_page.shape_recipe is not None
    elif method == "ncc":
        try:
            model_ready = ncc_locator.model_is_ready(tool_page.session.product_dir, current_role)
            display_names_by_label = ncc_locator.display_names_by_label_for_product(
                tool_page.session.product_dir,
                current_role,
            ) if model_ready else {}
        except Exception:
            model_ready = False
    existing_items = load_inspection_items(path)
    current_role_items = [
        item for item in existing_items
        if str(getattr(item, "camera_id", "") or "").strip() == current_role
    ]
    other_role_items = [
        item for item in existing_items
        if str(getattr(item, "camera_id", "") or "").strip() != current_role
    ]
    if not model_ready and not current_role_items:
        synced_current_role_items = []
    elif not model_ready:
        synced_current_role_items = current_role_items
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
    tool_page._refresh_inspection_items_table()
    tool_page.inspectionItemsChanged.emit()


def _missing_roi_files(tool_page, paths: List[str], camera_role=None) -> List[str]:
    labels = tool_page._loc_output_labels(camera_role)
    return service_missing_roi_files(paths, labels)


def _existing_roi_like_labels(tool_page, paths: List[str]) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()
    for path in paths:
        jpath = labelme_io.labelme_json_of_image(path)
        if not os.path.exists(jpath):
            continue
        try:
            shapes = labelme_io.list_shapes_from_labelme(jpath)
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
    current_labels = tool_page._loc_output_labels(camera_role)
    method_getter = getattr(tool_page, "loc_method_for_role", None)
    method = method_getter(camera_role) if callable(method_getter) else getattr(tool_page, "loc_method", "shape")
    prefer_stale_only = bool(
        method in {"shape", "ncc"}
        and getattr(tool_page, "chk_only_missing", None) is not None
        and tool_page.chk_only_missing.isChecked()
    )
    return clearable_roi_labels(
        current_labels,
        tool_page._existing_roi_like_labels(paths),
        prefer_stale_only=prefer_stale_only,
    )



