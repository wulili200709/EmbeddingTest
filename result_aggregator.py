"""
result_aggregator.py

把运行链路相机级结果，转换成 UI 可消费的：
  - item 级结果
  - camera 级结果
  - final 级结果

当前第一版限制：
  - InspectionRuntime 只返回相机级结果
  - 因此 item 级结果先继承所属相机结果
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping

from domain import (
    CameraRuntimeResult,
    InspectionItem,
    InspectionItemResult,
    RuntimeInspectionResult,
    build_task_id,
)


def build_pending_result(
    *,
    product_name: str,
    recipe_name: str,
    items: Iterable[InspectionItem],
    active_roles: Iterable[str],
    status: str = "PENDING",
) -> RuntimeInspectionResult:
    active_role_set = {str(role).strip() for role in active_roles if str(role).strip()}
    item_results: list[InspectionItemResult] = []
    camera_results: dict[str, CameraRuntimeResult] = {}

    for role in sorted(active_role_set):
        camera_results[role] = CameraRuntimeResult(camera_id=role, result="")

    for item in items:
        if not item.enabled:
            result = "DISABLED"
        elif item.camera_id not in active_role_set:
            result = "INACTIVE"
        else:
            result = status
        item_results.append(
            InspectionItemResult(
                item_id=item.item_id,
                display_name=item.display_name,
                camera_id=item.camera_id,
                roi_label=item.roi_label,
                enabled=item.enabled,
                result=result,
            )
        )

    return RuntimeInspectionResult(
        task_id=build_task_id("pending"),
        product_name=product_name,
        recipe_name=recipe_name,
        final_result="",
        camera_results=camera_results,
        item_results=item_results,
    )


def aggregate_runtime_outcome(
    *,
    product_name: str,
    recipe_name: str,
    items: Iterable[InspectionItem],
    active_roles: Iterable[str],
    camera_outcomes: Mapping[str, object],
    final_result: str,
    duration_ms: int,
    error_message: str = "",
    capture_paths: Mapping[str, str] | None = None,
    item_results_by_camera: Mapping[str, Iterable[InspectionItemResult]] | None = None,
) -> RuntimeInspectionResult:
    active_role_set = {str(role).strip() for role in active_roles if str(role).strip()}
    capture_paths = dict(capture_paths or {})
    item_results_by_camera = dict(item_results_by_camera or {})
    camera_results: dict[str, CameraRuntimeResult] = {}

    for role in sorted(active_role_set):
        outcome = camera_outcomes.get(role)
        result = str(getattr(outcome, "result", "") or "")
        detail = str(getattr(outcome, "message", "") or "")
        camera_results[role] = CameraRuntimeResult(
            camera_id=role,
            result=result or ("NG" if error_message else ""),
            detail=detail,
            image_path=str(capture_paths.get(role, "") or ""),
        )

    explicit_item_results: dict[str, InspectionItemResult] = {}
    for role, rows in item_results_by_camera.items():
        for row in rows or []:
            if isinstance(row, InspectionItemResult):
                explicit_item_results[row.item_id] = row

    item_results: list[InspectionItemResult] = []
    for item in items:
        explicit = explicit_item_results.get(item.item_id)
        if explicit is not None:
            item_results.append(explicit)
            continue
        if not item.enabled:
            item_result = "DISABLED"
            detail = ""
        elif item.camera_id not in active_role_set:
            item_result = "INACTIVE"
            detail = ""
        else:
            camera_result = camera_results.get(item.camera_id)
            item_result = (camera_result.result if camera_result else "") or ("NG" if error_message else "PENDING")
            detail = camera_result.detail if camera_result else ""
        item_results.append(
            InspectionItemResult(
                item_id=item.item_id,
                display_name=item.display_name,
                camera_id=item.camera_id,
                roi_label=item.roi_label,
                enabled=item.enabled,
                result=item_result,
                detail=detail,
            )
        )

    for item_result in item_results:
        camera_result = camera_results.get(item_result.camera_id)
        if camera_result is not None:
            camera_result.item_results.append(item_result)

    is_system_error = bool(error_message) or str(final_result).upper() in {"ERROR", "PRECHECK_FAILED"}
    return RuntimeInspectionResult(
        task_id=build_task_id("runtime"),
        product_name=product_name,
        recipe_name=recipe_name,
        final_result=str(final_result or ""),
        duration_ms=int(duration_ms),
        error_message=str(error_message or ""),
        is_system_error=is_system_error,
        camera_results=camera_results,
        item_results=item_results,
    )


def recipe_name_from_path(path: str) -> str:
    return os.path.basename(path) if path else ""


__all__ = [
    "aggregate_runtime_outcome",
    "build_pending_result",
    "recipe_name_from_path",
]
