"""Status emission helpers for RuntimeController."""

from __future__ import annotations

import json

from domain import build_pending_result, recipe_name_from_path

_RUN_STATE_ZH_FOR_STATUS = {
    "WaitingTrigger": "等待触发",
    "ReleasedPendingConsume": "已放行，待消耗",
    "CapturingCam1": "采集中(相机1)",
    "CapturingCam2": "采集中(相机2)",
    "Inspecting": "检测中",
    "Aggregating": "汇总结论",
    "CompletedOk": "本轮完成 OK",
    "CompletedNg": "本轮 NG",
    "LockedByNg": "NG 锁定",
    "Error": "运行异常",
    "Unavailable": "服务不可用",
}


def _update_status(runtime, message=None) -> None:
    runtime._emit_runtime_context()
    if runtime._import_error is not None:
        runtime.runtimeStateChanged.emit("Unavailable")
        runtime.permissionStatusChanged.emit("-")
        runtime.connectionStatusChanged.emit("服务导入失败")
        runtime.towerLightStatusChanged.emit("-")
        runtime.statusMessageChanged.emit(
            message or f"运行服务不可用: {runtime._import_error}"
        )
        runtime.recordPathChanged.emit("-")
        return

    state_text = "WaitingTrigger"
    if runtime._scheduler is not None:
        state_value = runtime._scheduler.state
        state_text = state_value.value if hasattr(state_value, "value") else str(state_value)
    runtime.runtimeStateChanged.emit(state_text)

    if runtime._permission_manager is not None:
        release_status = runtime._permission_manager.status()
        if release_status.is_locked:
            permission_text = "NG锁定"
        elif release_status.has_pending_release:
            permission_text = "已放行，待消耗"
        elif release_status.release_consumed:
            permission_text = "已消耗一次放行"
        else:
            permission_text = "未锁定"
    else:
        permission_text = "未初始化"
    runtime.permissionStatusChanged.emit(permission_text)

    roles = runtime._connected_roles()
    connection_text = "已连接: " + ", ".join(roles) if roles else "未连接相机"
    runtime.connectionStatusChanged.emit(connection_text)

    tower_state = getattr(runtime._tower_light_controller, "state", "waiting")
    tower_text_map = {
        "waiting": "蓝灯等待",
        "inspecting": "检测中",
        "ok": "绿灯结果",
        "ng": "红灯结果",
    }
    runtime.towerLightStatusChanged.emit(tower_text_map.get(tower_state, str(tower_state)))
    runtime.statusMessageChanged.emit(message or _RUN_STATE_ZH_FOR_STATUS.get(state_text, state_text))
    runtime.recordPathChanged.emit(runtime._last_record_path or "-")


def _connected_roles(runtime) -> list[str]:
    roles: list[str] = []
    if runtime._frame_grab_service is not None:
        try:
            roles = [str(role) for role in runtime._frame_grab_service.roles()]
        except Exception:
            roles = []
    return roles


def _current_item_signature(runtime) -> list[tuple[str, str, str, str, str, bool, str]]:
    return [
        (
            str(item.item_id),
            str(item.display_name),
            str(item.camera_id),
            str(item.roi_label),
            str(getattr(item, "algorithm_code", "")),
            bool(item.enabled),
            json.dumps(dict(getattr(item, "params", {}) or {}), ensure_ascii=False, sort_keys=True),
        )
        for item in runtime._runtime_context.inspection_items
    ]


def _result_item_signature(runtime) -> list[tuple[str, str, str, str, str, bool, str]]:
    if runtime._last_runtime_result is None:
        return []
    return [
        (
            str(item.item_id),
            str(item.display_name),
            str(item.camera_id),
            str(item.roi_label),
            str(getattr(item, "algorithm_code", "")),
            bool(item.enabled),
            json.dumps(dict(getattr(item, "params", {}) or {}), ensure_ascii=False, sort_keys=True),
        )
        for item in runtime._last_runtime_result.item_results
    ]


def _runtime_result_is_stale(runtime) -> bool:
    if runtime._last_runtime_result is None:
        return True
    if str(runtime._last_runtime_result.product_name or "") != str(runtime._session.current_product or ""):
        return True
    if set(runtime._last_runtime_result.camera_results.keys()) != set(runtime._connected_roles()):
        return True
    return runtime._current_item_signature() != runtime._result_item_signature()


def _emit_runtime_context(runtime) -> None:
    if runtime._runtime_result_is_stale():
        runtime._last_runtime_result = runtime._build_pending_runtime_result(status="PENDING")
    runtime.productNameChanged.emit(runtime._session.current_product)
    runtime.activeCameraRolesChanged.emit(runtime._connected_roles())
    runtime.inspectionItemsChanged.emit(runtime._last_runtime_result.item_rows())
    runtime.cameraResultsChanged.emit(runtime._last_runtime_result.camera_result_map())
    runtime.durationChanged.emit(int(getattr(runtime._last_runtime_result, "duration_ms", 0) or 0))
    runtime.timingBreakdownChanged.emit(runtime._last_runtime_result.timing_breakdown())


def _build_pending_runtime_result(runtime, *, status: str):
    return build_pending_result(
        product_name=runtime._session.current_product,
        recipe_name=recipe_name_from_path(runtime._session.line2dup_recipe_path),
        items=runtime._runtime_context.inspection_items,
        active_roles=runtime._connected_roles(),
        status=status,
    )


def _current_runtime_state_text(runtime) -> str:
    if runtime._scheduler is None:
        return "Uninitialized"
    state_value = runtime._scheduler.state
    return state_value.value if hasattr(state_value, "value") else str(state_value)
