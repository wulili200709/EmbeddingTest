"""Execution-oriented RuntimeController helpers."""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from domain import aggregate_runtime_outcome, recipe_name_from_path
from common.algorithm_codes import learning_backbone_storage_code
from common.camera_roles import configured_camera_roles, camera_index_for_role

from .capture_policy import normalize_capture_retention_policy
from .capture_channels import (
    channels_for_roles,
    light_output_index,
    physical_connected_roles,
    required_runtime_roles,
)
from .preview_frame import RuntimePreviewFrame, build_runtime_preview_frame, export_runtime_preview_frame

_MIN_RUNTIME_CAPTURE_FREE_BYTES = 1 * 1024 * 1024 * 1024
_DATE_DIRECTORY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_INVALID_PATH_SEGMENT_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _recipe_path_for_role(runtime, role: str) -> str:
    getter = getattr(runtime._session, "shape_recipe_path_for_role", None)
    if callable(getter):
        try:
            return str(getter(role) or "")
        except Exception:
            return str(getattr(runtime._session, "shape_recipe_path", "") or "")
    return str(getattr(runtime._session, "shape_recipe_path", "") or "")


def _recipe_name_for_roles(runtime, roles) -> str:
    role_list = [str(role).strip() for role in roles if str(role).strip()]
    for role in role_list:
        path = _recipe_path_for_role(runtime, role)
        if path:
            return recipe_name_from_path(path)
    return recipe_name_from_path(str(getattr(runtime._session, "shape_recipe_path", "") or ""))


def _outcome_roles(outcome) -> tuple[str, ...]:
    camera_outcomes = getattr(outcome, "camera_outcomes", None)
    if not isinstance(camera_outcomes, dict):
        return ()
    roles = [
        str(role).strip()
        for role in camera_outcomes.keys()
        if str(role).strip()
    ]
    return tuple(sorted(set(roles)))


def _capture_export_roles(outcome, preview_frames: dict[str, object]) -> tuple[str, ...]:
    roles = set(_outcome_roles(outcome))
    if not roles:
        roles = {
            str(role).strip()
            for role, frame in dict(preview_frames or {}).items()
            if str(role).strip() and isinstance(frame, RuntimePreviewFrame)
        }
    return tuple(sorted(roles))


def _should_export_captures(runtime, final_result: str) -> bool:
    policy = normalize_capture_retention_policy(runtime._capture_retention_policy)
    return policy == "all" or str(final_result or "").strip().upper() == "NG"


def _capture_root_directory(runtime) -> Path:
    configured_dir = str(getattr(runtime, "_runtime_capture_dir", "") or "").strip()
    if configured_dir:
        return Path(configured_dir)
    product_dir = str(getattr(runtime._session, "product_dir", "") or "").strip()
    if product_dir:
        return Path(product_dir) / "runtime_capture"
    return Path(os.path.abspath(".")) / "runtime_capture"


def _sanitize_path_segment(text: str, *, default: str) -> str:
    sanitized = _INVALID_PATH_SEGMENT_CHARS_RE.sub("_", str(text or "").strip())
    sanitized = sanitized.strip(" .")
    return sanitized or default


def _capture_directory(runtime, target_dt: datetime | None = None) -> str:
    timestamp = target_dt or datetime.now()
    root_dir = _capture_root_directory(runtime)
    product_name = _sanitize_path_segment(
        str(getattr(runtime._session, "current_product", "") or ""),
        default="Default",
    )
    return str(root_dir / timestamp.strftime("%Y-%m-%d") / product_name)


def _disk_usage_anchor(path: Path) -> Path:
    current = Path(path)
    while not current.exists():
        parent = current.parent
        if parent == current:
            return Path(os.path.abspath("."))
        current = parent
    return current


def _log_capture_storage_event(runtime, message: str) -> None:
    try:
        runtime.logAppended.emit(f"[runtime-capture] {message}")
    except Exception:
        pass


def _ensure_capture_free_space(runtime, target_dir: str | Path) -> None:
    root_dir = _capture_root_directory(runtime)
    root_dir.mkdir(parents=True, exist_ok=True)
    target_date_dir_name = Path(target_dir).parent.name
    while shutil.disk_usage(str(_disk_usage_anchor(root_dir))).free < _MIN_RUNTIME_CAPTURE_FREE_BYTES:
        candidates = sorted(
            (
                path
                for path in root_dir.iterdir()
                if path.is_dir()
                and _DATE_DIRECTORY_RE.match(path.name)
                and path.name != target_date_dir_name
            ),
            key=lambda path: path.name,
        )
        if not candidates:
            free_bytes = shutil.disk_usage(str(_disk_usage_anchor(root_dir))).free
            raise RuntimeError(
                "runtime capture disk space below 1 GB and no older date folders can be removed "
                f"(root={root_dir}, free={free_bytes})"
            )
        oldest_dir = candidates[0]
        shutil.rmtree(oldest_dir)
        _log_capture_storage_event(runtime, f"deleted old capture folder: {oldest_dir}")


def _ensure_capture_free_space_for_root(runtime, target_dir: str | Path, *, root_dir: str | Path) -> None:
    target_root = Path(root_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    target_date_dir_name = Path(target_dir).parent.name
    while shutil.disk_usage(str(_disk_usage_anchor(target_root))).free < _MIN_RUNTIME_CAPTURE_FREE_BYTES:
        candidates = sorted(
            (
                path
                for path in target_root.iterdir()
                if path.is_dir()
                and _DATE_DIRECTORY_RE.match(path.name)
                and path.name != target_date_dir_name
            ),
            key=lambda path: path.name,
        )
        if not candidates:
            free_bytes = shutil.disk_usage(str(_disk_usage_anchor(target_root))).free
            raise RuntimeError(
                "runtime capture disk space below 1 GB and no older date folders can be removed "
                f"(root={target_root}, free={free_bytes})"
            )
        oldest_dir = candidates[0]
        shutil.rmtree(oldest_dir)
        _log_capture_storage_event(runtime, f"deleted old capture folder: {oldest_dir}")


def _record_path_for_service(record_service) -> str:
    if record_service is None:
        return ""
    try:
        return str(record_service.writer.file_path_for_date())
    except Exception:
        return ""


def _write_release_log_sync(
    runtime,
    release_log_service,
    *,
    product_name: str,
    recipe_name: str,
    event_type: str,
    result: str,
    message: str,
    runtime_state: str,
) -> None:
    if release_log_service is None:
        return
    try:
        release_log_service.write_event(
            product_name=product_name,
            recipe_name=recipe_name,
            event_type=event_type,
            result=result,
            message=message,
            runtime_state=runtime_state,
        )
    except Exception as exc:
        runtime.logAppended.emit(f"[release] failed to write release log: {exc}")


def _write_runtime_record_sync(runtime, record_service, runtime_result, *, lock_on_ng: bool) -> None:
    runtime_results_store = getattr(runtime, "_runtime_results_store", None)
    if record_service is None and runtime_results_store is None:
        return
    if record_service is not None:
        try:
            record_service.write_product_result(
                product_name=runtime_result.product_name,
                recipe_name=runtime_result.recipe_name,
                final_result=runtime_result.final_result,
                camera1_result=runtime_result.camera_results.get("cam1", None).result
                if runtime_result.camera_results.get("cam1") is not None
                else "",
                camera2_result=runtime_result.camera_results.get("cam2", None).result
                if runtime_result.camera_results.get("cam2") is not None
                else "",
                camera3_result=runtime_result.camera_results.get("cam3", None).result
                if runtime_result.camera_results.get("cam3") is not None
                else "",
                duration_ms=runtime_result.duration_ms,
                is_error=runtime_result.is_system_error,
                error_message=runtime_result.error_message,
                lock_required=(runtime_result.final_result == "NG" and lock_on_ng),
                release_required=(runtime_result.final_result == "NG" and lock_on_ng),
                release_result="pending" if runtime_result.final_result == "NG" and lock_on_ng else "",
                extra_fields=runtime_result.to_record_extra_fields(),
            )
        except Exception as exc:
            runtime.logAppended.emit(f"[runtime] failed to write CSV runtime record: {exc}")
    if runtime_results_store is not None:
        try:
            runtime_results_store.write_runtime_result(runtime_result)
        except Exception as exc:
            runtime.logAppended.emit(f"[runtime] failed to write database runtime record: {exc}")


def _export_runtime_captures_sync(
    runtime,
    *,
    task_id: str,
    preview_frames: dict[str, object],
    roles: tuple[str, ...],
    capture_dir: str,
    capture_root_dir: str,
    stamp: str,
) -> None:
    retained_capture_paths: dict[str, str] = {}
    exported_frames: dict[str, RuntimePreviewFrame] = {}
    for role in roles:
        source = preview_frames.get(role)
        if not isinstance(source, RuntimePreviewFrame):
            continue
        try:
            _ensure_capture_free_space_for_root(runtime, capture_dir, root_dir=capture_root_dir)
            exported_frame = export_runtime_preview_frame(source, capture_dir, stamp=stamp)
        except Exception as exc:
            runtime.logAppended.emit(f"[runtime-capture] failed to export {role}: {exc}")
            continue
        retained_capture_paths[role] = exported_frame.source_path
        exported_frames[role] = exported_frame

    if not retained_capture_paths:
        return

    with runtime._frame_lock:
        current_result = getattr(runtime, "_last_runtime_result", None)
        if current_result is None or str(getattr(current_result, "task_id", "") or "") != str(task_id or ""):
            return
        runtime._last_capture_paths = dict(retained_capture_paths)
        for role, exported_frame in exported_frames.items():
            runtime._last_preview_frames[role] = exported_frame
            camera_result = current_result.camera_results.get(role)
            if camera_result is not None:
                camera_result.image_path = exported_frame.source_path


def _finalize_trigger_outcome(
    runtime,
    outcome,
    release_status_before,
    *,
    active_roles=None,
) -> None:
    trigger_id = str(
        getattr(getattr(runtime, "_last_runtime_result", None), "task_id", "") or ""
    )
    current_preview_frames = dict(runtime._last_preview_frames)
    current_roles = _capture_export_roles(outcome, current_preview_frames)
    result_roles = configured_camera_roles(active_roles or [])
    if not result_roles:
        result_roles = list(_outcome_roles(outcome)) or required_runtime_roles(runtime)
    item_results_by_camera = {
        str(role): list(rows or [])
        for role, rows in dict(runtime._last_item_results_by_camera).items()
    }
    runtime._last_record_path = _record_path_for_service(runtime._record_service)
    runtime._last_capture_paths = {}
    runtime._last_runtime_result = aggregate_runtime_outcome(
        product_name=runtime._session.current_product,
        recipe_name=_recipe_name_for_roles(runtime, result_roles),
        items=runtime._runtime_context.inspection_items,
        active_roles=result_roles,
        camera_outcomes=outcome.camera_outcomes,
        final_result=outcome.final_result,
        duration_ms=outcome.duration_ms,
        error_message=outcome.error_message,
        capture_paths={},
        item_results_by_camera=item_results_by_camera,
    )
    detail_text = runtime._last_runtime_result.summary_text()

    runtime.recordPathChanged.emit(runtime._last_record_path or "-")
    runtime.triggerResultReady.emit(outcome.final_result, detail_text)
    runtime.logAppended.emit(
        _trigger_summary_message(
            trigger_id=trigger_id,
            outcome=outcome,
            roles=result_roles,
            preview_frames=current_preview_frames,
        )
    )
    runtime.logAppended.emit(f"[runtime] result={outcome.final_result} detail={detail_text}")
    runtime._update_status(detail_text or f"result={outcome.final_result}")

    # Publish the frame only after final item statuses have reached the UI. The
    # preview renderer can then color the ROI overlays correctly in one pass.
    for role in current_roles:
        source = current_preview_frames.get(role)
        if source is None:
            source = ""
        runtime.previewUpdated.emit(role, source)

    should_export_captures = _should_export_captures(runtime, outcome.final_result)
    if should_export_captures:
        capture_dt = datetime.now()
        runtime._submit_persistence_task(
            _export_runtime_captures_sync,
            runtime,
            task_id=runtime._last_runtime_result.task_id,
            preview_frames=current_preview_frames,
            roles=current_roles,
            capture_dir=_capture_directory(runtime, capture_dt),
            capture_root_dir=str(_capture_root_directory(runtime)),
            stamp=capture_dt.strftime("%Y%m%d_%H%M%S_%f"),
            description=f"capture export {runtime._last_runtime_result.task_id}",
        )

    runtime._write_runtime_record(runtime._last_runtime_result)

    if (
        release_status_before is not None
        and release_status_before.has_pending_release
        and outcome.final_result != "PRECHECK_FAILED"
    ):
        runtime._write_release_log(
            event_type="release_consumed",
            result="consumed",
            message=f"release consumed when valid inspection started, result={outcome.final_result}",
        )
    if outcome.final_result == "NG" and runtime._lock_on_ng:
        runtime._write_release_log(
            event_type="ng_lock",
            result="locked",
            message=detail_text,
        )
    elif outcome.error_message and runtime._lock_on_ng:
        runtime._write_release_log(
            event_type="runtime_error_lock",
            result="locked",
            message=outcome.error_message,
        )


def _trigger_summary_message(
    *,
    trigger_id: str,
    outcome,
    roles,
    preview_frames: dict[str, object],
) -> str:
    channel_parts: list[str] = []
    camera_outcomes = dict(getattr(outcome, "camera_outcomes", {}) or {})
    for role in configured_camera_roles(roles or []):
        frame = preview_frames.get(role)
        camera_outcome = camera_outcomes.get(role)
        result = str(getattr(camera_outcome, "result", "") or "-").strip()
        physical_role = str(getattr(frame, "physical_role", "") or role).strip()
        serial = str(getattr(frame, "camera_serial", "") or "-").strip()
        frame_number = int(getattr(frame, "frame_number", 0) or 0)
        channel_parts.append(
            f"{role}:{result}/physical={physical_role}/serial={serial}/frame={frame_number}"
        )
    error_text = str(getattr(outcome, "error_message", "") or "").strip()
    error_part = f" error={error_text!r}" if error_text else ""
    return (
        f"[trigger-summary] trigger={str(trigger_id or '-').strip()} "
        f"result={str(getattr(outcome, 'final_result', '') or '-').strip()} "
        f"duration_ms={int(getattr(outcome, 'duration_ms', 0) or 0)} "
        f"channels={','.join(channel_parts) or '-'}{error_part}"
    )


def _precheck(runtime):
    return _precheck_for_roles(runtime, required_runtime_roles(runtime))


def _enabled_items_for_roles(runtime, active_roles: set[str]) -> list[object]:
    return [
        item
        for item in runtime._runtime_context.inspection_items
        if item.enabled and item.camera_id in active_roles
    ]


def _enabled_item_roles(runtime, active_roles: set[str]) -> set[str]:
    return {
        str(item.camera_id or "").strip()
        for item in _enabled_items_for_roles(runtime, active_roles)
        if str(item.camera_id or "").strip()
    }


def _capture_channels_with_enabled_items(runtime, channels: list[dict]) -> list[dict]:
    channel_roles = {
        str(channel.get("role", "")).strip()
        for channel in channels
        if str(channel.get("role", "")).strip()
    }
    item_roles = _enabled_item_roles(runtime, channel_roles)
    return [
        channel
        for channel in channels
        if str(channel.get("role", "")).strip() in item_roles
    ]


def _warm_runtime_models(runtime, enabled_items: list[object]) -> None:
    learning_items = [
        item
        for item in enabled_items
        if runtime._algo.is_learning_tool(item.algorithm_code)
    ]
    if learning_items:
        learning_groups: dict[str, list[object]] = {}
        for item in learning_items:
            algorithm = runtime._algo.resolve_tool_algorithm(
                item.algorithm_code,
                item.camera_id,
            )
            if str(algorithm or "").strip():
                learning_groups.setdefault(algorithm, []).append(item)
        for algorithm, group in learning_groups.items():
            try:
                for item in group:
                    runtime._runtime_context.load_embedding_model(
                        algorithm,
                        model_key=item.model_key,
                    )
                runtime._algo.get_feat_net(algorithm)
            except Exception:
                pass

    traditional_items = [
        item
        for item in enabled_items
        if runtime._algo.is_traditional_tool(item.algorithm_code)
    ]
    for item in traditional_items:
        try:
            algorithm = runtime._algo.resolve_tool_algorithm(item.algorithm_code, item.camera_id)
            runtime._algo.get_traditional_model_dict(algorithm, model_key=item.model_key)
        except Exception:
            pass


def _unsupported_localization_roles(runtime, roles) -> list[str]:
    method_getter = getattr(runtime._runtime_context, "loc_method_for_role", None)
    unsupported_roles: list[str] = []
    for role in configured_camera_roles(roles or []):
        method = (
            method_getter(role)
            if callable(method_getter)
            else getattr(runtime._runtime_context, "loc_method", "")
        )
        if str(method or "").strip().lower() not in {"shape", "ncc"}:
            unsupported_roles.append(role)
    return unsupported_roles


def _precheck_for_capture_channels(runtime, channels: list[dict]) -> tuple[bool, str]:
    from . import controller as runtime_controller_module

    if runtime._frame_grab_service is None or not runtime._frame_grab_service.roles():
        return False, "camera not connected"

    if not channels:
        return False, "capture channel is not enabled"

    active_roles = {
        str(channel.get("role", "")).strip()
        for channel in channels
        if str(channel.get("role", "")).strip()
    }
    if not active_roles:
        return False, "capture channel is not enabled"

    connected_physical_roles = set(physical_connected_roles(runtime))
    missing_physical_roles = sorted(
        {
            str(channel.get("physical_role", "")).strip()
            for channel in channels
            if str(channel.get("physical_role", "")).strip() not in connected_physical_roles
        }
    )
    if missing_physical_roles:
        return False, "required cameras are not connected: " + ", ".join(missing_physical_roles)

    enabled_items = _enabled_items_for_roles(runtime, active_roles)
    enabled_item_roles = {
        str(item.camera_id or "").strip()
        for item in enabled_items
        if str(item.camera_id or "").strip()
    }
    roles_without_items = sorted(active_roles - enabled_item_roles)
    if roles_without_items:
        return False, "please enable at least one inspection tool for " + ", ".join(roles_without_items)

    unsupported_roles = _unsupported_localization_roles(runtime, active_roles)
    if unsupported_roles:
        return (
            False,
            "runtime currently only supports shape/NCC localization for "
            + ", ".join(unsupported_roles),
        )
    _warm_runtime_models(runtime, enabled_items)

    if runtime_controller_module.frame_to_bgr_image is None:
        return False, "camera frame conversion service is unavailable"
    return True, ""


def _channel_float(channel: dict, key: str, default: float) -> float:
    try:
        return float(channel.get(key, default))
    except Exception:
        return float(default)


def _channel_int(channel: dict, key: str, default: int) -> int:
    try:
        return max(0, int(float(channel.get(key, default))))
    except Exception:
        return int(default)


def _apply_capture_channel_camera_settings(runtime, channel: dict) -> None:
    from . import controller as runtime_controller_module

    if runtime._frame_grab_service is None or runtime_controller_module.HikCameraSettings is None:
        return
    physical_role = str(channel.get("physical_role", "")).strip()
    if not physical_role:
        return
    device = runtime._frame_grab_service.get_device(physical_role)
    serial = str(getattr(device, "serial_number", "") or "").strip()
    payload = runtime._camera_settings_store.load_for_role(physical_role, serial=serial) or {}
    payload = dict(payload)
    payload["exposure_time_us"] = _channel_float(channel, "exposure_time_us", 5000.0)
    payload["gain"] = _channel_float(channel, "gain", 0.0)
    settings = runtime_controller_module.HikCameraSettings(
        **runtime_controller_module.hik_settings_kwargs_from_mapping(
            payload,
            default_trigger_mode="software",
            force_trigger_mode="software",
        )
    )
    device.apply_settings(settings)


def _run_single_multi_light_trigger(runtime, requested_roles=None):
    from . import controller as runtime_controller_module

    channels = channels_for_roles(runtime, requested_roles)
    decision = runtime._scheduler.begin_precheck()
    if not decision.allowed:
        return None

    ok, reason = _precheck_for_capture_channels(runtime, channels)
    if not ok:
        runtime._scheduler.on_precheck_failed()
        return runtime_controller_module.FinalInspectionOutcome(
            final_result="PRECHECK_FAILED",
            camera_outcomes={},
            duration_ms=0,
            error_message=reason,
        )

    started_at = time.perf_counter()
    camera_outcomes = {}
    try:
        runtime._tower_light_controller.enter_inspecting()
        try:
            runtime.logAppended.emit(
                "[runtime] single-multi-light channels: "
                + ", ".join(
                    f"{channel.get('role')}<=physical:{channel.get('physical_role')} "
                    f"light:{channel.get('light_output')}"
                    for channel in channels
                )
            )
        except Exception:
            pass
        for channel in channels:
            virtual_role = str(channel.get("role", "")).strip()
            physical_role = str(channel.get("physical_role", "")).strip()
            light_index = light_output_index(channel)
            virtual_index = camera_index_for_role(virtual_role, default=light_index)

            capture_t0 = time.perf_counter()
            light_prepared = False
            try:
                runtime.logAppended.emit(
                    f"[runtime] single-multi-light capture {virtual_role} "
                    f"<= {physical_role}, light={channel.get('light_output')}, "
                    f"exposure={channel.get('exposure_time_us')}, gain={channel.get('gain')}"
                )
            except Exception:
                pass
            try:
                _apply_capture_channel_camera_settings(runtime, channel)
                if hasattr(runtime._light_controller, "set_camera_light_mode"):
                    runtime._light_controller.set_camera_light_mode(light_index, "board_io")
                try:
                    runtime._light_controller.prepare_capture(light_index)
                    light_prepared = True
                    requires_stable_delay = True
                    requires_stable_delay_getter = getattr(
                        runtime._light_controller,
                        "requires_stable_delay",
                        None,
                    )
                    if callable(requires_stable_delay_getter):
                        requires_stable_delay = bool(requires_stable_delay_getter(light_index))
                    stable_delay_ms = _channel_int(channel, "stable_delay_ms", 50)
                    if stable_delay_ms > 0 and requires_stable_delay:
                        time.sleep(stable_delay_ms / 1000.0)
                    runtime._scheduler.on_capture_started(virtual_index)
                    frame = runtime._frame_grab_service.capture_once(physical_role, timeout_ms=1000)
                finally:
                    if light_prepared:
                        runtime._light_controller.finish_capture(light_index)
                capture_ms = (time.perf_counter() - capture_t0) * 1000.0
                outcome = runtime._inspect_frame(
                    virtual_role,
                    frame,
                    physical_role=physical_role,
                )
                camera_outcomes[virtual_role] = replace(outcome, capture_ms=float(capture_ms))
                try:
                    runtime.logAppended.emit(
                        f"[runtime] single-multi-light {virtual_role} "
                        f"captured by {physical_role}, result={outcome.result}"
                    )
                except Exception:
                    pass
            except Exception as channel_exc:
                capture_ms = (time.perf_counter() - capture_t0) * 1000.0
                camera_outcomes[virtual_role] = runtime_controller_module.CameraInspectionOutcome(
                    role=virtual_role,
                    result="NG",
                    message=f"{virtual_role} {channel_exc}",
                    capture_ms=float(capture_ms),
                    match_ms=0.0,
                    infer_ms=0.0,
                )
                try:
                    runtime.logAppended.emit(
                        f"[runtime] single-multi-light {virtual_role} failed: {channel_exc}"
                    )
                except Exception:
                    pass

        runtime._scheduler.on_inspecting_started()
        runtime._scheduler.on_aggregating_started()
        final_ok = all(outcome.result == "OK" for outcome in camera_outcomes.values())
        duration_ms = int((time.perf_counter() - started_at) * 1000.0)
        runtime._scheduler.on_completed(final_ok=final_ok)
        if final_ok:
            runtime._tower_light_controller.show_ok()
        else:
            runtime._tower_light_controller.show_ng()
        return runtime_controller_module.FinalInspectionOutcome(
            final_result="OK" if final_ok else "NG",
            camera_outcomes=camera_outcomes,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000.0)
        try:
            runtime.logAppended.emit(f"[runtime] single-multi-light error: {exc}")
        except Exception:
            pass
        runtime._scheduler.on_error(lock_as_ng=True)
        runtime._tower_light_controller.show_ng()
        return runtime_controller_module.FinalInspectionOutcome(
            final_result="NG",
            camera_outcomes=camera_outcomes,
            duration_ms=duration_ms,
            error_message=str(exc),
        )


def _precheck_for_roles(runtime, roles) -> tuple[bool, str]:
    from . import controller as runtime_controller_module

    if runtime._frame_grab_service is None or not runtime._frame_grab_service.roles():
        return False, "camera not connected"

    requested_role_list = configured_camera_roles(roles or [])
    if not requested_role_list:
        return False, "please enable at least one inspection tool"

    connected_roles = set(runtime._connected_roles())
    missing_roles = [role for role in requested_role_list if role not in connected_roles]
    if missing_roles:
        return False, "required cameras are not connected: " + ", ".join(missing_roles)

    active_roles = set(requested_role_list)

    enabled_items = _enabled_items_for_roles(runtime, active_roles)
    enabled_item_roles = {
        str(item.camera_id or "").strip()
        for item in enabled_items
        if str(item.camera_id or "").strip()
    }
    roles_without_items = [role for role in requested_role_list if role not in enabled_item_roles]
    if roles_without_items:
        return False, "please enable at least one inspection tool for " + ", ".join(roles_without_items)

    unsupported_roles = _unsupported_localization_roles(runtime, requested_role_list)
    if unsupported_roles:
        return (
            False,
            "runtime currently only supports shape/NCC localization for "
            + ", ".join(unsupported_roles),
        )

    _warm_runtime_models(runtime, enabled_items)

    if runtime_controller_module.frame_to_bgr_image is None:
        return False, "camera frame conversion service is unavailable"
    return True, ""


def _save_frame(runtime, role: str, image) -> str:
    product_dir = str(getattr(runtime._session, "product_dir", "") or "").strip()
    preview_frame = build_runtime_preview_frame(
        role=role,
        image_bgr=image,
        source_path="",
        product_dir=product_dir,
        camera_role=role,
    )
    capture_dt = datetime.now()
    capture_dir = _capture_directory(runtime, capture_dt)
    _ensure_capture_free_space(runtime, capture_dir)
    exported_frame = export_runtime_preview_frame(preview_frame, capture_dir)
    with runtime._frame_lock:
        runtime._last_capture_paths[role] = exported_frame.source_path
        runtime._last_preview_frames[role] = exported_frame
    return exported_frame.source_path


def _inspect_frame(runtime, role: str, frame, *, physical_role: str = ""):
    from . import controller as runtime_controller_module

    product_dir = str(getattr(runtime._session, "product_dir", "") or "")
    trigger_id = str(
        getattr(getattr(runtime, "_last_runtime_result", None), "task_id", "") or ""
    )
    physical_role_text = str(physical_role or role or "").strip()
    camera_serial = str(getattr(frame, "camera_serial", "") or "").strip()
    frame_number = int(getattr(frame, "frame_num", 0) or 0)
    capture_timestamp = int(getattr(frame, "host_timestamp", 0) or 0)
    image = runtime_controller_module.frame_to_bgr_image(frame)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    runtime.logAppended.emit(
        "[capture] "
        f"trigger={trigger_id or '-'} logical={role} physical={physical_role_text} "
        f"serial={camera_serial or '-'} frame={frame_number} timestamp={capture_timestamp} "
        f"shape={tuple(image.shape)} dtype={image.dtype}"
    )
    preview_frame = build_runtime_preview_frame(
        role=role,
        image_bgr=image,
        trigger_id=trigger_id,
        physical_role=physical_role_text,
        camera_serial=camera_serial,
        frame_number=frame_number,
        capture_timestamp=capture_timestamp,
        source_path="",
        product_dir=product_dir,
        camera_role=role,
    )
    with runtime._frame_lock:
        runtime._last_preview_frames[role] = preview_frame
    # Publish the acquired image immediately.  The finalized preview (with ROI
    # shapes and measurements) is published again after inspection completes.
    runtime.previewUpdated.emit(role, preview_frame)
    with runtime._inspect_lock:
        response = runtime._inspection_executor.execute(
            runtime_controller_module.InspectionExecutionRequest(
                camera_id=role,
                image_path="",
                image_bgr=image,
                items=[item for item in runtime._runtime_context.inspection_items if item.camera_id == role],
            )
        )
        runtime._last_item_results_by_camera[role] = list(response.item_results)
    roi_shapes = tuple(getattr(response, "roi_shapes", ()) or ())
    measurements = tuple(getattr(response, "measurements", ()) or ())
    preview_frame = build_runtime_preview_frame(
        role=role,
        image_bgr=image,
        trigger_id=trigger_id,
        physical_role=physical_role_text,
        camera_serial=camera_serial,
        frame_number=frame_number,
        capture_timestamp=capture_timestamp,
        source_path="",
        product_dir=product_dir,
        camera_role=role,
        roi_shapes=roi_shapes,
        measurements=measurements,
    )
    with runtime._frame_lock:
        runtime._last_preview_frames[role] = preview_frame
    message = f"{role} pred={response.result}"
    if response.detail:
        message += f" {response.detail}"
    runtime.logAppended.emit(
        "[inspect] "
        f"trigger={trigger_id or '-'} logical={role} physical={physical_role_text} "
        f"serial={camera_serial or '-'} result={response.result} "
        f"match={float(response.match_ms or 0.0):.1f}ms "
        f"infer={float(response.infer_ms or 0.0):.1f}ms"
    )
    return runtime_controller_module.CameraInspectionOutcome(
        role=role,
        result=response.result,
        message=message,
        match_ms=float(response.match_ms or 0.0),
        infer_ms=float(response.infer_ms or 0.0),
    )


def _write_release_log(runtime, *, event_type: str, result: str, message: str = "") -> None:
    release_log_service = runtime._release_log_service
    if release_log_service is None:
        return
    runtime._submit_persistence_task(
        _write_release_log_sync,
        runtime,
        release_log_service,
        product_name=runtime._session.current_product,
        recipe_name=_recipe_name_for_roles(runtime, runtime._connected_roles()),
        event_type=event_type,
        result=result,
        message=message,
        runtime_state=runtime._current_runtime_state_text(),
        description=f"release log {event_type}",
    )


def _write_runtime_record(runtime, runtime_result) -> None:
    record_service = runtime._record_service
    runtime_results_store = getattr(runtime, "_runtime_results_store", None)
    if record_service is None and runtime_results_store is None:
        return
    runtime._submit_persistence_task(
        _write_runtime_record_sync,
        runtime,
        record_service,
        runtime_result,
        lock_on_ng=runtime._lock_on_ng,
        description=f"runtime record {getattr(runtime_result, 'task_id', '')}",
    )


def _reload_runtime_context(runtime) -> None:
    try:
        runtime._runtime_context.reload()
    except Exception as exc:
        runtime.logAppended.emit(f"[runtime] failed to reload runtime context: {exc}")


