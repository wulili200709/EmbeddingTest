"""Execution-oriented RuntimeController helpers."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from domain import aggregate_runtime_outcome, recipe_name_from_path
from algorithms.registry import learning_backbone_storage_code

from .capture_policy import normalize_capture_retention_policy
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
    if record_service is None:
        return
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
            duration_ms=runtime_result.duration_ms,
            is_error=runtime_result.is_system_error,
            error_message=runtime_result.error_message,
            lock_required=(runtime_result.final_result == "NG" and lock_on_ng),
            release_required=(runtime_result.final_result == "NG" and lock_on_ng),
            release_result="pending" if runtime_result.final_result == "NG" and lock_on_ng else "",
            extra_fields=runtime_result.to_record_extra_fields(),
        )
    except Exception as exc:
        runtime.logAppended.emit(f"[runtime] failed to write runtime record: {exc}")


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


def _finalize_trigger_outcome(runtime, outcome, release_status_before) -> None:
    current_preview_frames = dict(runtime._last_preview_frames)
    current_roles = _capture_export_roles(outcome, current_preview_frames)
    active_roles = runtime._connected_roles()
    item_results_by_camera = {
        str(role): list(rows or [])
        for role, rows in dict(runtime._last_item_results_by_camera).items()
    }
    runtime._last_record_path = _record_path_for_service(runtime._record_service)
    runtime._last_capture_paths = {}
    runtime._last_runtime_result = aggregate_runtime_outcome(
        product_name=runtime._session.current_product,
        recipe_name=_recipe_name_for_roles(runtime, active_roles),
        items=runtime._runtime_context.inspection_items,
        active_roles=active_roles,
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
    for role in ("cam1", "cam2"):
        source = current_preview_frames.get(role)
        if source is None:
            source = ""
        runtime.previewUpdated.emit(role, source)
    runtime.logAppended.emit(f"[runtime] result={outcome.final_result} detail={detail_text}")
    runtime._emit_runtime_context()
    runtime._update_status(detail_text or f"result={outcome.final_result}")

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


def _precheck(runtime):
    return _precheck_for_roles(runtime, runtime._connected_roles())


def _precheck_for_roles(runtime, roles) -> tuple[bool, str]:
    from . import controller as runtime_controller_module

    if runtime._frame_grab_service is None or not runtime._frame_grab_service.roles():
        return False, "camera not connected"

    if runtime._runtime_context.loc_method != "shape":
        return False, "runtime currently only supports shape localization"

    active_roles = {
        str(role).strip()
        for role in runtime._connected_roles()
        if str(role).strip()
    }
    requested_roles = {
        str(role).strip()
        for role in roles
        if str(role).strip()
    }
    if requested_roles:
        active_roles &= requested_roles
    if not active_roles:
        return False, "requested camera is not connected"
    missing_recipe_roles = [
        role for role in sorted(active_roles)
        if not os.path.exists(_recipe_path_for_role(runtime, role))
    ]
    if missing_recipe_roles:
        if len(missing_recipe_roles) == 1:
            return False, f"please generate and save a shape template first for {missing_recipe_roles[0]}"
        return False, "please generate and save shape templates first for connected cameras"
    enabled_items = [
        item
        for item in runtime._runtime_context.inspection_items
        if item.enabled and item.camera_id in active_roles
    ]
    if not enabled_items:
        if len(active_roles) == 1:
            role = next(iter(active_roles))
            return False, f"please enable at least one inspection tool for {role}"
        return False, "please enable at least one inspection tool for connected cameras"

    unknown_algorithms = sorted(
        {
            str(item.algorithm_code or "").strip()
            for item in enabled_items
            if runtime._algo.tool_algorithm_spec(item.algorithm_code) is None
        }
    )
    if unknown_algorithms:
        return False, f"unsupported inspection algorithm: {unknown_algorithms[0]}"

    learning_items = [
        item
        for item in enabled_items
        if runtime._algo.is_learning_tool(item.algorithm_code)
    ]
    if learning_items:
        algorithm = runtime._algo.current_learning_backbone()
        if not str(algorithm or "").strip():
            return False, "please choose a learning tool subtype first"
        try:
            for item in learning_items:
                runtime._runtime_context.load_embedding_model(algorithm, model_key=item.model_key)
            if runtime._algo.model is not None:
                runtime._algo.get_feat_net(
                    runtime._algo.model.backbone,
                    getattr(runtime._algo.model, "device", None),
                )
        except Exception as exc:
            return False, f"failed to load model: {exc}"
        if runtime._algo.model is None:
            return False, f"algorithm {learning_backbone_storage_code(algorithm)} does not have a trained model yet"
    traditional_items = [
        item
        for item in enabled_items
        if runtime._algo.is_traditional_tool(item.algorithm_code)
    ]
    for item in traditional_items:
        algorithm = runtime._algo.resolve_tool_algorithm(item.algorithm_code)
        model_dict = runtime._algo.get_traditional_model_dict(algorithm, model_key=item.model_key)
        if not isinstance(model_dict, dict):
            return False, f"traditional algorithm {algorithm} is not trained yet"

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


def _inspect_frame(runtime, role: str, frame):
    from . import controller as runtime_controller_module

    product_dir = str(getattr(runtime._session, "product_dir", "") or "")
    image = runtime_controller_module.frame_to_bgr_image(frame)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    preview_frame = build_runtime_preview_frame(
        role=role,
        image_bgr=image,
        source_path="",
        product_dir=product_dir,
        camera_role=role,
    )
    with runtime._frame_lock:
        runtime._last_preview_frames[role] = preview_frame
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
    if record_service is None:
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


