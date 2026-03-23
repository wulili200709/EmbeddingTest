"""Execution-oriented RuntimeController helpers."""

from __future__ import annotations

import os
from datetime import datetime

import cv2

from domain import aggregate_runtime_outcome, recipe_name_from_path

from .capture_policy import delete_capture_artifacts, retained_capture_paths_for_policy


def _finalize_trigger_outcome(runtime, outcome, release_status_before) -> None:
    runtime._last_record_path = ""
    if runtime._record_service is not None:
        runtime._last_record_path = str(runtime._record_service.writer.file_path_for_date())
    current_capture_paths = dict(runtime._last_capture_paths)

    for role in ("cam1", "cam2"):
        path = current_capture_paths.get(role, "")
        runtime.previewUpdated.emit(role, path)

    runtime.recordPathChanged.emit(runtime._last_record_path or "-")
    retained_capture_paths = retained_capture_paths_for_policy(
        runtime._capture_retention_policy,
        outcome.final_result,
        current_capture_paths,
    )

    runtime._last_runtime_result = aggregate_runtime_outcome(
        product_name=runtime._session.current_product,
        recipe_name=recipe_name_from_path(runtime._session.line2dup_recipe_path),
        items=runtime._runtime_context.inspection_items,
        active_roles=runtime._connected_roles(),
        camera_outcomes=outcome.camera_outcomes,
        final_result=outcome.final_result,
        duration_ms=outcome.duration_ms,
        error_message=outcome.error_message,
        capture_paths=retained_capture_paths,
        item_results_by_camera=runtime._last_item_results_by_camera,
    )
    runtime._write_runtime_record(runtime._last_runtime_result)
    transient_capture_paths = {
        role: path
        for role, path in current_capture_paths.items()
        if path and retained_capture_paths.get(role) != path
    }
    delete_capture_artifacts(transient_capture_paths)
    runtime._last_capture_paths = dict(retained_capture_paths)
    if runtime._record_service is not None:
        runtime._last_record_path = str(runtime._record_service.writer.file_path_for_date())
        runtime.recordPathChanged.emit(runtime._last_record_path or "-")
    detail_text = runtime._last_runtime_result.summary_text()

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
    if outcome.final_result == "NG":
        runtime._write_release_log(
            event_type="ng_lock",
            result="locked",
            message=detail_text,
        )
    elif outcome.error_message:
        runtime._write_release_log(
            event_type="runtime_error_lock",
            result="locked",
            message=outcome.error_message,
        )

    runtime.triggerResultReady.emit(outcome.final_result, detail_text)
    runtime.logAppended.emit(f"[runtime] result={outcome.final_result} detail={detail_text}")
    runtime._emit_runtime_context()
    runtime._update_status(f"result={outcome.final_result}")


def _precheck(runtime):
    from . import controller as runtime_controller_module

    if runtime._frame_grab_service is None or not runtime._frame_grab_service.roles():
        return False, "camera not connected"

    if runtime._runtime_context.loc_method != "line2dup":
        return False, "runtime currently only supports line2dup localization"

    if not os.path.exists(runtime._session.line2dup_recipe_path):
        return False, "please generate and save a line2dup template first"

    algorithm = runtime._runtime_context.current_algorithm()
    if not algorithm:
        return False, "please select an algorithm first"
    if runtime._algo.is_embedding_algorithm(algorithm):
        try:
            runtime._runtime_context.load_embedding_model(algorithm)
            if runtime._algo.model is not None:
                runtime._algo.get_feat_net(
                    runtime._algo.model.backbone,
                    getattr(runtime._algo.model, "device", None),
                )
        except Exception as exc:
            return False, f"failed to load model: {exc}"
        if runtime._algo.model is None:
            return False, f"algorithm {algorithm} does not have a trained model yet"
    else:
        model_dict = runtime._algo.product_params.traditional_models.get(algorithm)
        if not isinstance(model_dict, dict):
            return False, f"traditional algorithm {algorithm} is not trained yet"

    if runtime_controller_module.frame_to_bgr_image is None:
        return False, "camera frame conversion service is unavailable"
    return True, ""


def _save_frame(runtime, role: str, frame) -> str:
    from . import controller as runtime_controller_module

    capture_dir = os.path.join(runtime._session.product_dir, "runtime_capture")
    os.makedirs(capture_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(capture_dir, f"{stamp}_{role}.png")
    image = runtime_controller_module.frame_to_bgr_image(frame)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"failed to save runtime capture: {path}")
    with runtime._frame_lock:
        runtime._last_capture_paths[role] = path
    return path


def _inspect_frame(runtime, role: str, frame):
    from . import controller as runtime_controller_module

    path = runtime._save_frame(role, frame)
    with runtime._inspect_lock:
        response = runtime._inspection_executor.execute(
            runtime_controller_module.InspectionExecutionRequest(
                camera_id=role,
                image_path=path,
                items=[item for item in runtime._runtime_context.inspection_items if item.camera_id == role],
            )
        )
        runtime._last_item_results_by_camera[role] = list(response.item_results)
    message = f"{os.path.basename(path)} pred={response.result}"
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
    if runtime._release_log_service is None:
        return
    try:
        runtime._release_log_service.write_event(
            product_name=runtime._session.current_product,
            recipe_name=recipe_name_from_path(runtime._session.line2dup_recipe_path),
            event_type=event_type,
            result=result,
            message=message,
            runtime_state=runtime._current_runtime_state_text(),
        )
    except Exception as exc:
        runtime.logAppended.emit(f"[release] failed to write release log: {exc}")


def _write_runtime_record(runtime, runtime_result) -> None:
    if runtime._record_service is None:
        return
    try:
        runtime._record_service.write_product_result(
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
            lock_required=(runtime_result.final_result == "NG"),
            release_required=(runtime_result.final_result == "NG"),
            release_result="pending" if runtime_result.final_result == "NG" else "",
            extra_fields=runtime_result.to_record_extra_fields(),
        )
    except Exception as exc:
        runtime.logAppended.emit(f"[runtime] failed to write runtime record: {exc}")


def _reload_runtime_context(runtime) -> None:
    try:
        runtime._runtime_context.reload()
    except Exception as exc:
        runtime.logAppended.emit(f"[runtime] failed to reload runtime context: {exc}")
