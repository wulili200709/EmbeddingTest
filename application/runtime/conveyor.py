"""Continuous conveyor integration for :class:`RuntimeController`."""

from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path

from PySide6 import QtCore

from common.app_paths import packaged_embedding_test_root
from common.camera_roles import camera_index_for_role
from domain import InspectionItemResult
from domain.conveyor_line import ConveyorConfig, ConveyorLineController
from application.inspection_executor import InspectionExecutionRequest
from .capture_channels import channels_for_roles, required_runtime_roles
from .capture_pipeline import capture_runtime_channel
from .preview_frame import build_runtime_preview_frame


_TEMPLATE_MATCH_FAILURE_MARKERS = (
    "match failure",
    "match failed",
    "matching failed",
    "did not find any match",
    "no match",
    "\u5339\u914d\u5931\u8d25",
    "\u672a\u5339\u914d",
)


def _is_template_match_failure(error: BaseException) -> bool:
    detail = str(error or "").strip().lower()
    return bool(detail) and any(marker in detail for marker in _TEMPLATE_MATCH_FAILURE_MARKERS)


def _load_conveyor_config(runtime) -> ConveyorConfig:
    path = packaged_embedding_test_root(__file__) / "config" / "defaults" / "conveyor_control.json"
    runtime._conveyor_config_path = Path(path)
    try:
        return ConveyorConfig.from_json_file(path)
    except Exception as exc:
        runtime.logAppended.emit(f"[conveyor] invalid configuration {path}: {exc}")
        raise RuntimeError(f"invalid conveyor configuration: {exc}") from exc


def _initialize_conveyor_controller(runtime, input_snapshot=None) -> bool:
    if runtime._io_controller is None or not getattr(runtime._io_controller, "is_open", False):
        return False
    if runtime._conveyor_controller is not None:
        if input_snapshot is not None:
            runtime._conveyor_controller.initialize_inputs(input_snapshot, io_ready=True)
        return True

    config = runtime._load_conveyor_config()

    def _write_output(name: str, on: bool) -> None:
        controller = runtime._io_controller
        if controller is None or not getattr(controller, "is_open", False):
            raise RuntimeError("IO controller is not open")
        arbiter = runtime._output_arbiter
        if arbiter is not None:
            arbiter.set_line_output(name, bool(on))
        else:
            controller.set_output(name, bool(on))

    def _write_outputs(updates) -> None:
        controller = runtime._io_controller
        if controller is None or not getattr(controller, "is_open", False):
            raise RuntimeError("IO controller is not open")
        normalized = {str(name): bool(on) for name, on in dict(updates).items()}
        arbiter = runtime._output_arbiter
        if arbiter is not None:
            arbiter.set_line_outputs(normalized)
        else:
            controller.set_outputs(normalized)

    runtime._conveyor_controller = ConveyorLineController(
        config=config,
        output_writer=_write_output,
        output_batch_writer=_write_outputs,
        inspection_requester=runtime._enqueue_conveyor_inspection,
        state_listener=runtime._publish_conveyor_state,
        log_writer=lambda message: runtime.logAppended.emit(f"[conveyor] {message}"),
        start_authorizer=runtime._prepare_conveyor_start,
        inspection_result_listener=lambda _sequence_id, _epoch, result, _detail: (
            _show_conveyor_inspection_result(runtime, result)
        ),
    )

    # Normal conveyor production must not lock the inspection scheduler after
    # every NG. NG is retained per workpiece and consumed later at DI1.
    runtime._lock_on_ng = False
    if runtime._scheduler is not None:
        runtime._scheduler._lock_on_ng = False

    snapshot = dict(input_snapshot or runtime._io_controller.snapshot_inputs())
    runtime._conveyor_controller.initialize_inputs(snapshot, io_ready=True)
    interval = max(5, min(100, int(config.poll_interval_ms)))
    runtime._conveyor_timer.setInterval(interval)
    runtime._conveyor_timer.start()
    runtime.logAppended.emit(f"[conveyor] controller initialized; config={runtime._conveyor_config_path}")
    return True


def _shutdown_conveyor_controller(runtime) -> None:
    runtime._conveyor_timer.stop()
    controller = runtime._conveyor_controller
    if controller is not None:
        try:
            controller.shutdown()
        except Exception as exc:
            runtime.logAppended.emit(f"[conveyor] shutdown failed: {exc}")
    runtime._conveyor_controller = None


def _wait_for_conveyor_inspections(runtime) -> None:
    with runtime._trigger_lock:
        pending = list(runtime._conveyor_inspection_futures)
    for future in pending:
        try:
            future.result()
        except Exception:
            pass
        finally:
            runtime._conveyor_inspection_futures.discard(future)


def _publish_conveyor_state(runtime, snapshot) -> None:
    payload = dict(snapshot or {})
    runtime._last_conveyor_snapshot = payload
    runtime.conveyorStateChanged.emit(payload)


@QtCore.Slot()
def _tick_conveyor(runtime) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        return
    try:
        controller.tick()
    except Exception as exc:
        runtime.logAppended.emit(f"[conveyor] control tick failed: {exc}")


@QtCore.Slot(str, bool, float, float, int)
def _handle_conveyor_di_event(
    runtime,
    name: str,
    state: bool,
    sample_wall_s: float = 0.0,
    sample_monotonic_s: float = 0.0,
    raw_word: int = -1,
) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        return
    try:
        if str(name) in {
            "camera_trigger_sensor",
            "reject_position_sensor",
            "end_test_sensor",
            "good_outlet_sensor",
            "waste_outlet_sensor",
        }:
            channel = -1
            active_high = None
            io_controller = getattr(runtime, "_io_controller", None)
            mapping = getattr(io_controller, "mapping", None)
            if mapping is not None:
                try:
                    input_config = mapping.get_input(str(name))
                    channel = int(input_config.channel)
                    active_high = bool(input_config.active_high)
                except Exception:
                    pass
            delay_ms = (
                max(0.0, time.perf_counter() - float(sample_monotonic_s)) * 1000.0
                if float(sample_monotonic_s) > 0.0
                else -1.0
            )
            sample_time = "unknown"
            if float(sample_wall_s) > 0.0:
                whole_seconds = int(float(sample_wall_s))
                milliseconds = int((float(sample_wall_s) - whole_seconds) * 1000.0)
                sample_time = (
                    f"{time.strftime('%H:%M:%S', time.localtime(whole_seconds))}."
                    f"{milliseconds:03d}"
                )
            raw_text = f"0x{int(raw_word) & 0xFFFF:04X}" if int(raw_word) >= 0 else "unknown"
            runtime.logAppended.emit(
                "[DI-sample] "
                f"logical={name}; physical=DI{channel if channel >= 0 else '?'}; "
                f"state={'ON' if state else 'OFF'}; raw_word={raw_text}; "
                f"active_high={active_high if active_high is not None else 'unknown'}; "
                f"sample_time={sample_time}; "
                f"sample_epoch_ms={float(sample_wall_s) * 1000.0:.1f}; "
                f"ui_delay_ms={delay_ms:.1f}"
            )
        controller.handle_input_change(str(name), bool(state))
    except Exception as exc:
        runtime.logAppended.emit(f"[conveyor] DI event failed ({name}={state}): {exc}")
        controller.set_io_ready(False, detail=str(exc))


@QtCore.Slot(str, str)
def _handle_conveyor_io_error(runtime, name: str, detail: str) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        return
    message = f"DI read failed ({name}): {detail}"
    runtime.logAppended.emit(f"[conveyor] {message}")
    controller.set_io_ready(False, detail=message)


def _enqueue_conveyor_inspection(runtime, sequence_id: int, epoch: int) -> None:
    with runtime._trigger_lock:
        executor = runtime._trigger_executor
        if executor is None or not runtime._accept_trigger_jobs:
            raise RuntimeError("inspection worker is not available")
        future = executor.submit(runtime._run_conveyor_capture, int(sequence_id), int(epoch))
        runtime._conveyor_inspection_futures.add(future)

    def _done(done: Future) -> None:
        runtime._conveyorCaptureTaskFinished.emit(int(sequence_id), int(epoch), done)

    future.add_done_callback(_done)


def _prepare_conveyor_start(runtime) -> tuple[bool, str]:
    with runtime._trigger_lock:
        manual_task = runtime._trigger_future
        if runtime._busy or (manual_task is not None and not manual_task.done()):
            return False, "manual inspection is still running"
    runtime._reload_runtime_context()
    roles = required_runtime_roles(runtime)
    warmup_started = time.perf_counter()
    ok, reason = runtime._precheck_for_roles(roles)
    if ok:
        runtime._conveyor_required_roles = list(roles)
        runtime.logAppended.emit(
            "[conveyor] inspection pipeline ready; "
            f"precheck_and_warmup_ms={(time.perf_counter() - warmup_started) * 1000.0:.1f}"
        )
    else:
        runtime._conveyor_required_roles = []
    return bool(ok), str(reason or "")


def _run_conveyor_capture(runtime, sequence_id: int, epoch: int) -> list[dict[str, object]]:
    """Capture immediately in FIFO order without waiting for prior inference."""
    from . import execution as runtime_execution

    controller = runtime._conveyor_controller
    if controller is None or controller.epoch != int(epoch):
        raise RuntimeError("capture belongs to an inactive conveyor epoch")
    if runtime._frame_grab_service is None:
        raise RuntimeError("camera service is not connected")

    roles = list(runtime._conveyor_required_roles or required_runtime_roles(runtime))
    if not roles:
        raise RuntimeError("no inspection camera role is configured")

    configured_channels = channels_for_roles(runtime, roles)
    if configured_channels:
        channels = configured_channels
    else:
        channels = [
            {
                "role": role,
                "physical_role": role,
                "light_index": camera_index_for_role(role),
            }
            for role in roles
        ]

    captured: list[dict[str, object]] = []
    for channel in channels:
        controller = runtime._conveyor_controller
        if controller is None or controller.epoch != int(epoch):
            raise RuntimeError("capture invalidated by purge or shutdown")
        captured_frame = capture_runtime_channel(
            runtime,
            channel,
            apply_camera_settings=(
                lambda current: runtime_execution._apply_capture_channel_camera_settings(
                    runtime, current
                )
                if configured_channels
                else None
            ),
        )
        captured.append(captured_frame.to_payload())
    runtime.logAppended.emit(
        f"[conveyor] capture completed: item={sequence_id}, epoch={epoch}, roles={roles}"
    )
    return captured


@QtCore.Slot(int, int, object)
def _on_conveyor_capture_task_finished(runtime, sequence_id: int, epoch: int, done: Future) -> None:
    runtime._conveyor_inspection_futures.discard(done)
    controller = runtime._conveyor_controller
    if controller is not None:
        controller.capture_completed(sequence_id, epoch)
    try:
        captured = done.result()
    except Exception as exc:
        if controller is not None:
            controller.inspection_completed(sequence_id, epoch, "ERROR", detail=str(exc))
        return
    if controller is None or controller.epoch != int(epoch):
        return
    executor = runtime._conveyor_infer_executor
    if executor is None:
        controller.inspection_completed(
            sequence_id,
            epoch,
            "ERROR",
            detail="conveyor inference worker is not available",
        )
        return
    future = executor.submit(
        runtime._run_conveyor_inspection,
        int(sequence_id),
        int(epoch),
        list(captured),
    )
    runtime._conveyor_inspection_futures.add(future)

    def _done(infer_done: Future) -> None:
        runtime._conveyorInspectionTaskFinished.emit(int(sequence_id), int(epoch), infer_done)

    future.add_done_callback(_done)


def _run_conveyor_inspection(
    runtime,
    sequence_id: int,
    epoch: int,
    captured: list[dict[str, object]],
) -> tuple[str, str]:
    """Inspect a captured workpiece using task-local result containers."""
    from . import controller as runtime_controller_module
    started_at = time.perf_counter()
    camera_outcomes: dict[str, object] = {}
    item_results_by_camera: dict[str, list] = {}
    preview_frames: dict[str, object] = {}
    roles: list[str] = []
    multiple_product_reports: list[str] = []
    trigger_id = f"conveyor-{epoch}-{sequence_id}"

    for capture in captured:
        role = str(capture.get("role", "") or "").strip()
        physical_role = str(capture.get("physical_role", role) or role).strip()
        frame = capture.get("frame")
        roles.append(role)
        image = runtime_controller_module.frame_to_bgr_image(frame)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        camera_serial = str(getattr(frame, "camera_serial", "") or "").strip()
        frame_number = int(getattr(frame, "frame_num", 0) or 0)
        capture_timestamp = int(getattr(frame, "host_timestamp", 0) or 0)
        preview = build_runtime_preview_frame(
            role=role,
            image_bgr=image,
            trigger_id=trigger_id,
            physical_role=physical_role,
            camera_serial=camera_serial,
            frame_number=frame_number,
            capture_timestamp=capture_timestamp,
            source_path="",
            product_dir=str(getattr(runtime._session, "product_dir", "") or ""),
            camera_role=role,
        )
        runtime.previewCycleStarted.emit(role, trigger_id)
        runtime.previewUpdated.emit(role, preview)

        camera_items = [
            item
            for item in runtime._runtime_context.inspection_items
            if item.camera_id == role
        ]
        inspect_t0 = time.perf_counter()
        try:
            with runtime._inspect_lock:
                response = runtime._inspection_executor.execute(
                    InspectionExecutionRequest(
                        camera_id=role,
                        image_path="",
                        image_bgr=image,
                        items=camera_items,
                    )
                )
        except Exception as exc:
            if not _is_template_match_failure(exc):
                raise
            match_ms = (time.perf_counter() - inspect_t0) * 1000.0
            failure_detail = str(exc or "match failure").strip() or "match failure"
            item_results_by_camera[role] = [
                InspectionItemResult(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    camera_id=item.camera_id,
                    roi_label=item.roi_label,
                    algorithm_code=item.algorithm_code,
                    enabled=item.enabled,
                    params=dict(item.params or {}),
                    result="NG" if item.enabled else "DISABLED",
                    detail=failure_detail if item.enabled else "",
                )
                for item in camera_items
            ]
            preview_frames[role] = preview
            camera_outcomes[role] = runtime_controller_module.CameraInspectionOutcome(
                role=role,
                result="NG",
                message=f"{role} {failure_detail}",
                capture_ms=float(capture.get("capture_ms", 0.0) or 0.0),
                match_ms=float(match_ms),
                infer_ms=0.0,
            )
            runtime.logAppended.emit(
                f"[conveyor] template match failed: item={sequence_id}, role={role}, "
                f"treated as NG: {failure_detail}"
            )
            continue
        inspect_ms = (time.perf_counter() - inspect_t0) * 1000.0
        product_count = _reported_product_count(response)
        if product_count is not None and product_count > 1:
            multiple_product_reports.append(f"{role} reported {product_count} products")
        item_results_by_camera[role] = list(response.item_results)
        preview = build_runtime_preview_frame(
            role=role,
            image_bgr=image,
            trigger_id=trigger_id,
            physical_role=physical_role,
            camera_serial=camera_serial,
            frame_number=frame_number,
            capture_timestamp=capture_timestamp,
            source_path="",
            product_dir=str(getattr(runtime._session, "product_dir", "") or ""),
            camera_role=role,
            roi_shapes=tuple(getattr(response, "roi_shapes", ()) or ()),
            measurements=tuple(getattr(response, "measurements", ()) or ()),
        )
        preview_frames[role] = preview
        camera_outcomes[role] = runtime_controller_module.CameraInspectionOutcome(
            role=role,
            result=str(response.result or "NG"),
            message=f"{role} pred={response.result} {response.detail or ''}".strip(),
            capture_ms=float(capture.get("capture_ms", 0.0) or 0.0),
            match_ms=float(response.match_ms or 0.0),
            infer_ms=float(response.infer_ms or inspect_ms),
        )

    final_ok = bool(camera_outcomes) and all(
        str(getattr(outcome, "result", "") or "").upper() == "OK"
        for outcome in camera_outcomes.values()
    )
    final_result = (
        "MULTIPLE_PRODUCTS_IN_FOV"
        if multiple_product_reports
        else ("OK" if final_ok else "NG")
    )
    duration_ms = int((time.perf_counter() - started_at) * 1000.0)
    outcome = runtime_controller_module.FinalInspectionOutcome(
        final_result=final_result,
        camera_outcomes=camera_outcomes,
        duration_ms=duration_ms,
    )

    # Finalization uses the existing record/UI pipeline. Keep its legacy global
    # compatibility fields together under one lock so out-of-order completion
    # cannot mix two workpieces.
    with runtime._frame_lock:
        runtime._last_preview_frames = dict(preview_frames)
        runtime._last_item_results_by_camera = dict(item_results_by_camera)
        runtime._last_runtime_result = runtime._build_pending_runtime_result(status="RUNNING")
        runtime._finalize_trigger_outcome(outcome, None, active_roles=roles)
        result_object = runtime._last_runtime_result
        detail = str(
            result_object.summary_text()
            if result_object is not None and hasattr(result_object, "summary_text")
            else ""
        )
        if multiple_product_reports:
            detail = "; ".join(multiple_product_reports)
    runtime.logAppended.emit(
        f"[conveyor] inspection completed: item={sequence_id}, result={final_result}"
    )
    return final_result, detail


def _reported_product_count(response) -> int | None:
    """Read an optional product-count result without coupling to one algorithm."""
    keys = ("product_count", "detected_product_count", "instance_count", "object_count")
    candidates: list[object] = []
    raw_row = getattr(response, "raw_row", None)
    if isinstance(raw_row, dict):
        candidates.extend(raw_row.get(key) for key in keys if key in raw_row)
        for item_row in list(raw_row.get("item_rows", []) or []):
            if isinstance(item_row, dict):
                candidates.extend(item_row.get(key) for key in keys if key in item_row)
    for measurement in tuple(getattr(response, "measurements", ()) or ()):
        if isinstance(measurement, dict):
            candidates.extend(
                measurement.get(key) for key in keys if key in measurement
            )
    reported_counts: list[int] = []
    for value in candidates:
        try:
            reported_counts.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(reported_counts) if reported_counts else None


def _show_conveyor_inspection_result(runtime, result: str) -> None:
    """Drive the shared result lamps/buzzer for an accepted conveyor result."""
    controller = getattr(runtime, "_tower_light_controller", None)
    if controller is None:
        return
    normalized = str(result or "").strip().upper()
    if normalized in {"OK", "GOOD"}:
        controller.show_ok()
    else:
        # TowerLightController uses the live ng_buzzer_ms setting, so conveyor
        # NG results follow the same configurable pulse duration as manual runs.
        controller.show_ng()


@QtCore.Slot(int, int, object)
def _on_conveyor_inspection_task_finished(runtime, sequence_id: int, epoch: int, done: Future) -> None:
    runtime._conveyor_inspection_futures.discard(done)
    try:
        result, detail = done.result()
    except Exception as exc:
        result, detail = "ERROR", str(exc)
    controller = runtime._conveyor_controller
    if controller is None:
        return
    controller.inspection_completed(
        int(sequence_id),
        int(epoch),
        str(result),
        detail=str(detail),
    )


@QtCore.Slot()
def start_conveyor(runtime) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        runtime.warningOccurred.emit("皮带控制器未初始化，请先连接相机和IO。")
        return
    controller.request_start()


@QtCore.Slot()
def stop_conveyor(runtime) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        return
    controller.request_controlled_stop()


@QtCore.Slot()
def start_conveyor_purge(runtime) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        runtime.warningOccurred.emit("皮带控制器未初始化，请先连接相机和IO。")
        return
    if not controller.request_purge():
        runtime.warningOccurred.emit("当前状态不允许清线，请确认皮带已停止、DI5有效且门已关闭。")


@QtCore.Slot()
def continue_conveyor_purge(runtime) -> None:
    controller = runtime._conveyor_controller
    if controller is None:
        return
    if not controller.continue_purge():
        runtime.warningOccurred.emit("当前没有可继续的清线流程，或安全许可尚未恢复。")


@QtCore.Slot()
def acknowledge_conveyor_alarm(runtime) -> None:
    tower_controller = getattr(runtime, "_tower_light_controller", None)
    if tower_controller is not None and hasattr(tower_controller, "silence_buzzer"):
        tower_controller.silence_buzzer()
    controller = runtime._conveyor_controller
    if controller is None:
        return
    try:
        controller.acknowledge_alarm()
    except Exception as exc:
        runtime.logAppended.emit(f"[conveyor] alarm buzzer silence failed: {exc}")
        runtime.warningOccurred.emit("蜂鸣器消音失败，请检查IO连接。")


__all__ = [
    "_load_conveyor_config",
    "_initialize_conveyor_controller",
    "_shutdown_conveyor_controller",
    "_wait_for_conveyor_inspections",
    "_publish_conveyor_state",
    "_tick_conveyor",
    "_handle_conveyor_di_event",
    "_handle_conveyor_io_error",
    "_enqueue_conveyor_inspection",
    "_prepare_conveyor_start",
    "_run_conveyor_capture",
    "_on_conveyor_capture_task_finished",
    "_run_conveyor_inspection",
    "_reported_product_count",
    "_on_conveyor_inspection_task_finished",
    "start_conveyor",
    "stop_conveyor",
    "start_conveyor_purge",
    "continue_conveyor_purge",
    "acknowledge_conveyor_alarm",
]
