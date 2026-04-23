"""Hardware-oriented RuntimeController helpers."""

from __future__ import annotations

import configparser
import json
import os
import threading
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from app_paths import packaged_embedding_test_root, packaged_repo_root
from .capture_policy import DEFAULT_LIGHT_STABLE_MS


_CONVEYOR_RUN_OUTPUT_CANDIDATES = ("conveyor_run", "reserved_out_1")
_BUZZER_OUTPUT_CANDIDATES = ("buzzer", "light_cam2", "reserved_out_2")
_START_BUTTON_LIGHT_OUTPUT_CANDIDATES = ("button_green", "start_button_light")
_STOP_BUTTON_LIGHT_OUTPUT_CANDIDATES = ("button_red", "stop_button_light")
_RESET_BUTTON_LIGHT_OUTPUT_CANDIDATES = ("button_blue", "reset_button_light")
_CONVEYOR_START_INPUT_CANDIDATES = ("conveyor_start", "reserved_in_1")
_CONVEYOR_STOP_INPUT_CANDIDATES = ("conveyor_stop", "reserved_in_2")
_RESET_BUTTON_INPUT_CANDIDATES = ("reset_button", "release_button", "reserved_in_3")
_CONVEYOR_TOGGLE_INPUT_CANDIDATES = ("conveyor_toggle", "reserved_in_1")
_FOOT_SWITCH_INPUT_CANDIDATES = ("foot_switch",)
_SPLIT_BUTTON_BOX_INPUT_NAMES = {"conveyor_start", "conveyor_stop", "reset_button", "release_button"}
_RUNTIME_IO_LOGIC_FILENAME = "runtime_io_logic.json"
_BUILTIN_RUNTIME_IO_LOGIC_DEFAULTS = {
    "startup": [
        {"type": "set_conveyor_run", "value": True, "reason": "startup default"},
        {"type": "set_buzzer", "value": False, "reason": "startup default"},
        {"type": "set_output", "name": "button_green", "value": True, "reason": "startup default"},
        {"type": "set_output", "name": "button_red", "value": False, "reason": "startup default"},
        {"type": "set_output", "name": "button_blue", "value": False, "reason": "startup default"},
    ],
    "shutdown": [
        {"type": "set_conveyor_run", "value": False, "reason": "shutdown default"},
        {"type": "set_buzzer", "value": False, "reason": "shutdown default"},
        {"type": "set_output", "name": "button_green", "value": False, "reason": "shutdown default"},
        {"type": "set_output", "name": "button_red", "value": True, "reason": "shutdown default"},
        {"type": "set_output", "name": "button_blue", "value": False, "reason": "shutdown default"},
    ],
    "ng": [
        {"type": "set_conveyor_run", "value": False, "reason": "NG result"},
        {"type": "set_buzzer", "value": True, "reason": "NG result"},
        {"type": "set_output", "name": "button_green", "value": False, "reason": "NG result"},
        {"type": "set_output", "name": "button_red", "value": True, "reason": "NG result"},
        {"type": "set_output", "name": "button_blue", "value": False, "reason": "NG result"},
    ],
    "release_granted": [
        {"type": "set_conveyor_run", "value": True, "reason": "release granted"},
        {"type": "set_output", "name": "button_green", "value": True, "reason": "release granted"},
        {"type": "set_output", "name": "button_red", "value": False, "reason": "release granted"},
        {"type": "set_output", "name": "button_blue", "value": False, "reason": "release granted"},
    ],
}


def _load_nkio_runtime_options(mapping_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("nkio_config_path", "nkio_dll_path"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result[key] = text
    return result


def _resolve_input_name(runtime, controller, *candidates: str) -> str | None:
    mapping = getattr(controller, "mapping", None)
    if mapping is None:
        return candidates[-1] if candidates else None
    available = set(mapping.di_names())
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _resolve_output_name(runtime, controller, *candidates: str) -> str | None:
    mapping = getattr(controller, "mapping", None)
    if mapping is None:
        return candidates[-1] if candidates else None
    available = set(mapping.do_names())
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _emit_conveyor_run_state(runtime, *, available: bool, running: bool, detail: str = "") -> None:
    runtime._conveyor_running = bool(running)
    signal = getattr(runtime, "conveyorRunStateChanged", None)
    if signal is not None:
        signal.emit(bool(available), bool(running), str(detail or ""))


def _emit_io_status(runtime, ready: bool, detail: str, controller=None) -> None:
    runtime._io_ready = bool(ready)
    runtime._io_status_detail = str(detail or "")
    runtime.ioStatusChanged.emit(runtime._io_ready, runtime._io_status_detail, controller)
    running = bool(getattr(runtime, "_conveyor_running", False))
    if runtime._io_ready and controller is not None:
        try:
            output_name = _resolve_output_name(runtime, controller, *_CONVEYOR_RUN_OUTPUT_CANDIDATES)
            if output_name:
                running = bool(controller.read_output(output_name))
        except Exception:
            pass
    _emit_conveyor_run_state(
        runtime,
        available=runtime._io_ready,
        running=running if runtime._io_ready else False,
        detail=runtime._io_status_detail,
    )


def _coerce_logic_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "on", "yes"}:
        return True
    if text in {"0", "false", "off", "no"}:
        return False
    return bool(default)


def _coerce_logic_int(value: object, *, default: int = 0, minimum: int = 0) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = int(default)
    return max(int(minimum), normalized)


def _normalize_io_logic_action(raw_action: object) -> dict[str, object] | None:
    if not isinstance(raw_action, dict):
        return None
    action_type = str(raw_action.get("type", "") or "").strip().lower()
    if not action_type:
        return None
    action: dict[str, object] = {"type": action_type}
    if action_type in {"set_conveyor_run", "set_buzzer", "set_output", "pulse_output"}:
        action["value"] = _coerce_logic_bool(raw_action.get("value"), default=False)
    if action_type == "pulse_output":
        action["value"] = _coerce_logic_bool(raw_action.get("value"), default=True)
        action["reset_value"] = _coerce_logic_bool(raw_action.get("reset_value"), default=False)
        action["duration_ms"] = _coerce_logic_int(raw_action.get("duration_ms", 200), default=200, minimum=1)
    if action_type in {"set_output", "pulse_output"}:
        name = str(raw_action.get("name", "") or "").strip()
        if not name:
            return None
        action["name"] = name
    for key in ("reason", "message"):
        if key not in raw_action:
            continue
        text = str(raw_action.get(key, "") or "").strip()
        if text:
            action[key] = text
    return action


def _load_runtime_io_logic_file(path: Path) -> dict[str, list[dict[str, object]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    events_payload = payload.get("events") if isinstance(payload.get("events"), dict) else payload
    if not isinstance(events_payload, dict):
        return {}

    result: dict[str, list[dict[str, object]]] = {}
    for raw_event_name, raw_actions in events_payload.items():
        event_name = str(raw_event_name or "").strip().lower()
        if not event_name:
            continue
        if raw_actions is None:
            result[event_name] = []
            continue
        if not isinstance(raw_actions, list):
            continue
        actions: list[dict[str, object]] = []
        for raw_action in raw_actions:
            normalized = _normalize_io_logic_action(raw_action)
            if normalized is not None:
                actions.append(normalized)
        result[event_name] = actions
    return result


def _default_runtime_io_logic_path() -> Path:
    return packaged_embedding_test_root(__file__) / "config" / "defaults" / _RUNTIME_IO_LOGIC_FILENAME


def _product_runtime_io_logic_path(runtime) -> Path | None:
    product_dir = str(getattr(getattr(runtime, "_session", None), "product_dir", "") or "").strip()
    if not product_dir:
        return None
    return Path(product_dir) / _RUNTIME_IO_LOGIC_FILENAME


def _runtime_io_logic_for_event(runtime, event_name: str) -> tuple[bool, list[dict[str, object]]]:
    normalized_event = str(event_name or "").strip().lower()
    if not normalized_event:
        return False, []

    merged: dict[str, list[dict[str, object]]] = {
        key: [dict(action) for action in actions]
        for key, actions in _BUILTIN_RUNTIME_IO_LOGIC_DEFAULTS.items()
    }
    default_path = _default_runtime_io_logic_path()
    if default_path.exists():
        merged.update(_load_runtime_io_logic_file(default_path))
    product_path = _product_runtime_io_logic_path(runtime)
    if product_path is not None and product_path.exists():
        merged.update(_load_runtime_io_logic_file(product_path))

    if normalized_event not in merged:
        return False, []
    return True, [dict(action) for action in merged.get(normalized_event, [])]


def _format_io_logic_text(template: object, context: dict[str, object] | None) -> str:
    text = str(template or "").strip()
    if not text:
        return ""
    if not context:
        return text
    try:
        return text.format(**context)
    except Exception:
        return text


def _resolve_runtime_output_name(runtime, controller, name: str) -> str:
    resolved_name = str(name or "").strip()
    if resolved_name == "conveyor_run":
        return _resolve_output_name(runtime, controller, *_CONVEYOR_RUN_OUTPUT_CANDIDATES) or resolved_name
    if resolved_name == "buzzer":
        return _resolve_output_name(runtime, controller, *_BUZZER_OUTPUT_CANDIDATES) or resolved_name
    return resolved_name


def _set_button_box_lights(
    runtime,
    active_button: str,
    *,
    controller=None,
    reason: str = "",
) -> bool:
    active_controller = controller if controller is not None else getattr(runtime, "_io_controller", None)
    if active_controller is None or not getattr(active_controller, "is_open", False):
        return False

    states_by_button = {
        "start": (
            (_START_BUTTON_LIGHT_OUTPUT_CANDIDATES, True),
            (_STOP_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
            (_RESET_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
        ),
        "stop": (
            (_START_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
            (_STOP_BUTTON_LIGHT_OUTPUT_CANDIDATES, True),
            (_RESET_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
        ),
        "reset": (
            (_START_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
            (_STOP_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
            (_RESET_BUTTON_LIGHT_OUTPUT_CANDIDATES, True),
        ),
        "all_off": (
            (_START_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
            (_STOP_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
            (_RESET_BUTTON_LIGHT_OUTPUT_CANDIDATES, False),
        ),
    }
    plans = states_by_button.get(str(active_button or "").strip().lower())
    if plans is None:
        return False

    applied_any = False
    for candidates, on in plans:
        output_name = _resolve_output_name(runtime, active_controller, *candidates)
        if not output_name:
            continue
        changed = _set_named_output(
            runtime,
            output_name,
            bool(on),
            controller=active_controller,
            reason=reason,
        )
        applied_any = changed or applied_any
    return applied_any


def _io_logic_timer_lock(runtime):
    lock = getattr(runtime, "_io_logic_pulse_timer_lock", None)
    if lock is None:
        lock = threading.RLock()
        runtime._io_logic_pulse_timer_lock = lock
    return lock


def _cancel_io_logic_pulse_timer(runtime, name: str) -> None:
    timers = getattr(runtime, "_io_logic_pulse_timers", None)
    if not isinstance(timers, dict):
        return
    with _io_logic_timer_lock(runtime):
        timer = timers.pop(str(name or "").strip(), None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass


def _cancel_all_io_logic_pulse_timers(runtime) -> None:
    timers = getattr(runtime, "_io_logic_pulse_timers", None)
    if not isinstance(timers, dict):
        runtime._io_logic_pulse_timers = {}
        return
    with _io_logic_timer_lock(runtime):
        active_timers = list(timers.values())
        timers.clear()
    for timer in active_timers:
        try:
            timer.cancel()
        except Exception:
            pass


def _schedule_io_logic_pulse_reset(
    runtime,
    *,
    name: str,
    reset_value: bool,
    duration_ms: int,
    controller=None,
    reason: str = "",
) -> None:
    active_controller = controller if controller is not None else getattr(runtime, "_io_controller", None)
    if active_controller is None:
        return
    resolved_name = _resolve_runtime_output_name(runtime, active_controller, name)
    _cancel_io_logic_pulse_timer(runtime, resolved_name)

    def _reset_output() -> None:
        try:
            _set_named_output(
                runtime,
                resolved_name,
                bool(reset_value),
                controller=active_controller,
                reason=reason or "pulse reset",
            )
        finally:
            timers = getattr(runtime, "_io_logic_pulse_timers", None)
            if isinstance(timers, dict):
                with _io_logic_timer_lock(runtime):
                    current = timers.get(resolved_name)
                    if current is timer:
                        timers.pop(resolved_name, None)

    timer = threading.Timer(max(0.001, float(duration_ms) / 1000.0), _reset_output)
    timer.daemon = True
    with _io_logic_timer_lock(runtime):
        timers = getattr(runtime, "_io_logic_pulse_timers", None)
        if not isinstance(timers, dict):
            timers = {}
            runtime._io_logic_pulse_timers = timers
        timers[resolved_name] = timer
    timer.start()


def _set_named_output(
    runtime,
    name: str,
    on: bool,
    *,
    controller=None,
    reason: str = "",
    phase: str = "",
) -> bool:
    active_controller = controller if controller is not None else getattr(runtime, "_io_controller", None)
    if active_controller is None or not getattr(active_controller, "is_open", False):
        return False
    resolved_name = _resolve_runtime_output_name(runtime, active_controller, name)
    try:
        active_controller.set_output(resolved_name, bool(on))
    except Exception as exc:
        prefix = f"{phase} output default" if phase else "io logic output"
        runtime.logAppended.emit(f"[IO] failed to apply {prefix}: {resolved_name}={on}: {exc}")
        return False
    if phase:
        runtime.logAppended.emit(f"[IO] {phase} output default applied: {resolved_name}={'ON' if on else 'OFF'}")
    else:
        detail = f" ({reason})" if str(reason or "").strip() else ""
        runtime.logAppended.emit(
            f"[IO] io logic output applied: {resolved_name}={'ON' if on else 'OFF'}{detail}"
        )
    conveyor_name = _resolve_output_name(runtime, active_controller, *_CONVEYOR_RUN_OUTPUT_CANDIDATES)
    if conveyor_name and resolved_name == conveyor_name:
        _emit_conveyor_run_state(
            runtime,
            available=True,
            running=bool(on),
            detail=reason or (f"{phase} output default" if phase else ""),
        )
    return True


def _execute_io_logic_action(
    runtime,
    action: dict[str, object],
    *,
    controller=None,
    event_name: str,
    context: dict[str, object] | None = None,
) -> None:
    action_type = str(action.get("type", "") or "").strip().lower()
    reason = _format_io_logic_text(action.get("reason", event_name), context)
    if action_type == "set_conveyor_run":
        if controller is not None:
            _set_conveyor_run(
                runtime,
                _coerce_logic_bool(action.get("value"), default=False),
                reason=reason,
                controller=controller,
            )
            return
        setter = getattr(runtime, "_set_conveyor_run", None)
        if callable(setter):
            setter(_coerce_logic_bool(action.get("value"), default=False), reason=reason)
        return
    if action_type == "set_buzzer":
        if controller is not None:
            _set_buzzer(
                runtime,
                _coerce_logic_bool(action.get("value"), default=False),
                reason=reason,
                controller=controller,
            )
            return
        setter = getattr(runtime, "_set_buzzer", None)
        if callable(setter):
            setter(_coerce_logic_bool(action.get("value"), default=False), reason=reason)
        return
    if action_type == "set_output":
        name = str(action.get("name", "") or "").strip()
        if not name:
            return
        phase = event_name if event_name in {"startup", "shutdown"} else ""
        _set_named_output(
            runtime,
            name,
            _coerce_logic_bool(action.get("value"), default=False),
            controller=controller,
            reason=reason,
            phase=phase,
        )
        return
    if action_type == "pulse_output":
        name = str(action.get("name", "") or "").strip()
        if not name:
            return
        pulse_value = _coerce_logic_bool(action.get("value"), default=True)
        reset_value = _coerce_logic_bool(action.get("reset_value"), default=False)
        duration_ms = _coerce_logic_int(action.get("duration_ms", 200), default=200, minimum=1)
        if not _set_named_output(
            runtime,
            name,
            pulse_value,
            controller=controller,
            reason=reason or f"{event_name} pulse",
        ):
            return
        _schedule_io_logic_pulse_reset(
            runtime,
            name=name,
            reset_value=reset_value,
            duration_ms=duration_ms,
            controller=controller,
            reason=f"{reason or event_name} pulse reset",
        )
        return
    if action_type == "log":
        message = _format_io_logic_text(action.get("message", ""), context)
        if message:
            runtime.logAppended.emit(message)
        return
    if action_type == "status":
        message = _format_io_logic_text(action.get("message", ""), context)
        if message and hasattr(runtime, "_update_status"):
            runtime._update_status(message)
        return
    runtime.logAppended.emit(f"[IO] unsupported io logic action: {action_type} (event={event_name})")


def _apply_io_logic_event(
    runtime,
    event_name: str,
    *,
    controller=None,
    context: dict[str, object] | None = None,
) -> bool:
    configured, actions = _runtime_io_logic_for_event(runtime, event_name)
    if not configured:
        return False
    normalized_event = str(event_name or "").strip().lower()
    for action in actions:
        _execute_io_logic_action(
            runtime,
            action,
            controller=controller,
            event_name=normalized_event,
            context=context,
        )
    return True


def _apply_output_defaults(
    runtime,
    controller,
    defaults: tuple[tuple[tuple[str, ...], bool, str], ...],
    *,
    phase: str,
) -> None:
    for candidates, on, label in defaults:
        name = _resolve_output_name(runtime, controller, *candidates)
        if not name:
            runtime.logAppended.emit(
                f"[IO] skipped {phase} output default for {label}: no mapping found among {list(candidates)}"
            )
            continue
        try:
            controller.set_output(name, on)
        except Exception as exc:
            runtime.logAppended.emit(
                f"[IO] failed to apply {phase} output default: {name}={on}: {exc}"
            )
            continue
        runtime.logAppended.emit(f"[IO] {phase} output default applied: {name}={'ON' if on else 'OFF'}")
        if candidates == _CONVEYOR_RUN_OUTPUT_CANDIDATES:
            _emit_conveyor_run_state(
                runtime,
                available=True,
                running=bool(on),
                detail=f"{phase} output default",
            )


def _apply_startup_output_defaults(runtime, controller) -> None:
    if _apply_io_logic_event(runtime, "startup", controller=controller):
        return
    _apply_output_defaults(
        runtime,
        controller,
        (
            (_CONVEYOR_RUN_OUTPUT_CANDIDATES, True, "conveyor"),
            (_BUZZER_OUTPUT_CANDIDATES, False, "buzzer"),
        ),
        phase="startup",
    )


def _apply_shutdown_output_defaults(runtime, controller) -> None:
    if _apply_io_logic_event(runtime, "shutdown", controller=controller):
        return
    _apply_output_defaults(
        runtime,
        controller,
        (
            (_CONVEYOR_RUN_OUTPUT_CANDIDATES, False, "conveyor"),
            (_BUZZER_OUTPUT_CANDIDATES, False, "buzzer"),
        ),
        phase="shutdown",
    )


def _set_buzzer(runtime, on: bool, *, reason: str = "", controller=None) -> bool:
    controller = controller if controller is not None else getattr(runtime, "_io_controller", None)
    if controller is None or not getattr(controller, "is_open", False):
        return False
    output_name = _resolve_output_name(runtime, controller, *_BUZZER_OUTPUT_CANDIDATES)
    if not output_name:
        runtime.logAppended.emit(
            f"[IO] missing buzzer output mapping; expected one of {list(_BUZZER_OUTPUT_CANDIDATES)}"
        )
        return False
    try:
        controller.set_output(output_name, bool(on))
    except Exception as exc:
        runtime.logAppended.emit(
            f"[IO] failed to set buzzer output: {output_name}={'ON' if on else 'OFF'}: {exc}"
        )
        return False
    detail = f" ({reason})" if str(reason or "").strip() else ""
    runtime.logAppended.emit(
        f"[IO] buzzer output applied: {output_name}={'ON' if on else 'OFF'}{detail}"
    )
    return True


def _set_conveyor_run(runtime, running: bool, *, reason: str = "", controller=None) -> bool:
    controller = controller if controller is not None else getattr(runtime, "_io_controller", None)
    if controller is None or not getattr(controller, "is_open", False):
        return False
    runtime._pending_ng_stop_delay_ms = 0
    runtime._ng_stop_delay_pending = False
    runtime._ng_stop_delay_sequence = int(getattr(runtime, "_ng_stop_delay_sequence", 0) or 0) + 1
    output_name = _resolve_output_name(runtime, controller, *_CONVEYOR_RUN_OUTPUT_CANDIDATES)
    if not output_name:
        runtime.logAppended.emit(
            f"[IO] missing conveyor output mapping; expected one of {list(_CONVEYOR_RUN_OUTPUT_CANDIDATES)}"
        )
        return False
    try:
        controller.set_output(output_name, bool(running))
    except Exception as exc:
        runtime.logAppended.emit(
            f"[IO] failed to set conveyor output: {output_name}="
            f"{'ON' if running else 'OFF'}: {exc}"
        )
        return False
    detail = f" ({reason})" if str(reason or "").strip() else ""
    runtime.logAppended.emit(
        f"[IO] conveyor output applied: {output_name}="
        f"{'ON' if running else 'OFF'}{detail}"
    )
    _emit_conveyor_run_state(
        runtime,
        available=True,
        running=bool(running),
        detail=str(reason or "").strip(),
    )
    return True


def set_conveyor_run(runtime, running: bool) -> None:
    if _set_conveyor_run(runtime, bool(running), reason="manual UI"):
        if bool(running):
            _set_button_box_lights(runtime, "start", reason="manual UI")
        else:
            _set_button_box_lights(runtime, "stop", reason="manual UI")
        runtime._update_status("皮带已启动" if running else "皮带已停止")
        return
    runtime.warningOccurred.emit("IO未就绪，无法控制皮带")
    _emit_conveyor_run_state(
        runtime,
        available=False,
        running=False,
        detail="IO未就绪，无法控制皮带",
    )


def _rebuild_runner(runtime) -> bool:
    from . import controller as runtime_controller_module

    if runtime._import_error is not None:
        runtime._update_status(f"运行服务不可用: {runtime._import_error}")
        return False

    runtime._stop_di_poller()
    runtime._last_record_path = None
    runtime._last_capture_paths = {}
    runtime._light_controller = runtime_controller_module._UiOnlyLightController()
    runtime._tower_light_controller = runtime_controller_module._UiOnlyTowerLightController()
    if runtime._io_controller is None or not getattr(runtime._io_controller, "is_open", False):
        runtime._io_controller = runtime._try_create_io_controller()
    else:
        runtime._emit_io_status(
            True,
            runtime._io_status_detail or "real IO ready",
            runtime._io_controller,
        )
    if runtime._io_controller is not None:
        runtime._light_controller = runtime_controller_module.LightController(runtime._io_controller)
        tower_settings = dict(getattr(runtime, "_tower_light_settings", {}) or {})
        runtime._tower_light_controller = runtime_controller_module.TowerLightController(
            runtime._io_controller,
            ok_flash_ms=int(tower_settings.get("ok_flash_ms", 200) or 200),
            ng_flash_ms=int(tower_settings.get("ng_flash_ms", 200) or 200),
            idle_blue_delay_s=float(int(tower_settings.get("idle_blue_delay_ms", 30000) or 30000)) / 1000.0,
        )
    runtime._camera_manager = runtime_controller_module.HikCameraManager()
    runtime._frame_grab_service = runtime_controller_module.FrameGrabService(runtime._camera_manager)
    runtime._permission_manager = runtime_controller_module.PermissionManager(runtime._release_password)
    runtime._scheduler = runtime_controller_module.InspectionScheduler(
        state_machine=runtime_controller_module.RunStateMachine(),
        permission_manager=runtime._permission_manager,
        lock_on_ng=runtime._lock_on_ng,
    )
    records_dir = str(getattr(runtime, "_runtime_records_dir", Path(runtime._session.product_dir) / "runtime_records"))
    release_logs_dir = os.path.join(runtime._session.product_dir, "release_logs")
    os.makedirs(records_dir, exist_ok=True)
    os.makedirs(release_logs_dir, exist_ok=True)
    runtime._record_service = runtime_controller_module.TestRecordService(
        runtime_controller_module.CsvRecordWriter(records_dir)
    )
    runtime._release_log_service = runtime_controller_module.ReleaseLogService(
        runtime_controller_module.CsvReleaseLogWriter(release_logs_dir)
    )
    runtime._runner = runtime_controller_module.InspectionRuntime(
        scheduler=runtime._scheduler,
        permission_manager=runtime._permission_manager,
        frame_grab_service=runtime._frame_grab_service,
        light_controller=runtime._light_controller,
        tower_light_controller=runtime._tower_light_controller,
        inspect_callback=runtime._inspect_frame,
        precheck_callback=runtime._precheck,
        role_to_camera_index={"cam1": 1, "cam2": 2},
        light_stable_ms=DEFAULT_LIGHT_STABLE_MS,
    )
    runtime._tower_light_controller.enter_waiting()
    return True


def _try_create_io_controller(runtime):
    from . import controller as runtime_controller_module

    if (
        runtime_controller_module.DiMonitor is None
        or runtime_controller_module.IoManager is None
        or runtime_controller_module.LightController is None
        or runtime_controller_module.TowerLightController is None
    ):
        runtime.logAppended.emit("[IO] real IO controller unavailable, fallback to UI-only mode")
        runtime._emit_io_status(False, "real IO controller unavailable")
        return None

    mapping_path = packaged_embedding_test_root(__file__) / "config" / "defaults" / "io_mapping.json"
    if not mapping_path.exists():
        runtime.logAppended.emit(f"[IO] missing IO mapping config: {mapping_path}")
        runtime._emit_io_status(False, f"missing IO mapping config: {mapping_path}")
        return None

    runtime_options = _load_nkio_runtime_options(mapping_path)
    configured_board_config = runtime_options.get("nkio_config_path")
    board_config_path = Path(configured_board_config) if configured_board_config else runtime._find_nkio_config_path()
    if board_config_path is None:
        runtime.logAppended.emit("[IO] missing nkio_config.ini, fallback to UI-only mode")
        runtime._emit_io_status(False, "missing nkio_config.ini")
        return None
    if not Path(board_config_path).exists():
        runtime.logAppended.emit(f"[IO] configured nkio_config.ini missing: {board_config_path}")
        runtime._emit_io_status(False, f"configured nkio_config.ini missing: {board_config_path}")
        return None

    dll_path = runtime_options.get("nkio_dll_path")

    try:
        controller = runtime_controller_module.IoManager.from_config_file(
            board_config_path,
            mapping_path,
            dll_path=dll_path,
        )
        controller.open()
        _apply_startup_output_defaults(runtime, controller)
    except Exception as exc:
        runtime.logAppended.emit(f"[IO] failed to initialize real IO: {exc}")
        runtime._emit_io_status(False, f"failed to initialize real IO: {exc}")
        return None

    runtime.logAppended.emit(f"[IO] using real IO: {board_config_path}")
    runtime._emit_io_status(True, f"real IO ready: {board_config_path}", controller)
    return controller


def _initialize_startup_io(runtime, force: bool = False) -> bool:
    if runtime._import_error is not None:
        runtime._emit_io_status(False, f"runtime import error: {runtime._import_error}")
        return False
    if runtime._io_controller is not None and getattr(runtime._io_controller, "is_open", False) and not force:
        runtime._emit_io_status(True, runtime._io_status_detail or "real IO ready", runtime._io_controller)
        return True
    if runtime._io_controller is not None:
        runtime._close_io_controller()
    runtime._io_controller = runtime._try_create_io_controller()
    return runtime._io_controller is not None


def _close_io_controller(runtime) -> None:
    _cancel_all_io_logic_pulse_timers(runtime)
    if hasattr(runtime._tower_light_controller, "close"):
        try:
            runtime._tower_light_controller.close()
        except Exception:
            pass
    if hasattr(runtime._light_controller, "turn_off_all"):
        try:
            runtime._light_controller.turn_off_all()
        except Exception:
            pass
    if runtime._io_controller is not None:
        try:
            if hasattr(runtime._tower_light_controller, "all_off"):
                runtime._tower_light_controller.all_off()
        except Exception:
            pass
        try:
            _apply_shutdown_output_defaults(runtime, runtime._io_controller)
        except Exception:
            pass
        try:
            runtime._io_controller.close()
        except Exception:
            pass
    runtime._emit_io_status(False, runtime._io_status_detail or "IO closed")


def _start_di_poller_if_available(runtime) -> None:
    from . import controller as runtime_controller_module

    runtime._stop_di_poller()
    if runtime._io_controller is None or runtime_controller_module.DiMonitor is None:
        return
    foot_switch_name = _resolve_input_name(runtime, runtime._io_controller, *_FOOT_SWITCH_INPUT_CANDIDATES)
    mapping = getattr(runtime._io_controller, "mapping", None)
    available_inputs = set(mapping.di_names()) if mapping is not None else set()
    split_button_box_mode = bool(available_inputs.intersection(_SPLIT_BUTTON_BOX_INPUT_NAMES))

    conveyor_start_name = None
    conveyor_stop_name = None
    reset_button_name = None
    conveyor_toggle_name = None
    if split_button_box_mode:
        conveyor_start_name = _resolve_input_name(
            runtime, runtime._io_controller, *_CONVEYOR_START_INPUT_CANDIDATES
        )
        conveyor_stop_name = _resolve_input_name(
            runtime, runtime._io_controller, *_CONVEYOR_STOP_INPUT_CANDIDATES
        )
        reset_button_name = _resolve_input_name(
            runtime, runtime._io_controller, *_RESET_BUTTON_INPUT_CANDIDATES
        )
    else:
        conveyor_toggle_name = _resolve_input_name(
            runtime, runtime._io_controller, *_CONVEYOR_TOGGLE_INPUT_CANDIDATES
        )
    input_names = list(
        dict.fromkeys(
            name
            for name in (
                foot_switch_name,
                conveyor_start_name,
                conveyor_stop_name,
                reset_button_name,
                conveyor_toggle_name,
            )
            if name
        )
    )
    if not input_names:
        runtime.logAppended.emit("[IO] skipped DI monitor start: no mapped DI inputs for trigger/control")
        return
    try:
        poller = runtime_controller_module.DiMonitor(
            runtime._io_controller,
            input_names=input_names,
            poll_interval_ms=20,
            debounce_ms=50,
        )
        poller.add_rising_callback(runtime._on_foot_switch_rising)
        if split_button_box_mode:
            poller.add_rising_callback(runtime._on_conveyor_start_rising)
            poller.add_falling_callback(runtime._on_conveyor_stop_falling)
            poller.add_rising_callback(runtime._on_reset_button_rising)
            poller.add_falling_callback(runtime._on_reset_button_falling)
        else:
            poller.add_rising_callback(runtime._on_conveyor_toggle_rising)
        poller.start()
    except Exception as exc:
        runtime.logAppended.emit(f"[IO] failed to start DI monitor: {exc}")
        return
    runtime._di_poller = poller
    runtime.logAppended.emit(f"[IO] DI monitor started for: {', '.join(input_names)}")


def _stop_di_poller(runtime) -> None:
    if runtime._di_poller is None:
        return
    try:
        runtime._di_poller.stop()
    except Exception:
        pass
    runtime._di_poller = None


def _on_foot_switch_rising(runtime, event) -> None:
    if str(getattr(event, "name", "") or "") not in _FOOT_SWITCH_INPUT_CANDIDATES:
        return
    runtime.logAppended.emit(f"[foot-switch] rising edge detected: {event.name}")
    delay_ms = max(0, int(getattr(runtime, "_foot_trigger_delay_ms", 0) or 0))
    if delay_ms <= 0:
        QtCore.QMetaObject.invokeMethod(runtime, "_trigger_from_di", QtCore.Qt.QueuedConnection)
        return
    if getattr(runtime, "_di_trigger_delay_pending", False):
        runtime.logAppended.emit("[foot-switch] delayed trigger already pending; ignoring new edge")
        return
    runtime._pending_di_trigger_delay_ms = delay_ms
    runtime._di_trigger_delay_pending = True
    QtCore.QMetaObject.invokeMethod(runtime, "_schedule_trigger_from_di", QtCore.Qt.QueuedConnection)


def _on_conveyor_toggle_rising(runtime, event) -> None:
    if str(getattr(event, "name", "") or "") not in _CONVEYOR_TOGGLE_INPUT_CANDIDATES:
        return
    runtime.logAppended.emit(f"[conveyor-toggle] rising edge detected: {event.name}")
    QtCore.QMetaObject.invokeMethod(runtime, "_toggle_conveyor_run_from_di", QtCore.Qt.QueuedConnection)


def _on_conveyor_start_rising(runtime, event) -> None:
    if str(getattr(event, "name", "") or "") not in _CONVEYOR_START_INPUT_CANDIDATES:
        return
    runtime.logAppended.emit(f"[conveyor-start] rising edge detected: {event.name}")
    QtCore.QMetaObject.invokeMethod(runtime, "_start_conveyor_from_di", QtCore.Qt.QueuedConnection)


def _on_conveyor_stop_falling(runtime, event) -> None:
    if str(getattr(event, "name", "") or "") not in _CONVEYOR_STOP_INPUT_CANDIDATES:
        return
    runtime.logAppended.emit(f"[conveyor-stop] falling edge detected: {event.name}")
    QtCore.QMetaObject.invokeMethod(runtime, "_stop_conveyor_from_di", QtCore.Qt.QueuedConnection)


def _on_reset_button_rising(runtime, event) -> None:
    if str(getattr(event, "name", "") or "") not in _RESET_BUTTON_INPUT_CANDIDATES:
        return
    runtime.logAppended.emit(f"[reset-button] rising edge detected: {event.name}")
    QtCore.QMetaObject.invokeMethod(runtime, "_handle_reset_button_from_di", QtCore.Qt.QueuedConnection)


def _on_reset_button_falling(runtime, event) -> None:
    if str(getattr(event, "name", "") or "") not in _RESET_BUTTON_INPUT_CANDIDATES:
        return
    runtime.logAppended.emit(f"[reset-button] falling edge detected: {event.name}")
    QtCore.QMetaObject.invokeMethod(
        runtime,
        "_handle_reset_button_release_from_di",
        QtCore.Qt.QueuedConnection,
    )


@QtCore.Slot()
def _schedule_trigger_from_di(runtime) -> None:
    delay_ms = max(0, int(getattr(runtime, "_pending_di_trigger_delay_ms", 0) or 0))
    if delay_ms <= 0:
        runtime._di_trigger_delay_pending = False
        runtime._pending_di_trigger_delay_ms = 0
        runtime.trigger()
        return
    runtime.logAppended.emit(f"[foot-switch] waiting {delay_ms} ms before trigger")
    QtCore.QTimer.singleShot(delay_ms, lambda: runtime._fire_delayed_trigger_from_di())


@QtCore.Slot()
def _fire_delayed_trigger_from_di(runtime) -> None:
    if not getattr(runtime, "_di_trigger_delay_pending", False):
        return
    runtime._di_trigger_delay_pending = False
    runtime._pending_di_trigger_delay_ms = 0
    runtime.logAppended.emit("[foot-switch] delayed trigger fired")
    runtime.trigger()


@QtCore.Slot()
def _trigger_from_di(runtime) -> None:
    runtime.trigger()


@QtCore.Slot()
def _start_conveyor_from_di(runtime) -> None:
    controller = getattr(runtime, "_io_controller", None)
    if _apply_io_logic_event(runtime, "start_button_pressed", controller=controller):
        runtime._update_status("DI2 启动皮带")
        return
    if _set_conveyor_run(runtime, True, reason="DI2 start"):
        _set_button_box_lights(runtime, "start", reason="DI2 start")
        runtime._update_status("DI2 启动皮带")
        return
    runtime.warningOccurred.emit("IO未就绪，无法响应DI2启动皮带")


@QtCore.Slot()
def _stop_conveyor_from_di(runtime) -> None:
    controller = getattr(runtime, "_io_controller", None)
    if _apply_io_logic_event(runtime, "stop_button_pressed", controller=controller):
        runtime._update_status("DI3 停止皮带")
        return
    if _set_conveyor_run(runtime, False, reason="DI3 stop"):
        _set_button_box_lights(runtime, "stop", reason="DI3 stop")
        runtime._update_status("DI3 停止皮带")
        return
    runtime.warningOccurred.emit("IO未就绪，无法响应DI3停止皮带")


@QtCore.Slot()
def _handle_reset_button_from_di(runtime) -> None:
    controller = getattr(runtime, "_io_controller", None)
    if _apply_io_logic_event(runtime, "reset_button_pressed", controller=controller):
        runtime._update_status("DI4 复位按钮按下")
        return
    buzzer_changed = _set_buzzer(runtime, False, reason="DI4 reset")
    lights_changed = _set_button_box_lights(runtime, "reset", reason="DI4 reset")
    if buzzer_changed or lights_changed:
        runtime._update_status("DI4 复位按钮按下")
        return
    runtime.warningOccurred.emit("IO未就绪，无法响应DI4复位按钮")


@QtCore.Slot()
def _handle_reset_button_release_from_di(runtime) -> None:
    controller = getattr(runtime, "_io_controller", None)
    if _apply_io_logic_event(runtime, "reset_button_released", controller=controller):
        runtime._update_status("DI4 复位按钮松开")
        return
    if _set_named_output(runtime, "button_blue", False, controller=controller, reason="DI4 reset release"):
        runtime._update_status("DI4 复位按钮松开")
        return
    runtime.warningOccurred.emit("IO未就绪，无法响应DI4复位按钮释放")


@QtCore.Slot()
def _toggle_conveyor_run_from_di(runtime) -> None:
    controller = getattr(runtime, "_io_controller", None)
    current_running = bool(getattr(runtime, "_conveyor_running", False))
    if controller is not None and getattr(controller, "is_open", False):
        output_name = _resolve_output_name(runtime, controller, *_CONVEYOR_RUN_OUTPUT_CANDIDATES)
        if output_name:
            try:
                current_running = bool(controller.read_output(output_name))
            except Exception:
                pass
    target_running = not current_running
    if _set_conveyor_run(runtime, target_running, reason="DI2 toggle"):
        if target_running:
            _set_button_box_lights(runtime, "start", reason="DI2 toggle")
        else:
            _set_button_box_lights(runtime, "stop", reason="DI2 toggle")
        runtime._update_status("DI2 启动皮带" if target_running else "DI2 停止皮带")
        return
    runtime.warningOccurred.emit("IO未就绪，无法响应DI2皮带启停")


def _find_nkio_config_path(runtime) -> Optional[Path]:
    repo_root = packaged_repo_root(__file__)
    select_ini = repo_root / "NKDIOLC_SDK" / "Bin" / "select.ini"
    if select_ini.exists():
        parser = configparser.ConfigParser()
        parser.optionxform = str
        try:
            parser.read(select_ini, encoding="utf-8")
        except Exception:
            parser = None
        if parser is not None and parser.has_section("SELECTED"):
            config_path = str(parser.get("SELECTED", "ConfigPath", fallback="") or "").strip()
            if config_path:
                relative_path = config_path.lstrip("/\\").replace("/", "\\")
                candidate = repo_root / "NKDIOLC_SDK" / "Bin" / Path(relative_path)
                if candidate.exists():
                    return candidate

    candidates = [
        repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Debug" / "NP-6133-16I16O" / "nkio_config.ini",
        repo_root / "NKDIOLC_SDK" / "Bin" / "NP-6133-16I16O" / "nkio_config.ini",
        repo_root / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
        repo_root / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
        repo_root / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _matching_runtime_roles_by_serial(runtime, serial: str) -> list[str]:
    serial_text = str(serial).strip()
    if not serial_text or runtime._frame_grab_service is None:
        return []
    matched_roles: list[str] = []
    for role in runtime._connected_roles():
        try:
            device = runtime._frame_grab_service.get_device(role)
        except Exception:
            continue
        if str(getattr(device, "serial_number", "") or "").strip() == serial_text:
            matched_roles.append(role)
    return matched_roles


def _apply_camera_settings_now(
    runtime,
    serial: str,
    settings_payload,
    *,
    matched_roles: Optional[list[str]] = None,
) -> None:
    from . import controller as runtime_controller_module

    if runtime._frame_grab_service is None or runtime_controller_module.HikCameraSettings is None:
        return
    roles = matched_roles if matched_roles is not None else runtime._matching_runtime_roles_by_serial(serial)
    if not roles:
        return
    for role in roles:
        effective_payload = (
            dict(settings_payload)
            if settings_payload
            else (runtime._camera_settings_store.load_for_role(role, serial=serial) or {})
        )
        if hasattr(runtime._light_controller, "set_camera_light_mode"):
            camera_index = 1 if role == "cam1" else 2 if role == "cam2" else 0
            if camera_index > 0:
                runtime._light_controller.set_camera_light_mode(
                    camera_index,
                    runtime_controller_module.light_source_mode_from_mapping(effective_payload),
                )
        settings = runtime_controller_module.HikCameraSettings(
            **runtime_controller_module.hik_settings_kwargs_from_mapping(
                effective_payload,
                default_trigger_mode="continuous",
            )
        )
        device = runtime._frame_grab_service.get_device(role)
        device.apply_settings(settings)


def reset_all_camera_triggers_off(runtime) -> None:
    from . import controller as runtime_controller_module

    if (
        runtime._import_error is not None
        or runtime_controller_module.HikCameraManager is None
        or runtime_controller_module.HikCameraSettings is None
    ):
        return

    manager = runtime_controller_module.HikCameraManager()
    failures: list[str] = []
    try:
        infos = manager.enumerate_cameras()
        for info in infos:
            serial_text = str(info.serial_number or "").strip()
            if not serial_text:
                continue
            saved_settings = runtime._camera_settings_store.load_for_serial(serial_text) or {}
            settings = runtime_controller_module.HikCameraSettings(
                **runtime_controller_module.hik_settings_kwargs_from_mapping(
                    saved_settings,
                    default_trigger_mode="continuous",
                    force_trigger_mode="continuous",
                )
            )
            try:
                device = manager.open_camera(serial_text, settings=settings)
                device.close()
            except Exception as exc:
                failures.append(f"{serial_text}: {exc}")
    except Exception as exc:
        runtime.logAppended.emit(f"[camera] startup trigger reset failed: {exc}")
        runtime._update_status(f"startup trigger reset failed: {exc}")
        return
    finally:
        try:
            manager.close()
        except Exception:
            pass

    if failures:
        runtime.logAppended.emit("[camera] startup trigger reset partial failure")
        runtime._update_status("startup trigger reset partial failure")
        return

    runtime.logAppended.emit("[camera] startup trigger reset complete")
    runtime._update_status("startup camera trigger=off")


def _set_busy(runtime, busy: bool) -> None:
    runtime._busy = bool(busy)
    runtime.busyChanged.emit(runtime._busy)
    if not runtime._busy:
        runtime._flush_pending_camera_settings()


def _flush_pending_camera_settings(runtime) -> None:
    if not runtime._pending_camera_settings_by_serial:
        return
    pending = dict(runtime._pending_camera_settings_by_serial)
    runtime._pending_camera_settings_by_serial.clear()
    for serial_text, payload in pending.items():
        matched_roles = runtime._matching_runtime_roles_by_serial(serial_text)
        if not matched_roles:
            continue
        try:
            runtime._apply_camera_settings_now(serial_text, payload, matched_roles=matched_roles)
        except Exception as exc:
            runtime.errorOccurred.emit(f"Failed to sync runtime camera settings: {exc}")
            runtime.logAppended.emit(f"[camera] runtime settings sync failed for {serial_text}: {exc}")
            runtime._update_status(f"camera settings sync failed: {exc}")
            continue
        runtime.logAppended.emit(f"[camera] runtime settings synced for {serial_text}")
        runtime._update_status(f"camera settings synced: {serial_text}")
