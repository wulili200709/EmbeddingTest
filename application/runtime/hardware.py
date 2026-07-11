"""Hardware-oriented RuntimeController helpers."""

from __future__ import annotations

import configparser
import json
import os
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from common.app_paths import packaged_embedding_test_root, packaged_repo_root
from common.camera_roles import ROLE_TO_CAMERA_INDEX, camera_index_for_role
from .capture_policy import DEFAULT_LIGHT_STABLE_MS
from .capture_channels import physical_connected_roles


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


def _emit_io_status(runtime, ready: bool, detail: str, controller=None) -> None:
    runtime._io_ready = bool(ready)
    runtime._io_status_detail = str(detail or "")
    runtime.ioStatusChanged.emit(runtime._io_ready, runtime._io_status_detail, controller)


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
        runtime._light_controller = runtime_controller_module.LightController(
            runtime._io_controller,
            supported_cameras=ROLE_TO_CAMERA_INDEX.values(),
        )
        tower_settings = dict(getattr(runtime, "_tower_light_settings", {}) or {})
        runtime._tower_light_controller = runtime_controller_module.TowerLightController(
            runtime._io_controller,
            ok_flash_ms=int(tower_settings.get("ok_flash_ms", 200) or 200),
            ng_flash_ms=int(tower_settings.get("ng_flash_ms", 200) or 200),
            ng_buzzer_ms=int(tower_settings.get("ng_buzzer_ms", 500) or 0),
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
        role_to_camera_index=dict(ROLE_TO_CAMERA_INDEX),
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
            runtime._io_controller.close()
        except Exception:
            pass
    runtime._emit_io_status(False, runtime._io_status_detail or "IO closed")


def _start_di_poller_if_available(runtime) -> None:
    from . import controller as runtime_controller_module

    runtime._stop_di_poller()
    if runtime._io_controller is None or runtime_controller_module.DiMonitor is None:
        return
    try:
        poller = runtime_controller_module.DiMonitor(
            runtime._io_controller,
            input_names=["foot_switch"],
            poll_interval_ms=20,
            debounce_ms=50,
        )
        poller.add_rising_callback(runtime._on_foot_switch_rising)
        poller.start()
    except Exception as exc:
        runtime.logAppended.emit(f"[IO] failed to start foot-switch DI monitor: {exc}")
        return
    runtime._di_poller = poller
    runtime.logAppended.emit("[IO] foot-switch DI monitor started")


def _stop_di_poller(runtime) -> None:
    if runtime._di_poller is None:
        return
    try:
        runtime._di_poller.stop()
    except Exception:
        pass
    runtime._di_poller = None


def _on_foot_switch_rising(runtime, event) -> None:
    runtime.logAppended.emit(f"[foot-switch] rising edge detected: {event.name}")
    QtCore.QMetaObject.invokeMethod(runtime, "_trigger_from_di", QtCore.Qt.QueuedConnection)


@QtCore.Slot()
def _trigger_from_di(runtime) -> None:
    runtime.trigger()


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
    for role in physical_connected_roles(runtime):
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
            camera_index = camera_index_for_role(role)
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
