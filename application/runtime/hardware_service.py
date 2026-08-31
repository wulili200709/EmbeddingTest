from __future__ import annotations

from typing import Optional

from . import hardware


class RuntimeHardwareService:
    """Object-composed facade for runtime hardware operations."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def rebuild_runner(self) -> bool:
        return hardware._rebuild_runner(self._runtime)

    def emit_io_status(self, ready: bool, detail: str, controller=None) -> None:
        hardware._emit_io_status(self._runtime, ready, detail, controller)

    def try_create_io_controller(self):
        return hardware._try_create_io_controller(self._runtime)

    def initialize_startup_io(self, *, force: bool = False) -> bool:
        return hardware._initialize_startup_io(self._runtime, force=force)

    def close_io_controller(self) -> None:
        hardware._close_io_controller(self._runtime)

    def start_di_poller(self) -> bool:
        return hardware._start_di_poller_if_available(self._runtime)

    def stop_di_poller(self) -> None:
        hardware._stop_di_poller(self._runtime)

    def on_foot_switch_rising(self, event) -> None:
        hardware._on_foot_switch_rising(self._runtime, event)

    def on_conveyor_di_change(self, event) -> None:
        hardware._on_conveyor_di_change(self._runtime, event)

    def on_conveyor_io_error(self, name: str, error: Exception) -> None:
        hardware._on_conveyor_io_error(self._runtime, name, error)

    def trigger_from_di(self) -> None:
        hardware._trigger_from_di(self._runtime)

    def find_nkio_config_path(self):
        return hardware._find_nkio_config_path(self._runtime)

    def matching_runtime_roles_by_serial(self, serial: str):
        return hardware._matching_runtime_roles_by_serial(self._runtime, serial)

    def apply_camera_settings_now(
        self,
        serial: str,
        settings_payload,
        *,
        matched_roles: Optional[list[str]] = None,
    ):
        return hardware._apply_camera_settings_now(
            self._runtime,
            serial,
            settings_payload,
            matched_roles=matched_roles,
        )

    def reset_all_camera_triggers_off(self):
        return hardware.reset_all_camera_triggers_off(self._runtime)

    def set_busy(self, busy: bool) -> None:
        hardware._set_busy(self._runtime, busy)

    def flush_pending_camera_settings(self) -> None:
        hardware._flush_pending_camera_settings(self._runtime)


__all__ = ["RuntimeHardwareService"]
