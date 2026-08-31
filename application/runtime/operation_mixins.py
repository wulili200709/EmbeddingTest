"""Explicit compatibility mixins for runtime operations being migrated to services.

Unlike the previous binding registry, these methods are part of the controller's
static MRO and are visible to introspection, IDE navigation, and type tooling.
"""

from __future__ import annotations

class RuntimeConveyorMixin:
    def _load_conveyor_config(self):
        return self._conveyor_service.load_config()

    def _initialize_conveyor_controller(self, input_snapshot=None) -> bool:
        return self._conveyor_service.initialize(input_snapshot)

    def _shutdown_conveyor_controller(self) -> None:
        self._conveyor_service.shutdown()

    def _wait_for_conveyor_inspections(self) -> None:
        self._conveyor_service.wait_for_inspections()

    def _publish_conveyor_state(self, snapshot) -> None:
        self._conveyor_service.publish_state(snapshot)

    def _tick_conveyor(self) -> None:
        self._conveyor_service.tick()

    def _handle_conveyor_di_event(self, name: str, state: bool) -> None:
        self._conveyor_service.handle_di_event(name, state)

    def _handle_conveyor_io_error(self, name: str, detail: str) -> None:
        self._conveyor_service.handle_io_error(name, detail)

    def _enqueue_conveyor_inspection(self, sequence_id: int, epoch: int) -> None:
        self._conveyor_service.enqueue_inspection(sequence_id, epoch)

    def _prepare_conveyor_start(self) -> tuple[bool, str]:
        return self._conveyor_service.prepare_start()

    def _run_conveyor_capture(self, sequence_id: int, epoch: int):
        return self._conveyor_service.run_capture(sequence_id, epoch)

    def _on_conveyor_capture_task_finished(self, sequence_id: int, epoch: int, done) -> None:
        self._conveyor_service.capture_task_finished(sequence_id, epoch, done)

    def _run_conveyor_inspection(self, sequence_id: int, epoch: int, captured):
        return self._conveyor_service.run_inspection(sequence_id, epoch, captured)

    def _on_conveyor_inspection_task_finished(self, sequence_id: int, epoch: int, done) -> None:
        self._conveyor_service.inspection_task_finished(sequence_id, epoch, done)

    def start_conveyor(self):
        return self._conveyor_service.start()

    def stop_conveyor(self):
        return self._conveyor_service.stop()

    def start_conveyor_purge(self):
        return self._conveyor_service.start_purge()

    def continue_conveyor_purge(self):
        return self._conveyor_service.continue_purge()

    def acknowledge_conveyor_alarm(self):
        return self._conveyor_service.acknowledge_alarm()


class RuntimeHardwareMixin:
    def _rebuild_runner(self) -> bool:
        return self._hardware_service.rebuild_runner()

    def _emit_io_status(self, ready: bool, detail: str, controller=None) -> None:
        self._hardware_service.emit_io_status(ready, detail, controller)

    def _try_create_io_controller(self):
        return self._hardware_service.try_create_io_controller()

    def _initialize_startup_io(self, *, force: bool = False) -> bool:
        return self._hardware_service.initialize_startup_io(force=force)

    def _close_io_controller(self) -> None:
        self._hardware_service.close_io_controller()

    def _start_di_poller_if_available(self) -> bool:
        return self._hardware_service.start_di_poller()

    def _stop_di_poller(self) -> None:
        self._hardware_service.stop_di_poller()

    def _on_foot_switch_rising(self, event) -> None:
        self._hardware_service.on_foot_switch_rising(event)

    def _on_conveyor_di_change(self, event) -> None:
        self._hardware_service.on_conveyor_di_change(event)

    def _on_conveyor_io_error(self, name: str, error: Exception) -> None:
        self._hardware_service.on_conveyor_io_error(name, error)

    def _trigger_from_di(self) -> None:
        self._hardware_service.trigger_from_di()

    def _find_nkio_config_path(self):
        return self._hardware_service.find_nkio_config_path()

    def _matching_runtime_roles_by_serial(self, serial: str):
        return self._hardware_service.matching_runtime_roles_by_serial(serial)

    def _apply_camera_settings_now(self, serial: str, settings_payload, *, matched_roles=None):
        return self._hardware_service.apply_camera_settings_now(
            serial,
            settings_payload,
            matched_roles=matched_roles,
        )

    def reset_all_camera_triggers_off(self):
        return self._hardware_service.reset_all_camera_triggers_off()

    def _set_busy(self, busy: bool) -> None:
        self._hardware_service.set_busy(busy)

    def _flush_pending_camera_settings(self) -> None:
        self._hardware_service.flush_pending_camera_settings()


class RuntimeExecutionMixin:
    def _finalize_trigger_outcome(self, outcome, release_status_before, *, active_roles=None):
        return self._execution_service.finalize_trigger_outcome(
            outcome,
            release_status_before,
            active_roles=active_roles,
        )

    def _run_single_multi_light_trigger(self, requested_roles=None):
        return self._execution_service.run_single_multi_light_trigger(requested_roles)

    def _precheck(self):
        return self._execution_service.precheck()

    def _precheck_for_roles(self, roles) -> tuple[bool, str]:
        return self._execution_service.precheck_for_roles(roles)

    def _save_frame(self, role: str, image):
        return self._execution_service.save_frame(role, image)

    def _inspect_frame(self, role: str, frame, *, physical_role: str = ""):
        return self._execution_service.inspect_frame(
            role,
            frame,
            physical_role=physical_role,
        )

    def _write_release_log(self, *, event_type: str, result: str, message: str = "") -> None:
        self._execution_service.write_release_log(
            event_type=event_type,
            result=result,
            message=message,
        )

    def _write_runtime_record(self, runtime_result) -> None:
        self._execution_service.write_runtime_record(runtime_result)

    def _reload_runtime_context(self) -> None:
        self._execution_service.reload_runtime_context()


__all__ = ["RuntimeConveyorMixin", "RuntimeExecutionMixin", "RuntimeHardwareMixin"]
