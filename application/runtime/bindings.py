"""RuntimeController helper binding registry."""

from __future__ import annotations

from . import conveyor, execution, hardware, status_bus

_RUNTIME_CONTROLLER_BINDINGS = (
    (
        conveyor,
        (
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
            "_on_conveyor_inspection_task_finished",
            "start_conveyor",
            "stop_conveyor",
            "start_conveyor_purge",
            "continue_conveyor_purge",
            "acknowledge_conveyor_alarm",
        ),
    ),
    (
        hardware,
        (
            "_rebuild_runner",
            "_emit_io_status",
            "_try_create_io_controller",
            "_initialize_startup_io",
            "_close_io_controller",
            "_start_di_poller_if_available",
            "_stop_di_poller",
            "_on_foot_switch_rising",
            "_on_conveyor_di_change",
            "_on_conveyor_io_error",
            "_trigger_from_di",
            "_find_nkio_config_path",
            "_matching_runtime_roles_by_serial",
            "_apply_camera_settings_now",
            "reset_all_camera_triggers_off",
            "_set_busy",
            "_flush_pending_camera_settings",
        ),
    ),
    (
        execution,
        (
            "_finalize_trigger_outcome",
            "_run_single_multi_light_trigger",
            "_precheck",
            "_precheck_for_roles",
            "_save_frame",
            "_inspect_frame",
            "_write_release_log",
            "_write_runtime_record",
            "_reload_runtime_context",
        ),
    ),
    (
        status_bus,
        (
            "_update_status",
            "_connected_roles",
            "_required_roles",
            "_current_item_signature",
            "_result_item_signature",
            "_runtime_result_is_stale",
            "_emit_runtime_context",
            "_build_pending_runtime_result",
            "_current_runtime_state_text",
        ),
    ),
)


def bind_runtime_controller(cls):
    for module, names in _RUNTIME_CONTROLLER_BINDINGS:
        for name in names:
            setattr(cls, name, getattr(module, name))
    return cls
