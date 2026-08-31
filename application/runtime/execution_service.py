from __future__ import annotations

from . import execution


class RuntimeExecutionService:
    """Object-composed facade for manual/runtime inspection operations."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def finalize_trigger_outcome(self, outcome, release_status_before, *, active_roles=None):
        return execution._finalize_trigger_outcome(
            self._runtime,
            outcome,
            release_status_before,
            active_roles=active_roles,
        )

    def run_single_multi_light_trigger(self, requested_roles=None):
        return execution._run_single_multi_light_trigger(self._runtime, requested_roles)

    def precheck(self):
        return execution._precheck(self._runtime)

    def precheck_for_roles(self, roles) -> tuple[bool, str]:
        return execution._precheck_for_roles(self._runtime, roles)

    def save_frame(self, role: str, image):
        return execution._save_frame(self._runtime, role, image)

    def inspect_frame(self, role: str, frame, *, physical_role: str = ""):
        return execution._inspect_frame(
            self._runtime,
            role,
            frame,
            physical_role=physical_role,
        )

    def write_release_log(self, *, event_type: str, result: str, message: str = "") -> None:
        execution._write_release_log(
            self._runtime,
            event_type=event_type,
            result=result,
            message=message,
        )

    def write_runtime_record(self, runtime_result) -> None:
        execution._write_runtime_record(self._runtime, runtime_result)

    def reload_runtime_context(self) -> None:
        execution._reload_runtime_context(self._runtime)


__all__ = ["RuntimeExecutionService"]
