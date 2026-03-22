from __future__ import annotations

from dataclasses import dataclass

from .permission_manager import PermissionManager
from .run_state import RunState, RunStateMachine


@dataclass(frozen=True)
class StartDecision:
    allowed: bool
    reason: str = ""


class InspectionScheduler:
    """Coordinate runtime state, trigger gating, and NG release consumption."""

    def __init__(
        self,
        state_machine: RunStateMachine,
        permission_manager: PermissionManager,
    ) -> None:
        self.state_machine = state_machine
        self.permission_manager = permission_manager

    @property
    def state(self) -> RunState:
        return self.state_machine.state

    def can_accept_trigger(self) -> StartDecision:
        if self.permission_manager.is_locked:
            return StartDecision(False, "system is locked by NG")
        if self.state not in {RunState.WaitingTrigger, RunState.ReleasedPendingConsume}:
            return StartDecision(False, f"state={self.state} does not allow a new trigger")
        return StartDecision(True)

    def try_release_ng_lock(self, password: str) -> bool:
        if not self.permission_manager.try_release_once(password):
            return False
        if self.state == RunState.LockedByNg:
            self.state_machine.transition_to(RunState.ReleasedPendingConsume)
        return True

    def begin_precheck(self) -> StartDecision:
        return self.can_accept_trigger()

    def on_precheck_failed(self) -> None:
        if self.state == RunState.ReleasedPendingConsume:
            return
        if self.state == RunState.Error:
            self.state_machine.transition_to(RunState.WaitingTrigger)

    def on_capture_started(self, camera_index: int) -> None:
        self._consume_release_if_needed()
        if int(camera_index) == 1:
            self.state_machine.transition_to(RunState.CapturingCam1)
            return
        if int(camera_index) == 2:
            self.state_machine.transition_to(RunState.CapturingCam2)
            return
        raise ValueError(f"unsupported camera index: {camera_index}")

    def on_inspecting_started(self) -> None:
        self._consume_release_if_needed()
        self.state_machine.transition_to(RunState.Inspecting)

    def on_aggregating_started(self) -> None:
        self.state_machine.transition_to(RunState.Aggregating)

    def on_completed(self, *, final_ok: bool) -> None:
        if final_ok:
            self.state_machine.transition_to(RunState.CompletedOk)
            self.permission_manager.reset_after_success()
            self.state_machine.transition_to(RunState.WaitingTrigger)
            return

        self.state_machine.transition_to(RunState.CompletedNg)
        self.permission_manager.lock_for_ng()
        self.state_machine.transition_to(RunState.LockedByNg)

    def on_error(self, *, lock_as_ng: bool = True) -> None:
        if self.state != RunState.Error:
            self.state_machine.transition_to(RunState.Error)
        if lock_as_ng:
            self.permission_manager.lock_for_ng()
            self.state_machine.transition_to(RunState.LockedByNg)
        else:
            self.permission_manager.restore_pending_release()
            self.state_machine.transition_to(RunState.WaitingTrigger)

    def reset_to_waiting(self) -> None:
        if self.permission_manager.is_locked:
            raise RuntimeError("cannot reset to waiting while system is locked")
        if self.state == RunState.Error:
            self.state_machine.transition_to(RunState.WaitingTrigger)

    def _consume_release_if_needed(self) -> None:
        consumed = self.permission_manager.consume_release_if_needed()
        if consumed and self.state == RunState.ReleasedPendingConsume:
            return
