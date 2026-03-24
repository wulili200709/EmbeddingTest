from __future__ import annotations

from enum import Enum


class RunState(str, Enum):
    WaitingTrigger = "WaitingTrigger"
    ReleasedPendingConsume = "ReleasedPendingConsume"
    CapturingCam1 = "CapturingCam1"
    CapturingCam2 = "CapturingCam2"
    Inspecting = "Inspecting"
    Aggregating = "Aggregating"
    CompletedOk = "CompletedOk"
    CompletedNg = "CompletedNg"
    LockedByNg = "LockedByNg"
    Error = "Error"


class RunStateMachine:
    """Lightweight runtime state holder with guarded transitions."""

    _ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
        RunState.WaitingTrigger: {
            RunState.CapturingCam1,
            RunState.CapturingCam2,
            RunState.Inspecting,
            RunState.LockedByNg,
            RunState.Error,
            RunState.ReleasedPendingConsume,
        },
        RunState.ReleasedPendingConsume: {
            RunState.CapturingCam1,
            RunState.CapturingCam2,
            RunState.Inspecting,
            RunState.LockedByNg,
            RunState.Error,
            RunState.WaitingTrigger,
        },
        RunState.CapturingCam1: {
            RunState.CapturingCam2,
            RunState.Inspecting,
            RunState.Error,
        },
        RunState.CapturingCam2: {
            RunState.Inspecting,
            RunState.Error,
        },
        RunState.Inspecting: {
            RunState.Aggregating,
            RunState.Error,
        },
        RunState.Aggregating: {
            RunState.CompletedOk,
            RunState.CompletedNg,
            RunState.Error,
        },
        RunState.CompletedOk: {
            RunState.WaitingTrigger,
            RunState.Error,
        },
        RunState.CompletedNg: {
            RunState.LockedByNg,
            RunState.WaitingTrigger,
            RunState.Error,
        },
        RunState.LockedByNg: {
            RunState.ReleasedPendingConsume,
            RunState.Error,
        },
        RunState.Error: {
            RunState.WaitingTrigger,
            RunState.LockedByNg,
        },
    }

    def __init__(self, initial_state: RunState = RunState.WaitingTrigger) -> None:
        self._state = initial_state

    @property
    def state(self) -> RunState:
        return self._state

    def can_transition_to(self, new_state: RunState) -> bool:
        return new_state in self._ALLOWED_TRANSITIONS.get(self._state, set())

    def transition_to(self, new_state: RunState) -> None:
        if new_state == self._state:
            return
        if not self.can_transition_to(new_state):
            raise RuntimeError(f"invalid run-state transition: {self._state} -> {new_state}")
        self._state = new_state
