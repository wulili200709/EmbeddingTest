from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Deque, Iterable, Mapping


class InspectionStatus(str, Enum):
    PENDING = "PENDING"
    GOOD = "GOOD"
    NG = "NG"
    ERROR = "ERROR"
    PURGED = "PURGED"


@dataclass
class WorkpieceRecord:
    sequence_id: int
    epoch: int
    created_at: float
    created_motion_s: float
    inspection_status: InspectionStatus = InspectionStatus.PENDING
    inspection_detail: str = ""
    result_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["inspection_status"] = self.inspection_status.value
        return payload


class WorkpieceTracker:
    def __init__(self) -> None:
        self.fifo: Deque[WorkpieceRecord] = deque()
        self._records_by_id: dict[int, WorkpieceRecord] = {}
        self._sequence = 0
        self.epoch = 1

    def create(self, *, now: float, motion_s: float) -> WorkpieceRecord:
        self._sequence += 1
        record = WorkpieceRecord(
            sequence_id=self._sequence,
            epoch=self.epoch,
            created_at=float(now),
            created_motion_s=float(motion_s),
        )
        self.fifo.append(record)
        self._records_by_id[record.sequence_id] = record
        return record

    def get(self, sequence_id: int) -> WorkpieceRecord | None:
        return self._records_by_id.get(int(sequence_id))

    def head(self) -> WorkpieceRecord | None:
        return self.fifo[0] if self.fifo else None

    def pop_head(self) -> WorkpieceRecord | None:
        if not self.fifo:
            return None
        record = self.fifo.popleft()
        self._records_by_id.pop(record.sequence_id, None)
        return record

    def invalidate_for_purge(self) -> None:
        self.epoch += 1
        for record in self.fifo:
            record.inspection_status = InspectionStatus.PURGED
        self.fifo.clear()
        self._records_by_id.clear()


@dataclass
class OutletExpectation:
    sequence_id: int
    epoch: int
    inspection_status: InspectionStatus
    expected_input: str
    started_motion_s: float
    earliest_motion_s: float
    deadline_motion_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "epoch": self.epoch,
            "inspection_status": self.inspection_status.value,
            "expected_input": self.expected_input,
            "started_motion_s": self.started_motion_s,
            "earliest_motion_s": self.earliest_motion_s,
            "deadline_motion_s": self.deadline_motion_s,
            "tracking_status": (
                "WAITING_GOOD_OUTLET"
                if self.expected_input == "good_outlet_sensor"
                else "WAITING_WASTE_OUTLET"
            ),
        }


class OutletConfirmationTracker:
    """Track workpieces from DI1 until their expected outlet edge arrives."""

    def __init__(self) -> None:
        self.pending: list[OutletExpectation] = []

    def expect(
        self,
        record: WorkpieceRecord,
        *,
        expected_input: str,
        motion_s: float,
        min_run_s: float,
        max_run_s: float,
    ) -> OutletExpectation:
        started = float(motion_s)
        minimum = max(0.0, float(min_run_s))
        maximum = max(minimum + 0.001, float(max_run_s))
        expectation = OutletExpectation(
            sequence_id=int(record.sequence_id),
            epoch=int(record.epoch),
            inspection_status=record.inspection_status,
            expected_input=str(expected_input),
            started_motion_s=started,
            earliest_motion_s=started + minimum,
            deadline_motion_s=started + maximum,
        )
        self.pending.append(expectation)
        return expectation

    def confirm(
        self,
        input_name: str,
        *,
        motion_s: float,
    ) -> tuple[str, OutletExpectation | None]:
        """Return CONFIRMED, WRONG_OUTLET or UNEXPECTED for an outlet edge."""
        name = str(input_name)
        motion = float(motion_s)
        eligible = [
            item
            for item in self.pending
            if item.earliest_motion_s <= motion <= item.deadline_motion_s
        ]
        correct = next(
            (item for item in eligible if item.expected_input == name),
            None,
        )
        if correct is not None:
            self.pending.remove(correct)
            return "CONFIRMED", correct
        wrong = next(
            (item for item in eligible if item.expected_input != name),
            None,
        )
        if wrong is not None:
            return "WRONG_OUTLET", wrong
        return "UNEXPECTED", None

    def first_expired(self, motion_s: float) -> OutletExpectation | None:
        motion = float(motion_s)
        return next(
            (item for item in self.pending if motion > item.deadline_motion_s),
            None,
        )

    def count_for_input(self, input_name: str) -> int:
        name = str(input_name)
        return sum(1 for item in self.pending if item.expected_input == name)

    def clear(self) -> None:
        self.pending.clear()


@dataclass
class RejectWindow:
    sequence_id: int
    start_motion_s: float
    end_motion_s: float


class RejectBlowController:
    def __init__(self) -> None:
        self.windows: list[RejectWindow] = []

    def clear(self) -> None:
        self.windows.clear()

    def schedule(
        self,
        *,
        sequence_id: int,
        motion_s: float,
        delay_s: float,
        duration_s: float,
    ) -> None:
        start = float(motion_s) + max(0.0, float(delay_s))
        duration = max(0.001, float(duration_s))
        self.windows.append(RejectWindow(int(sequence_id), start, start + duration))

    def prune(self, motion_s: float) -> None:
        motion = float(motion_s)
        self.windows = [window for window in self.windows if window.end_motion_s > motion]

    def is_active(self, motion_s: float) -> bool:
        motion = float(motion_s)
        return any(
            window.start_motion_s <= motion < window.end_motion_s
            for window in self.windows
        )

    @property
    def has_pending(self) -> bool:
        return bool(self.windows)


@dataclass
class PurgeContext:
    requested_at: float
    conveyor_started_at: float | None = None
    last_activity_at: float | None = None
    waste_clear_since: float | None = None


class AutoPurgeController:
    def __init__(self) -> None:
        self.context: PurgeContext | None = None

    @property
    def active(self) -> bool:
        return self.context is not None

    def begin(self, *, now: float, waste_active: bool) -> None:
        self.context = PurgeContext(
            requested_at=float(now),
            last_activity_at=float(now),
            waste_clear_since=None if waste_active else float(now),
        )

    def restart_lead(self, *, now: float) -> bool:
        if self.context is None:
            return False
        self.context.requested_at = float(now)
        self.context.conveyor_started_at = None
        return True

    def clear(self) -> None:
        self.context = None

    def record_activity(self, *, now: float, waste_active: bool) -> None:
        if self.context is None:
            return
        self.context.last_activity_at = float(now)
        if waste_active:
            self.context.waste_clear_since = None

    def update_waste_state(self, *, now: float, waste_active: bool) -> None:
        if self.context is None:
            return
        if waste_active:
            self.context.waste_clear_since = None
        elif self.context.waste_clear_since is None:
            self.context.waste_clear_since = float(now)


class JamMonitor:
    def __init__(self, input_names: Iterable[str]) -> None:
        self._active_motion_since = {str(name): None for name in input_names}

    def observe_input(
        self,
        name: str,
        *,
        active: bool,
        motion_s: float,
        conveyor_running: bool,
    ) -> None:
        self._active_motion_since[str(name)] = (
            float(motion_s) if active and conveyor_running else None
        )

    def reset(self) -> None:
        for name in self._active_motion_since:
            self._active_motion_since[name] = None

    def first_timeout(
        self,
        *,
        inputs: Mapping[str, bool],
        motion_s: float,
        thresholds: Mapping[str, float],
        disabled_inputs: Iterable[str] = (),
    ) -> tuple[str, float] | None:
        disabled = {str(name) for name in disabled_inputs}
        motion = float(motion_s)
        for name, threshold in thresholds.items():
            if name in disabled or not inputs.get(name, False):
                self._active_motion_since[name] = None
                continue
            since = self._active_motion_since.get(name)
            if since is None:
                self._active_motion_since[name] = motion
                continue
            if motion - since >= max(0.001, float(threshold)):
                return name, float(threshold)
        return None


__all__ = [
    "AutoPurgeController",
    "InspectionStatus",
    "JamMonitor",
    "OutletConfirmationTracker",
    "OutletExpectation",
    "PurgeContext",
    "RejectBlowController",
    "RejectWindow",
    "WorkpieceRecord",
    "WorkpieceTracker",
]
