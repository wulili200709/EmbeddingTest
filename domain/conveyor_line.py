from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Deque, Mapping


class ConveyorState(str, Enum):
    STARTING = "STARTING"
    SAFETY_LOCKED = "SAFETY_LOCKED"
    DOOR_OPEN_STOPPED = "DOOR_OPEN_STOPPED"
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    CONTROLLED_STOPPING = "CONTROLLED_STOPPING"
    SAFETY_PAUSED = "SAFETY_PAUSED"
    DOOR_PAUSED = "DOOR_PAUSED"
    PURGING = "PURGING"
    PURGE_PAUSED = "PURGE_PAUSED"
    FAULT = "FAULT"


class InspectionStatus(str, Enum):
    PENDING = "PENDING"
    GOOD = "GOOD"
    NG = "NG"
    ERROR = "ERROR"
    PURGED = "PURGED"


@dataclass(frozen=True)
class ConveyorConfig:
    poll_interval_ms: int = 10
    debounce_ms: int = 20
    capture_commit_guard_ms: int = 250
    controlled_stop_timeout_ms: int = 1500
    reject_delay_ms: int = 0
    reject_duration_ms: int = 300
    fifo_max_items: int = 128
    item_to_reject_timeout_s: float = 10.0
    end_sensor_enabled: bool = True
    upper_door_sensor_enabled: bool = False
    end_sensor_jam_s: float = 3.0
    good_outlet_jam_s: float = 3.0
    waste_outlet_jam_s: float = 3.0
    purge_air_lead_ms: int = 200
    purge_min_run_s: float = 10.0
    purge_tail_run_s: float = 5.0
    purge_quiet_s: float = 2.0
    purge_max_run_s: float = 30.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "ConveyorConfig":
        values = dict(payload or {})
        known = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in known if key in values})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ConveyorConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("conveyor control config must be a JSON object")
        return cls.from_mapping(payload)


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


@dataclass
class _RejectWindow:
    sequence_id: int
    start_motion_s: float
    end_motion_s: float


@dataclass
class _PurgeContext:
    requested_at: float
    conveyor_started_at: float | None = None
    last_activity_at: float | None = None
    waste_clear_since: float | None = None


OutputWriter = Callable[[str, bool], None]
InspectionRequester = Callable[[int, int], None]
StateListener = Callable[[dict[str, object]], None]
LogWriter = Callable[[str], None]
StartAuthorizer = Callable[[], tuple[bool, str]]
InspectionResultListener = Callable[[int, int, str, str], None]


class ConveyorLineController:
    """Deterministic owner of conveyor state, FIFO and actuator decisions.

    All public methods are thread-safe, but the intended integration is to call
    them on one Qt/control thread. DI and inspection workers publish events only;
    they must never write conveyor outputs directly.
    """

    MATERIAL_INPUTS = (
        "camera_trigger_sensor",
        "reject_position_sensor",
        "end_test_sensor",
        "good_outlet_sensor",
        "waste_outlet_sensor",
    )
    JAM_INPUTS = (
        "end_test_sensor",
        "good_outlet_sensor",
        "waste_outlet_sensor",
    )
    DOOR_INPUTS = (
        "door_closed",
        "door_upper_closed",
    )
    CONTROL_OUTPUTS = (
        "conveyor_run",
        "waste_removal",
        "button_green",
        "button_red",
        "button_blue",
        "buzzer",
    )

    def __init__(
        self,
        *,
        config: ConveyorConfig | None = None,
        output_writer: OutputWriter,
        inspection_requester: InspectionRequester,
        state_listener: StateListener | None = None,
        log_writer: LogWriter | None = None,
        start_authorizer: StartAuthorizer | None = None,
        inspection_result_listener: InspectionResultListener | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or ConveyorConfig()
        self._write_output_callback = output_writer
        self._request_inspection_callback = inspection_requester
        self._state_listener = state_listener
        self._log_writer = log_writer
        self._start_authorizer = start_authorizer
        self._inspection_result_listener = inspection_result_listener
        self._clock = clock
        self._lock = threading.RLock()

        self.state = ConveyorState.STARTING
        self.inputs: dict[str, bool] = {}
        # Start unknown so initialization actively writes every safety-related
        # OFF/indicator state instead of assuming the board already matches.
        self.outputs: dict[str, bool] = {}
        self.fifo: Deque[WorkpieceRecord] = deque()
        self._records_by_id: dict[int, WorkpieceRecord] = {}
        self._reject_windows: list[_RejectWindow] = []
        self._active_captures: set[tuple[int, int]] = set()
        self._sequence = 0
        self._epoch = 1
        self._io_ready = False
        self._fault_code = ""
        self._fault_detail = ""
        self._controlled_stop_started_at: float | None = None
        self._last_capture_edge_at: float | None = None
        self._purge: _PurgeContext | None = None
        self._resume_purge_after_interlock = False
        self._motion_elapsed_s = 0.0
        self._last_tick_at = self._clock()
        self._jam_active_motion_since: dict[str, float | None] = {
            name: None for name in self.JAM_INPUTS
        }
        self._apply_indicator_outputs()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def run_permitted(self) -> bool:
        return (
            self._io_ready
            and self.inputs.get("safety_ok", False)
            and self._doors_closed()
            and not self._fault_code
        )

    def initialize_inputs(
        self,
        inputs: Mapping[str, bool],
        *,
        io_ready: bool = True,
        now: float | None = None,
    ) -> None:
        with self._lock:
            current = self._now(now)
            self._last_tick_at = current
            self.inputs = {str(name): bool(value) for name, value in inputs.items()}
            self._io_ready = bool(io_ready)
            self._fault_code = ""
            self._fault_detail = ""
            self._force_motion_off()
            self._select_nonrunning_interlock_state()
            self._publish()

    def set_io_ready(self, ready: bool, *, detail: str = "", now: float | None = None) -> None:
        with self._lock:
            self.tick(now)
            self._io_ready = bool(ready)
            if not ready:
                self._trip_fault("IO_NOT_READY", detail or "IO controller is not ready")
            elif self._fault_code == "IO_NOT_READY":
                self._fault_code = ""
                self._fault_detail = ""
                self._select_nonrunning_interlock_state()
            self._publish()

    def handle_input_change(self, name: str, state: bool, *, now: float | None = None) -> None:
        with self._lock:
            current = self._now(now)
            self.tick(current)
            input_name = str(name)
            previous = self.inputs.get(input_name)
            business_state = bool(state)
            self.inputs[input_name] = business_state
            if previous is business_state:
                return

            if input_name in self.MATERIAL_INPUTS:
                self._record_material_activity(current)
            if input_name in self.JAM_INPUTS:
                self._jam_active_motion_since[input_name] = (
                    self._motion_elapsed_s
                    if business_state and self.outputs.get("conveyor_run", False)
                    else None
                )

            if input_name == "safety_ok" or input_name in self.DOOR_INPUTS:
                self._handle_interlock_change(current)
            elif business_state and input_name == "start_button":
                self.request_start(now=current)
            elif business_state and input_name == "stop_button":
                self.request_controlled_stop(now=current)
            elif business_state and input_name == "camera_trigger_sensor":
                self._on_camera_sensor(current)
            elif business_state and input_name == "reject_position_sensor":
                self._on_reject_sensor(current)
            self._apply_indicator_outputs()
            self._publish()

    def request_start(self, *, now: float | None = None) -> bool:
        with self._lock:
            self.tick(now)
            if not self.run_permitted:
                self._log("start rejected: safety/door/IO/alarm permission is not satisfied")
                self._apply_indicator_outputs()
                self._publish()
                return False
            if self._start_authorizer is not None:
                allowed, reason = self._start_authorizer()
                if not allowed:
                    self._log(f"start rejected: {reason or 'inspection precheck failed'}")
                    self._publish()
                    return False
            if self.state == ConveyorState.PURGE_PAUSED and self._purge is not None:
                return self.continue_purge(now=now)
            if self.state not in {
                ConveyorState.STOPPED,
                ConveyorState.SAFETY_PAUSED,
                ConveyorState.DOOR_PAUSED,
            }:
                return self.state == ConveyorState.RUNNING
            self.state = ConveyorState.RUNNING
            self._controlled_stop_started_at = None
            self._set_output("conveyor_run", True)
            self._apply_indicator_outputs()
            self._log("production started")
            self._publish()
            return True

    def request_controlled_stop(self, *, now: float | None = None) -> bool:
        with self._lock:
            current = self._now(now)
            self.tick(current)
            if self.state == ConveyorState.PURGING:
                self._pause_purge("operator stop")
                self._publish()
                return True
            if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}:
                self._force_motion_off()
                self._apply_indicator_outputs()
                self._publish()
                return False
            if self.state != ConveyorState.CONTROLLED_STOPPING:
                self.state = ConveyorState.CONTROLLED_STOPPING
                self._controlled_stop_started_at = current
                self._log("controlled stop requested")
            self._evaluate_controlled_stop(current)
            self._apply_indicator_outputs()
            self._publish()
            return True

    def request_purge(self, *, now: float | None = None) -> bool:
        with self._lock:
            current = self._now(now)
            self.tick(current)
            if not (
                self._io_ready
                and self.inputs.get("safety_ok", False)
                and self._doors_closed()
            ):
                self._log("purge rejected: safety/door/IO permission is not satisfied")
                return False
            if self.state in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}:
                self._log("purge rejected: stop production first")
                return False

            self._fault_code = ""
            self._fault_detail = ""
            self._epoch += 1
            for record in self.fifo:
                record.inspection_status = InspectionStatus.PURGED
            self.fifo.clear()
            self._records_by_id.clear()
            self._reject_windows.clear()
            self._active_captures.clear()
            self._controlled_stop_started_at = None
            self._purge = _PurgeContext(
                requested_at=current,
                last_activity_at=current,
                waste_clear_since=current if not self.inputs.get("waste_outlet_sensor", False) else None,
            )
            self._resume_purge_after_interlock = False
            self.state = ConveyorState.PURGING
            self._set_output("conveyor_run", False)
            self._set_output("waste_removal", True)
            self._apply_indicator_outputs()
            self._log("one-click purge started; inspection results from the old epoch are invalid")
            self._publish()
            return True

    def continue_purge(self, *, now: float | None = None) -> bool:
        with self._lock:
            current = self._now(now)
            self.tick(current)
            if self.state != ConveyorState.PURGE_PAUSED or self._purge is None:
                return False
            if not self.run_permitted:
                return False
            self.state = ConveyorState.PURGING
            self._purge.requested_at = current
            self._purge.conveyor_started_at = None
            self._set_output("waste_removal", True)
            self._set_output("conveyor_run", False)
            self._resume_purge_after_interlock = False
            self._apply_indicator_outputs()
            self._log("purge continued after operator confirmation")
            self._publish()
            return True

    def acknowledge_alarm(self, *, now: float | None = None) -> bool:
        with self._lock:
            self.tick(now)
            if self.state != ConveyorState.FAULT:
                return False
            if not (
                self._io_ready
                and self.inputs.get("safety_ok", False)
                and self._doors_closed()
            ):
                return False
            self._fault_code = ""
            self._fault_detail = ""
            self._select_nonrunning_interlock_state()
            self._apply_indicator_outputs()
            self._log("alarm acknowledged")
            self._publish()
            return True

    def inspection_completed(
        self,
        sequence_id: int,
        epoch: int,
        result: str,
        *,
        detail: str = "",
        now: float | None = None,
    ) -> bool:
        with self._lock:
            current = self._now(now)
            self.tick(current)
            if int(epoch) != self._epoch:
                self._log(f"ignored stale inspection result: item={sequence_id}, epoch={epoch}")
                return False
            record = self._records_by_id.get(int(sequence_id))
            if record is None:
                self._log(f"ignored result for unknown/purged item: {sequence_id}")
                return False
            normalized = str(result or "").strip().upper()
            status = (
                InspectionStatus.GOOD
                if normalized in {"OK", "GOOD"}
                else InspectionStatus.NG
            )
            record.inspection_status = status
            record.inspection_detail = str(detail or "")
            record.result_at = current
            self._log(f"inspection completed: item={sequence_id}, result={status.value}")
            self._notify_inspection_result(record)
            self._publish()
            return True

    def capture_completed(self, sequence_id: int, epoch: int, *, now: float | None = None) -> bool:
        with self._lock:
            self.tick(now)
            key = (int(sequence_id), int(epoch))
            if key not in self._active_captures:
                return False
            self._active_captures.discard(key)
            self._publish()
            return True

    def tick(self, now: float | None = None) -> None:
        with self._lock:
            current = self._now(now)
            delta = max(0.0, current - self._last_tick_at)
            self._last_tick_at = current
            if self.outputs.get("conveyor_run", False):
                self._motion_elapsed_s += delta

            self._update_reject_output()
            self._monitor_jams()
            self._monitor_fifo_timeout()
            if self.state == ConveyorState.CONTROLLED_STOPPING:
                self._evaluate_controlled_stop(current)
            elif self.state == ConveyorState.PURGING:
                self._update_purge(current)
            self._apply_indicator_outputs()
            self._publish()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self.state.value,
                "run_permitted": self.run_permitted,
                "io_ready": self._io_ready,
                "safety_ok": self.inputs.get("safety_ok", False),
                "door_closed": self._doors_closed(),
                "door_lower_closed": self.inputs.get("door_closed", False),
                "door_upper_closed": self.inputs.get("door_upper_closed", False),
                "fault_code": self._fault_code,
                "fault_detail": self._fault_detail,
                "fifo_count": len(self.fifo),
                "capture_pending_count": len(self._active_captures),
                "fifo": [record.to_dict() for record in self.fifo],
                "epoch": self._epoch,
                "motion_elapsed_s": self._motion_elapsed_s,
                "outputs": dict(self.outputs),
                "inputs": dict(self.inputs),
                "purge_active": self.state == ConveyorState.PURGING,
                "purge_paused": self.state == ConveyorState.PURGE_PAUSED,
            }

    def shutdown(self) -> None:
        with self._lock:
            self._force_motion_off()
            self._set_output("buzzer", False)
            self.state = ConveyorState.STOPPED
            self._publish()

    def _on_camera_sensor(self, now: float) -> None:
        if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}:
            return
        if len(self.fifo) >= max(1, int(self.config.fifo_max_items)):
            self._trip_fault("FIFO_OVERFLOW", "in-flight workpiece queue is full")
            return
        self._sequence += 1
        record = WorkpieceRecord(
            sequence_id=self._sequence,
            epoch=self._epoch,
            created_at=now,
            created_motion_s=self._motion_elapsed_s,
        )
        self.fifo.append(record)
        self._records_by_id[record.sequence_id] = record
        self._active_captures.add((record.sequence_id, record.epoch))
        self._last_capture_edge_at = now
        self._log(f"camera sensor created item={record.sequence_id}, fifo={len(self.fifo)}")
        try:
            self._request_inspection_callback(record.sequence_id, record.epoch)
        except Exception as exc:
            self._active_captures.discard((record.sequence_id, record.epoch))
            record.inspection_status = InspectionStatus.NG
            record.inspection_detail = str(exc)
            record.result_at = now
            self._log(f"inspection enqueue failed; item={record.sequence_id} treated as NG: {exc}")
            self._notify_inspection_result(record)

    def _on_reject_sensor(self, now: float) -> None:
        if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}:
            return
        if not self.fifo:
            self._trip_fault("FIFO_UNDERFLOW", "DI1 triggered while FIFO is empty")
            return
        record = self.fifo[0]
        if record.inspection_status == InspectionStatus.PENDING:
            self._trip_fault(
                "RESULT_NOT_READY",
                f"item {record.sequence_id} reached DI1 before inspection completed",
            )
            return
        if record.inspection_status == InspectionStatus.ERROR:
            self._trip_fault(
                "INSPECTION_ERROR",
                f"item {record.sequence_id}: {record.inspection_detail or 'inspection failed'}",
            )
            return

        self.fifo.popleft()
        self._records_by_id.pop(record.sequence_id, None)
        if record.inspection_status == InspectionStatus.NG:
            start = self._motion_elapsed_s + max(0.0, self.config.reject_delay_ms / 1000.0)
            duration = max(0.001, self.config.reject_duration_ms / 1000.0)
            self._reject_windows.append(
                _RejectWindow(record.sequence_id, start, start + duration)
            )
            self._log(f"NG item={record.sequence_id} scheduled for blow-off")
        else:
            self._log(f"GOOD item={record.sequence_id} passed without blow-off")
        self._update_reject_output()

    def _update_reject_output(self) -> None:
        motion = self._motion_elapsed_s
        self._reject_windows = [window for window in self._reject_windows if window.end_motion_s > motion]
        should_blow = (
            self.state in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}
            and self.outputs.get("conveyor_run", False)
            and any(window.start_motion_s <= motion < window.end_motion_s for window in self._reject_windows)
        )
        if self.state != ConveyorState.PURGING:
            self._set_output("waste_removal", should_blow)

    def _evaluate_controlled_stop(self, now: float) -> None:
        started = self._controlled_stop_started_at or now
        timeout_s = max(0.0, self.config.controlled_stop_timeout_ms / 1000.0)
        if timeout_s and now - started >= timeout_s:
            self._trip_fault(
                "CONTROLLED_STOP_TIMEOUT",
                f"committed action did not finish within {self.config.controlled_stop_timeout_ms} ms",
            )
            return
        capture_guard_s = max(0.0, self.config.capture_commit_guard_ms / 1000.0)
        capture_committed = (
            bool(self._active_captures)
            or (
                self._last_capture_edge_at is not None
                and now < self._last_capture_edge_at + capture_guard_s
            )
        )
        reject_committed = bool(self._reject_windows)
        if capture_committed or reject_committed:
            return
        self._force_motion_off()
        self.state = ConveyorState.STOPPED
        self._controlled_stop_started_at = None
        self._log("controlled stop completed")

    def _handle_interlock_change(self, now: float) -> None:
        safety_ok = self.inputs.get("safety_ok", False)
        door_closed = self._doors_closed()
        if not safety_ok or not door_closed:
            was_purging = self.state in {ConveyorState.PURGING, ConveyorState.PURGE_PAUSED}
            was_moving = self.state in {
                ConveyorState.RUNNING,
                ConveyorState.CONTROLLED_STOPPING,
                ConveyorState.PURGING,
            }
            self._resume_purge_after_interlock = was_purging and self._purge is not None
            self._force_motion_off()
            if not safety_ok:
                self.state = ConveyorState.SAFETY_PAUSED if was_moving else ConveyorState.SAFETY_LOCKED
            else:
                self.state = ConveyorState.DOOR_PAUSED if was_moving else ConveyorState.DOOR_OPEN_STOPPED
            self._log("motion stopped immediately by safety interlock")
            return

        if self._resume_purge_after_interlock and self._purge is not None:
            self.state = ConveyorState.PURGE_PAUSED
        elif self._fault_code:
            self.state = ConveyorState.FAULT
        elif self.state in {
            ConveyorState.SAFETY_LOCKED,
            ConveyorState.SAFETY_PAUSED,
            ConveyorState.DOOR_OPEN_STOPPED,
            ConveyorState.DOOR_PAUSED,
        }:
            self.state = ConveyorState.STOPPED
        self._log("safety permission restored; manual restart is required")

    def _select_nonrunning_interlock_state(self) -> None:
        if not self._io_ready or not self.inputs.get("safety_ok", False):
            self.state = ConveyorState.SAFETY_LOCKED
        elif not self._doors_closed():
            self.state = ConveyorState.DOOR_OPEN_STOPPED
        elif self._fault_code:
            self.state = ConveyorState.FAULT
        else:
            self.state = ConveyorState.STOPPED
        self._apply_indicator_outputs()

    def _doors_closed(self) -> bool:
        if not self.inputs.get("door_closed", False):
            return False
        if not self.config.upper_door_sensor_enabled:
            return True
        return self.inputs.get("door_upper_closed", False)

    def _record_material_activity(self, now: float) -> None:
        if self._purge is None:
            return
        self._purge.last_activity_at = now
        if self.inputs.get("waste_outlet_sensor", False):
            self._purge.waste_clear_since = None

    def _update_purge(self, now: float) -> None:
        purge = self._purge
        if purge is None:
            self._trip_fault("PURGE_STATE_ERROR", "purge context is missing")
            return
        if now - purge.requested_at >= max(0.0, self.config.purge_max_run_s):
            self._trip_fault("PURGE_TIMEOUT", "one-click purge exceeded maximum run time")
            return
        lead_s = max(0.0, self.config.purge_air_lead_ms / 1000.0)
        if purge.conveyor_started_at is None and now - purge.requested_at >= lead_s:
            purge.conveyor_started_at = now
            self._set_output("conveyor_run", True)
            self._log("purge air lead complete; conveyor started")
        if purge.conveyor_started_at is None:
            return

        waste_active = self.inputs.get("waste_outlet_sensor", False)
        if waste_active:
            purge.waste_clear_since = None
        elif purge.waste_clear_since is None:
            purge.waste_clear_since = now

        run_s = now - purge.conveyor_started_at
        last_activity = purge.last_activity_at or purge.conveyor_started_at
        all_clear = not any(self.inputs.get(name, False) for name in self.MATERIAL_INPUTS)
        quiet_since = purge.waste_clear_since or now
        if (
            run_s >= max(0.0, self.config.purge_min_run_s)
            and all_clear
            and now - last_activity >= max(0.0, self.config.purge_tail_run_s)
            and now - quiet_since >= max(0.0, self.config.purge_quiet_s)
        ):
            self._force_motion_off()
            self._purge = None
            self.state = ConveyorState.STOPPED
            self._log("one-click purge completed")

    def _pause_purge(self, reason: str) -> None:
        self._force_motion_off()
        self.state = ConveyorState.PURGE_PAUSED
        self._resume_purge_after_interlock = True
        self._log(f"purge paused: {reason}; operator confirmation is required")

    def _monitor_jams(self) -> None:
        if not self.outputs.get("conveyor_run", False):
            for name in self.JAM_INPUTS:
                self._jam_active_motion_since[name] = None
            return
        thresholds = {
            "end_test_sensor": self.config.end_sensor_jam_s,
            "good_outlet_sensor": self.config.good_outlet_jam_s,
            "waste_outlet_sensor": self.config.waste_outlet_jam_s,
        }
        for name, threshold in thresholds.items():
            if name == "end_test_sensor" and not self.config.end_sensor_enabled:
                self._jam_active_motion_since[name] = None
                continue
            if not self.inputs.get(name, False):
                self._jam_active_motion_since[name] = None
                continue
            since = self._jam_active_motion_since.get(name)
            if since is None:
                self._jam_active_motion_since[name] = self._motion_elapsed_s
                continue
            if self._motion_elapsed_s - since >= max(0.001, float(threshold)):
                self._trip_fault("JAM_DETECTED", f"{name} remained active for {threshold} s")
                return

    def _monitor_fifo_timeout(self) -> None:
        if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING} or not self.fifo:
            return
        head = self.fifo[0]
        if self._motion_elapsed_s - head.created_motion_s >= max(0.1, self.config.item_to_reject_timeout_s):
            self._trip_fault(
                "ITEM_ARRIVAL_TIMEOUT",
                f"item {head.sequence_id} did not reach DI1 in time",
            )

    def _trip_fault(self, code: str, detail: str) -> None:
        self._fault_code = str(code)
        self._fault_detail = str(detail)
        self._reject_windows.clear()
        self._force_motion_off()
        self.state = ConveyorState.FAULT
        self._set_output("buzzer", True)
        self._apply_indicator_outputs()
        self._log(f"fault {self._fault_code}: {self._fault_detail}")

    def _force_motion_off(self) -> None:
        self._set_output("conveyor_run", False)
        self._set_output("waste_removal", False)

    def _apply_indicator_outputs(self) -> None:
        safety_ok = self.inputs.get("safety_ok", False)
        running = self.state in {ConveyorState.RUNNING, ConveyorState.PURGING}
        self._set_output("button_green", running)
        self._set_output("button_red", not running)
        self._set_output("button_blue", not safety_ok)
        self._set_output("buzzer", self.state == ConveyorState.FAULT)

    def _set_output(self, name: str, on: bool) -> None:
        value = bool(on)
        if self.outputs.get(name) == value:
            return
        self.outputs[name] = value
        try:
            self._write_output_callback(name, value)
        except Exception as exc:
            self._fault_code = "OUTPUT_WRITE_FAILED"
            self._fault_detail = f"{name}: {exc}"
            self.state = ConveyorState.FAULT
            for safe_name in ("conveyor_run", "waste_removal"):
                self.outputs[safe_name] = False
                try:
                    self._write_output_callback(safe_name, False)
                except Exception:
                    pass
            raise RuntimeError(self._fault_detail) from exc

    def _notify_inspection_result(self, record: WorkpieceRecord) -> None:
        callback = self._inspection_result_listener
        if callback is None:
            return
        try:
            callback(
                int(record.sequence_id),
                int(record.epoch),
                str(record.inspection_status.value),
                str(record.inspection_detail or ""),
            )
        except Exception as exc:
            self._log(f"inspection result notification failed: {exc}")

    def _publish(self) -> None:
        if self._state_listener is not None:
            self._state_listener(self.snapshot())

    def _log(self, message: str) -> None:
        if self._log_writer is not None:
            self._log_writer(str(message))

    def _now(self, value: float | None) -> float:
        return self._clock() if value is None else float(value)


__all__ = [
    "ConveyorConfig",
    "ConveyorLineController",
    "ConveyorState",
    "InspectionStatus",
    "WorkpieceRecord",
]
