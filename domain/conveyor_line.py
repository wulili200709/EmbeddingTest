from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from .conveyor_components import (
    AutoPurgeController,
    InspectionStatus,
    JamMonitor,
    RejectBlowController,
    WorkpieceRecord,
    WorkpieceTracker,
)


class ConveyorState(str, Enum):
    BOOTING = "BOOTING"
    SAFETY_LOCKED = "SAFETY_LOCKED"
    DOOR_OPEN_STOPPED = "DOOR_OPEN_STOPPED"
    READY_STOPPED = "READY_STOPPED"
    RUNNING = "RUNNING"
    CONTROLLED_STOPPING = "CONTROLLED_STOPPING"
    SAFETY_PAUSED = "SAFETY_PAUSED"
    DOOR_PAUSED = "DOOR_PAUSED"
    READY_TO_RESUME = "READY_TO_RESUME"
    PURGE_PREPARING = "PURGE_PREPARING"
    PURGE_RUNNING = "PURGE_RUNNING"
    PURGE_PAUSED = "PURGE_PAUSED"
    FAULT_STOPPED = "FAULT_STOPPED"


class FaultRecovery(str, Enum):
    ACKNOWLEDGE = "ACKNOWLEDGE"
    PURGE_REQUIRED = "PURGE_REQUIRED"
    RECONNECT_IO = "RECONNECT_IO"


@dataclass(frozen=True)
class ConveyorConfig:
    poll_interval_ms: int = 10
    debounce_ms: int = 20
    capture_commit_guard_ms: int = 250
    controlled_stop_timeout_ms: int = 1500
    reject_blow_delay_ms: int = 0
    reject_blow_duration_ms: int = 300
    max_inflight_items: int = 128
    front_to_reject_max_run_ms: int = 10000
    end_test_sensor_enabled: bool = True
    upper_door_sensor_enabled: bool = False
    end_test_jam_timeout_s: float = 3.0
    good_jam_timeout_s: float = 3.0
    waste_jam_timeout_s: float = 3.0
    purge_air_lead_ms: int = 200
    purge_min_run_s: float = 10.0
    purge_tail_run_s: float = 5.0
    purge_quiet_s: float = 2.0
    purge_max_run_s: float = 30.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "ConveyorConfig":
        values = dict(payload or {})
        aliases = {
            "reject_delay_ms": "reject_blow_delay_ms",
            "reject_duration_ms": "reject_blow_duration_ms",
            "fifo_max_items": "max_inflight_items",
            "end_sensor_enabled": "end_test_sensor_enabled",
            "end_sensor_jam_s": "end_test_jam_timeout_s",
            "good_outlet_jam_s": "good_jam_timeout_s",
            "waste_outlet_jam_s": "waste_jam_timeout_s",
        }
        for old_name, canonical_name in aliases.items():
            if canonical_name not in values and old_name in values:
                values[canonical_name] = values[old_name]
        if "front_to_reject_max_run_ms" not in values and "item_to_reject_timeout_s" in values:
            values["front_to_reject_max_run_ms"] = int(
                float(values["item_to_reject_timeout_s"]) * 1000.0
            )
        known = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in known if key in values})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ConveyorConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("conveyor control config must be a JSON object")
        return cls.from_mapping(payload)

    # Read-only aliases keep integrations using the previous attribute names
    # working while configuration files and implementation use the design names.
    @property
    def reject_delay_ms(self) -> int:
        return self.reject_blow_delay_ms

    @property
    def reject_duration_ms(self) -> int:
        return self.reject_blow_duration_ms

    @property
    def fifo_max_items(self) -> int:
        return self.max_inflight_items

    @property
    def item_to_reject_timeout_s(self) -> float:
        return self.front_to_reject_max_run_ms / 1000.0

    @property
    def end_sensor_enabled(self) -> bool:
        return self.end_test_sensor_enabled

    @property
    def end_sensor_jam_s(self) -> float:
        return self.end_test_jam_timeout_s

    @property
    def good_outlet_jam_s(self) -> float:
        return self.good_jam_timeout_s

    @property
    def waste_outlet_jam_s(self) -> float:
        return self.waste_jam_timeout_s


OutputWriter = Callable[[str, bool], None]
OutputBatchWriter = Callable[[Mapping[str, bool]], None]
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
        output_batch_writer: OutputBatchWriter | None = None,
        state_listener: StateListener | None = None,
        log_writer: LogWriter | None = None,
        start_authorizer: StartAuthorizer | None = None,
        inspection_result_listener: InspectionResultListener | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or ConveyorConfig()
        self._write_output_callback = output_writer
        self._write_outputs_callback = output_batch_writer
        self._request_inspection_callback = inspection_requester
        self._state_listener = state_listener
        self._log_writer = log_writer
        self._start_authorizer = start_authorizer
        self._inspection_result_listener = inspection_result_listener
        self._clock = clock
        self._lock = threading.RLock()

        self.state = ConveyorState.BOOTING
        self.inputs: dict[str, bool] = {}
        # Start unknown so initialization actively writes every safety-related
        # OFF/indicator state instead of assuming the board already matches.
        self.outputs: dict[str, bool] = {}
        self._tracker = WorkpieceTracker()
        self.fifo = self._tracker.fifo
        self._reject = RejectBlowController()
        self._active_captures: set[tuple[int, int]] = set()
        self._io_ready = False
        self._fault_code = ""
        self._fault_detail = ""
        self._fault_recovery = FaultRecovery.ACKNOWLEDGE
        self._fault_input = ""
        self._controlled_stop_started_at: float | None = None
        self._last_capture_edge_at: float | None = None
        self._purge = AutoPurgeController()
        self._resume_purge_after_interlock = False
        self._motion_elapsed_s = 0.0
        self._last_tick_at = self._clock()
        self._jam_monitor = JamMonitor(self.JAM_INPUTS)
        self._apply_indicator_outputs()

    @property
    def epoch(self) -> int:
        return self._tracker.epoch

    @property
    def run_permitted(self) -> bool:
        return (
            self._io_ready
            and self.inputs.get("safety_ok", False)
            and self._doors_closed()
            and not self._fault_code
        )

    @property
    def manual_operations_permitted(self) -> bool:
        return (
            self.state == ConveyorState.READY_STOPPED
            and not self.fifo
            and not self._active_captures
            and not self._purge.active
            and not self._fault_code
        )

    @property
    def configuration_operations_permitted(self) -> bool:
        return (
            self.state
            in {
                ConveyorState.SAFETY_LOCKED,
                ConveyorState.DOOR_OPEN_STOPPED,
                ConveyorState.READY_STOPPED,
                ConveyorState.FAULT_STOPPED,
            }
            and not self.fifo
            and not self._active_captures
            and not self._purge.active
            and not self.outputs.get("conveyor_run", False)
            and not self.outputs.get("waste_removal", False)
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
            self._fault_recovery = FaultRecovery.ACKNOWLEDGE
            self._fault_input = ""
            self._force_motion_off()
            self._select_nonrunning_interlock_state()
            self._publish()

    def set_io_ready(self, ready: bool, *, detail: str = "", now: float | None = None) -> None:
        with self._lock:
            current = self._now(now)
            if not ready:
                self._advance_motion_clock(current)
            else:
                self.tick(current)
            self._io_ready = bool(ready)
            if not ready:
                self._trip_fault(
                    "IO_NOT_READY",
                    detail or "IO controller is not ready",
                    recovery=FaultRecovery.RECONNECT_IO,
                )
            elif self._fault_code == "IO_NOT_READY":
                self._fault_code = ""
                self._fault_detail = ""
                self._fault_recovery = FaultRecovery.ACKNOWLEDGE
                self._fault_input = ""
                self._select_nonrunning_interlock_state()
            self._publish()

    def handle_input_change(self, name: str, state: bool, *, now: float | None = None) -> None:
        with self._lock:
            current = self._now(now)
            input_name = str(name)
            previous = self.inputs.get(input_name)
            business_state = bool(state)

            # Safety events must never run normal timers/actions using the old
            # interlock state. Only account for elapsed belt motion, then apply
            # the new safety state and force safe outputs first.
            if input_name == "safety_ok" or input_name in self.DOOR_INPUTS:
                self._advance_motion_clock(current)
                self.inputs[input_name] = business_state
                if previous is not business_state:
                    self._handle_interlock_change(current)
                self._apply_indicator_outputs()
                self._publish()
                return

            self.tick(current)
            self.inputs[input_name] = business_state
            if previous is business_state:
                return

            if input_name in self.MATERIAL_INPUTS:
                self._record_material_activity(current)
            if input_name in self.JAM_INPUTS:
                self._jam_monitor.observe_input(
                    input_name,
                    active=business_state,
                    motion_s=self._motion_elapsed_s,
                    conveyor_running=self.outputs.get("conveyor_run", False),
                )

            if business_state and input_name == "start_button":
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
            if self.state == ConveyorState.PURGE_PAUSED and self._purge.active:
                return self.continue_purge(now=now)
            if self.state not in {
                ConveyorState.READY_STOPPED,
                ConveyorState.READY_TO_RESUME,
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
            if self.state in {ConveyorState.PURGE_PREPARING, ConveyorState.PURGE_RUNNING}:
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
            allowed_states = {ConveyorState.READY_STOPPED, ConveyorState.READY_TO_RESUME}
            if self.state == ConveyorState.FAULT_STOPPED:
                if self._fault_recovery != FaultRecovery.PURGE_REQUIRED:
                    self._log("purge rejected: acknowledge or reconnect is required for this fault")
                    return False
            elif self.state not in allowed_states:
                self._log("purge rejected: line must be stopped first")
                return False

            self._fault_code = ""
            self._fault_detail = ""
            self._fault_recovery = FaultRecovery.ACKNOWLEDGE
            self._fault_input = ""
            self._tracker.invalidate_for_purge()
            self._reject.clear()
            self._active_captures.clear()
            self._controlled_stop_started_at = None
            self._purge.begin(
                now=current,
                waste_active=self.inputs.get("waste_outlet_sensor", False),
            )
            self._resume_purge_after_interlock = False
            self.state = ConveyorState.PURGE_PREPARING
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
            if self.state != ConveyorState.PURGE_PAUSED or not self._purge.active:
                return False
            if not self.run_permitted:
                return False
            self.state = ConveyorState.PURGE_PREPARING
            self._purge.restart_lead(now=current)
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
            if self.state != ConveyorState.FAULT_STOPPED:
                return False
            if self._fault_recovery != FaultRecovery.ACKNOWLEDGE:
                return False
            if not (
                self._io_ready
                and self.inputs.get("safety_ok", False)
                and self._doors_closed()
            ):
                return False
            if self._fault_input and self.inputs.get(self._fault_input, False):
                return False
            self._fault_code = ""
            self._fault_detail = ""
            self._fault_recovery = FaultRecovery.ACKNOWLEDGE
            self._fault_input = ""
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
            if int(epoch) != self.epoch:
                self._log(f"ignored stale inspection result: item={sequence_id}, epoch={epoch}")
                return False
            record = self._tracker.get(int(sequence_id))
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
            self._advance_motion_clock(current)

            self._update_reject_output()
            self._monitor_jams()
            self._monitor_fifo_timeout()
            if self.state == ConveyorState.CONTROLLED_STOPPING:
                self._evaluate_controlled_stop(current)
            elif self.state in {ConveyorState.PURGE_PREPARING, ConveyorState.PURGE_RUNNING}:
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
                "fault_recovery": self._fault_recovery.value,
                "fault_input": self._fault_input,
                "manual_operations_permitted": self.manual_operations_permitted,
                "configuration_operations_permitted": self.configuration_operations_permitted,
                "fifo_count": len(self.fifo),
                "capture_pending_count": len(self._active_captures),
                "fifo": [record.to_dict() for record in self.fifo],
                "epoch": self.epoch,
                "motion_elapsed_s": self._motion_elapsed_s,
                "outputs": dict(self.outputs),
                "inputs": dict(self.inputs),
                "purge_active": self.state in {
                    ConveyorState.PURGE_PREPARING,
                    ConveyorState.PURGE_RUNNING,
                },
                "purge_paused": self.state == ConveyorState.PURGE_PAUSED,
            }

    def shutdown(self) -> None:
        with self._lock:
            self._force_motion_off()
            self._set_output("buzzer", False)
            self.state = ConveyorState.READY_STOPPED
            self._publish()

    def _on_camera_sensor(self, now: float) -> None:
        if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}:
            return
        if len(self.fifo) >= max(1, int(self.config.max_inflight_items)):
            self._trip_fault("FIFO_OVERFLOW", "in-flight workpiece queue is full")
            return
        record = self._tracker.create(
            now=now,
            motion_s=self._motion_elapsed_s,
        )
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
        record = self._tracker.head()
        if record is None:
            self._trip_fault("FIFO_UNDERFLOW", "DI1 triggered while FIFO is empty")
            return
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

        if (
            record.inspection_status == InspectionStatus.GOOD
            and self.outputs.get("waste_removal", False)
        ):
            self._trip_fault(
                "BLOW_WINDOW_CONFLICT",
                f"GOOD item {record.sequence_id} reached DI1 while waste_removal was active",
                recovery=FaultRecovery.PURGE_REQUIRED,
            )
            return

        self._tracker.pop_head()
        if record.inspection_status == InspectionStatus.NG:
            self._reject.schedule(
                sequence_id=record.sequence_id,
                motion_s=self._motion_elapsed_s,
                delay_s=self.config.reject_blow_delay_ms / 1000.0,
                duration_s=self.config.reject_blow_duration_ms / 1000.0,
            )
            self._log(f"NG item={record.sequence_id} scheduled for blow-off")
        else:
            self._log(f"GOOD item={record.sequence_id} passed without blow-off")
        self._update_reject_output()

    def _update_reject_output(self) -> None:
        motion = self._motion_elapsed_s
        self._reject.prune(motion)
        should_blow = (
            self.state in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}
            and self.outputs.get("conveyor_run", False)
            and self._reject.is_active(motion)
        )
        if self.state not in {ConveyorState.PURGE_PREPARING, ConveyorState.PURGE_RUNNING}:
            self._set_output("waste_removal", should_blow)

    def _evaluate_controlled_stop(self, now: float) -> None:
        started = (
            self._controlled_stop_started_at
            if self._controlled_stop_started_at is not None
            else now
        )
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
        reject_committed = self._reject.has_pending
        if capture_committed or reject_committed:
            return
        self._force_motion_off()
        self.state = ConveyorState.READY_STOPPED
        self._controlled_stop_started_at = None
        self._log("controlled stop completed")

    def _handle_interlock_change(self, now: float) -> None:
        safety_ok = self.inputs.get("safety_ok", False)
        door_closed = self._doors_closed()
        if not safety_ok or not door_closed:
            was_purging = self.state in {
                ConveyorState.PURGE_PREPARING,
                ConveyorState.PURGE_RUNNING,
                ConveyorState.PURGE_PAUSED,
            }
            was_moving = self.state in {
                ConveyorState.RUNNING,
                ConveyorState.CONTROLLED_STOPPING,
                ConveyorState.PURGE_PREPARING,
                ConveyorState.PURGE_RUNNING,
            }
            self._resume_purge_after_interlock = was_purging and self._purge.active
            if (
                self.state in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}
                and self.outputs.get("waste_removal", False)
            ):
                self._fault_code = "BLOW_INTERRUPTED"
                self._fault_detail = "normal NG blow-off was interrupted by a safety interlock"
                self._fault_recovery = FaultRecovery.PURGE_REQUIRED
                self._fault_input = ""
                self._reject.clear()
            self._force_motion_off()
            if not safety_ok:
                self.state = ConveyorState.SAFETY_PAUSED if was_moving else ConveyorState.SAFETY_LOCKED
            else:
                self.state = ConveyorState.DOOR_PAUSED if was_moving else ConveyorState.DOOR_OPEN_STOPPED
            self._log("motion stopped immediately by safety interlock")
            return

        if self._resume_purge_after_interlock and self._purge.active:
            self.state = ConveyorState.PURGE_PAUSED
        elif self._fault_code:
            self.state = ConveyorState.FAULT_STOPPED
        elif self.state in {
            ConveyorState.SAFETY_PAUSED,
            ConveyorState.DOOR_PAUSED,
        }:
            self.state = ConveyorState.READY_TO_RESUME
        elif self.state in {
            ConveyorState.SAFETY_LOCKED,
            ConveyorState.DOOR_OPEN_STOPPED,
        }:
            self.state = ConveyorState.READY_STOPPED
        self._log("safety permission restored; manual restart is required")

    def _select_nonrunning_interlock_state(self) -> None:
        if not self._io_ready or not self.inputs.get("safety_ok", False):
            self.state = ConveyorState.SAFETY_LOCKED
        elif not self._doors_closed():
            self.state = ConveyorState.DOOR_OPEN_STOPPED
        elif self._fault_code:
            self.state = ConveyorState.FAULT_STOPPED
        else:
            self.state = ConveyorState.READY_STOPPED
        self._apply_indicator_outputs()

    def _doors_closed(self) -> bool:
        if not self.inputs.get("door_closed", False):
            return False
        if not self.config.upper_door_sensor_enabled:
            return True
        return self.inputs.get("door_upper_closed", False)

    def _record_material_activity(self, now: float) -> None:
        self._purge.record_activity(
            now=now,
            waste_active=self.inputs.get("waste_outlet_sensor", False),
        )

    def _update_purge(self, now: float) -> None:
        purge = self._purge.context
        if purge is None:
            self._trip_fault("PURGE_STATE_ERROR", "purge context is missing")
            return
        if now - purge.requested_at >= max(0.0, self.config.purge_max_run_s):
            self._trip_fault("PURGE_TIMEOUT", "one-click purge exceeded maximum run time")
            return
        lead_s = max(0.0, self.config.purge_air_lead_ms / 1000.0)
        if purge.conveyor_started_at is None and now - purge.requested_at >= lead_s:
            purge.conveyor_started_at = now
            self.state = ConveyorState.PURGE_RUNNING
            self._set_output("conveyor_run", True)
            self._log("purge air lead complete; conveyor started")
        if purge.conveyor_started_at is None:
            return

        self._purge.update_waste_state(
            now=now,
            waste_active=self.inputs.get("waste_outlet_sensor", False),
        )

        run_s = now - purge.conveyor_started_at
        last_activity = (
            purge.last_activity_at
            if purge.last_activity_at is not None
            else purge.conveyor_started_at
        )
        all_clear = not any(self.inputs.get(name, False) for name in self.MATERIAL_INPUTS)
        quiet_since = (
            purge.waste_clear_since
            if purge.waste_clear_since is not None
            else now
        )
        if (
            run_s >= max(0.0, self.config.purge_min_run_s)
            and all_clear
            and now - last_activity >= max(0.0, self.config.purge_tail_run_s)
            and now - quiet_since >= max(0.0, self.config.purge_quiet_s)
        ):
            # Normal purge completion is intentionally ordered: stop the belt
            # first, then stop the purge air.
            self._set_output("conveyor_run", False)
            self._set_output("waste_removal", False)
            self._purge.clear()
            self.state = ConveyorState.READY_STOPPED
            self._log("one-click purge completed")

    def _pause_purge(self, reason: str) -> None:
        self._force_motion_off()
        self.state = ConveyorState.PURGE_PAUSED
        self._resume_purge_after_interlock = True
        self._log(f"purge paused: {reason}; operator confirmation is required")

    def _monitor_jams(self) -> None:
        if not self.outputs.get("conveyor_run", False):
            self._jam_monitor.reset()
            return
        thresholds = {
            "end_test_sensor": self.config.end_test_jam_timeout_s,
            "good_outlet_sensor": self.config.good_jam_timeout_s,
            "waste_outlet_sensor": self.config.waste_jam_timeout_s,
        }
        timed_out = self._jam_monitor.first_timeout(
            inputs=self.inputs,
            motion_s=self._motion_elapsed_s,
            thresholds=thresholds,
            disabled_inputs=(
                () if self.config.end_test_sensor_enabled else ("end_test_sensor",)
            ),
        )
        if timed_out is not None:
            name, threshold = timed_out
            self._trip_fault(
                "JAM_DETECTED",
                f"{name} remained active for {threshold} s",
                recovery=FaultRecovery.ACKNOWLEDGE,
                source_input=name,
            )

    def _monitor_fifo_timeout(self) -> None:
        if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING} or not self.fifo:
            return
        head = self._tracker.head()
        if head is None:
            return
        timeout_s = max(0.1, self.config.front_to_reject_max_run_ms / 1000.0)
        if self._motion_elapsed_s - head.created_motion_s >= timeout_s:
            self._trip_fault(
                "ITEM_ARRIVAL_TIMEOUT",
                f"item {head.sequence_id} did not reach DI1 in time",
            )

    def _trip_fault(
        self,
        code: str,
        detail: str,
        *,
        recovery: FaultRecovery | None = None,
        source_input: str = "",
    ) -> None:
        self._fault_code = str(code)
        self._fault_detail = str(detail)
        self._fault_recovery = recovery or self._default_fault_recovery(code)
        self._fault_input = str(source_input or "")
        self._reject.clear()
        self._force_motion_off()
        self.state = ConveyorState.FAULT_STOPPED
        self._set_output("buzzer", True)
        self._apply_indicator_outputs()
        self._log(f"fault {self._fault_code}: {self._fault_detail}")

    def _force_motion_off(self) -> None:
        self._set_outputs(
            {"conveyor_run": False, "waste_removal": False},
            force=True,
        )

    def _apply_indicator_outputs(self) -> None:
        safety_ok = self.inputs.get("safety_ok", False)
        running = self.state in {
            ConveyorState.RUNNING,
            ConveyorState.PURGE_PREPARING,
            ConveyorState.PURGE_RUNNING,
        }
        self._set_output("button_green", running)
        self._set_output("button_red", not running)
        self._set_output("button_blue", not safety_ok)
        self._set_output("buzzer", self.state == ConveyorState.FAULT_STOPPED)

    def _set_output(self, name: str, on: bool) -> None:
        value = bool(on)
        previous = self.outputs.get(name)
        if previous == value:
            return
        self.outputs[name] = value
        if name == "conveyor_run":
            self._sync_jam_monitor_for_belt_transition(bool(previous), value)
        try:
            self._write_output_callback(name, value)
        except Exception as exc:
            self._handle_output_write_failure(name, exc)
            raise RuntimeError(self._fault_detail) from exc

    def _set_outputs(self, updates: Mapping[str, bool], *, force: bool = False) -> None:
        normalized = {str(name): bool(on) for name, on in updates.items()}
        previous_conveyor_run = bool(self.outputs.get("conveyor_run", False))
        changed = {
            name: value
            for name, value in normalized.items()
            if force or self.outputs.get(name) != value
        }
        if not changed:
            return
        self.outputs.update(changed)
        if "conveyor_run" in changed:
            self._sync_jam_monitor_for_belt_transition(
                previous_conveyor_run,
                changed["conveyor_run"],
            )
        try:
            if self._write_outputs_callback is not None:
                self._write_outputs_callback(changed)
            else:
                for name, value in changed.items():
                    self._write_output_callback(name, value)
        except Exception as exc:
            self._handle_output_write_failure(",".join(changed), exc)
            raise RuntimeError(self._fault_detail) from exc

    def _handle_output_write_failure(self, name: str, exc: Exception) -> None:
        self._fault_code = "OUTPUT_WRITE_FAILED"
        self._fault_detail = f"{name}: {exc}"
        self._fault_recovery = FaultRecovery.RECONNECT_IO
        self._fault_input = ""
        self.state = ConveyorState.FAULT_STOPPED
        for safe_name in ("conveyor_run", "waste_removal"):
            self.outputs[safe_name] = False
            try:
                self._write_output_callback(safe_name, False)
            except Exception:
                pass

    def _advance_motion_clock(self, now: float) -> None:
        delta = max(0.0, float(now) - self._last_tick_at)
        self._last_tick_at = float(now)
        if self.outputs.get("conveyor_run", False):
            self._motion_elapsed_s += delta

    def _sync_jam_monitor_for_belt_transition(
        self,
        was_running: bool,
        is_running: bool,
    ) -> None:
        if bool(was_running) == bool(is_running):
            return
        if not is_running:
            self._jam_monitor.reset()
            return
        for name in self.JAM_INPUTS:
            self._jam_monitor.observe_input(
                name,
                active=self.inputs.get(name, False),
                motion_s=self._motion_elapsed_s,
                conveyor_running=True,
            )

    @staticmethod
    def _default_fault_recovery(code: str) -> FaultRecovery:
        normalized = str(code or "").strip().upper()
        if normalized in {"IO_NOT_READY", "OUTPUT_WRITE_FAILED"}:
            return FaultRecovery.RECONNECT_IO
        if normalized in {
            "FIFO_OVERFLOW",
            "FIFO_UNDERFLOW",
            "RESULT_NOT_READY",
            "ITEM_ARRIVAL_TIMEOUT",
            "BLOW_INTERRUPTED",
            "BLOW_WINDOW_CONFLICT",
        }:
            return FaultRecovery.PURGE_REQUIRED
        return FaultRecovery.ACKNOWLEDGE

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
    "FaultRecovery",
    "InspectionStatus",
    "WorkpieceRecord",
]
