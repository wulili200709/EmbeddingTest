from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from .conveyor_components import (
    AutoPurgeController,
    InspectionStatus,
    JamMonitor,
    OutletExpectation,
    OutletConfirmationTracker,
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
    WAITING_INSPECTION = "WAITING_INSPECTION"
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
    inspection_result_wait_timeout_ms: int = 3000
    controlled_stop_timeout_ms: int = 1500
    reject_blow_delay_ms: int = 0
    reject_blow_duration_ms: int = 300
    reject_following_item_guard_ms: int = 100
    max_inflight_items: int = 20
    front_to_reject_max_run_ms: int = 5000
    front_sensor_max_active_ms: int = 0
    front_sensor_min_clear_ms: int = 0
    good_outlet_arrival_min_run_ms: int = 500
    good_outlet_arrival_max_run_ms: int = 3000
    waste_outlet_arrival_min_run_ms: int = 500
    waste_outlet_arrival_max_run_ms: int = 3000
    end_test_sensor_enabled: bool = True
    waste_outlet_confirmation_enabled: bool = False
    upper_door_sensor_enabled: bool = False
    end_test_blocked_timeout_s: float = 3.0
    good_outlet_blocked_timeout_s: float = 3.0
    waste_outlet_blocked_timeout_s: float = 3.0
    purge_air_lead_ms: int = 200
    purge_min_run_s: float = 10.0
    purge_tail_run_s: float = 5.0
    purge_quiet_s: float = 2.0
    purge_max_run_s: float = 30.0

    _INTEGER_FIELDS = (
        "poll_interval_ms",
        "debounce_ms",
        "capture_commit_guard_ms",
        "inspection_result_wait_timeout_ms",
        "controlled_stop_timeout_ms",
        "reject_blow_delay_ms",
        "reject_blow_duration_ms",
        "reject_following_item_guard_ms",
        "max_inflight_items",
        "front_to_reject_max_run_ms",
        "front_sensor_max_active_ms",
        "front_sensor_min_clear_ms",
        "good_outlet_arrival_min_run_ms",
        "good_outlet_arrival_max_run_ms",
        "waste_outlet_arrival_min_run_ms",
        "waste_outlet_arrival_max_run_ms",
        "purge_air_lead_ms",
    )
    _FLOAT_FIELDS = (
        "end_test_blocked_timeout_s",
        "good_outlet_blocked_timeout_s",
        "waste_outlet_blocked_timeout_s",
        "purge_min_run_s",
        "purge_tail_run_s",
        "purge_quiet_s",
        "purge_max_run_s",
    )
    _BOOLEAN_FIELDS = (
        "end_test_sensor_enabled",
        "waste_outlet_confirmation_enabled",
        "upper_door_sensor_enabled",
    )

    def __post_init__(self) -> None:
        for name in self._INTEGER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to 0")

        for name in self._FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite number greater than or equal to 0")

        for name in self._BOOLEAN_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

        if self.poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be greater than 0")
        if self.max_inflight_items <= 0:
            raise ValueError("max_inflight_items must be greater than 0")
        self._validate_time_window(
            "good_outlet_arrival",
            self.good_outlet_arrival_min_run_ms,
            self.good_outlet_arrival_max_run_ms,
        )
        self._validate_time_window(
            "waste_outlet_arrival",
            self.waste_outlet_arrival_min_run_ms,
            self.waste_outlet_arrival_max_run_ms,
        )
        if self.purge_min_run_s > self.purge_max_run_s:
            raise ValueError("purge_min_run_s must not exceed purge_max_run_s")
        if self.purge_tail_run_s > self.purge_max_run_s:
            raise ValueError("purge_tail_run_s must not exceed purge_max_run_s")
        if self.purge_quiet_s > self.purge_max_run_s:
            raise ValueError("purge_quiet_s must not exceed purge_max_run_s")

    @staticmethod
    def _validate_time_window(name: str, minimum: int, maximum: int) -> None:
        if minimum > maximum:
            raise ValueError(f"{name}_min_run_ms must not exceed {name}_max_run_ms")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "ConveyorConfig":
        values = dict(payload or {})
        aliases = {
            "reject_delay_ms": "reject_blow_delay_ms",
            "reject_duration_ms": "reject_blow_duration_ms",
            "fifo_max_items": "max_inflight_items",
            "end_sensor_enabled": "end_test_sensor_enabled",
            "end_test_jam_timeout_s": "end_test_blocked_timeout_s",
            "good_jam_timeout_s": "good_outlet_blocked_timeout_s",
            "waste_jam_timeout_s": "waste_outlet_blocked_timeout_s",
            "end_sensor_jam_s": "end_test_blocked_timeout_s",
            "good_outlet_jam_s": "good_outlet_blocked_timeout_s",
            "waste_outlet_jam_s": "waste_outlet_blocked_timeout_s",
        }
        known = cls.__dataclass_fields__
        allowed_keys = set(known) | set(aliases) | {"item_to_reject_timeout_s", "_comments"}
        unknown_keys = sorted(str(key) for key in values if key not in allowed_keys)
        if unknown_keys:
            raise ValueError(f"unknown conveyor configuration field(s): {', '.join(unknown_keys)}")
        for old_name, canonical_name in aliases.items():
            if canonical_name not in values and old_name in values:
                values[canonical_name] = values[old_name]
        if "front_to_reject_max_run_ms" not in values and "item_to_reject_timeout_s" in values:
            values["front_to_reject_max_run_ms"] = int(
                float(values["item_to_reject_timeout_s"]) * 1000.0
            )
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
        return self.end_test_blocked_timeout_s

    @property
    def good_outlet_jam_s(self) -> float:
        return self.good_outlet_blocked_timeout_s

    @property
    def waste_outlet_jam_s(self) -> float:
        return self.waste_outlet_blocked_timeout_s

    @property
    def end_test_jam_timeout_s(self) -> float:
        return self.end_test_blocked_timeout_s

    @property
    def good_jam_timeout_s(self) -> float:
        return self.good_outlet_blocked_timeout_s

    @property
    def waste_jam_timeout_s(self) -> float:
        return self.waste_outlet_blocked_timeout_s


@dataclass(frozen=True)
class ConveyorSnapshot:
    state: str
    run_permitted: bool
    io_ready: bool
    safety_ok: bool
    door_closed: bool
    door_lower_closed: bool
    door_upper_closed: bool
    fault_code: str
    fault_detail: str
    fault_recovery: str
    fault_input: str
    manual_operations_permitted: bool
    configuration_operations_permitted: bool
    fifo_count: int
    outlet_pending_count: int
    good_outlet_pending_count: int
    waste_outlet_pending_count: int
    inflight_count: int
    capture_pending_count: int
    fifo: list[dict[str, object]]
    outlet_pending: list[dict[str, object]]
    inflight: list[dict[str, object]]
    epoch: int
    motion_elapsed_s: float
    outputs: dict[str, bool]
    inputs: dict[str, bool]
    purge_active: bool
    purge_paused: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
    OUTLET_INPUTS = (
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
        self._outlet = OutletConfirmationTracker()
        self._active_captures: set[tuple[int, int]] = set()
        self._io_ready = False
        self._fault_code = ""
        self._fault_detail = ""
        self._fault_recovery = FaultRecovery.ACKNOWLEDGE
        self._fault_input = ""
        self._alarm_silenced = False
        self._controlled_stop_started_at: float | None = None
        self._di1_waiting_sequence_id: int | None = None
        self._di1_wait_started_at: float | None = None
        self._last_capture_edge_at: float | None = None
        self._front_sensor_active_since_motion_s: float | None = None
        self._front_sensor_clear_since_motion_s: float | None = None
        self._purge = AutoPurgeController()
        self._resume_purge_after_interlock = False
        self._motion_elapsed_s = 0.0
        self._last_tick_at = self._clock()
        self._outlet_active_since_at: dict[str, float | None] = {
            name: None for name in self.OUTLET_INPUTS
        }
        self._outlet_active_since_motion_s: dict[str, float | None] = {
            name: None for name in self.OUTLET_INPUTS
        }
        self._outlet_clear_since_at: dict[str, float | None] = {
            name: None for name in self.OUTLET_INPUTS
        }
        self._outlet_clear_since_motion_s: dict[str, float | None] = {
            name: None for name in self.OUTLET_INPUTS
        }
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
            and not self._outlet.pending
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
            and not self._outlet.pending
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
            self._alarm_silenced = False
            self._di1_waiting_sequence_id = None
            self._di1_wait_started_at = None
            self._initialize_outlet_signal_timing(current)
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

            # Outlet edges are both confirmation events and blockage signals.
            # Apply the observed edge before checking arrival deadlines so a
            # valid edge at the end of its window is not turned into a timeout.
            if input_name in self.OUTLET_INPUTS:
                self._advance_motion_clock(current)
                self.inputs[input_name] = business_state
                if previous is not business_state:
                    self._record_material_activity(current)
                    self._jam_monitor.observe_input(
                        input_name,
                        active=business_state,
                        motion_s=self._motion_elapsed_s,
                        conveyor_running=self.outputs.get("conveyor_run", False),
                    )
                    self._log_outlet_signal_transition(
                        input_name,
                        active=business_state,
                        now=current,
                    )
                    if business_state:
                        self._on_outlet_sensor(input_name)
                self.tick(current)
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
            elif input_name == "camera_trigger_sensor":
                if business_state:
                    self._on_camera_sensor(current)
                else:
                    self._front_sensor_active_since_motion_s = None
                    self._front_sensor_clear_since_motion_s = self._motion_elapsed_s
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
            waiting_record = self._waiting_di1_record()
            if waiting_record is not None:
                if waiting_record.inspection_status == InspectionStatus.PENDING:
                    self.state = ConveyorState.WAITING_INSPECTION
                    self._di1_wait_started_at = self._now(now)
                    self._force_motion_off()
                    self._apply_indicator_outputs()
                    self._log(
                        f"item={waiting_record.sequence_id} remains stopped at DI1 awaiting inspection"
                    )
                    self._publish()
                    return True
                return self._resume_waiting_di1(waiting_record)
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
            if self.state == ConveyorState.WAITING_INSPECTION:
                self.state = ConveyorState.READY_STOPPED
                self._force_motion_off()
                self._apply_indicator_outputs()
                self._log("operator stopped while DI1 was awaiting inspection")
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
            self._outlet.clear()
            self._front_sensor_active_since_motion_s = None
            self._front_sensor_clear_since_motion_s = None
            self._active_captures.clear()
            self._controlled_stop_started_at = None
            self._di1_waiting_sequence_id = None
            self._di1_wait_started_at = None
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
            # The operator alarm-reset button is always usable as a buzzer
            # silence control.  Fault recovery remains subject to the existing
            # safety and recovery-mode checks below.
            self._alarm_silenced = True
            can_clear_fault = (
                self.state == ConveyorState.FAULT_STOPPED
                and self._fault_recovery == FaultRecovery.ACKNOWLEDGE
                and self._io_ready
                and self.inputs.get("safety_ok", False)
                and self._doors_closed()
                and not (
                    self._fault_input
                    and self.inputs.get(self._fault_input, False)
                )
            )
            if not can_clear_fault:
                self._apply_indicator_outputs()
                self._log("alarm buzzer silenced")
                self._publish()
                return True
            self._fault_code = ""
            self._fault_detail = ""
            self._fault_recovery = FaultRecovery.ACKNOWLEDGE
            self._fault_input = ""
            self._select_nonrunning_interlock_state()
            self._apply_indicator_outputs()
            self._log("alarm acknowledged and buzzer silenced")
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
            if normalized in {"MULTIPLE_PRODUCTS", "MULTIPLE_PRODUCTS_IN_FOV"}:
                record.inspection_status = InspectionStatus.ERROR
                record.inspection_detail = str(
                    detail or "multiple products were detected in one inspection frame"
                )
                record.result_at = current
                self._notify_inspection_result(record)
                self._trip_fault(
                    "MULTIPLE_PRODUCTS_IN_FOV",
                    f"item {record.sequence_id}: {record.inspection_detail}",
                    recovery=FaultRecovery.PURGE_REQUIRED,
                )
                self._publish()
                return True
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
            if (
                self._di1_waiting_sequence_id == record.sequence_id
                and self.state == ConveyorState.WAITING_INSPECTION
            ):
                self._resume_waiting_di1(record)
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
            self._monitor_inspection_wait(current)
            self._monitor_outlet_timeouts()
            self._monitor_front_sensor_spacing()
            self._monitor_fifo_timeout()
            if self.state == ConveyorState.CONTROLLED_STOPPING:
                self._evaluate_controlled_stop(current)
            elif self.state in {ConveyorState.PURGE_PREPARING, ConveyorState.PURGE_RUNNING}:
                self._update_purge(current)
            self._apply_indicator_outputs()
            self._publish()

    def snapshot(self) -> dict[str, object]:
        return self.snapshot_model().to_dict()

    def snapshot_model(self) -> ConveyorSnapshot:
        with self._lock:
            fifo = [record.to_dict() for record in self.fifo]
            outlet_pending = [item.to_dict() for item in self._outlet.pending]
            return ConveyorSnapshot(
                state=self.state.value,
                run_permitted=self.run_permitted,
                io_ready=self._io_ready,
                safety_ok=self.inputs.get("safety_ok", False),
                door_closed=self._doors_closed(),
                door_lower_closed=self.inputs.get("door_closed", False),
                door_upper_closed=self.inputs.get("door_upper_closed", False),
                fault_code=self._fault_code,
                fault_detail=self._fault_detail,
                fault_recovery=self._fault_recovery.value,
                fault_input=self._fault_input,
                manual_operations_permitted=self.manual_operations_permitted,
                configuration_operations_permitted=self.configuration_operations_permitted,
                fifo_count=len(fifo),
                outlet_pending_count=len(outlet_pending),
                good_outlet_pending_count=self._outlet.count_for_input("good_outlet_sensor"),
                waste_outlet_pending_count=self._outlet.count_for_input("waste_outlet_sensor"),
                inflight_count=len(fifo) + len(outlet_pending),
                capture_pending_count=len(self._active_captures),
                fifo=fifo,
                outlet_pending=outlet_pending,
                inflight=fifo + outlet_pending,
                epoch=self.epoch,
                motion_elapsed_s=self._motion_elapsed_s,
                outputs=dict(self.outputs),
                inputs=dict(self.inputs),
                purge_active=self.state in {
                    ConveyorState.PURGE_PREPARING,
                    ConveyorState.PURGE_RUNNING,
                },
                purge_paused=self.state == ConveyorState.PURGE_PAUSED,
            )

    def shutdown(self) -> None:
        with self._lock:
            self._force_motion_off()
            self._set_output("buzzer", False)
            self._di1_waiting_sequence_id = None
            self._di1_wait_started_at = None
            self.state = ConveyorState.READY_STOPPED
            self._publish()

    def _on_camera_sensor(self, now: float) -> None:
        if self.state not in {ConveyorState.RUNNING, ConveyorState.CONTROLLED_STOPPING}:
            return
        min_clear_s = max(0.0, self.config.front_sensor_min_clear_ms / 1000.0)
        if (
            min_clear_s > 0.0
            and self._front_sensor_clear_since_motion_s is not None
            and self._motion_elapsed_s - self._front_sensor_clear_since_motion_s
            < min_clear_s
        ):
            self._trip_fault(
                "PRODUCT_SPACING_TOO_SMALL",
                "DI0 clear interval was shorter than the configured minimum",
                recovery=FaultRecovery.PURGE_REQUIRED,
            )
            return
        self._front_sensor_active_since_motion_s = self._motion_elapsed_s
        if len(self.fifo) + len(self._outlet.pending) >= max(
            1,
            int(self.config.max_inflight_items),
        ):
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
            if self.outputs.get("waste_removal", False):
                self._trip_fault(
                    "BLOW_WINDOW_CONFLICT",
                    f"item {record.sequence_id} reached DI1 while waste_removal was active",
                    recovery=FaultRecovery.PURGE_REQUIRED,
                )
                return
            self._di1_waiting_sequence_id = record.sequence_id
            self._di1_wait_started_at = now
            self.state = ConveyorState.WAITING_INSPECTION
            self._force_motion_off()
            self._log(
                f"item={record.sequence_id} stopped at DI1 awaiting inspection result"
            )
            return
        self._route_di1_record(record)

    def _route_di1_record(self, record: WorkpieceRecord) -> None:
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

        if record.inspection_status == InspectionStatus.NG and len(self.fifo) > 1:
            following = self.fifo[1]
            spacing_s = max(
                0.0,
                float(following.created_motion_s) - float(record.created_motion_s),
            )
            required_s = max(
                0.0,
                (
                    self.config.reject_blow_delay_ms
                    + self.config.reject_blow_duration_ms
                    + self.config.reject_following_item_guard_ms
                )
                / 1000.0,
            )
            if spacing_s < required_s:
                self._trip_fault(
                    "PRODUCT_SPACING_TOO_SMALL",
                    (
                        f"NG item {record.sequence_id} blow-off blocked: following item "
                        f"{following.sequence_id} is only {spacing_s * 1000.0:.1f} ms behind; "
                        f"required spacing is {required_s * 1000.0:.1f} ms"
                    ),
                    recovery=FaultRecovery.PURGE_REQUIRED,
                    source_input="camera_trigger_sensor",
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
            expectation = self._outlet.expect(
                record,
                expected_input="waste_outlet_sensor",
                motion_s=self._motion_elapsed_s,
                min_run_s=self.config.waste_outlet_arrival_min_run_ms / 1000.0,
                max_run_s=self.config.waste_outlet_arrival_max_run_ms / 1000.0,
            )
            self._log(
                (
                    f"NG item={record.sequence_id} scheduled for blow-off and DI8 confirmation"
                    if self.config.waste_outlet_confirmation_enabled
                    else f"NG item={record.sequence_id} scheduled for blow-off and DI7 reject guard"
                )
                + self._format_outlet_window(expectation)
            )
        else:
            expectation = self._outlet.expect(
                record,
                expected_input="good_outlet_sensor",
                motion_s=self._motion_elapsed_s,
                min_run_s=self.config.good_outlet_arrival_min_run_ms / 1000.0,
                max_run_s=self.config.good_outlet_arrival_max_run_ms / 1000.0,
            )
            self._log(
                f"GOOD item={record.sequence_id} passed DI1 and awaits DI7 confirmation"
                + self._format_outlet_window(expectation)
            )
        self._update_reject_output()

    def _waiting_di1_record(self) -> WorkpieceRecord | None:
        sequence_id = self._di1_waiting_sequence_id
        if sequence_id is None:
            return None
        return self._tracker.get(sequence_id)

    def _resume_waiting_di1(self, record: WorkpieceRecord) -> bool:
        if record.inspection_status == InspectionStatus.PENDING:
            return False
        if not self.run_permitted:
            return False
        wait_started_at = self._di1_wait_started_at
        wait_ms = (
            max(0.0, self._clock() - wait_started_at) * 1000.0
            if wait_started_at is not None
            else 0.0
        )
        self._di1_waiting_sequence_id = None
        self._di1_wait_started_at = None
        self.state = ConveyorState.RUNNING
        self._route_di1_record(record)
        if self.state == ConveyorState.FAULT_STOPPED:
            return False
        should_blow = self._reject.is_active(self._motion_elapsed_s)
        self._set_outputs(
            {
                "waste_removal": should_blow,
                "conveyor_run": True,
            }
        )
        self._apply_indicator_outputs()
        self._log(
            f"item={record.sequence_id} inspection ready at DI1; "
            f"inspection_wait_ms={wait_ms:.1f}; production resumed"
        )
        return True

    def _on_outlet_sensor(self, input_name: str) -> None:
        if self.state not in {
            ConveyorState.RUNNING,
            ConveyorState.CONTROLLED_STOPPING,
        }:
            return
        if (
            input_name == "waste_outlet_sensor"
            and not self.config.waste_outlet_confirmation_enabled
        ):
            self._log("DI8 edge observed; per-item NG confirmation is disabled")
            return
        motion_s = self._motion_elapsed_s
        outlet = "DI7" if input_name == "good_outlet_sensor" else "DI8"
        self._log(
            f"{outlet} edge: motion_ms={motion_s * 1000.0:.1f}; "
            f"active_candidates={self._format_outlet_candidates(motion_s)}"
        )
        outcome, expectation = self._outlet.confirm(
            input_name,
            motion_s=motion_s,
        )
        if outcome == "CONFIRMED" and expectation is not None:
            travel_ms = max(0.0, motion_s - expectation.started_motion_s) * 1000.0
            self._log(
                f"outlet confirmed: item={expectation.sequence_id}, outlet={outlet}, "
                f"travel_ms={travel_ms:.1f}"
                + self._format_outlet_window(expectation)
            )
            return
        if outcome == "WRONG_OUTLET" and expectation is not None:
            travel_ms = max(0.0, motion_s - expectation.started_motion_s) * 1000.0
            if input_name == "good_outlet_sensor":
                code = "REJECT_FAILED_WRONG_OUTLET"
                detail = (
                    f"NG item {expectation.sequence_id} candidate reached DI7 after blow-off; "
                    f"DI1_to_DI7_motion_ms={travel_ms:.1f}"
                )
            else:
                code = "GOOD_WRONG_OUTLET"
                detail = (
                    f"GOOD item {expectation.sequence_id} candidate reached DI8 instead of DI7; "
                    f"DI1_to_DI8_motion_ms={travel_ms:.1f}"
                )
            self._trip_fault(code, detail, recovery=FaultRecovery.PURGE_REQUIRED)
            return
        code = (
            "UNEXPECTED_GOOD_OUTLET"
            if input_name == "good_outlet_sensor"
            else "UNEXPECTED_WASTE_OUTLET"
        )
        outlet = "DI7" if input_name == "good_outlet_sensor" else "DI8"
        self._trip_fault(
            code,
            f"{outlet} triggered without an eligible outlet expectation",
            recovery=FaultRecovery.PURGE_REQUIRED,
        )

    def _format_outlet_window(self, expectation: OutletExpectation) -> str:
        minimum_ms = max(
            0.0,
            expectation.earliest_motion_s - expectation.started_motion_s,
        ) * 1000.0
        maximum_ms = max(
            0.0,
            expectation.deadline_motion_s - expectation.started_motion_s,
        ) * 1000.0
        return (
            f"; di1_motion_ms={expectation.started_motion_s * 1000.0:.1f}; "
            f"window_ms={minimum_ms:.1f}..{maximum_ms:.1f}"
        )

    def _format_outlet_candidates(self, motion_s: float) -> str:
        active = [
            expectation
            for expectation in self._outlet.pending
            if expectation.started_motion_s
            <= motion_s
            <= expectation.deadline_motion_s
        ]
        if not active:
            return "none"
        descriptions: list[str] = []
        for expectation in active:
            elapsed_ms = max(
                0.0,
                motion_s - expectation.started_motion_s,
            ) * 1000.0
            expected = (
                "DI7"
                if expectation.expected_input == "good_outlet_sensor"
                else (
                    "DI8"
                    if self.config.waste_outlet_confirmation_enabled
                    else "DI7_REJECT_GUARD"
                )
            )
            descriptions.append(
                f"item={expectation.sequence_id}/result={expectation.inspection_status.value}"
                f"/expected={expected}/elapsed_ms={elapsed_ms:.1f}"
            )
        return "[" + ", ".join(descriptions) + "]"

    def _initialize_outlet_signal_timing(self, now: float) -> None:
        current = float(now)
        motion_s = float(self._motion_elapsed_s)
        for name in self.OUTLET_INPUTS:
            if self.inputs.get(name, False):
                self._outlet_active_since_at[name] = current
                self._outlet_active_since_motion_s[name] = motion_s
                self._outlet_clear_since_at[name] = None
                self._outlet_clear_since_motion_s[name] = None
            else:
                self._outlet_active_since_at[name] = None
                self._outlet_active_since_motion_s[name] = None
                self._outlet_clear_since_at[name] = current
                self._outlet_clear_since_motion_s[name] = motion_s

    @staticmethod
    def _format_signal_interval_ms(
        current: float,
        started_at: float | None,
    ) -> str:
        if started_at is None:
            return "unknown"
        return f"{max(0.0, float(current) - float(started_at)) * 1000.0:.1f}"

    def _log_outlet_signal_transition(
        self,
        input_name: str,
        *,
        active: bool,
        now: float,
    ) -> None:
        outlet = "DI7" if input_name == "good_outlet_sensor" else "DI8"
        current = float(now)
        motion_s = float(self._motion_elapsed_s)
        if active:
            low_wall_ms = self._format_signal_interval_ms(
                current,
                self._outlet_clear_since_at.get(input_name),
            )
            low_motion_ms = self._format_signal_interval_ms(
                motion_s,
                self._outlet_clear_since_motion_s.get(input_name),
            )
            self._outlet_active_since_at[input_name] = current
            self._outlet_active_since_motion_s[input_name] = motion_s
            self._outlet_clear_since_at[input_name] = None
            self._outlet_clear_since_motion_s[input_name] = None
            self._log(
                f"{outlet} signal ON: motion_ms={motion_s * 1000.0:.1f}; "
                f"low_wall_ms={low_wall_ms}; low_motion_ms={low_motion_ms}; "
                f"debounce_ms={self.config.debounce_ms}"
            )
            return

        high_wall_ms = self._format_signal_interval_ms(
            current,
            self._outlet_active_since_at.get(input_name),
        )
        high_motion_ms = self._format_signal_interval_ms(
            motion_s,
            self._outlet_active_since_motion_s.get(input_name),
        )
        self._outlet_active_since_at[input_name] = None
        self._outlet_active_since_motion_s[input_name] = None
        self._outlet_clear_since_at[input_name] = current
        self._outlet_clear_since_motion_s[input_name] = motion_s
        self._log(
            f"{outlet} signal OFF: motion_ms={motion_s * 1000.0:.1f}; "
            f"high_wall_ms={high_wall_ms}; high_motion_ms={high_motion_ms}; "
            f"debounce_ms={self.config.debounce_ms}"
        )

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
                self._alarm_silenced = False
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
            "end_test_sensor": self.config.end_test_blocked_timeout_s,
            "good_outlet_sensor": self.config.good_outlet_blocked_timeout_s,
            "waste_outlet_sensor": self.config.waste_outlet_blocked_timeout_s,
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

    def _monitor_inspection_wait(self, now: float) -> None:
        if self.state != ConveyorState.WAITING_INSPECTION:
            return
        started_at = self._di1_wait_started_at
        record = self._waiting_di1_record()
        if started_at is None or record is None:
            self._trip_fault(
                "RESULT_NOT_READY",
                "DI1 inspection wait state lost its tracked workpiece",
                recovery=FaultRecovery.PURGE_REQUIRED,
            )
            return
        timeout_s = max(
            0.001,
            self.config.inspection_result_wait_timeout_ms / 1000.0,
        )
        if now - started_at >= timeout_s:
            self._trip_fault(
                "RESULT_NOT_READY",
                (
                    f"item {record.sequence_id} inspection did not complete within "
                    f"{self.config.inspection_result_wait_timeout_ms} ms after reaching DI1"
                ),
                recovery=FaultRecovery.PURGE_REQUIRED,
            )

    def _monitor_outlet_timeouts(self) -> None:
        if self.state not in {
            ConveyorState.RUNNING,
            ConveyorState.CONTROLLED_STOPPING,
        }:
            return
        while True:
            expired = self._outlet.first_expired(self._motion_elapsed_s)
            if expired is None:
                return
            if expired.expected_input == "good_outlet_sensor":
                self._trip_fault(
                    "GOOD_OUTLET_TIMEOUT",
                    f"GOOD item {expired.sequence_id} did not reach DI7 in time",
                    recovery=FaultRecovery.PURGE_REQUIRED,
                )
                return
            if self.config.waste_outlet_confirmation_enabled:
                self._trip_fault(
                    "WASTE_OUTLET_TIMEOUT",
                    f"NG item {expired.sequence_id} did not reach DI8 in time",
                    recovery=FaultRecovery.PURGE_REQUIRED,
                )
                return
            self._outlet.remove(expired)
            elapsed_ms = max(
                0.0,
                self._motion_elapsed_s - expired.started_motion_s,
            ) * 1000.0
            self._log(
                f"NG item={expired.sequence_id} reject guard passed without DI7 signal; "
                f"elapsed_ms={elapsed_ms:.1f}"
                + self._format_outlet_window(expired)
            )

    def _monitor_front_sensor_spacing(self) -> None:
        if self.state not in {
            ConveyorState.RUNNING,
            ConveyorState.CONTROLLED_STOPPING,
        }:
            return
        maximum_s = max(0.0, self.config.front_sensor_max_active_ms / 1000.0)
        since = self._front_sensor_active_since_motion_s
        if (
            maximum_s <= 0.0
            or since is None
            or not self.inputs.get("camera_trigger_sensor", False)
        ):
            return
        if self._motion_elapsed_s - since >= maximum_s:
            self._trip_fault(
                "PRODUCT_SPACING_TOO_SMALL",
                "DI0 remained active beyond the configured product-spacing limit",
                recovery=FaultRecovery.PURGE_REQUIRED,
                source_input="camera_trigger_sensor",
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
        self._alarm_silenced = False
        self._di1_waiting_sequence_id = None
        self._di1_wait_started_at = None
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
        self._set_output(
            "buzzer",
            self.state == ConveyorState.FAULT_STOPPED and not self._alarm_silenced,
        )

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
        self._alarm_silenced = False
        self.state = ConveyorState.FAULT_STOPPED
        safe_off_failures: list[str] = []
        for safe_name in ("conveyor_run", "waste_removal"):
            self.outputs[safe_name] = False
            try:
                self._write_output_callback(safe_name, False)
            except Exception as safe_exc:
                safe_off_failures.append(f"{safe_name}: {safe_exc}")
        if safe_off_failures:
            failure_detail = "; ".join(safe_off_failures)
            self._fault_detail = (
                f"{self._fault_detail}; safe-off failed, physical output state unknown: "
                f"{failure_detail}"
            )
        self._log(f"fault {self._fault_code}: {self._fault_detail}")

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
            "GOOD_OUTLET_TIMEOUT",
            "WASTE_OUTLET_TIMEOUT",
            "REJECT_FAILED_WRONG_OUTLET",
            "GOOD_WRONG_OUTLET",
            "UNEXPECTED_GOOD_OUTLET",
            "UNEXPECTED_WASTE_OUTLET",
            "MULTIPLE_PRODUCTS_IN_FOV",
            "PRODUCT_SPACING_TOO_SMALL",
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
    "ConveyorSnapshot",
    "ConveyorState",
    "FaultRecovery",
    "InspectionStatus",
    "WorkpieceRecord",
]
