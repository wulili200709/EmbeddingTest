from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from domain.conveyor_line import (
    ConveyorConfig,
    ConveyorLineController,
    ConveyorSnapshot,
    ConveyorState,
    InspectionStatus,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += float(seconds)
        return self.value


class ConveyorLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.outputs: dict[str, bool] = {}
        self.output_events: list[tuple[str, bool]] = []
        self.output_batches: list[dict[str, bool]] = []
        self.requests: list[tuple[int, int]] = []
        self.logs: list[str] = []
        self.config = ConveyorConfig(
            capture_commit_guard_ms=100,
            inspection_result_wait_timeout_ms=3000,
            controlled_stop_timeout_ms=1500,
            reject_blow_delay_ms=0,
            reject_blow_duration_ms=300,
            max_inflight_items=8,
            front_to_reject_max_run_ms=5000,
            good_outlet_arrival_min_run_ms=0,
            good_outlet_arrival_max_run_ms=500,
            waste_outlet_arrival_min_run_ms=0,
            waste_outlet_arrival_max_run_ms=500,
            end_test_sensor_enabled=True,
            waste_outlet_confirmation_enabled=True,
            upper_door_sensor_enabled=True,
            end_test_blocked_timeout_s=1.0,
            good_outlet_blocked_timeout_s=1.0,
            waste_outlet_blocked_timeout_s=1.0,
            purge_air_lead_ms=100,
            purge_min_run_s=0.5,
            purge_tail_run_s=0.5,
            purge_quiet_s=0.5,
            purge_max_run_s=5.0,
        )
        def write_output(name: str, on: bool) -> None:
            self.output_events.append((name, bool(on)))
            self.outputs[name] = bool(on)

        def write_outputs(updates: dict[str, bool]) -> None:
            normalized = {str(name): bool(on) for name, on in updates.items()}
            self.output_batches.append(normalized)
            self.outputs.update(normalized)

        self.line = ConveyorLineController(
            config=self.config,
            output_writer=write_output,
            output_batch_writer=write_outputs,
            inspection_requester=lambda sequence_id, epoch: self.requests.append((sequence_id, epoch)),
            log_writer=self.logs.append,
            clock=self.clock,
        )
        self.line.initialize_inputs(
            {
                "camera_trigger_sensor": False,
                "reject_position_sensor": False,
                "start_button": False,
                "stop_button": False,
                "safety_ok": True,
                "end_test_sensor": False,
                "good_outlet_sensor": False,
                "waste_outlet_sensor": False,
                "door_closed": True,
                "door_upper_closed": True,
            },
            io_ready=True,
        )

    def edge(self, name: str) -> None:
        self.line.handle_input_change(name, True)
        self.line.handle_input_change(name, False)

    def test_start_and_safety_restore_never_auto_restarts(self) -> None:
        self.assertTrue(self.line.request_start())
        self.assertEqual(self.line.state, ConveyorState.RUNNING)
        self.assertTrue(self.outputs["conveyor_run"])

        self.line.handle_input_change("safety_ok", False)
        self.assertEqual(self.line.state, ConveyorState.SAFETY_PAUSED)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])
        self.assertTrue(self.outputs["button_blue"])

        self.line.handle_input_change("safety_ok", True)
        self.assertEqual(self.line.state, ConveyorState.READY_TO_RESUME)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertTrue(self.line.request_start())

    def test_typed_snapshot_matches_compatible_dictionary_snapshot(self) -> None:
        model = self.line.snapshot_model()
        self.assertIsInstance(model, ConveyorSnapshot)
        self.assertEqual(model.to_dict(), self.line.snapshot())

    def test_interlock_event_never_advances_purge_before_stopping_outputs(self) -> None:
        self.assertTrue(self.line.request_purge())
        self.output_events.clear()
        self.output_batches.clear()

        self.line.handle_input_change("door_closed", False, now=self.clock.advance(0.11))

        self.assertNotIn(("conveyor_run", True), self.output_events)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])
        self.assertIn(
            {"conveyor_run": False, "waste_removal": False},
            self.output_batches,
        )

    def test_door_is_independent_from_safety_ok(self) -> None:
        self.line.request_start()
        self.line.handle_input_change("door_closed", False)
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "DOOR_PAUSED")
        self.assertTrue(snapshot["safety_ok"])
        self.assertFalse(snapshot["door_closed"])
        self.assertFalse(self.outputs["button_blue"])
        self.assertFalse(self.outputs["conveyor_run"])

    def test_upper_door_is_also_required_for_motion(self) -> None:
        self.assertTrue(self.line.request_start())
        self.line.handle_input_change("door_upper_closed", False)
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "DOOR_PAUSED")
        self.assertFalse(snapshot["door_closed"])
        self.assertTrue(snapshot["door_lower_closed"])
        self.assertFalse(snapshot["door_upper_closed"])
        self.assertFalse(self.outputs["conveyor_run"])

        self.line.handle_input_change("door_upper_closed", True)
        self.assertEqual(self.line.state, ConveyorState.READY_TO_RESUME)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertTrue(self.line.request_start())

    def test_unwired_upper_door_can_be_disabled_for_debugging(self) -> None:
        self.line.config = ConveyorConfig(upper_door_sensor_enabled=False)
        self.line.handle_input_change("door_upper_closed", False)
        snapshot = self.line.snapshot()
        self.assertTrue(snapshot["door_closed"])
        self.assertFalse(snapshot["door_upper_closed"])
        self.assertTrue(self.line.request_start())
        self.assertTrue(self.outputs["conveyor_run"])

    def test_fifo_order_is_not_changed_by_out_of_order_results(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.edge("camera_trigger_sensor")
        self.assertEqual([item.sequence_id for item in self.line.fifo], [1, 2])

        self.line.inspection_completed(2, self.line.epoch, "NG")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")
        self.assertEqual([item.sequence_id for item in self.line.fifo], [2])
        self.assertFalse(self.outputs["waste_removal"])

        self.edge("reject_position_sensor")
        self.assertEqual(len(self.line.fifo), 0)
        self.assertTrue(self.outputs["waste_removal"])

    def test_good_arrival_during_ng_blow_window_stops_with_conflict(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.tick(self.clock.advance(0.5))
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.line.inspection_completed(2, self.line.epoch, "OK")

        self.edge("reject_position_sensor")
        self.assertTrue(self.outputs["waste_removal"])
        self.edge("reject_position_sensor")

        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT_STOPPED")
        self.assertEqual(snapshot["fault_code"], "BLOW_WINDOW_CONFLICT")
        self.assertEqual(snapshot["fault_recovery"], "PURGE_REQUIRED")
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])
        self.assertEqual(len(self.line.fifo), 1)

    def test_close_following_item_blocks_ng_blow_before_fifo_can_desynchronize(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.tick(self.clock.advance(0.2))
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.line.inspection_completed(2, self.line.epoch, "OK")

        self.edge("reject_position_sensor")

        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT_STOPPED")
        self.assertEqual(snapshot["fault_code"], "PRODUCT_SPACING_TOO_SMALL")
        self.assertEqual(snapshot["fault_recovery"], "PURGE_REQUIRED")
        self.assertIn("following item 2", snapshot["fault_detail"])
        self.assertEqual(snapshot["fifo_count"], 2)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

    def test_safely_spaced_following_item_allows_ng_blow(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.tick(self.clock.advance(0.5))
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.line.inspection_completed(2, self.line.epoch, "OK")

        self.edge("reject_position_sensor")

        self.assertEqual(self.line.state, ConveyorState.RUNNING)
        self.assertEqual([item.sequence_id for item in self.line.fifo], [2])
        self.assertTrue(self.outputs["waste_removal"])

    def test_pending_result_at_di1_stops_then_good_result_resumes(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.edge("reject_position_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "WAITING_INSPECTION")
        self.assertEqual(snapshot["fault_code"], "")
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

        self.assertTrue(
            self.line.inspection_completed(1, self.line.epoch, "OK")
        )
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "RUNNING")
        self.assertTrue(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])
        self.assertEqual(snapshot["fifo_count"], 0)
        self.assertEqual(snapshot["good_outlet_pending_count"], 1)

    def test_pending_result_at_di1_stops_then_ng_result_resumes_with_blow(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.edge("reject_position_sensor")

        self.assertTrue(
            self.line.inspection_completed(1, self.line.epoch, "NG")
        )
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "RUNNING")
        self.assertTrue(self.outputs["conveyor_run"])
        self.assertTrue(self.outputs["waste_removal"])
        self.assertEqual(snapshot["fifo_count"], 0)
        self.assertEqual(snapshot["waste_outlet_pending_count"], 1)

    def test_pending_result_at_di1_faults_only_after_wait_timeout(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.edge("reject_position_sensor")

        self.line.tick(self.clock.advance(2.9))
        self.assertEqual(self.line.state, ConveyorState.WAITING_INSPECTION)
        self.line.tick(self.clock.advance(0.2))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT_STOPPED")
        self.assertEqual(snapshot["fault_code"], "RESULT_NOT_READY")
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

    def test_inspection_error_is_rejected_as_ng_instead_of_faulting(self) -> None:
        self.assertTrue(self.line.request_start())
        self.edge("camera_trigger_sensor")
        self.assertTrue(
            self.line.inspection_completed(
                1,
                self.line.epoch,
                "ERROR",
                detail="match failure",
            )
        )
        self.assertEqual(self.line.fifo[0].inspection_status, InspectionStatus.NG)
        self.edge("reject_position_sensor")
        self.assertEqual(self.line.state, ConveyorState.RUNNING)
        self.assertEqual(len(self.line.fifo), 0)
        self.assertTrue(self.outputs["waste_removal"])

    def test_controlled_stop_waits_only_for_committed_blow_window(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.capture_completed(1, self.line.epoch)
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")
        self.assertTrue(self.outputs["waste_removal"])

        self.line.request_controlled_stop()
        self.assertEqual(self.line.state, ConveyorState.CONTROLLED_STOPPING)
        self.line.tick(self.clock.advance(0.2))
        self.assertEqual(self.line.state, ConveyorState.CONTROLLED_STOPPING)
        self.line.tick(self.clock.advance(0.2))
        self.assertEqual(self.line.state, ConveyorState.READY_STOPPED)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

    def test_controlled_stop_waits_for_capture_but_not_inference(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.request_controlled_stop()
        self.line.tick(self.clock.advance(0.2))
        self.assertEqual(self.line.state, ConveyorState.CONTROLLED_STOPPING)
        self.line.capture_completed(1, self.line.epoch)
        self.line.tick(self.clock.advance(0.01))
        self.assertEqual(self.line.state, ConveyorState.READY_STOPPED)
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.assertEqual(self.line.fifo[0].inspection_status.value, "GOOD")

    def test_controlled_stop_timeout_forces_safe_outputs_and_alarm(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.request_controlled_stop()
        self.line.tick(self.clock.advance(1.6))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT_STOPPED")
        self.assertEqual(snapshot["fault_code"], "CONTROLLED_STOP_TIMEOUT")
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

    def test_safety_pause_freezes_belt_distance_timers_and_fifo(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.line.tick(self.clock.advance(0.1))
        motion_before = self.line.snapshot()["motion_elapsed_s"]

        self.line.handle_input_change("safety_ok", False)
        self.line.tick(self.clock.advance(2.0))
        self.assertEqual(self.line.snapshot()["motion_elapsed_s"], motion_before)
        self.assertEqual(len(self.line.fifo), 1)
        self.line.handle_input_change("safety_ok", True)
        self.line.request_start()
        self.line.tick(self.clock.advance(0.1))
        self.assertFalse(self.outputs["waste_removal"])
        self.assertAlmostEqual(
            self.line.snapshot()["motion_elapsed_s"],
            motion_before + 0.1,
        )

    def test_safety_pause_preserves_and_freezes_outlet_expectation(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")
        self.line.tick(self.clock.advance(0.2))
        motion_before = self.line.snapshot()["motion_elapsed_s"]

        self.line.handle_input_change("safety_ok", False)
        self.line.tick(self.clock.advance(2.0))
        paused = self.line.snapshot()
        self.assertEqual(paused["good_outlet_pending_count"], 1)
        self.assertEqual(paused["motion_elapsed_s"], motion_before)
        self.assertEqual(paused["fault_code"], "")

        self.line.handle_input_change("safety_ok", True)
        self.assertTrue(self.line.request_start())
        self.line.tick(self.clock.advance(0.2))
        self.edge("good_outlet_sensor")
        resumed = self.line.snapshot()
        self.assertEqual(resumed["good_outlet_pending_count"], 0)
        self.assertEqual(resumed["fault_code"], "")

    def test_interrupted_blow_requires_purge_before_production_can_resume(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")
        self.assertTrue(self.outputs["waste_removal"])

        self.line.handle_input_change("safety_ok", False)
        self.assertEqual(self.line.state, ConveyorState.SAFETY_PAUSED)
        self.line.handle_input_change("safety_ok", True)

        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT_STOPPED")
        self.assertEqual(snapshot["fault_code"], "BLOW_INTERRUPTED")
        self.assertEqual(snapshot["fault_recovery"], "PURGE_REQUIRED")
        self.assertFalse(self.line.request_start())
        self.assertTrue(self.outputs["buzzer"])
        self.assertTrue(self.line.acknowledge_alarm())
        self.assertFalse(self.outputs["buzzer"])
        self.assertEqual(self.line.snapshot()["state"], "FAULT_STOPPED")
        self.assertEqual(self.line.snapshot()["fault_code"], "BLOW_INTERRUPTED")
        self.assertTrue(self.line.request_purge())

    def test_purge_invalidates_old_results_and_finishes_when_line_is_quiet(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.capture_completed(1, self.line.epoch)
        old_epoch = self.line.epoch
        self.line.request_controlled_stop()
        self.line.tick(self.clock.advance(0.2))
        self.assertTrue(self.line.request_purge())
        self.assertEqual(self.line.state, ConveyorState.PURGE_PREPARING)
        self.assertFalse(self.line.inspection_completed(1, old_epoch, "NG"))
        self.assertEqual(len(self.line.fifo), 0)
        self.assertTrue(self.outputs["waste_removal"])
        self.assertFalse(self.outputs["conveyor_run"])

        self.line.tick(self.clock.advance(0.11))
        self.assertEqual(self.line.state, ConveyorState.PURGE_RUNNING)
        self.assertTrue(self.outputs["conveyor_run"])
        self.line.tick(self.clock.advance(0.6))
        self.assertEqual(self.line.state, ConveyorState.READY_STOPPED)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

    def test_purge_interlock_requires_explicit_continue(self) -> None:
        self.assertTrue(self.line.request_purge())
        self.line.tick(self.clock.advance(0.11))
        self.assertTrue(self.outputs["conveyor_run"])
        self.line.handle_input_change("door_closed", False)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])
        self.line.handle_input_change("door_closed", True)
        self.assertEqual(self.line.state, ConveyorState.PURGE_PAUSED)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertTrue(self.line.continue_purge())
        self.assertTrue(self.outputs["waste_removal"])
        self.assertFalse(self.outputs["conveyor_run"])

    def test_outlet_jam_is_latched_fault(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")
        self.line.handle_input_change("good_outlet_sensor", True)
        self.line.tick(self.clock.advance(1.1))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT_STOPPED")
        self.assertEqual(snapshot["fault_code"], "JAM_DETECTED")
        self.assertTrue(self.outputs["buzzer"])
        self.assertTrue(self.line.acknowledge_alarm())
        self.assertFalse(self.outputs["buzzer"])
        self.assertEqual(self.line.snapshot()["state"], "FAULT_STOPPED")
        self.line.tick(self.clock.advance(0.1))
        self.assertFalse(self.outputs["buzzer"])
        self.line.handle_input_change("good_outlet_sensor", False)
        self.assertTrue(self.line.acknowledge_alarm())
        self.assertEqual(self.line.snapshot()["state"], "READY_STOPPED")

        self.assertTrue(self.line.request_start())
        self.line.handle_input_change("end_test_sensor", True)
        self.line.tick(self.clock.advance(1.1))
        self.assertEqual(self.line.snapshot()["fault_code"], "JAM_DETECTED")
        self.assertTrue(self.outputs["buzzer"])

    def test_good_requires_di7_confirmation(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")

        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fifo_count"], 0)
        self.assertEqual(snapshot["good_outlet_pending_count"], 1)
        self.assertEqual(snapshot["inflight_count"], 1)

        self.edge("good_outlet_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["good_outlet_pending_count"], 0)
        self.assertEqual(snapshot["inflight_count"], 0)
        self.assertEqual(snapshot["fault_code"], "")

    def test_outlet_calibration_logs_use_millisecond_motion_time(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")

        self.clock.advance(0.234)
        self.edge("good_outlet_sensor")

        self.assertTrue(
            any(
                "GOOD item=1 passed DI1" in entry
                and "window_ms=0.0..500.0" in entry
                for entry in self.logs
            )
        )
        self.assertTrue(
            any(
                "DI7 edge:" in entry
                and "item=1/result=GOOD/expected=DI7/elapsed_ms=234.0" in entry
                for entry in self.logs
            )
        )
        self.assertTrue(
            any(
                "outlet confirmed: item=1, outlet=DI7, travel_ms=234.0" in entry
                for entry in self.logs
            )
        )

    def test_outlet_signal_logs_include_high_and_low_millisecond_intervals(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")

        self.clock.advance(0.234)
        self.line.handle_input_change("good_outlet_sensor", True)
        self.clock.advance(0.125)
        self.line.handle_input_change("good_outlet_sensor", False)

        self.assertTrue(
            any(
                "DI7 signal ON:" in entry
                and "low_wall_ms=234.0" in entry
                and "low_motion_ms=234.0" in entry
                and "debounce_ms=20" in entry
                for entry in self.logs
            )
        )
        self.assertTrue(
            any(
                "DI7 signal OFF:" in entry
                and "high_wall_ms=125.0" in entry
                and "high_motion_ms=125.0" in entry
                and "debounce_ms=20" in entry
                for entry in self.logs
            )
        )

    def test_ng_requires_di8_confirmation(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")
        self.assertEqual(self.line.snapshot()["waste_outlet_pending_count"], 1)

        self.edge("waste_outlet_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["waste_outlet_pending_count"], 0)
        self.assertEqual(snapshot["inflight_count"], 0)
        self.assertEqual(snapshot["fault_code"], "")

    def test_ng_guard_completes_without_di8_when_confirmation_is_disabled(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")
        self.assertEqual(self.line.snapshot()["waste_outlet_pending_count"], 1)

        self.line.tick(self.clock.advance(0.51))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["waste_outlet_pending_count"], 0)
        self.assertEqual(snapshot["inflight_count"], 0)
        self.assertEqual(snapshot["fault_code"], "")
        self.assertTrue(any("reject guard passed" in entry for entry in self.logs))

    def test_di6_short_pulse_does_not_fail_ng_guard(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")

        self.edge("end_test_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "")
        self.assertEqual(snapshot["waste_outlet_pending_count"], 1)

        self.line.tick(self.clock.advance(0.51))
        self.assertEqual(self.line.snapshot()["waste_outlet_pending_count"], 0)

    def test_ng_guard_reports_reject_failure_on_di7(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")

        self.edge("good_outlet_sensor")
        self.assertEqual(
            self.line.snapshot()["fault_code"],
            "REJECT_FAILED_WRONG_OUTLET",
        )

    def test_di8_edge_does_not_complete_ng_guard_when_confirmation_is_disabled(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")

        self.edge("waste_outlet_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["waste_outlet_pending_count"], 1)
        self.assertEqual(snapshot["fault_code"], "")

    def test_di8_jam_monitor_remains_enabled_when_confirmation_is_disabled(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.line.handle_input_change("waste_outlet_sensor", True)
        self.line.tick(self.clock.advance(1.1))

        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "JAM_DETECTED")
        self.assertEqual(snapshot["fault_input"], "waste_outlet_sensor")

    def test_di6_remains_a_dedicated_jam_sensor(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.line.handle_input_change("end_test_sensor", True)
        self.line.tick(self.clock.advance(1.1))

        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "JAM_DETECTED")
        self.assertEqual(snapshot["fault_input"], "end_test_sensor")

    def test_di7_cannot_skip_earlier_ng_to_confirm_later_good(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.tick(self.clock.advance(0.5))
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.line.inspection_completed(2, self.line.epoch, "OK")
        self.edge("reject_position_sensor")
        self.line.tick(self.clock.advance(0.31))
        self.edge("reject_position_sensor")

        self.edge("good_outlet_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "REJECT_FAILED_WRONG_OUTLET")
        self.assertEqual(snapshot["good_outlet_pending_count"], 1)
        edge_log = next(entry for entry in self.logs if "DI7 edge:" in entry)
        self.assertIn(
            "item=1/result=NG/expected=DI7_REJECT_GUARD/elapsed_ms=310.0",
            edge_log,
        )
        self.assertIn(
            "item=2/result=GOOD/expected=DI7/elapsed_ms=0.0",
            edge_log,
        )

    def test_missing_good_outlet_confirmation_times_out(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.edge("reject_position_sensor")

        self.line.tick(self.clock.advance(0.51))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "GOOD_OUTLET_TIMEOUT")
        self.assertEqual(snapshot["fault_recovery"], "PURGE_REQUIRED")

    def test_ng_reaching_di7_reports_reject_failure(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")

        self.line.handle_input_change("good_outlet_sensor", True)
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "REJECT_FAILED_WRONG_OUTLET")
        self.assertEqual(snapshot["fault_recovery"], "PURGE_REQUIRED")

    def test_two_good_items_need_two_distinct_di7_edges(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.line.inspection_completed(2, self.line.epoch, "OK")
        self.edge("reject_position_sensor")
        self.edge("reject_position_sensor")
        self.assertEqual(self.line.snapshot()["good_outlet_pending_count"], 2)

        self.edge("good_outlet_sensor")
        self.assertEqual(self.line.snapshot()["good_outlet_pending_count"], 1)
        self.line.tick(self.clock.advance(0.51))
        self.assertEqual(self.line.snapshot()["fault_code"], "GOOD_OUTLET_TIMEOUT")

    def test_multiple_products_result_stops_and_requires_purge(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.assertTrue(
            self.line.inspection_completed(
                1,
                self.line.epoch,
                "MULTIPLE_PRODUCTS_IN_FOV",
                detail="two products detected",
            )
        )
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "MULTIPLE_PRODUCTS_IN_FOV")
        self.assertEqual(snapshot["fault_recovery"], "PURGE_REQUIRED")

    def test_di0_minimum_clear_interval_can_detect_tight_spacing(self) -> None:
        self.line.config = replace(self.line.config, front_sensor_min_clear_ms=100)
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.handle_input_change("camera_trigger_sensor", True)

        self.assertEqual(
            self.line.snapshot()["fault_code"],
            "PRODUCT_SPACING_TOO_SMALL",
        )

    def test_di0_maximum_active_time_can_detect_one_long_pulse(self) -> None:
        self.line.config = replace(self.line.config, front_sensor_max_active_ms=100)
        self.line.request_start()
        self.line.handle_input_change("camera_trigger_sensor", True)
        self.line.tick(self.clock.advance(0.11))

        self.assertEqual(
            self.line.snapshot()["fault_code"],
            "PRODUCT_SPACING_TOO_SMALL",
        )

    def test_raw_di8_presence_does_not_block_purge_start(self) -> None:
        self.line.config = replace(
            self.line.config,
            waste_outlet_confirmation_enabled=False,
        )
        self.line.handle_input_change("waste_outlet_sensor", True)
        self.assertTrue(self.line.request_purge())
        self.assertEqual(self.line.state, ConveyorState.PURGE_PREPARING)

        self.line.tick(self.clock.advance(0.11))
        self.assertEqual(self.line.state, ConveyorState.PURGE_RUNNING)
        self.line.tick(self.clock.advance(1.1))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["fault_code"], "JAM_DETECTED")
        self.assertEqual(snapshot["fault_input"], "waste_outlet_sensor")

    def test_purge_requires_the_line_to_be_stopped(self) -> None:
        self.assertTrue(self.line.request_start())
        self.assertFalse(self.line.request_purge())
        self.assertTrue(self.outputs["conveyor_run"])

    def test_manual_and_configuration_operations_are_locked_during_production(self) -> None:
        ready = self.line.snapshot()
        self.assertTrue(ready["manual_operations_permitted"])
        self.assertTrue(ready["configuration_operations_permitted"])

        self.assertTrue(self.line.request_start())
        running = self.line.snapshot()
        self.assertFalse(running["manual_operations_permitted"])
        self.assertFalse(running["configuration_operations_permitted"])

        self.edge("camera_trigger_sensor")
        self.line.capture_completed(1, self.line.epoch)
        self.line.request_controlled_stop()
        self.line.tick(self.clock.advance(0.2))
        stopped_with_fifo = self.line.snapshot()
        self.assertFalse(stopped_with_fifo["manual_operations_permitted"])
        self.assertFalse(stopped_with_fifo["configuration_operations_permitted"])

    def test_legacy_config_keys_map_to_canonical_names(self) -> None:
        config = ConveyorConfig.from_mapping(
            {
                "reject_delay_ms": 12,
                "reject_duration_ms": 345,
                "fifo_max_items": 9,
                "item_to_reject_timeout_s": 6.5,
                "end_sensor_enabled": False,
                "end_sensor_jam_s": 1.1,
                "good_outlet_jam_s": 1.2,
                "waste_outlet_jam_s": 1.3,
            }
        )
        self.assertEqual(config.reject_blow_delay_ms, 12)
        self.assertEqual(config.reject_blow_duration_ms, 345)
        self.assertEqual(config.max_inflight_items, 9)
        self.assertEqual(config.front_to_reject_max_run_ms, 6500)
        self.assertFalse(config.end_test_sensor_enabled)
        self.assertEqual(config.end_test_blocked_timeout_s, 1.1)
        self.assertEqual(config.good_outlet_blocked_timeout_s, 1.2)
        self.assertEqual(config.waste_outlet_blocked_timeout_s, 1.3)

    def test_config_rejects_string_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "upper_door_sensor_enabled must be a boolean"):
            ConveyorConfig.from_mapping({"upper_door_sensor_enabled": "false"})

    def test_config_rejects_unknown_field_instead_of_ignoring_typo(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown conveyor configuration field"):
            ConveyorConfig.from_mapping({"reject_blow_duraton_ms": 300})

    def test_config_rejects_invalid_outlet_arrival_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "good_outlet_arrival_min_run_ms"):
            ConveyorConfig.from_mapping(
                {
                    "good_outlet_arrival_min_run_ms": 3001,
                    "good_outlet_arrival_max_run_ms": 3000,
                }
            )

    def test_config_rejects_non_positive_capacity_and_poll_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_inflight_items"):
            ConveyorConfig.from_mapping({"max_inflight_items": 0})
        with self.assertRaisesRegex(ValueError, "poll_interval_ms"):
            ConveyorConfig.from_mapping({"poll_interval_ms": 0})

    def test_config_rejects_impossible_purge_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "purge_min_run_s"):
            ConveyorConfig.from_mapping({"purge_min_run_s": 31.0, "purge_max_run_s": 30.0})

    def test_output_failure_reports_unknown_physical_safe_state(self) -> None:
        failures_enabled = False

        def write_output(name: str, on: bool) -> None:
            if failures_enabled:
                raise OSError(f"single write failed for {name}")

        def write_outputs(updates: dict[str, bool]) -> None:
            if failures_enabled:
                raise OSError("batch write failed")

        logs: list[str] = []
        line = ConveyorLineController(
            config=self.config,
            output_writer=write_output,
            output_batch_writer=write_outputs,
            inspection_requester=lambda _sequence_id, _epoch: None,
            log_writer=logs.append,
            clock=self.clock,
        )
        line.initialize_inputs(
            {"safety_ok": True, "door_closed": True, "door_upper_closed": True},
            io_ready=True,
        )
        failures_enabled = True

        with self.assertRaisesRegex(RuntimeError, "physical output state unknown"):
            line.request_start()

        snapshot = line.snapshot()
        self.assertEqual(snapshot["fault_code"], "OUTPUT_WRITE_FAILED")
        self.assertIn("safe-off failed", snapshot["fault_detail"])
        self.assertTrue(any("physical output state unknown" in entry for entry in logs))

    def test_default_config_uses_only_canonical_control_names(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "defaults"
            / "conveyor_control.json"
        )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(
            {
                "reject_blow_delay_ms",
                "reject_blow_duration_ms",
                "reject_following_item_guard_ms",
                "max_inflight_items",
                "front_to_reject_max_run_ms",
                "inspection_result_wait_timeout_ms",
                "end_test_sensor_enabled",
                "waste_outlet_confirmation_enabled",
                "end_test_blocked_timeout_s",
                "good_outlet_blocked_timeout_s",
                "waste_outlet_blocked_timeout_s",
                "good_outlet_arrival_min_run_ms",
                "good_outlet_arrival_max_run_ms",
                "waste_outlet_arrival_min_run_ms",
                "waste_outlet_arrival_max_run_ms",
            }.issubset(payload)
        )
        self.assertFalse(
            {
                "reject_delay_ms",
                "reject_duration_ms",
                "fifo_max_items",
                "item_to_reject_timeout_s",
                "end_sensor_enabled",
                "end_sensor_jam_s",
                "good_outlet_jam_s",
                "waste_outlet_jam_s",
                "end_test_jam_timeout_s",
                "good_jam_timeout_s",
                "waste_jam_timeout_s",
            }.intersection(payload)
        )
        self.assertIsInstance(payload.get("_comments"), dict)
        self.assertIn("reject_blow_delay_ms", payload["_comments"])
        self.assertIn("good_outlet_arrival_max_run_ms", payload["_comments"])
        self.assertEqual(
            set(payload) - {"_comments"},
            set(payload["_comments"]),
        )

    def test_latched_fault_survives_safety_loss_and_restore(self) -> None:
        self.line.request_start()
        self.edge("reject_position_sensor")
        self.assertEqual(self.line.state, ConveyorState.FAULT_STOPPED)
        self.line.handle_input_change("safety_ok", False)
        self.assertEqual(self.line.state, ConveyorState.SAFETY_LOCKED)
        self.line.handle_input_change("safety_ok", True)
        self.assertEqual(self.line.state, ConveyorState.FAULT_STOPPED)
        self.assertEqual(self.line.snapshot()["fault_code"], "FIFO_UNDERFLOW")

    def test_runtime_io_mapping_matches_confirmed_field_polarity(self) -> None:
        mapping_path = Path(__file__).resolve().parents[1] / "config" / "defaults" / "io_mapping.json"
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        expected_di = {
            "camera_trigger_sensor": (0, False),
            "reject_position_sensor": (1, False),
            "start_button": (2, True),
            "stop_button": (3, False),
            "safety_ok": (5, True),
            "end_test_sensor": (6, True),
            "good_outlet_sensor": (7, True),
            "waste_outlet_sensor": (8, True),
            "door_closed": (9, True),
            "door_upper_closed": (10, True),
        }
        for name, (channel, active_high) in expected_di.items():
            self.assertEqual(payload["di"][name]["channel"], channel)
            self.assertIs(payload["di"][name]["active_high"], active_high)
        for name, channel in {
            "waste_removal": 3,
            "conveyor_run": 4,
            "button_green": 5,
            "button_blue": 7,
            "buzzer": 8,
            "button_red": 9,
        }.items():
            self.assertEqual(payload["do"][name]["channel"], channel)
            self.assertIs(payload["do"][name]["active_high"], False)


if __name__ == "__main__":
    unittest.main()
