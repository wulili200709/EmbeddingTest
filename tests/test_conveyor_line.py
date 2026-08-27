from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "domain" / "conveyor_line.py"
SPEC = importlib.util.spec_from_file_location("conveyor_line_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ConveyorConfig = MODULE.ConveyorConfig
ConveyorLineController = MODULE.ConveyorLineController
ConveyorState = MODULE.ConveyorState
InspectionStatus = MODULE.InspectionStatus


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
        self.requests: list[tuple[int, int]] = []
        self.logs: list[str] = []
        self.config = ConveyorConfig(
            capture_commit_guard_ms=100,
            controlled_stop_timeout_ms=1500,
            reject_delay_ms=0,
            reject_duration_ms=300,
            fifo_max_items=8,
            item_to_reject_timeout_s=5.0,
            end_sensor_enabled=True,
            upper_door_sensor_enabled=True,
            end_sensor_jam_s=1.0,
            good_outlet_jam_s=1.0,
            waste_outlet_jam_s=1.0,
            purge_air_lead_ms=100,
            purge_min_run_s=0.5,
            purge_tail_run_s=0.5,
            purge_quiet_s=0.5,
            purge_max_run_s=5.0,
        )
        self.line = ConveyorLineController(
            config=self.config,
            output_writer=lambda name, on: self.outputs.__setitem__(name, on),
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
        self.assertEqual(self.line.state, ConveyorState.STOPPED)
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertTrue(self.line.request_start())

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
        self.assertEqual(self.line.state, ConveyorState.STOPPED)
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

    def test_pending_result_at_di1_faults_instead_of_guessing(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.edge("reject_position_sensor")
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT")
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
        self.assertEqual(self.line.state, ConveyorState.STOPPED)
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
        self.assertEqual(self.line.state, ConveyorState.STOPPED)
        self.line.inspection_completed(1, self.line.epoch, "OK")
        self.assertEqual(self.line.fifo[0].inspection_status.value, "GOOD")

    def test_controlled_stop_timeout_forces_safe_outputs_and_alarm(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.request_controlled_stop()
        self.line.tick(self.clock.advance(1.6))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT")
        self.assertEqual(snapshot["fault_code"], "CONTROLLED_STOP_TIMEOUT")
        self.assertFalse(self.outputs["conveyor_run"])
        self.assertFalse(self.outputs["waste_removal"])

    def test_safety_pause_freezes_belt_distance_timers_and_fifo(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.inspection_completed(1, self.line.epoch, "NG")
        self.edge("reject_position_sensor")
        self.line.tick(self.clock.advance(0.1))
        motion_before = self.line.snapshot()["motion_elapsed_s"]

        self.line.handle_input_change("safety_ok", False)
        self.line.tick(self.clock.advance(2.0))
        self.assertEqual(self.line.snapshot()["motion_elapsed_s"], motion_before)
        self.line.handle_input_change("safety_ok", True)
        self.line.request_start()
        self.line.tick(self.clock.advance(0.1))
        self.assertTrue(self.outputs["waste_removal"])
        self.line.tick(self.clock.advance(0.11))
        self.assertFalse(self.outputs["waste_removal"])

    def test_purge_invalidates_old_results_and_finishes_when_line_is_quiet(self) -> None:
        self.line.request_start()
        self.edge("camera_trigger_sensor")
        self.line.capture_completed(1, self.line.epoch)
        old_epoch = self.line.epoch
        self.line.request_controlled_stop()
        self.line.tick(self.clock.advance(0.2))
        self.assertTrue(self.line.request_purge())
        self.assertFalse(self.line.inspection_completed(1, old_epoch, "NG"))
        self.assertEqual(len(self.line.fifo), 0)
        self.assertTrue(self.outputs["waste_removal"])
        self.assertFalse(self.outputs["conveyor_run"])

        self.line.tick(self.clock.advance(0.11))
        self.assertTrue(self.outputs["conveyor_run"])
        self.line.tick(self.clock.advance(0.6))
        self.assertEqual(self.line.state, ConveyorState.STOPPED)
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
        self.line.handle_input_change("good_outlet_sensor", True)
        self.line.tick(self.clock.advance(1.1))
        snapshot = self.line.snapshot()
        self.assertEqual(snapshot["state"], "FAULT")
        self.assertEqual(snapshot["fault_code"], "JAM_DETECTED")
        self.assertTrue(self.outputs["buzzer"])

    def test_latched_fault_survives_safety_loss_and_restore(self) -> None:
        self.line.request_start()
        self.edge("reject_position_sensor")
        self.assertEqual(self.line.state, ConveyorState.FAULT)
        self.line.handle_input_change("safety_ok", False)
        self.assertEqual(self.line.state, ConveyorState.SAFETY_LOCKED)
        self.line.handle_input_change("safety_ok", True)
        self.assertEqual(self.line.state, ConveyorState.FAULT)
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
