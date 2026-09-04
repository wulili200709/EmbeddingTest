from __future__ import annotations

import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from application.runtime.conveyor import (
    _is_template_match_failure,
    _reported_product_count,
    _run_conveyor_inspection,
    _show_conveyor_inspection_result,
    acknowledge_conveyor_alarm,
)
from domain import InspectionItem


class _FakeTowerLightController:
    def __init__(self) -> None:
        self.events: list[str] = []

    def show_ok(self) -> None:
        self.events.append("OK")

    def show_ng(self) -> None:
        self.events.append("NG")

    def silence_buzzer(self) -> None:
        self.events.append("SILENCE")


class _FakeConveyorController:
    def __init__(self) -> None:
        self.acknowledged = False

    def acknowledge_alarm(self) -> bool:
        self.acknowledged = True
        return True


class _FakeRuntime:
    def __init__(self) -> None:
        self._tower_light_controller = _FakeTowerLightController()
        self._conveyor_controller = _FakeConveyorController()


class _Signal:
    def __init__(self) -> None:
        self.values: list[tuple] = []

    def emit(self, *args) -> None:
        self.values.append(tuple(args))


class ConveyorResultSignalTests(unittest.TestCase):
    def test_alarm_reset_silences_result_buzzer_and_notifies_line_controller(self) -> None:
        runtime = _FakeRuntime()
        acknowledge_conveyor_alarm(runtime)
        self.assertEqual(runtime._tower_light_controller.events, ["SILENCE"])
        self.assertTrue(runtime._conveyor_controller.acknowledged)

    def test_ng_uses_shared_configurable_tower_light_buzzer(self) -> None:
        runtime = _FakeRuntime()
        _show_conveyor_inspection_result(runtime, "NG")
        self.assertEqual(runtime._tower_light_controller.events, ["NG"])

    def test_ok_uses_result_light_without_ng_buzzer(self) -> None:
        runtime = _FakeRuntime()
        _show_conveyor_inspection_result(runtime, "OK")
        self.assertEqual(runtime._tower_light_controller.events, ["OK"])

    def test_every_non_ok_result_uses_ng_signal(self) -> None:
        runtime = _FakeRuntime()
        _show_conveyor_inspection_result(runtime, "ERROR")
        self.assertEqual(runtime._tower_light_controller.events, ["NG"])

    def test_product_count_reads_nested_algorithm_result(self) -> None:
        response = SimpleNamespace(
            raw_row={"item_rows": [{"detected_product_count": 2}]},
            measurements=(),
        )
        self.assertEqual(_reported_product_count(response), 2)

    def test_product_count_uses_largest_reported_value(self) -> None:
        response = SimpleNamespace(
            raw_row={"product_count": 1},
            measurements=({"object_count": "3"},),
        )
        self.assertEqual(_reported_product_count(response), 3)

    def test_product_count_is_optional(self) -> None:
        response = SimpleNamespace(raw_row={}, measurements=())
        self.assertIsNone(_reported_product_count(response))

    def test_template_match_failure_is_recognized(self) -> None:
        self.assertTrue(_is_template_match_failure(RuntimeError("match failure")))
        self.assertFalse(_is_template_match_failure(RuntimeError("camera disconnected")))

    def test_first_conveyor_match_failure_is_finalized_as_ng(self) -> None:
        item = InspectionItem("item-1", "roi1", "cam1", "roi1", "meanintensity")
        runtime = SimpleNamespace(
            _session=SimpleNamespace(product_dir="."),
            _runtime_context=SimpleNamespace(inspection_items=[item]),
            _inspection_executor=SimpleNamespace(
                execute=lambda _request: (_ for _ in ()).throw(RuntimeError("match failure"))
            ),
            _inspect_lock=threading.RLock(),
            _frame_lock=threading.RLock(),
            previewCycleStarted=_Signal(),
            previewUpdated=_Signal(),
            logAppended=_Signal(),
            _last_preview_frames={},
            _last_item_results_by_camera={},
            _last_runtime_result=None,
        )
        finalized: list[object] = []

        def finalize(outcome, _release_status, *, active_roles=None) -> None:
            finalized.append(outcome)
            runtime._last_runtime_result = SimpleNamespace(
                summary_text=lambda: "cam1=NG; match failure"
            )

        runtime._build_pending_runtime_result = lambda **_kwargs: SimpleNamespace(task_id="pending")
        runtime._finalize_trigger_outcome = finalize
        frame = SimpleNamespace(camera_serial="serial", frame_num=1, host_timestamp=1)

        with patch(
            "application.runtime.controller.frame_to_bgr_image",
            return_value=np.zeros((16, 16, 3), dtype=np.uint8),
        ):
            result, detail = _run_conveyor_inspection(
                runtime,
                sequence_id=1,
                epoch=1,
                captured=[
                    {
                        "role": "cam1",
                        "physical_role": "cam1",
                        "frame": frame,
                        "capture_ms": 2.0,
                    }
                ],
            )

        self.assertEqual(result, "NG")
        self.assertIn("match failure", detail)
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].final_result, "NG")
        self.assertEqual(runtime._last_item_results_by_camera["cam1"][0].result, "NG")


if __name__ == "__main__":
    unittest.main()
