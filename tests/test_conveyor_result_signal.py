from __future__ import annotations

import unittest
from types import SimpleNamespace

from application.runtime.conveyor import (
    _reported_product_count,
    _show_conveyor_inspection_result,
)


class _FakeTowerLightController:
    def __init__(self) -> None:
        self.events: list[str] = []

    def show_ok(self) -> None:
        self.events.append("OK")

    def show_ng(self) -> None:
        self.events.append("NG")


class _FakeRuntime:
    def __init__(self) -> None:
        self._tower_light_controller = _FakeTowerLightController()


class ConveyorResultSignalTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
