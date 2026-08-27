from __future__ import annotations

import unittest

from application.runtime.conveyor import _show_conveyor_inspection_result


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


if __name__ == "__main__":
    unittest.main()
