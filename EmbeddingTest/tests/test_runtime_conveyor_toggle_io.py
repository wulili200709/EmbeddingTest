from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6 import QtCore


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.hardware import (
    _on_conveyor_toggle_rising,
    _toggle_conveyor_run_from_di,
)


class _FakeSignal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(str(message))


class _FakeMapping:
    def di_names(self) -> list[str]:
        return ["foot_switch", "conveyor_toggle"]

    def do_names(self) -> list[str]:
        return ["conveyor_run", "buzzer"]


class _FakeController:
    def __init__(self) -> None:
        self.mapping = _FakeMapping()
        self.outputs = {
            "conveyor_run": False,
            "buzzer": True,
        }
        self.is_open = True

    def set_output(self, name: str, on: bool) -> None:
        self.outputs[str(name)] = bool(on)

    def read_output(self, name: str) -> bool:
        return bool(self.outputs[str(name)])


class _FakeEvent:
    def __init__(self, name: str = "conveyor_toggle") -> None:
        self.name = name


class _FakeRuntime:
    def __init__(self) -> None:
        self._io_controller = _FakeController()
        self._conveyor_running = False
        self.logAppended = _FakeSignal()
        self.warningOccurred = _FakeSignal()
        self.status_messages: list[str] = []

    def _update_status(self, message: str) -> None:
        self.status_messages.append(str(message))


class RuntimeConveyorToggleIoTest(unittest.TestCase):
    def test_conveyor_toggle_rising_queues_toggle_slot(self) -> None:
        runtime = _FakeRuntime()

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_conveyor_toggle_rising(runtime, _FakeEvent("conveyor_toggle"))

        invoke_method.assert_called_once_with(
            runtime,
            "_toggle_conveyor_run_from_di",
            QtCore.Qt.QueuedConnection,
        )

    def test_conveyor_toggle_ignores_other_di_events(self) -> None:
        runtime = _FakeRuntime()

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_conveyor_toggle_rising(runtime, _FakeEvent("foot_switch"))

        invoke_method.assert_not_called()

    def test_di2_toggle_starts_conveyor_and_silences_buzzer(self) -> None:
        runtime = _FakeRuntime()

        _toggle_conveyor_run_from_di(runtime)

        self.assertTrue(runtime._io_controller.outputs["conveyor_run"])
        self.assertFalse(runtime._io_controller.outputs["buzzer"])
        self.assertIn("DI2 started conveyor", runtime.status_messages)

    def test_di2_toggle_stops_conveyor(self) -> None:
        runtime = _FakeRuntime()
        runtime._io_controller.outputs["conveyor_run"] = True
        runtime._conveyor_running = True
        runtime._io_controller.outputs["buzzer"] = True

        _toggle_conveyor_run_from_di(runtime)

        self.assertFalse(runtime._io_controller.outputs["conveyor_run"])
        self.assertTrue(runtime._io_controller.outputs["buzzer"])
        self.assertIn("DI2 stopped conveyor", runtime.status_messages)


if __name__ == "__main__":
    unittest.main()
