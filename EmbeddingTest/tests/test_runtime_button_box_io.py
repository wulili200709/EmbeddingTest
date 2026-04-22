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
    _handle_reset_button_from_di,
    _handle_reset_button_release_from_di,
    _on_conveyor_start_rising,
    _on_conveyor_stop_falling,
    _on_reset_button_falling,
    _on_reset_button_rising,
    _start_conveyor_from_di,
    _stop_conveyor_from_di,
)


class _FakeSignal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(str(message))


class _FakeMapping:
    def di_names(self) -> list[str]:
        return ["foot_switch", "conveyor_start", "conveyor_stop", "reset_button"]

    def do_names(self) -> list[str]:
        return ["conveyor_run", "buzzer", "button_green", "button_red", "button_blue"]


class _FakeController:
    def __init__(self) -> None:
        self.mapping = _FakeMapping()
        self.outputs = {
            "conveyor_run": False,
            "buzzer": True,
            "button_green": False,
            "button_red": False,
            "button_blue": False,
        }
        self.is_open = True

    def set_output(self, name: str, on: bool) -> None:
        self.outputs[str(name)] = bool(on)

    def read_output(self, name: str) -> bool:
        return bool(self.outputs[str(name)])


class _FakeEvent:
    def __init__(self, name: str) -> None:
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


class RuntimeButtonBoxIoTest(unittest.TestCase):
    def test_conveyor_start_rising_queues_start_slot(self) -> None:
        runtime = _FakeRuntime()

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_conveyor_start_rising(runtime, _FakeEvent("conveyor_start"))

        invoke_method.assert_called_once_with(
            runtime,
            "_start_conveyor_from_di",
            QtCore.Qt.QueuedConnection,
        )

    def test_conveyor_stop_falling_queues_stop_slot(self) -> None:
        runtime = _FakeRuntime()

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_conveyor_stop_falling(runtime, _FakeEvent("conveyor_stop"))

        invoke_method.assert_called_once_with(
            runtime,
            "_stop_conveyor_from_di",
            QtCore.Qt.QueuedConnection,
        )

    def test_reset_button_rising_queues_reset_slot(self) -> None:
        runtime = _FakeRuntime()

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_reset_button_rising(runtime, _FakeEvent("reset_button"))

        invoke_method.assert_called_once_with(
            runtime,
            "_handle_reset_button_from_di",
            QtCore.Qt.QueuedConnection,
        )

    def test_reset_button_falling_queues_release_slot(self) -> None:
        runtime = _FakeRuntime()

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_reset_button_falling(runtime, _FakeEvent("reset_button"))

        invoke_method.assert_called_once_with(
            runtime,
            "_handle_reset_button_release_from_di",
            QtCore.Qt.QueuedConnection,
        )

    def test_di2_start_turns_on_conveyor_and_green_button_light(self) -> None:
        runtime = _FakeRuntime()

        _start_conveyor_from_di(runtime)

        self.assertTrue(runtime._io_controller.outputs["conveyor_run"])
        self.assertFalse(runtime._io_controller.outputs["buzzer"])
        self.assertTrue(runtime._io_controller.outputs["button_green"])
        self.assertFalse(runtime._io_controller.outputs["button_red"])
        self.assertFalse(runtime._io_controller.outputs["button_blue"])
        self.assertIn("DI2 started conveyor", runtime.status_messages)

    def test_di3_stop_turns_on_red_button_light(self) -> None:
        runtime = _FakeRuntime()
        runtime._io_controller.outputs["conveyor_run"] = True
        runtime._conveyor_running = True
        runtime._io_controller.outputs["button_green"] = True

        _stop_conveyor_from_di(runtime)

        self.assertFalse(runtime._io_controller.outputs["conveyor_run"])
        self.assertFalse(runtime._io_controller.outputs["button_green"])
        self.assertTrue(runtime._io_controller.outputs["button_red"])
        self.assertFalse(runtime._io_controller.outputs["button_blue"])
        self.assertIn("DI3 stopped conveyor", runtime.status_messages)

    def test_di4_reset_turns_on_blue_button_light_only(self) -> None:
        runtime = _FakeRuntime()
        runtime._io_controller.outputs["conveyor_run"] = True
        runtime._io_controller.outputs["button_green"] = True

        _handle_reset_button_from_di(runtime)

        self.assertTrue(runtime._io_controller.outputs["conveyor_run"])
        self.assertFalse(runtime._io_controller.outputs["button_green"])
        self.assertFalse(runtime._io_controller.outputs["button_red"])
        self.assertTrue(runtime._io_controller.outputs["button_blue"])
        self.assertIn("DI4 reset button pressed", runtime.status_messages)

    def test_di4_reset_release_turns_off_blue_button_light(self) -> None:
        runtime = _FakeRuntime()
        runtime._io_controller.outputs["button_blue"] = True

        _handle_reset_button_release_from_di(runtime)

        self.assertFalse(runtime._io_controller.outputs["button_blue"])
        self.assertIn("DI4 reset button released", runtime.status_messages)


if __name__ == "__main__":
    unittest.main()
