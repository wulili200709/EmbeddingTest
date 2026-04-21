from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.hardware import (
    _apply_shutdown_output_defaults,
    _apply_startup_output_defaults,
    _set_conveyor_run,
)


class _FakeSignal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(str(message))


class _FakeRuntime:
    def __init__(self) -> None:
        self.logAppended = _FakeSignal()
        self._io_controller = None


class _FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.is_open = True

    def set_output(self, name: str, on: bool) -> None:
        self.calls.append((str(name), bool(on)))


class _FailingController:
    is_open = True

    def set_output(self, name: str, on: bool) -> None:
        raise RuntimeError("write failed")


class StartupOutputDefaultsTest(unittest.TestCase):
    def test_reserved_output_is_enabled_on_startup(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        _apply_startup_output_defaults(runtime, controller)

        self.assertEqual(
            controller.calls,
            [("reserved_out_1", True), ("reserved_out_2", False)],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            [
                "[IO] startup output default applied: reserved_out_1=ON",
                "[IO] startup output default applied: reserved_out_2=OFF",
            ],
        )

    def test_reserved_output_is_disabled_on_shutdown(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        _apply_shutdown_output_defaults(runtime, controller)

        self.assertEqual(
            controller.calls,
            [("reserved_out_1", False), ("reserved_out_2", False)],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            [
                "[IO] shutdown output default applied: reserved_out_1=OFF",
                "[IO] shutdown output default applied: reserved_out_2=OFF",
            ],
        )

    def test_startup_output_failure_is_logged_without_raising(self) -> None:
        runtime = _FakeRuntime()

        _apply_startup_output_defaults(runtime, _FailingController())

        self.assertEqual(len(runtime.logAppended.messages), 2)
        self.assertIn("failed to apply startup output default", runtime.logAppended.messages[0])
        self.assertIn("reserved_out_1=True", runtime.logAppended.messages[0])
        self.assertIn("failed to apply startup output default", runtime.logAppended.messages[1])
        self.assertIn("reserved_out_2=False", runtime.logAppended.messages[1])

    def test_shutdown_output_failure_is_logged_without_raising(self) -> None:
        runtime = _FakeRuntime()

        _apply_shutdown_output_defaults(runtime, _FailingController())

        self.assertEqual(len(runtime.logAppended.messages), 2)
        self.assertIn("failed to apply shutdown output default", runtime.logAppended.messages[0])
        self.assertIn("reserved_out_1=False", runtime.logAppended.messages[0])
        self.assertIn("failed to apply shutdown output default", runtime.logAppended.messages[1])
        self.assertIn("reserved_out_2=False", runtime.logAppended.messages[1])

    def test_conveyor_output_is_disabled_for_ng_stop(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()
        runtime._io_controller = controller

        changed = _set_conveyor_run(runtime, False, reason="NG result")

        self.assertTrue(changed)
        self.assertEqual(controller.calls, [("reserved_out_1", False)])
        self.assertEqual(
            runtime.logAppended.messages,
            ["[IO] conveyor output applied: reserved_out_1=OFF (NG result)"],
        )

    def test_conveyor_output_is_enabled_after_release(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()
        runtime._io_controller = controller

        changed = _set_conveyor_run(runtime, True, reason="release granted")

        self.assertTrue(changed)
        self.assertEqual(
            controller.calls,
            [("reserved_out_1", True), ("reserved_out_2", False)],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            [
                "[IO] conveyor output applied: reserved_out_1=ON (release granted)",
                "[IO] buzzer output applied: reserved_out_2=OFF (conveyor running: release granted)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
