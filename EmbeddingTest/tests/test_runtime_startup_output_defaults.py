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
)


class _FakeSignal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(str(message))


class _FakeRuntime:
    def __init__(self) -> None:
        self.logAppended = _FakeSignal()


class _FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def set_output(self, name: str, on: bool) -> None:
        self.calls.append((str(name), bool(on)))


class _FailingController:
    def set_output(self, name: str, on: bool) -> None:
        raise RuntimeError("write failed")


class StartupOutputDefaultsTest(unittest.TestCase):
    def test_reserved_output_is_enabled_on_startup(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        _apply_startup_output_defaults(runtime, controller)

        self.assertEqual(controller.calls, [("reserved_out_1", True)])
        self.assertEqual(
            runtime.logAppended.messages,
            ["[IO] startup output default applied: reserved_out_1=ON"],
        )

    def test_reserved_output_is_disabled_on_shutdown(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        _apply_shutdown_output_defaults(runtime, controller)

        self.assertEqual(controller.calls, [("reserved_out_1", False)])
        self.assertEqual(
            runtime.logAppended.messages,
            ["[IO] shutdown output default applied: reserved_out_1=OFF"],
        )

    def test_startup_output_failure_is_logged_without_raising(self) -> None:
        runtime = _FakeRuntime()

        _apply_startup_output_defaults(runtime, _FailingController())

        self.assertEqual(len(runtime.logAppended.messages), 1)
        self.assertIn("failed to apply startup output default", runtime.logAppended.messages[0])
        self.assertIn("reserved_out_1=True", runtime.logAppended.messages[0])

    def test_shutdown_output_failure_is_logged_without_raising(self) -> None:
        runtime = _FakeRuntime()

        _apply_shutdown_output_defaults(runtime, _FailingController())

        self.assertEqual(len(runtime.logAppended.messages), 1)
        self.assertIn("failed to apply shutdown output default", runtime.logAppended.messages[0])
        self.assertIn("reserved_out_1=False", runtime.logAppended.messages[0])


if __name__ == "__main__":
    unittest.main()
