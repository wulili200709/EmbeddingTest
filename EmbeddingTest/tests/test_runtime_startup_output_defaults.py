from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.hardware import (
    _apply_io_logic_event,
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


class _FakeTimer:
    def __init__(self, interval: float, callback) -> None:
        self.interval = float(interval)
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


class StartupOutputDefaultsTest(unittest.TestCase):
    def test_reserved_output_is_enabled_on_startup(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        _apply_startup_output_defaults(runtime, controller)

        self.assertEqual(
            controller.calls,
            [
                ("reserved_out_1", True),
                ("reserved_out_2", False),
                ("button_green", True),
                ("button_red", False),
                ("button_blue", False),
            ],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            [
                "[IO] startup output default applied: reserved_out_1=ON",
                "[IO] startup output default applied: reserved_out_2=OFF",
                "[IO] startup output default applied: button_green=ON",
                "[IO] startup output default applied: button_red=OFF",
                "[IO] startup output default applied: button_blue=OFF",
            ],
        )

    def test_reserved_output_is_disabled_on_shutdown(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        _apply_shutdown_output_defaults(runtime, controller)

        self.assertEqual(
            controller.calls,
            [
                ("reserved_out_1", False),
                ("reserved_out_2", False),
                ("button_green", False),
                ("button_red", True),
                ("button_blue", False),
            ],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            [
                "[IO] shutdown output default applied: reserved_out_1=OFF",
                "[IO] shutdown output default applied: reserved_out_2=OFF",
                "[IO] shutdown output default applied: button_green=OFF",
                "[IO] shutdown output default applied: button_red=ON",
                "[IO] shutdown output default applied: button_blue=OFF",
            ],
        )

    def test_startup_output_failure_is_logged_without_raising(self) -> None:
        runtime = _FakeRuntime()

        _apply_startup_output_defaults(runtime, _FailingController())

        self.assertEqual(len(runtime.logAppended.messages), 5)
        self.assertIn("failed to apply startup output default", runtime.logAppended.messages[0])
        self.assertIn("reserved_out_1=True", runtime.logAppended.messages[0])
        self.assertIn("failed to apply startup output default", runtime.logAppended.messages[1])
        self.assertIn("reserved_out_2=False", runtime.logAppended.messages[1])
        self.assertIn("button_green=True", runtime.logAppended.messages[2])
        self.assertIn("button_red=False", runtime.logAppended.messages[3])
        self.assertIn("button_blue=False", runtime.logAppended.messages[4])

    def test_shutdown_output_failure_is_logged_without_raising(self) -> None:
        runtime = _FakeRuntime()

        _apply_shutdown_output_defaults(runtime, _FailingController())

        self.assertEqual(len(runtime.logAppended.messages), 5)
        self.assertIn("failed to apply shutdown output default", runtime.logAppended.messages[0])
        self.assertIn("reserved_out_1=False", runtime.logAppended.messages[0])
        self.assertIn("failed to apply shutdown output default", runtime.logAppended.messages[1])
        self.assertIn("reserved_out_2=False", runtime.logAppended.messages[1])
        self.assertIn("button_green=False", runtime.logAppended.messages[2])
        self.assertIn("button_red=True", runtime.logAppended.messages[3])
        self.assertIn("button_blue=False", runtime.logAppended.messages[4])

    def test_product_override_replaces_startup_actions(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()

        with TemporaryDirectory() as tmp:
            runtime._session = SimpleNamespace(product_dir=tmp)
            override_path = Path(tmp) / "runtime_io_logic.json"
            override_path.write_text(
                json.dumps(
                    {
                        "startup": [
                            {"type": "set_output", "name": "buzzer", "value": True},
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            _apply_startup_output_defaults(runtime, controller)

        self.assertEqual(controller.calls, [("reserved_out_2", True)])
        self.assertEqual(runtime.logAppended.messages, ["[IO] startup output default applied: reserved_out_2=ON"])

    def test_pulse_output_turns_on_then_resets_after_timer(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()
        runtime._io_controller = controller
        created_timers: list[_FakeTimer] = []

        def _build_timer(interval: float, callback):
            timer = _FakeTimer(interval, callback)
            created_timers.append(timer)
            return timer

        with TemporaryDirectory() as tmp:
            runtime._session = SimpleNamespace(product_dir=tmp)
            override_path = Path(tmp) / "runtime_io_logic.json"
            override_path.write_text(
                json.dumps(
                    {
                        "ng": [
                            {
                                "type": "pulse_output",
                                "name": "reject_output",
                                "duration_ms": 150,
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with patch("application.runtime.hardware.threading.Timer", side_effect=_build_timer):
                applied = _apply_io_logic_event(runtime, "ng")

        self.assertTrue(applied)
        self.assertEqual(controller.calls, [("reject_output", True)])
        self.assertEqual(len(created_timers), 1)
        self.assertTrue(created_timers[0].started)
        self.assertAlmostEqual(created_timers[0].interval, 0.15, places=3)

        created_timers[0].callback()

        self.assertEqual(controller.calls, [("reject_output", True), ("reject_output", False)])

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
            [("reserved_out_1", True)],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            ["[IO] conveyor output applied: reserved_out_1=ON (release granted)"],
        )

    def test_release_granted_logic_turns_on_conveyor_and_turns_off_buzzer(self) -> None:
        runtime = _FakeRuntime()
        controller = _FakeController()
        runtime._io_controller = controller

        applied = _apply_io_logic_event(runtime, "release_granted", controller=controller)

        self.assertTrue(applied)
        self.assertEqual(
            controller.calls,
            [
                ("reserved_out_1", True),
                ("reserved_out_2", False),
                ("button_green", True),
                ("button_red", False),
                ("button_blue", False),
            ],
        )
        self.assertEqual(
            runtime.logAppended.messages,
            [
                "[IO] conveyor output applied: reserved_out_1=ON (release granted)",
                "[IO] buzzer output applied: reserved_out_2=OFF (release granted)",
                "[IO] io logic output applied: button_green=ON (release granted)",
                "[IO] io logic output applied: button_red=OFF (release granted)",
                "[IO] io logic output applied: button_blue=OFF (release granted)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
