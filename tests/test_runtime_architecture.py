from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from application.runtime import conveyor
from application.runtime.controller import RuntimeController
from application.runtime.operation_mixins import RuntimeConveyorMixin


class _FakeConveyorService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def start(self):
        self.calls.append(("start",))
        return True

    def handle_di_event(
        self,
        name: str,
        state: bool,
        sample_wall_s: float = 0.0,
        sample_monotonic_s: float = 0.0,
        raw_word: int = -1,
    ) -> None:
        self.calls.append(
            ("di", name, state, sample_wall_s, sample_monotonic_s, raw_word)
        )


class _ConveyorMixinHarness(RuntimeConveyorMixin):
    def __init__(self) -> None:
        self._conveyor_service = _FakeConveyorService()


class RuntimeArchitectureTests(unittest.TestCase):
    def test_status_behavior_is_explicitly_composed_not_monkey_patched(self) -> None:
        self.assertIn("_update_status", RuntimeController.__dict__)
        self.assertIn("_emit_runtime_context", RuntimeController.__dict__)

    def test_runtime_operations_are_available_without_dynamic_binding(self) -> None:
        self.assertTrue(hasattr(RuntimeController, "start_conveyor"))
        self.assertTrue(hasattr(RuntimeController, "_run_single_multi_light_trigger"))
        self.assertTrue(hasattr(RuntimeController, "_close_io_controller"))

    def test_conveyor_api_delegates_to_composed_service(self) -> None:
        runtime = _ConveyorMixinHarness()
        self.assertTrue(runtime.start_conveyor())
        runtime._handle_conveyor_di_event("camera_trigger_sensor", True)
        self.assertEqual(
            runtime._conveyor_service.calls,
            [("start",), ("di", "camera_trigger_sensor", True, 0.0, 0.0, -1)],
        )

    def test_di_diagnostic_log_keeps_physical_channel_and_raw_word(self) -> None:
        logs: list[str] = []
        handled: list[tuple[str, bool]] = []

        class _Signal:
            def emit(self, message: str) -> None:
                logs.append(message)

        class _Controller:
            def handle_input_change(self, name: str, state: bool) -> None:
                handled.append((name, state))

        mapping = SimpleNamespace(
            get_input=lambda _name: SimpleNamespace(channel=6, active_high=True)
        )
        runtime = SimpleNamespace(
            _conveyor_controller=_Controller(),
            _io_controller=SimpleNamespace(mapping=mapping),
            logAppended=_Signal(),
        )

        conveyor._handle_conveyor_di_event(
            runtime,
            "good_outlet_sensor",
            True,
            123.456,
            time.perf_counter(),
            0x0041,
        )

        self.assertEqual(handled, [("good_outlet_sensor", True)])
        self.assertTrue(any("physical=DI6" in entry for entry in logs))
        self.assertTrue(any("raw_word=0x0041" in entry for entry in logs))
        self.assertTrue(any("logical=good_outlet_sensor" in entry for entry in logs))


if __name__ == "__main__":
    unittest.main()
