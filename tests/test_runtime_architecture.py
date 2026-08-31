from __future__ import annotations

import unittest

from application.runtime.controller import RuntimeController
from application.runtime.operation_mixins import RuntimeConveyorMixin


class _FakeConveyorService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def start(self):
        self.calls.append(("start",))
        return True

    def handle_di_event(self, name: str, state: bool) -> None:
        self.calls.append(("di", name, state))


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
            [("start",), ("di", "camera_trigger_sensor", True)],
        )


if __name__ == "__main__":
    unittest.main()
