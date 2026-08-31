from __future__ import annotations

import unittest

from application.runtime.capture_pipeline import capture_runtime_channel


class _LightController:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def set_camera_light_mode(self, index: int, mode: str) -> None:
        self._events.append(("mode", index, mode))

    def prepare_capture(self, index: int) -> None:
        self._events.append(("prepare", index))

    def requires_stable_delay(self, _index: int) -> bool:
        return False

    def finish_capture(self, index: int) -> None:
        self._events.append(("finish", index))


class _FrameGrabService:
    def __init__(self, events: list[object], *, fail: bool = False) -> None:
        self._events = events
        self._fail = fail

    def capture_once(self, role: str, *, timeout_ms: int) -> object:
        self._events.append(("capture", role, timeout_ms))
        if self._fail:
            raise RuntimeError("camera failed")
        return object()


class _Runtime:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[object] = []
        self._light_controller = _LightController(self.events)
        self._frame_grab_service = _FrameGrabService(self.events, fail=fail)


class CapturePipelineTests(unittest.TestCase):
    def test_shared_capture_pipeline_orders_settings_light_capture_and_cleanup(self) -> None:
        runtime = _Runtime()
        result = capture_runtime_channel(
            runtime,
            {"role": "cam2", "physical_role": "cam1", "light_index": 3},
            apply_camera_settings=lambda _channel: runtime.events.append("settings"),
            before_capture=lambda role, index: runtime.events.append(("before", role, index)),
            force_board_io_light=True,
        )

        self.assertEqual(result.role, "cam2")
        self.assertEqual(result.physical_role, "cam1")
        self.assertEqual(
            runtime.events,
            [
                "settings",
                ("mode", 3, "board_io"),
                ("prepare", 3),
                ("before", "cam2", 3),
                ("capture", "cam1", 1000),
                ("finish", 3),
            ],
        )

    def test_shared_capture_pipeline_turns_light_off_when_capture_fails(self) -> None:
        runtime = _Runtime(fail=True)

        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            capture_runtime_channel(
                runtime,
                {"role": "cam1", "physical_role": "cam1", "light_index": 1},
            )

        self.assertEqual(runtime.events[-1], ("finish", 1))


if __name__ == "__main__":
    unittest.main()
