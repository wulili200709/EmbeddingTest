from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


camera_stub = types.ModuleType("services.camera")
camera_stub.FrameGrabService = object
camera_stub.HikCameraDevice = object
camera_stub.HikCameraError = RuntimeError
camera_stub.HikCameraInfo = object
camera_stub.HikCameraManager = object
camera_stub.HikCameraSettings = object
camera_stub.HikFrame = object
camera_stub.frame_to_bgr_image = lambda frame: frame
camera_stub.frame_to_rgb_image = lambda frame: frame
sys.modules.setdefault("services.camera", camera_stub)

from services.inspection_runtime import CameraInspectionOutcome, InspectionRuntime


class _FakeScheduler:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_capture_started(self, camera_index: int) -> None:
        self.events.append(f"capture_started:{camera_index}")

    def on_inspecting_started(self) -> None:
        self.events.append("inspecting_started")


class _FakeFrameGrabService:
    def __init__(self, cam1_inspect_started: threading.Event, events: list[str]) -> None:
        self._cam1_inspect_started = cam1_inspect_started
        self._events = events

    def roles(self) -> list[str]:
        return ["cam2", "cam1"]

    def capture_once(self, role: str, *, timeout_ms: int = 1000):
        if role == "cam2":
            assert self._cam1_inspect_started.wait(0.2), "cam1 检测未在线程中先启动"
        self._events.append(f"capture:{role}")
        return {"role": role, "timeout_ms": timeout_ms}


class _FakeLightController:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def prepare_capture(self, camera_index: int) -> None:
        self._events.append(f"light_on:{camera_index}")

    def finish_capture(self, camera_index: int) -> None:
        self._events.append(f"light_off:{camera_index}")


class _FakeCameraStrobeLightController(_FakeLightController):
    def requires_stable_delay(self, camera_index: int) -> bool:
        return int(camera_index) != 1


class InspectionRuntimePipelineTest(unittest.TestCase):
    def test_capture_pipeline_orders_roles_and_submits_cam1_inspection_before_cam2_capture(self) -> None:
        events: list[str] = []
        cam1_inspect_started = threading.Event()
        scheduler = _FakeScheduler()

        def inspect_callback(role: str, frame) -> CameraInspectionOutcome:
            events.append(f"inspect:{role}")
            if role == "cam1":
                cam1_inspect_started.set()
                time.sleep(0.05)
            return CameraInspectionOutcome(role=role, result="OK")

        runtime = InspectionRuntime(
            scheduler=scheduler,
            permission_manager=object(),
            frame_grab_service=_FakeFrameGrabService(cam1_inspect_started, events),
            light_controller=_FakeLightController(events),
            tower_light_controller=object(),
            inspect_callback=inspect_callback,
            role_to_camera_index={"cam1": 1, "cam2": 2},
        )

        outcomes = runtime._capture_and_inspect_pipeline(timeout_ms=1000)

        self.assertEqual(list(outcomes.keys()), ["cam1", "cam2"])
        self.assertEqual(
            [events[i] for i in range(6)],
            [
                "light_on:1",
                "capture:cam1",
                "light_off:1",
                "inspect:cam1",
                "light_on:2",
                "capture:cam2",
            ],
        )
        self.assertEqual(
            scheduler.events,
            [
                "capture_started:1",
                "capture_started:2",
                "inspecting_started",
            ],
        )

    def test_camera_line1_strobe_mode_skips_light_stable_sleep(self) -> None:
        events: list[str] = []
        scheduler = _FakeScheduler()

        class _SingleRoleFrameGrabService:
            def roles(self) -> list[str]:
                return ["cam1"]

            def capture_once(self, role: str, *, timeout_ms: int = 1000):
                events.append(f"capture:{role}")
                return {"role": role, "timeout_ms": timeout_ms}

        runtime = InspectionRuntime(
            scheduler=scheduler,
            permission_manager=object(),
            frame_grab_service=_SingleRoleFrameGrabService(),
            light_controller=_FakeCameraStrobeLightController(events),
            tower_light_controller=object(),
            inspect_callback=lambda role, frame: CameraInspectionOutcome(role=role, result="OK"),
            role_to_camera_index={"cam1": 1},
            light_stable_ms=80,
        )

        with mock.patch("services.inspection_runtime.time.sleep") as sleep_mock:
            runtime._capture_and_inspect_for_roles(["cam1"], timeout_ms=1000)

        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
