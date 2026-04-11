from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from devices.light_controller import LightController


class _FakeIo:
    def __init__(self) -> None:
        self.outputs = {
            "light_cam1": False,
            "light_cam2": False,
        }

    def set_camera_light(self, camera_index: int, on: bool) -> None:
        self.outputs[f"light_cam{int(camera_index)}"] = bool(on)

    def read_output(self, name: str) -> bool:
        return bool(self.outputs.get(str(name), False))


class LightControllerModeTest(unittest.TestCase):
    def test_camera_line1_strobe_mode_skips_board_do_output(self) -> None:
        io = _FakeIo()
        controller = LightController(io)
        controller.set_camera_light_mode(1, "camera_line1_strobe")

        controller.prepare_capture(1)
        self.assertFalse(io.outputs["light_cam1"])
        self.assertFalse(io.outputs["light_cam2"])
        self.assertFalse(controller.requires_stable_delay(1))
        self.assertTrue(controller.requires_stable_delay(2))

        controller.finish_capture(1)
        self.assertFalse(io.outputs["light_cam1"])


if __name__ == "__main__":
    unittest.main()
