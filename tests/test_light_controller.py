from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from devices.light_controller import LightController
from devices.tower_light_controller import TowerLightController


class _FakeIo:
    def __init__(self) -> None:
        self.outputs = {
            "light_cam1": False,
            "light_cam2": False,
        }
        self.events: list[tuple[str, object]] = []

    def set_camera_light(self, camera_index: int, on: bool) -> None:
        self.outputs[f"light_cam{int(camera_index)}"] = bool(on)

    def read_output(self, name: str) -> bool:
        return bool(self.outputs.get(str(name), False))

    def set_buzzer(self, on: bool) -> None:
        self.outputs["buzzer"] = bool(on)
        self.events.append(("buzzer", bool(on)))

    def set_tower_light(self, *, red=None, green=None, blue=None) -> None:
        updates = {}
        if red is not None:
            updates["tower_red"] = bool(red)
        if green is not None:
            updates["tower_green"] = bool(green)
        if blue is not None:
            updates["tower_blue"] = bool(blue)
        self.outputs.update(updates)
        self.events.append(("tower", updates))

    def tower_all_off(self) -> None:
        self.set_tower_light(red=False, green=False, blue=False)


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

    def test_tower_light_ng_starts_red_signal_and_buzzer_together(self) -> None:
        io = _FakeIo()
        controller = TowerLightController(
            io,
            ok_flash_ms=10,
            ng_flash_ms=1000,
            ng_buzzer_ms=250,
            idle_blue_delay_s=0,
        )

        def fake_sleep(seconds: float) -> None:
            io.events.append(("sleep", round(float(seconds), 3)))

        with patch("devices.tower_light_controller.time.sleep", side_effect=fake_sleep):
            controller.show_ng()

        self.assertEqual(
            io.events[:5],
            [
                ("tower", {"tower_red": False, "tower_green": False, "tower_blue": False}),
                ("tower", {"tower_red": True}),
                ("buzzer", True),
                ("sleep", 0.25),
                ("buzzer", False),
            ],
        )
        controller.close()


if __name__ == "__main__":
    unittest.main()
