from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from infrastructure.camera_settings_store import (
    CameraSettingsStore,
    LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE,
    hik_settings_kwargs_from_mapping,
    light_source_mode_from_mapping,
)


class CameraSettingsStoreTest(unittest.TestCase):
    def test_store_and_hik_kwargs_support_digital_shift_without_leaking_light_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CameraSettingsStore(Path(tmp_dir) / "camera_settings.json")
            store.save_for_role(
                "cam1",
                "ABC123",
                {
                    "exposure_time_us": 1234.5,
                    "gain": 6.7,
                    "trigger_mode": "software",
                    "digital_shift_enable": True,
                    "digital_shift": 5.9994,
                    "light_source_mode": LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE,
                },
            )

            payload = store.load_for_role("cam1", serial="ABC123")
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertTrue(payload["digital_shift_enable"])
            self.assertAlmostEqual(float(payload["digital_shift"]), 5.9994, places=4)
            self.assertEqual(
                light_source_mode_from_mapping(payload),
                LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE,
            )

            hik_kwargs = hik_settings_kwargs_from_mapping(payload)
            self.assertEqual(hik_kwargs["trigger_mode"], "software")
            self.assertTrue(hik_kwargs["digital_shift_enable"])
            self.assertAlmostEqual(float(hik_kwargs["digital_shift"]), 5.9994, places=4)
            self.assertNotIn("light_source_mode", hik_kwargs)


if __name__ == "__main__":
    unittest.main()
