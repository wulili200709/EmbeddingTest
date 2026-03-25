from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from PySide6 import QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ui.runtime.runtime_mode_pyside6 import RuntimeModePage


class RuntimeModeTriggerButtonsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_trigger_buttons_follow_enabled_items_per_camera(self) -> None:
        page = RuntimeModePage()
        page.set_active_camera_roles(["cam1", "cam2"])
        page.set_inspection_items(
            [
                {
                    "item_id": "roi1",
                    "display_name": "ROI1",
                    "camera_id": "cam1",
                    "enabled": True,
                    "status_kind": "pending",
                    "status_text": "PENDING",
                },
                {
                    "item_id": "roi2",
                    "display_name": "ROI2",
                    "camera_id": "cam2",
                    "enabled": False,
                    "status_kind": "disabled",
                    "status_text": "DISABLED",
                },
            ]
        )

        self.assertTrue(page.btn_trigger_cam1.isEnabled())
        self.assertFalse(page.btn_trigger_cam2.isEnabled())


if __name__ == "__main__":
    unittest.main()
