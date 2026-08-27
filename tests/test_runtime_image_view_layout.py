from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets

from ui.runtime.runtime_mode_pyside6 import RuntimeImageView, RuntimeModePage


class RuntimeImageViewLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_large_camera_frame_does_not_control_page_layout_size(self) -> None:
        view = RuntimeImageView("Cam1")
        self.assertEqual(
            view.sizePolicy().horizontalPolicy(),
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.assertEqual(
            view.sizePolicy().verticalPolicy(),
            QtWidgets.QSizePolicy.Policy.Ignored,
        )

        view.resize(640, 360)
        view.set_runtime_pixmap(QtGui.QPixmap(2448, 2048))
        self.assertEqual(view.minimumWidth(), 160)
        self.assertEqual(view.minimumHeight(), 120)
        self.assertEqual(view.pixmap().size(), view.size())

    def test_conveyor_controls_and_fifo_lights_follow_runtime_state(self) -> None:
        page = RuntimeModePage()
        self.assertTrue(page.btn_conveyor_start.isHidden())
        self.assertTrue(page.btn_conveyor_stop.isHidden())
        self.assertTrue(page.btn_conveyor_continue.isHidden())

        purge_events: list[str] = []
        page.conveyorPurgeRequested.connect(lambda: purge_events.append("start"))
        page.conveyorPurgeContinueRequested.connect(lambda: purge_events.append("continue"))
        base = {
            "io_ready": True,
            "safety_ok": True,
            "door_closed": True,
            "run_permitted": True,
            "fault_code": "",
            "fifo_count": 3,
            "fifo": [
                {"sequence_id": 1, "inspection_status": "PENDING"},
                {"sequence_id": 2, "inspection_status": "GOOD"},
                {"sequence_id": 3, "inspection_status": "NG"},
            ],
        }

        page.set_conveyor_state({**base, "state": "STOPPED"})
        self.assertEqual(page.btn_conveyor_purge.text(), "一键清线")
        page.btn_conveyor_purge.click()
        self.assertEqual(purge_events, ["start"])
        self.assertEqual(
            [label.property("fifoStatus") for label in page._fifo_indicator_labels],
            ["PENDING", "GOOD", "NG"],
        )

        page.set_conveyor_state({**base, "state": "PURGING"})
        self.assertEqual(page.btn_conveyor_purge.text(), "清线中…")
        self.assertFalse(page.btn_conveyor_purge.isEnabled())

        page.set_conveyor_state({**base, "state": "PURGE_PAUSED"})
        self.assertEqual(page.btn_conveyor_purge.text(), "继续清线")
        page.btn_conveyor_purge.click()
        self.assertEqual(purge_events, ["start", "continue"])

        page.set_conveyor_state({**base, "state": "FAULT", "fault_code": "JAM_DETECTED"})
        self.assertTrue(page.btn_conveyor_ack.isEnabled())
        page.set_conveyor_state({**base, "state": "FAULT", "safety_ok": False})
        self.assertFalse(page.btn_conveyor_ack.isEnabled())


if __name__ == "__main__":
    unittest.main()
