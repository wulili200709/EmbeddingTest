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

        page.set_conveyor_state({**base, "state": "READY_STOPPED"})
        self.assertEqual(page.btn_conveyor_purge.text(), "一键清线")
        page.btn_conveyor_purge.click()
        self.assertEqual(purge_events, ["start"])
        self.assertEqual(
            [label.property("fifoStatus") for label in page._fifo_indicator_labels],
            ["PENDING", "GOOD", "NG"],
        )

        page.set_conveyor_state({**base, "state": "PURGE_PREPARING"})
        self.assertEqual(page.btn_conveyor_purge.text(), "清线中…")
        self.assertFalse(page.btn_conveyor_purge.isEnabled())

        page.set_conveyor_state({**base, "state": "PURGE_RUNNING"})
        self.assertEqual(page.btn_conveyor_purge.text(), "清线中…")
        self.assertFalse(page.btn_conveyor_purge.isEnabled())

        page.set_conveyor_state({**base, "state": "PURGE_PAUSED"})
        self.assertEqual(page.btn_conveyor_purge.text(), "继续清线")
        page.btn_conveyor_purge.click()
        self.assertEqual(purge_events, ["start", "continue"])

        page.set_conveyor_state(
            {
                **base,
                "state": "FAULT_STOPPED",
                "fault_code": "JAM_DETECTED",
                "fault_recovery": "ACKNOWLEDGE",
            }
        )
        self.assertTrue(page.btn_conveyor_ack.isEnabled())
        page.set_conveyor_state(
            {
                **base,
                "state": "FAULT_STOPPED",
                "safety_ok": False,
                "fault_recovery": "ACKNOWLEDGE",
            }
        )
        self.assertFalse(page.btn_conveyor_ack.isEnabled())

    def test_conveyor_faults_are_shown_as_operator_friendly_chinese(self) -> None:
        page = RuntimeModePage()
        base = {
            "state": "FAULT_STOPPED",
            "io_ready": True,
            "safety_ok": True,
            "door_closed": True,
            "run_permitted": False,
            "fifo_count": 0,
            "fifo": [],
            "fault_code": "JAM_DETECTED",
        }
        jam_cases = (
            ("end_test_sensor remained active for 3.0 s", "DI6末端检测位置堵料"),
            ("good_outlet_sensor remained active for 3.0 s", "DI7 GOOD出口堵料"),
            ("waste_outlet_sensor remained active for 3.0 s", "DI8废料出口堵料"),
        )
        for detail, expected in jam_cases:
            page.set_conveyor_state({**base, "fault_detail": detail})
            self.assertIn("皮带: 故障停机", page.lbl_conveyor_state.text())
            self.assertIn(expected, page.lbl_conveyor_state.text())
            self.assertIn("JAM_DETECTED", page.lbl_conveyor_state.toolTip())
            self.assertIn(detail, page.lbl_conveyor_state.toolTip())

        page.set_conveyor_state(
            {
                **base,
                "fault_code": "ITEM_ARRIVAL_TIMEOUT",
                "fault_detail": "item 1 did not reach DI1 in time",
            }
        )
        self.assertIn("物料未在规定时间到达DI1", page.lbl_conveyor_state.text())

    def test_camera_trigger_buttons_follow_physical_connections(self) -> None:
        page = RuntimeModePage()
        self.assertTrue(page.btn_trigger_cam1.isHidden())
        self.assertTrue(page.btn_trigger_cam2.isHidden())
        self.assertTrue(page.btn_trigger_cam3.isHidden())

        page.set_physical_camera_roles(["cam1"])
        self.assertFalse(page.btn_trigger_cam1.isHidden())
        self.assertTrue(page.btn_trigger_cam2.isHidden())
        self.assertTrue(page.btn_trigger_cam3.isHidden())

        page.set_physical_camera_roles(["cam1", "cam3"])
        self.assertFalse(page.btn_trigger_cam1.isHidden())
        self.assertTrue(page.btn_trigger_cam2.isHidden())
        self.assertFalse(page.btn_trigger_cam3.isHidden())

        page.set_physical_camera_roles([])
        self.assertTrue(page.btn_trigger_cam1.isHidden())
        self.assertTrue(page.btn_trigger_cam2.isHidden())
        self.assertTrue(page.btn_trigger_cam3.isHidden())


if __name__ == "__main__":
    unittest.main()
