from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6 import QtCore, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ui.runtime.runtime_mode_pyside6 import RuntimeModePage, _NG_RED
from ui.debug.tool_page import camera_debug
from ui.i18n import language_code, set_language, tr
from ui.window_common import update_runtime_preview
from application.runtime.preview_frame import build_runtime_preview_frame


class _CaptureStore:
    def __init__(self) -> None:
        self.saved: list[tuple[object, object]] = []

    def save_capture_config(self, mode, channels) -> None:
        self.saved.append((mode, channels))


class _CaptureHarness(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._capture_config_loading = False
        self._camera_settings_store = _CaptureStore()
        self.inspection_items = []
        self.lbl_capture_channel_count = QtWidgets.QLabel()
        self.lbl_capture_mode_help = QtWidgets.QLabel()
        self.capture_channel_frame = QtWidgets.QFrame()
        self.cmb_capture_mode = QtWidgets.QComboBox()
        self.cmb_capture_mode.addItem("Independent", "independent")
        self.cmb_capture_mode.addItem("Flexible", "flexible")
        self.cmb_capture_mode.setCurrentIndex(1)
        self.capture_channel_table = QtWidgets.QTableWidget(3, 6)
        for row, role in enumerate(("cam1", "cam2", "cam3")):
            enabled = QtWidgets.QTableWidgetItem()
            enabled.setCheckState(QtCore.Qt.CheckState.Checked)
            self.capture_channel_table.setItem(row, 0, enabled)
            self.capture_channel_table.setItem(row, 1, QtWidgets.QTableWidgetItem(role))


class RuntimeModeTriggerButtonsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_enabled_channel_badge_uses_checked_rows(self) -> None:
        harness = _CaptureHarness()
        harness.capture_channel_table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)

        camera_debug._update_capture_channel_count(harness)

        self.assertEqual(
            harness.lbl_capture_channel_count.text(),
            tr("debug.capture_channel_count", count=2),
        )

    def test_cancel_disable_restores_channel_and_does_not_save(self) -> None:
        harness = _CaptureHarness()
        harness.inspection_items = [
            SimpleNamespace(
                item_id="roi",
                display_name="CAM3 ROI",
                roi_label="roi",
                camera_id="cam3",
                enabled=False,
            )
        ]
        item = harness.capture_channel_table.item(2, 0)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        original_question = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.No
        )
        try:
            camera_debug._on_capture_channel_item_changed(harness, item)
        finally:
            QtWidgets.QMessageBox.question = original_question

        self.assertEqual(item.checkState(), QtCore.Qt.CheckState.Checked)
        self.assertEqual(harness._camera_settings_store.saved, [])
        self.assertEqual(
            harness.lbl_capture_channel_count.text(),
            tr("debug.capture_channel_count", count=3),
        )

    def test_confirm_disable_preserves_inspection_item_and_saves_two_channels(self) -> None:
        harness = _CaptureHarness()
        inspection_item = SimpleNamespace(
            item_id="roi",
            display_name="CAM3 ROI",
            roi_label="roi",
            camera_id="cam3",
            enabled=True,
        )
        harness.inspection_items = [inspection_item]
        item = harness.capture_channel_table.item(2, 0)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        original_question = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes
        )
        try:
            camera_debug._on_capture_channel_item_changed(harness, item)
        finally:
            QtWidgets.QMessageBox.question = original_question

        self.assertIs(harness.inspection_items[0], inspection_item)
        self.assertEqual(item.checkState(), QtCore.Qt.CheckState.Unchecked)
        self.assertEqual(len(harness._camera_settings_store.saved), 1)
        _mode, channels = harness._camera_settings_store.saved[0]
        self.assertEqual(
            [channel["role"] for channel in channels if channel["enabled"]],
            ["cam1", "cam2"],
        )
        self.assertEqual(
            harness.lbl_capture_channel_count.text(),
            tr("debug.capture_channel_count", count=2),
        )

    def test_switch_to_mapped_mode_warns_about_previously_unchecked_channel(self) -> None:
        harness = _CaptureHarness()
        harness.inspection_items = [
            SimpleNamespace(
                item_id="roi",
                display_name="CAM3 ROI",
                roi_label="roi",
                camera_id="cam3",
                enabled=True,
            )
        ]
        harness.capture_channel_table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        original_question = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.No
        )
        try:
            camera_debug._on_capture_mode_changed(harness)
        finally:
            QtWidgets.QMessageBox.question = original_question

        self.assertEqual(
            harness.capture_channel_table.item(2, 0).checkState(),
            QtCore.Qt.CheckState.Checked,
        )

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

    def test_top_timing_labels_hidden_by_default(self) -> None:
        page = RuntimeModePage()

        self.assertFalse(page.lbl_capture_time.isVisible())
        self.assertFalse(page.lbl_match_time.isVisible())
        self.assertFalse(page.lbl_infer_time.isVisible())
        self.assertFalse(page.lbl_duration.isVisible())

    def test_runtime_preview_refresh_keeps_in_memory_frame_when_source_path_is_missing(self) -> None:
        page = RuntimeModePage()
        preview = build_runtime_preview_frame(
            role="cam1",
            image_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
            source_path=str(ROOT / "missing_runtime_capture.png"),
            camera_role="cam1",
        )

        update_runtime_preview(page, "cam1", preview)
        page.set_inspection_items(
            [
                {
                    "item_id": "roi1",
                    "display_name": "ROI1",
                    "camera_id": "cam1",
                    "enabled": True,
                    "status_kind": "ok",
                    "status_text": "OK",
                }
            ]
        )

        self.assertIsNotNone(page.view_cam1.pixmap())
        self.assertFalse(page.view_cam1.pixmap().isNull())
        self.assertEqual(page.view_cam1.text(), "")

    def test_legacy_source_path_aliases_to_preview_source_cache(self) -> None:
        page = RuntimeModePage()

        page.set_camera_source_path("cam1", "demo.png")

        self.assertEqual(page._camera_preview_sources["cam1"], "demo.png")

    def test_single_camera_configuration_hides_cam2_view(self) -> None:
        page = RuntimeModePage()

        page.set_configured_camera_roles(["cam1"])
        page.clear_camera_views()

        self.assertTrue(page.view_cam2.isHidden())
        self.assertFalse(page.btn_trigger_cam2.isEnabled())

    def test_cam2_cam3_product_hides_cam1_and_enables_order_trigger(self) -> None:
        page = RuntimeModePage()
        page.set_configured_camera_roles(["cam2", "cam3"])
        page.set_active_camera_roles(["cam1", "cam2", "cam3"])
        page.set_inspection_items(
            [
                {
                    "item_id": "cam2-roi",
                    "display_name": "CAM2 ROI",
                    "camera_id": "cam2",
                    "enabled": True,
                    "status_kind": "pending",
                    "status_text": "PENDING",
                },
                {
                    "item_id": "cam3-roi",
                    "display_name": "CAM3 ROI",
                    "camera_id": "cam3",
                    "enabled": True,
                    "status_kind": "pending",
                    "status_text": "PENDING",
                },
            ]
        )

        self.assertTrue(page.view_cam1.isHidden())
        self.assertFalse(page.view_cam2.isHidden())
        self.assertFalse(page.view_cam3.isHidden())
        self.assertTrue(page.btn_simulate_foot.isEnabled())
        self.assertTrue(page.btn_trigger_cam2.isEnabled())
        self.assertTrue(page.btn_trigger_cam3.isEnabled())

    def test_runtime_item_list_hides_disabled_channel_rows_without_deleting_them(self) -> None:
        page = RuntimeModePage()
        rows = [
            {
                "item_id": "cam1-roi",
                "display_name": "CAM1 ROI",
                "camera_id": "cam1",
                "enabled": True,
                "status_kind": "pending",
                "status_text": "PENDING",
            },
            {
                "item_id": "cam3-roi",
                "display_name": "CAM3 ROI",
                "camera_id": "cam3",
                "enabled": True,
                "status_kind": "inactive",
                "status_text": "相机未接入",
            },
        ]

        page.set_configured_camera_roles(["cam1", "cam2"])
        page.set_inspection_items(rows)

        self.assertEqual(page._inspection_rows, rows)
        self.assertIn("cam1-roi", page._item_indicators_by_item_id)
        self.assertNotIn("cam3-roi", page._item_indicators_by_item_id)
        self.assertNotIn("cam3", page._camera_section_headers)

        page.set_configured_camera_roles(["cam1", "cam2", "cam3"])

        self.assertIn("cam3-roi", page._item_indicators_by_item_id)
        self.assertIn("cam3", page._camera_section_headers)

    def test_physical_camera_bindings_are_independent_from_product_roles(self) -> None:
        page = RuntimeModePage()
        page.set_configured_camera_roles(["cam2", "cam3"])
        page.set_camera_serial("cam1", "SERIAL-1")
        page.set_camera_serial("cam2", "SERIAL-2")
        page.set_camera_serial("cam3", "SERIAL-3")

        self.assertEqual(
            page.camera_bindings(),
            {
                "cam1": "SERIAL-1",
                "cam2": "SERIAL-2",
                "cam3": "SERIAL-3",
            },
        )

    def test_runtime_item_list_hides_find_line_helpers_when_line_distance_exists(self) -> None:
        page = RuntimeModePage()

        page.set_inspection_items(
            [
                {
                    "item_id": "left",
                    "display_name": "左",
                    "camera_id": "cam1",
                    "algorithm_code": "find_line",
                    "enabled": True,
                    "status_kind": "ok",
                    "status_text": "OK",
                },
                {
                    "item_id": "right",
                    "display_name": "右",
                    "camera_id": "cam1",
                    "algorithm_code": "find_line",
                    "enabled": True,
                    "status_kind": "ok",
                    "status_text": "OK",
                },
                {
                    "item_id": "line_distance",
                    "display_name": "Line Distance",
                    "camera_id": "cam1",
                    "algorithm_code": "line_distance",
                    "params": {
                        "line_a_item_id": "left",
                        "line_b_item_id": "right",
                    },
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                },
            ]
        )

        self.assertEqual(len(page._item_indicators), 1)
        self.assertIn("line_distance", page._item_indicators_by_item_id)
        self.assertNotIn("left", page._item_indicators_by_item_id)
        self.assertNotIn("right", page._item_indicators_by_item_id)

    def test_runtime_item_list_translates_default_line_distance_name(self) -> None:
        previous = language_code()
        try:
            set_language("zh_CN", persist=False)
            page = RuntimeModePage()

            page.set_inspection_items(
                [
                    {
                        "item_id": "line_distance",
                        "display_name": "Line Distance",
                        "camera_id": "cam1",
                        "algorithm_code": "line_distance",
                        "enabled": True,
                        "status_kind": "pending",
                        "status_text": "PENDING",
                    },
                ]
            )

            self.assertEqual(len(page._item_indicators), 1)
            self.assertEqual(page._item_indicators[0].lbl_name._full_text, tr("debug.algorithm.line_distance"))
        finally:
            set_language(previous, persist=False)

    def test_runtime_item_list_translates_legacy_name_for_reference_normal_distance(self) -> None:
        previous = language_code()
        try:
            set_language("zh_CN", persist=False)
            page = RuntimeModePage()

            page.set_inspection_items(
                [
                    {
                        "item_id": "line_distance",
                        "display_name": tr("debug.algorithm.line_distance"),
                        "camera_id": "cam1",
                        "algorithm_code": "line_distance_ref_normal",
                        "enabled": True,
                        "status_kind": "pending",
                        "status_text": "PENDING",
                    },
                ]
            )

            self.assertEqual(len(page._item_indicators), 1)
            self.assertEqual(page._item_indicators[0].lbl_name._full_text, tr("debug.algorithm.line_distance_ref_normal"))
        finally:
            set_language(previous, persist=False)

    def test_runtime_item_list_keeps_non_helper_items_with_line_distance(self) -> None:
        page = RuntimeModePage()

        page.set_inspection_items(
            [
                {
                    "item_id": "left",
                    "display_name": "左",
                    "camera_id": "cam1",
                    "algorithm_code": "find_line",
                    "enabled": True,
                    "status_kind": "ok",
                    "status_text": "OK",
                },
                {
                    "item_id": "right",
                    "display_name": "右",
                    "camera_id": "cam1",
                    "algorithm_code": "find_line",
                    "enabled": True,
                    "status_kind": "ok",
                    "status_text": "OK",
                },
                {
                    "item_id": "roi3",
                    "display_name": "roi3",
                    "camera_id": "cam1",
                    "algorithm_code": "shared_backbone_register",
                    "enabled": True,
                    "status_kind": "pending",
                    "status_text": "PENDING",
                },
                {
                    "item_id": "line_distance",
                    "display_name": "Line Distance",
                    "camera_id": "cam1",
                    "algorithm_code": "line_distance",
                    "params": {
                        "line_a_item_id": "left",
                        "line_b_item_id": "right",
                    },
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                },
            ]
        )

        self.assertEqual(len(page._item_indicators), 2)
        self.assertIn("roi3", page._item_indicators_by_item_id)
        self.assertIn("line_distance", page._item_indicators_by_item_id)
        self.assertNotIn("left", page._item_indicators_by_item_id)
        self.assertNotIn("right", page._item_indicators_by_item_id)

    def test_ng_summary_shows_first_ng_item_for_each_available_camera(self) -> None:
        previous = language_code()
        try:
            set_language("en_US", persist=False)
            page = RuntimeModePage()
            rows = [
                {
                    "item_id": "cam1-first",
                    "display_name": "Cam1 First",
                    "camera_id": "cam1",
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                },
                {
                    "item_id": "cam1-second",
                    "display_name": "Cam1 Second",
                    "camera_id": "cam1",
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                },
                {
                    "item_id": "cam2-first",
                    "display_name": "Cam2 First",
                    "camera_id": "cam2",
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                },
                {
                    "item_id": "cam3-first",
                    "display_name": "Cam3 First",
                    "camera_id": "cam3",
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                },
            ]

            page.set_inspection_items(rows)
            summary = page.lbl_ng_summary.text()
            self.assertIn("Cam1 Cam1 First NG", summary)
            self.assertNotIn("Cam1 Second", summary)
            self.assertIn("Cam2 Cam2 First NG", summary)
            self.assertIn("Cam3 Cam3 First NG", summary)
            self.assertFalse(page.lbl_ng_summary.wordWrap())
            self.assertNotIn("border", page.lbl_ng_summary.styleSheet())
            self.assertGreater(page.lbl_ng_summary.maximumWidth(), 720)
            self.assertTrue(page.ng_summary_bar.isVisibleTo(page))
            self.assertIn(_NG_RED, page.lbl_ng_summary.styleSheet())
            self.assertNotIn(_NG_RED, page.ng_summary_bar.styleSheet())

            page.set_inspection_items(rows[:1])
            self.assertEqual(page.lbl_ng_summary.text(), "Cam1 Cam1 First NG")

            page.set_inspection_items([rows[0], rows[2]])
            summary = page.lbl_ng_summary.text()
            self.assertIn("Cam1 Cam1 First NG", summary)
            self.assertIn("Cam2 Cam2 First NG", summary)
            self.assertNotIn("Cam3", summary)
        finally:
            set_language(previous, persist=False)

    def test_ng_summary_bar_contains_camera_timing_and_hides_right_timing_rows(self) -> None:
        page = RuntimeModePage()
        page.set_timing_breakdown(
            {
                "cam1_capture_ms": 88.2,
                "cam1_match_ms": 123.8,
                "cam1_infer_ms": 531.8,
                "cam1_total_ms": 743.8,
                "cam2_capture_ms": 89.6,
                "cam2_match_ms": 26.0,
                "cam2_infer_ms": 130.5,
                "cam2_total_ms": 246.1,
            }
        )
        page.set_inspection_items(
            [
                {
                    "item_id": "bu-p",
                    "display_name": "BU-P",
                    "camera_id": "cam1",
                    "enabled": True,
                    "status_kind": "ng",
                    "status_text": "NG",
                }
            ]
        )

        timing_text = page.lbl_ng_timing_summary.toolTip()
        self.assertIn("CAM1", timing_text)
        self.assertIn("88.2", timing_text)
        self.assertIn("743.8", timing_text)
        self.assertIn("CAM2", timing_text)
        self.assertNotIn("CAM3", timing_text)
        summary_layout = page.ng_summary_bar.layout()
        self.assertLess(
            summary_layout.indexOf(page.lbl_ng_timing_summary),
            summary_layout.indexOf(page.lbl_ng_summary),
        )
        self.assertIn(_NG_RED, page.lbl_ng_summary.styleSheet())
        self.assertTrue(page.ng_summary_bar.isVisibleTo(page))
        self.assertFalse(page.lbl_cam1_timing.isVisibleTo(page))
        self.assertFalse(page.lbl_cam2_timing.isVisibleTo(page))
        self.assertFalse(page.lbl_cam3_timing.isVisibleTo(page))

    def test_ng_summary_reports_template_match_failure(self) -> None:
        previous = language_code()
        try:
            set_language("zh_CN", persist=False)
            page = RuntimeModePage()

            for detail in ("cam3 match failure", "cam3 NCC did not find any match."):
                with self.subTest(detail=detail):
                    page.set_inspection_items(
                        [
                            {
                                "item_id": "cam3-roi",
                                "display_name": "ROI",
                                "camera_id": "cam3",
                                "enabled": True,
                                "status_kind": "ng",
                                "status_text": f"NG ({detail})",
                            }
                        ]
                    )
                    self.assertEqual(
                        page.lbl_ng_summary.text(),
                        tr("runtime.ng_summary_match_failed", camera="Cam3"),
                    )
        finally:
            set_language(previous, persist=False)


if __name__ == "__main__":
    unittest.main()
