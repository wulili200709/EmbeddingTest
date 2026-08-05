from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from PySide6 import QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ui.runtime.runtime_mode_pyside6 import RuntimeModePage, _NG_RED
from ui.i18n import language_code, set_language, tr
from ui.window_common import update_runtime_preview
from application.runtime.preview_frame import build_runtime_preview_frame


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
