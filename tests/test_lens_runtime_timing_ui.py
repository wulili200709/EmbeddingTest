from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from PySide6 import QtWidgets
from ui.i18n import LANG_EN, LANG_ZH, language_code, set_language
from ui.runtime.runtime_mode_pyside6 import RuntimeModePage


class LensRuntimeTimingUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_correction_is_visible_in_summary_tooltip_and_follows_language(self):
        previous = language_code()
        page = RuntimeModePage()
        try:
            page.set_timing_breakdown({
                "duration_ms": 40, "correction_ms": 7.5,
                "cam1_capture_ms": 10, "cam1_correction_ms": 7.5,
                "cam1_match_ms": 4, "cam1_infer_ms": 6, "cam1_total_ms": 27.5,
            })
            for language, title in ((LANG_ZH, "镜头矫正"), (LANG_EN, "Lens correction")):
                set_language(language, persist=False)
                page.retranslate_ui()
                self.assertIn(title, page.lbl_ng_timing_summary.toolTip())
                self.assertIn("7.5", page.lbl_ng_timing_summary.toolTip())
                self.assertIn("27.5", page.lbl_ng_timing_summary.toolTip())
                self.assertIn(title, page.lbl_footer_time.toolTip())
                self.assertIn("40.0", page.lbl_footer_time.text())
        finally:
            page.close()
            page.deleteLater()
            set_language(previous, persist=False)

    def test_disabled_correction_keeps_previous_summary_text(self):
        previous = language_code()
        page = RuntimeModePage()
        try:
            set_language(LANG_EN, persist=False)
            page.set_timing_breakdown({"cam1_capture_ms": 2, "cam1_match_ms": 3,
                                       "cam1_infer_ms": 4, "cam1_total_ms": 9})
            self.assertNotIn("Lens correction", page.lbl_ng_timing_summary.toolTip())
            self.assertEqual(page.lbl_footer_time.toolTip(), "")
        finally:
            page.close()
            page.deleteLater()
            set_language(previous, persist=False)


if __name__ == "__main__":
    unittest.main()
