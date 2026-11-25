from __future__ import annotations

import copy
from pathlib import Path
import string
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from algorithms.image_io import imwrite
from shape.core.localization_groups import LocalizationGroupError, clear_reference_cache, localize_groups
from ui.i18n import LANG_EN, LANG_ZH, _LOCALIZATION_TRANSLATIONS, language_code, set_language, tr
from ui.shape_template.localization_text import group_error_message, group_reason


class LocalizationDiagnosticI18nTests(unittest.TestCase):
    def setUp(self):
        current = language_code()
        self.addCleanup(set_language, current, persist=False)

    def test_real_failed_result_changes_language_without_changing_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "reference.png")
            reference = np.random.default_rng(83).integers(0, 255, (120, 160), np.uint8)
            self.assertTrue(imwrite(path, reference))
            clear_reference_cache()
            group = {"id": "anchor", "name": "Housing", "points": [[30, 30], [90, 80]],
                     "search_radius": 10, "min_score": 0.65}
            records = localize_groups(np.zeros_like(reference), path, [group], ["anchor"], np.eye(3)[:2])
        cached = copy.deepcopy(records)
        self.assertFalse(records[0]["valid"])
        self.assertEqual(records[0]["reason_code"], "low_score")
        self.assertEqual(records[0]["reason_params"], {"score": 0.0, "minimum": 0.65})
        error = LocalizationGroupError(records)
        for locale, expected in [(LANG_ZH, "匹配分数不足"), (LANG_EN, "Correlation is too low")]:
            set_language(locale, persist=False)
            self.assertIn(expected, group_reason(records[0]))
            self.assertIn(expected, group_error_message(error.group_results))
            self.assertIn("0.000 < 0.650", group_reason(records[0]))
            self.assertFalse(records[0]["valid"])
        self.assertEqual(records, cached)

    def test_cached_legacy_boundary_reason_and_unused_are_translated(self):
        boundary = {"valid": False, "reason": "最佳位置贴近有效图像边界，无法可靠细化"}
        set_language(LANG_EN, persist=False)
        self.assertIn("valid image boundary", group_reason(boundary))
        self.assertEqual(group_reason({"used": False, "valid": True, "reason_code": "unused"}), "No ROI assigned")
        invalid_config = {"id": "__configuration__", "name": "定位组配置", "valid": False,
                          "reason_code": "configuration_invalid", "reason_params": {}}
        self.assertIn("Group configuration", group_error_message([invalid_config]))
        self.assertNotIn("定位组", group_error_message([invalid_config]))

    def test_translation_placeholders_match_and_chinese_buttons_are_localized(self):
        formatter = string.Formatter()
        for key, (zh, en) in _LOCALIZATION_TRANSLATIONS.items():
            fields = lambda value: {part[1] for part in formatter.parse(value) if part[1] is not None}
            self.assertEqual(fields(zh), fields(en), key)
        set_language(LANG_ZH, persist=False)
        self.assertEqual(tr("template.tab.create"), "创建模板")
        self.assertEqual(tr("template.run_selected"), "运行所选")
        self.assertEqual(tr("template.add_images"), "添加图片")

    def test_core_diagnostics_import_without_ui_or_qt(self):
        code = ("import sys; import shape.core.localization_groups; "
                "assert 'ui.i18n' not in sys.modules; "
                "assert not any(x.startswith(('PySide', 'PyQt')) for x in sys.modules)")
        completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
