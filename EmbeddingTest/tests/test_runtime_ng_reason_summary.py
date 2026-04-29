from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from domain.inspection_models import InspectionItemResult, RuntimeInspectionResult
from ui.runtime.runtime_mode_pyside6 import _result_reason_display_text


class RuntimeNgReasonSummaryTest(unittest.TestCase):
    def test_item_ng_summary_uses_display_name(self) -> None:
        result = RuntimeInspectionResult(
            task_id="demo",
            product_name="1841678",
            final_result="NG",
            item_results=[
                InspectionItemResult(
                    item_id="roi33",
                    display_name="多料",
                    camera_id="cam1",
                    roi_label="roi33",
                    result="NG",
                )
            ],
        )

        self.assertEqual(result.failure_summary_text(), "多料NG")
        self.assertTrue(result.summary_text().startswith("多料NG"))

    def test_template_match_failure_is_localized(self) -> None:
        result = RuntimeInspectionResult(
            task_id="demo",
            product_name="1841678",
            final_result="NG",
            error_message="match failure: expected 1 instances, got 0",
        )

        self.assertEqual(result.failure_summary_text(), "模板匹配失败")

    def test_result_reason_display_picks_first_human_reason(self) -> None:
        self.assertEqual(
            _result_reason_display_text("NG", "多料NG；cam1=NG；match 10.0 ms"),
            "多料NG",
        )
        self.assertEqual(
            _result_reason_display_text("NG", "NCC did not find any match."),
            "模板匹配失败",
        )


if __name__ == "__main__":
    unittest.main()
