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

from ui.debug.tool_page import test_runner


class _PopulateResultsHarness:
    def __init__(self) -> None:
        self._current_result_rows = []
        self.table = QtWidgets.QTableWidget(0, 11)
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setValue(0.02)


class ToolPageResultExplanationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_populate_results_table_sets_anomaly_explanation_tooltip(self) -> None:
        harness = _PopulateResultsHarness()

        test_runner._populate_results_table(
            harness,
            [
                {
                    "file_name": "demo.png",
                    "pred": "OK",
                    "diff": 0.0123,
                    "value": 0.0877,
                    "threshold": 0.1000,
                }
            ],
        )

        self.assertEqual(harness.table.horizontalHeaderItem(6).text(), "score/value")
        item = harness.table.item(0, 2)
        self.assertIsNotNone(item)
        self.assertIn("anomaly score=0.0877 <= threshold=0.1000", item.toolTip())
        self.assertIn("threshold-score=0.0123", item.toolTip())


if __name__ == "__main__":
    unittest.main()
