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

from ui.debug.tool_page.page import ToolPage


class _ToolPageRuntimeWidgetsHarness:
    _update_runtime_widgets = ToolPage._update_runtime_widgets

    def __init__(self) -> None:
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["proto", "topk"])
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_topk = QtWidgets.QSpinBox()
        self.btn_train = QtWidgets.QPushButton()
        self.btn_test = QtWidgets.QPushButton()
        self.btn_validate_margin = QtWidgets.QPushButton()
        self.btn_embedding_analysis = QtWidgets.QPushButton()
        self.lbl_topk = QtWidgets.QLabel("TopK")
        self._algo_param_label_style = "color:#9a9a9a;"
        self._algo_param_label_disabled_style = "color:#6a6a6a;"

    def current_algorithm(self) -> str:
        return "efficientnet_b0"

    def _is_embedding_algorithm(self, algorithm: str | None = None) -> bool:
        return True


class ToolPageRuntimeWidgetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_topk_widgets_follow_score_mode(self) -> None:
        harness = _ToolPageRuntimeWidgetsHarness()

        harness.cmb_mode.setCurrentText("proto")
        harness._update_runtime_widgets()
        self.assertFalse(harness.spin_topk.isEnabled())
        self.assertFalse(harness.lbl_topk.isEnabled())
        self.assertEqual(harness.lbl_topk.styleSheet(), harness._algo_param_label_disabled_style)

        harness.cmb_mode.setCurrentText("topk")
        harness._update_runtime_widgets()
        self.assertTrue(harness.spin_topk.isEnabled())
        self.assertTrue(harness.lbl_topk.isEnabled())
        self.assertEqual(harness.lbl_topk.styleSheet(), harness._algo_param_label_style)


if __name__ == "__main__":
    unittest.main()
