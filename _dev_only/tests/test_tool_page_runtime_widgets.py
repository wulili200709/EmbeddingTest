from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.inspection_items = []
        self.algo = SimpleNamespace(
            is_measurement_tool=lambda algorithm: algorithm == "line_distance",
        )

    def current_camera_role(self) -> str:
        return "cam1"

    def window(self):
        return self

    def _selected_inspection_item(self):
        return None

    def _sync_training_action_buttons(self) -> None:
        return None

    def _update_sample_panel_widgets(self) -> None:
        return None

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

    def test_measurement_only_item_keeps_test_button_enabled(self) -> None:
        harness = _ToolPageRuntimeWidgetsHarness()
        harness.inspection_items = [
            SimpleNamespace(enabled=True, camera_id="cam1", algorithm_code="line_distance"),
        ]

        harness._update_runtime_widgets()

        self.assertTrue(harness.btn_test.isEnabled())
        self.assertFalse(harness.btn_train.isEnabled())

    def test_test_button_requires_enabled_item_for_current_camera(self) -> None:
        harness = _ToolPageRuntimeWidgetsHarness()
        harness.inspection_items = [
            SimpleNamespace(enabled=False, camera_id="cam1", algorithm_code="line_distance"),
            SimpleNamespace(enabled=True, camera_id="cam2", algorithm_code="line_distance"),
        ]

        harness._update_runtime_widgets()

        self.assertFalse(harness.btn_test.isEnabled())


if __name__ == "__main__":
    unittest.main()
