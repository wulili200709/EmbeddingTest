from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from PySide6 import QtCore, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from ui.debug.tool_page.page import ToolPage


class _ToolSectionToggleHarness:
    _toggle_tool_config_section = ToolPage._toggle_tool_config_section

    def __init__(self) -> None:
        self.tool_config_frame = QtWidgets.QWidget()
        self.btn_toggle_tools = QtWidgets.QToolButton()


class ToolPageToolSectionToggleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_toggle_tool_config_section_updates_visibility_and_arrow(self) -> None:
        harness = _ToolSectionToggleHarness()

        harness._toggle_tool_config_section(False)
        self.assertFalse(harness.tool_config_frame.isVisible())
        self.assertEqual(harness.btn_toggle_tools.arrowType(), QtCore.Qt.ArrowType.RightArrow)

        harness._toggle_tool_config_section(True)
        self.assertTrue(harness.tool_config_frame.isVisible())
        self.assertEqual(harness.btn_toggle_tools.arrowType(), QtCore.Qt.ArrowType.DownArrow)


if __name__ == "__main__":
    unittest.main()
