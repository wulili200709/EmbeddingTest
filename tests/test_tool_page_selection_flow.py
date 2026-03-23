from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from PySide6 import QtGui, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from ui.debug.tool_page.page import ToolPage
from ui.shell.main_window import _normalize_application_font


class _DummyCanvas:
    def __init__(self) -> None:
        self._path: str | None = None

    def image_path(self) -> str | None:
        return self._path


class _ToolPageSelectionHarness:
    _current_selected_path = ToolPage._current_selected_path
    _show_selected_image_path = ToolPage._show_selected_image_path
    _on_select_ok = ToolPage._on_select_ok
    _on_select_ng = ToolPage._on_select_ng
    _on_select_test = ToolPage._on_select_test
    _on_tab_changed = ToolPage._on_tab_changed

    def __init__(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(QtWidgets.QWidget(), "OK")
        self.tabs.addTab(QtWidgets.QWidget(), "NG")
        self.tabs.addTab(QtWidgets.QWidget(), "TEST")
        self.ok_list = QtWidgets.QListWidget()
        self.ng_list = QtWidgets.QListWidget()
        self.test_list = QtWidgets.QListWidget()
        self.ok_files: list[str] = []
        self.ng_files: list[str] = []
        self.test_files: list[str] = []
        self.canvas = _DummyCanvas()
        self.load_calls: list[str] = []
        self.status_calls: list[str] = []

    def _load_canvas_image(self, path: str) -> None:
        self.load_calls.append(path)
        self.canvas._path = path

    def _set_status_for_current_image(self, path: str) -> None:
        self.status_calls.append(path)


class ToolPageSelectionFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_tab_change_loads_first_image_once(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.ok_files = ["ok_a.png", "ok_b.png"]
        harness.ok_list.addItems(["ok_a.png", "ok_b.png"])

        harness.tabs.setCurrentIndex(0)
        harness._on_tab_changed(0)

        self.assertEqual(harness.ok_list.currentRow(), 0)
        self.assertEqual(harness.load_calls, ["ok_a.png"])
        self.assertEqual(harness.status_calls, ["ok_a.png"])

        harness._on_tab_changed(0)
        self.assertEqual(harness.load_calls, ["ok_a.png"])
        self.assertEqual(harness.status_calls, ["ok_a.png", "ok_a.png"])

    def test_selecting_same_test_image_does_not_reload(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.test_files = ["test_a.png"]
        harness.test_list.addItem("test_a.png")
        harness.tabs.setCurrentIndex(2)
        harness.test_list.setCurrentRow(0)
        harness.canvas._path = "test_a.png"

        harness._on_select_test()

        self.assertEqual(harness.load_calls, [])
        self.assertEqual(harness.status_calls, ["test_a.png"])

    def test_application_font_normalization_sets_valid_point_size(self) -> None:
        app = self.app
        original_font = QtGui.QFont(app.font())
        try:
            app.setFont(QtGui.QFont())
            _normalize_application_font(app)
            self.assertGreater(app.font().pointSizeF(), 0.0)
        finally:
            app.setFont(original_font)

    def test_stylesheet_font_sizes_are_normalized_to_points(self) -> None:
        style_sheet = "QListWidget{font-size:12px;} QLabel{font-size:13px;}"
        normalized = ToolPage._normalize_font_size_units(style_sheet)
        self.assertEqual(
            normalized,
            "QListWidget{font-size:9pt;} QLabel{font-size:10pt;}",
        )


if __name__ == "__main__":
    unittest.main()
