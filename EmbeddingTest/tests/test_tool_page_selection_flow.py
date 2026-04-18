from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

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
    _move_selected_sample_to = ToolPage._move_selected_sample_to
    _clear_selected_inspection_item = ToolPage._clear_selected_inspection_item
    _show_selected_image_path = ToolPage._show_selected_image_path
    _on_select_ok = ToolPage._on_select_ok
    _on_select_ng = ToolPage._on_select_ng
    _on_select_test = ToolPage._on_select_test
    _on_tab_changed = ToolPage._on_tab_changed
    _remove_selected_from = ToolPage._remove_selected_from
    _select_path_in_sample_list = ToolPage._select_path_in_sample_list
    _sync_sample_selection_to_path = ToolPage._sync_sample_selection_to_path

    def __init__(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(QtWidgets.QWidget(), "训练样本")
        self.tabs.addTab(QtWidgets.QWidget(), "测试样本")
        self.ok_list = QtWidgets.QListWidget()
        self.ng_list = QtWidgets.QListWidget()
        self.test_list = QtWidgets.QListWidget()
        self.train_files: list[str] = []
        self.ok_files: list[str] = []
        self.ng_files: list[str] = []
        self.test_files: list[str] = []
        self.canvas = _DummyCanvas()
        self.inspection_items_table = QtWidgets.QTableWidget(1, 1)
        self.inspection_items_table.setItem(0, 0, QtWidgets.QTableWidgetItem("roi1"))
        self.load_calls: list[str] = []
        self.status_calls: list[str] = []
        self.refresh_calls = 0
        self.save_calls = 0
        self.selection_calls: list[str] = []

    def _load_canvas_image(self, path: str) -> None:
        self.load_calls.append(path)
        self.canvas._path = path

    def _set_status_for_current_image(self, path: str) -> None:
        self.status_calls.append(path)

    def _sample_paths_for_kind(self, kind: str, _camera_role=None) -> list[str]:
        if str(kind) == "train":
            return list(self.train_files)
        return list(self.test_files)

    def _current_sample_tab_kind(self) -> str:
        return "train" if self.tabs.currentIndex() == 0 else "test"

    def _update_sample_panel_widgets(self) -> None:
        return None

    def _refresh_lists(self) -> None:
        self.refresh_calls += 1
        current_train = self.ok_list.currentItem()
        current_test = self.test_list.currentItem()
        current_train_path = current_train.data(QtCore.Qt.UserRole) if current_train is not None else None
        current_test_path = current_test.data(QtCore.Qt.UserRole) if current_test is not None else None

        def fill(list_widget: QtWidgets.QListWidget, paths: list[str], current_path: object) -> None:
            blocker = QtCore.QSignalBlocker(list_widget)
            list_widget.clear()
            selected_row = -1
            for row, path in enumerate(paths):
                item = QtWidgets.QListWidgetItem(path)
                item.setData(QtCore.Qt.UserRole, path)
                list_widget.addItem(item)
                if current_path == path:
                    selected_row = row
            if selected_row >= 0:
                list_widget.setCurrentRow(selected_row)
            del blocker

        fill(self.ok_list, self.train_files, current_train_path)
        fill(self.test_list, self.test_files, current_test_path)

    def _clear_training_roi_review_state(self) -> None:
        return None

    def _save_session(self) -> None:
        self.save_calls += 1

    def _select_path_in_current_tab(self, path: str) -> None:
        self.selection_calls.append(path)

    def current_camera_role(self) -> str:
        return "cam1"


class ToolPageSelectionFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_tab_change_loads_first_image_once(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.train_files = ["ok_a.png", "ok_b.png"]
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
        harness.tabs.setCurrentIndex(1)
        harness.test_list.setCurrentRow(0)
        harness.canvas._path = "test_a.png"

        harness._on_select_test()

        self.assertEqual(harness.load_calls, [])
        self.assertEqual(harness.status_calls, ["test_a.png"])

    def test_selecting_image_clears_inspection_item_selection(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.train_files = ["ok_a.png"]
        harness.ok_list.addItem("ok_a.png")
        harness.tabs.setCurrentIndex(0)
        harness.ok_list.setCurrentRow(0)
        harness.inspection_items_table.setCurrentCell(0, 0)

        harness._on_select_ok()

        self.assertEqual(harness.inspection_items_table.currentRow(), -1)
        self.assertEqual(harness.status_calls, ["ok_a.png"])

    def test_remove_selected_from_test_tab_removes_current_image(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.test_files = ["test_a.png", "test_b.png"]
        harness.test_list.addItems(["test_a.png", "test_b.png"])
        harness.test_list.setCurrentRow(0)

        harness._remove_selected_from("TEST")

        self.assertEqual(harness.test_files, ["test_b.png"])

    def test_move_test_sample_into_training(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.test_files = ["test_a.png"]
        item = QtWidgets.QListWidgetItem("test_a.png")
        item.setData(QtCore.Qt.UserRole, "test_a.png")
        harness.test_list.addItem(item)
        harness.tabs.setCurrentIndex(1)
        harness.test_list.setCurrentRow(0)

        harness._move_selected_sample_to("TRAIN")

        self.assertEqual(harness.train_files, ["test_a.png"])
        self.assertEqual(harness.test_files, [])
        self.assertEqual(harness.tabs.currentIndex(), 0)
        self.assertEqual(harness.save_calls, 1)
        self.assertEqual(harness.refresh_calls, 1)
        self.assertEqual(harness.selection_calls, ["test_a.png"])

    def test_move_test_sample_preserves_neighbor_selection_in_test_tab(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.test_files = ["test_a.png", "test_b.png", "test_c.png"]
        for path in harness.test_files:
            item = QtWidgets.QListWidgetItem(path)
            item.setData(QtCore.Qt.UserRole, path)
            harness.test_list.addItem(item)
        harness.tabs.setCurrentIndex(1)
        harness.test_list.setCurrentRow(1)

        harness._move_selected_sample_to("TRAIN")

        self.assertEqual(harness.test_files, ["test_a.png", "test_c.png"])
        self.assertEqual(harness.test_list.currentRow(), 1)
        self.assertEqual(
            harness.test_list.currentItem().data(QtCore.Qt.UserRole),
            "test_c.png",
        )

    def test_sync_sample_selection_switches_to_test_path_after_test_run(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.train_files = ["train_a.png", "train_b.png"]
        harness.test_files = ["test_a.png", "test_b.png"]
        for path in harness.train_files:
            harness.ok_list.addItem(path)
        for path in harness.test_files:
            harness.test_list.addItem(path)
        harness.tabs.setCurrentIndex(0)
        harness.ok_list.setCurrentRow(1)

        synced = harness._sync_sample_selection_to_path("test_a.png", switch_tab=True)

        self.assertTrue(synced)
        self.assertEqual(harness.tabs.currentIndex(), 1)
        self.assertEqual(harness.test_list.currentRow(), 0)
        self.assertEqual(harness.ok_list.currentRow(), -1)

    def test_sync_sample_selection_keeps_training_path_selected(self) -> None:
        harness = _ToolPageSelectionHarness()
        harness.train_files = ["train_a.png", "train_b.png"]
        harness.test_files = ["test_a.png"]
        for path in harness.train_files:
            harness.ok_list.addItem(path)
        for path in harness.test_files:
            harness.test_list.addItem(path)
        harness.tabs.setCurrentIndex(0)
        harness.ok_list.setCurrentRow(1)

        synced = harness._sync_sample_selection_to_path("train_a.png", switch_tab=True)

        self.assertTrue(synced)
        self.assertEqual(harness.tabs.currentIndex(), 0)
        self.assertEqual(harness.ok_list.currentRow(), 0)

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
