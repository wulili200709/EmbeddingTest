from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PySide6 import QtCore, QtGui, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from algorithms.registry import SHARED_BACKBONE_ALGORITHM_CODE
from domain.inspection_items import InspectionItem
from line2dup.core.recipe import Line2DupRecipe
from ui.debug.tool_page import roi_ops, tool_config


class _DummyAlgo:
    def __init__(self, product_dir: str) -> None:
        self.product_params = SimpleNamespace(algorithm="")
        self._product_dir = product_dir

    def is_learning_tool(self, algorithm_code: str) -> bool:
        return algorithm_code == SHARED_BACKBONE_ALGORITHM_CODE

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def embedding_model_path(self, backbone: str, product_dir: str, model_key: str | None = None) -> str:
        suffix = model_key or "shared"
        return os.path.join(product_dir, f"{suffix}_{backbone}.npz")

    def algorithm_display_name(self, algorithm: str) -> str:
        return algorithm

    def resolve_tool_algorithm(self, algorithm_code: str) -> str:
        return str(algorithm_code or "").strip()

    def get_traditional_model_dict(self, algorithm: str, model_key: str | None = None):
        return None

    def traditional_model_storage_key(self, algorithm: str, model_key: str | None = None) -> str:
        return f"{algorithm}::{model_key or 'shared'}"


class _DummyCanvas:
    def __init__(self) -> None:
        self._path: str | None = None

    def image_path(self) -> str | None:
        return self._path


class _ToolConfigHarness:
    def __init__(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.session = SimpleNamespace(
            product_dir=self._tmpdir.name,
            inspection_items_path=os.path.join(self._tmpdir.name, "inspection_items.json"),
        )
        self.algo = _DummyAlgo(self.session.product_dir)
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="roi1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code=SHARED_BACKBONE_ALGORITHM_CODE,
            ),
            InspectionItem(
                item_id="roi2",
                display_name="roi2",
                camera_id="cam2",
                roi_label="roi2",
                algorithm_code=SHARED_BACKBONE_ALGORITHM_CODE,
            ),
        ]
        self.inspection_items_table = QtWidgets.QTableWidget(0, 6)
        self.lbl_tool_config_hint = QtWidgets.QLabel("")
        self.lbl_tool_config_hint.hide()
        self.canvas = _DummyCanvas()
        self._inspection_items_table_loading = False
        self._updating_runtime_params = False
        self.current_algorithm_value = ""
        self.runtime_update_calls = 0
        self.load_shape_calls: list[tuple[str, str]] = []
        self.refresh_list_calls = 0
        self.current_camera = "cam1"
        self._recipe_by_role = {
            "cam1": Line2DupRecipe(
                reference_regions=[
                    {"output_label": "roi1", "display_name": "hole"},
                    {"output_label": "roi2", "display_name": "hole"},
                    {"output_label": "roi3", "display_name": "pusher"},
                ]
            ),
            "cam2": Line2DupRecipe(
                reference_regions=[
                    {"output_label": "roi1", "display_name": "cam2_group"},
                ]
            ),
        }

    def cleanup(self) -> None:
        self._tmpdir.cleanup()

    def _set_current_algorithm(self, algorithm: str) -> None:
        self.current_algorithm_value = algorithm

    def _update_runtime_widgets(self) -> None:
        self.runtime_update_calls += 1

    def _load_shape_for_label(self, image_path: str, label_name: str) -> None:
        self.load_shape_calls.append((image_path, label_name))

    def _refresh_lists(self) -> None:
        self.refresh_list_calls += 1

    def current_camera_role(self) -> str:
        return self.current_camera

    def line2dup_recipe_for_role(self, role: str, force_reload: bool = False):
        return self._recipe_by_role.get(str(role or "cam1"))

    def _current_label(self) -> str:
        return "roi"

    def _roi_status_for_path(self, image_path: str, label: str):
        return None

    def _refresh_inspection_items_table(self) -> None:
        tool_config._refresh_inspection_items_table(self)

    def _selected_inspection_item(self):
        return tool_config._selected_inspection_item(self)

    def _on_inspection_items_selection_changed(self) -> None:
        tool_config._on_inspection_items_selection_changed(self)

    def _update_learning_backbone_hint(self) -> None:
        tool_config._update_learning_backbone_hint(self)

    def _on_inspection_item_group_changed(self, row: int, group_name: object) -> None:
        tool_config._on_inspection_item_group_changed(self, row, group_name)


class ToolPageToolConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_refresh_does_not_select_any_tool_by_default(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness._refresh_inspection_items_table()

            self.assertEqual(harness.inspection_items_table.currentRow(), -1)
            self.assertIsNone(harness._selected_inspection_item())
            self.assertEqual(harness.lbl_tool_config_hint.text(), "")
            self.assertFalse(harness.lbl_tool_config_hint.isVisible())
            self.assertEqual(harness.current_algorithm_value, "")
            self.assertGreaterEqual(harness.runtime_update_calls, 1)
            self.assertEqual(harness.inspection_items_table.rowCount(), 1)
        finally:
            harness.cleanup()

    def test_selected_tool_roi_uses_cyan_overlay_style(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness._refresh_inspection_items_table()
            harness.inspection_items_table.setCurrentCell(0, 1)
            selection_model = harness.inspection_items_table.selectionModel()
            self.assertIsNotNone(selection_model)
            selection_model.select(
                harness.inspection_items_table.model().index(0, 1),
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
            harness._on_inspection_items_selection_changed()

            color, width, dash = roi_ops._overlay_style_for_tool_label(harness, "img.png", "roi1")

            self.assertEqual(QtGui.QColor(color).name().lower(), "#00c8c8")
            self.assertGreaterEqual(width, 3.0)
            self.assertFalse(dash)
            self.assertEqual(harness.current_algorithm_value, "efficientnet_b0")
        finally:
            harness.cleanup()

    def test_clear_selection_removes_partial_highlight_state(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness._refresh_inspection_items_table()
            harness.inspection_items_table.setCurrentCell(0, 1)
            harness.inspection_items_table.clearSelection()
            harness._on_inspection_items_selection_changed()

            camera_combo = harness.inspection_items_table.cellWidget(0, 2)
            algorithm_combo = harness.inspection_items_table.cellWidget(0, 3)

            self.assertIsNotNone(camera_combo)
            self.assertIsNotNone(algorithm_combo)
            self.assertIn("background:#3a3a3a", camera_combo.styleSheet())
            self.assertIn("background:#3a3a3a", algorithm_combo.styleSheet())
            self.assertEqual(harness.lbl_tool_config_hint.text(), "")
        finally:
            harness.cleanup()

    def test_table_only_shows_items_for_current_camera_role(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness.current_camera = "cam2"
            harness._refresh_inspection_items_table()

            self.assertEqual(harness.inspection_items_table.rowCount(), 1)
            item = harness.inspection_items_table.item(0, 1)
            self.assertIsNotNone(item)
            self.assertEqual(item.text(), "roi2")
        finally:
            harness.cleanup()

    def test_row_highlight_and_selected_tool_follow_selected_row_not_current_row(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness.inspection_items.append(
                InspectionItem(
                    item_id="roi3",
                    display_name="roi3",
                    camera_id="cam1",
                    roi_label="roi3",
                    algorithm_code=SHARED_BACKBONE_ALGORITHM_CODE,
                )
            )
            harness._refresh_inspection_items_table()
            table = harness.inspection_items_table

            table.setCurrentCell(0, 1)
            selection_model = table.selectionModel()
            self.assertIsNotNone(selection_model)
            selection_model.select(
                table.model().index(0, 1),
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
            table.setCurrentCell(1, 1, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)

            tool_config._sync_inspection_items_row_highlight(harness)

            row0_camera = table.cellWidget(0, 2)
            row1_camera = table.cellWidget(1, 2)
            self.assertIsNotNone(row0_camera)
            self.assertIsNotNone(row1_camera)
            self.assertIn("background:#6ec0ff", row0_camera.styleSheet())
            self.assertIn("background:#3a3a3a", row1_camera.styleSheet())
            self.assertEqual(harness._selected_inspection_item().item_id, "roi1")
        finally:
            harness.cleanup()

    def test_group_column_uses_reference_roi_names_as_dropdown_options(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness._refresh_inspection_items_table()

            group_combo = harness.inspection_items_table.cellWidget(0, 4)

            self.assertIsNotNone(group_combo)
            values = [group_combo.itemData(index) for index in range(group_combo.count())]
            self.assertEqual(values, ["", "hole", "pusher"])
        finally:
            harness.cleanup()

    def test_group_dropdown_selection_persists_task_group(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness._refresh_inspection_items_table()

            group_combo = harness.inspection_items_table.cellWidget(0, 4)
            self.assertIsNotNone(group_combo)
            group_combo.setCurrentIndex(group_combo.findData("hole"))

            self.assertEqual(harness.inspection_items[0].task_group, "hole")
        finally:
            harness.cleanup()


if __name__ == "__main__":
    unittest.main()
