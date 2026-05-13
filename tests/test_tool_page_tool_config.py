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
from domain.inspection_items import InspectionItem, load_inspection_items
from ui.i18n import language_code, set_language, tr
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

    def embedding_model_storage_paths(self, backbone: str, product_dir: str):
        return [self.embedding_model_path(backbone, product_dir)]

    def algorithm_display_name(self, algorithm: str) -> str:
        return algorithm

    def resolve_tool_algorithm(self, algorithm_code: str) -> str:
        return str(algorithm_code or "").strip()

    def is_measurement_tool(self, algorithm_code: str) -> bool:
        return str(algorithm_code or "").strip() in {
            "find_line",
            "find_line_subpix",
            "line_distance",
            "line_distance_ref_normal",
        }

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
        self.inspection_items_table = QtWidgets.QTableWidget(0, 5)
        self.lbl_tool_config_hint = QtWidgets.QLabel("")
        self.lbl_tool_config_hint.hide()
        self.lbl_status = QtWidgets.QLabel("")
        self.btn_delete_line_distance_tool = QtWidgets.QPushButton()
        self.inspectionItemsChanged = SimpleNamespace(emit=lambda: None)
        self.canvas = _DummyCanvas()
        self._inspection_items_table_loading = False
        self._updating_runtime_params = False
        self.current_algorithm_value = ""
        self.runtime_update_calls = 0
        self.load_shape_calls: list[tuple[str, str]] = []
        self.refresh_list_calls = 0
        self.current_camera = "cam1"

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

    def _update_measurement_params_panel(self) -> None:
        return None


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

    def test_subpixel_find_line_stays_config_only_in_algorithm_combo(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness.inspection_items.append(
                InspectionItem(
                    item_id="subpix_line",
                    display_name="subpix_line",
                    camera_id="cam1",
                    roi_label="roi3",
                    algorithm_code="find_line_subpix",
                    params={"line": {"direction": "left_right", "edge_detector": "subpix_shen"}},
                )
            )

            harness._refresh_inspection_items_table()

            algorithm_combo = harness.inspection_items_table.cellWidget(1, 3)
            self.assertIsNotNone(algorithm_combo)
            self.assertEqual(algorithm_combo.findData("find_line_subpix"), -1)
            self.assertGreaterEqual(algorithm_combo.findData("find_line"), 0)
            self.assertEqual(algorithm_combo.currentData(), "find_line")
            self.assertEqual(harness.inspection_items[-1].algorithm_code, "find_line_subpix")

            harness.inspection_items_table.setCurrentCell(1, 1)
            harness.inspection_items_table.selectRow(1)
            harness._on_inspection_items_selection_changed()

            self.assertEqual(harness.current_algorithm_value, "find_line")
            self.assertEqual(harness.algo.product_params.algorithm, "find_line_subpix")
        finally:
            harness.cleanup()

    def test_delete_selected_line_distance_tool_removes_only_selected_measurement(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness.inspection_items.extend(
                [
                    InspectionItem(
                        item_id="line_distance",
                        display_name="Line Distance",
                        camera_id="cam1",
                        roi_label="",
                        algorithm_code="line_distance",
                    ),
                    InspectionItem(
                        item_id="line_distance_2",
                        display_name="Line Distance",
                        camera_id="cam1",
                        roi_label="",
                        algorithm_code="line_distance",
                    ),
                ]
            )

            harness._refresh_inspection_items_table()
            harness.inspection_items_table.setCurrentCell(2, 1)
            harness.inspection_items_table.selectRow(2)
            tool_config._delete_selected_line_distance_tool(harness)

            remaining_ids = [item.item_id for item in harness.inspection_items]
            self.assertIn("line_distance", remaining_ids)
            self.assertNotIn("line_distance_2", remaining_ids)
            self.assertEqual(harness.inspection_items_table.rowCount(), 2)
            self.assertFalse(harness.btn_delete_line_distance_tool.isEnabled())
            persisted_ids = [item.item_id for item in load_inspection_items(harness.session.inspection_items_path)]
            self.assertEqual(persisted_ids, remaining_ids)
        finally:
            harness.cleanup()

    def test_delete_line_distance_button_only_visible_for_selected_distance_tool(self) -> None:
        harness = _ToolConfigHarness()
        try:
            harness.inspection_items.append(
                InspectionItem(
                    item_id="line_distance",
                    display_name="Line Distance",
                    camera_id="cam1",
                    roi_label="",
                    algorithm_code="line_distance",
                )
            )

            harness._refresh_inspection_items_table()
            self.assertFalse(harness.btn_delete_line_distance_tool.isVisible())

            harness.inspection_items_table.setCurrentCell(0, 1)
            harness.inspection_items_table.selectRow(0)
            harness._on_inspection_items_selection_changed()
            self.assertFalse(harness.btn_delete_line_distance_tool.isVisible())

            harness.inspection_items_table.setCurrentCell(1, 1)
            harness.inspection_items_table.selectRow(1)
            harness._on_inspection_items_selection_changed()
            self.assertTrue(harness.btn_delete_line_distance_tool.isVisible())
            self.assertTrue(harness.btn_delete_line_distance_tool.isEnabled())
        finally:
            harness.cleanup()

    def test_line_distance_default_name_is_translated_in_table(self) -> None:
        previous = language_code()
        harness = _ToolConfigHarness()
        try:
            set_language("zh_CN", persist=False)
            harness.inspection_items.append(
                InspectionItem(
                    item_id="line_distance",
                    display_name="Line Distance",
                    camera_id="cam1",
                    roi_label="",
                    algorithm_code="line_distance",
                )
            )

            harness._refresh_inspection_items_table()
            item = harness.inspection_items_table.item(1, 1)
            self.assertIsNotNone(item)
            self.assertEqual(item.text(), tr("debug.algorithm.line_distance"))
        finally:
            set_language(previous, persist=False)
            harness.cleanup()

    def test_line_distance_algorithm_combo_is_config_only(self) -> None:
        previous = language_code()
        harness = _ToolConfigHarness()
        try:
            set_language("zh_CN", persist=False)
            harness.inspection_items.append(
                InspectionItem(
                    item_id="line_distance",
                    display_name="Reference Normal Distance",
                    camera_id="cam1",
                    roi_label="",
                    algorithm_code="line_distance_ref_normal",
                )
            )

            harness._refresh_inspection_items_table()

            name_item = harness.inspection_items_table.item(1, 1)
            self.assertIsNotNone(name_item)
            self.assertEqual(name_item.text(), tr("debug.algorithm.line_distance"))
            algorithm_combo = harness.inspection_items_table.cellWidget(1, 3)
            self.assertIsNotNone(algorithm_combo)
            self.assertFalse(algorithm_combo.isEnabled())
            self.assertEqual(algorithm_combo.currentText(), tr("debug.algorithm.line_distance"))
            self.assertNotEqual(algorithm_combo.currentText(), tr("debug.algorithm.line_distance_ref_normal"))
        finally:
            set_language(previous, persist=False)
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

    def test_measurement_params_use_combo_data_not_translated_text(self) -> None:
        harness = _ToolConfigHarness()
        try:
            item = InspectionItem(
                item_id="line1",
                display_name="line1",
                camera_id="cam1",
                roi_label="line1",
                algorithm_code="find_line",
                params={"line": {"direction": "left_right", "polarity": "any"}},
            )
            harness.inspection_items = [item]
            harness._visible_inspection_item_indexes = [0]
            harness.inspection_items_table.setRowCount(1)
            harness.inspection_items_table.selectRow(0)
            harness.inspectionItemsChanged = SimpleNamespace(emit=lambda: None)
            harness._measurement_params_loading = False

            harness.cmb_measurement_unit = QtWidgets.QComboBox()
            harness.cmb_measurement_unit.addItems(["px", "mm"])
            harness.cmb_measurement_line_a_tool = QtWidgets.QComboBox()
            harness.cmb_measurement_line_b_tool = QtWidgets.QComboBox()
            harness.cmb_measurement_line_a_direction = QtWidgets.QComboBox()
            harness.cmb_measurement_line_b_direction = QtWidgets.QComboBox()
            for text, value in (
                ("左到右", "left_right"),
                ("右到左", "right_left"),
                ("上到下", "top_down"),
                ("下到上", "bottom_up"),
            ):
                harness.cmb_measurement_line_a_direction.addItem(text, value)
                harness.cmb_measurement_line_b_direction.addItem(text, value)
            harness.cmb_measurement_polarity = QtWidgets.QComboBox()
            for text, value in (
                ("任意", "any"),
                ("暗到亮", "dark_to_bright"),
                ("亮到暗", "bright_to_dark"),
            ):
                harness.cmb_measurement_polarity.addItem(text, value)
            harness.spin_measurement_edge_threshold = QtWidgets.QDoubleSpinBox()
            harness.spin_measurement_scan_step = QtWidgets.QSpinBox()
            harness.spin_measurement_min_points = QtWidgets.QSpinBox()
            harness.spin_measurement_lower = QtWidgets.QDoubleSpinBox()
            harness.spin_measurement_upper = QtWidgets.QDoubleSpinBox()
            harness.spin_measurement_pixel_size = QtWidgets.QDoubleSpinBox()
            harness.chk_measurement_lower = QtWidgets.QCheckBox()
            harness.chk_measurement_upper = QtWidgets.QCheckBox()

            harness.cmb_measurement_line_a_direction.setCurrentIndex(1)
            harness.cmb_measurement_polarity.setCurrentIndex(1)
            harness.spin_measurement_edge_threshold.setValue(12.5)
            harness.spin_measurement_scan_step.setValue(4)
            harness.spin_measurement_min_points.setValue(9)

            tool_config._on_measurement_params_changed(harness)

            self.assertEqual(item.params["line"]["direction"], "right_left")
            self.assertEqual(item.params["line"]["polarity"], "dark_to_bright")
            self.assertEqual(item.params["line"]["edge_threshold"], 12.5)
            self.assertEqual(item.params["line"]["scan_step"], 4)
            self.assertEqual(item.params["line"]["min_points"], 9)
            self.assertEqual(item.params["line"]["edge_detector"], "canny")
        finally:
            harness.cleanup()


if __name__ == "__main__":
    unittest.main()
