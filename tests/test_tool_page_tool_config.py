from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PySide6 import QtCore, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from domain.inspection_items import InspectionItem
from algorithms.registry import is_learning_tool_algorithm
from ui.debug.tool_page import tool_config


class _FakeSignal:
    def __init__(self) -> None:
        self.count = 0

    def emit(self) -> None:
        self.count += 1


class _FakeAlgo:
    def __init__(self) -> None:
        self.product_params = type("Params", (), {"algorithm": "", "traditional_models": {}})()

    def current_learning_backbone(self) -> str:
        return "mobilenet_v3_small"

    def is_learning_tool(self, code) -> bool:
        return is_learning_tool_algorithm(code)

    def resolve_tool_algorithm(self, code) -> str:
        return str(code or "").strip()

    def embedding_model_path(self, algorithm: str, product_dir: str, *, model_key: object = "") -> str:
        suffix = f"{model_key}_register_model_{algorithm}.npz" if model_key else f"register_model_{algorithm}.npz"
        return str(Path(product_dir) / suffix)

    def algorithm_display_name(self, algorithm: object) -> str:
        return str(algorithm or "")

    def traditional_model_storage_key(self, algorithm: str, *, model_key: object = "") -> str:
        return f"{algorithm}::{model_key}" if model_key else algorithm

    def get_traditional_model_dict(self, algorithm: str, *, model_key: object = ""):
        key = self.traditional_model_storage_key(algorithm, model_key=model_key)
        return self.product_params.traditional_models.get(key) or self.product_params.traditional_models.get(algorithm)


class _ToolConfigHarness:
    _selected_inspection_item_row = tool_config._selected_inspection_item_row
    _selected_inspection_item = tool_config._selected_inspection_item
    _on_inspection_items_selection_changed = tool_config._on_inspection_items_selection_changed
    _persist_inspection_items = tool_config._persist_inspection_items
    _refresh_inspection_items_table = tool_config._refresh_inspection_items_table
    _update_learning_backbone_hint = tool_config._update_learning_backbone_hint
    _on_inspection_items_table_item_changed = tool_config._on_inspection_items_table_item_changed
    _on_inspection_item_camera_changed = tool_config._on_inspection_item_camera_changed
    _on_inspection_item_algorithm_changed = tool_config._on_inspection_item_algorithm_changed

    def __init__(self, inspection_items_path: str) -> None:
        self.session = type(
            "Session",
            (),
            {
                "inspection_items_path": inspection_items_path,
                "product_dir": str(Path(inspection_items_path).parent),
            },
        )()
        self.algo = _FakeAlgo()
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code="shared_backbone_register",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam2",
                roi_label="roi2",
                algorithm_code="meanintensity",
            ),
        ]
        self._inspection_items_table_loading = False
        self._updating_runtime_params = False
        self.inspectionItemsChanged = _FakeSignal()
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.setStyleSheet("QComboBox{font-size:12px;}")
        self.lbl_tool_config_hint = QtWidgets.QLabel("")
        self.inspection_items_table = QtWidgets.QTableWidget(0, 5)
        self.inspection_items_table.setHorizontalHeaderLabels(["启用", "名称", "相机", "算法", "状态"])
        self.inspection_items_table.itemChanged.connect(self._on_inspection_items_table_item_changed)
        self.current_algorithm_value = ""
    def _set_current_algorithm(self, algorithm: str) -> None:
        self.current_algorithm_value = str(algorithm or "")

    def _update_runtime_widgets(self) -> None:
        return None


class ToolPageToolConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_tool_config_table_persists_algorithm_and_enabled_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inspection_items_path = str(Path(tmpdir) / "inspection_items.json")
            harness = _ToolConfigHarness(inspection_items_path)

            harness._refresh_inspection_items_table()

            self.assertEqual(harness.inspection_items_table.rowCount(), 2)
            self.assertEqual(harness.inspection_items_table.columnCount(), 5)
            headers = [
                harness.inspection_items_table.horizontalHeaderItem(i).text()
                for i in range(harness.inspection_items_table.columnCount())
            ]
            self.assertEqual(headers, ["启用", "名称", "相机", "算法", "状态"])
            self.assertIn("当前工具：ROI1", harness.lbl_tool_config_hint.text())
            self.assertTrue(harness.lbl_tool_config_hint.isVisible())
            self.assertEqual(harness.current_algorithm_value, "mobilenet_v3_small")
            self.assertEqual(harness.inspection_items_table.item(0, 4).text(), "未训练")
            self.assertEqual(harness.inspection_items_table.item(1, 4).text(), "未标定")

            algo_combo = harness.inspection_items_table.cellWidget(0, 3)
            self.assertIsNotNone(algo_combo)
            self.assertEqual(algo_combo.currentData(), "shared_backbone_register")
            self.assertEqual(algo_combo.currentText(), "学习工具")
            algo_combo.setCurrentIndex(algo_combo.findData("meanintensity"))
            self.assertEqual(harness.inspection_items_table.item(0, 4).text(), "未标定")

            enabled_item = harness.inspection_items_table.item(1, 0)
            self.assertIsNotNone(enabled_item)
            enabled_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.assertEqual(harness.inspection_items_table.item(1, 4).text(), "已禁用")

            self.assertEqual(harness.inspection_items[0].algorithm_code, "meanintensity")
            self.assertFalse(harness.inspection_items[1].enabled)
            self.assertGreaterEqual(harness.inspectionItemsChanged.count, 2)

            with open(inspection_items_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload[0]["algorithm_code"], "meanintensity")
            self.assertFalse(payload[1]["enabled"])


if __name__ == "__main__":
    unittest.main()
