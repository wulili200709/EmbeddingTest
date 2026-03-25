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

from application.algorithm_controller import TrainResult
from domain.inspection_items import InspectionItem
from ui.debug.tool_page.page import ToolPage


class _FakeAlgo:
    def __init__(self) -> None:
        self.product_params = SimpleNamespace(
            algorithm="",
            score_mode="proto",
            margin=0.02,
            topk=3,
        )
        self.train_calls: list[dict[str, object]] = []

    def is_learning_tool(self, code) -> bool:
        return str(code or "").strip() == "shared_backbone_register"

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, code) -> str:
        return str(code or "").strip()

    def train(
        self,
        ok_files,
        ng_files,
        *,
        algorithm,
        product_dir,
        label_names,
        model_key,
    ):
        self.train_calls.append(
            {
                "ok_files": list(ok_files),
                "ng_files": list(ng_files),
                "algorithm": str(algorithm),
                "product_dir": str(product_dir),
                "label_names": list(label_names),
                "model_key": str(model_key),
            }
        )
        return TrainResult(
            algorithm=str(algorithm),
            is_embedding=True,
            status_message="trained",
            dialog_message="done",
            result_rows=[],
        )


class _RoleFilterHarness:
    _refresh_lists = ToolPage._refresh_lists
    _current_selected_path = ToolPage._current_selected_path
    _resolve_training_algorithm = ToolPage._resolve_training_algorithm
    _train_inspection_item = ToolPage._train_inspection_item

    def __init__(self) -> None:
        self.lbl_images_section = QtWidgets.QLabel("")
        self.ok_list = QtWidgets.QListWidget()
        self.ng_list = QtWidgets.QListWidget()
        self.test_list = QtWidgets.QListWidget()
        self.ok_files = ["cam1_ok_a.png", "cam2_ok_b.png"]
        self.ng_files = ["cam1_ng_a.png", "cam2_ng_b.png"]
        self.test_files = ["cam1_test_a.png", "cam2_test_b.png"]
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
                algorithm_code="shared_backbone_register",
            ),
        ]
        self._selected_item = None
        self._debug_role = "cam1"
        self.algo = _FakeAlgo()
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["proto", "topk"])
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setValue(0.02)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setValue(3)
        self.loc_method = "line2dup"
        self.session = SimpleNamespace(product_dir="demo_product")

    def _selected_inspection_item(self):
        return self._selected_item

    def _selected_debug_camera_role(self) -> str:
        return self._debug_role

    def current_camera_role(self) -> str:
        return self._debug_role

    def _missing_training_roi_paths(self, roi_label: str, candidate_paths: list[str]):
        return []


class ToolPageCameraRoleFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_refresh_lists_follows_current_camera_role(self) -> None:
        harness = _RoleFilterHarness()

        harness._debug_role = "cam1"
        harness._refresh_lists()
        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam1）")
        self.assertEqual(harness.ok_list.count(), 1)
        self.assertEqual(harness.ok_list.item(0).text(), "cam1_ok_a.png")
        self.assertEqual(harness.test_list.count(), 1)
        self.assertEqual(harness.test_list.item(0).text(), "cam1_test_a.png")

        harness._debug_role = "cam2"
        harness._refresh_lists()
        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam2）")
        self.assertEqual(harness.ok_list.count(), 1)
        self.assertEqual(harness.ok_list.item(0).text(), "cam2_ok_b.png")
        self.assertEqual(harness.ng_list.item(0).text(), "cam2_ng_b.png")

    def test_refresh_lists_ignore_selected_tool_camera_role(self) -> None:
        harness = _RoleFilterHarness()
        harness._debug_role = "cam1"
        harness._selected_item = harness.inspection_items[1]

        harness._refresh_lists()

        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam1）")
        self.assertEqual(harness.ok_list.item(0).text(), "cam1_ok_a.png")

    def test_train_inspection_item_uses_only_matching_camera_samples(self) -> None:
        harness = _RoleFilterHarness()

        result = harness._train_inspection_item(harness.inspection_items[0])

        self.assertEqual(result.algorithm, "efficientnet_b0")
        self.assertEqual(
            harness.algo.train_calls,
            [
                {
                    "ok_files": ["cam1_ok_a.png"],
                    "ng_files": ["cam1_ng_a.png"],
                    "algorithm": "efficientnet_b0",
                    "product_dir": "demo_product",
                    "label_names": ["roi1"],
                    "model_key": "cam1__roi1",
                }
            ],
        )

    def test_single_camera_product_still_follows_current_role(self) -> None:
        harness = _RoleFilterHarness()
        harness.inspection_items = [harness.inspection_items[0]]
        harness.ok_files = ["cam1_ok.png", "cam2_ok.png"]
        harness.ng_files = ["cam1_ng.png"]
        harness.test_files = ["cam1_test.png", "cam2_test.png"]

        harness._refresh_lists()

        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam1）")
        self.assertEqual(harness.ok_list.count(), 1)
        self.assertEqual(harness.ng_list.count(), 1)
        self.assertEqual(harness.test_list.count(), 1)


if __name__ == "__main__":
    unittest.main()
