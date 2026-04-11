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
            traditional_models={},
        )
        self.train_calls: list[dict[str, object]] = []

    def tool_model_key(self, model_key: object) -> str:
        return str(model_key or "").strip()

    def is_learning_tool(self, code) -> bool:
        return str(code or "").strip() == "shared_backbone_register"

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, code) -> str:
        return str(code or "").strip()

    def get_traditional_model_dict(self, algorithm: str, *, model_key: object = ""):
        storage_key = self.traditional_model_storage_key(algorithm, model_key=model_key)
        return self.product_params.traditional_models.get(storage_key)

    def traditional_model_storage_key(self, algorithm: str, *, model_key: object = "") -> str:
        normalized = str(model_key or "").strip()
        if normalized:
            return f"{algorithm}::{normalized}"
        return str(algorithm or "").strip()

    def train(
        self,
        ok_files,
        ng_files,
        *,
        algorithm,
        product_dir,
        label_names,
        model_key,
        ok_samples=None,
        ng_samples=None,
    ):
        self.train_calls.append(
            {
                "ok_files": list(ok_files),
                "ng_files": list(ng_files),
                "algorithm": str(algorithm),
                "product_dir": str(product_dir),
                "label_names": list(label_names),
                "model_key": str(model_key),
                "ok_samples": list(ok_samples or []),
                "ng_samples": list(ng_samples or []),
            }
        )
        if str(algorithm or "").strip() != "efficientnet_b0":
            self.product_params.traditional_models[
                self.traditional_model_storage_key(algorithm, model_key=model_key)
            ] = {
                "algorithm": str(algorithm),
                "threshold": 0.5,
                "ok_when": "greater_equal",
                "accuracy": 1.0,
            }
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
    _effective_model_key_for_item = ToolPage._effective_model_key_for_item
    _group_items_for_inspection_item = ToolPage._group_items_for_inspection_item
    _training_samples_for_inspection_item = ToolPage._training_samples_for_inspection_item
    _train_sample_paths_for_role = ToolPage._train_sample_paths_for_role
    _store_runtime_params_for_group = ToolPage._store_runtime_params_for_group
    _clear_previous_training_output = ToolPage._clear_previous_training_output
    _resolve_training_algorithm = ToolPage._resolve_training_algorithm
    _train_inspection_item = ToolPage._train_inspection_item
    current_camera_role = ToolPage.current_camera_role
    configured_camera_roles = ToolPage.configured_camera_roles
    set_configured_camera_roles = ToolPage.set_configured_camera_roles
    _apply_camera_role_options_to_combo = ToolPage._apply_camera_role_options_to_combo
    _apply_configured_camera_roles_to_ui = ToolPage._apply_configured_camera_roles_to_ui
    _set_current_camera_role = ToolPage._set_current_camera_role

    def __init__(self) -> None:
        self.lbl_images_section = QtWidgets.QLabel("")
        self.ok_list = QtWidgets.QListWidget()
        self.ng_list = QtWidgets.QListWidget()
        self.test_list = QtWidgets.QListWidget()
        self.cmb_current_camera_role = QtWidgets.QComboBox()
        self.cmb_current_camera_role.addItem("cam1", "cam1")
        self.cmb_current_camera_role.addItem("cam2", "cam2")
        self.cmb_debug_camera_role = QtWidgets.QComboBox()
        self.cmb_debug_camera_role.addItem("cam1", "cam1")
        self.cmb_debug_camera_role.addItem("cam2", "cam2")
        self.train_files = ["cam1_ok_a.png", "cam1_ng_a.png", "cam2_ok_b.png", "cam2_ng_b.png"]
        self.ok_files = []
        self.ng_files = []
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
        self._current_camera_role = "cam1"
        self._configured_camera_roles = ["cam1", "cam2"]
        self._sample_annotation_preview_dialog = None
        self.algo = _FakeAlgo()
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["proto", "topk"])
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setValue(0.02)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setValue(3)
        self.loc_method = "line2dup"
        self.session = SimpleNamespace(product_dir="demo_product")
        self.canvas = SimpleNamespace(image_path=lambda: None)
        self.persist_calls = 0

    def _selected_inspection_item(self):
        return self._selected_item

    def _selected_debug_camera_role(self) -> str:
        return self._debug_role

    def current_camera_role(self) -> str:
        return ToolPage.current_camera_role(self)

    def _clear_image_view_for_role_switch(self) -> None:
        return None

    def _apply_current_role_recipe_state(self) -> None:
        return None

    def _refresh_inspection_items_table(self) -> None:
        return None

    def _update_runtime_widgets(self) -> None:
        return None

    def _persist_inspection_items(self) -> None:
        self.persist_calls += 1

    def _save_runtime_params(self) -> None:
        return None

    def _refresh_debug_role_status(self) -> None:
        self._debug_role = self.current_camera_role()

    def _training_sample_groups_for_role(self, camera_role=None, *, roi_label=None):
        role = str(camera_role or self._debug_role)
        train_files = [path for path in self.train_files if role in path]
        ok_files = [path for path in train_files if "_ok" in path]
        ng_files = [path for path in train_files if "_ng" in path]
        return ok_files, ng_files, list(train_files)

    def _path_has_roi_geometry(self, path: str, roi_label: str) -> bool:
        return True

    def _sample_roi_status_for_path(self, path: str, camera_role: object, roi_label: str) -> str:
        lower_name = os.path.basename(path).lower()
        if "_ok" in lower_name:
            return "OK"
        if "_ng" in lower_name:
            return "NG"
        return ""

    def _sample_paths_for_kind(self, kind: str, camera_role=None):
        role = str(camera_role or self._debug_role)
        if str(kind) == "train":
            _ok_files, _ng_files, train_files = self._training_sample_groups_for_role(role)
            return train_files
        return [path for path in self.test_files if role in path]

    def _sample_item_display_text(self, path: str, _sample_kind: str, _camera_role=None) -> str:
        return os.path.basename(path)

    def _update_sample_panel_widgets(self) -> None:
        return None

    def _missing_training_roi_paths(self, roi_label: str, candidate_paths: list[str]):
        return []


class ToolPageCameraRoleFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_refresh_lists_follows_current_camera_role(self) -> None:
        harness = _RoleFilterHarness()

        harness._debug_role = "cam1"
        harness._current_camera_role = "cam1"
        harness.cmb_current_camera_role.setCurrentIndex(harness.cmb_current_camera_role.findData("cam1"))
        harness._refresh_lists()
        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam1）")
        self.assertEqual(harness.ok_list.count(), 2)
        self.assertEqual(harness.ok_list.item(0).text(), "cam1_ok_a.png")
        self.assertEqual(harness.ok_list.item(1).text(), "cam1_ng_a.png")
        self.assertEqual(harness.test_list.count(), 1)
        self.assertEqual(harness.test_list.item(0).text(), "cam1_test_a.png")

        harness._debug_role = "cam2"
        harness._current_camera_role = "cam2"
        harness.cmb_current_camera_role.setCurrentIndex(harness.cmb_current_camera_role.findData("cam2"))
        harness._refresh_lists()
        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam2）")
        self.assertEqual(harness.ok_list.count(), 2)
        self.assertEqual(harness.ok_list.item(0).text(), "cam2_ok_b.png")
        self.assertEqual(harness.ok_list.item(1).text(), "cam2_ng_b.png")
        self.assertEqual(harness.ng_list.count(), 0)

    def test_refresh_lists_ignore_selected_tool_camera_role(self) -> None:
        harness = _RoleFilterHarness()
        harness._debug_role = "cam1"
        harness._current_camera_role = "cam1"
        harness.cmb_current_camera_role.setCurrentIndex(harness.cmb_current_camera_role.findData("cam1"))
        harness._selected_item = harness.inspection_items[1]

        harness._refresh_lists()

        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam1）")
        self.assertEqual(harness.ok_list.item(0).text(), "cam1_ok_a.png")
        self.assertEqual(harness.ok_list.item(1).text(), "cam1_ng_a.png")

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
                    "ok_samples": [("cam1_ok_a.png", "roi1")],
                    "ng_samples": [("cam1_ng_a.png", "roi1")],
                }
            ],
        )

    def test_single_camera_product_still_follows_current_role(self) -> None:
        harness = _RoleFilterHarness()
        harness.inspection_items = [harness.inspection_items[0]]
        harness.train_files = ["cam1_ok.png", "cam1_ng.png", "cam2_ok.png"]
        harness.test_files = ["cam1_test.png", "cam2_test.png"]

        harness._refresh_lists()

        self.assertEqual(harness.lbl_images_section.text(), "  图片列表（cam1）")
        self.assertEqual(harness.ok_list.count(), 2)
        self.assertEqual(harness.ng_list.count(), 0)
        self.assertEqual(harness.test_list.count(), 1)


    def test_configured_camera_roles_disable_cam2_selection(self) -> None:
        harness = _RoleFilterHarness()
        harness._debug_role = "cam2"
        harness._current_camera_role = "cam2"
        harness.cmb_current_camera_role.setCurrentIndex(harness.cmb_current_camera_role.findData("cam2"))
        harness.cmb_debug_camera_role.setCurrentIndex(harness.cmb_debug_camera_role.findData("cam2"))

        harness.set_configured_camera_roles(["cam1"])

        self.assertEqual(harness.current_camera_role(), "cam1")
        self.assertFalse(harness.cmb_current_camera_role.isEnabled())
        self.assertFalse(harness.cmb_debug_camera_role.isEnabled())
        self.assertFalse(harness.cmb_current_camera_role.model().item(1).isEnabled())
        self.assertFalse(harness.cmb_debug_camera_role.model().item(1).isEnabled())


if __name__ == "__main__":
    unittest.main()
