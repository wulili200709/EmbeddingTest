from __future__ import annotations

import os
import sys
import tempfile
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


class _RuntimeParamsAlgo:
    def __init__(self) -> None:
        self.product_params = SimpleNamespace(
            algorithm="efficientnet_b0",
            score_mode="proto",
            margin=0.02,
            topk=3,
        )
        self.model = SimpleNamespace(
            backbone="efficientnet_b0",
            device="cpu",
            score_mode="proto",
            margin=0.02,
            topk=3,
        )
        self.load_calls: list[dict[str, object]] = []
        self.train_calls: list[dict[str, object]] = []
        self.apply_calls = 0

    def tool_model_key(self, model_key: object) -> str:
        return str(model_key or "").strip()

    def is_learning_tool(self, code) -> bool:
        return str(code or "").strip() == "shared_backbone_register"

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, code) -> str:
        normalized = str(code or "").strip()
        if normalized == "shared_backbone_register":
            return "efficientnet_b0"
        return normalized

    def resolve_learning_algorithm(self, code) -> str:
        return self.resolve_tool_algorithm(code)

    def is_embedding_algorithm(self, algorithm) -> bool:
        return str(algorithm or "").strip() in {"efficientnet_b0", "patchcore_lite"}

    def is_anomaly_tool(self, code) -> bool:
        return str(code or "").strip() == "patchcore_lite"

    def is_anomaly_algorithm(self, algorithm) -> bool:
        return str(algorithm or "").strip() == "patchcore_lite"

    def apply_params_to_model(self) -> None:
        self.apply_calls += 1
        if self.model is not None:
            self.model.score_mode = str(self.product_params.score_mode)
            self.model.margin = float(self.product_params.margin)
            self.model.topk = int(self.product_params.topk)

    def load_model_for_algorithm(self, algorithm: str, product_dir: str, *, model_key: object = ""):
        self.load_calls.append(
            {
                "algorithm": str(algorithm),
                "product_dir": str(product_dir),
                "model_key": str(model_key or ""),
                "score_mode": str(self.product_params.score_mode),
                "margin": float(self.product_params.margin),
                "topk": int(self.product_params.topk),
            }
        )
        if str(algorithm) == "patchcore_lite":
            self.model = SimpleNamespace(
                backbone="efficientnet_b0",
                device="cpu",
                threshold=float(self.product_params.margin),
                topk=int(self.product_params.topk),
            )
        else:
            self.model = SimpleNamespace(
                backbone=str(algorithm),
                device="cpu",
                score_mode=str(self.product_params.score_mode),
                margin=float(self.product_params.margin),
                topk=int(self.product_params.topk),
            )
        return self.model, "loaded"

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
                "score_mode": str(self.product_params.score_mode),
                "margin": float(self.product_params.margin),
                "topk": int(self.product_params.topk),
            }
        )
        if str(algorithm) == "patchcore_lite":
            model = SimpleNamespace(
                algorithm="patchcore_lite",
                backbone="efficientnet_b0",
                threshold=0.44,
                topk=int(self.product_params.topk),
                device="cpu",
            )
        else:
            model = SimpleNamespace(
                backbone=str(algorithm),
                score_mode=str(self.product_params.score_mode),
                margin=float(self.product_params.margin),
                topk=int(self.product_params.topk),
                device="cpu",
            )
        return TrainResult(
            algorithm=str(algorithm),
            is_embedding=True,
            status_message="trained",
            dialog_message="done",
            model=model,
            saved_model_path="demo.npz",
            result_rows=[],
        )


class _RuntimeParamsHarness:
    _apply_runtime_params_to_ui = ToolPage._apply_runtime_params_to_ui
    _on_runtime_params_changed = ToolPage._on_runtime_params_changed
    _resolve_training_algorithm = ToolPage._resolve_training_algorithm
    _train_inspection_item = ToolPage._train_inspection_item
    _is_embedding_algorithm = ToolPage._is_embedding_algorithm
    _is_anomaly_algorithm = ToolPage._is_anomaly_algorithm
    load_embedding_model = ToolPage.load_embedding_model

    def __init__(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.session = SimpleNamespace(
            product_dir=self._tmpdir.name,
            current_product="demo",
        )
        self.algo = _RuntimeParamsAlgo()
        self.lbl_status = QtWidgets.QLabel("")
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["proto", "topk"])
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setDecimals(4)
        self.spin_margin.setValue(0.02)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setValue(3)
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code="shared_backbone_register",
                params={"score_mode": "topk", "margin": 0.11, "topk": 7},
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam1",
                roi_label="roi2",
                algorithm_code="shared_backbone_register",
                params={"score_mode": "proto", "margin": 0.33, "topk": 2},
            ),
            InspectionItem(
                item_id="roi3",
                display_name="ROI3",
                camera_id="cam1",
                roi_label="roi3",
                algorithm_code="patchcore_lite",
                params={"score_mode": "topk", "margin": 0.21, "topk": 5},
            ),
        ]
        self._selected_index = 0
        self.current_algorithm_value = "efficientnet_b0"
        self._updating_runtime_params = False
        self.persist_calls = 0
        self.save_runtime_calls = 0
        self.update_calls = 0
        self.hint_calls = 0

    def cleanup(self) -> None:
        self._tmpdir.cleanup()

    def current_algorithm(self) -> str:
        return self.current_algorithm_value

    def _set_current_algorithm(self, algorithm: str) -> None:
        self.current_algorithm_value = str(algorithm or "").strip()

    def _selected_inspection_item(self):
        if 0 <= self._selected_index < len(self.inspection_items):
            return self.inspection_items[self._selected_index]
        return None

    def _persist_inspection_items(self) -> None:
        self.persist_calls += 1

    def _save_runtime_params(self) -> None:
        self.save_runtime_calls += 1

    def _update_runtime_widgets(self) -> None:
        self.update_calls += 1

    def _update_learning_backbone_hint(self) -> None:
        self.hint_calls += 1

    def _training_sample_groups_for_role(self, camera_role=None, *, roi_label=None):
        return ["ok.png"], [], ["ok.png"]

    def _missing_training_roi_paths(self, roi_label: str, candidate_paths: list[str]):
        return []


class ToolPageItemRuntimeParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_apply_runtime_params_to_ui_uses_selected_item_params(self) -> None:
        harness = _RuntimeParamsHarness()
        try:
            harness._selected_index = 0
            harness.current_algorithm_value = "efficientnet_b0"

            harness._apply_runtime_params_to_ui()

            self.assertEqual(harness.current_algorithm_value, "efficientnet_b0")
            self.assertEqual(harness.cmb_mode.currentText(), "topk")
            self.assertAlmostEqual(harness.spin_margin.value(), 0.11)
            self.assertEqual(harness.spin_topk.value(), 7)
            self.assertAlmostEqual(harness.algo.product_params.margin, 0.11)
            self.assertEqual(harness.algo.product_params.topk, 7)
        finally:
            harness.cleanup()

    def test_runtime_param_change_persists_selected_item(self) -> None:
        harness = _RuntimeParamsHarness()
        try:
            harness._selected_index = 1
            harness.current_algorithm_value = "efficientnet_b0"
            harness.cmb_mode.setCurrentText("topk")
            harness.spin_margin.setValue(0.58)
            harness.spin_topk.setValue(9)

            harness._on_runtime_params_changed()

            self.assertEqual(
                harness.inspection_items[1].params,
                {"score_mode": "topk", "margin": 0.58, "topk": 9},
            )
            self.assertEqual(harness.persist_calls, 1)
            self.assertEqual(harness.save_runtime_calls, 1)
            self.assertEqual(harness.algo.apply_calls, 1)
            self.assertAlmostEqual(harness.algo.product_params.margin, 0.58)
        finally:
            harness.cleanup()

    def test_load_embedding_model_uses_model_key_item_params(self) -> None:
        harness = _RuntimeParamsHarness()
        try:
            harness._selected_index = 0

            harness.load_embedding_model("efficientnet_b0", model_key="cam1__roi2")

            self.assertEqual(len(harness.algo.load_calls), 1)
            self.assertEqual(harness.algo.load_calls[0]["model_key"], "cam1__roi2")
            self.assertAlmostEqual(harness.algo.load_calls[0]["margin"], 0.33)
            self.assertEqual(harness.algo.load_calls[0]["topk"], 2)
            self.assertEqual(harness.algo.load_calls[0]["score_mode"], "proto")
            self.assertEqual(harness.lbl_status.text(), "loaded")
        finally:
            harness.cleanup()

    def test_train_inspection_item_updates_anomaly_threshold_for_item(self) -> None:
        harness = _RuntimeParamsHarness()
        try:
            anomaly_item = harness.inspection_items[2]

            result = harness._train_inspection_item(anomaly_item)

            self.assertTrue(result.is_embedding)
            self.assertEqual(
                harness.algo.train_calls,
                [
                    {
                        "ok_files": ["ok.png"],
                        "ng_files": [],
                        "algorithm": "patchcore_lite",
                        "product_dir": harness.session.product_dir,
                        "label_names": ["roi3"],
                        "model_key": "cam1__roi3",
                        "score_mode": "topk",
                        "margin": 0.21,
                        "topk": 5,
                    }
                ],
            )
            self.assertEqual(
                anomaly_item.params,
                {"score_mode": "topk", "margin": 0.44, "topk": 5},
            )
        finally:
            harness.cleanup()


if __name__ == "__main__":
    unittest.main()
