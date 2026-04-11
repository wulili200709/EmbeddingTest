from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PySide6 import QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from algorithms.anomaly import AnomalyModel
from algorithms.anomaly_heatmap import AnomalyHeatmapResult
from ui.debug.anomaly_heatmap_dialog import AnomalyHeatmapDialog


class _AlgoHarness:
    def __init__(self, model: AnomalyModel) -> None:
        self.model = model

    def anomaly_model_path(self, algorithm: str, product_dir: str, *, model_key: object = "") -> str:
        return os.path.join(str(product_dir or ""), "demo_model.npz")

    def load_model_for_algorithm(self, algorithm: str, product_dir: str, *, model_key: object = ""):
        return self.model, "ok"

    def get_feat_net(self, backbone: str, device=None):
        return object()


class AnomalyHeatmapDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_dialog_renders_summary_and_tabs(self) -> None:
        model = AnomalyModel(
            algorithm="patchcore_lite",
            backbone="efficientnet_b0",
            threshold=0.10,
            topk=1,
            label_name="roi1",
            label_names=["roi1"],
            device="cpu",
            ok_bank=np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        result = AnomalyHeatmapResult(
            image_path="demo.png",
            roi_label="roi1",
            roi_xywh=(1, 2, 4, 4),
            pred="OK",
            score=0.08,
            threshold=0.10,
            diff=0.02,
            topk=1,
            ok_image_count=4,
            ok_patch_count=196,
            patch_max=0.40,
            patch_mean=0.10,
            coarse_patch_scores=np.asarray([[0.0, 1.0]], dtype=np.float32),
            heatmap_scores=np.zeros((4, 4), dtype=np.float32),
            heatmap_display=np.zeros((4, 4), dtype=np.float32),
            full_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
            roi_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            overlay_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
            roi_overlay_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            roi_heatmap_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        )

        with mock.patch("ui.debug.anomaly_heatmap_dialog.generate_anomaly_heatmap", return_value=result):
            dialog = AnomalyHeatmapDialog(
                algo_controller=_AlgoHarness(model),
                product_dir="demo_product",
                image_path="demo.png",
                algorithm="patchcore_lite",
                model_key="cam1__roi1",
                tool_name="ROI1",
                roi_label="roi1",
                ok_files=["ok1.png"],
            )
            dialog._refresh_heatmap()

        self.assertEqual(dialog.tabs.count(), 4)
        self.assertIn("整图 anomaly score: 0.0800", dialog.txt_summary.toPlainText())
        self.assertIn("patch max: 0.4000", dialog.txt_summary.toPlainText())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
