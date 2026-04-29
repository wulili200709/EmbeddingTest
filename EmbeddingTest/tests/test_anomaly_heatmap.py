from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from algorithms.anomaly import AnomalyModel
from algorithms import anomaly_heatmap


class AnomalyHeatmapTest(unittest.TestCase):
    def test_score_patch_embeddings_highlights_outlier_patch(self) -> None:
        patch_bank = np.asarray(
            [
                [1.0, 0.0],
                [0.98, 0.02],
            ],
            dtype=np.float32,
        )
        patch_embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        scores = anomaly_heatmap.score_patch_embeddings_against_bank(
            patch_embeddings,
            patch_bank,
            topk=1,
        )

        self.assertEqual(scores.shape, (2,))
        self.assertLess(scores[0], 0.05)
        self.assertGreater(scores[1], 0.95)

    def test_generate_anomaly_heatmap_returns_overlay_images(self) -> None:
        model = AnomalyModel(
            algorithm="patchcore_lite",
            backbone="b0",
            threshold=0.20,
            topk=1,
            label_name="roi1",
            label_names=["roi1"],
            device="cpu",
            ok_bank=np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        full_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        roi_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        feature_map = np.asarray(
            [
                [[2.0, 0.0], [2.0, 0.0]],
                [[0.0, 2.0], [0.0, 2.0]],
            ],
            dtype=np.float32,
        )

        with mock.patch.object(anomaly_heatmap, "imread", return_value=full_bgr), mock.patch.object(
            anomaly_heatmap,
            "_extract_roi_feature_map",
            return_value=(feature_map, roi_rgb, (2, 3, 4, 4)),
        ), mock.patch.object(
            anomaly_heatmap,
            "predict_one_with_anomaly_model",
            return_value=("NG", -0.05, 0.25),
        ):
            result = anomaly_heatmap.generate_anomaly_heatmap(
                "demo.png",
                ok_files=["ok_1.png"],
                model=model,
                label_name="roi1",
                feat_net=object(),
                patch_bank=np.asarray([[1.0, 0.0]], dtype=np.float32),
                device="cpu",
            )

        self.assertEqual(result.roi_xywh, (2, 3, 4, 4))
        self.assertEqual(result.pred, "NG")
        self.assertEqual(result.overlay_bgr.shape, (10, 10, 3))
        self.assertEqual(result.roi_overlay_bgr.shape, (4, 4, 3))
        self.assertEqual(result.roi_heatmap_bgr.shape, (4, 4, 3))
        self.assertGreater(result.patch_max, result.patch_mean)
        self.assertEqual(result.pred, "NG")
        self.assertAlmostEqual(result.score, 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
