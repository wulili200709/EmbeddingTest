from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import application.algorithm_controller as algorithm_controller
from algorithms.traditional import TraditionalThresholdModel
from application.algorithm_controller import AlgorithmController


class AlgorithmControllerToolScopedModelsTest(unittest.TestCase):
    def test_embedding_model_path_uses_tool_key(self) -> None:
        controller = AlgorithmController()
        path = controller.embedding_model_path(
            "efficientnet_b0",
            "C:/demo/product",
            model_key="cam1__roi1",
        )
        self.assertTrue(path.endswith("cam1__roi1_register_model_lt01.npz"))

    def test_train_stores_traditional_model_by_tool_key(self) -> None:
        controller = AlgorithmController()
        fake_model = TraditionalThresholdModel(
            algorithm="meanintensity",
            threshold=0.5,
            ok_when="greater_equal",
            ok_mean=0.7,
            ng_mean=0.3,
            accuracy=1.0,
            roi_label="roi1",
        )

        with mock.patch.object(
            algorithm_controller,
            "train_threshold_model",
            return_value=(fake_model, []),
        ):
            result = controller.train(
                ["ok.png"],
                ["ng.png"],
                algorithm="meanintensity",
                product_dir="C:/demo/product",
                label_names=["roi1"],
                model_key="cam1__roi1",
            )

        self.assertFalse(result.is_embedding)
        self.assertIn("meanintensity::cam1__roi1", controller.product_params.traditional_models)

    def test_predict_image_uses_tool_scoped_traditional_model(self) -> None:
        controller = AlgorithmController()
        controller.product_params.traditional_models = {
            "meanintensity::cam1__roi1": {
                "algorithm": "meanintensity",
                "threshold": 0.40,
                "ok_when": "greater_equal",
                "ok_mean": 0.7,
                "ng_mean": 0.2,
                "accuracy": 1.0,
                "roi_label": "roi1",
            },
            "meanintensity::cam1__roi2": {
                "algorithm": "meanintensity",
                "threshold": 0.80,
                "ok_when": "greater_equal",
                "ok_mean": 0.9,
                "ng_mean": 0.3,
                "accuracy": 1.0,
                "roi_label": "roi2",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.png"
            image_path.write_bytes(b"demo")

            with mock.patch.object(
                algorithm_controller,
                "compute_roi_metrics",
                return_value={"meanintensity": 0.5},
            ), mock.patch.object(
                algorithm_controller,
                "metric_value",
                side_effect=lambda metrics, _algorithm: float(metrics["meanintensity"]),
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "labelme_json_of_image",
                return_value="demo.json",
                create=True,
            ):
                roi1_result = controller.predict_image(
                    str(image_path),
                    labels=["roi1"],
                    algorithm_override="meanintensity",
                    model_key_override="cam1__roi1",
                )
                roi2_result = controller.predict_image(
                    str(image_path),
                    labels=["roi2"],
                    algorithm_override="meanintensity",
                    model_key_override="cam1__roi2",
                )

        self.assertEqual(roi1_result.pred, "OK")
        self.assertEqual(roi2_result.pred, "NG")


if __name__ == "__main__":
    unittest.main()
