from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
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
            "b0",
            "C:/demo/product",
            model_key="cam1__roi1",
        )
        self.assertTrue(path.endswith("cam1__roi1_register_model_lt01.npz"))

    def test_load_embedding_model_reuses_unchanged_model_file(self) -> None:
        controller = AlgorithmController()
        controller.set_learning_backbone("b0")
        fake_model = SimpleNamespace(
            backbone="b0",
            device="cpu",
            score_mode="proto",
            margin=0.1,
            topk=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            product_dir = Path(tmpdir)
            model_path = Path(
                controller.embedding_model_path(
                    "b0",
                    str(product_dir),
                    model_key="cam1__roi1",
                )
            )
            model_path.write_bytes(b"model-v1")

            load_mock = mock.Mock(return_value=fake_model)
            with mock.patch.dict(
                algorithm_controller.qr_core.__dict__,
                {"load_register_model_npz": load_mock},
            ):
                first_model, _first_msg = controller.load_model_for_algorithm(
                    "b0",
                    str(product_dir),
                    model_key="cam1__roi1",
                )
                controller.product_params.margin = 0.6
                second_model, _second_msg = controller.load_model_for_algorithm(
                    "b0",
                    str(product_dir),
                    model_key="cam1__roi1",
                )

        self.assertIs(first_model, fake_model)
        self.assertIs(second_model, fake_model)
        self.assertEqual(load_mock.call_count, 1)
        self.assertAlmostEqual(fake_model.margin, 0.6)

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

    def test_predict_image_prefers_requested_roi_over_group_model_roi_label(self) -> None:
        controller = AlgorithmController()
        controller.product_params.traditional_models = {
            "meanintensity::cam1__hole": {
                "algorithm": "meanintensity",
                "threshold": 0.40,
                "ok_when": "greater_equal",
                "ok_mean": 0.7,
                "ng_mean": 0.2,
                "accuracy": 1.0,
                "roi_label": "hole",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.png"
            image_path.write_bytes(b"demo")

            with mock.patch.object(
                algorithm_controller,
                "compute_roi_metrics",
                return_value={"meanintensity": 0.5},
            ) as compute_mock, mock.patch.object(
                algorithm_controller,
                "metric_value",
                side_effect=lambda metrics, _algorithm: float(metrics["meanintensity"]),
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "labelme_json_of_image",
                return_value="demo.json",
                create=True,
            ):
                controller.predict_image(
                    str(image_path),
                    labels=["roi7"],
                    algorithm_override="meanintensity",
                    model_key_override="cam1__hole",
                )

        self.assertEqual(compute_mock.call_args.kwargs["preferred_label"], "roi7")

    def test_clear_training_output_removes_traditional_scoped_model(self) -> None:
        controller = AlgorithmController()
        controller.product_params.traditional_models = {
            "meanintensity::cam1__roi1": {"threshold": 0.4},
            "meanintensity::cam1__roi2": {"threshold": 0.8},
        }

        changed = controller.clear_training_output(
            "meanintensity",
            "C:/demo/product",
            model_key="cam1__roi1",
        )

        self.assertTrue(changed)
        self.assertNotIn("meanintensity::cam1__roi1", controller.product_params.traditional_models)
        self.assertIn("meanintensity::cam1__roi2", controller.product_params.traditional_models)

    def test_clear_obsolete_traditional_models_prunes_stale_camera_keys(self) -> None:
        controller = AlgorithmController()
        controller.product_params.traditional_models = {
            "meanstd::cam1__roi1": {"threshold": 0.4},
            "meanstd::cam1__roi2": {"threshold": 0.5},
            "meanstd::cam1__roi24": {"threshold": 0.9},
            "meanstd::cam2__roi1": {"threshold": 0.6},
            "meanstd": {"threshold": 0.7},
        }

        changed = controller.clear_obsolete_traditional_models(
            camera_role="cam1",
            valid_model_keys_by_algorithm={"meanstd": {"cam1__roi1", "cam1__roi2"}},
        )

        self.assertTrue(changed)
        self.assertIn("meanstd::cam1__roi1", controller.product_params.traditional_models)
        self.assertIn("meanstd::cam1__roi2", controller.product_params.traditional_models)
        self.assertNotIn("meanstd::cam1__roi24", controller.product_params.traditional_models)
        self.assertIn("meanstd::cam2__roi1", controller.product_params.traditional_models)
        self.assertNotIn("meanstd", controller.product_params.traditional_models)


    def test_anomaly_model_path_uses_tool_key(self) -> None:
        controller = AlgorithmController()
        path = controller.anomaly_model_path(
            "patchcore_lite",
            "C:/demo/product",
            model_key="cam1__roi1",
        )
        self.assertTrue(path.endswith("cam1__roi1_anomaly_model_patchcore_lite.npz"))

    def test_train_stores_anomaly_model_by_tool_key(self) -> None:
        controller = AlgorithmController()
        fake_model = SimpleNamespace(
            algorithm="patchcore_lite",
            backbone="b0",
            threshold=0.12,
            topk=5,
            device="cpu",
        )

        with mock.patch.object(
            algorithm_controller,
            "train_patchcore_lite_model",
            return_value=fake_model,
        ) as train_mock, mock.patch.object(
            algorithm_controller,
            "save_anomaly_model_npz",
        ) as save_mock:
            result = controller.train(
                ["ok.png"],
                [],
                algorithm="patchcore_lite",
                product_dir="C:/demo/product",
                label_names=["roi1"],
                model_key="cam1__roi1",
            )

        self.assertTrue(result.is_embedding)
        self.assertEqual(controller.product_params.margin, fake_model.threshold)
        train_mock.assert_called_once()
        save_mock.assert_called_once()
        self.assertTrue(result.saved_model_path.endswith("cam1__roi1_anomaly_model_patchcore_lite.npz"))

    def test_predict_image_uses_tool_scoped_anomaly_model(self) -> None:
        controller = AlgorithmController()
        controller.model = SimpleNamespace(
            algorithm="patchcore_lite",
            backbone="b0",
            device="cpu",
            topk=3,
            threshold=0.25,
            label_name="roi1",
            effective_label_names=lambda: ["roi1"],
        )
        controller._loaded_embedding_model_key = ("patchcore_lite", "cam1__roi1")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.png"
            image_path.write_bytes(b"demo")

            with mock.patch.object(
                controller,
                "get_feat_net",
                return_value=object(),
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "embed_one",
                return_value=np.asarray([1.0, 0.0], dtype=np.float32),
                create=True,
            ), mock.patch.object(
                algorithm_controller,
                "predict_one_with_anomaly_model",
                return_value=("OK", 0.05, 0.20),
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "labelme_json_of_image",
                return_value="demo.json",
                create=True,
            ):
                result = controller.predict_image(
                    str(image_path),
                    labels=["roi1"],
                    roi=(0, 0, 1, 1),
                    algorithm_override="patchcore_lite",
                    model_key_override="cam1__roi1",
                )

        self.assertEqual(result.pred, "OK")
        self.assertAlmostEqual(result.value, 0.20)
        self.assertAlmostEqual(result.threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
