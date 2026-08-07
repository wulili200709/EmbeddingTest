from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import application.algorithm_controller as algorithm_controller
from application.algorithm_controller import AlgorithmController


class AlgorithmControllerFeatNetCacheTest(unittest.TestCase):
    def test_successful_embedding_training_clears_cached_models(self) -> None:
        controller = AlgorithmController()
        controller.product_params.algorithm = "efficientnet_b0"
        controller.product_params.learning_backbone = "efficientnet_b0"
        controller._embedding_model_cache["old-product-model"] = ((1, 1), object())
        trained_model = SimpleNamespace(backbone="efficientnet_b0")

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            controller,
            "get_feat_net",
            return_value=object(),
        ), mock.patch.dict(
            algorithm_controller.qr_core.__dict__,
            {
                "get_device": mock.Mock(return_value="cpu"),
                "train_register_model": mock.Mock(return_value=trained_model),
                "save_register_model_npz": mock.Mock(),
            },
        ):
            result = controller.train(
                ["ok.png"],
                ["ng.png"],
                algorithm="efficientnet_b0",
                product_dir=tmpdir,
                label_names=["roi1"],
                model_key="cam1__roi1",
            )

        self.assertIs(result.model, trained_model)
        self.assertEqual(controller._embedding_model_cache, {})

    def test_load_model_reuses_cached_npz_until_file_changes(self) -> None:
        controller = AlgorithmController()
        controller.product_params.algorithm = "efficientnet_b0"
        loaded_models: list[object] = []

        def _fake_load_model(_path: str):
            model = SimpleNamespace(backbone="efficientnet_b0")
            loaded_models.append(model)
            return model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "cam1__roi1_register_model_b0.npz"
            model_path.write_bytes(b"model-v1")
            with mock.patch.dict(
                algorithm_controller.qr_core.__dict__,
                {"load_register_model_npz": mock.Mock(side_effect=_fake_load_model)},
            ):
                first, _message = controller.load_model_for_algorithm(
                    "efficientnet_b0",
                    tmpdir,
                    model_key="cam1__roi1",
                )
                second, _message = controller.load_model_for_algorithm(
                    "efficientnet_b0",
                    tmpdir,
                    model_key="cam1__roi1",
                )
                model_path.write_bytes(b"model-v2-with-new-size")
                third, _message = controller.load_model_for_algorithm(
                    "efficientnet_b0",
                    tmpdir,
                    model_key="cam1__roi1",
                )
                params_path = Path(tmpdir) / "product_params.json"
                params_path.write_text("{}", encoding="utf-8")
                controller.load_params(str(params_path))
                fourth, _message = controller.load_model_for_algorithm(
                    "efficientnet_b0",
                    tmpdir,
                    model_key="cam1__roi1",
                )

        self.assertIs(first, second)
        self.assertIsNot(second, third)
        self.assertIsNot(third, fourth)
        self.assertEqual(len(loaded_models), 3)

    def test_get_feat_net_reuses_cached_backbone_for_same_device(self) -> None:
        controller = AlgorithmController()
        load_calls: list[tuple[str, str | None]] = []

        def _fake_load_backbone(name: str, device: str | None = None, **_kwargs):
            load_calls.append((name, device))
            return object(), 1280

        with mock.patch.object(
            algorithm_controller.qr_core,
            "load_backbone",
            side_effect=_fake_load_backbone,
            create=True,
        ), mock.patch.object(
            algorithm_controller.qr_core,
            "get_device",
            return_value="cpu",
            create=True,
        ):
            feat_net_1 = controller.get_feat_net("efficientnet_b0", "cpu")
            feat_net_2 = controller.get_feat_net("efficientnet_b0", "cpu")

        self.assertIs(feat_net_1, feat_net_2)
        self.assertEqual(load_calls, [("efficientnet_b0", "cpu")])

    def test_predict_image_uses_cached_feat_net_when_not_provided(self) -> None:
        controller = AlgorithmController()
        controller.product_params.algorithm = "efficientnet_b0"
        controller.model = SimpleNamespace(
            backbone="efficientnet_b0",
            device="cpu",
            score_mode="proto",
            margin=0.02,
            topk=3,
        )
        load_calls: list[tuple[str, str | None]] = []

        def _fake_load_backbone(name: str, device: str | None = None, **_kwargs):
            load_calls.append((name, device))
            return object(), 1280

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.png"
            image_path.write_bytes(b"demo")

            with mock.patch.object(
                algorithm_controller.qr_core,
                "load_backbone",
                side_effect=_fake_load_backbone,
                create=True,
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "embed_many",
                return_value="embedding",
                create=True,
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "predict_one_with_model",
                return_value=("OK", 0.12, 0.88, 0.11),
                create=True,
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "labelme_json_of_image",
                return_value="demo.json",
                create=True,
            ):
                controller.predict_image(str(image_path), labels=["roi1", "roi2"])
                controller.predict_image(str(image_path), labels=["roi1", "roi2"])

        self.assertEqual(load_calls, [("efficientnet_b0", "cpu")])

    def test_train_embedding_passes_cached_feat_net_to_core_training(self) -> None:
        controller = AlgorithmController()
        controller.product_params.score_mode = "proto"
        controller.product_params.learning_backbone = "efficientnet_b0"
        controller.product_params.margin = 0.02
        controller.product_params.topk = 3
        feat_net = object()
        train_calls: list[dict[str, object]] = []

        def _fake_train_register_model(*args, **kwargs):
            train_calls.append(dict(kwargs))
            return SimpleNamespace(backbone="efficientnet_b0")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                controller,
                "get_feat_net",
                return_value=feat_net,
            ) as get_feat_net, mock.patch.object(
                algorithm_controller.qr_core,
                "get_device",
                return_value="cpu",
                create=True,
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "train_register_model",
                side_effect=_fake_train_register_model,
                create=True,
            ), mock.patch.object(
                algorithm_controller.qr_core,
                "save_register_model_npz",
                return_value=None,
                create=True,
            ):
                controller.train(
                    ["ok.png"],
                    ["ng.png"],
                    algorithm="efficientnet_b0",
                    product_dir=tmpdir,
                    label_names=["roi1"],
                    model_key="cam1__roi1",
                )

        get_feat_net.assert_called_once_with("efficientnet_b0", "cpu")
        self.assertEqual(train_calls[0]["feat_net"], feat_net)
        self.assertEqual(train_calls[0]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
