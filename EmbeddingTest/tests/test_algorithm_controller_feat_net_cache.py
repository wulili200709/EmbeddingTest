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
    def test_get_feat_net_reuses_cached_backbone_for_same_device(self) -> None:
        controller = AlgorithmController()
        load_calls: list[tuple[str, str | None]] = []

        def _fake_load_backbone(name: str, device: str | None = None):
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
            feat_net_1 = controller.get_feat_net("b0", "cpu")
            feat_net_2 = controller.get_feat_net("b0", "cpu")

        self.assertIs(feat_net_1, feat_net_2)
        self.assertEqual(load_calls, [("b0", "cpu")])

    def test_predict_image_uses_cached_feat_net_when_not_provided(self) -> None:
        controller = AlgorithmController()
        controller.product_params.algorithm = "b0"
        controller.model = SimpleNamespace(
            backbone="b0",
            device="cpu",
            score_mode="proto",
            margin=0.02,
            topk=3,
        )
        load_calls: list[tuple[str, str | None]] = []

        def _fake_load_backbone(name: str, device: str | None = None):
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

        self.assertEqual(load_calls, [("b0", "cpu")])


if __name__ == "__main__":
    unittest.main()
