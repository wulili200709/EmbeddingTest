from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from algorithms.image_io import imwrite
from application.algorithm_controller import AlgorithmController
from common.model_library import ModelLibrary
from domain import InspectionItem
from services import lens_correction as lens


class LensModelIsolationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "corrected-product"
        self.raw_root = Path(self.directory.name) / "raw-product"
        self.root.mkdir()
        self.raw_root.mkdir()
        self.params = self.root / "product_params.json"
        self.params.write_text("{}", encoding="utf-8")
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps({
            "schema": "lc-lens-calibration-v1", "image_size": [96, 72],
            "camera_matrix": [[100., 0., 47.5], [0., 99., 35.5], [0., 0., 1.]],
            "dist_coeffs": [-0.18, 0.03, 0.001, -0.002, 0.0], "pattern": "chessboard",
            "rms": 0.2, "views": [], "warnings": [],
        }), encoding="utf-8")
        role_patch = patch("services.lens_camera_roles.resolve_correction_role", side_effect=lambda _root, role: role)
        role_patch.start()
        self.addCleanup(role_patch.stop)
        self.controller = AlgorithmController()
        self.controller.load_params(str(self.params))
        self.controller.model_library = ModelLibrary(Path(self.directory.name) / "library")
        self.key = "cam1__roi1"

    def enable(self, alpha=1):
        return lens.save_config(self.root, "cam1", enabled=True, profile_path=self.profile, camera_serial="SERIAL-A", alpha=alpha)

    def prepared(self, name, intensity):
        source = self.root / f"{name}.png"
        imwrite(source, np.full((72, 96, 3), intensity, np.uint8))
        result = Path(lens.prepare_path(source, self.root, "cam1"))
        result.with_suffix(".json").write_text(json.dumps({
            "imagePath": result.name, "imageWidth": 96, "imageHeight": 72,
            "shapes": [{"label": "roi1", "shape_type": "rectangle", "points": [[20, 20], [40, 40]], "flags": {}}],
        }), encoding="utf-8")
        return str(result)

    def test_disabled_paths_match_legacy_and_explicit_product_never_uses_stale_root(self):
        self.assertEqual(self.controller.tool_model_key(self.key), self.key)
        self.assertEqual(Path(self.controller.embedding_model_path("b0", str(self.root), model_key=self.key)).name,
                         "cam1__roi1_register_model_b0.npz")
        raw_yolo = self.controller.yolo_model_paths(str(self.root), model_key=self.key)
        self.enable()
        self.assertIn("__lens_", self.controller.tool_model_key(self.key))
        self.assertNotIn("__lens_", self.controller.embedding_model_path("b0", str(self.raw_root), model_key=self.key))
        self.assertNotIn("__lens_", str(self.controller.yolo_model_paths(str(self.raw_root), model_key=self.key).onnx))
        lens.save_config(self.root, "cam1", enabled=False)
        self.assertEqual(self.controller.yolo_model_paths(str(self.root), model_key=self.key), raw_yolo)

    def test_keys_are_idempotent_and_replace_versions(self):
        first = self.enable()
        key = self.controller.tool_model_key(self.key)
        self.assertEqual(key, f"{self.key}__lens_{first.profile_id}")
        self.assertEqual(self.controller.tool_model_key(key), key)
        second = self.enable(alpha=0.6)
        changed = self.controller.tool_model_key(key)
        self.assertEqual(changed, f"{self.key}__lens_{second.profile_id}")
        self.assertEqual(changed.count("__lens_"), 1)
        self.assertEqual(self.controller.tool_model_key("cam2__roi1"), "cam2__roi1")
        lens.save_config(self.root, "cam1", enabled=False)
        self.assertEqual(self.controller.tool_model_key(changed), self.key)

    def test_corrected_embedding_never_falls_back_to_raw_keyed_or_unkeyed_model(self):
        (self.root / "cam1__roi1_register_model_b0.npz").write_bytes(b"raw keyed")
        (self.root / "register_model_b0.npz").write_bytes(b"raw fallback")
        self.enable()
        with patch("application.algorithm_controller.qr_core", new=SimpleNamespace(load_register_model_npz=Mock())) as backend:
            result, _ = self.controller.load_model_for_algorithm("b0", str(self.root), model_key=self.key)
            self.assertIsNone(result)
            backend.load_register_model_npz.assert_not_called()
        corrected = Path(self.controller.embedding_model_path("b0", str(self.root), model_key=self.key))
        corrected.write_bytes(b"new corrected model")
        with patch("application.algorithm_controller.qr_core", new=SimpleNamespace(load_register_model_npz=Mock())) as backend:
            backend.load_register_model_npz.return_value = SimpleNamespace(backbone="efficientnet_b0")
            result, _ = self.controller.load_model_for_algorithm("b0", str(self.root), model_key=self.key)
            self.assertIsNotNone(result)
            self.assertEqual(Path(backend.load_register_model_npz.call_args.args[0]), corrected)
        self.controller.model = SimpleNamespace(backbone="efficientnet_b0", label_name="roi1")
        self.controller._loaded_embedding_model_key = ("", "")
        self.assertFalse(self.controller._loaded_embedding_matches("efficientnet_b0", labels=["roi1"], model_key=self.key))

    def test_raw_second_camera_fallback_is_not_redirected_to_corrected_first_camera(self):
        raw = self.root / "register_model_b0.npz"
        raw.write_bytes(b"raw fallback")
        self.enable()
        corrected_default = Path(self.controller.embedding_model_path("b0", str(self.root)))
        corrected_default.write_bytes(b"cam1 corrected fallback")
        selected = self.controller._local_embedding_file("b0", str(self.root), "cam2__roi1")
        self.assertEqual(Path(selected), raw)

    def test_traditional_and_yolo_versions_are_isolated(self):
        raw = {"threshold": 40.0}
        self.controller.product_params.traditional_models["meanintensity"] = raw
        self.assertIs(self.controller.get_traditional_model_dict("meanintensity", model_key=self.key), raw)
        config = self.enable()
        self.assertIsNone(self.controller.get_traditional_model_dict("meanintensity", model_key=self.key))
        key = self.controller.traditional_model_storage_key("meanintensity", model_key=self.key)
        self.assertIn(config.profile_id, key)
        self.controller.product_params.traditional_models[key] = {"threshold": 80.0}
        self.assertEqual(self.controller.get_traditional_model_dict("meanintensity", model_key=self.key)["threshold"], 80.)
        self.assertIn(config.profile_id, str(self.controller.yolo_model_paths(str(self.root), model_key=self.key).onnx))
        self.assertIn(config.profile_id, self.controller.embedding_cache_dir("b0", str(self.root), model_key=self.key))

    def test_training_rejects_raw_images_before_any_algorithm_runs(self):
        self.enable()
        raw = self.root / "raw.png"
        imwrite(raw, np.full((72, 96, 3), 80, np.uint8))
        for algorithm in ("b0", "yolov8n_cls", "meanintensity"):
            with self.subTest(algorithm=algorithm), \
                 patch("application.algorithm_controller.train_yolo_classifier") as yolo, \
                 patch("application.algorithm_controller.qr_core", new=SimpleNamespace(get_device=Mock())) as backend, \
                 patch("application.algorithm_controller.train_threshold_model") as threshold:
                with self.assertRaises(lens.LensCorrectionError):
                    self.controller.train([str(raw)], [], algorithm=algorithm, product_dir=str(self.root),
                                          label_names=["roi1"], model_key=self.key)
                yolo.assert_not_called()
                backend.get_device.assert_not_called()
                threshold.assert_not_called()

    def test_corrected_training_and_publication_record_coordinate_version(self):
        config = self.enable()
        ok = self.prepared("ok", 40)
        ng = self.prepared("ng", 180)
        result = self.controller.train([ok], [ng], algorithm="meanintensity", product_dir=str(self.root),
                                       label_names=["roi1"], model_key=self.key)
        self.assertFalse(result.is_embedding)
        key = self.controller.traditional_model_storage_key("meanintensity", model_key=self.key)
        self.assertIn(config.profile_id, key)
        self.assertIn(key, self.controller.product_params.traditional_models)
        self.assertNotIn("meanintensity", self.controller.product_params.traditional_models)
        item = InspectionItem("roi1", "ROI 1", "cam1", "roi1", "meanintensity")
        published = self.controller.publish_model_to_library(item, str(self.root), "Corrected threshold")
        manifest = self.controller.model_library.resolve(published["model_id"], published["version"])
        self.assertEqual(manifest["source"]["lens_correction"]["profile_id"], config.profile_id)
        shared = InspectionItem("roi2", "ROI 2", "cam1", "roi2", "meanintensity", model_source="library",
                                library_model_id=published["model_id"], library_version=published["version"])
        self.assertTrue(self.controller.describe_model_binding(shared, str(self.root))["ready"])
        lens.save_config(self.root, "cam1", enabled=False)
        self.assertFalse(self.controller.describe_model_binding(shared, str(self.root))["ready"])

    def test_library_without_coordinate_metadata_is_explicitly_rejected(self):
        manifest = {"algorithm": "meanintensity", "kind": "traditional", "source": {}, "traditional_model": {}}
        self.enable()
        key = "cam1__library__" + "a" * 32 + "__v000001"
        with patch.object(self.controller.model_library, "resolve", return_value=manifest):
            with self.assertRaisesRegex(lens.LensCorrectionError, "缺少标定版本"):
                self.controller.get_traditional_model_dict("meanintensity", model_key=key)
            self.controller._active_product_dir = str(self.raw_root)
            with self.assertRaises(lens.LensCorrectionError):
                self.controller.load_model_for_algorithm("meanintensity", str(self.root), model_key=key)


if __name__ == "__main__":
    unittest.main()
