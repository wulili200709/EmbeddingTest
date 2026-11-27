from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from algorithms.image_io import imread, imwrite
from services import lens_correction as lens
from ui.debug.tool_page.lens_correction_flow import prepare_sample, refresh_sample_paths
from ui.debug.tool_page.training_assets import ensure_training_image_local


class LensCorrectionDebugTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        role_patch = patch("services.lens_camera_roles.resolve_correction_role", side_effect=lambda _root, role: role)
        role_patch.start()
        self.addCleanup(role_patch.stop)
        self.profile = self.root / "new_profile.json"
        self.profile.write_text(json.dumps({
            "schema": "lc-lens-calibration-v1", "image_size": [96, 72],
            "camera_matrix": [[100., 0., 47.5], [0., 99., 35.5], [0., 0., 1.]],
            "dist_coeffs": [-0.18, 0.03, 0.001, -0.002, 0.], "pattern": "chessboard",
            "rms": 0.2, "views": [], "warnings": [],
        }), encoding="utf-8")
        self.image = np.random.default_rng(92).integers(0, 256, (72, 96, 3), np.uint8)
        self.raw = str(self.root / "cam1_test.bmp")
        imwrite(self.raw, self.image)
        self.owner = SimpleNamespace(
            session=SimpleNamespace(product_dir=str(self.root)), current_camera_role=lambda: "cam1",
            train_files=[self.raw], test_files=[], ok_files=[], ng_files=[],
            _sample_roi_annotations_by_path={self.raw: {"cam1::roi1": "OK"}},
        )

    def enable(self):
        lens.save_config(self.root, "cam1", enabled=True, profile_path=self.profile)

    def test_disabled_keeps_original_paths_and_annotations(self):
        with patch.object(lens, "_cached", side_effect=AssertionError("map built")):
            refresh_sample_paths(self.owner)
        self.assertEqual(self.owner.train_files, [self.raw])
        self.assertEqual(self.owner._sample_roi_annotations_by_path, {self.raw: {"cam1::roi1": "OK"}})
        self.assertFalse((self.root / "lens_correction_settings.json").exists())

    def test_disabled_preserves_preexisting_manual_demo_inputs(self):
        old = str(self.root / "cam1_legacy_export.png")
        imwrite(old, self.image)
        Path(old).with_suffix(".undistortion.json").write_text(json.dumps({
            "already_undistorted": True, "source_image": self.raw,
            "profile": json.loads(self.profile.read_text()), "alpha": 1.,
        }), encoding="utf-8")
        self.assertEqual(prepare_sample(self.owner, old), old)

    def test_switch_on_off_preserves_raw_geometry_and_restores_paths(self):
        original_json = Path(self.raw).with_suffix(".json")
        original_json.write_text('{"shapes":[{"label":"roi1","points":[[1,2],[3,4]]}]}', encoding="utf-8")
        before = original_json.read_bytes()
        self.enable()
        refresh_sample_paths(self.owner)
        corrected = self.owner.train_files[0]
        self.assertNotEqual(corrected, self.raw)
        self.assertFalse(Path(corrected).with_suffix(".json").exists())
        self.assertEqual(self.owner._sample_roi_annotations_by_path[corrected], {"cam1::roi1": "OK"})
        self.assertEqual(original_json.read_bytes(), before)
        lens.save_config(self.root, "cam1", enabled=False)
        refresh_sample_paths(self.owner)
        self.assertEqual(self.owner.train_files, [self.raw])
        self.assertEqual(original_json.read_bytes(), before)

    def test_repeated_refresh_uses_cache_and_new_parameters_get_new_pixels(self):
        self.enable()
        first = prepare_sample(self.owner, self.raw)
        with patch.object(lens, "prepare_path", side_effect=AssertionError("decoded again")):
            self.assertEqual(prepare_sample(self.owner, self.raw), first)
        payload = json.loads(self.profile.read_text())
        payload["dist_coeffs"][0] = -0.34
        self.profile.write_text(json.dumps(payload))
        self.enable()
        second = prepare_sample(self.owner, first)
        self.assertNotEqual(first, second)
        self.assertEqual(lens.source_image_path(second), self.raw)
        self.assertFalse(np.array_equal(imread(first), imread(second)))

    def test_bad_resolution_stays_visible_and_is_rejected_on_open(self):
        self.enable()
        bad = str(self.root / "cam1_bad.bmp")
        imwrite(bad, np.zeros((32, 40, 3), np.uint8))
        self.owner.test_files = [bad]
        refresh_sample_paths(self.owner)
        self.assertEqual(self.owner.test_files, [bad])
        self.assertIn(bad, self.owner._lens_sample_errors)
        with self.assertRaises(lens.LensCorrectionError):
            prepare_sample(self.owner, bad)

    def test_training_copy_keeps_corrected_metadata(self):
        self.enable()
        corrected = lens.prepare_path(self.raw, self.root, "cam1")
        destination = self.root / "another_product"
        copied = ensure_training_image_local(corrected, product_dir=destination)
        self.assertEqual(
            Path(corrected).with_suffix(".undistortion.json").read_bytes(),
            Path(copied).with_suffix(".undistortion.json").read_bytes(),
        )
        self.assertEqual(ensure_training_image_local(corrected, product_dir=destination), copied)

    def test_auto_roi_only_missing_checks_current_corrected_annotations(self):
        from application.auto_roi_service import run_auto_roi_batch

        self.enable()
        corrected = prepare_sample(self.owner, self.raw)
        Path(corrected).with_suffix(".json").write_text(json.dumps({
            "shapes": [{"label": "roi1", "shape_type": "rectangle", "points": [[1, 2], [5, 6]]}],
        }))
        with patch("application.auto_roi_service.shape_locator.autogen_roi_json_from_shape_timed") as run:
            result = run_auto_roi_batch(
                paths=[self.raw], method="shape", ref_image=corrected, product_dir=str(self.root),
                camera_role="cam1", labels=["roi1"], only_missing=True,
            )
        run.assert_not_called()
        self.assertTrue(result["no_work"])
        self.assertFalse(Path(self.raw).with_suffix(".json").exists())

    def test_optional_lite_input_guard_runs_before_remap(self):
        self.enable()
        self.owner._validate_lens_input = MagicMock(side_effect=ValueError("Need full camera image"))
        with patch.object(lens, "prepare_path") as prepare:
            with self.assertRaisesRegex(ValueError, "full camera image"):
                prepare_sample(self.owner, self.raw)
        prepare.assert_not_called()

    def test_debug_inspection_passes_once_corrected_pixels_and_records_ct(self):
        from ui.debug.tool_page.test_execution_controller import TestExecutionController

        self.enable()
        cached = prepare_sample(self.owner, self.raw)
        owner = MagicMock()
        owner.session = self.owner.session
        owner.current_camera_role.return_value = "cam1"
        owner.inspection_items = [SimpleNamespace(enabled=True, camera_id="cam1", model_source="independent")]
        owner._lens_sample_path_cache = {}
        controller = TestExecutionController(owner)
        controller._show_runtime_roi_shapes = MagicMock()
        response = SimpleNamespace(match_ms=1., infer_ms=2., raw_row={}, item_results=[], result="OK", roi_shapes=())
        with patch("ui.debug.tool_page.test_execution_controller.ProductRuntimeContext"), \
             patch("ui.debug.tool_page.test_execution_controller.InspectionExecutor") as factory:
            factory.return_value.execute.return_value = response
            result = controller.execute_image(cached)
        request = factory.return_value.execute.call_args.args[0]
        self.assertTrue(request.image_prepared)
        np.testing.assert_array_equal(request.image_bgr, imread(cached))
        self.assertGreaterEqual(request.correction_ms, 0.)
        self.assertEqual(result["correction_ms"], request.correction_ms)
        self.assertIn("remap=", result["status_text"])

    def test_disabled_orphan_corrected_export_rejected_before_inspection(self):
        from ui.debug.tool_page.test_execution_controller import TestExecutionController

        self.enable()
        exported = str(self.root / "cam1_export.png")
        imwrite(exported, self.image)
        lens.mark_corrected_path(exported, self.root, "cam1", profile_id=lens.active_signature(self.root, "cam1"))
        lens.save_config(self.root, "cam1", enabled=False)
        owner = MagicMock()
        owner.session = self.owner.session
        owner.current_camera_role.return_value = "cam1"
        owner.inspection_items = [SimpleNamespace(enabled=True, camera_id="cam1", model_source="independent")]
        controller = TestExecutionController(owner)
        with patch("ui.debug.tool_page.test_execution_controller.ProductRuntimeContext"), \
             patch("ui.debug.tool_page.test_execution_controller.InspectionExecutor") as factory:
            with self.assertRaises(lens.LensCorrectionError):
                controller.execute_image(exported)
        factory.return_value.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
