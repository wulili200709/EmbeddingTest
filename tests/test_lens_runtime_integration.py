"""Synthetic integration checks: one correction, one coordinate space, reversible assets."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from algorithms.image_io import imwrite
from algorithms.lens_calibration import manual_profile, save_profile
from application.inspection_executor import InspectionExecutionRequest, InspectionExecutor
from application.runtime import execution
from application.runtime.preview_frame import export_runtime_preview_frame
from application.runtime_context import ProductRuntimeContext
from domain import InspectionItem, aggregate_runtime_outcome
from domain.inspection_items import save_inspection_items as save_raw_items
from ncc import locator as ncc_locator
from services import lens_correction, lens_inspection_items
from services.lens_camera_roles import resolve_correction_role
from shape.core import locator as shape_locator
from shape.core.recipe import ShapeRecipe


class Signal:
    def __init__(self):
        self.events = []

    def emit(self, *args):
        self.events.append(args)


class LensRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lens-runtime-test-")
        self.root = Path(self.temp.name)
        self.image = np.random.default_rng(12).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        self.routing = patch("services.lens_camera_roles._read_capture_config", return_value={})
        self.routing.start()

    def tearDown(self):
        self.routing.stop()
        self.temp.cleanup()

    def enable(self, role="cam1", coefficient=-0.2):
        path = self.root / f"profile-{role}-{coefficient}.json"
        save_profile(manual_profile((64, 48), coefficient), path)
        return lens_correction.save_config(self.root, role, enabled=True, profile_path=str(path))

    def runtime(self):
        return SimpleNamespace(
            _session=SimpleNamespace(product_dir=str(self.root)),
            _last_runtime_result=SimpleNamespace(task_id="synthetic-trigger"),
            _frame_lock=threading.RLock(), _inspect_lock=threading.RLock(),
            _last_preview_frames={}, _last_item_results_by_camera={},
            _runtime_context=SimpleNamespace(inspection_items=[]),
            _inspection_executor=SimpleNamespace(execute=Mock(return_value=SimpleNamespace(
                item_results=[], roi_shapes=(), measurements=(), result="OK", detail="",
                match_ms=4.0, infer_ms=6.0,
            ))),
            previewUpdated=Signal(), logAppended=Signal(),
        )

    def inspect(self, runtime, role="cam1", physical_role=""):
        frame = SimpleNamespace(camera_serial="", frame_num=7, host_timestamp=8)
        with patch("application.runtime.controller.frame_to_bgr_image", return_value=self.image):
            return execution._inspect_frame(runtime, role, frame, physical_role=physical_role)

    def test_disabled_frame_keeps_pixels_and_creates_no_configuration(self):
        runtime = self.runtime()
        outcome = self.inspect(runtime)
        request = runtime._inspection_executor.execute.call_args.args[0]
        self.assertIs(request.image_bgr, self.image)
        self.assertTrue(request.image_prepared)
        self.assertEqual(outcome.correction_ms, 0.0)
        self.assertEqual(list(self.root.iterdir()), [])
        np.testing.assert_array_equal(runtime._last_preview_frames["cam1"].image_bgr, self.image)

    def test_enabled_frame_is_corrected_once_and_shared_with_both_previews(self):
        config = self.enable()
        expected = lens_correction.prepare_array(self.image, self.root, "cam1").image
        runtime = self.runtime()
        with patch.object(lens_correction, "prepare_array", wraps=lens_correction.prepare_array) as prepare:
            outcome = self.inspect(runtime)
        prepare.assert_called_once()
        request = runtime._inspection_executor.execute.call_args.args[0]
        np.testing.assert_array_equal(request.image_bgr, expected)
        self.assertFalse(np.array_equal(expected, self.image))
        np.testing.assert_array_equal(runtime.previewUpdated.events[0][1].image_bgr, expected)
        np.testing.assert_array_equal(runtime._last_preview_frames["cam1"].image_bgr, expected)
        self.assertEqual(runtime._last_preview_frames["cam1"].correction_profile_id, config.profile_id)
        self.assertGreaterEqual(outcome.correction_ms, 0)
        self.assertEqual(request.correction_ms, outcome.correction_ms)

    def test_correction_failure_never_runs_detector_or_publishes_raw_preview(self):
        runtime = self.runtime()
        with patch.object(lens_correction, "prepare_array", side_effect=lens_correction.LensCorrectionError("size mismatch")):
            with self.assertRaisesRegex(lens_correction.LensCorrectionError, "size mismatch"):
                self.inspect(runtime)
        runtime._inspection_executor.execute.assert_not_called()
        self.assertEqual(runtime.previewUpdated.events, [])

    def test_live_executor_context_chain_does_not_remap_again_for_roi_batch(self):
        self.enable()
        item = InspectionItem("a", "a", "cam1", "roi1")
        context = SimpleNamespace(
            session=SimpleNamespace(product_dir=str(self.root)),
            algo=SimpleNamespace(is_learning_tool=lambda code: True),
            loc_method_for_role=lambda role: "none", load_embedding_model=lambda *a, **k: None,
        )
        context.predict_items_batch_from_frame = MethodType(ProductRuntimeContext.predict_items_batch_from_frame, context)
        runtime = self.runtime()
        runtime._runtime_context.inspection_items = [item]
        runtime._inspection_executor = InspectionExecutor(context)
        with patch.object(lens_correction, "prepare_array", wraps=lens_correction.prepare_array) as prepare, patch(
            "application.runtime_context._predict_learning_items_batch_rows_from_frame",
            return_value={item.model_key: {"pred": "OK", "match_ms": 0, "infer_ms": 2}},
        ) as predict:
            self.inspect(runtime)
        prepare.assert_called_once()
        np.testing.assert_array_equal(predict.call_args.kwargs["image_bgr"],
                                      runtime._last_preview_frames["cam1"].image_bgr)

    def test_two_cameras_keep_separate_maps_and_previews(self):
        self.enable("cam1", -0.2)
        self.enable("cam2", 0.12)
        runtime = self.runtime()
        self.inspect(runtime, "cam1")
        self.inspect(runtime, "cam2")
        for role in ("cam1", "cam2"):
            expected = lens_correction.prepare_array(self.image, self.root, role).image
            np.testing.assert_array_equal(runtime._last_preview_frames[role].image_bgr, expected)
        self.assertFalse(np.array_equal(runtime._last_preview_frames["cam1"].image_bgr,
                                       runtime._last_preview_frames["cam2"].image_bgr))

    def test_virtual_channel_uses_physical_profile_but_keeps_its_own_template_assets(self):
        config = self.enable("cam2", -0.2)
        routing = {"capture_mode": "flexible", "capture_channels": [
            {"role": "cam1", "physical_role": "cam2"}, {"role": "cam3", "physical_role": "cam2"}]}
        with patch("services.lens_camera_roles._read_capture_config", return_value=routing):
            runtime = self.runtime()
            self.inspect(runtime, "cam1", physical_role="cam2")
            self.assertEqual(runtime._last_preview_frames["cam1"].correction_profile_id, config.profile_id)
            self.assertEqual(lens_correction.active_signature(self.root, "cam1"),
                             lens_correction.active_signature(self.root, "cam3"))
            self.assertNotEqual(shape_locator.product_paths(str(self.root), "cam1").model_path,
                                shape_locator.product_paths(str(self.root), "cam3").model_path)

    def test_prepared_context_bypasses_second_remap_and_raw_context_does_one(self):
        context = SimpleNamespace(session=SimpleNamespace(product_dir=str(self.root)),
                                  algo=SimpleNamespace(), loc_method_for_role=lambda role: "none",
                                  load_embedding_model=lambda *a, **k: None)
        self.enable()
        with patch.object(lens_correction, "prepare_array", wraps=lens_correction.prepare_array) as prepare:
            ProductRuntimeContext.predict_items_batch_from_frame(context, self.image, camera_role="cam1", items=[])
            self.assertEqual(prepare.call_count, 1)
            ProductRuntimeContext.predict_items_batch_from_frame(
                context, self.image, camera_role="cam1", items=[], image_prepared=True)
            self.assertEqual(prepare.call_count, 1)

    def test_executor_forwards_prepared_flag_and_counts_correction_once(self):
        class Predictor:
            def predict_items_batch_from_frame(self, image, *, camera_role, items, image_prepared=False):
                self.prepared = image_prepared
                return SimpleNamespace(rows=[dict(pred="OK", match_ms=4.0, infer_ms=6.0)],
                                       roi_shapes=(), timing_breakdown={})
        predictor = Predictor()
        response = InspectionExecutor(predictor).execute(InspectionExecutionRequest(
            camera_id="cam1", image_bgr=self.image, image_prepared=True, correction_ms=7.0,
            items=[InspectionItem("a", "a", "cam1", "roi1")],
        ))
        self.assertTrue(predictor.prepared)
        self.assertEqual(response.total_ms, 17.0)
        self.assertEqual(response.raw_row["timing_breakdown"]["correction_ms"], 7.0)

    def test_shape_ncc_and_legacy_paths_are_reversible_and_never_fall_back(self):
        originals = []
        for locator in (shape_locator, ncc_locator):
            paths = locator.product_paths(str(self.root), "cam1")
            Path(paths.legacy_model_path).write_bytes(b"original-model")
            originals.append(paths)
        self.enable()
        for locator, old in zip((shape_locator, ncc_locator), originals):
            scoped = locator.product_paths(str(self.root), "cam1")
            self.assertIn("lens_assets", scoped.model_path)
            self.assertNotEqual(scoped.legacy_model_path, old.legacy_model_path)
            self.assertEqual(locator.resolved_model_path_for_product(str(self.root)), scoped.model_path)
        with self.assertRaises(lens_correction.LensCorrectionError):
            ncc_locator._validate_model_scope(originals[1].legacy_model_path, str(self.root), "cam1")
        lens_correction.save_config(self.root, "cam1", enabled=False)
        for locator, old in zip((shape_locator, ncc_locator), originals):
            self.assertEqual(locator.product_paths(str(self.root)), old)
            self.assertEqual(Path(old.legacy_model_path).read_bytes(), b"original-model")

    def test_export_keeps_captured_profile_even_after_switching_settings(self):
        first = self.enable()
        runtime = self.runtime()
        self.inspect(runtime)
        frame = runtime._last_preview_frames["cam1"]
        self.enable(coefficient=0.1)
        exported = export_runtime_preview_frame(frame, self.root / "exports")
        self.assertEqual(exported.correction_profile_id, first.profile_id)
        with self.assertRaises(lens_correction.LensCorrectionError):
            lens_correction.prepare_path(exported.source_path, self.root, "cam1")

    def test_ncc_source_must_belong_to_current_calibration(self):
        self.enable()
        raw = self.root / "ncc-source.png"
        self.assertTrue(imwrite(raw, self.image))
        model_path = ncc_locator.product_paths(str(self.root)).model_path
        with self.assertRaises(lens_correction.LensCorrectionError):
            ncc_locator._validate_source_image(SimpleNamespace(source_image_path=str(raw)),
                                               model_path, str(self.root), "cam1")
        prepared = lens_correction.prepare_path(str(raw), self.root, "cam1")
        ncc_locator._validate_source_image(SimpleNamespace(source_image_path=prepared),
                                           model_path, str(self.root), "cam1")

    def test_shape_path_localization_writes_only_corrected_coordinate_sidecar(self):
        self.enable()
        raw = self.root / "raw.png"
        self.assertTrue(imwrite(raw, self.image))
        prepared = lens_correction.prepare_path(str(raw), self.root, "cam1")
        paths = shape_locator.product_paths(str(self.root))
        Path(paths.role_dir).mkdir(parents=True)
        Path(paths.model_path).write_text("{}", encoding="utf-8")
        recipe = ShapeRecipe(reference_image=prepared)
        region = SimpleNamespace(source_shape_type="rectangle", bbox=(2, 3, 10, 8), label_name="roi1")
        run = SimpleNamespace(result=SimpleNamespace(regions=[region]), locate_ms=1.0)
        with patch.object(shape_locator, "load_recipe", return_value=recipe), patch.object(
            shape_locator._LOCATE_SERVICE, "locate", return_value=run
        ) as locate:
            result = shape_locator.autogen_roi_json_from_shape_timed(str(raw), prepared, str(self.root))
        self.assertEqual(Path(result.jpath), Path(prepared).with_suffix(".json"))
        self.assertFalse(raw.with_suffix(".json").exists())
        np.testing.assert_array_equal(locate.call_args.args[0],
                                      lens_correction.prepare_array(self.image, self.root, "cam1").image)
        with patch.object(shape_locator, "load_recipe", return_value=recipe), patch.object(
            shape_locator._LOCATE_SERVICE, "locate"
        ) as locate:
            with self.assertRaises(lens_correction.LensCorrectionError):
                shape_locator.autogen_roi_json_from_shape_timed(str(raw), str(raw), str(self.root))
            locate.assert_not_called()

    def test_item_configuration_new_version_starts_empty_and_off_restores_raw_bytes(self):
        path = str(self.root / "inspection_items.json")
        raw = InspectionItem("raw", "raw", "cam1", "roi1", params={"line": {"points": [1, 2]}})
        other = InspectionItem("other", "other", "cam2", "roi2")
        save_raw_items([raw, other], path)
        before = Path(path).read_bytes()
        self.enable()
        loaded = lens_inspection_items.load_inspection_items(path)
        self.assertEqual([item.item_id for item in loaded], ["other"])
        corrected = InspectionItem("corrected", "corrected", "cam1", "roi3")
        lens_inspection_items.save_inspection_items([*loaded, corrected], path)
        self.assertEqual(Path(path).read_bytes(), before)
        self.assertEqual({i.item_id for i in lens_inspection_items.load_inspection_items(path)}, {"other", "corrected"})
        lens_correction.save_config(self.root, "cam1", enabled=False)
        self.assertEqual([i.item_id for i in lens_inspection_items.load_inspection_items(path)], ["raw", "other"])
        self.assertEqual(Path(path).read_bytes(), before)

    def test_raw_mode_ignores_managed_session_reference_and_recovers_raw_recipe(self):
        self.enable()
        raw = self.root / "reference.png"
        self.assertTrue(imwrite(raw, self.image))
        corrected = lens_correction.prepare_path(str(raw), self.root, "cam1")
        lens_correction.save_config(self.root, "cam1", enabled=False)
        context = SimpleNamespace(session=SimpleNamespace(product_dir=str(self.root)), _ref_image=corrected)
        self.assertEqual(ProductRuntimeContext._reference_image(context, ShapeRecipe(reference_image=str(raw))), str(raw))

    def test_aggregated_camera_totals_include_correction_without_changing_wall_ct(self):
        outcome = SimpleNamespace(result="OK", capture_ms=2, correction_ms=3, match_ms=4, infer_ms=5)
        result = aggregate_runtime_outcome(product_name="p", recipe_name="r", items=[],
                                          active_roles=["cam1"], camera_outcomes={"cam1": outcome},
                                          final_result="OK", duration_ms=20)
        timing = result.timing_breakdown()
        self.assertEqual(timing["correction_ms"], 3)
        self.assertEqual(timing["cam1_total_ms"], 14)
        self.assertEqual(timing["duration_ms"], 20)


class LensCameraRoleTests(unittest.TestCase):
    def test_independent_and_multi_light_follow_acquisition_defaults(self):
        self.assertEqual(resolve_correction_role("unused", "cam2", capture_config={}), "cam2")
        self.assertEqual(resolve_correction_role("unused", "cam3", capture_config={"capture_mode": "single_multi_light"}), "cam1")

    def test_flexible_mapping_and_shared_physical_camera(self):
        config = {"capture_mode": "flexible", "capture_channels": [
            {"role": "cam1", "physical_role": "cam2"}, {"role": "cam3", "physical_role": "cam2"}]}
        self.assertEqual(resolve_correction_role("unused", "cam1", capture_config=config), "cam2")
        self.assertEqual(resolve_correction_role("unused", "cam3", capture_config=config), "cam2")

    def test_backup_read_does_not_restore_or_migrate_machine_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "camera_settings.json"
            path.write_text("invalid", encoding="utf-8")
            Path(str(path) + ".bak").write_text(json.dumps({"capture_mode": "single_multi_light"}), encoding="utf-8")
            with patch("services.lens_camera_roles.default_camera_settings_path", return_value=path):
                self.assertEqual(resolve_correction_role(temp, "cam2"), "cam1")
            self.assertEqual(path.read_text(encoding="utf-8"), "invalid")

    def test_product_routing_wins_and_missing_product_never_uses_global_routing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            product = root / "product"
            product.mkdir()
            global_path = root / "global.json"
            global_path.write_text(json.dumps({"capture_mode": "single_multi_light"}), encoding="utf-8")
            with patch("services.lens_camera_roles.default_camera_settings_path", return_value=global_path):
                self.assertEqual(resolve_correction_role(str(product), "cam2"), "cam2")
                (product / "camera_settings.json").write_text(json.dumps({
                    "capture_mode": "flexible", "capture_channels": [{"role": "cam2", "physical_role": "cam3"}],
                }), encoding="utf-8")
                self.assertEqual(resolve_correction_role(str(product), "cam2"), "cam3")


if __name__ == "__main__":
    unittest.main()
