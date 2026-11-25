"""Failed local groups cannot reach persisted or live inspection ROIs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from application.auto_roi_service import run_auto_roi_batch
from application.inspection_executor import InspectionExecutionRequest, InspectionExecutor
from application.runtime_context import ProductRuntimeContext, ToolPageRuntimeContext
from domain import InspectionItem
from shape.core import locator, services
from shape.core.localization_groups import LocalizationGroupError
from shape.core.recipe import ShapeRecipe
from ui.debug.tool_page.test_runner import _predict_image
from ui.debug.tool_page.auto_roi_reference_flow import _set_reference


class LocalizationGroupConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "frame_Cam1.png"
        self.image = np.zeros((64, 80, 3), np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image_path), self.image))
        self.annotation_path = self.image_path.with_suffix(".json")
        self.annotation_path.write_text(json.dumps({
            "shapes": [{"label": "roi1", "shape_type": "rectangle", "points": [[1, 2], [20, 22]]}],
        }), encoding="utf-8")
        self.model_path = self.root / "shape" / "cam1" / "model.json"
        self.model_path.parent.mkdir(parents=True)
        self.model_path.write_text("model sentinel", encoding="utf-8")
        self.recipe = ShapeRecipe(
            model_path=str(self.model_path), reference_image=str(self.root / "missing_reference.png"),
            reference_regions=[{"output_label": "roi1", "shape_type": "rectangle",
                                "points": [[1, 2], [20, 22]], "group_id": "upper-left"}],
            localization_groups=[{"id": "upper-left", "name": "Upper left", "points": [], "needs_rebuild": True}],
        )
        self.failure = LocalizationGroupError([
            {"id": "upper-left", "name": "Upper left", "used": True, "valid": False,
             "reason": "reference missing / needs rebuild"},
        ], preview_result=SimpleNamespace(regions=[object()]))
        self.item = InspectionItem("roi1", "Part", "cam1", "roi1", algorithm_code="meanintensity")
        self.algo = SimpleNamespace(
            is_learning_tool=lambda _algorithm: False,
            is_measurement_tool=lambda _algorithm: False,
            is_yolo_algorithm=lambda _algorithm: False,
            is_embedding_algorithm=lambda _algorithm: False,
            resolve_tool_algorithm=lambda algorithm, _role: algorithm,
            _library_model_for_key=lambda _key: None,
            product_params=SimpleNamespace(algorithm="meanintensity"),
            predict_image=Mock(return_value=SimpleNamespace(to_dict=lambda: {"pred": "OK", "total_ms": 1.0})),
        )

    def runtime_context(self):
        context = ProductRuntimeContext.__new__(ProductRuntimeContext)
        context.session = SimpleNamespace(product_dir=str(self.root))
        context.algo = self.algo
        context._loc_method = "shape"
        context._loc_methods_by_role = {"cam1": "shape"}
        context._recipes_by_role = {"cam1": self.recipe}
        context._recipe = self.recipe
        context._ref_image = ""
        context._shape_match_ms_by_image = {str(self.image_path): 12.0}
        return context

    def tool_page(self):
        return SimpleNamespace(
            algo=self.algo, session=SimpleNamespace(product_dir=str(self.root)),
            current_algorithm=lambda: "meanintensity", current_camera_role=lambda: "cam1",
            shape_recipe_for_role=lambda _role: self.recipe,
            loc_method="shape", loc_method_for_role=lambda _role: "shape", ref_image="",
            _shape_match_ms_by_image={}, _shape_autogen_ms_by_image={},
            _loc_output_labels=lambda _role: ["roi1"], load_embedding_model=Mock(),
        )

    def test_cached_runner_never_reuses_a_previous_scene_result_after_group_failure(self):
        first, third = object(), object()
        with patch("shape.core.services.load_detector_model", return_value=object()) as load, patch(
            "shape.core.services.locate_and_follow", side_effect=[first, self.failure, third]
        ) as follow:
            service = services.ShapeLocateService()
            self.assertIs(service.locate(self.image, self.recipe).result, first)
            with self.assertRaises(LocalizationGroupError) as caught:
                service.locate(self.image, self.recipe)
            self.assertIs(caught.exception, self.failure)
            self.recipe.localization_groups[0]["needs_rebuild"] = False
            self.assertIs(service.locate(self.image, self.recipe).result, third)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(follow.call_count, 3)
        self.assertIs(follow.call_args.args[2], self.recipe)

    def test_file_autogen_group_failure_leaves_existing_annotations_untouched(self):
        before = self.annotation_path.read_bytes()
        with patch.object(locator, "load_recipe", return_value=self.recipe), patch.object(
            locator._LOCATE_SERVICE, "locate", side_effect=self.failure
        ), patch.object(locator, "_delete_stale_shape_roi_shapes") as delete, patch.object(
            locator.labelme_io, "upsert_labelme_rect"
        ) as write:
            with self.assertRaises(LocalizationGroupError) as caught:
                locator.autogen_roi_json_from_shape_timed(str(self.image_path), "", str(self.root))
        self.assertIs(caught.exception, self.failure)
        self.assertEqual(self.annotation_path.read_bytes(), before)
        delete.assert_not_called()
        write.assert_not_called()

    def test_live_autogen_group_failure_never_converts_preview_regions(self):
        with patch.object(locator, "load_recipe", return_value=self.recipe), patch.object(
            locator._LOCATE_SERVICE, "locate", side_effect=self.failure
        ), patch.object(locator, "runtime_shapes_from_follow_result") as convert:
            with self.assertRaises(LocalizationGroupError):
                locator.autogen_runtime_roi_shapes_timed(self.image, "", str(self.root))
        convert.assert_not_called()

    def test_auto_roi_batch_reports_failure_without_marking_it_successful(self):
        with patch("application.auto_roi_service.shape_locator.autogen_roi_json_from_shape_timed",
                   side_effect=self.failure):
            report = run_auto_roi_batch(
                paths=[str(self.image_path)], method="shape", ref_image="", product_dir=str(self.root),
                camera_role="cam1", labels=["roi1"], only_missing=False,
            )
        self.assertEqual(report["ok"], 0)
        self.assertEqual(report["ok_paths"], [])
        self.assertEqual(len(report["errs"]), 1)
        self.assertIn("Upper left", report["errs"][0])
        self.assertEqual(report["timings"], {})

    def test_missing_reference_cannot_bypass_bound_group_for_file_inspection(self):
        for method in ("predict_image", "predict_items_batch"):
            with self.subTest(method=method), patch(
                "application.runtime_context.shape_locator.autogen_roi_json_from_shape_timed",
                side_effect=self.failure,
            ) as locate:
                context = self.runtime_context()
                with self.assertRaises(LocalizationGroupError):
                    if method == "predict_image":
                        context.predict_image(str(self.image_path), labels_override=["roi1"], algorithm_override="meanintensity")
                    else:
                        context.predict_items_batch(str(self.image_path), items=[self.item])
                locate.assert_called_once()
        self.algo.predict_image.assert_not_called()

    def test_dangling_group_binding_also_cannot_use_stale_file_roi(self):
        self.recipe.localization_groups = []
        with patch("application.runtime_context.shape_locator.autogen_roi_json_from_shape_timed",
                   side_effect=self.failure) as locate:
            with self.assertRaises(LocalizationGroupError):
                self.runtime_context().predict_image(str(self.image_path), labels_override=["roi1"])
        locate.assert_called_once()
        self.algo.predict_image.assert_not_called()

    def test_live_frame_group_failure_stops_before_any_inference(self):
        with patch("application.runtime_context.shape_locator.autogen_runtime_roi_shapes_timed",
                   side_effect=self.failure) as locate, patch(
            "application.runtime_context._predict_learning_items_batch_rows_from_frame"
        ) as predict:
            with self.assertRaises(LocalizationGroupError):
                self.runtime_context().predict_items_batch_from_frame(self.image, camera_role="cam1", items=[self.item])
        locate.assert_called_once()
        predict.assert_not_called()
        self.algo.predict_image.assert_not_called()

    def test_tool_page_batch_and_direct_image_test_do_not_bypass_missing_reference(self):
        page = self.tool_page()
        for direct in (False, True):
            with self.subTest(direct=direct), patch(
                "application.runtime_context.shape_locator.autogen_roi_json_from_shape_timed", side_effect=self.failure
            ) as locate:
                with self.assertRaises(LocalizationGroupError):
                    if direct:
                        _predict_image(page, str(self.image_path), labels_override=["roi1"])
                    else:
                        ToolPageRuntimeContext(page).predict_items_batch(str(self.image_path), items=[self.item])
                locate.assert_called_once()
        self.algo.predict_image.assert_not_called()

    def test_legacy_and_unbound_draft_keep_manual_roi_workflow_without_reference(self):
        self.recipe.reference_regions[0]["group_id"] = ""
        for draft in (False, True):
            with self.subTest(unbound_draft=draft), patch(
                "application.runtime_context.shape_locator.autogen_roi_json_from_shape_timed"
            ) as locate:
                self.recipe.localization_groups = ([{"id": "draft", "points": [], "needs_rebuild": True}] if draft else [])
                result = self.runtime_context().predict_image(
                    str(self.image_path), labels_override=["roi1"], algorithm_override="meanintensity",
                )
                self.assertEqual(result["pred"], "OK")
                locate.assert_not_called()

    def test_executor_never_falls_back_to_file_or_previous_results_after_live_group_failure(self):
        predictor = SimpleNamespace(predict_items_batch_from_frame=Mock(side_effect=self.failure), predict_image=Mock())
        executor = InspectionExecutor(predictor)
        with self.assertRaises(LocalizationGroupError) as caught:
            executor.execute(InspectionExecutionRequest(camera_id="cam1", image_path="", image_bgr=self.image, items=[self.item]))
        self.assertIs(caught.exception, self.failure)
        predictor.predict_image.assert_not_called()

    def test_reference_path_change_invalidates_groups_and_refreshes_runtime(self):
        self.recipe.localization_groups[0]["needs_rebuild"] = False
        page = self.tool_page()
        page.ref_image = self.recipe.reference_image
        page.window = lambda: SimpleNamespace()
        page.lbl_ref = None
        page._clear_training_roi_review_state = Mock()
        page._save_session = Mock()
        page._shape_recipes_by_role = {}
        page.shape_model_path_for_role = lambda _role: str(self.model_path)
        page.inspectionItemsChanged = SimpleNamespace(emit=Mock())
        replacement = str(self.root / "replacement_reference.png")
        with patch("ui.debug.tool_page.auto_roi_reference_flow.shape_locator.save_recipe_for_product") as save:
            _set_reference(page, replacement)
        self.assertTrue(page.shape_recipe.localization_groups[0]["needs_rebuild"])
        self.assertFalse(self.recipe.localization_groups[0]["needs_rebuild"])
        self.assertEqual(page.shape_recipe.reference_image, replacement)
        self.assertIs(page._shape_recipes_by_role["cam1"], page.shape_recipe)
        save.assert_called_once()
        page.inspectionItemsChanged.emit.assert_called_once()

    def test_selecting_same_normalized_reference_does_not_invalidate_groups(self):
        self.recipe.localization_groups[0]["needs_rebuild"] = False
        page = self.tool_page()
        page.ref_image = self.recipe.reference_image
        page.window = lambda: SimpleNamespace()
        page.lbl_ref = None
        page._clear_training_roi_review_state = Mock()
        page._save_session = Mock()
        page._shape_recipes_by_role = {}
        page.shape_model_path_for_role = lambda _role: str(self.model_path)
        page.inspectionItemsChanged = SimpleNamespace(emit=Mock())
        equivalent = str(self.root) + "/./missing_reference.png"
        with patch("ui.debug.tool_page.auto_roi_reference_flow.shape_locator.save_recipe_for_product"):
            _set_reference(page, equivalent)
        self.assertFalse(self.recipe.localization_groups[0]["needs_rebuild"])
        page.inspectionItemsChanged.emit.assert_not_called()

    def test_failed_reference_save_keeps_old_reference_and_group_state(self):
        self.recipe.localization_groups[0]["needs_rebuild"] = False
        page = self.tool_page()
        page.ref_image = self.recipe.reference_image
        page.window = lambda: SimpleNamespace()
        page.lbl_ref = Mock()
        page.lbl_status = Mock()
        page._clear_training_roi_review_state = Mock()
        page._save_session = Mock()
        page._shape_recipes_by_role = {"cam1": self.recipe}
        page.shape_recipe = self.recipe
        page.shape_model_path_for_role = lambda _role: str(self.model_path)
        page.inspectionItemsChanged = SimpleNamespace(emit=Mock())
        original_path = page.ref_image
        with patch("ui.debug.tool_page.auto_roi_reference_flow.shape_locator.save_recipe_for_product",
                   side_effect=OSError("disk full")):
            _set_reference(page, str(self.root / "replacement_reference.png"))
        self.assertEqual(page.ref_image, original_path)
        self.assertIs(page.shape_recipe, self.recipe)
        self.assertFalse(self.recipe.localization_groups[0]["needs_rebuild"])
        page.lbl_ref.setText.assert_not_called()
        page._save_session.assert_not_called()
        page.inspectionItemsChanged.emit.assert_not_called()
        self.assertIn("not saved", page.lbl_status.setText.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
