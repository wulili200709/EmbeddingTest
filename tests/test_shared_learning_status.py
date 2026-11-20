"""Switching the camera learning subtype must update shared model readiness."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets

from algorithms.embedding import RegisterModel, save_register_model_npz
from application import AlgorithmController, ProductSession
from application.runtime_context import ProductRuntimeContext
from common.model_library import ModelLibrary
from domain import InspectionItem, load_inspection_items, save_inspection_items
from shape.core.recipe import ShapeRecipe, save_recipe
from ui.debug.tool_page import ToolPage
from ui.i18n import LANG_EN, LANG_ZH, set_language
from ui.window_common import connect_runtime_refresh_sources


class SharedLearningStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        set_language(LANG_EN, persist=False)
        self.session = ProductSession(str(self.root / "orders"))
        self.session.load()
        self.session.switch_product("Default")
        self.algo = AlgorithmController()
        self.algo.model_library = ModelLibrary(self.root / "library")
        for camera in ("cam1", "cam2"):
            self.algo.set_learning_backbone("mobilenet_v3_large", camera)
        self.algo.product_params.algorithm = "mobilenet_v3_large"
        self.algo.save_params(self.session.product_params_path)
        self.items = [InspectionItem(label, label, "cam1", label, model_source="group", task_group="PIN")
                      for label in ("roi1", "roi2")]
        self.other_camera = InspectionItem("roi3", "Other camera", "cam2", "roi3", model_source="group", task_group="PIN")
        self.items.append(self.other_camera)
        self.models = []
        for item in (self.items[0], self.other_camera):
            path = Path(self.algo.embedding_model_path("mobilenet_v3_large", self.session.product_dir,
                                                      model_key=item.effective_model_key))
            ok, ng = np.eye(4, dtype=np.float32)[:2]
            save_register_model_npz(RegisterModel(
                backbone="mobilenet_v3_large", score_mode="proto", margin=.02, topk=1,
                label_name=item.roi_label, ok_proto=ok, ng_proto=ng,
                ok_bank=np.stack([ok]), ng_bank=np.stack([ng]),
            ), str(path))
            self.models.append(path)
        save_inspection_items(self.items, self.session.inspection_items_path)
        for camera in ("cam1", "cam2"):
            save_recipe(ShapeRecipe(reference_regions=[
                {"reference_label": item.roi_label, "output_label": item.roi_label,
                 "display_name": item.display_name, "shape_type": "rectangle", "points": [[0., 0.], [12., 20.]]}
                for item in self.items if item.camera_id == camera
            ]), self.session.shape_recipe_path_for_role(camera))
        self.page = ToolPage(self.session, self.algo, lite_mode=True)
        self.page._defer_initial_session_load = False
        self.page.inspection_items = self.items
        self.page._refresh_inspection_items_table()
        self.context = ProductRuntimeContext(self.session, self.algo)
        runtime = SimpleNamespace(refresh_all_status=lambda _message: self.context.reload())
        connect_runtime_refresh_sources(self.page, runtime, session_loaded_message="test")
        self.app.processEvents()
        self.page.inspection_items_table.clearSelection()
        self.assertIsNone(self.page._selected_inspection_item())

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def choose_tool(self, algorithm):
        self.page._algorithm_actions[algorithm].trigger()
        self.app.processEvents()
        self.assertEqual(self.page.current_algorithm(), algorithm)

    def assert_table_ready(self, ready):
        table = self.page.inspection_items_table
        self.assertEqual(table.rowCount(), 2)
        for row in range(table.rowCount()):
            status = table.item(row, 4)
            self.assertIn("PIN", status.text())
            self.assertEqual(status.foreground().color().name(), "#79d279" if ready else "#d98c8c")
            if not ready:
                self.assertIn("untrained", status.text().lower())
                self.assertIn(self.algo.algorithm_display_name("efficientnet_b0"), status.toolTip())

    def test_no_roi_selection_refreshes_every_shared_row_and_switch_back_restores_ready(self):
        original = {path: path.read_bytes() for path in self.models}
        self.assert_table_ready(True)
        self.choose_tool("efficientnet_b0")
        self.assert_table_ready(False)
        self.assertIsNone(self.page._selected_inspection_item())
        self.assertFalse(self.algo.describe_model_binding(self.items[0], self.session.product_dir)["ready"])
        self.assertTrue(self.algo.describe_model_binding(self.other_camera, self.session.product_dir)["ready"])
        self.assertEqual({item.model_source for item in load_inspection_items(self.session.inspection_items_path)}, {"group"})
        self.choose_tool("mobilenet_v3_large")
        self.assert_table_ready(True)
        self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))

    def test_global_tool_switch_leaves_pinned_library_reference_ready(self):
        publication = self.algo.model_library.publish("Published PIN", "mobilenet_v3_large", model_file=self.models[0])
        reference = InspectionItem("roi4", "Published pin", "cam1", "roi4", model_source="library",
                                   library_model_id=publication["model_id"], library_version=publication["version"])
        self.page.inspection_items.append(reference)
        self.page._persist_inspection_items()
        self.page._refresh_inspection_items_table()
        self.choose_tool("efficientnet_b0")
        statuses = [self.page.inspection_items_table.item(row, 4) for row in range(3)]
        self.assertEqual([status.foreground().color().name() for status in statuses], ["#d98c8c", "#d98c8c", "#79d279"])
        self.assertEqual(self.algo.resolve_item_algorithm(reference), "mobilenet_v3_large")

    def test_missing_shared_subtype_is_reported_before_decoding_or_running_inference(self):
        set_language(LANG_ZH, persist=False)
        self.page.retranslate_ui()
        self.choose_tool("efficientnet_b0")
        image = self.root / "test-image.png"
        image.write_bytes(b"precheck runs before decoding")
        with patch("ui.debug.tool_page.test_execution_controller.imread") as decode, \
             patch("ui.debug.tool_page.test_execution_controller.ProductRuntimeContext") as runtime:
            with self.assertRaises(RuntimeError) as error:
                self.page._execute_test_image(str(image))
            decode.assert_not_called()
            runtime.assert_not_called()
        self.assertIn("PIN", str(error.exception))
        self.assertIn("高精度学习工具", str(error.exception))
        self.assertIn("未训练", str(error.exception))

    def test_batch_test_reports_missing_model_once_before_attempting_any_images(self):
        self.choose_tool("efficientnet_b0")
        controller = self.page.test_execution_controller
        with patch.object(self.page, "_sample_paths_for_kind", return_value=["first.png", "second.png"]), \
             patch.object(controller, "execute_image") as execute, \
             patch.object(QtWidgets.QMessageBox, "critical") as message:
            controller.run_all_test_samples()
        execute.assert_not_called()
        message.assert_called_once()
        self.assertIn("PIN", message.call_args.args[2])
        self.assertIn(self.algo.algorithm_display_name("efficientnet_b0"), message.call_args.args[2])

    def test_ready_models_pass_precheck_and_unavailable_inactive_rois_do_not_block(self):
        self.algo.set_learning_backbone("efficientnet_b0", "cam2")
        self.assertFalse(self.algo.describe_model_binding(self.other_camera, self.session.product_dir)["ready"])
        publication = self.algo.model_library.publish("Published PIN", "mobilenet_v3_large", model_file=self.models[0])
        reference = InspectionItem("roi4", "Published pin", "cam1", "roi4", model_source="library",
                                   library_model_id=publication["model_id"], library_version=publication["version"])
        disabled = InspectionItem("disabled", "Disabled pin", "cam1", "disabled", enabled=False,
                                  model_source="group", task_group="Untrained group")
        self.page.inspection_items.extend([reference, disabled])
        self.page.test_execution_controller._validate_model_bindings(self.page._test_target_inspection_items())
        self.assertTrue(self.algo.describe_model_binding(reference, self.session.product_dir)["ready"])


if __name__ == "__main__":
    unittest.main()
