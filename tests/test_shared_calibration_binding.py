"""Shared calibration must commit the selected model source before refresh."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PySide6 import QtWidgets

from application import AlgorithmController, ProductSession
from application.runtime_context import ProductRuntimeContext
from common.labelme_io import roi_xywh_to_labelme_shape, upsert_labelme_shapes
from common.model_library import ModelLibrary
from domain import InspectionItem, load_inspection_items, save_inspection_items
from shape.core.recipe import ShapeRecipe, load_recipe, save_recipe
from ui.debug.tool_page import ToolPage
from ui.debug.tool_page.training_worker import TrainingJobWorker
from ui.i18n import language_code, LANG_EN, set_language


class SharedCalibrationBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.addCleanup(set_language, language_code(), persist=False)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.session = ProductSession(str(self.root / "orders"))
        self.session.load()
        self.session.switch_product("Default")
        self.algo = AlgorithmController()
        self.algo.model_library = ModelLibrary(self.root / "library")
        self.algo.product_params.algorithm = "meanintensity"
        self.algo.save_params(self.session.product_params_path)
        self.group = "Shared presence"
        initial = [
            InspectionItem("roi1", "First part", "cam1", "roi1", "meanintensity", model_source="group", task_group=self.group),
            InspectionItem("roi2", "Second part", "cam1", "roi2", "meanintensity"),
        ]
        save_inspection_items(initial, self.session.inspection_items_path)
        save_recipe(ShapeRecipe(reference_regions=[
            {"reference_label": "roi1", "output_label": "roi1", "display_name": "First part", "shape_type": "rectangle", "points": [[0., 0.], [12., 20.]]},
            {"reference_label": "roi2", "output_label": "roi2", "display_name": "Second part", "shape_type": "rectangle", "points": [[20., 0.], [32., 20.]]},
        ]), self.session.shape_recipe_path)
        set_language(LANG_EN, persist=False)
        self.runtime = ProductRuntimeContext(self.session, self.algo)
        self.page = ToolPage(self.session, self.algo, lite_mode=True)
        self.page.inspection_items = load_inspection_items(self.session.inspection_items_path)
        self.reload_count = 0

        def reload_runtime():
            self.reload_count += 1
            self.runtime.reload()

        self.page.inspectionItemsChanged.connect(reload_runtime)
        self.good = self._sample("good_cam1.png", 220, 30)
        self.bad = self._sample("bad_cam1.png", 30, 220)
        self.page.train_files = [self.good, self.bad]
        self.annotations = {(self.good, "roi1"): "OK", (self.bad, "roi1"): "NG"}
        self.page._path_has_roi_geometry = lambda _path, _label: True
        self.page._sample_roi_status_for_path = lambda path, _role, label: self.annotations.get((path, label), "")
        self.page._filter_paths_for_camera = lambda paths, _role: list(paths)
        self.page._sample_annotation_counts_for_roi = self._counts
        self.page._warn_mixed_training_camera_samples = Mock(return_value=False)
        self._select_second()

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def _sample(self, filename, left, right):
        path = Path(self.session.product_dir) / filename
        pixels = np.zeros((20, 40, 3), dtype=np.uint8)
        pixels[:, :12] = left
        pixels[:, 20:32] = right
        Image.fromarray(pixels).save(path)
        upsert_labelme_shapes(str(path), [
            roi_xywh_to_labelme_shape(0, 0, 12, 20, "roi1"),
            roi_xywh_to_labelme_shape(20, 0, 12, 20, "roi2"),
        ])
        return str(path)

    def _counts(self, label, _role, *, paths=None):
        statuses = [self.annotations.get((path, label), "") for path in (paths if paths is not None else self.page.train_files)]
        return statuses.count("OK"), statuses.count("NG"), statuses.count("")

    def _select_second(self):
        self.page._refresh_inspection_items_table()
        self.page.inspection_items_table.selectRow(1)
        self.app.processEvents()

    def _set_group_draft(self, name=None):
        panel = self.page.model_source_panel
        panel.mode.setCurrentIndex(panel.mode.findData("group"))
        panel.group.setCurrentText(self.group if name is None else name)

    def _assert_group_everywhere(self):
        selected = self.page._selected_inspection_item()
        self.assertEqual(selected.roi_label, "roi2")
        for item in (selected, load_inspection_items(self.session.inspection_items_path)[1], self.runtime.inspection_items[1]):
            self.assertEqual((item.model_source, item.task_group), ("group", self.group))
            self.assertEqual(item.effective_model_key, self.page.inspection_items[0].effective_model_key)
        self.assertEqual(self.page.model_source_panel.mode.currentData(), "group")
        self.assertEqual(self.page.model_source_panel.group.currentText(), self.group)

    def _calibration_payload(self, *, all_tools=False):
        controller = self.page.training_controller
        action = controller.train_all_tools if all_tools else controller.train_current
        with patch.object(controller, "start_worker") as start, patch.object(QtWidgets.QMessageBox, "warning") as warning, patch.object(QtWidgets.QMessageBox, "information"):
            action()
            self._assert_group_everywhere()
            start.assert_not_called()
            action()
            warning.assert_not_called()
            start.assert_called_once()
            return start.call_args.args[0]

    def _run_real_worker_and_finish(self, payload):
        result = TrainingJobWorker(self.algo, payload)._run_training()
        self.assertEqual(result["failure_messages"], [])
        self.assertEqual(len(result["success_names"]), 1)
        with patch.object(QtWidgets.QMessageBox, "information"), patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.page.training_controller.on_finished(result)
            warning.assert_not_called()
        self.runtime.reload()
        self._assert_group_everywhere()
        for item in self.page.inspection_items:
            self.assertTrue(self.algo.describe_model_binding(item, self.session.product_dir)["ready"])
        model = self.algo.get_traditional_model_dict("meanintensity", model_key=self.page.inspection_items[0].effective_model_key)
        self.assertIsNotNone(model)
        self.assertEqual(model["ok_mean"], 220.)
        self.assertEqual(model["ng_mean"], 30.)
        self.assertEqual(self.algo.predict_image(self.good, labels=["roi1"], algorithm_override="meanintensity", model_key_override=self.page.inspection_items[0].effective_model_key).pred, "OK")
        self.assertEqual(self.algo.predict_image(self.good, labels=["roi2"], algorithm_override="meanintensity", model_key_override=self.page.inspection_items[1].effective_model_key).pred, "NG")

    def test_current_calibration_commits_unapplied_group_and_uses_only_annotated_peer_samples(self):
        self._set_group_draft()
        self.assertEqual(self.page.inspection_items[1].model_source, "independent")
        payload = self._calibration_payload()
        self.assertGreater(self.reload_count, 0)
        task = payload["tasks"][0]
        self.assertEqual(task["model_key"], self.page.inspection_items[0].effective_model_key)
        self.assertEqual(task["model_source"], "group")
        self.assertEqual(task["label_names"], ["roi1", "roi2"])
        self.assertEqual(task["ok_samples"], [(self.good, "roi1")])
        self.assertEqual(task["ng_samples"], [(self.bad, "roi1")])
        self._run_real_worker_and_finish(payload)

    def test_already_applied_group_survives_worker_completion_refresh_and_runtime_reload(self):
        self._set_group_draft()
        self.assertTrue(self.page.model_source_panel.apply())
        self._assert_group_everywhere()
        payload = self._calibration_payload()
        self._run_real_worker_and_finish(payload)
        self.page._refresh_inspection_items_table()
        self.page._update_runtime_widgets()
        self._assert_group_everywhere()

    def test_empty_group_draft_rejects_current_calibration_without_starting_worker(self):
        self._set_group_draft("")
        with patch.object(self.page.training_controller, "start_worker") as start, patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.page.training_controller.train_current()
        start.assert_not_called()
        warning.assert_called_once()
        self.assertEqual(self.reload_count, 0)
        self.assertEqual(self.page.inspection_items[1].model_source, "independent")
        self.assertEqual(load_inspection_items(self.session.inspection_items_path)[1].model_source, "independent")
        self.assertEqual(self.page.model_source_panel.mode.currentData(), "group")

    def test_all_calibration_commits_group_draft_and_trains_shared_target_once(self):
        self._set_group_draft()
        payload = self._calibration_payload(all_tools=True)
        self.assertEqual(len(payload["tasks"]), 1)
        self.assertEqual(payload["tasks"][0]["model_source"], "group")
        self._run_real_worker_and_finish(payload)

    def test_empty_group_draft_rejects_all_calibration_without_starting_worker(self):
        self._set_group_draft("")
        with patch.object(self.page.training_controller, "start_worker") as start, patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.page.training_controller.train_all_tools()
        start.assert_not_called()
        warning.assert_called_once()
        self.assertEqual(self.page.inspection_items[1].model_source, "independent")
        self.assertEqual(load_inspection_items(self.session.inspection_items_path)[1].model_source, "independent")

    def test_all_calibration_with_measurement_selected_does_not_apply_hidden_group_controls(self):
        self.page.inspection_items[1].enabled = False
        measurement = InspectionItem("roi3", "Reference edge", "cam1", "roi3", "find_line")
        self.page.inspection_items.append(measurement)
        save_inspection_items(self.page.inspection_items, self.session.inspection_items_path)
        recipe = load_recipe(self.session.shape_recipe_path)
        recipe.reference_regions.append({"reference_label": "roi3", "output_label": "roi3", "display_name": "Reference edge", "shape_type": "rectangle", "points": [[0., 0.], [12., 20.]]})
        save_recipe(recipe, self.session.shape_recipe_path)
        self.page._refresh_inspection_items_table()
        self.page.inspection_items_table.selectRow(0)
        self.assertEqual(self.page.model_source_panel.mode.currentData(), "group")
        self.page.inspection_items_table.selectRow(2)
        self.assertTrue(self.page.model_source_frame.isHidden())
        self.assertFalse(self.page.model_source_panel.has_pending_changes())
        snapshot = measurement.to_dict()
        controller = self.page.training_controller
        with patch.object(controller, "start_worker") as start, patch.object(QtWidgets.QMessageBox, "warning") as warning, patch.object(QtWidgets.QMessageBox, "information"):
            controller.train_all_tools()
            controller.train_all_tools()
            warning.assert_not_called()
            start.assert_called_once()
            payload = start.call_args.args[0]
        self.assertEqual(len(payload["tasks"]), 1)
        self.assertEqual(payload["tasks"][0]["model_key"], self.page.inspection_items[0].effective_model_key)
        self.assertEqual(self.reload_count, 0)
        self.assertEqual(measurement.to_dict(), snapshot)
        result = TrainingJobWorker(self.algo, payload)._run_training()
        self.assertEqual(result["failure_messages"], [])
        with patch.object(QtWidgets.QMessageBox, "information"), patch.object(QtWidgets.QMessageBox, "warning"):
            controller.on_finished(result)
        self.runtime.reload()
        saved = {item.item_id: item for item in load_inspection_items(self.session.inspection_items_path)}
        reloaded = {item.item_id: item for item in self.runtime.inspection_items}
        self.assertEqual(saved["roi3"].to_dict(), snapshot)
        self.assertEqual(reloaded["roi3"].to_dict(), snapshot)
        self.assertTrue(self.algo.describe_model_binding(self.page.inspection_items[0], self.session.product_dir)["ready"])


if __name__ == "__main__":
    unittest.main()
