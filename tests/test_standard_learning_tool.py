from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from PySide6 import QtCore, QtWidgets
from application import AlgorithmController, ProductSession
from application.runtime.execution import _validate_runtime_model_bindings, _warm_runtime_models
from application.runtime_context import _predict_learning_items_batch_rows, _predict_learning_items_batch_rows_from_frame
from application.runtime.preview_frame import RuntimePreviewShape
from algorithms.yolo_classifier import classifier_paths, _write_dataset_group
from common.model_library import ModelLibrary
from domain import InspectionItem, save_inspection_items, load_inspection_items
from test_model_source_ui import _Owner
from ui.debug.tool_page import ToolPage
from ui.debug.tool_page.training_worker import TrainingJobWorker
from ui.debug.tool_page.training_task_builder import TrainingTaskBuilder
from ui.algorithm_labels import algorithm_display_name
from ui.i18n import LANG_ZH, LANG_EN, language_code, set_language


class StandardLearningClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.algo = AlgorithmController()
        self.algo.model_library = ModelLibrary(self.root / "library")
        self.algo.set_learning_backbone("yolov8n_cls", "cam1")
        self.algo.product_params.yolo_ng_threshold = .6
        self.source = self.root / "source"
        self.source.mkdir()
        self.item = InspectionItem("roi1", "Source pin", "cam1", "roi1", model_source="group", task_group="PIN")
        paths = self.algo.yolo_model_paths(str(self.source), model_key=self.item.effective_model_key)
        nodes = [helper.make_node("ReduceMean", ["images"], ["mean"], axes=[0, 1, 2, 3], keepdims=0),
                 helper.make_node("Mul", ["mean", "zero"], ["offset"]),
                 helper.make_node("Add", ["probabilities", "offset"], ["scores"])]
        graph = helper.make_graph(nodes, "tiny-classifier",
                                  [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
                                  [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
                                  [numpy_helper.from_array(np.array(0., np.float32), "zero"),
                                   numpy_helper.from_array(np.array([[.25, .75]], np.float32), "probabilities")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8)
        onnx.save(model, str(paths.onnx))
        paths.metadata.write_text(json.dumps({"names": {"0": "ok", "1": "ng"}, "image_size": 8}), encoding="utf-8")
        self.publication = self.algo.publish_model_to_library(self.item, str(self.source), "PIN classifier")
        paths.onnx.unlink()
        paths.metadata.unlink()
        self.target = self.root / "target"
        self.target.mkdir()
        self.bound = InspectionItem("roi2", "Target pin", "cam1", "roi2", model_source="library",
                                    library_model_id=self.publication["model_id"], library_version=self.publication["version"])

    def test_classifier_publication_survives_source_removal_and_keeps_published_threshold(self):
        self.algo.set_learning_backbone("efficientnet_b0", "cam1")
        self.algo.product_params.yolo_ng_threshold = .9
        self.assertEqual(self.algo.resolve_item_algorithm(self.bound), "yolov8n_cls")
        self.algo.load_model_for_algorithm("efficientnet_b0", str(self.target), model_key=self.bound.effective_model_key)
        image = np.zeros((30, 50, 3), np.uint8)
        result = self.algo.predict_yolo_from_array(image, {"shape_type": "rectangle", "points": [[2, 3], [40, 20]]},
                                                  model_key=self.bound.effective_model_key)
        self.assertEqual(result.pred, "NG")
        self.assertAlmostEqual(result.threshold, .6)
        self.assertEqual(list(self.target.iterdir()), [])

    def test_classifier_metadata_is_verified_and_public_model_cannot_be_retrained(self):
        with self.assertRaisesRegex(ValueError, "只读"):
            self.algo.train([], [], algorithm="yolov8n_cls", product_dir=str(self.target),
                            label_names=["roi2"], model_key=self.bound.effective_model_key)
        Path(self.publication["classifier_metadata"]).write_text('{"names": {}}', encoding="utf-8")
        self.assertFalse(self.algo.describe_model_binding(self.bound, str(self.target))["ready"])

    def test_runtime_precheck_uses_classifier_runner_without_embedding_backbone(self):
        runtime = SimpleNamespace(_algo=self.algo, _session=SimpleNamespace(product_dir=str(self.target)),
                                  _runtime_context=SimpleNamespace(load_embedding_model=lambda algorithm, model_key:
                                      self.algo.load_model_for_algorithm(algorithm, str(self.target), model_key=model_key)))
        self.assertEqual(_validate_runtime_model_bindings(runtime, [self.bound]), (True, ""))
        with patch.object(self.algo, "get_feat_net") as embedding_warmup:
            _warm_runtime_models(runtime, [self.bound])
        embedding_warmup.assert_not_called()
        self.assertIsNotNone(self.algo.model)

    def test_file_and_live_shared_rois_use_their_own_crop_without_embedding_backend(self):
        items = [InspectionItem(label, label, "cam1", label, model_source="group", task_group="POSITION")
                 for label in ("dark", "bright")]
        paths = classifier_paths(self.target, items[0].effective_model_key)
        nodes = [helper.make_node("ReduceMean", ["images"], ["mean"], axes=[0, 1, 2, 3], keepdims=0),
                 helper.make_node("Sub", ["one", "mean"], ["inverse"]),
                 helper.make_node("Unsqueeze", ["mean"], ["ok"], axes=[0, 1]),
                 helper.make_node("Unsqueeze", ["inverse"], ["ng"], axes=[0, 1]),
                 helper.make_node("Concat", ["ok", "ng"], ["scores"], axis=1)]
        graph = helper.make_graph(nodes, "brightness-classifier",
                                  [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
                                  [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
                                  [numpy_helper.from_array(np.array(1., np.float32), "one")])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8), str(paths.onnx))
        paths.metadata.write_text(json.dumps({"names": {"0": "ok", "1": "ng"}, "image_size": 8}), encoding="utf-8")
        image = np.zeros((20, 60, 3), np.uint8)
        image[:, 40:] = 255
        image_path = self.target / "positions.png"
        Image.fromarray(image).save(image_path)
        shapes = [dict(label=label, shape_type="rectangle", points=points) for label, points in
                  (("dark", [[0, 0], [20, 20]]), ("bright", [[40, 0], [60, 20]]))]
        image_path.with_suffix(".json").write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
        loader = lambda algorithm, model_key: self.algo.load_model_for_algorithm(algorithm, str(self.target), model_key=model_key)
        with patch.object(self.algo, "get_feat_net", side_effect=AssertionError("Classifier must use ONNX")):
            disk = _predict_learning_items_batch_rows(path=str(image_path), items=items, match_ms=0.,
                                                      algo=self.algo, load_embedding_model=loader)
            live = _predict_learning_items_batch_rows_from_frame(
                image_bgr=image, roi_shapes=tuple(RuntimePreviewShape(label=shape["label"], shape_type="rectangle",
                    points=tuple(tuple(point) for point in shape["points"])) for shape in shapes),
                items=items, match_ms=0., algo=self.algo, load_embedding_model=loader)
        for rows in (disk, live):
            self.assertEqual([rows[item.model_key]["pred"] for item in items], ["NG", "OK"])

    def test_repeated_tests_reuse_classifier_and_explicit_reload_releases_it(self):
        params = str(self.target / "product_params.json")
        self.algo.save_params(params)
        self.algo.load_params(params)
        first, _ = self.algo.load_model_for_algorithm("yolov8n_cls", str(self.target), model_key=self.bound.effective_model_key)
        self.algo.load_params(params, reuse_embedding_models=True)
        second, _ = self.algo.load_model_for_algorithm("yolov8n_cls", str(self.target), model_key=self.bound.effective_model_key)
        self.assertIs(first, second)
        self.algo.load_params(params)
        third, _ = self.algo.load_model_for_algorithm("yolov8n_cls", str(self.target), model_key=self.bound.effective_model_key)
        self.assertIsNot(second, third)


class StandardLearningUiAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.addCleanup(set_language, language_code(), persist=False)
        set_language(LANG_ZH, persist=False)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_menu_name_and_parameter_persistence(self):
        session = ProductSession(str(self.root / "orders"))
        session.load()
        session.switch_product("Default")
        save_inspection_items([InspectionItem("roi1", "ROI1", "cam1", "roi1")], session.inspection_items_path)
        algo = AlgorithmController()
        algo.model_library = ModelLibrary(self.root / "library")
        page = ToolPage(session, algo, lite_mode=True)
        try:
            page.inspection_items = load_inspection_items(session.inspection_items_path)
            page._refresh_inspection_items_table()
            page.inspection_items_table.selectRow(0)
            page._set_current_algorithm("yolov8n_cls")
            self.assertEqual(page.btn_algorithm_picker.text(), "标准学习工具")
            self.assertEqual(algo.current_learning_backbone("cam1"), "yolov8n_cls")
            self.assertFalse(page.spin_yolo_epochs.isHidden())
            self.assertFalse(page.cmb_mode.isEnabled())
            page.spin_yolo_epochs.setValue(14)
            page.spin_yolo_batch.setValue(3)
            page.spin_yolo_ng_threshold.setValue(.65)
            saved = json.loads(Path(session.product_params_path).read_text(encoding="utf-8"))
            self.assertEqual((saved["algorithm"], saved["yolo_epochs"], saved["yolo_batch"]), ("y8ncls", 14, 3))
            self.assertAlmostEqual(saved["yolo_ng_threshold"], .65)
            set_language(LANG_EN, persist=False)
            page.retranslate_ui()
            self.assertEqual(page.btn_algorithm_picker.text(), "Standard Learning Tool")
            self.assertEqual(algorithm_display_name("y8ncls"), "Standard Learning Tool")
            page._set_current_algorithm("efficientnet_b0")
            self.assertTrue(page.spin_yolo_epochs.isHidden())
        finally:
            self.app.processEvents()
            page.close()
            page.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            self.app.processEvents()

    def test_worker_preserves_every_group_roi_sample_and_keeps_independent_roi_separate(self):
        items = [InspectionItem(label, label, "cam1", label, model_source="group", task_group="PIN")
                 for label in ("roi1", "roi2")]
        items.append(InspectionItem("roi3", "roi3", "cam1", "roi3"))
        owner = _Owner(self.root, items)
        owner.algo = AlgorithmController()
        owner.algo.set_learning_backbone("yolov8n_cls", "cam1")
        shapes = [dict(label=item.roi_label, shape_type="rectangle", points=[[i*10, 0], [(i+1)*10, 10]])
                  for i, item in enumerate(items)]
        owner.train_files = []
        owner.annotations = {}
        for index, status in enumerate(("OK", "NG", "OK", "NG")):
            path = self.root / f"sample{index}.png"
            Image.new("RGB", (30, 10), (220, 220, 220) if index == 0 else (20, 20, 20)).save(path)
            path.with_suffix(".json").write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
            owner.train_files.append(str(path))
            for item in items:
                owner.annotations[str(path), item.roi_label] = status if item is not items[2] else ("NG" if status == "OK" else "OK")
        tasks = [TrainingTaskBuilder(owner).build_task_for_item(items[index]) for index in (0, 2)]
        captured = []

        def train(ok_samples, ng_samples, *, product_dir, model_key, **kwargs):
            captured.append((model_key, list(ok_samples), list(ng_samples)))
            dataset = self.root / "datasets" / model_key
            for category, samples in (("ok", ok_samples), ("ng", ng_samples)):
                _write_dataset_group(samples, category, dataset, size=8, seed=1, progress_callback=None)
            return SimpleNamespace(), classifier_paths(product_dir, model_key), dataset

        try:
            with patch("application.algorithm_controller.train_yolo_classifier", side_effect=train), \
                 patch.object(owner.algo, "_load_yolo_model_cached", return_value=SimpleNamespace()):
                result = TrainingJobWorker(owner.algo, {"mode": "all", "product_dir": str(self.root), "tasks": tasks})._run_training()
            self.assertFalse(result["failure_messages"])
            self.assertEqual(len(captured), 2)
            self.assertEqual({label for _, label in captured[0][1] + captured[0][2]}, {"roi1", "roi2"})
            self.assertEqual({label for _, label in captured[1][1] + captured[1][2]}, {"roi3"})
            group_files = list((self.root / "datasets" / items[0].effective_model_key).rglob("*.png"))
            self.assertEqual(len(group_files), 8)
            self.assertEqual(captured[1][1], [(owner.train_files[1], "roi3"), (owner.train_files[3], "roi3")])
        finally:
            owner.close()
            owner.deleteLater()
            self.app.processEvents()

    def test_minimum_sample_check_counts_group_crops_instead_of_image_files(self):
        items = [InspectionItem(label, label, "cam1", label, model_source="group", task_group="PIN")
                 for label in ("roi1", "roi2")]
        owner = _Owner(self.root, items)
        owner.algo = AlgorithmController()
        owner.algo.set_learning_backbone("yolov8n_cls", "cam1")
        try:
            with self.assertRaisesRegex(RuntimeError, "2"):
                TrainingTaskBuilder(owner).build_task_for_item(items[0])
            owner.annotations.update({("good.png", "roi2"): "OK", ("bad.png", "roi2"): "NG"})
            task = TrainingTaskBuilder(owner).build_task_for_item(items[0])
            self.assertEqual(len(task["ok_samples"]), 2)
            self.assertEqual(len(task["ng_samples"]), 2)
            self.assertEqual(len(task["ok_files"]), 1)
            self.assertEqual(len(task["ng_files"]), 1)
        finally:
            owner.close()
            owner.deleteLater()
            self.app.processEvents()

