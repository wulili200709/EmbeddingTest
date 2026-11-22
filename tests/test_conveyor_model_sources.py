"""Conveyor legacy groups, explicit independence and position-specific samples."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PySide6 import QtWidgets

from application import AlgorithmController
from application.runtime.execution import _validate_runtime_model_bindings, _warm_runtime_models
from common.model_library import ModelLibrary
from common.model_groups import OrderModelGroups, make_group_model_key
from domain import InspectionItem, load_inspection_items, save_inspection_items, sync_items_with_labels
from test_model_source_ui import _Owner
from ui.debug.tool_page.model_source_panel import ModelSourcePanel
from ui.debug.tool_page.training_task_builder import TrainingTaskBuilder
from ui.debug.tool_page.tool_config import _persist_inspection_items


class ConveyorModelSourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.app.processEvents()
        self.temp.cleanup()

    def test_explicit_independence_survives_template_group_names_and_serialization(self):
        item = InspectionItem("roi1", "Hole", "cam1", "roi1", model_source="independent")
        path = self.root / "inspection_items.json"
        save_inspection_items([item], str(path))
        synced = sync_items_with_labels(load_inspection_items(str(path)), ["roi1"],
                                        display_names_by_label={"roi1": "Hole"}, task_groups_by_label={"roi1": "Hole"})
        self.assertEqual((synced[0].model_source, synced[0].task_group), ("independent", ""))
        self.assertEqual(synced[0].effective_model_key, "cam1__roi1")

    def test_new_template_instance_joins_existing_legacy_group_without_rekeying_weights(self):
        old = InspectionItem.from_dict({"item_id": "roi1", "camera_id": "cam1", "roi_label": "roi1", "task_group": "孔"})
        synced = sync_items_with_labels([old], ["roi1", "roi2"], task_groups_by_label={"roi1": "孔", "roi2": "孔"})
        self.assertEqual([item.effective_model_key for item in synced], ["cam1__孔", "cam1__孔"])
        path = self.root / "inspection_items.json"
        save_inspection_items(synced, str(path))
        self.assertEqual([item.effective_model_key for item in load_inspection_items(str(path))], ["cam1__孔", "cam1__孔"])

    def test_new_group_keys_do_not_collide_with_independent_roi_keys(self):
        independent = InspectionItem("roi1", "Independent", "cam1", "roi1", model_source="independent")
        group = InspectionItem("roi2", "Shared", "cam1", "roi2", model_source="group", task_group="roi1")
        self.assertNotEqual(independent.effective_model_key, group.effective_model_key)

    def test_legacy_key_collision_forks_group_assets_before_independent_retraining(self):
        first = InspectionItem("roi1", "Original", "cam1", "roi1", model_source="independent")
        second = InspectionItem("roi2", "Copy", "cam1", "roi2", task_group="roi1")
        old_file = self.root / "cam1__roi1_register_model_b2.npz"
        old_file.write_bytes(b"original shared weights")
        path = self.root / "inspection_items.json"
        save_inspection_items([first, second], str(path))
        self.assertNotEqual(first.effective_model_key, second.effective_model_key)
        group_file = self.root / f"{second.effective_model_key}_register_model_b2.npz"
        self.assertEqual(group_file.read_bytes(), b"original shared weights")
        old_file.write_bytes(b"independent retraining")
        self.assertEqual(group_file.read_bytes(), b"original shared weights")
        reloaded = load_inspection_items(str(path))
        self.assertEqual(reloaded[1].effective_model_key, second.effective_model_key)
        self.assertEqual(OrderModelGroups(self.root).list_groups("cam1")[0]["model_key"], second.effective_model_key)

    def test_last_member_leaving_legacy_group_archives_weights_away_from_independent_key(self):
        item = InspectionItem("roi1", "Original", "cam1", "roi1", task_group="roi1")
        old_file = self.root / "cam1__roi1_register_model_b2.npz"
        old_file.write_bytes(b"shared model to retain")
        path = self.root / "inspection_items.json"
        save_inspection_items([item], str(path))
        item.model_source, item.task_group, item.legacy_model_key = "independent", "", ""
        save_inspection_items([item], str(path))
        key = make_group_model_key("cam1", "roi1")
        old_file.write_bytes(b"new independent model")
        self.assertEqual((self.root / f"{key}_register_model_b2.npz").read_bytes(), b"shared model to retain")
        self.assertEqual(OrderModelGroups(self.root).list_groups("cam1")[0]["model_key"], key)

    def test_legacy_ascii_group_asset_is_forked_when_chinese_roi_names_collide(self):
        first = InspectionItem("孔", "Original", "cam1", "roi1", model_source="independent")
        second = InspectionItem("roi2", "Copy", "cam1", "roi2", task_group="孔")
        source = self.root / "cam1__group_register_model_lt03.npz"
        source.write_bytes(b"old ASCII group model")
        save_inspection_items([first, second], str(self.root / "inspection_items.json"))
        copied = self.root / f"{second.effective_model_key}_register_model_lt03.npz"
        self.assertEqual(copied.read_bytes(), b"old ASCII group model")
        self.assertEqual(source.read_bytes(), b"old ASCII group model")

    def test_legacy_threshold_fork_remains_ready_in_memory_and_survives_parameter_save(self):
        first = InspectionItem("roi1", "Independent", "cam1", "roi1", "meanintensity", model_source="independent")
        second = InspectionItem("roi2", "Shared", "cam1", "roi2", "meanintensity", task_group="roi1")
        owner = _Owner(self.root, [first, second])
        owner.algo = AlgorithmController()
        owner.algo.model_library = ModelLibrary(self.root / "library")
        owner.algo.product_params.traditional_models = {
            "meanintensity::cam1__roi1": {"algorithm": "meanintensity", "threshold": 120., "ok_when": "greater_equal"}
        }
        params_path = str(self.root / "product_params.json")
        owner.algo.save_params(params_path)
        try:
            _persist_inspection_items(owner)
            group = owner.algo.get_traditional_model_dict("meanintensity", model_key=second.effective_model_key)
            self.assertEqual(group["threshold"], 120.)
            self.assertTrue(owner.algo.describe_model_binding(second, self.root)["ready"])
            independent = owner.algo.get_traditional_model_dict("meanintensity", model_key=first.effective_model_key)
            independent["threshold"] = 190.
            self.assertEqual(group["threshold"], 120.)
            owner.algo.save_params(params_path)
            owner.algo.load_params(params_path)
            self.assertEqual(owner.algo.get_traditional_model_dict("meanintensity", model_key=second.effective_model_key)["threshold"], 120.)
            self.assertEqual(owner.algo.get_traditional_model_dict("meanintensity", model_key=first.effective_model_key)["threshold"], 190.)
        finally:
            owner.close()
            owner.deleteLater()

    def test_legacy_group_fork_rejects_conflict_without_overwriting_existing_assets(self):
        first = InspectionItem("roi1", "Independent", "cam1", "roi1", model_source="independent")
        second = InspectionItem("roi2", "Shared", "cam1", "roi2", task_group="roi1")
        old_file = self.root / "cam1__roi1_register_model_b2.npz"
        existing = self.root / f"{make_group_model_key('cam1', 'roi1')}_register_model_b2.npz"
        old_file.write_bytes(b"original")
        existing.write_bytes(b"another trained model")
        with self.assertRaises(FileExistsError):
            save_inspection_items([first, second], str(self.root / "inspection_items.json"))
        self.assertEqual(old_file.read_bytes(), b"original")
        self.assertEqual(existing.read_bytes(), b"another trained model")
        self.assertEqual(second.effective_model_key, "cam1__roi1")
        self.assertFalse(list(self.root.glob(".group-fork-*")))

    def test_legacy_group_remains_selectable_after_every_member_leaves(self):
        items = [InspectionItem(label, "Hole", "cam1", label, "meanintensity", task_group="Hole")
                 for label in ("roi1", "roi2")]
        path = self.root / "cam1__Hole_register_model_b2.npz"
        path.write_bytes(b"legacy trained weights")
        owner = _Owner(self.root, items)
        panel = ModelSourcePanel(owner, QtWidgets.QVBoxLayout(owner))
        try:
            save_inspection_items(items, owner.session.inspection_items_path)
            for item in items:
                owner.selected = item
                panel.refresh()
                panel.mode.setCurrentIndex(panel.mode.findData("independent"))
                self.assertTrue(panel.apply())
            panel.mode.setCurrentIndex(panel.mode.findData("group"))
            self.assertEqual(panel.group.currentText(), "Hole")
            self.assertTrue(panel.apply())
            self.assertEqual(owner.selected.effective_model_key, "cam1__Hole")
            self.assertEqual(path.read_bytes(), b"legacy trained weights")
        finally:
            owner.close()
            owner.deleteLater()

    def test_disabled_group_member_does_not_contribute_training_samples(self):
        first = InspectionItem("roi1", "Hole", "cam1", "roi1", "meanintensity", task_group="Hole")
        second = InspectionItem("roi2", "Hole", "cam1", "roi2", "meanintensity", task_group="Hole", enabled=False)
        owner = _Owner(self.root, [first, second])
        owner.annotations.update({("good.png", "roi2"): "NG", ("bad.png", "roi2"): "OK"})
        try:
            task = TrainingTaskBuilder(owner).build_task_for_item(first)
            self.assertEqual(task["ok_samples"], [("good.png", "roi1")])
            self.assertEqual(task["ng_samples"], [("bad.png", "roi1")])
        finally:
            owner.close()
            owner.deleteLater()

    def test_shared_training_accepts_different_crop_sizes_and_keeps_independent_samples_separate(self):
        items = [InspectionItem(label, label, "cam1", label, "meanintensity", model_source="group", task_group="PIN")
                 for label in ("roi1", "roi2")]
        items.append(InspectionItem("roi3", "roi3", "cam1", "roi3", "meanintensity", model_source="independent"))
        owner = _Owner(self.root, items)
        owner.algo = AlgorithmController()
        owner.algo.model_library = ModelLibrary(self.root / "library")
        shapes = [dict(label="roi1", shape_type="rectangle", points=[[10, 10], [50, 110]]),
                  dict(label="roi2", shape_type="rectangle", points=[[80, 30], [230, 70]]),
                  dict(label="roi3", shape_type="rectangle", points=[[80, 80], [230, 110]])]
        paths = []
        for index, intensities in enumerate(((220, 205, 30), (25, 35, 220))):
            image = np.zeros((120, 240, 3), np.uint8)
            for shape, value in zip(shapes, intensities):
                (x0, y0), (x1, y1) = shape["points"]
                image[y0:y1, x0:x1] = value
            path = self.root / f"sample_{index}_cam1.png"
            cv2.imwrite(str(path), image)
            path.with_suffix(".json").write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
            paths.append(str(path))
        owner.train_files = paths
        owner.annotations = {(paths[0], "roi1"): "OK", (paths[0], "roi2"): "OK", (paths[0], "roi3"): "NG",
                             (paths[1], "roi1"): "NG", (paths[1], "roi2"): "NG", (paths[1], "roi3"): "OK"}
        try:
            builder = TrainingTaskBuilder(owner)
            group_task = builder.build_task_for_item(items[0])
            separate_task = builder.build_task_for_item(items[2])
            self.assertEqual(len(group_task["ok_samples"]), 2)
            self.assertEqual(len(group_task["ng_samples"]), 2)
            self.assertEqual(separate_task["ok_samples"], [(paths[1], "roi3")])
            for task in (group_task, separate_task):
                owner.algo.train(task["ok_files"], task["ng_files"], algorithm=task["algorithm"], product_dir=str(self.root),
                                 label_names=task["label_names"], model_key=task["model_key"],
                                 ok_samples=task["ok_samples"], ng_samples=task["ng_samples"])
            group = owner.algo.get_traditional_model_dict("meanintensity", model_key=items[0].effective_model_key)
            independent = owner.algo.get_traditional_model_dict("meanintensity", model_key=items[2].effective_model_key)
            self.assertAlmostEqual(group["ok_mean"], 212.5)
            self.assertAlmostEqual(independent["ok_mean"], 220.)
            results = [owner.algo.predict_image(paths[0], labels=[item.roi_label], algorithm_override="meanintensity",
                                                model_key_override=item.effective_model_key).pred for item in items]
            self.assertEqual(results, ["OK", "OK", "NG"])
        finally:
            owner.close()
            owner.deleteLater()


class ConveyorPublicClassifierTests(unittest.TestCase):
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
        with patch.object(self.algo, "warmup_feat_net") as embedding_warmup:
            _warm_runtime_models(runtime, [self.bound])
        embedding_warmup.assert_not_called()
        self.assertIsNotNone(self.algo.model)


if __name__ == "__main__":
    unittest.main()
