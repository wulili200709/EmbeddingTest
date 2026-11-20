from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ui.debug.tool_page.training_controller import TrainingController
from ui.debug.tool_page.training_task_builder import TrainingTaskBuilder
from ui.debug.tool_page.training_worker import TrainingJobWorker as TrainingWorker


def item(label, *, source="group", group="pins", algorithm="learning", camera="cam1", enabled=True):
    key = f"{camera}__group__{group}" if source == "group" else f"{camera}__{label}"
    if source == "library":
        key = f"{camera}__library__{'a' * 32}__v000001"
    return SimpleNamespace(
        item_id=label, display_name=label, roi_label=label, camera_id=camera,
        enabled=enabled, algorithm_code=algorithm, model_source=source,
        task_group=group if source == "group" else "", model_key=f"{camera}__{label}",
        effective_model_key=key,
    )


class TrainingAlgo:
    def __init__(self):
        self.model = None
        self.product_params = SimpleNamespace()
        self.describe_model_binding = Mock(return_value={"ready": True})

    def is_learning_tool(self, algorithm):
        return algorithm == "learning"

    def current_learning_backbone(self, _camera):
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, algorithm, _camera):
        return algorithm

    def is_embedding_algorithm(self, algorithm):
        return algorithm == "efficientnet_b0"

    def embedding_cache_dir(self, _algorithm, product_dir):
        return product_dir + "/cache"


def owner_for(items, annotations=None):
    annotations = annotations or {}
    owner = SimpleNamespace(
        algo=TrainingAlgo(), inspection_items=items,
        train_files=list(dict.fromkeys(path for path, _label in annotations)),
        ok_files=[], ng_files=[], session=SimpleNamespace(product_dir="order-b"),
        current_camera_role=lambda: "cam1",
        _filter_paths_for_camera=lambda paths, _role: paths,
        _path_has_roi_geometry=lambda path, label: (path, label) in annotations,
        _sample_roi_status_for_path=lambda path, _role, label: annotations[(path, label)],
        _training_in_progress=False,
        _warn_mixed_training_camera_samples=Mock(return_value=False),
        _selected_inspection_item=lambda: None,
        _ensure_training_roi_reviewed=Mock(return_value=True),
        window=lambda: SimpleNamespace(),
        lbl_status=SimpleNamespace(setText=Mock()),
        table=SimpleNamespace(setRowCount=Mock()),
        cmb_mode=SimpleNamespace(currentText=lambda: "proto"),
        spin_margin=SimpleNamespace(value=lambda: 0.02),
        spin_topk=SimpleNamespace(value=lambda: 3),
        _training_roi_ready_signatures={}, _training_roi_pending_actions={},
        _training_roi_confirmed_signatures={}, _update_runtime_widgets=Mock(),
    )
    return owner


class SharedModelTrainingTests(unittest.TestCase):
    def test_only_one_group_roi_needs_annotated_ok_and_ng(self):
        first, second = item("roi1"), item("roi2")
        owner = owner_for([first, second], {("ok.png", "roi1"): "OK", ("ng.png", "roi1"): "NG"})
        builder = TrainingTaskBuilder(owner)
        task = builder.build_task_for_item(second)
        self.assertEqual(task["label_names"], ["roi1", "roi2"])
        self.assertEqual(task["ok_samples"], [("ok.png", "roi1")])
        self.assertEqual(task["ng_samples"], [("ng.png", "roi1")])
        self.assertEqual(task["model_key"], first.effective_model_key)
        self.assertEqual(builder.sample_groups_for_item(second), (["ok.png"], ["ng.png"], ["ok.png", "ng.png"]))

    def test_mixed_image_preserves_each_roi_label(self):
        owner = owner_for([item("roi1"), item("roi2")], {
            ("ok.png", "roi1"): "OK", ("ok.png", "roi2"): "OK",
            ("mixed.png", "roi1"): "NG", ("mixed.png", "roi2"): "OK",
        })
        task = TrainingTaskBuilder(owner).build_task_for_item(owner.inspection_items[0])
        self.assertEqual(task["ok_samples"], [("ok.png", "roi1"), ("ok.png", "roi2"), ("mixed.png", "roi2")])
        self.assertEqual(task["ng_samples"], [("mixed.png", "roi1")])

    def test_group_with_conflicting_algorithms_is_rejected(self):
        owner = owner_for([item("roi1"), item("roi2", algorithm="meanhsv_h")])
        with self.assertRaises(RuntimeError):
            TrainingTaskBuilder(owner).build_task_for_item(owner.inspection_items[0])

    def test_disabled_and_other_camera_items_do_not_join_group(self):
        first = item("roi1")
        owner = owner_for([first, item("roi2", enabled=False), item("roi3", camera="cam2")])
        self.assertEqual(TrainingTaskBuilder(owner).group_items_for_item(first), [first])

    def test_all_training_deduplicates_groups_and_skips_ready_library_reference(self):
        first, second, reference = item("roi1"), item("roi2"), item("roi3", source="library")
        owner = owner_for([first, second, reference], {("ok.png", "roi1"): "OK", ("ng.png", "roi1"): "NG"})
        controller = TrainingController(owner, TrainingWorker)
        controller.start_worker = Mock()
        controller.train_all_tools()
        payload = controller.start_worker.call_args.args[0]
        self.assertEqual(len(payload["tasks"]), 1)
        self.assertEqual(payload["tasks"][0]["model_key"], first.effective_model_key)
        owner.algo.describe_model_binding.assert_called_once_with(reference, "order-b")

    def test_order_with_only_references_needs_no_training_images(self):
        owner = owner_for([item("roi1", source="library"), item("roi2", source="library")])
        controller = TrainingController(owner, TrainingWorker)
        controller.start_worker = Mock()
        with patch("ui.debug.tool_page.training_controller.QtWidgets.QMessageBox.information") as info:
            controller.train_all_tools()
        info.assert_called_once()
        controller.start_worker.assert_not_called()
        owner._warn_mixed_training_camera_samples.assert_not_called()
        owner._ensure_training_roi_reviewed.assert_not_called()

    def test_missing_reference_stops_all_training(self):
        owner = owner_for([item("roi1", source="library")])
        owner.algo.describe_model_binding.return_value = {"ready": False, "message": "missing published model"}
        controller = TrainingController(owner, TrainingWorker)
        controller.start_worker = Mock()
        with patch("ui.debug.tool_page.training_controller.QtWidgets.QMessageBox.warning") as warning:
            controller.train_all_tools()
        warning.assert_called_once()
        self.assertIn("missing published model", warning.call_args.args[2])
        controller.start_worker.assert_not_called()

    def test_current_reference_never_starts_training(self):
        reference = item("roi1", source="library")
        owner = owner_for([reference])
        owner._selected_inspection_item = lambda: reference
        controller = TrainingController(owner, TrainingWorker)
        controller.start_worker = Mock()
        with patch("ui.debug.tool_page.training_controller.QtWidgets.QMessageBox.information"):
            controller.train_current()
        controller.start_worker.assert_not_called()
        owner._warn_mixed_training_camera_samples.assert_not_called()

    def test_worker_rejects_library_key_before_calling_algorithm(self):
        algo = SimpleNamespace(train=Mock(), product_params=SimpleNamespace())
        worker = TrainingWorker(algo, {"tasks": [{"model_key": "cam1__library__abc__v000001"}]})
        result = worker._run_training()
        algo.train.assert_not_called()
        self.assertEqual(result["success_names"], [])
        self.assertIn("cannot be trained", result["failure_messages"][0])

    def test_confirmation_token_includes_per_roi_annotation(self):
        base = {"model_key": "cam1__group__pins", "ok_files": ["mixed.png"]}
        a = TrainingController.confirmation_token("all", [{**base, "ok_samples": [("mixed.png", "roi1")]}])
        b = TrainingController.confirmation_token("all", [{**base, "ok_samples": [("mixed.png", "roi2")]}])
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
