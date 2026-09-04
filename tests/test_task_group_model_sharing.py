from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from application.algorithm_controller import AlgorithmController
from application.runtime_context import _group_inspection_items, _predict_grouped_items_from_path
from domain import InspectionItem, load_inspection_items, save_inspection_items, sync_items_with_labels
from ui.debug.tool_page.auto_roi import _task_groups_from_display_names
from ui.debug.tool_page.inspection_item_status import _inspection_item_status
from ui.debug.tool_page.training_task_builder import TrainingTaskBuilder


class _TrainingAlgo:
    @staticmethod
    def is_learning_tool(code: object) -> bool:
        return str(code) == "shared_backbone_register"

    @staticmethod
    def is_embedding_algorithm(code: object) -> bool:
        return str(code) == "mobilenet_v3_large"

    @staticmethod
    def current_learning_backbone(_camera_id: object = None) -> str:
        return "mobilenet_v3_large"

    def resolve_tool_algorithm(self, code: object, _camera_id: object = None) -> str:
        if self.is_learning_tool(code):
            return self.current_learning_backbone()
        return str(code or "")

    @staticmethod
    def embedding_cache_dir(_algorithm: str, product_dir: str) -> str:
        return str(Path(product_dir) / "embedding_cache" / "b2")


class TaskGroupModelSharingTests(unittest.TestCase):
    def test_reference_display_names_restore_groups_without_overwriting_existing_names(self) -> None:
        existing = InspectionItem(
            "roi1",
            "Hole",
            "cam1",
            "roi1",
            task_group="",
        )
        groups = _task_groups_from_display_names({"roi1": "Hole", "roi2": "roi2"})
        synced = sync_items_with_labels(
            [existing],
            ["roi1"],
            display_names_by_label={"roi1": "roi1"},
            task_groups_by_label={"roi1": "Hole"},
        )

        self.assertEqual(groups, {"roi1": "Hole"})
        self.assertEqual(synced[0].display_name, "Hole")
        self.assertEqual(synced[0].task_group, "Hole")

    def test_task_group_round_trip_and_effective_model_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inspection_items.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "item_id": "roi1",
                            "display_name": "Hole",
                            "camera_id": "cam1",
                            "roi_label": "roi1",
                            "task_group": "Hole",
                            "algorithm_code": "shared_backbone_register",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            item = load_inspection_items(str(path))[0]
            self.assertEqual(item.task_group, "Hole")
            self.assertEqual(item.model_key, "cam1__roi1")
            self.assertEqual(item.effective_model_key, "cam1__Hole")

            save_inspection_items([item], str(path))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["task_group"], "Hole")

    def test_status_recognizes_grouped_lt03_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cam1__Hole_register_model_lt03.npz"
            model_path.touch()
            algo = AlgorithmController()
            algo.product_params.learning_backbone = "mobilenet_v3_large"
            page = SimpleNamespace(
                algo=algo,
                session=SimpleNamespace(product_dir=temp_dir),
            )
            item = InspectionItem(
                "roi1",
                "Hole",
                "cam1",
                "roi1",
                task_group="Hole",
            )

            status, tooltip, _color = _inspection_item_status(page, item)

            self.assertIn("旧", status)
            self.assertIn(model_path.name, tooltip)

    def test_status_recognizes_legacy_ascii_key_for_chinese_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cam1__group_register_model_lt03.npz"
            model_path.touch()
            algo = AlgorithmController()
            algo.product_params.learning_backbone = "mobilenet_v3_large"
            page = SimpleNamespace(
                algo=algo,
                session=SimpleNamespace(product_dir=temp_dir),
            )
            item = InspectionItem(
                "roi33",
                "多料",
                "cam1",
                "roi33",
                task_group="多料",
            )

            status, tooltip, _color = _inspection_item_status(page, item)

            self.assertIn("旧", status)
            self.assertIn(model_path.name, tooltip)
            self.assertEqual(
                algo.tool_model_storage_keys(item.effective_model_key),
                ["cam1__多料", "cam1__group"],
            )

    def test_training_builder_combines_group_rois_into_one_task(self) -> None:
        items = [
            InspectionItem(
                "roi1",
                "Hole",
                "cam1",
                "roi1",
                task_group="Hole",
            ),
            InspectionItem(
                "roi2",
                "Hole",
                "cam1",
                "roi2",
                task_group="Hole",
            ),
        ]
        statuses = {
            ("ok.png", "roi1"): "OK",
            ("ok.png", "roi2"): "OK",
            ("mixed.png", "roi1"): "NG",
            ("mixed.png", "roi2"): "OK",
        }
        owner = SimpleNamespace(
            algo=_TrainingAlgo(),
            inspection_items=items,
            train_files=["ok.png", "mixed.png"],
            ok_files=[],
            ng_files=[],
            session=SimpleNamespace(product_dir="product"),
            current_camera_role=lambda: "cam1",
            _filter_paths_for_camera=lambda paths, _role: paths,
            _path_has_roi_geometry=lambda path, label: (path, label) in statuses,
            _sample_roi_status_for_path=lambda path, _role, label: statuses[(path, label)],
        )

        task = TrainingTaskBuilder(owner).build_task_for_item(items[0])

        self.assertEqual(task["display_name"], "Hole")
        self.assertEqual(task["model_key"], "cam1__Hole")
        self.assertEqual(task["label_names"], ["roi1", "roi2"])
        self.assertEqual(
            task["ok_samples"],
            [("ok.png", "roi1"), ("ok.png", "roi2"), ("mixed.png", "roi2")],
        )
        self.assertEqual(task["ng_samples"], [("mixed.png", "roi1")])

    def test_runtime_uses_shared_key_but_keeps_one_result_per_roi(self) -> None:
        algo = _TrainingAlgo()
        items = [
            InspectionItem(
                "roi17",
                "Pusher",
                "cam1",
                "roi17",
                "meanhsv_h",
                task_group="Pusher",
            ),
            InspectionItem(
                "roi18",
                "Pusher",
                "cam1",
                "roi18",
                "meanhsv_h",
                task_group="Pusher",
            ),
        ]
        calls: list[tuple[str, str]] = []

        def predict_image(_path: str, **kwargs) -> dict[str, str]:
            calls.append((kwargs["labels_override"][0], kwargs["model_key_override"]))
            return {"pred": "OK", "roi": kwargs["labels_override"][0]}

        rows = _predict_grouped_items_from_path(
            path="sample.png",
            groups=_group_inspection_items(items, algo),
            match_ms=0.0,
            algo=algo,
            predict_image=predict_image,
            load_embedding_model=lambda *_args, **_kwargs: None,
        )

        self.assertEqual(calls, [("roi17", "cam1__Pusher"), ("roi18", "cam1__Pusher")])
        self.assertEqual([row["roi"] for row in rows], ["roi17", "roi18"])


if __name__ == "__main__":
    unittest.main()
