from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from domain.inspection_items import InspectionItem
from ui.debug.tool_page.page import ToolPage


class _MixedGroupHarness:
    _normalized_sample_path = ToolPage._normalized_sample_path
    _sample_annotation_store_path = ToolPage._sample_annotation_store_path
    _sample_annotation_path_key = ToolPage._sample_annotation_path_key
    _sample_roi_annotation_key = ToolPage._sample_roi_annotation_key
    _invalidate_sample_annotation_state_cache = ToolPage._invalidate_sample_annotation_state_cache
    _invalidate_shape_lookup_cache = ToolPage._invalidate_shape_lookup_cache
    _shape_lookup_for_path = ToolPage._shape_lookup_for_path
    _shape_entry_for_path = ToolPage._shape_entry_for_path
    _save_sample_roi_annotations = ToolPage._save_sample_roi_annotations
    _path_has_roi_geometry = ToolPage._path_has_roi_geometry
    _set_sample_roi_status_for_path = ToolPage._set_sample_roi_status_for_path
    _sample_roi_status_for_path = ToolPage._sample_roi_status_for_path
    _effective_model_key_for_item = ToolPage._effective_model_key_for_item
    _group_items_for_inspection_item = ToolPage._group_items_for_inspection_item
    _inspection_label_names_for_role = ToolPage._inspection_label_names_for_role
    _training_samples_for_inspection_item = ToolPage._training_samples_for_inspection_item
    _training_validation_text = ToolPage._training_validation_text
    _resolve_training_algorithm = ToolPage._resolve_training_algorithm
    _train_sample_paths_for_role = ToolPage._train_sample_paths_for_role

    def __init__(self, product_dir: str) -> None:
        self.session = SimpleNamespace(product_dir=product_dir)
        self.algo = SimpleNamespace(
            tool_model_key=lambda value: str(value or "").strip(),
            is_learning_tool=lambda code: str(code or "").strip() == "shared_backbone_register",
            current_learning_backbone=lambda: "b0",
            resolve_tool_algorithm=lambda code: str(code or "").strip(),
            is_anomaly_tool=lambda _code: False,
        )
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="hole",
                camera_id="cam1",
                roi_label="roi1",
                task_group="hole",
                algorithm_code="shared_backbone_register",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="hole",
                camera_id="cam1",
                roi_label="roi2",
                task_group="hole",
                algorithm_code="shared_backbone_register",
            ),
        ]
        self.train_files: list[str] = []
        self.ok_files: list[str] = []
        self.ng_files: list[str] = []
        self.test_files: list[str] = []
        self._sample_roi_annotations_by_path: dict[str, dict[str, str]] = {}
        self._shape_lookup_cache_by_path: dict[str, dict[str, object]] = {}
        self._sample_annotation_state_cache: dict[tuple[str, str], str] = {}

    def current_camera_role(self) -> str:
        return "cam1"

    def _sample_paths_for_kind(self, kind: str, _camera_role=None) -> list[str]:
        if str(kind) == "train":
            return list(self.train_files)
        return list(self.test_files)

    def _selected_inspection_item(self):
        return self.inspection_items[0]


class ToolPageGroupMixedAnnotationsTest(unittest.TestCase):
    def _write_labelme_json(self, image_path: Path, labels: list[str]) -> None:
        payload = {
            "version": "5.0.0",
            "flags": {},
            "shapes": [
                {
                    "label": label,
                    "points": [[0, 0], [10, 10]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
                for label in labels
            ],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": 10,
            "imageWidth": 10,
        }
        image_path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_group_allows_mixed_ok_ng_annotations_in_same_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "cam1_group.png"
            image_path.write_bytes(b"img")
            self._write_labelme_json(image_path, ["roi1", "roi2"])

            harness = _MixedGroupHarness(tmpdir)
            harness.train_files = [str(image_path)]
            harness._set_sample_roi_status_for_path(str(image_path), "cam1", "roi1", "OK")
            harness._set_sample_roi_status_for_path(str(image_path), "cam1", "roi2", "NG")

            training = harness._training_samples_for_inspection_item(harness.inspection_items[0])

            self.assertEqual(training["ok_files"], [str(image_path)])
            self.assertEqual(training["ng_files"], [str(image_path)])
            self.assertEqual(training["ok_samples"], [(str(image_path), "roi1")])
            self.assertEqual(training["ng_samples"], [(str(image_path), "roi2")])
            self.assertEqual(training["missing_annotation_paths"], [])
            self.assertEqual(training["conflicting_status_paths"], [])
            self.assertIn("可以训练", harness._training_validation_text())


if __name__ == "__main__":
    unittest.main()
