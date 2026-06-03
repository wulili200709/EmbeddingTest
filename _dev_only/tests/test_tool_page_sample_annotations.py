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


class _SampleAnnotationHarness:
    _train_sample_paths_for_role = ToolPage._train_sample_paths_for_role
    _training_sample_groups_for_role = ToolPage._training_sample_groups_for_role
    _sample_annotation_store_path = ToolPage._sample_annotation_store_path
    _sample_annotation_path_key = ToolPage._sample_annotation_path_key
    _sample_roi_annotation_key = ToolPage._sample_roi_annotation_key
    _load_sample_roi_annotations = ToolPage._load_sample_roi_annotations
    _save_sample_roi_annotations = ToolPage._save_sample_roi_annotations
    _delete_sample_annotation_file = ToolPage._delete_sample_annotation_file
    _path_has_roi_geometry = ToolPage._path_has_roi_geometry
    _set_sample_roi_status_for_path = ToolPage._set_sample_roi_status_for_path
    _sample_roi_status_for_path = ToolPage._sample_roi_status_for_path
    _sample_annotation_progress_for_path = ToolPage._sample_annotation_progress_for_path
    _sample_annotation_state_for_path = ToolPage._sample_annotation_state_for_path
    _inspection_label_names_for_role = ToolPage._inspection_label_names_for_role
    _sample_annotation_counts_for_roi = ToolPage._sample_annotation_counts_for_roi
    _current_tool_sample_stats_text = ToolPage._current_tool_sample_stats_text
    _training_validation_text = ToolPage._training_validation_text

    def __init__(self, product_dir: str) -> None:
        self.session = SimpleNamespace(product_dir=product_dir)
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code="shared_backbone_register",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam1",
                roi_label="roi2",
                algorithm_code="shared_backbone_register",
            ),
        ]
        self.train_files: list[str] = []
        self.ok_files: list[str] = []
        self.ng_files: list[str] = []
        self.test_files: list[str] = []
        self._sample_roi_annotations_by_path: dict[str, dict[str, str]] = {}

    def current_camera_role(self) -> str:
        return "cam1"

    def _sample_paths_for_kind(self, kind: str, _camera_role=None) -> list[str]:
        if str(kind) == "train":
            return list(self.train_files)
        return list(self.test_files)

    def _selected_inspection_item(self):
        return self.inspection_items[0]


class ToolPageSampleAnnotationsTest(unittest.TestCase):
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

    def test_sample_roi_status_persists_to_json_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "cam1_sample.png"
            image_path.write_bytes(b"img")
            self._write_labelme_json(image_path, ["roi1", "roi2"])

            harness = _SampleAnnotationHarness(tmpdir)
            harness._set_sample_roi_status_for_path(str(image_path), "cam1", "roi1", "OK")
            harness._set_sample_roi_status_for_path(str(image_path), "cam1", "roi2", "NG")

            reloaded = _SampleAnnotationHarness(tmpdir)
            reloaded._load_sample_roi_annotations()

            self.assertEqual(reloaded._sample_roi_status_for_path(str(image_path), "cam1", "roi1"), "OK")
            self.assertEqual(reloaded._sample_roi_status_for_path(str(image_path), "cam1", "roi2"), "NG")
            self.assertEqual(reloaded._sample_annotation_state_for_path(str(image_path), "cam1"), "已完成")

    def test_tool_stats_and_validation_use_roi_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ok_path = Path(tmpdir) / "cam1_ok.png"
            ng_path = Path(tmpdir) / "cam1_ng.png"
            ok_path.write_bytes(b"ok")
            ng_path.write_bytes(b"ng")
            self._write_labelme_json(ok_path, ["roi1"])
            self._write_labelme_json(ng_path, ["roi1"])

            harness = _SampleAnnotationHarness(tmpdir)
            harness.train_files = [str(ok_path), str(ng_path)]
            harness._set_sample_roi_status_for_path(str(ok_path), "cam1", "roi1", "OK")

            self.assertEqual(
                harness._sample_annotation_counts_for_roi("roi1", "cam1"),
                (1, 0, 1),
            )
            self.assertIn("OK 1 / NG 0 / 未标注 1", harness._current_tool_sample_stats_text())
            self.assertIn("还有 1 张未标注", harness._training_validation_text())


if __name__ == "__main__":
    unittest.main()
