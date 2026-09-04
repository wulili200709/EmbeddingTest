from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ui.debug.tool_page import roi_ops
from ui.debug.tool_page.roi_annotation_controller import RoiAnnotationController


class _Owner:
    def __init__(self, product_dir: str, labels: list[str]) -> None:
        self.session = SimpleNamespace(product_dir=product_dir)
        self._sample_roi_annotations_by_path: dict[str, dict[str, str]] = {}
        self.inspection_items = [
            SimpleNamespace(
                camera_id="cam1",
                roi_label=label,
                enabled=True,
                algorithm_code="learning",
            )
            for label in labels
        ]
        self._labels = labels

    @staticmethod
    def current_camera_role() -> str:
        return "cam1"

    def _inspection_label_names_for_role(self, _role=None) -> list[str]:
        return list(self._labels)

    @staticmethod
    def _sample_paths_for_kind(_kind: str, _role: str) -> list[str]:
        return []


def _write_sidecar(image_path: str, labels: list[str]) -> None:
    sidecar = os.path.splitext(image_path)[0] + ".json"
    shapes = [
        {
            "label": label,
            "shape_type": "rectangle",
            "points": [[0, 0], [10, 10]],
        }
        for label in labels
    ]
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump({"shapes": shapes}, handle)


class RoiAnnotationPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.product_dir = self._temp_dir.name

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def image_path(self) -> str:
        return str(Path(self.product_dir) / "sample_cam1.png")

    def test_geometry_sidecar_is_parsed_once_for_many_labels(self) -> None:
        image_path = self.image_path()
        labels = [f"roi{index}" for index in range(40)]
        _write_sidecar(image_path, labels)
        owner = _Owner(self.product_dir, labels)
        controller = RoiAnnotationController(owner)

        from common import labelme_io

        original = labelme_io.list_shapes_from_labelme
        calls = 0

        def counted_read(path: str, label_prefix=None):
            nonlocal calls
            calls += 1
            return original(path, label_prefix=label_prefix)

        with mock.patch.object(labelme_io, "list_shapes_from_labelme", counted_read):
            for label in labels:
                self.assertTrue(controller.has_geometry(image_path, label))
            controller.state_for_path(image_path, "cam1")

        self.assertEqual(calls, 1)

    def test_geometry_cache_refreshes_when_sidecar_changes(self) -> None:
        image_path = self.image_path()
        _write_sidecar(image_path, ["roi1"])
        controller = RoiAnnotationController(_Owner(self.product_dir, ["roi1", "roi2"]))

        self.assertTrue(controller.has_geometry(image_path, "roi1"))
        self.assertFalse(controller.has_geometry(image_path, "roi2"))

        _write_sidecar(image_path, ["roi1", "roi2"])

        self.assertTrue(controller.has_geometry(image_path, "roi2"))

    def test_mark_all_and_clear_each_save_only_once(self) -> None:
        image_path = self.image_path()
        labels = [f"roi{index}" for index in range(25)]
        _write_sidecar(image_path, labels)
        owner = _Owner(self.product_dir, labels)
        controller = RoiAnnotationController(owner)
        save_calls = 0

        def counted_save() -> None:
            nonlocal save_calls
            save_calls += 1

        with mock.patch.object(controller, "save", counted_save):
            controller.mark_all_ok(image_path, "cam1")
            self.assertEqual(save_calls, 1)
            self.assertEqual(
                len(owner._sample_roi_annotations_by_path[os.path.normpath(image_path)]),
                len(labels),
            )

            controller.clear_path(image_path, "cam1")

        self.assertEqual(save_calls, 2)
        self.assertNotIn(os.path.normpath(image_path), owner._sample_roi_annotations_by_path)

    def test_main_canvas_builds_all_roi_overlays_from_one_parse(self) -> None:
        image_path = self.image_path()
        labels = [f"roi{index}" for index in range(40)]
        _write_sidecar(image_path, labels)

        class Canvas:
            def clear_roi(self) -> None:
                pass

            def set_roi_style(self, **_kwargs) -> None:
                pass

            def set_roi_rect(self, _xywh) -> None:
                pass

            def set_roi_polygon(self, _points) -> None:
                pass

            def set_overlays(self, overlays) -> None:
                self.overlays = overlays

        canvas = Canvas()
        combo = SimpleNamespace(setCurrentText=lambda _value: None)
        tool_page = SimpleNamespace(
            canvas=canvas,
            cmb_shape=combo,
            inspection_items=[],
            loc_method="ncc",
            current_camera_role=lambda: "cam1",
            loc_method_for_role=lambda _role: "ncc",
            shape_recipe_for_role=lambda _role: None,
            _selected_inspection_item=lambda: None,
            _roi_status_for_path=lambda _path, _label: "",
            _on_shapes_changed=lambda: None,
        )
        tool_page._set_overlay_shapes = lambda path, label, **kwargs: roi_ops._set_overlay_shapes(
            tool_page,
            path,
            label,
            **kwargs,
        )

        from common import labelme_io

        original = labelme_io.list_shapes_from_labelme
        calls = 0

        def counted_read(path: str, label_prefix=None):
            nonlocal calls
            calls += 1
            return original(path, label_prefix=label_prefix)

        with (
            mock.patch.object(labelme_io, "list_shapes_from_labelme", counted_read),
            mock.patch.object(roi_ops, "measurement_overlays_for_path", return_value=[]),
        ):
            roi_ops._load_shape_for_label(tool_page, image_path, "roi1")

        self.assertEqual(calls, 1)
        self.assertEqual(len(canvas.overlays), len(labels) - 1)


if __name__ == "__main__":
    unittest.main()
