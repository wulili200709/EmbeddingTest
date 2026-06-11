from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PySide6 import QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from line2dup.ui import template_page_pyside6 as template_page_module


class Line2DupReferenceRegionsSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_canvas_shape_change_emits_reference_region_sync_for_new_and_updated_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                emissions: list[str] = []
                dialog.referenceRegionsChanged.connect(lambda: emissions.append("changed"))

                with mock.patch.object(
                    dialog,
                    "_region_points_from_canvas",
                    return_value=("rectangle", [[1.0, 2.0], [11.0, 12.0]]),
                ):
                    dialog._on_reference_canvas_shape_changed()

                self.assertEqual(len(emissions), 1)
                self.assertEqual(len(dialog._reference_regions), 1)
                self.assertEqual(dialog._reference_regions[0]["output_label"], "roi1")

                with mock.patch.object(
                    dialog,
                    "_region_points_from_canvas",
                    return_value=("rectangle", [[5.0, 6.0], [15.0, 16.0]]),
                ):
                    dialog._on_reference_canvas_shape_changed()

                self.assertEqual(len(emissions), 2)
                self.assertEqual(dialog._reference_regions[0]["points"], [[5.0, 6.0], [15.0, 16.0]])
            finally:
                dialog.close()

    def test_new_overlapping_roi_does_not_replace_existing_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                original_points = [[10.0, 10.0], [30.0, 30.0]]
                dialog._reference_regions = [
                    {
                        "reference_label": "roi1",
                        "output_label": "roi1",
                        "display_name": "roi1",
                        "shape_type": "rectangle",
                        "points": original_points,
                    }
                ]
                dialog._selected_reference_idx = 0

                dialog._prepare_new_reference_roi()
                dialog._on_reference_canvas_pressed(
                    template_page_module._button_left(),
                    20,
                    20,
                )

                self.assertTrue(dialog._adding_reference_roi)
                self.assertIsNone(dialog._selected_reference_idx)

                with mock.patch.object(
                    dialog,
                    "_region_points_from_canvas",
                    return_value=("rectangle", [[20.0, 20.0], [40.0, 40.0]]),
                ):
                    dialog._on_reference_canvas_shape_changed()

                self.assertEqual(len(dialog._reference_regions), 2)
                self.assertEqual(dialog._reference_regions[0]["points"], original_points)
                self.assertEqual(
                    dialog._reference_regions[1]["points"],
                    [[20.0, 20.0], [40.0, 40.0]],
                )
                self.assertEqual(dialog._selected_reference_idx, 1)
                self.assertFalse(dialog._adding_reference_roi)
            finally:
                dialog.close()

    def test_loading_labelme_roi_preserves_recipe_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            image_path = product_dir / "reference.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"stub")
            image_path.with_suffix(".json").write_text("{}", encoding="utf-8")
            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                dialog.image_path = str(image_path)
                dialog._reference_regions = [
                    {
                        "reference_label": "roi34",
                        "output_label": "roi34",
                        "display_name": "Spring",
                        "shape_type": "rectangle",
                        "points": [[1.0, 2.0], [11.0, 12.0]],
                    }
                ]
                with mock.patch.object(
                    template_page_module.qr_core,
                    "list_shapes_from_labelme",
                    return_value=[
                        {
                            "label": "roi34",
                            "shape_type": "rectangle",
                            "points": [[5.0, 6.0], [15.0, 16.0]],
                        }
                    ],
                ):
                    dialog._load_reference_roi_from_json(silent=True)

                self.assertEqual(dialog._reference_regions[0]["output_label"], "roi34")
                self.assertEqual(dialog._reference_regions[0]["display_name"], "Spring")
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
