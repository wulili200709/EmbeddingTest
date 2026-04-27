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


if __name__ == "__main__":
    unittest.main()
