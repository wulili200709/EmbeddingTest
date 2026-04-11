from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.preview_frame import (  # noqa: E402
    RuntimePreviewShape,
    build_runtime_preview_frame,
    export_runtime_preview_frame,
    read_exported_runtime_preview_shapes,
)


class RuntimePreviewFrameExportTest(unittest.TestCase):
    def test_export_runtime_preview_frame_writes_png_and_labelme_json(self) -> None:
        preview = build_runtime_preview_frame(
            role="cam1",
            image_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
            product_dir="demo_product",
            camera_role="cam1",
            roi_shapes=(
                RuntimePreviewShape(
                    label="roi1",
                    shape_type="rectangle",
                    points=((1.0, 2.0), (10.0, 12.0)),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = export_runtime_preview_frame(
                preview,
                tmpdir,
                stamp="20260331_120000_000001",
            )

            image_path = Path(exported.source_path)
            json_path = image_path.with_suffix(".json")

            self.assertTrue(image_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(image_path.name, "20260331_120000_000001_cam1.png")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["imagePath"], image_path.name)
            self.assertEqual(payload["imageWidth"], 32)
            self.assertEqual(payload["imageHeight"], 24)
            self.assertEqual(payload["shapes"][0]["label"], "roi1")

            shapes = read_exported_runtime_preview_shapes(str(image_path))
            self.assertEqual(len(shapes), 1)
            self.assertEqual(shapes[0].label, "roi1")


if __name__ == "__main__":
    unittest.main()
