from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shape.core.locator import _delete_stale_line2dup_roi_shapes
from shape.core.recipe import Line2DupRecipe


class Line2DupLocatorCleanupTest(unittest.TestCase):
    def test_delete_stale_line2dup_roi_shapes_removes_only_obsolete_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            json_path = Path(tmpdir) / "sample.json"
            image_path.write_bytes(b"")
            json_path.write_text(
                json.dumps(
                    {
                        "shapes": [
                            {"label": "roi1", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
                            {"label": "roi2", "shape_type": "rectangle", "points": [[10, 10], [20, 20]]},
                            {"label": "anchor", "shape_type": "rectangle", "points": [[2, 2], [6, 6]]},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            recipe = Line2DupRecipe(
                reference_regions=[
                    {
                        "reference_label": "roi1",
                        "output_label": "roi1",
                        "display_name": "roi1",
                        "shape_type": "rectangle",
                        "points": [[0, 0], [10, 10]],
                    }
                ]
            )

            removed = _delete_stale_line2dup_roi_shapes(str(image_path), recipe)

            self.assertEqual(removed, ["roi2"])
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [shape["label"] for shape in payload["shapes"]],
                ["roi1", "anchor"],
            )


if __name__ == "__main__":
    unittest.main()
