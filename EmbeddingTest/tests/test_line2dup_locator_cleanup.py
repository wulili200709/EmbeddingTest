from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from algorithms.labelme import upsert_labelme_shapes
from line2dup.core import locator
from line2dup.core.locator import _delete_stale_line2dup_roi_shapes
from line2dup.core.recipe import Line2DupRecipe
from line2dup.core.roi_follow import FollowRegion, FollowResult
from line2dup.like_matcher import Match


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

    def test_delete_stale_line2dup_roi_shapes_handles_array_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            json_path = Path(tmpdir) / "sample.json"
            image_path.write_bytes(b"")
            json_path.write_text(
                json.dumps(
                    {
                        "shapes": [
                            {"label": "roi1__01", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
                            {"label": "roi1__02", "shape_type": "rectangle", "points": [[10, 10], [20, 20]]},
                            {"label": "roi2__01", "shape_type": "rectangle", "points": [[20, 20], [30, 30]]},
                            {"label": "anchor", "shape_type": "rectangle", "points": [[2, 2], [6, 6]]},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            recipe = Line2DupRecipe(
                array_count=2,
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

            self.assertEqual(removed, ["roi2__01"])
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [shape["label"] for shape in payload["shapes"]],
                ["roi1__01", "roi1__02", "anchor"],
            )

    def test_autogen_writes_all_regions_in_one_labelme_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            product_dir = Path(tmpdir) / "product"
            role_dir = product_dir / "line2dup" / "cam1"
            role_dir.mkdir(parents=True)
            (role_dir / "model.json").write_text("{}", encoding="utf-8")

            image_path = Path(tmpdir) / "sample.png"
            cv2.imwrite(str(image_path), np.zeros((100, 120, 3), dtype=np.uint8))
            regions = [
                FollowRegion(
                    label_name=f"roi{index + 1}",
                    points=[
                        (float(index), 10.0),
                        (float(index + 5), 10.0),
                        (float(index + 5), 20.0),
                        (float(index), 20.0),
                    ],
                    bbox=(index, 10, 5, 10),
                    source_shape_type="rectangle",
                )
                for index in range(49)
            ]
            follow_result = FollowResult(
                match=Match(
                    x=0,
                    y=0,
                    similarity=100.0,
                    class_id="demo",
                    template_id=0,
                ),
                regions=regions,
                points=list(regions[0].points),
                bbox=regions[0].bbox,
                source_shape_type="rectangle",
            )
            recipe = Line2DupRecipe(
                reference_regions=[
                    {
                        "reference_label": region.label_name,
                        "output_label": region.label_name,
                        "shape_type": "rectangle",
                        "points": [[0, 0], [5, 10]],
                    }
                    for region in regions
                ]
            )

            with (
                mock.patch.object(locator, "locate_and_follow", return_value=follow_result),
                mock.patch.object(
                    locator,
                    "upsert_labelme_shapes",
                    wraps=upsert_labelme_shapes,
                ) as batch_upsert,
            ):
                run = locator.autogen_roi_json_from_line2dup_timed(
                    str(image_path),
                    "",
                    str(product_dir),
                    camera_role="cam1",
                    recipe=recipe,
                    detector=object(),
                )

            batch_upsert.assert_called_once()
            payload = json.loads(Path(run.jpath).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["shapes"]), 49)
            self.assertEqual(payload["shapes"][0]["label"], "roi1")
            self.assertEqual(payload["shapes"][-1]["label"], "roi49")


if __name__ == "__main__":
    unittest.main()
