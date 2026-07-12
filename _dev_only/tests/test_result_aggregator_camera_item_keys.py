from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from domain.inspection_items import InspectionItem
from domain.inspection_models import InspectionItemResult
from domain.result_aggregator import aggregate_runtime_outcome


class ResultAggregatorCameraItemKeysTest(unittest.TestCase):
    def test_same_item_id_across_cameras_keeps_each_camera_result(self) -> None:
        items = [
            InspectionItem(item_id="roi1", display_name="密封圈", camera_id="cam1", roi_label="roi1"),
            InspectionItem(item_id="roi1", display_name="密封圈", camera_id="cam2", roi_label="roi1"),
        ]
        item_results_by_camera = {
            "cam1": [
                InspectionItemResult(
                    item_id="roi1",
                    display_name="密封圈",
                    camera_id="cam1",
                    roi_label="roi1",
                    result="OK",
                )
            ],
            "cam2": [
                InspectionItemResult(
                    item_id="roi1",
                    display_name="密封圈",
                    camera_id="cam2",
                    roi_label="roi1",
                    result="NG",
                )
            ],
        }

        result = aggregate_runtime_outcome(
            product_name="demo",
            recipe_name="recipe.json",
            items=items,
            active_roles=["cam1", "cam2"],
            camera_outcomes={},
            final_result="NG",
            duration_ms=12,
            item_results_by_camera=item_results_by_camera,
        )

        rows = {(row.camera_id, row.item_id): row.result for row in result.item_results}
        self.assertEqual(rows[("cam1", "roi1")], "OK")
        self.assertEqual(rows[("cam2", "roi1")], "NG")
        self.assertEqual(
            result.to_record_extra_fields(),
            {
                "cam1.密封圈": "OK",
                "cam2.密封圈": "NG",
            },
        )


if __name__ == "__main__":
    unittest.main()
