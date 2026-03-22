from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from inspection_items import InspectionItem, sync_items_with_labels
from line2dup_recipe import Line2DupRecipe
from domain.recipe_manager import inspection_item_specs_from_line2dup_recipe


class InspectionItemDisplayNameTest(unittest.TestCase):
    def test_recipe_regions_provide_display_name_specs(self) -> None:
        recipe = Line2DupRecipe(
            reference_regions=[
                {
                    "reference_label": "roi1",
                    "output_label": "roi1",
                    "display_name": "PusherL2",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [10, 10]],
                },
                {
                    "reference_label": "roi2",
                    "output_label": "roi2",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [20, 20]],
                },
            ]
        )

        specs = inspection_item_specs_from_line2dup_recipe(recipe)

        self.assertEqual(
            specs,
            [
                {"roi_label": "roi1", "display_name": "PusherL2"},
                {"roi_label": "roi2", "display_name": "roi2"},
            ],
        )

    def test_sync_items_uses_recipe_display_name_and_preserves_existing_fields(self) -> None:
        existing_items = [
            InspectionItem(
                item_id="roi1",
                display_name="旧名称",
                camera_id="cam2",
                roi_label="roi1",
                algorithm_type="inherit_product",
                enabled=True,
            )
        ]

        synced = sync_items_with_labels(
            existing_items,
            ["roi1", "roi2"],
            display_names_by_label={
                "roi1": "PusherL2",
                "roi2": "SpringL2",
            },
        )

        self.assertEqual(synced[0].display_name, "PusherL2")
        self.assertEqual(synced[0].camera_id, "cam2")
        self.assertEqual(synced[1].display_name, "SpringL2")
        self.assertEqual(synced[1].roi_label, "roi2")


if __name__ == "__main__":
    unittest.main()
