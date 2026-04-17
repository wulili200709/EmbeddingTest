from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from domain.inspection_items import InspectionItem, sync_items_with_labels
from domain.recipe_manager import inspection_item_specs_from_line2dup_recipe
from line2dup.core.recipe import Line2DupRecipe


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
                algorithm_code="shared_backbone_register",
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
        self.assertEqual(synced[0].algorithm_code, "shared_backbone_register")
        self.assertEqual(synced[1].display_name, "SpringL2")
        self.assertEqual(synced[1].roi_label, "roi2")

    def test_sync_items_can_default_task_group_from_reference_names(self) -> None:
        synced = sync_items_with_labels(
            [],
            ["roi1", "roi2", "roi9"],
            display_names_by_label={
                "roi1": "Hole",
                "roi2": "Hole",
                "roi9": "Pusher",
            },
            task_groups_by_label={
                "roi1": "Hole",
                "roi2": "Hole",
                "roi9": "Pusher",
            },
        )

        self.assertEqual([item.task_group for item in synced], ["Hole", "Hole", "Pusher"])
        self.assertEqual(synced[0].effective_model_key, "cam1__Hole")
        self.assertEqual(synced[2].effective_model_key, "cam1__Pusher")

    def test_sync_items_replaces_stale_roi_task_group_with_reference_name(self) -> None:
        existing_items = [
            InspectionItem(
                item_id="roi1",
                display_name="Hole",
                camera_id="cam1",
                roi_label="roi1",
                task_group="roi1",
                algorithm_code="shared_backbone_register",
            )
        ]

        synced = sync_items_with_labels(
            existing_items,
            ["roi1"],
            display_names_by_label={"roi1": "Hole"},
            task_groups_by_label={"roi1": "Hole"},
        )

        self.assertEqual(synced[0].task_group, "Hole")
        self.assertEqual(synced[0].effective_model_key, "cam1__Hole")

    def test_recipe_array_count_expands_item_specs(self) -> None:
        recipe = Line2DupRecipe(
            array_count=3,
            reference_regions=[
                {
                    "reference_label": "roi1",
                    "output_label": "roi1",
                    "display_name": "Pusher",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [10, 10]],
                },
                {
                    "reference_label": "roi2",
                    "output_label": "roi2",
                    "display_name": "Hole",
                    "shape_type": "rectangle",
                    "points": [[20, 20], [30, 30]],
                },
            ],
        )

        specs = inspection_item_specs_from_line2dup_recipe(recipe)

        self.assertEqual(
            specs,
            [
                {"roi_label": "roi1__01", "display_name": "Pusher #1"},
                {"roi_label": "roi2__01", "display_name": "Hole #1"},
                {"roi_label": "roi1__02", "display_name": "Pusher #2"},
                {"roi_label": "roi2__02", "display_name": "Hole #2"},
                {"roi_label": "roi1__03", "display_name": "Pusher #3"},
                {"roi_label": "roi2__03", "display_name": "Hole #3"},
            ],
        )

    def test_legacy_algorithm_type_payload_maps_to_algorithm_code(self) -> None:
        item = InspectionItem.from_dict(
            {
                "item_id": "roi1",
                "display_name": "ROI1",
                "camera_id": "cam1",
                "roi_label": "roi1",
                "algorithm_type": "inherit_product",
                "enabled": True,
            }
        )

        self.assertEqual(item.algorithm_code, "shared_backbone_register")
        self.assertEqual(item.algorithm_type, "shared_backbone_register")


if __name__ == "__main__":
    unittest.main()
