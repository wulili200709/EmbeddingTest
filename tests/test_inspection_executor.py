from __future__ import annotations

import unittest

from application.inspection_executor import (
    InspectionExecutionPlan,
    InspectionExecutionRequest,
    InspectionExecutor,
)
from domain import InspectionItem


class _Predictor:
    def predict_image(self, _path: str, **kwargs) -> dict:
        model_key = str(kwargs.get("model_key_override", "") or "")
        return {
            "pred": "NG" if model_key.endswith("second") else "OK",
            "detail": model_key,
            "match_ms": 2.0,
            "infer_ms": 3.0,
        }


class InspectionExecutorTests(unittest.TestCase):
    def test_plan_separates_primary_distance_and_disabled_items(self) -> None:
        items = [
            InspectionItem("first", "First", "cam1", "roi1", "meanintensity"),
            InspectionItem("distance", "Distance", "cam1", "roi2", "line_distance"),
            InspectionItem("disabled", "Disabled", "cam1", "roi3", "meanstd", enabled=False),
        ]

        plan = InspectionExecutionPlan.from_items(items)

        self.assertEqual([item.item_id for item in plan.enabled_items], ["first", "distance"])
        self.assertEqual([item.item_id for item in plan.predicted_items], ["first"])
        self.assertEqual([item.item_id for item in plan.distance_items], ["distance"])

    def test_executor_keeps_item_order_and_aggregates_ng(self) -> None:
        items = [
            InspectionItem("first", "First", "cam1", "roi1", "meanintensity"),
            InspectionItem("second", "Second", "cam1", "roi2", "meanstd"),
            InspectionItem("disabled", "Disabled", "cam1", "roi3", "meanstd", enabled=False),
        ]

        response = InspectionExecutor(_Predictor()).execute(
            InspectionExecutionRequest(camera_id="cam1", image_path="sample.bmp", items=items)
        )

        self.assertEqual(response.result, "NG")
        self.assertEqual(
            [item.result for item in response.item_results],
            ["OK", "NG", "DISABLED"],
        )


if __name__ == "__main__":
    unittest.main()
