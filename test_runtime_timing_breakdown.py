from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.inspection_executor import InspectionExecutionRequest, InspectionExecutor
from inspection_items import InspectionItem
from result_aggregator import aggregate_runtime_outcome


class _FakePredictor:
    def predict_image(self, path: str, *, feat_net=None, labels_override=None) -> dict:
        return {
            "pred": "OK",
            "diff": 0.1234,
            "match_ms": 12.5,
            "infer_ms": 34.0,
            "total_ms": 46.8,
        }


class InspectionExecutorTimingTest(unittest.TestCase):
    def test_executor_exposes_structured_match_and_infer_timings(self) -> None:
        executor = InspectionExecutor(_FakePredictor())
        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam1",
                image_path="demo.png",
                items=[
                    InspectionItem(
                        item_id="roi1",
                        display_name="ROI1",
                        camera_id="cam1",
                        roi_label="roi1",
                    )
                ],
            )
        )

        self.assertAlmostEqual(response.match_ms, 12.5)
        self.assertAlmostEqual(response.infer_ms, 34.0)
        self.assertIn("match=12.5ms", response.detail)
        self.assertIn("infer=34.0ms", response.detail)


class AggregateRuntimeTimingBreakdownTest(unittest.TestCase):
    def test_aggregate_runtime_outcome_sums_stage_timings(self) -> None:
        result = aggregate_runtime_outcome(
            product_name="demo",
            recipe_name="recipe.json",
            items=[
                InspectionItem(
                    item_id="roi1",
                    display_name="ROI1",
                    camera_id="cam1",
                    roi_label="roi1",
                ),
                InspectionItem(
                    item_id="roi2",
                    display_name="ROI2",
                    camera_id="cam2",
                    roi_label="roi2",
                ),
            ],
            active_roles=["cam1", "cam2"],
            camera_outcomes={
                "cam1": type(
                    "Outcome",
                    (),
                    {
                        "result": "OK",
                        "message": "cam1",
                        "capture_ms": 20.0,
                        "match_ms": 11.5,
                        "infer_ms": 31.0,
                    },
                )(),
                "cam2": type(
                    "Outcome",
                    (),
                    {
                        "result": "NG",
                        "message": "cam2",
                        "capture_ms": 22.0,
                        "match_ms": 13.5,
                        "infer_ms": 35.0,
                    },
                )(),
            },
            final_result="NG",
            duration_ms=95,
            capture_paths={"cam1": "cam1.png", "cam2": "cam2.png"},
        )

        self.assertAlmostEqual(result.capture_ms, 42.0)
        self.assertAlmostEqual(result.match_ms, 25.0)
        self.assertAlmostEqual(result.infer_ms, 66.0)
        self.assertEqual(result.duration_ms, 95)
        self.assertAlmostEqual(result.camera_results["cam1"].capture_ms, 20.0)
        self.assertAlmostEqual(result.camera_results["cam2"].infer_ms, 35.0)

        extra = result.to_record_extra_fields()
        self.assertEqual(extra["capture_ms"], 42.0)
        self.assertEqual(extra["match_ms"], 25.0)
        self.assertEqual(extra["infer_ms"], 66.0)
        self.assertEqual(extra["cam1_capture_ms"], 20.0)
        self.assertEqual(extra["cam2_infer_ms"], 35.0)


if __name__ == "__main__":
    unittest.main()
