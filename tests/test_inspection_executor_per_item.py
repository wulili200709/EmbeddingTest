from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.inspection_executor import InspectionExecutionRequest, InspectionExecutor
from domain.inspection_items import InspectionItem


class _RecordingPredictor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override=None,
        algorithm_override=None,
        model_key_override=None,
    ) -> dict:
        label = labels_override[0] if labels_override else ""
        self.calls.append(
            {
                "path": path,
                "labels_override": list(labels_override or []),
                "algorithm_override": algorithm_override,
                "model_key_override": model_key_override,
            }
        )
        if label == "roi1":
            return {
                "pred": "OK",
                "diff": 0.10,
                "match_ms": 12.5,
                "infer_ms": 10.0,
            }
        if label == "roi2":
            return {
                "pred": "NG",
                "diff": 0.70,
                "match_ms": 12.5,
                "infer_ms": 8.0,
            }
        raise AssertionError(f"unexpected ROI label: {label}")


class InspectionExecutorPerItemTest(unittest.TestCase):
    def test_execute_runs_each_enabled_item_independently(self) -> None:
        predictor = _RecordingPredictor()
        executor = InspectionExecutor(predictor)

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
                        algorithm_code="shared_backbone_register",
                    ),
                    InspectionItem(
                        item_id="roi2",
                        display_name="ROI2",
                        camera_id="cam1",
                        roi_label="roi2",
                        algorithm_code="meanintensity",
                    ),
                    InspectionItem(
                        item_id="roi3",
                        display_name="ROI3",
                        camera_id="cam1",
                        roi_label="roi3",
                        algorithm_code="meanhsv_h",
                        enabled=False,
                    ),
                ],
            )
        )

        self.assertEqual(
            predictor.calls,
            [
                {
                    "path": "demo.png",
                    "labels_override": ["roi1"],
                    "algorithm_override": "shared_backbone_register",
                    "model_key_override": "cam1__roi1",
                },
                {
                    "path": "demo.png",
                    "labels_override": ["roi2"],
                    "algorithm_override": "meanintensity",
                    "model_key_override": "cam1__roi2",
                },
            ],
        )
        self.assertEqual(response.result, "NG")
        self.assertAlmostEqual(response.match_ms, 12.5)
        self.assertAlmostEqual(response.infer_ms, 18.0)
        self.assertIn("NG: ROI2", response.detail)
        self.assertIn("diff=0.7000", response.detail)
        self.assertIn("match=12.5ms", response.detail)
        self.assertIn("infer=18.0ms", response.detail)

        item_results = {item.item_id: item for item in response.item_results}
        self.assertEqual(item_results["roi1"].result, "OK")
        self.assertEqual(item_results["roi2"].result, "NG")
        self.assertEqual(item_results["roi3"].result, "DISABLED")
        self.assertEqual(item_results["roi2"].algorithm_code, "meanintensity")


if __name__ == "__main__":
    unittest.main()
