from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.inspection_executor import InspectionExecutionRequest, InspectionExecutor
from domain.inspection_items import InspectionItem


class _BatchPredictor:
    def __init__(self) -> None:
        self.batch_calls: list[dict] = []
        self.single_calls: list[dict] = []

    def predict_items_batch(self, path: str, *, items, feat_net=None):
        self.batch_calls.append(
            {
                "path": path,
                "item_ids": [item.item_id for item in items],
            }
        )
        return [
            {"pred": "OK", "diff": 0.1, "match_ms": 11.0, "infer_ms": 4.0},
            {"pred": "NG", "diff": 0.6, "match_ms": 11.0, "infer_ms": 6.0},
        ]

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override=None,
        algorithm_override=None,
        model_key_override=None,
    ) -> dict:
        self.single_calls.append(
            {
                "path": path,
                "labels_override": list(labels_override or []),
                "algorithm_override": algorithm_override,
                "model_key_override": model_key_override,
            }
        )
        raise AssertionError("predict_image should not be called when batch predictor is available")


class _FrameBatchPredictor(_BatchPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.frame_batch_calls: list[dict] = []

    def predict_items_batch_from_frame(self, image_bgr, *, camera_role, items, feat_net=None):
        self.frame_batch_calls.append(
            {
                "camera_role": camera_role,
                "image_shape": tuple(getattr(image_bgr, "shape", ())),
                "item_ids": [item.item_id for item in items],
            }
        )
        return type(
            "FrameBatchPrediction",
            (),
            {
                "rows": [
                    {"pred": "OK", "diff": 0.1, "match_ms": 8.0, "infer_ms": 5.0},
                    {"pred": "NG", "diff": 0.6, "match_ms": 8.0, "infer_ms": 7.0},
                ],
                "roi_shapes": (),
            },
        )()

    def predict_items_batch(self, path: str, *, items, feat_net=None):
        raise AssertionError("path-based batch predictor should not be called when frame batch predictor is available")


class InspectionExecutorBatchPredictorTest(unittest.TestCase):
    def test_execute_prefers_batch_predictor_when_available(self) -> None:
        predictor = _BatchPredictor()
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
                        algorithm_code="shared_backbone_register",
                    ),
                ],
            )
        )

        self.assertEqual(
            predictor.batch_calls,
            [{"path": "demo.png", "item_ids": ["roi1", "roi2"]}],
        )
        self.assertEqual(predictor.single_calls, [])
        self.assertEqual(response.result, "NG")
        self.assertAlmostEqual(response.match_ms, 11.0)
        self.assertAlmostEqual(response.infer_ms, 10.0)
        self.assertEqual([item.result for item in response.item_results], ["OK", "NG"])

    def test_execute_prefers_frame_batch_predictor_when_image_is_provided(self) -> None:
        import numpy as np

        predictor = _FrameBatchPredictor()
        executor = InspectionExecutor(predictor)

        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam2",
                image_path="demo.png",
                image_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
                items=[
                    InspectionItem(
                        item_id="roi1",
                        display_name="ROI1",
                        camera_id="cam2",
                        roi_label="roi1",
                        algorithm_code="shared_backbone_register",
                    ),
                    InspectionItem(
                        item_id="roi2",
                        display_name="ROI2",
                        camera_id="cam2",
                        roi_label="roi2",
                        algorithm_code="shared_backbone_register",
                    ),
                ],
            )
        )

        self.assertEqual(
            predictor.frame_batch_calls,
            [{"camera_role": "cam2", "image_shape": (24, 32, 3), "item_ids": ["roi1", "roi2"]}],
        )
        self.assertEqual(response.result, "NG")
        self.assertAlmostEqual(response.match_ms, 8.0)
        self.assertAlmostEqual(response.infer_ms, 12.0)


if __name__ == "__main__":
    unittest.main()
