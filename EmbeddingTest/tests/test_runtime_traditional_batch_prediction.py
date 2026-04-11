from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime_context import ProductRuntimeContext  # noqa: E402
from domain.inspection_items import InspectionItem  # noqa: E402


class _FakeTraditionalAlgo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def is_learning_tool(self, code) -> bool:
        return False

    def resolve_tool_algorithm(self, code) -> str:
        return str(code or "").strip()

    def get_traditional_model_dict(self, algorithm: str, *, model_key: object = ""):
        self.calls.append((str(algorithm or "").strip(), str(model_key or "").strip()))
        if str(model_key or "").strip() == "cam1__hole":
            return {
                "algorithm": algorithm,
                "threshold": 0.0,
                "ok_when": "greater_equal",
                "ok_mean": 1.0,
                "ng_mean": -1.0,
                "accuracy": 1.0,
                "roi_label": "roi1",
            }
        return None


class RuntimeTraditionalBatchPredictionTest(unittest.TestCase):
    def test_predict_items_batch_batches_path_based_traditional_metrics(self) -> None:
        runtime = ProductRuntimeContext.__new__(ProductRuntimeContext)
        runtime.session = SimpleNamespace(product_dir="demo")
        runtime.algo = _FakeTraditionalAlgo()
        runtime._loc_method = ""
        runtime._line2dup_match_ms_by_image = {}

        items = [
            InspectionItem(
                item_id="roi1",
                display_name="hole",
                camera_id="cam1",
                roi_label="roi1",
                task_group="hole",
                algorithm_code="meanstd",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="hole",
                camera_id="cam1",
                roi_label="roi2",
                task_group="hole",
                algorithm_code="meanstd",
            ),
        ]

        captured: dict[str, object] = {}

        def _fake_compute_metrics_batch(path, *, preferred_labels, required_algorithms=None, json_path=None):
            captured["path"] = path
            captured["preferred_labels"] = list(preferred_labels)
            captured["required_algorithms"] = list(required_algorithms or [])
            return [
                {"roi_label": "roi1", "meanstd": 1.0},
                {"roi_label": "roi2", "meanstd": 1.0},
            ]

        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "application.runtime_context.compute_roi_metrics_batch",
            side_effect=_fake_compute_metrics_batch,
        ):
            rows = runtime.predict_items_batch("demo.png", items=items)

        self.assertEqual(captured["path"], "demo.png")
        self.assertEqual(captured["preferred_labels"], ["roi1", "roi2"])
        self.assertEqual(captured["required_algorithms"], ["meanstd", "meanstd"])
        self.assertEqual([row["roi_label"] for row in rows], ["roi1", "roi2"])
        self.assertEqual([row["pred"] for row in rows], ["OK", "OK"])
        self.assertEqual(
            runtime.algo.calls,
            [("meanstd", "cam1__hole")],
        )

    def test_predict_items_batch_from_frame_batches_current_item_roi_labels(self) -> None:
        runtime = ProductRuntimeContext.__new__(ProductRuntimeContext)
        runtime.session = SimpleNamespace(product_dir="demo")
        runtime.algo = _FakeTraditionalAlgo()
        runtime._loc_method = ""
        runtime._line2dup_match_ms_by_image = {}

        items = [
            InspectionItem(
                item_id="roi1",
                display_name="hole",
                camera_id="cam1",
                roi_label="roi1",
                task_group="hole",
                algorithm_code="meanstd",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="hole",
                camera_id="cam1",
                roi_label="roi2",
                task_group="hole",
                algorithm_code="meanstd",
            ),
        ]

        captured: dict[str, object] = {}

        def _fake_compute_metrics_batch(image_bgr, *, shape_by_label, preferred_labels, required_algorithms=None):
            captured["preferred_labels"] = list(preferred_labels)
            captured["required_algorithms"] = list(required_algorithms or [])
            return [
                {"roi_label": "roi1", "meanstd": 1.0},
                {"roi_label": "roi2", "meanstd": 1.0},
            ]

        with mock.patch(
            "application.runtime_context.compute_roi_metrics_batch_from_array",
            side_effect=_fake_compute_metrics_batch,
        ):
            prediction = runtime.predict_items_batch_from_frame(
                np.zeros((24, 32, 3), dtype=np.uint8),
                camera_role="cam1",
                items=items,
            )

        self.assertEqual(captured["preferred_labels"], ["roi1", "roi2"])
        self.assertEqual(captured["required_algorithms"], ["meanstd", "meanstd"])
        self.assertEqual([row["roi_label"] for row in prediction.rows], ["roi1", "roi2"])
        self.assertEqual([row["pred"] for row in prediction.rows], ["OK", "OK"])


if __name__ == "__main__":
    unittest.main()
