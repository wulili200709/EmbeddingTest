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

    def test_measurement_item_result_participates_in_ok_ng(self) -> None:
        class _MeasurementPredictor:
            def predict_image(self, path: str, **kwargs) -> dict:
                params = dict(kwargs.get("params_override") or {})
                value = 41.0
                lower = float(params.get("lower_limit", 0.0))
                upper = float(params.get("upper_limit", 999.0))
                return {
                    "pred": "OK" if lower <= value <= upper else "NG",
                    "value": value,
                    "threshold": upper,
                    "detail": f"distance={value:.3f}px spec={lower:.1f}..{upper:.1f}px",
                }

        executor = InspectionExecutor(_MeasurementPredictor())
        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam1",
                image_path="demo.png",
                items=[
                    InspectionItem(
                        item_id="width",
                        display_name="Width",
                        camera_id="cam1",
                        roi_label="roi1",
                        algorithm_code="edge_distance",
                        params={"lower_limit": 40.0, "upper_limit": 42.0},
                    )
                ],
            )
        )

        self.assertEqual(response.result, "OK")
        self.assertEqual(response.item_results[0].result, "OK")
        self.assertIn("distance=41.000px", response.item_results[0].detail)

        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam1",
                image_path="demo.png",
                items=[
                    InspectionItem(
                        item_id="width",
                        display_name="Width",
                        camera_id="cam1",
                        roi_label="roi1",
                        algorithm_code="edge_distance",
                        params={"lower_limit": 45.0, "upper_limit": 50.0},
                    )
                ],
            )
        )

        self.assertEqual(response.result, "NG")
        self.assertEqual(response.item_results[0].result, "NG")

    def test_line_distance_item_uses_explicit_find_line_pair(self) -> None:
        class _FindLinePredictor:
            def predict_image(self, path: str, **kwargs) -> dict:
                label = kwargs.get("labels_override", [""])[0]
                x = 10.0 if label == "left" else 52.0
                return {
                    "pred": "OK",
                    "diff": 0.1,
                    "measurement": {
                        "roi_label": label,
                        "line_segment": [[x, 0.0], [x, 100.0]],
                        "edge_points": [[x, 0.0], [x, 50.0], [x, 100.0]],
                    },
                }

        executor = InspectionExecutor(_FindLinePredictor())
        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam1",
                image_path="demo.png",
                items=[
                    InspectionItem(
                        item_id="left",
                        display_name="Left",
                        camera_id="cam1",
                        roi_label="left",
                        algorithm_code="find_line",
                    ),
                    InspectionItem(
                        item_id="right",
                        display_name="Right",
                        camera_id="cam1",
                        roi_label="right",
                        algorithm_code="find_line",
                    ),
                    InspectionItem(
                        item_id="width",
                        display_name="Width",
                        camera_id="cam1",
                        roi_label="",
                        algorithm_code="line_distance",
                        params={
                            "line_a_item_id": "left",
                            "line_b_item_id": "right",
                            "lower_limit": 40.0,
                            "upper_limit": 44.0,
                        },
                    ),
                ],
            )
        )

        self.assertEqual(response.result, "OK")
        self.assertEqual(len(response.item_results), 3)
        distance_result = response.item_results[-1]
        self.assertEqual(distance_result.item_id, "width")
        self.assertEqual(distance_result.result, "OK")
        self.assertIn("distance=42.000px", distance_result.detail)
        assert response.raw_row is not None
        distance_row = response.raw_row["item_rows"][-1]
        self.assertEqual(distance_row["tool_name"], "Width")
        self.assertAlmostEqual(distance_row["value"], 42.0)
        self.assertEqual(distance_row["measurement"]["line_a_item_id"], "left")
        self.assertEqual(distance_row["measurement"]["line_b_item_id"], "right")
        self.assertIn("dimension_segment", distance_row["measurement"])
        self.assertEqual(len(response.measurements), 1)
        self.assertEqual(response.measurements[0]["type"], "line_distance")

        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam1",
                image_path="demo.png",
                items=[
                    InspectionItem(
                        item_id="left",
                        display_name="Left",
                        camera_id="cam1",
                        roi_label="left",
                        algorithm_code="find_line",
                    ),
                    InspectionItem(
                        item_id="right",
                        display_name="Right",
                        camera_id="cam1",
                        roi_label="right",
                        algorithm_code="find_line",
                    ),
                    InspectionItem(
                        item_id="width",
                        display_name="Width",
                        camera_id="cam1",
                        roi_label="",
                        algorithm_code="line_distance",
                        params={
                            "line_a_item_id": "left",
                            "line_b_item_id": "right",
                            "lower_limit": 45.0,
                            "upper_limit": 50.0,
                        },
                    ),
                ],
            )
        )

        self.assertEqual(response.result, "NG")
        self.assertEqual(response.item_results[-1].result, "NG")

    def test_line_distance_mm_can_use_bound_find_line_pixel_size(self) -> None:
        class _FindLinePredictor:
            def predict_image(self, path: str, **kwargs) -> dict:
                label = kwargs.get("labels_override", [""])[0]
                x = 10.0 if label == "left" else 52.0
                return {
                    "pred": "OK",
                    "measurement": {
                        "roi_label": label,
                        "line_segment": [[x, 0.0], [x, 100.0]],
                    },
                }

        executor = InspectionExecutor(_FindLinePredictor())
        response = executor.execute(
            InspectionExecutionRequest(
                camera_id="cam1",
                image_path="demo.png",
                items=[
                    InspectionItem(
                        item_id="left",
                        display_name="Left",
                        camera_id="cam1",
                        roi_label="left",
                        algorithm_code="find_line",
                        params={"pixel_size_mm": 0.05},
                    ),
                    InspectionItem(
                        item_id="right",
                        display_name="Right",
                        camera_id="cam1",
                        roi_label="right",
                        algorithm_code="find_line",
                        params={"pixel_size_mm": 0.05},
                    ),
                    InspectionItem(
                        item_id="width",
                        display_name="Width",
                        camera_id="cam1",
                        roi_label="",
                        algorithm_code="line_distance",
                        params={
                            "line_a_item_id": "left",
                            "line_b_item_id": "right",
                            "limit_unit": "mm",
                            "lower_limit": 2.0,
                            "upper_limit": 2.2,
                        },
                    ),
                ],
            )
        )

        self.assertEqual(response.result, "OK")
        assert response.raw_row is not None
        distance_row = response.raw_row["item_rows"][-1]
        self.assertAlmostEqual(distance_row["value"], 2.1)
        self.assertEqual(distance_row["measurement"]["unit"], "mm")
        self.assertAlmostEqual(distance_row["measurement"]["pixel_size_mm"], 0.05)
        self.assertIn("distance=2.100mm", response.item_results[-1].detail)


if __name__ == "__main__":
    unittest.main()
