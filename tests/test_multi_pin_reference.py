"""Regression checks for multi-pin measurement with an external reference line."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from algorithms.measurement import (
    evaluate_multi_pin_tip_height,
    measure_multi_pin_tip_height_from_array,
    judge_multi_pin_tip_height,
    multi_pin_tip_judgment_payload,
)
from application.runtime_context import _measurement_execution_order, _measurement_execution_params
from domain import InspectionItem
from application import AlgorithmController
from application.runtime_context import ProductRuntimeContext
from ui.debug.tool_page.roi_measurement_overlays import measurement_overlays_for_path


class MultiPinReferenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((260, 640, 3), 255, dtype=np.uint8)
        cv2.rectangle(self.image, (20, 30), (620, 100), (0, 0, 0), -1)
        for index in range(8):
            x = 65 + index * 70
            cv2.rectangle(self.image, (x - 9, 95), (x + 9, 190), (0, 0, 0), -1)
            cv2.ellipse(self.image, (x, 190), (9, 10), 0, 0, 180, (0, 0, 0), -1)
        self.shapes = {"roi1": {"shape_type": "rectangle", "points": [[30, 110], [610, 230]]}}
        self.group = InspectionItem("roi1", "Pins", "cam1", "roi1", "multi_pin_tip_height",
                                    params={"expected_pin_count": 8, "reference_line_item_id": "roi2"})
        self.line = InspectionItem("roi2", "Housing", "cam1", "roi2", "find_line")
        self.line_row = {"pred": "OK", "measurement": {"line_segment": [[20, 30], [620, 30]]}}

    def measure(self, params):
        return measure_multi_pin_tip_height_from_array(self.image, shape_by_label=self.shapes, params=params)

    def test_reference_runs_first_but_output_identity_is_preserved(self):
        items = [self.group, self.line]
        self.assertEqual([item.item_id for item in _measurement_execution_order(items)], ["roi2", "roi1"])
        params = _measurement_execution_params(self.group, items, {self.line.model_key: self.line_row})
        result = self.measure(params)
        self.assertEqual(len(result.tip_points), 8)
        self.assertTrue(all(abs(distance - 170) < 2 for distance in result.distances_px))
        self.assertEqual(result.reference_line_segment, ((20.0, 30.0), (620.0, 30.0)))
        self.assertNotIn("_reference_line_segment", self.group.params)
        self.assertEqual(judge_multi_pin_tip_height(result, {**params, "lower_limit": 168, "upper_limit": 172})[0], "OK")
        self.assertEqual(judge_multi_pin_tip_height(result, {**params, "upper_limit": 160})[0], "NG")
        self.assertEqual(judge_multi_pin_tip_height(result, {**params, "expected_pin_count": 9})[0], "NG")

    def test_missing_or_failed_reference_does_not_fall_back(self):
        for reference, row in ((self.line, {}), (self.line, {"pred": "NG", "measurement": self.line_row["measurement"]})):
            params = _measurement_execution_params(self.group, [self.group, reference], {reference.model_key: row})
            with self.assertRaisesRegex(RuntimeError, "reference line"):
                self.measure(params)
        self.line.enabled = False
        params = _measurement_execution_params(self.group, [self.group, self.line], {self.line.model_key: self.line_row})
        with self.assertRaisesRegex(RuntimeError, "reference line"):
            self.measure(params)

    def test_external_reference_uses_current_line_result(self):
        row = {"pred": "OK", "measurement": {"line_segment": [[20, 50], [620, 50]]}}
        params = _measurement_execution_params(self.group, [self.group, self.line], {self.line.model_key: row})
        result = self.measure(params)
        self.assertEqual(len(result.tip_points), 8)
        self.assertTrue(all(abs(distance - 150) < 2 for distance in result.distances_px))

    def test_height_spacing_switches_and_custom_gap_specs(self):
        params = _measurement_execution_params(
            self.group, [self.group, self.line], {self.line.model_key: self.line_row}
        )
        result = self.measure(params)
        self.assertEqual(len(result.spacings_px), 7)
        self.assertTrue(all(abs(value - 70.0) < 1.0 for value in result.spacings_px))
        specs = [
            {"nominal": 70.0, "lower_tolerance": 1.0, "upper_tolerance": 1.0}
            for _index in range(7)
        ]

        spacing_only = {
            **params,
            "height_check_enabled": False,
            "spacing_check_enabled": True,
            "upper_limit": 1.0,
            "spacing_specs": specs,
        }
        evaluation = evaluate_multi_pin_tip_height(result, spacing_only)
        self.assertEqual(evaluation["pred"], "OK")
        self.assertTrue(all(evaluation["height_in_spec"]))

        wrong_specs = [dict(item) for item in specs]
        wrong_specs[3]["nominal"] = 100.0
        spacing_ng = evaluate_multi_pin_tip_height(
            result, {**spacing_only, "spacing_specs": wrong_specs}
        )
        self.assertEqual(spacing_ng["pred"], "NG")
        self.assertEqual(spacing_ng["spacing_results"][3]["pred"], "NG")

        height_only = evaluate_multi_pin_tip_height(
            result,
            {
                **params,
                "height_check_enabled": True,
                "spacing_check_enabled": False,
                "upper_limit": 160.0,
                "spacing_specs": wrong_specs,
            },
        )
        self.assertEqual(height_only["pred"], "NG")

        both = multi_pin_tip_judgment_payload(
            result,
            {
                **params,
                "height_check_enabled": True,
                "spacing_check_enabled": True,
                "lower_limit": 168.0,
                "upper_limit": 172.0,
                "spacing_specs": specs,
            },
        )
        self.assertEqual(both["pred"], "OK")
        self.assertEqual(len(both["pin_results"]), 8)
        self.assertEqual(len(both["spacing_results"]), 7)
        self.assertEqual(both["spacing_results"][0]["point_a"], both["pin_results"][0]["point"])
        self.assertEqual(both["spacing_results"][0]["point_b"], both["pin_results"][1]["point"])

    def test_frame_pipeline_computes_line_before_group_and_returns_original_order(self):
        self.line.params = {"line": {"direction": "top_down", "polarity": "bright_to_dark"}}
        shapes = {**self.shapes, "roi2": {"shape_type": "rectangle", "points": [[30, 20], [610, 45]]}}
        context = SimpleNamespace(algo=AlgorithmController(), loc_method_for_role=lambda role: "none",
                                  load_embedding_model=lambda *args, **kwargs: None)
        with patch("application.runtime_context._runtime_shape_by_label", return_value=shapes):
            batch = ProductRuntimeContext.predict_items_batch_from_frame(
                context, self.image, camera_role="cam1", items=[self.group, self.line],
            )
        self.assertEqual(batch.rows[0]["measurement"]["type"], "multi_pin_tip_height")
        self.assertEqual(batch.rows[0]["measurement"]["detected_pin_count"], 8)
        self.assertEqual(batch.rows[0]["pred"], "OK")
        self.assertEqual(batch.rows[1]["pred"], "OK")
        self.assertTrue(all(abs(value - 170) < 2 for value in batch.rows[0]["measurement"]["distances"]))

    def test_frame_pipeline_supports_spacing_only_mode(self):
        self.group.params.update(
            {
                "height_check_enabled": False,
                "spacing_check_enabled": True,
                "upper_limit": 1.0,
                "spacing_specs": [
                    {"nominal": 70.0, "lower_tolerance": 1.0, "upper_tolerance": 1.0}
                    for _index in range(7)
                ],
            }
        )
        self.line.params = {"line": {"direction": "top_down", "polarity": "bright_to_dark"}}
        shapes = {**self.shapes, "roi2": {"shape_type": "rectangle", "points": [[30, 20], [610, 45]]}}
        context = SimpleNamespace(
            algo=AlgorithmController(),
            loc_method_for_role=lambda role: "none",
            load_embedding_model=lambda *args, **kwargs: None,
        )
        with patch("application.runtime_context._runtime_shape_by_label", return_value=shapes):
            batch = ProductRuntimeContext.predict_items_batch_from_frame(
                context, self.image, camera_role="cam1", items=[self.group, self.line]
            )
        measurement = batch.rows[0]["measurement"]
        self.assertEqual(batch.rows[0]["pred"], "OK")
        self.assertFalse(measurement["height_check_enabled"])
        self.assertTrue(measurement["spacing_check_enabled"])
        self.assertEqual(len(measurement["spacing_results"]), 7)
        self.assertTrue(all(item["pred"] == "OK" for item in measurement["spacing_results"]))

        self.group.params["spacing_specs"][2]["nominal"] = 100.0
        with patch("application.runtime_context._runtime_shape_by_label", return_value=shapes):
            batch = ProductRuntimeContext.predict_items_batch_from_frame(
                context, self.image, camera_role="cam1", items=[self.group, self.line]
            )
        self.assertEqual(batch.rows[0]["pred"], "NG")
        self.assertEqual(batch.rows[0]["measurement"]["spacing_results"][2]["pred"], "NG")

    def test_each_pin_distance_gets_an_image_label(self):
        pin_results = [
            {
                "index": index,
                "point": [40.0 + index * 50.0, 200.0],
                "distance": 10.0 + index,
                "unit": "px",
                "pred": "NG" if index == 3 else "OK",
            }
            for index in range(1, 9)
        ]
        tool_page = SimpleNamespace(
            _current_result_rows=[
                {
                    "file_path": "sample.bmp",
                    "pred": "NG",
                    "measurement": {
                        "type": "multi_pin_tip_height",
                        "pin_results": pin_results,
                        "in_spec_points": [item["point"] for item in pin_results if item["pred"] == "OK"],
                        "out_of_spec_points": [item["point"] for item in pin_results if item["pred"] == "NG"],
                    },
                }
            ]
        )
        overlays = measurement_overlays_for_path(tool_page, "sample.bmp")
        labels = [overlay for overlay in overlays if overlay.shape_type == "point_text"]
        self.assertEqual(len(labels), 8)
        self.assertEqual(labels[0].text, "P1: 11.0px")
        self.assertEqual(labels[-1].text, "P8: 18.0px")
        self.assertNotEqual(labels[0].text_offset, labels[1].text_offset)
        self.assertEqual(labels[2].color.name().lower(), "#ff5252")

    def test_spacing_only_overlay_shows_each_gap_and_hides_heights(self):
        measurement = {
            "type": "multi_pin_tip_height",
            "height_check_enabled": False,
            "spacing_check_enabled": True,
            "in_spec_points": [[10.0, 20.0], [30.0, 20.0], [50.0, 20.0]],
            "pin_results": [
                {"index": 1, "point": [10.0, 20.0], "distance": 100.0, "unit": "px", "pred": "OK"},
                {"index": 2, "point": [30.0, 20.0], "distance": 100.0, "unit": "px", "pred": "OK"},
                {"index": 3, "point": [50.0, 20.0], "distance": 100.0, "unit": "px", "pred": "OK"},
            ],
            "spacing_results": [
                {"index": 1, "point_a": [10.0, 20.0], "point_b": [30.0, 20.0], "distance": 20.0, "unit": "px", "pred": "OK"},
                {"index": 2, "point_a": [30.0, 20.0], "point_b": [50.0, 20.0], "distance": 20.0, "unit": "px", "pred": "NG"},
            ],
        }
        tool_page = SimpleNamespace(
            _current_result_rows=[
                {"file_path": "sample.bmp", "pred": "NG", "measurement": measurement}
            ]
        )
        overlays = measurement_overlays_for_path(tool_page, "sample.bmp")
        labels = [overlay for overlay in overlays if overlay.shape_type == "point_text"]
        self.assertEqual([overlay.text for overlay in labels], ["P1-P2: 20.0px", "P2-P3: 20.0px"])
        self.assertEqual(labels[0].color.name().lower(), "#00e676")
        self.assertEqual(labels[1].color.name().lower(), "#ff5252")


if __name__ == "__main__":
    unittest.main()
