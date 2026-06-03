from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from algorithms.measurement import (  # noqa: E402
    fit_line_filtered,
    judge_edge_distance,
    judge_find_line,
    measure_edge_distance_from_array,
    measure_find_line_from_array,
)


class EdgeDistanceMeasurementTest(unittest.TestCase):
    def _image_and_shape(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, 35:76, :] = 220
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[10.0, 10.0], [100.0, 70.0]],
            }
        }
        return image, shape_by_label

    def test_find_line_measures_one_edge_in_one_roi(self) -> None:
        image, shape_by_label = self._image_and_shape()

        result = measure_find_line_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            params={
                "line": {
                    "direction": "left_right",
                    "edge_threshold": 30,
                    "scan_step": 2,
                    "blur_ksize": 0,
                    "min_points": 20,
                },
                "lower_limit": 24.0,
                "upper_limit": 26.0,
            },
        )

        self.assertEqual(result.roi_label, "roi1")
        self.assertAlmostEqual(result.position_px, 25.0, delta=0.5)
        self.assertGreaterEqual(result.line.point_count, 20)
        self.assertLess(result.line.residual, 0.5)
        self.assertEqual(result.roi_xywh, (10, 10, 90, 60))
        self.assertGreaterEqual(len(result.edge_points), 20)
        self.assertIsNotNone(result.line_segment)
        assert result.line_segment is not None
        for x, y in result.line_segment:
            self.assertGreaterEqual(x, 10.0)
            self.assertLessEqual(x, 99.0)
            self.assertGreaterEqual(y, 10.0)
            self.assertLessEqual(y, 69.0)
        payload = result.to_dict()
        self.assertIn("edge_points", payload)
        self.assertIn("line_segment", payload)

        pred, value, lower, upper, unit = judge_find_line(
            result,
            {"lower_limit": 24.0, "upper_limit": 26.0},
        )
        self.assertEqual(pred, "OK")
        self.assertAlmostEqual(value, result.position_px)
        self.assertEqual(lower, 24.0)
        self.assertEqual(upper, 26.0)
        self.assertEqual(unit, "px")

    def test_subpixel_find_line_uses_shen_style_detector(self) -> None:
        edge_x = 35.35
        xx = np.arange(120, dtype=np.float32)
        profile = 255.0 / (1.0 + np.exp(-(xx - edge_x) / 1.1))
        image_gray = np.repeat(profile.reshape(1, -1), 80, axis=0).astype(np.uint8)
        image = np.repeat(image_gray[:, :, None], 3, axis=2)
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[10.0, 10.0], [100.0, 70.0]],
            }
        }

        result = measure_find_line_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            algorithm="find_line_subpix",
            params={
                "line": {
                    "direction": "left_right",
                    "edge_threshold": 10,
                    "scan_step": 2,
                    "blur_ksize": 0,
                    "min_points": 20,
                },
            },
        )

        self.assertEqual(result.roi_label, "roi1")
        self.assertAlmostEqual(result.position_px, 25.85, delta=0.3)
        self.assertNotAlmostEqual(result.position_px, round(result.position_px), delta=0.05)
        self.assertGreaterEqual(result.line.point_count, 20)
        self.assertLess(result.line.residual, 0.5)

    def test_subpixel_find_line_prefers_dominant_peak_over_first_weak_peak(self) -> None:
        xx = np.arange(140, dtype=np.float32)
        profile = (
            55.0 / (1.0 + np.exp(-(xx - 30.0) / 1.0))
            + 180.0 / (1.0 + np.exp(-(xx - 55.0) / 1.0))
        )
        image_gray = np.repeat(profile.reshape(1, -1), 90, axis=0).astype(np.uint8)
        image = np.repeat(image_gray[:, :, None], 3, axis=2)
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[10.0, 10.0], [120.0, 80.0]],
            }
        }
        base_params = {
            "line": {
                "direction": "left_right",
                "polarity": "dark_to_bright",
                "edge_threshold": 5,
                "scan_step": 2,
                "blur_ksize": 0,
                "min_points": 20,
            },
        }

        first_result = measure_find_line_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            algorithm="find_line_subpix",
            params={"line": {**base_params["line"], "peak_selection": "first"}},
        )
        dominant_result = measure_find_line_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            algorithm="find_line_subpix",
            params=base_params,
        )

        self.assertLess(first_result.position_px, 30.0)
        self.assertGreater(dominant_result.position_px, 40.0)
        self.assertLess(dominant_result.line.residual, 0.5)

    def test_find_line_polarity_follows_reverse_horizontal_scan_direction(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, 50:, :] = 220
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[10.0, 10.0], [110.0, 70.0]],
            }
        }

        result = measure_find_line_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            params={
                "line": {
                    "direction": "right_left",
                    "polarity": "bright_to_dark",
                    "edge_threshold": 30,
                    "scan_step": 2,
                    "blur_ksize": 0,
                    "min_points": 20,
                },
            },
        )

        self.assertAlmostEqual(result.position_px, 40.0, delta=0.5)
        self.assertLess(result.line.residual, 0.5)

    def test_find_line_polarity_follows_reverse_vertical_scan_direction(self) -> None:
        image = np.zeros((120, 80, 3), dtype=np.uint8)
        image[55:, :, :] = 220
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[10.0, 10.0], [70.0, 110.0]],
            }
        }

        result = measure_find_line_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            params={
                "line": {
                    "direction": "bottom_up",
                    "polarity": "bright_to_dark",
                    "edge_threshold": 30,
                    "scan_step": 2,
                    "blur_ksize": 0,
                    "min_points": 20,
                },
            },
        )

        self.assertAlmostEqual(result.position_px, 45.0, delta=0.5)
        self.assertLess(result.line.residual, 0.5)

    def test_find_line_filters_obvious_outlier_points(self) -> None:
        true_points = np.asarray([(25.0, float(y)) for y in range(30)], dtype=np.float32)
        outliers = np.asarray([(60.0, 5.0), (60.0, 15.0), (60.0, 25.0)], dtype=np.float32)
        points = np.vstack([true_points, outliers])

        line, filtered_points = fit_line_filtered(points, min_points=20)

        self.assertLess(len(filtered_points), len(points))
        self.assertEqual(line.point_count, len(filtered_points))
        self.assertAlmostEqual(line.x0, 25.0, delta=0.2)
        self.assertLess(line.residual, 0.2)

    def test_measures_distance_between_two_vertical_edges(self) -> None:
        image, shape_by_label = self._image_and_shape()

        result = measure_edge_distance_from_array(
            image,
            shape_by_label=shape_by_label,
            preferred_label="roi1",
            params={
                "line_a": {
                    "direction": "left_right",
                    "edge_threshold": 30,
                    "scan_step": 2,
                    "blur_ksize": 0,
                    "min_points": 20,
                },
                "line_b": {
                    "direction": "right_left",
                    "edge_threshold": 30,
                    "scan_step": 2,
                    "blur_ksize": 0,
                    "min_points": 20,
                },
            },
        )

        self.assertEqual(result.roi_label, "roi1")
        self.assertAlmostEqual(result.distance_px, 41.0, delta=1.0)
        self.assertGreaterEqual(result.line_a.point_count, 20)
        self.assertGreaterEqual(result.line_b.point_count, 20)
        self.assertLess(result.line_a.residual, 0.5)
        self.assertLess(result.line_b.residual, 0.5)

        pred, value, lower, upper, unit = judge_edge_distance(
            result,
            {"lower_limit": 40.0, "upper_limit": 42.0},
        )
        self.assertEqual(pred, "OK")
        self.assertAlmostEqual(value, result.distance_px)
        self.assertEqual(lower, 40.0)
        self.assertEqual(upper, 42.0)
        self.assertEqual(unit, "px")

        pred, *_ = judge_edge_distance(
            result,
            {"lower_limit": 43.0, "upper_limit": 50.0},
        )
        self.assertEqual(pred, "NG")

        pred, value, lower, upper, unit = judge_edge_distance(
            result,
            {
                "pixel_size_mm": 0.01,
                "limit_unit": "mm",
                "lower_limit": 0.40,
                "upper_limit": 0.42,
            },
        )
        self.assertEqual(pred, "OK")
        self.assertAlmostEqual(value, result.distance_px * 0.01)
        self.assertEqual(lower, 0.40)
        self.assertEqual(upper, 0.42)
        self.assertEqual(unit, "mm")


if __name__ == "__main__":
    unittest.main()
