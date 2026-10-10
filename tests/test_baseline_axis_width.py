from __future__ import annotations

import math
import unittest

from algorithms.measurement import BASELINE_AXIS_WIDTH_ALGORITHM
from application.inspection_executor import InspectionExecutor
from domain import InspectionItem


def _line_row(points: list[tuple[float, float]]) -> dict:
    return {
        "pred": "OK",
        "measurement": {
            "type": "find_line",
            "edge_points": [[float(x), float(y)] for x, y in points],
            "line_segment": [list(points[0]), list(points[-1])],
        },
    }


class BaselineAxisWidthTests(unittest.TestCase):
    def setUp(self) -> None:
        ys = [float(value) for value in range(0, 41, 2)]
        self.row_a = _line_row([(10.0 + 0.1 * (y - 20.0), y) for y in ys])
        self.row_b = _line_row([(110.0 - 0.2 * (y - 20.0), y) for y in ys])
        self.reference_row = _line_row([(0.0, 100.0), (200.0, 100.0)])
        self.segment_a = ((8.0, 0.0), (12.0, 40.0))
        self.segment_b = ((114.0, 0.0), (106.0, 40.0))
        self.reference_segment = ((0.0, 100.0), (200.0, 100.0))

    def test_width_is_measured_on_axis_at_common_midpoint(self) -> None:
        info = InspectionExecutor._baseline_axis_width_between_rows(
            row_a=self.row_a,
            row_b=self.row_b,
            reference_row=self.reference_row,
            segment_a=self.segment_a,
            segment_b=self.segment_b,
            reference_segment=self.reference_segment,
            axis_position_ratio=0.5,
            sample_half_width_px=6.0,
            item_a_id="left",
            item_b_id="right",
            reference_item_id="base",
        )

        self.assertAlmostEqual(info["distance_px"], 100.0, places=5)
        p0, p1 = info["dimension_segment"]
        self.assertAlmostEqual(p0[1], 20.0, places=5)
        self.assertAlmostEqual(p1[1], 20.0, places=5)
        self.assertEqual(info["reference_line_item_id"], "base")
        self.assertGreaterEqual(info["line_a_common_points"], 2)
        self.assertGreaterEqual(info["line_b_common_points"], 2)

    def test_axis_position_ratio_selects_requested_height(self) -> None:
        info = InspectionExecutor._baseline_axis_width_between_rows(
            row_a=self.row_a,
            row_b=self.row_b,
            reference_row=self.reference_row,
            segment_a=self.segment_a,
            segment_b=self.segment_b,
            reference_segment=self.reference_segment,
            axis_position_ratio=0.0,
            sample_half_width_px=6.0,
            item_a_id="left",
            item_b_id="right",
            reference_item_id="base",
        )

        p0, p1 = info["dimension_segment"]
        self.assertAlmostEqual(p0[1], 0.0, places=5)
        self.assertAlmostEqual(p1[1], 0.0, places=5)
        self.assertAlmostEqual(info["distance_px"], 106.0, places=4)

    def test_composite_row_keeps_limits_and_overlay_metadata(self) -> None:
        line_items = [
            InspectionItem("left", "Left", "cam2", "roi1", "find_line"),
            InspectionItem("right", "Right", "cam2", "roi2", "find_line"),
            InspectionItem("base", "Base", "cam2", "roi3", "find_line"),
        ]
        rows = {
            "left": self.row_a,
            "right": self.row_b,
            "base": self.reference_row,
        }
        distance_item = InspectionItem(
            "baseline_axis_width",
            "Baseline Axis Width",
            "cam2",
            "",
            BASELINE_AXIS_WIDTH_ALGORITHM,
            params={
                "line_a_item_id": "left",
                "line_b_item_id": "right",
                "reference_line_item_id": "base",
                "axis_position_ratio": 0.5,
                "sample_half_width_px": 6.0,
                "limit_unit": "px",
                "lower_limit": 99.0,
                "upper_limit": 101.0,
            },
        )

        row = InspectionExecutor._build_find_line_distance_row(
            distance_item=distance_item,
            line_items=line_items,
            line_rows_by_item_id=rows,
            camera_id="cam2",
            image_path="sample.png",
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["pred"], "OK")
        self.assertEqual(row["algorithm"], BASELINE_AXIS_WIDTH_ALGORITHM)
        self.assertAlmostEqual(row["value"], 100.0, places=3)
        measurement = row["measurement"]
        self.assertEqual(measurement["distance_mode"], "baseline_axis_width")
        self.assertEqual(measurement["reference_line_item_id"], "base")
        self.assertEqual(measurement["reference_line_segment"], [[0.0, 100.0], [200.0, 100.0]])
        p0, p1 = measurement["dimension_segment"]
        self.assertAlmostEqual(math.atan2(p1[1] - p0[1], p1[0] - p0[0]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
