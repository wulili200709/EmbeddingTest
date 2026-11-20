import json
import math
import unittest

import numpy as np

from algorithms.measurement_opening_gap import OpeningGapRegion
from algorithms.measurement_opening_lines import measure_opening_lines


def synthetic(upper_slope=.2, lower_slope=.2, step=0.0, contrast=200.0):
    y, x = np.mgrid[:190, :240]
    upper = 60 + upper_slope * (x - 120) + step * (x >= 130)
    lower = 125 + lower_slope * (x - 120)
    sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return (25 + contrast * (1 - sigmoid(y - upper) + sigmoid(y - lower))).astype(np.float32)


REGION = OpeningGapRegion("opening", 120, (30, 95), (100, 165))


class OpeningLinesTests(unittest.TestCase):
    def test_parallel_slanted_edges_measure_normal_not_vertical_gap(self):
        image = synthetic()
        before = image.copy()
        result = measure_opening_lines(image, REGION)
        self.assertEqual(result["status"], "ok", result["reasons"])
        self.assertAlmostEqual(result["distance_px"], 65 / math.hypot(1, .2), delta=.1)
        self.assertAlmostEqual(result["reverse_distance_px"], result["distance_px"], delta=.1)
        self.assertGreater(65 - result["distance_px"], 1)
        start, end = np.asarray(result["dimension_segment"])
        self.assertAlmostEqual(float(np.dot(end - start, [1, .2])), 0, delta=.02)
        np.testing.assert_array_equal(image, before)
        json.dumps(result, allow_nan=False)

    def test_nonparallel_edges_preserve_both_directional_distances(self):
        result = measure_opening_lines(synthetic(.0, .4), REGION)
        self.assertEqual(result["status"], "review")
        self.assertIn("upper_lower_lines_not_parallel", result["reasons"])
        self.assertAlmostEqual(result["distance_px"], 65, delta=.1)
        self.assertAlmostEqual(result["reverse_distance_px"], 65 / math.hypot(1, .4), delta=.1)
        self.assertGreater(result["angle_difference_deg"], 20)

    def test_true_step_is_not_hidden_by_robust_fit(self):
        result = measure_opening_lines(synthetic(0, 0, step=8), REGION)
        self.assertIn(result["status"], ("review", "failed"))
        self.assertEqual(len(result["upper"]["points"]), 121)
        self.assertEqual(len(result["upper"]["inlier_mask"]), 121)
        retained = result["upper"]["strongest_points"]
        self.assertGreater(retained[-1][1] - retained[0][1], 7)

    def test_weaker_inner_edge_wins_over_brighter_backing_plate(self):
        image = np.full((190, 240), 225, np.float32)
        image[50:100] = 25
        image[100:115] = 125
        image[115:135] = 85
        region = OpeningGapRegion("backing", 120, (30, 70), (80, 155))
        result = measure_opening_lines(image, region)
        self.assertNotEqual(result["status"], "failed", result["reasons"])
        self.assertAlmostEqual(result["distance_px"], 50, delta=.3)
        self.assertGreater(np.median(np.asarray(result["lower"]["strongest_points"])[:, 1]), 130)
        self.assertLess(np.median(np.asarray(result["lower"]["points"])[:, 1]), 105)
        self.assertGreaterEqual(len(result["lower"]["branches"]), 2)

    def test_partly_supported_inner_branch_does_not_fall_back_to_plate(self):
        image = np.full((190, 240), 225, np.float32)
        image[50:135] = 25
        image[100:115, 80:120] = 125
        image[115:135, 80:120] = 85
        region = OpeningGapRegion("backing", 120, (30, 70), (80, 155))
        result = measure_opening_lines(image, region)
        self.assertEqual(result["status"], "failed")
        self.assertIn("lower:inner_branch_insufficient_support", result["reasons"])
        self.assertIsNone(result["distance_px"])
        self.assertIsNotNone(result["diagnostic_distance_px"])
        json.dumps(result, allow_nan=False)

    def test_low_contrast_and_absent_edges_do_not_become_good_measurements(self):
        weak = measure_opening_lines(synthetic(0, 0, contrast=50), REGION)
        self.assertEqual(weak["status"], "review")
        self.assertIn("upper:low_edge_contrast", weak["reasons"])
        absent = measure_opening_lines(np.full((190, 240), 100, np.uint8), REGION)
        self.assertEqual(absent["status"], "failed")
        self.assertIsNone(absent["distance_px"])
        self.assertFalse(any(absent["upper"]["inlier_mask"]))
        json.dumps(absent, allow_nan=False)

    def test_competing_edges_are_retained_and_flagged(self):
        image = np.full((190, 240), 225, np.float32)
        image[50:70] = 125
        image[70:125] = 25
        result = measure_opening_lines(image, REGION)
        self.assertEqual(result["status"], "review")
        self.assertIn("upper:multiple_competing_edges", result["reasons"])
        self.assertTrue(all(len(peaks) >= 2 for peaks in result["upper"]["candidate_peaks"]))

    def test_partial_support_stays_visible(self):
        image = synthetic(0, 0)
        image[:, 110:130] = 100
        result = measure_opening_lines(image, REGION)
        self.assertEqual(result["status"], "review")
        self.assertIn("upper:partial_edge_support", result["reasons"])
        self.assertIn("upper:uneven_horizontal_support", result["reasons"])
        self.assertEqual(result["upper"]["valid_columns"], 101)
        self.assertEqual(len(result["upper"]["points"]), 121)
        self.assertEqual(sum(result["upper"]["valid_mask"]), 101)


if __name__ == "__main__":
    unittest.main()
