from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from algorithms.traditional import compute_roi_metrics_from_array  # noqa: E402


class RuntimeTraditionalArrayMetricsTest(unittest.TestCase):
    def test_compute_roi_metrics_from_array_does_not_emit_path_fields(self) -> None:
        image = np.zeros((20, 24, 3), dtype=np.uint8)
        image[4:12, 6:18, :] = 128

        metrics = compute_roi_metrics_from_array(
            image,
            shape_by_label={
                "roi1": {
                    "label": "roi1",
                    "shape_type": "rectangle",
                    "points": [[6.0, 4.0], [18.0, 12.0]],
                }
            },
            preferred_label="roi1",
        )

        self.assertEqual(metrics["roi_label"], "roi1")
        self.assertNotIn("file_path", metrics)
        self.assertNotIn("file_name", metrics)
        self.assertIn("meanintensity", metrics)
        self.assertIn("bbox_xywh", metrics)


if __name__ == "__main__":
    unittest.main()
