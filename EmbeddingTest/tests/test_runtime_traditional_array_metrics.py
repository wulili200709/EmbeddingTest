from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import algorithms.traditional as traditional  # noqa: E402
from algorithms.traditional import compute_roi_metrics_batch_from_array, compute_roi_metrics_from_array  # noqa: E402


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

    def test_compute_roi_metrics_batch_from_array_reuses_frame_level_color_conversions(self) -> None:
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[4:18, 4:14, :] = (80, 90, 100)
        image[4:18, 18:28, :] = (120, 130, 140)
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[4.0, 4.0], [14.0, 18.0]],
            },
            "roi2": {
                "label": "roi2",
                "shape_type": "rectangle",
                "points": [[18.0, 4.0], [28.0, 18.0]],
            },
        }

        real_cvt_color = traditional.cv2.cvtColor
        calls: list[int] = []

        def _tracking_cvt_color(*args, **kwargs):
            calls.append(int(args[1]))
            return real_cvt_color(*args, **kwargs)

        with mock.patch.object(traditional.cv2, "cvtColor", side_effect=_tracking_cvt_color):
            rows = compute_roi_metrics_batch_from_array(
                image,
                shape_by_label=shape_by_label,
                preferred_labels=["roi1", "roi2"],
                required_algorithms=["meanstd", "meanhsv_h"],
            )

        self.assertEqual([row["roi_label"] for row in rows], ["roi1", "roi2"])
        self.assertEqual(
            calls,
            [traditional.cv2.COLOR_BGR2GRAY, traditional.cv2.COLOR_BGR2HSV],
        )


if __name__ == "__main__":
    unittest.main()
