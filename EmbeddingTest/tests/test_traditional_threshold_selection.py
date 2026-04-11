import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import algorithms.traditional as traditional


class TraditionalThresholdSelectionTest(unittest.TestCase):
    def test_selects_mid_gap_threshold_among_equal_accuracy_candidates(self) -> None:
        metric_map = {
            "ok1.png": 120.0,
            "ok2.png": 121.0,
            "ng1.png": 40.0,
            "ng2.png": 50.0,
        }

        def _fake_compute(path: str, preferred_label: str = "roi1"):
            return {
                "file_path": path,
                "file_name": path,
                "roi_label": preferred_label,
                "bbox_xywh": [0, 0, 10, 10],
                "meanintensity": metric_map[path],
                "mean_intensity": metric_map[path],
                "meanstd": metric_map[path] * 0.1,
                "mean_std": metric_map[path] * 0.1,
                "meanhsv_h": 0.0,
                "meanhsv_s": 0.0,
                "meanhsv_v": 0.0,
                "roi_area": 100,
            }

        with mock.patch.object(traditional, "compute_roi_metrics", side_effect=_fake_compute):
            model, rows = traditional.train_threshold_model(
                ["ok1.png", "ok2.png"],
                ["ng1.png", "ng2.png"],
                "meanintensity",
                preferred_label="roi1",
            )

        self.assertEqual(model.ok_when, "greater_equal")
        self.assertAlmostEqual(model.threshold, 85.0, places=4)
        self.assertAlmostEqual(model.accuracy, 1.0, places=8)
        self.assertEqual(len(rows), 4)

    def test_supports_mean_std_traditional_algorithm(self) -> None:
        metric_map = {
            "ok1.png": 12.0,
            "ok2.png": 14.0,
            "ng1.png": 3.0,
            "ng2.png": 4.0,
        }

        def _fake_compute(path: str, preferred_label: str = "roi1"):
            return {
                "file_path": path,
                "file_name": path,
                "roi_label": preferred_label,
                "bbox_xywh": [0, 0, 10, 10],
                "meanintensity": 0.0,
                "mean_intensity": 0.0,
                "meanstd": metric_map[path],
                "mean_std": metric_map[path],
                "meanhsv_h": 0.0,
                "meanhsv_s": 0.0,
                "meanhsv_v": 0.0,
                "roi_area": 100,
            }

        with mock.patch.object(traditional, "compute_roi_metrics", side_effect=_fake_compute):
            model, rows = traditional.train_threshold_model(
                ["ok1.png", "ok2.png"],
                ["ng1.png", "ng2.png"],
                "meanstd",
                preferred_label="roi1",
            )

        self.assertEqual(model.algorithm, "meanstd")
        self.assertEqual(model.ok_when, "greater_equal")
        self.assertAlmostEqual(model.threshold, 8.0, places=4)
        self.assertAlmostEqual(model.accuracy, 1.0, places=8)
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
