from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from line2dup.like_matcher import (
    Feature,
    TemplateLevel,
    detector_from_dict,
    encode_png_base64,
)


class Line2DupModelSingleLevelRepairTest(unittest.TestCase):
    def test_detector_from_dict_repairs_single_level_original_templates(self) -> None:
        level0 = TemplateLevel(
            width=7,
            height=7,
            tl_x=0,
            tl_y=0,
            pyramid_level=0,
            features=[
                Feature(x=2, y=2, label=0, theta=0.0),
                Feature(x=5, y=5, label=1, theta=45.0),
            ],
        )
        roi = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        model_dict = {
            "format": "line2dup_like_model_v2",
            "params": {
                "num_features": 128,
                "T_levels": [4, 8],
                "weak_threshold": 30.0,
                "strong_threshold": 60.0,
            },
            "classes": {
                "demo": {
                    "source": {
                        "image_path": "",
                        "roi_png": encode_png_base64(roi),
                        "mask_png": encode_png_base64(mask),
                        "roi_x": 0,
                        "roi_y": 0,
                        "roi_w": 8,
                        "roi_h": 8,
                        "mask_rects": [],
                    },
                    "pose_infos": {"items": [{"angle": 0.0, "scale": 1.0}], "ui": {}},
                    "original_mode": "manual_points",
                    "meta": [{"angle": 0.0, "scale": 1.0}],
                    "backends": {
                        "original": [{"template_id": 0, "levels": [self._level_to_dict(level0)]}],
                        "fusion": [],
                        "fusionv2": [],
                        "sim3": [],
                    },
                    "original_editor_levels": [self._level_to_dict(level0)],
                }
            },
        }

        detector = detector_from_dict(model_dict)

        self.assertEqual(len(detector.backend_templates["original"]["demo"][0]), 2)
        self.assertEqual(len(detector.get_original_editor_levels("demo")), 2)

    @staticmethod
    def _level_to_dict(level: TemplateLevel) -> dict:
        return {
            "width": int(level.width),
            "height": int(level.height),
            "tl_x": int(level.tl_x),
            "tl_y": int(level.tl_y),
            "pyramid_level": int(level.pyramid_level),
            "features": [
                {
                    "x": int(feature.x),
                    "y": int(feature.y),
                    "label": int(feature.label),
                    "theta": float(feature.theta),
                }
                for feature in level.features
            ],
        }


if __name__ == "__main__":
    unittest.main()
