from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from shape.like_matcher import (
    Feature,
    TemplateLevel,
    detector_from_dict,
    encode_png_base64,
)
from shape.core.services import ShapeLocateService


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


class ShapeLocateServiceCacheTest(unittest.TestCase):
    def test_unchanged_model_reuses_cached_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            model_path.write_text("first", encoding="utf-8")
            detectors = [object()]

            with patch("shape.core.services.load_detector_model", side_effect=detectors) as loader:
                service = ShapeLocateService()
                first = service.runner_for_model(str(model_path))
                second = service.runner_for_model(str(model_path))

            self.assertIs(first, second)
            self.assertEqual(loader.call_count, 1)

    def test_replaced_model_at_same_path_is_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            model_path.write_text("first", encoding="utf-8")
            detectors = [object(), object()]

            with patch("shape.core.services.load_detector_model", side_effect=detectors) as loader:
                service = ShapeLocateService()
                first = service.runner_for_model(str(model_path))
                replacement_path = Path(tmp) / "replacement.json"
                replacement_path.write_text("other", encoding="utf-8")
                os.replace(replacement_path, model_path)
                second = service.runner_for_model(str(model_path))

            self.assertIsNot(first, second)
            self.assertIs(first.detector, detectors[0])
            self.assertIs(second.detector, detectors[1])
            self.assertEqual(loader.call_count, 2)

    def test_explicit_invalidation_forces_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            model_path.write_text("same", encoding="utf-8")
            detectors = [object(), object()]

            with patch("shape.core.services.load_detector_model", side_effect=detectors) as loader:
                service = ShapeLocateService()
                first = service.runner_for_model(str(model_path))
                self.assertTrue(service.invalidate_model(str(model_path)))
                second = service.runner_for_model(str(model_path))

            self.assertIsNot(first, second)
            self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
