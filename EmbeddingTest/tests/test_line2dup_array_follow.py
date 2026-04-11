from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from line2dup.core import roi_follow
from line2dup.core.recipe import Line2DupRecipe
from line2dup.core.template_core import RoiRect
from line2dup.like_matcher import Match


class _FakeDetector:
    def __init__(self, matches):
        self._matches = list(matches)

    def class_ids(self):
        return ["demo"]

    def match(self, scene_bgr, threshold, class_ids, mask, backend):
        return list(self._matches)

    def get_template_meta(self, class_id, template_id):
        return {"angle": 0.0, "scale": 1.0}


class Line2DupArrayFollowTest(unittest.TestCase):
    def test_locate_and_follow_expands_array_by_pitch_from_single_match(self) -> None:
        recipe = Line2DupRecipe(
            class_id="demo",
            nms_iou=0.3,
            array_count=3,
            array_pitch_x=100.0,
            array_pitch_y=0.0,
            reference_regions=[
                {
                    "reference_label": "roi1",
                    "output_label": "roi1",
                    "display_name": "roi1",
                    "shape_type": "rectangle",
                    "points": [[1, 2], [6, 8]],
                }
            ],
        )
        matches = [
            Match(
                x=110,
                y=220,
                similarity=80.0,
                class_id="demo",
                template_id=0,
                refined_quad=[(110.0, 220.0), (130.0, 220.0), (130.0, 240.0), (110.0, 240.0)],
            ),
            Match(
                x=10,
                y=20,
                similarity=98.0,
                class_id="demo",
                template_id=0,
                refined_quad=[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
            ),
        ]
        detector = _FakeDetector(matches)
        scene = np.zeros((300, 400, 3), dtype=np.uint8)
        roi_img = np.zeros((20, 20, 3), dtype=np.uint8)
        roi_mask = np.ones((20, 20), dtype=np.uint8) * 255

        with (
            mock.patch.object(
                roi_follow,
                "load_class_source_assets",
                return_value=({}, roi_img, roi_mask, RoiRect(x=0, y=0, w=20, h=20), []),
            ),
            mock.patch.object(
                roi_follow,
                "_estimate_patch_to_scene_homography",
                return_value=None,
            ),
            mock.patch.object(
                roi_follow,
                "match_quad",
                side_effect=lambda _detector, match: list(match.refined_quad or []),
            ),
            mock.patch.object(
                roi_follow,
                "nms_matches",
                side_effect=lambda _detector, values, iou_threshold: list(values),
            ),
        ):
            result = roi_follow.locate_and_follow(scene, "ref.png", recipe, detector=detector)

        self.assertEqual(result.instance_count, 3)
        self.assertEqual(result.expected_instance_count, 3)
        self.assertEqual(result.match.x, 10)
        self.assertEqual([region.label_name for region in result.regions], ["roi1__01", "roi1__02", "roi1__03"])
        self.assertEqual(result.regions[0].bbox, (11, 22, 5, 6))
        self.assertEqual(result.regions[1].bbox, (111, 22, 5, 6))
        self.assertEqual(result.regions[2].bbox, (211, 22, 5, 6))
        self.assertEqual(len(result.matches or []), 1)
        self.assertEqual(result.transform_mode, "affine")

    def test_locate_and_follow_uses_projective_homography_when_available(self) -> None:
        recipe = Line2DupRecipe(
            class_id="demo",
            nms_iou=0.3,
            array_count=2,
            array_pitch_x=10.0,
            array_pitch_y=0.0,
            reference_regions=[
                {
                    "reference_label": "roi1",
                    "output_label": "roi1",
                    "display_name": "roi1",
                    "shape_type": "rectangle",
                    "points": [[1, 2], [6, 8]],
                }
            ],
        )
        match = Match(
            x=10,
            y=20,
            similarity=98.0,
            class_id="demo",
            template_id=0,
            refined_quad=[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
        )
        detector = _FakeDetector([match])
        scene = np.zeros((300, 400, 3), dtype=np.uint8)
        roi_img = np.zeros((20, 20, 3), dtype=np.uint8)
        roi_mask = np.ones((20, 20), dtype=np.uint8) * 255
        projective = np.array(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.005, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        with (
            mock.patch.object(
                roi_follow,
                "load_class_source_assets",
                return_value=({}, roi_img, roi_mask, RoiRect(x=0, y=0, w=20, h=20), []),
            ),
            mock.patch.object(
                roi_follow,
                "_estimate_patch_to_scene_homography",
                return_value=projective,
            ),
            mock.patch.object(
                roi_follow,
                "match_quad",
                side_effect=lambda _detector, value: list(value.refined_quad or []),
            ),
            mock.patch.object(
                roi_follow,
                "nms_matches",
                side_effect=lambda _detector, values, iou_threshold: list(values),
            ),
        ):
            result = roi_follow.locate_and_follow(scene, "ref.png", recipe, detector=detector)

        self.assertEqual(result.transform_mode, "projective_homography")
        self.assertEqual([region.label_name for region in result.regions], ["roi1__01", "roi1__02"])
        self.assertEqual(result.regions[0].bbox, (10, 21, 6, 7))
        self.assertEqual(result.regions[1].bbox, (19, 20, 6, 7))


if __name__ == "__main__":
    unittest.main()
