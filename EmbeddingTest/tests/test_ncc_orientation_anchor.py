from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ncc.model import (
    NccMatchBoundingBox,
    NccMatchModel,
    NccMatchRect,
    NccMatchResult,
    load_model,
    save_model,
)
from ncc.runtime_service import (
    _disambiguate_top_orientation,
    _refine_match_by_saturation_rect,
)


class NccOrientationAnchorTest(unittest.TestCase):
    def test_orientation_anchor_round_trips_with_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "model.json")
            save_model(
                model_path,
                NccMatchModel(
                    orientation_anchor=NccMatchRect(x=3, y=40, width=90, height=25),
                    pose_refinement="saturation_rect",
                ),
            )

            saved = load_model(model_path)

            self.assertIsNotNone(saved.orientation_anchor)
            self.assertEqual(saved.orientation_anchor.to_xywh(), (3, 40, 90, 25))
            self.assertEqual(saved.pose_refinement, "saturation_rect")

    def test_anchor_corrects_a_180_degree_ambiguous_match(self) -> None:
        rng = np.random.default_rng(3)
        template = np.full((100, 120), 50, dtype=np.uint8)
        for x in range(10, 111, 20):
            cv2.rectangle(template, (x, 15), (x + 9, 35), 120, -1)
            cv2.circle(template, (x + 5, 55), 6, 90, -1)
        template[70:100] = rng.integers(0, 256, size=(30, 120), dtype=np.uint8)

        scene = np.full((300, 320), 210, dtype=np.uint8)
        scene[90:190, 80:200] = template
        ambiguous = NccMatchResult(
            score=0.8,
            angle=180.0,
            center=(139.5, 139.5),
            quad=((199.0, 189.0), (80.0, 189.0), (80.0, 90.0), (199.0, 90.0)),
            bbox=NccMatchBoundingBox(x=80.0, y=90.0, width=119.0, height=99.0),
        )

        matches, checked = _disambiguate_top_orientation(
            scene,
            template,
            (ambiguous,),
            NccMatchRect(x=0, y=70, width=120, height=30),
        )

        self.assertTrue(checked)
        self.assertAlmostEqual(matches[0].angle, 0.0)
        self.assertAlmostEqual(matches[0].bbox.x, 80.0)
        self.assertAlmostEqual(matches[0].bbox.y, 90.0)

    def test_saturation_rect_refines_translation_scale_and_rotation(self) -> None:
        template = np.full((100, 120, 3), 230, dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (110, 90), (45, 110, 55), -1)
        cv2.rectangle(template, (35, 25), (85, 75), (20, 35, 20), -1)

        transform = cv2.getRotationMatrix2D((60.0, 50.0), -7.0, 1.08)
        transform[:, 2] += np.asarray([105.0, 85.0], dtype=np.float64)
        scene = cv2.warpAffine(
            template,
            transform,
            (360, 320),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(230, 230, 230),
        )
        template_corners = np.asarray(
            [[0.0, 0.0], [119.0, 0.0], [119.0, 99.0], [0.0, 99.0]],
            dtype=np.float32,
        )
        expected_quad = cv2.transform(template_corners.reshape(1, -1, 2), transform).reshape(-1, 2)
        initial_quad = expected_quad + np.asarray([7.0, 10.0], dtype=np.float32)
        initial_bbox = NccMatchBoundingBox(
            x=float(np.min(initial_quad[:, 0])),
            y=float(np.min(initial_quad[:, 1])),
            width=float(np.max(initial_quad[:, 0]) - np.min(initial_quad[:, 0])),
            height=float(np.max(initial_quad[:, 1]) - np.min(initial_quad[:, 1])),
        )
        initial = NccMatchResult(
            score=0.8,
            angle=-7.0,
            center=(
                initial_bbox.x + initial_bbox.width / 2.0,
                initial_bbox.y + initial_bbox.height / 2.0,
            ),
            quad=tuple((float(x), float(y)) for x, y in initial_quad),
            bbox=initial_bbox,
        )

        refined = _refine_match_by_saturation_rect(scene, template, initial)

        self.assertIsNotNone(refined)
        error = np.linalg.norm(
            np.asarray(refined.quad, dtype=np.float32) - expected_quad,
            axis=1,
        )
        self.assertLess(float(np.max(error)), 3.0)


if __name__ == "__main__":
    unittest.main()
