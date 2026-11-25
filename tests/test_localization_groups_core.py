from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from algorithms import image_io
from shape.core.localization_groups import clear_reference_cache, localize_groups, LocalizationGroupError
from shape.core.recipe import ShapeRecipe
from shape.core.roi_follow import locate_and_follow, _reference_to_scene_affine
from shape.like_matcher import Match


class LocalizationGroupsCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        clear_reference_cache()
        self.reference_path = str(Path(self.temp.name) / "参考.png")
        rng = np.random.default_rng(514)
        self.reference = np.full((240, 340), 30, np.uint8)
        self.reference[60:112, 50:114] = rng.integers(20, 235, (52, 64), np.uint8)
        self.reference[120:172, 210:274] = rng.integers(20, 235, (52, 64), np.uint8)
        self.reference = cv2.GaussianBlur(self.reference, (3, 3), 0.8)
        self.assertTrue(image_io.imwrite(self.reference_path, self.reference))
        self.groups = [
            {"id": "left", "name": "左组", "points": [[50, 60], [114, 112]], "search_radius": 24},
            {"id": "right", "name": "右组", "points": [[210, 120], [274, 172]], "search_radius": 24},
        ]

    def moved_reference(self, left=(9, -7), right=(-11, 12)):
        moved = np.full_like(self.reference, 30)
        for group, (dx, dy) in zip(self.groups, (left, right)):
            (x0, y0), (x1, y1) = group["points"]
            moved[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = self.reference[y0:y1, x0:x1]
        return moved

    def test_independent_groups_under_rotation_and_scale(self):
        # cv2's independent image warp provides the oracle, including forward
        # transform direction. Delta must be rotated/scaled before scene use.
        transform = cv2.getRotationMatrix2D((170, 120), 19.0, 1.08)
        transform[:, 2] += [80, 60]
        scene = cv2.warpAffine(self.moved_reference(), transform, (500, 400), borderValue=30)
        results = localize_groups(scene, self.reference_path, self.groups, ["left", "right"], transform)
        for result, delta in zip(results, ([9, -7], [-11, 12])):
            self.assertTrue(result["valid"], result)
            np.testing.assert_allclose(result["delta_ref"], delta, atol=0.18)
            np.testing.assert_allclose(result["delta_scene"], transform[:, :2] @ delta, atol=0.2)

    def test_repeat_texture_is_rejected_instead_of_picking_a_cell(self):
        tile = self.reference[60:112, 50:66]
        self.reference[60:112, 50:114] = np.tile(tile, (1, 4))
        image_io.imwrite(self.reference_path, self.reference)
        scene = np.full_like(self.reference, 30)
        scene[60:112, 18:146] = np.tile(tile, (1, 8))
        result = localize_groups(scene, self.reference_path, self.groups[:1], ["left"], np.eye(3)[:2])[0]
        self.assertFalse(result["valid"])
        self.assertIn("定位不唯一", result["reason"])

    def test_search_boundary_missing_and_stale_groups_fail(self):
        boundary = dict(self.groups[0], search_radius=9)
        result = localize_groups(self.moved_reference(left=(9, 0)), self.reference_path,
                                 [boundary], ["left"], np.eye(3)[:2])[0]
        self.assertFalse(result["valid"])
        self.assertIn("搜索边界", result["reason"])
        results = localize_groups(self.reference, self.reference_path,
                                  [dict(self.groups[0], needs_rebuild=True)], ["left", "missing"], np.eye(3)[:2])
        self.assertTrue(all(not r["valid"] for r in results))
        self.assertIn("参考图已更换", results[0]["reason"])
        self.assertIn("不存在", results[1]["reason"])

    def test_outside_image_is_not_matched_against_black_padding(self):
        transform = np.array([[1, 0, -300], [0, 1, -300]], float)
        result = localize_groups(self.reference, self.reference_path, self.groups[:1], ["left"], transform)[0]
        self.assertFalse(result["valid"])
        self.assertIn("图像外", result["reason"])

    def test_unused_group_does_not_read_reference(self):
        with patch("shape.core.localization_groups.image_io.imread", side_effect=AssertionError("must not read")):
            results = localize_groups(self.reference, "missing.png", [dict(self.groups[0], points=[])], [], np.eye(3)[:2])
        self.assertFalse(results[0]["used"])

    def test_reference_patch_cache_refreshes_after_reference_replacement(self):
        real_read = image_io.imread
        with patch("shape.core.localization_groups.image_io.imread", wraps=real_read) as reader:
            for _ in range(2):
                result = localize_groups(self.reference, self.reference_path, self.groups[:1], ["left"], np.eye(3)[:2])[0]
                self.assertTrue(result["valid"])
            self.assertEqual(reader.call_count, 1)
            image_io.imwrite(self.reference_path, np.full_like(self.reference, 100))
            result = localize_groups(self.reference, self.reference_path, self.groups[:1], ["left"], np.eye(3)[:2])[0]
            self.assertFalse(result["valid"])
            self.assertIn("纹理不足", result["reason"])
            self.assertEqual(reader.call_count, 2)

    def fake_detector(self):
        return SimpleNamespace(get_template_meta=lambda *_: {"angle": 0.0, "scale": 1.0})

    def test_public_follow_binds_groups_and_preserves_global_roi(self):
        recipe = ShapeRecipe(reference_image=self.reference_path, localization_groups=self.groups,
                             reference_regions=[
                                 {"output_label": "a", "points": [[65, 75], [85, 95]], "group_id": "left"},
                                 {"output_label": "b", "points": [[225, 130], [245, 150]], "group_id": "right"},
                                 {"output_label": "global", "points": [[140, 90], [160, 110]]},
                             ])
        transform = np.array([[1, 0, 30], [0, 1, 20]], float)
        scene = cv2.warpAffine(self.moved_reference(), transform, (400, 300), borderValue=30)
        matched = Match(30, 20, 90, "part", 0)
        rect = SimpleNamespace(x=0, y=0, w=340, h=240)
        with patch("shape.core.roi_follow._best_match", return_value=matched), \
             patch("shape.core.roi_follow.load_class_source_geometry", return_value=(340, 240, rect)), \
             patch("shape.core.roi_follow.match_quad", return_value=[(30, 20), (369, 20), (369, 259), (30, 259)]):
            result = locate_and_follow(scene, self.reference_path, recipe, detector=self.fake_detector())
            np.testing.assert_allclose(result.regions[0].points[0], [104, 88], atol=0.1)
            np.testing.assert_allclose(result.regions[1].points[0], [244, 162], atol=0.1)
            np.testing.assert_allclose(result.regions[2].points[0], [170, 110], atol=1e-8)
            np.testing.assert_allclose(result.regions[0].global_points[0], [95, 95], atol=1e-8)
            recipe.localization_groups[0]["needs_rebuild"] = True
            with self.assertRaises(LocalizationGroupError) as raised:
                locate_and_follow(scene, self.reference_path, recipe, detector=self.fake_detector())
            self.assertFalse(raised.exception.preview_result.regions[0].valid)
            self.assertTrue(raised.exception.preview_result.regions[1].valid)
            self.assertEqual(raised.exception.group_results[0]["name"], "左组")

    def test_full_global_pose_applies_refinement_once(self):
        base_quad = np.array([[30, 20], [369, 20], [369, 259], [30, 259]], float)
        refinement = cv2.getRotationMatrix2D((200, 150), 3.0, 1.02)
        refinement[:, 2] += [4, -2]
        final_quad = base_quad @ refinement[:, :2].T + refinement[:, 2]
        matched = Match(30, 20, 90, "part", 0, refined_transform=refinement, refined_quad=final_quad.tolist())
        rect = SimpleNamespace(x=0, y=0, w=340, h=240)
        def quad(_detector, match):
            return final_quad.tolist() if match.refined_quad is not None else base_quad.tolist()
        with patch("shape.core.roi_follow.load_class_source_geometry", return_value=(340, 240, rect)), \
             patch("shape.core.roi_follow.match_quad", side_effect=quad):
            actual = _reference_to_scene_affine(self.fake_detector(), matched)
        points = np.array([[100, 80], [200, 180]], float)
        expected = (points + [30, 20]) @ refinement[:, :2].T + refinement[:, 2]
        np.testing.assert_allclose(points @ actual[:, :2].T + actual[:, 2], expected, atol=1e-8)

    def test_malformed_bound_roi_cannot_fall_back_to_global_success(self):
        recipe = ShapeRecipe(reference_image=self.reference_path, localization_groups=self.groups,
                             reference_regions=[{"output_label": "", "points": [], "group_id": "left"}])
        matched = Match(0, 0, 99, "part", 0)
        rect = SimpleNamespace(x=0, y=0, w=340, h=240)
        with patch("shape.core.roi_follow._best_match", return_value=matched), \
             patch("shape.core.roi_follow.load_class_source_geometry", return_value=(340, 240, rect)), \
             patch("shape.core.roi_follow.match_quad", return_value=[(0, 0), (339, 0), (339, 239), (0, 239)]):
            with self.assertRaises(LocalizationGroupError) as raised:
                locate_and_follow(self.reference, self.reference_path, recipe, detector=self.fake_detector())
        self.assertEqual(raised.exception.group_results[0]["id"], "left")
        self.assertIn("绑定 ROI", str(raised.exception))
        self.assertFalse(raised.exception.preview_result.regions[0].valid)


if __name__ == "__main__":
    unittest.main()
