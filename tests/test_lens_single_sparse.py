"""Sparse-board calibration tests against independent OpenCV camera projection."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import unittest

import cv2
import numpy as np

from algorithms.lens_single_calibration import (
    SingleCalibrationProfile, SingleUndistorter, calibrate_single_points,
)


class SparseSingleCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_size, cls.pattern_size = (960, 720), (13, 10)
        cls.camera = np.array([[960., 0., 479.5], [0., 960., 359.5], [0., 0., 1.]])
        cls.distortion = np.array([-.30, .05, 0., 0., 0.])
        cls.rotation = np.array([.27, -.24, .06])
        cls.translation = np.array([.025, -.012, 1.35])
        cls.indices = np.mgrid[0:13, 0:10].T.reshape(-1, 2)
        normalized = (cls.indices - [6., 4.5]) / 12.
        cls.all_points = cls.project(normalized)
        # Remove two rectangular code regions and scattered corners, preserving
        # the integer lattice and true outer bounds. No synthetic points fill holes.
        x, y = cls.indices.T
        keep = ~(((x >= 2) & (x <= 4) & (y >= 2) & (y <= 4)) |
                 ((x >= 8) & (x <= 10) & (y >= 5) & (y <= 7)) |
                 (((x + 2 * y) % 11 == 3) & (x > 0) & (x < 12) & (y > 0) & (y < 9)))
        cls.sparse_indices, cls.sparse_points = cls.indices[keep], cls.all_points[keep]
        cls.profile = calibrate_single_points(
            cls.sparse_points, cls.image_size, cls.pattern_size, mode="radial_perspective",
            grid_indices=cls.sparse_indices, grid_step_confirmed=True,
        )
        cls.full_profile = calibrate_single_points(
            cls.all_points, cls.image_size, cls.pattern_size, mode="radial_perspective",
        )

    @classmethod
    def project(cls, normalized_grid):
        world = np.column_stack((normalized_grid, np.zeros(len(normalized_grid))))
        pixels, _ = cv2.projectPoints(world, cls.rotation, cls.translation, cls.camera, cls.distortion)
        return pixels.reshape(-1, 2)

    def test_holes_keep_real_correspondences_and_generalize_to_unobserved_grid_positions(self):
        self.assertLess(len(self.profile.points), np.prod(self.pattern_size))
        np.testing.assert_array_equal(self.profile.points, self.sparse_points)
        np.testing.assert_array_equal(self.profile.grid_indices, self.sparse_indices)
        np.testing.assert_allclose(self.profile.radial_coeffs, [-.30, .05], atol=2e-4)
        rectifier = SingleUndistorter(self.profile)
        xx, yy = np.meshgrid(np.linspace(-.46, .46, 23), np.linspace(-.335, .335, 17))
        held_out = np.column_stack((xx.ravel(), yy.ravel()))
        expected = self.project(held_out)
        destination = rectifier.output_center + rectifier.plane_scale * held_out
        mx, my = destination[:, 0].astype(np.float32).reshape(-1, 1), destination[:, 1].astype(np.float32).reshape(-1, 1)
        actual = np.column_stack((
            cv2.remap(rectifier.map_x, mx, my, cv2.INTER_LINEAR).ravel(),
            cv2.remap(rectifier.map_y, mx, my, cv2.INTER_LINEAR).ravel(),
        ))
        errors = np.linalg.norm(actual - expected, axis=1)
        self.assertLess(float(np.sqrt(np.mean(errors ** 2))), .06)
        self.assertLess(float(errors.max()), .15)

    def test_v1_dense_and_v2_sparse_json_round_trip_without_filling_holes(self):
        for profile, schema in [(self.full_profile, "lc-lens-single-v1"), (self.profile, "lc-lens-single-v2")]:
            payload = profile.to_dict()
            self.assertEqual(payload["schema"], schema)
            restored = SingleCalibrationProfile.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
            self.assertEqual(restored.to_dict(), payload)
        self.assertNotIn("grid_indices", self.full_profile.to_dict())
        v2_missing_indices = self.profile.to_dict()
        del v2_missing_indices["grid_indices"]
        with self.assertRaises(ValueError):
            SingleCalibrationProfile.from_dict(v2_missing_indices)
        mislabeled_v1 = self.profile.to_dict()
        mislabeled_v1["schema"] = "lc-lens-single-v1"
        with self.assertRaises(ValueError):
            SingleCalibrationProfile.from_dict(mislabeled_v1)

    def test_duplicate_fractional_nonfinite_and_out_of_bounds_indices_are_rejected(self):
        invalid_indices = []
        repeated = self.sparse_indices.copy()
        repeated[1] = repeated[0]
        invalid_indices.append(repeated)
        fractional = self.sparse_indices.astype(float)
        fractional[1, 0] += .25
        invalid_indices.append(fractional)
        nonfinite = self.sparse_indices.astype(float)
        nonfinite[1, 0] = np.nan
        invalid_indices.append(nonfinite)
        negative = self.sparse_indices.copy()
        negative[0, 0] = -1
        invalid_indices.append(negative)
        outside = self.sparse_indices.copy()
        outside[0, 0] = 13
        invalid_indices.append(outside)
        invalid_indices.append(self.sparse_indices[:-1])
        for indices in invalid_indices:
            with self.subTest(indices=indices[:2].tolist()):
                with self.assertRaises(ValueError):
                    calibrate_single_points(self.sparse_points, self.image_size, self.pattern_size, grid_indices=indices)
                payload = copy.deepcopy(self.profile.to_dict())
                payload["grid_indices"] = indices.tolist()
                with self.assertRaises(ValueError):
                    SingleCalibrationProfile.from_dict(payload)

    def test_unconfirmed_grid_steps_disable_physical_units_without_changing_correction(self):
        unconfirmed = replace(self.profile, grid_step_confirmed=False)
        self.assertIsNone(unconfirmed.pixel_size_mm)
        self.assertGreater(self.profile.pixel_size_mm, 0.)
        confirmed_rectifier = SingleUndistorter(self.profile)
        unconfirmed_rectifier = SingleUndistorter(unconfirmed)
        self.assertIsNone(unconfirmed_rectifier.pixel_size_mm)
        np.testing.assert_array_equal(confirmed_rectifier.map_x, unconfirmed_rectifier.map_x)
        np.testing.assert_array_equal(confirmed_rectifier.map_y, unconfirmed_rectifier.map_y)
        doubled_pitch = replace(self.profile, spacing_mm=4.)
        self.assertAlmostEqual(doubled_pitch.pixel_size_mm, self.profile.pixel_size_mm * 2.)

    def test_step_confirmation_requires_observed_unit_neighbors_in_both_axes(self):
        payload = self.profile.to_dict()
        payload["grid_indices"] = (self.sparse_indices * 2).tolist()
        payload["pattern_size"] = [25, 19]
        payload["grid_step_confirmed"] = True
        with self.assertRaises(ValueError):
            SingleCalibrationProfile.from_dict(payload)
        payload["grid_step_confirmed"] = False
        restored = SingleCalibrationProfile.from_dict(payload)
        self.assertIsNone(restored.pixel_size_mm)

    def test_local_paper_like_warp_does_not_bypass_the_two_pixel_residual_gate(self):
        warped = self.sparse_points.copy()
        # A spatially oscillating deformation cannot be explained by one planar
        # homography plus radial lens distortion; recognizing points is not enough.
        columns, rows = self.sparse_indices.T
        warped[:, 0] += 5.0 * np.sin(columns * 1.7 + rows * .3)
        warped[:, 1] += 4.0 * np.cos(rows * 1.5 - columns * .2)
        with self.assertRaisesRegex(ValueError, "拟合误差过大"):
            calibrate_single_points(warped, self.image_size, self.pattern_size,
                                    grid_indices=self.sparse_indices, grid_step_confirmed=True)


if __name__ == "__main__":
    unittest.main()
