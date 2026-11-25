"""Synthetic geometry checks for single-plane correction, not industrial accuracy claims."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from algorithms import lens_single_calibration as single
from algorithms.image_io import imwrite
from algorithms.lens_calibration import CalibrationProfile, manual_profile


def _known_camera(k1=-0.45, k2=0.08, pattern_size=(11, 8), image_size=(960, 720)):
    """Generate observations with OpenCV's independent 3D projection API."""
    width, height = image_size
    scale = max(image_size)
    camera = np.array(
        [[scale, 0.0, (width - 1) / 2], [0.0, scale, (height - 1) / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    rotation = np.array([0.27, -0.24, 0.06], dtype=np.float64)
    translation = np.array([0.025, -0.012, 1.35], dtype=np.float64)
    columns, rows = pattern_size
    grid_x, grid_y = np.meshgrid(np.arange(columns), np.arange(rows))
    grid = np.column_stack(
        (grid_x.ravel() - (columns - 1) / 2, grid_y.ravel() - (rows - 1) / 2)
    ).astype(np.float64) / max(columns - 1, rows - 1)

    def project(grid_points, coefficients=distortion):
        world_points = np.column_stack((grid_points, np.zeros(len(grid_points))))
        projected, _ = cv2.projectPoints(world_points, rotation, translation, camera, coefficients)
        return projected.reshape(-1, 2)

    return {
        "image_size": image_size,
        "pattern_size": pattern_size,
        "camera": camera,
        "distortion": distortion,
        "rotation": rotation,
        "translation": translation,
        "grid": grid,
        "project": project,
        "observed": project(grid),
        "ideal": project(grid, np.zeros(5)),
    }


def _held_out_grid(pattern_size):
    columns, rows = pattern_size
    extent_y = (rows - 1) / max(columns - 1, rows - 1) / 2
    xs, ys = np.meshgrid(np.linspace(-0.455, 0.455, 19), np.linspace(-extent_y * 0.906, extent_y * 0.906, 15))
    return np.column_stack((xs.ravel(), ys.ravel()))


def _sample_source_coordinates(rectifier, output_points):
    """Read where the production map samples, at independent validation positions."""
    output_points = np.asarray(output_points, np.float32)
    sample_x = output_points[:, 0].reshape(-1, 1)
    sample_y = output_points[:, 1].reshape(-1, 1)
    mapped_x = cv2.remap(rectifier.map_x, sample_x, sample_y, cv2.INTER_LINEAR).ravel()
    mapped_y = cv2.remap(rectifier.map_y, sample_x, sample_y, cv2.INTER_LINEAR).ravel()
    return np.column_stack((mapped_x, mapped_y))


def _render_board(fixture):
    """Inverse-project every sensor pixel into an ideal flat checkerboard texture."""
    columns, rows = fixture["pattern_size"]
    tile = 64
    texture = np.full(((rows + 1) * tile, (columns + 1) * tile), 255, np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                texture[row * tile:(row + 1) * tile, column * tile:(column + 1) * tile] = 0
    width, height = fixture["image_size"]
    ys, xs = np.indices((height, width), dtype=np.float32)
    sensor_points = np.stack((xs, ys), axis=2).reshape(-1, 1, 2)
    camera_rays = cv2.undistortPoints(
        sensor_points, fixture["camera"], fixture["distortion"]
    ).reshape(-1, 2)
    rotation, _ = cv2.Rodrigues(fixture["rotation"])
    plane_to_camera = np.column_stack((rotation[:, :2], fixture["translation"]))
    rays_homogeneous = np.column_stack((camera_rays, np.ones(len(camera_rays))))
    board_homogeneous = rays_homogeneous @ np.linalg.inv(plane_to_camera).T
    grid = board_homogeneous[:, :2] / board_homogeneous[:, 2:]
    grid_scale = max(columns - 1, rows - 1)
    # Texture boundaries are at half-integer pixel edges; the first internal
    # checkerboard corner is at tile - 0.5 in the texture coordinate system.
    texture_x = ((grid[:, 0] * grid_scale + (columns - 1) / 2 + 1) * tile - 0.5).reshape(height, width)
    texture_y = ((grid[:, 1] * grid_scale + (rows - 1) / 2 + 1) * tile - 0.5).reshape(height, width)
    image = cv2.remap(texture, texture_x.astype(np.float32), texture_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=255)
    return cv2.GaussianBlur(image, (3, 3), 0.55)


def _regular_grid_metrics(corners, pattern_size):
    columns, rows = pattern_size
    grid = corners.reshape(rows, columns, 2)
    horizontal = grid[:, 1:] - grid[:, :-1]
    vertical = grid[1:] - grid[:-1]
    horizontal_lengths = np.linalg.norm(horizontal, axis=2)
    vertical_lengths = np.linalg.norm(vertical, axis=2)
    x_axis = horizontal.mean(axis=(0, 1))
    y_axis = vertical.mean(axis=(0, 1))
    angle_error = abs(float(np.dot(x_axis, y_axis) / np.linalg.norm(x_axis) / np.linalg.norm(y_axis)))
    return (
        float(horizontal_lengths.std() / horizontal_lengths.mean()),
        float(vertical_lengths.std() / vertical_lengths.mean()),
        angle_error,
        abs(float(horizontal_lengths.mean() / vertical_lengths.mean()) - 1),
    )


class SingleCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _known_camera()
        cls.profile = single.calibrate_single_points(
            cls.fixture["observed"], cls.fixture["image_size"], cls.fixture["pattern_size"],
            spacing_mm=2.0, mode="radial_perspective",
        )

    def test_barrel_and_pincushion_correction_generalize_between_detected_corners(self):
        for k1 in (-0.45, 0.45):
            with self.subTest(k1=k1):
                fixture = _known_camera(k1=k1)
                profile = single.calibrate_single_points(
                    fixture["observed"], fixture["image_size"], fixture["pattern_size"],
                    mode="radial_perspective",
                )
                rectifier = single.SingleUndistorter(profile)
                held_out = _held_out_grid(fixture["pattern_size"])
                # Confirm these test positions are not the training corner set.
                nearest_training = np.linalg.norm(held_out[:, None] - fixture["grid"][None], axis=2).min(axis=1)
                self.assertGreater(float(nearest_training.min()), 0.001)
                output_positions = rectifier.output_center + rectifier.plane_scale * held_out
                actual_source = _sample_source_coordinates(rectifier, output_positions)
                expected_source = fixture["project"](held_out)
                errors = np.linalg.norm(actual_source - expected_source, axis=1)
                self.assertLess(float(np.sqrt(np.mean(errors ** 2))), 0.06)
                self.assertLess(float(errors.max()), 0.15)
                np.testing.assert_allclose(profile.radial_coeffs, [k1, 0.08], atol=0.002)

    def test_radial_mode_preserves_the_actual_plane_perspective(self):
        fixture = self.fixture
        profile = single.calibrate_single_points(
            fixture["observed"], fixture["image_size"], fixture["pattern_size"], mode="radial",
        )
        rectifier = single.SingleUndistorter(profile)
        held_out = _held_out_grid(fixture["pattern_size"])
        # Removing lens distortion alone must land at the original perspective
        # projection, rather than flattening the tilted plane into a square grid.
        perspective_positions = fixture["project"](held_out, np.zeros(5))
        actual_source = _sample_source_coordinates(rectifier, perspective_positions)
        expected_source = fixture["project"](held_out)
        self.assertLess(float(np.linalg.norm(actual_source - expected_source, axis=1).max()), 0.15)
        perspective_metrics = _regular_grid_metrics(fixture["ideal"], fixture["pattern_size"])
        self.assertGreater(max(perspective_metrics), 0.04)
        self.assertIsNone(profile.pixel_size_mm)

    def test_single_real_png_is_detected_and_rectified_as_a_regular_grid(self):
        fixture = _known_camera(k1=-0.32, k2=0.06, pattern_size=(9, 6))
        image = _render_board(fixture)
        with tempfile.TemporaryDirectory(prefix="single-lens-") as directory:
            path = Path(directory) / "单张棋盘格.png"
            self.assertTrue(imwrite(path, image))
            # No detector or image-IO mocks: exercise the actual one-image path.
            profile = single.calibrate_single_image(
                path, fixture["pattern_size"], spacing_mm=2.0, mode="radial_perspective",
            )
        corrected = single.SingleUndistorter(profile).apply(image)
        found, corners = cv2.findChessboardCornersSB(
            corrected, fixture["pattern_size"], flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        self.assertTrue(found, "the corrected real raster must retain a detectable checkerboard")
        self.assertLess(profile.rms, 0.5)
        for metric in _regular_grid_metrics(corners, fixture["pattern_size"]):
            self.assertLess(metric, 0.02)

    def test_single_profile_json_round_trip_and_schemas_are_distinct(self):
        encoded = json.dumps(self.profile.to_dict(), ensure_ascii=False, allow_nan=False)
        restored = single.SingleCalibrationProfile.from_dict(json.loads(encoded))
        self.assertEqual(restored.to_dict(), self.profile.to_dict())
        np.testing.assert_array_equal(restored.homography, self.profile.homography)
        np.testing.assert_array_equal(restored.radial_coeffs, self.profile.radial_coeffs)
        with self.assertRaises(ValueError):
            CalibrationProfile.from_dict(self.profile.to_dict())
        with self.assertRaises(ValueError):
            single.SingleCalibrationProfile.from_dict(manual_profile((960, 720), 0.0).to_dict())

    def test_nonfinite_parameters_and_observations_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            for field in ("homography", "radial_coeffs", "center"):
                with self.subTest(value=invalid, field=field):
                    payload = copy.deepcopy(self.profile.to_dict())
                    if field == "homography":
                        payload[field][0][0] = invalid
                    else:
                        payload[field][0] = invalid
                    with self.assertRaises(ValueError):
                        single.SingleCalibrationProfile.from_dict(payload)
            with self.subTest(value=invalid, field="map_input"):
                invalid_profile = replace(self.profile, radial_coeffs=np.array([invalid, 0.0]))
                with self.assertRaises(ValueError):
                    single.SingleUndistorter(invalid_profile)
        observations = self.fixture["observed"].copy()
        observations[0, 0] = np.nan
        with self.assertRaises(ValueError):
            single.calibrate_single_points(observations, self.fixture["image_size"], self.fixture["pattern_size"])

    def test_resolution_mismatch_and_unsupported_alpha_are_rejected(self):
        rectifier = single.SingleUndistorter(self.profile)
        with self.assertRaises(ValueError):
            rectifier.apply(np.zeros((720, 961, 3), np.uint8))
        with self.assertRaises(ValueError):
            single.SingleUndistorter(self.profile, alpha=0.0)
        self.assertTrue(np.isfinite(rectifier.map_build_ms))
        self.assertGreaterEqual(rectifier.map_build_ms, 0.0)

    def test_physical_spacing_changes_units_without_changing_the_geometry(self):
        doubled = replace(self.profile, spacing_mm=4.0)
        first = single.SingleUndistorter(self.profile)
        second = single.SingleUndistorter(doubled)
        self.assertGreater(first.pixel_size_mm, 0.0)
        self.assertAlmostEqual(second.pixel_size_mm, first.pixel_size_mm * 2.0)
        np.testing.assert_array_equal(first.map_x, second.map_x)
        np.testing.assert_array_equal(first.map_y, second.map_y)


if __name__ == "__main__":
    unittest.main()
