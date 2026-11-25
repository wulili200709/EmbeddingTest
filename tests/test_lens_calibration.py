from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from algorithms import lens_calibration as calibration


def _projected_board_views(count: int = 24):
    """Independent observations from a known pinhole camera, without image files."""
    image_size = (1280, 960)
    pattern_size = (9, 6)
    spacing_mm = 2.0
    camera_matrix = np.array(
        [[900.0, 0.0, 637.5], [0.0, 915.0, 481.25], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([-0.17, 0.04, 0.0012, -0.0009, 0.008], dtype=np.float64)
    columns, rows = np.meshgrid(np.arange(pattern_size[0]), np.arange(pattern_size[1]))
    board_points = np.column_stack(
        (columns.ravel(), rows.ravel(), np.zeros(columns.size))
    ).astype(np.float32) * spacing_mm
    observations = []
    for index in range(count):
        phase = index * 0.71
        rotation = np.array(
            [0.36 * np.sin(phase), 0.42 * np.cos(phase * 1.3), 0.18 * np.sin(phase * 0.7)]
        )
        translation = np.array(
            [-8.0 + 4.2 * np.sin(phase * 0.8), -5.0 + 2.8 * np.cos(phase), 39.0 + index % 7 * 2.1]
        )
        corners, _ = cv2.projectPoints(
            board_points, rotation, translation, camera_matrix, distortion
        )
        observations.append(corners.astype(np.float32))
    return image_size, pattern_size, spacing_mm, camera_matrix, distortion, observations


def _distorted_grid(camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Render straight physical grid lines through the known distorted camera."""
    width, height = 640, 480
    image = np.zeros((height, width, 3), dtype=np.uint8)

    def project_pixels(points: np.ndarray) -> np.ndarray:
        rays = np.column_stack(
            (
                (points[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
                (points[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
                np.ones(len(points)),
            )
        )
        projected, _ = cv2.projectPoints(
            rays, np.zeros(3), np.zeros(3), camera_matrix, distortion
        )
        return np.rint(projected).astype(np.int32).reshape(-1, 1, 2)

    # Distinct channels keep intersections from confusing the straightness metric.
    for x in (80, 160, 240, 320, 400, 480, 560):
        ys = np.linspace(25, height - 26, 700)
        points = np.column_stack((np.full_like(ys, x), ys))
        cv2.polylines(image, [project_pixels(points)], False, (0, 150, 0), 1, cv2.LINE_AA)
    for y in (80, 160, 320, 400):
        xs = np.linspace(25, width - 26, 900)
        points = np.column_stack((xs, np.full_like(xs, y)))
        cv2.polylines(image, [project_pixels(points)], False, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def _horizontal_line_errors(image: np.ndarray) -> list[float]:
    mask = (image[:, :, 2] > 60).astype(np.uint8)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, 8)
    errors = []
    for label in range(1, count):
        if (
            statistics[label, cv2.CC_STAT_AREA] < 200
            or statistics[label, cv2.CC_STAT_WIDTH] < 200
        ):
            continue
        ys, xs = np.where(labels == label)
        fitted = np.polyfit(xs, ys, 1)
        errors.append(float(np.sqrt(np.mean((ys - np.polyval(fitted, xs)) ** 2))))
    return errors


class LensCalibrationTests(unittest.TestCase):
    def _profile(self, **overrides):
        values = {
            "image_size": (640, 480),
            "camera_matrix": np.array(
                [[480.0, 0.0, 320.0], [0.0, 470.0, 240.0], [0.0, 0.0, 1.0]]
            ),
            "dist_coeffs": np.array([-0.28, 0.07, 0.001, -0.0007, 0.0]),
        }
        values.update(overrides)
        return calibration.CalibrationProfile(**values)

    def test_json_round_trip_preserves_parameters_and_view_evidence(self) -> None:
        profile = self._profile(
            rms=0.1287654321,
            pattern="chessboard",
            pattern_size=(9, 6),
            spacing_mm=2.5,
            views=[{"path": "标定图_01.png", "status": "used", "error_px": 0.123456789}],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "相机标定.json"
            calibration.save_profile(profile, path)
            restored = calibration.load_profile(path)
        np.testing.assert_array_equal(restored.camera_matrix, profile.camera_matrix)
        np.testing.assert_array_equal(restored.dist_coeffs.ravel(), profile.dist_coeffs.ravel())
        self.assertEqual(restored.image_size, profile.image_size)
        self.assertEqual(restored.pattern_size, profile.pattern_size)
        self.assertEqual(restored.spacing_mm, profile.spacing_mm)
        self.assertEqual(restored.rms, profile.rms)
        self.assertEqual(restored.views, profile.views)
        self.assertEqual(restored.to_dict(), profile.to_dict())
        self.assertEqual(
            calibration.CalibrationProfile.from_dict(profile.to_dict()).to_dict(),
            profile.to_dict(),
        )

    def test_nonfinite_camera_and_distortion_parameters_are_rejected(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(parameter="camera_matrix", value=invalid):
                camera_matrix = np.eye(3)
                camera_matrix[0, 0] = invalid
                profile = self._profile(camera_matrix=camera_matrix)
                with self.assertRaises(ValueError):
                    calibration.Undistorter(profile)
                with self.assertRaises(ValueError):
                    profile.to_dict()
                payload = self._profile().to_dict()
                payload["camera_matrix"][0][0] = invalid
                with self.assertRaises(ValueError):
                    calibration.CalibrationProfile.from_dict(payload)
            with self.subTest(parameter="dist_coeffs", value=invalid):
                coefficients = np.zeros(5)
                coefficients[0] = invalid
                profile = self._profile(dist_coeffs=coefficients)
                with self.assertRaises(ValueError):
                    calibration.Undistorter(profile)
                with self.assertRaises(ValueError):
                    profile.to_dict()
                payload = self._profile().to_dict()
                payload["dist_coeffs"][0] = invalid
                with self.assertRaises(ValueError):
                    calibration.CalibrationProfile.from_dict(payload)

    def test_wrong_frame_resolution_is_rejected(self) -> None:
        undistorter = calibration.Undistorter(self._profile())
        with self.assertRaises(ValueError):
            undistorter.apply(np.zeros((480, 641, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            undistorter.apply(np.zeros((479, 640, 3), dtype=np.uint8))

    def test_zero_distortion_preserves_pinhole_geometry(self) -> None:
        profile = self._profile(dist_coeffs=np.zeros(5))
        undistorter = calibration.Undistorter(profile, alpha=1.0)
        ys, xs = np.indices((480, 640))
        image = np.stack((xs // 4, ys // 4, (xs + ys) // 8), axis=2).astype(np.uint8)
        corrected = undistorter.apply(image)
        # A new output camera matrix may scale/translate a distortion-free image.
        # Its analytic mapping is a homography, so it must not introduce curvature.
        homography = undistorter.new_camera_matrix @ np.linalg.inv(profile.camera_matrix)
        expected = cv2.warpPerspective(image, homography, profile.image_size, flags=cv2.INTER_LINEAR)
        np.testing.assert_allclose(corrected, expected, atol=2)
        if np.allclose(undistorter.new_camera_matrix, profile.camera_matrix):
            np.testing.assert_array_equal(corrected, image)
        self.assertTrue(np.isfinite(undistorter.map_build_ms))
        self.assertGreaterEqual(undistorter.map_build_ms, 0.0)

    def test_synthetic_curved_grid_becomes_straight(self) -> None:
        profile = self._profile()
        distorted = _distorted_grid(profile.camera_matrix, profile.dist_coeffs)
        corrected = calibration.Undistorter(profile, alpha=1.0).apply(distorted)
        before = _horizontal_line_errors(distorted)
        after = _horizontal_line_errors(corrected)
        self.assertEqual(len(before), 4)
        self.assertEqual(len(after), 4)
        self.assertGreater(float(np.mean(before)), 2.0)
        self.assertLess(float(np.mean(after)), 0.7)
        self.assertLess(float(np.mean(after)), float(np.mean(before)) * 0.25)

    def test_manual_profile_carries_requested_radial_coefficients(self) -> None:
        profile = calibration.manual_profile((640, 480), k1=-0.2, k2=0.03)
        self.assertEqual(profile.image_size, (640, 480))
        self.assertEqual(profile.pattern, "manual")
        np.testing.assert_allclose(profile.dist_coeffs.ravel()[:2], [-0.2, 0.03])
        corrected = calibration.Undistorter(profile).apply(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(corrected.shape, (480, 640, 3))

    def test_calibration_recovers_a_known_camera_from_multiple_poses(self) -> None:
        size, pattern_size, spacing, camera, distortion, observations = _projected_board_views()
        paths = [f"virtual_view_{index:02d}.png" for index in range(len(observations))]

        def read_image(path, *_args, **_kwargs):
            index = paths.index(str(path))
            return np.full((size[1], size[0]), index + 1, dtype=np.uint8)

        def detect(gray, *_args, **_kwargs):
            return observations[int(gray[0, 0]) - 1].copy()

        # Image IO and corner detection are controlled; OpenCV's optimizer runs.
        with mock.patch.object(calibration.image_io, "imread", side_effect=read_image), mock.patch.object(
            calibration, "_detect_pattern", side_effect=detect
        ):
            fitted = calibration.calibrate_images(
                paths, pattern="chessboard", pattern_size=pattern_size, spacing_mm=spacing
            )
        self.assertEqual(fitted.image_size, size)
        self.assertEqual(sum(view.get("status") == "used" for view in fitted.views), len(paths))
        self.assertLess(fitted.rms, 0.001)
        np.testing.assert_allclose(fitted.camera_matrix, camera, atol=0.1)
        np.testing.assert_allclose(fitted.dist_coeffs.ravel(), distortion, atol=0.002)

    def test_fewer_than_ten_valid_views_are_rejected(self) -> None:
        size, pattern_size, spacing, _camera, _distortion, observations = _projected_board_views(9)
        paths = [f"virtual_view_{index:02d}.png" for index in range(12)]

        def read_image(path, *_args, **_kwargs):
            return np.full((size[1], size[0]), paths.index(str(path)) + 1, dtype=np.uint8)

        def detect(gray, *_args, **_kwargs):
            index = int(gray[0, 0]) - 1
            return observations[index] if index < len(observations) else None

        with mock.patch.object(calibration.image_io, "imread", side_effect=read_image), mock.patch.object(
            calibration, "_detect_pattern", side_effect=detect
        ):
            with self.assertRaises(calibration.CalibrationError) as caught:
                calibration.calibrate_images(
                    paths, pattern="chessboard", pattern_size=pattern_size, spacing_mm=spacing
                )
        self.assertEqual(len(caught.exception.views), 12)
        self.assertEqual(sum(view["status"] == "used" for view in caught.exception.views), 9)

    def test_copies_of_one_photo_do_not_count_as_ten_views(self) -> None:
        size, pattern_size, spacing, _camera, _distortion, observations = _projected_board_views(1)
        one_image = np.full((size[1], size[0]), 127, dtype=np.uint8)
        with mock.patch.object(calibration.image_io, "imread", return_value=one_image), mock.patch.object(
            calibration, "_detect_pattern", return_value=observations[0]
        ):
            with self.assertRaises(calibration.CalibrationError) as caught:
                calibration.calibrate_images(
                    [f"copy_{index:02d}.png" for index in range(10)],
                    pattern="chessboard",
                    pattern_size=pattern_size,
                    spacing_mm=spacing,
                )
        self.assertEqual(sum(view["status"] == "used" for view in caught.exception.views), 1)
        self.assertEqual(sum(view["status"] == "duplicate" for view in caught.exception.views), 9)


if __name__ == "__main__":
    unittest.main()
