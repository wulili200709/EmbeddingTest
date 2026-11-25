"""Exercise real image decoding, corner detection and calibration together."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from algorithms.image_io import imwrite
from algorithms.lens_calibration import calibrate_images


class RenderedCalibrationTest(unittest.TestCase):
    def setUp(self):
        # Avoid GPU kernel-cache warmup and shared driver-cache contention in this test.
        self.opencl_enabled = cv2.ocl.useOpenCL()
        cv2.ocl.setUseOpenCL(False)

    def tearDown(self):
        cv2.ocl.setUseOpenCL(self.opencl_enabled)

    def test_perspective_checkerboards_recover_an_undistorted_camera(self):
        """Multiple true perspective views must not invent substantial lens distortion."""
        width, height = 960, 720
        camera = np.array([[960.0, 0, 479.5], [0, 955.0, 359.5], [0, 0, 1.0]])
        columns, rows, tile = 9, 6, 70
        texture = np.zeros(((rows + 1) * tile, (columns + 1) * tile), np.uint8)
        for row in range(rows + 1):
            for column in range(columns + 1):
                texture[row * tile:(row + 1) * tile, column * tile:(column + 1) * tile] = (
                    255 if (row + column) % 2 else 0
                )
        # Pixel-edge coordinates make the first internal corner correspond to (0, 0).
        texture_quad = np.array([
            [-0.5, -0.5], [texture.shape[1] - 0.5, -0.5],
            [texture.shape[1] - 0.5, texture.shape[0] - 0.5],
            [-0.5, texture.shape[0] - 0.5],
        ], np.float32)
        world_quad = np.array([[-2, -2, 0], [18, -2, 0], [18, 12, 0], [-2, 12, 0]], np.float32)
        centers = [
            (270, 210), (670, 220), (680, 490), (280, 500), (480, 360),
            (430, 220), (560, 500), (300, 350), (660, 350), (480, 400),
            (500, 270), (400, 440), (560, 340), (350, 260),
        ]
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory(prefix="lens-rendered-") as directory:
            paths = []
            for index, (cx, cy) in enumerate(centers):
                rotation = np.array([
                    rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55), rng.uniform(-0.22, 0.22),
                ])
                rotation_matrix, _ = cv2.Rodrigues(rotation)
                depth = float(rng.uniform(48, 65))
                center_camera = np.array([
                    (cx - camera[0, 2]) * depth / camera[0, 0],
                    (cy - camera[1, 2]) * depth / camera[1, 1], depth,
                ])
                translation = center_camera - rotation_matrix @ np.array([8.0, 5.0, 0.0])
                projected, _ = cv2.projectPoints(world_quad, rotation, translation, camera, np.zeros(5))
                homography = cv2.getPerspectiveTransform(texture_quad, projected.reshape(-1, 2))
                image = cv2.warpPerspective(texture, homography, (width, height),
                                            flags=cv2.INTER_LINEAR, borderValue=210)
                image = cv2.GaussianBlur(image, (3, 3), 0.6)
                path = Path(directory) / f"board_{index:02d}.png"
                self.assertTrue(imwrite(path, image))
                paths.append(path)

            # No mocks: the complete pipeline reads actual PNGs and detects actual corners.
            profile = calibrate_images(paths, "chessboard", (columns, rows), 2.0)

        self.assertEqual(profile.image_size, (width, height))
        self.assertEqual(sum(view["status"] == "used" for view in profile.views), len(centers))
        self.assertLess(profile.rms, 0.35)
        np.testing.assert_allclose(profile.camera_matrix[[0, 1], [0, 1]],
                                   camera[[0, 1], [0, 1]], rtol=0.02)
        self.assertLess(np.max(np.abs(profile.camera_matrix[:2, 2] - camera[:2, 2])), 5.0)
        self.assertLess(abs(profile.dist_coeffs[0]), 0.05)
        self.assertLess(np.max(np.abs(profile.dist_coeffs[2:4])), 0.003)


if __name__ == "__main__":
    unittest.main()
