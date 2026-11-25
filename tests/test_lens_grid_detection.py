"""Independent raster checks for near-axis partial checkerboard detection.

These tests cover modest tilt (six degrees), cropped boards and local clutter.
They do not promise arbitrary board rotation, strong perspective, or industrial
measurement accuracy. No customer image or production rendering helper is used.
"""
from __future__ import annotations

import itertools
import json
import unittest

import cv2
import numpy as np

from algorithms.lens_grid_detection import detect_checker_grid


class _CheckerRaster:
    """Evaluate a known continuous checker texture at each sensor sample."""

    def __init__(self, angle=0.0, perspective=0.0, bend=0.0, code=False, clutter=False):
        self.width, self.height = 768, 544
        self.pitch = 32.0
        self.angle = np.deg2rad(angle)
        self.perspective, self.bend = perspective, bend
        self.code, self.clutter = code, clutter
        # Fractional offsets crop squares at every image edge. An arbitrary
        # absolute board origin lets the check ignore detector index offsets.
        self.offset = np.array([20.37, 15.28])
        self.code_bounds = (18.0, 13.0, 22.0, 17.0)
        self.clutter_bounds = (13.25, 9.35, 15.65, 11.75)

    def board_coordinates(self, pixels):
        pixels = np.asarray(pixels, np.float64)
        x = pixels[..., 0] - (self.width - 1) / 2
        y = pixels[..., 1] - (self.height - 1) / 2
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        denominator = 1.0 + self.perspective * (x / self.width - 0.4 * y / self.height)
        board_x = (ca * x + sa * y) / denominator
        board_y = (-sa * x + ca * y) / denominator
        # A small smooth field exercises curved grid lines without sharing the
        # calibration implementation's polynomial radial model.
        board_x += self.bend * np.sin(2.0 * np.pi * y / self.height)
        board_y += 0.65 * self.bend * np.sin(2.0 * np.pi * x / self.width)
        return np.stack((board_x, board_y), axis=-1) / self.pitch + self.offset

    def render(self):
        ys, xs = np.indices((self.height, self.width), dtype=np.float64)
        sensor = np.stack((xs, ys), axis=-1)
        accumulated = np.zeros((self.height, self.width), np.float64)
        for dx, dy in itertools.product((-0.25, 0.25), repeat=2):
            board = self.board_coordinates(sensor + [dx, dy])
            gx, gy = board[..., 0], board[..., 1]
            parity = (np.floor(gx).astype(np.int32) + np.floor(gy).astype(np.int32)) % 2
            intensity = np.where(parity == 0, 24.0, 231.0)
            if self.code:
                left, top, right, bottom = self.code_bounds
                covered = (gx > left) & (gx < right) & (gy > top) & (gy < bottom)
                # A dense, asymmetric binary block is deliberately not another
                # chessboard. Its corners must not create a finer grid unit.
                ix = np.floor((gx - left) * 4).astype(np.int32)
                iy = np.floor((gy - top) * 4).astype(np.int32)
                bit = ((ix * 13 + iy * 7 + (ix ^ iy) * 5) % 11) < 5
                intensity[covered] = np.where(bit[covered], 24.0, 231.0)
            if self.clutter:
                left, top, right, bottom = self.clutter_bounds
                covered = (gx > left) & (gx < right) & (gy > top) & (gy < bottom)
                # Small displaced squares create internally plausible corners
                # that disagree with the surrounding global lattice.
                local = (np.floor((gx - left) / 0.6).astype(np.int32)
                         + np.floor((gy - top) / 0.6).astype(np.int32)) % 2
                intensity[covered] = np.where(local[covered] == 0, 24.0, 231.0)
            accumulated += intensity
        return cv2.GaussianBlur(np.rint(accumulated / 4).astype(np.uint8), (3, 3), 0.5)


class _MagnifiedCheckerRaster:
    """Scale an independent raster using OpenCV's pixel-center convention."""

    def __init__(self, factor=4):
        self.base = _CheckerRaster(angle=4.0, code=True)
        self.factor = factor
        self.pitch = self.base.pitch * factor

    def render(self):
        return cv2.resize(self.base.render(), None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_LINEAR)

    def board_coordinates(self, pixels):
        return self.base.board_coordinates((np.asarray(pixels) + 0.5) / self.factor - 0.5)


class CheckerGridDetectionTests(unittest.TestCase):
    def assert_correct_lattice(self, fixture, minimum_points=60):
        points, indices, pattern_size, diagnostics = detect_checker_grid(fixture.render())
        points, indices = np.asarray(points), np.asarray(indices)
        self.assertEqual(points.shape, indices.shape)
        self.assertEqual(points.ndim, 2)
        self.assertEqual(points.shape[1], 2)
        self.assertGreaterEqual(len(points), minimum_points)
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.equal(indices, np.rint(indices)).all())
        np.testing.assert_array_equal(indices.min(axis=0), [0, 0])
        np.testing.assert_array_equal(indices.max(axis=0) + 1, pattern_size)
        self.assertEqual(len(np.unique(indices, axis=0)), len(indices))
        self.assertGreaterEqual(min(pattern_size), 5)

        continuous_truth = fixture.board_coordinates(points)
        true_indices = np.rint(continuous_truth).astype(np.int64)
        # All retained observations must be true original checker intersections,
        # including when a code block produces many other apparent corners.
        corner_error = np.linalg.norm(continuous_truth - true_indices, axis=1)
        self.assertLess(float(corner_error.max()), 0.10)

        # Grid origin, signs and axis names are free. Grid scale is NOT free:
        # allowing only signed permutations catches every-other-corner mistakes
        # and an accidental smaller grid introduced by the code block.
        best_mismatches = len(points)
        for swap in (False, True):
            truth = true_indices[:, ::-1] if swap else true_indices
            for sx, sy in itertools.product((-1, 1), repeat=2):
                aligned = truth * [sx, sy]
                translation = np.rint(np.median(indices - aligned, axis=0)).astype(np.int64)
                mismatches = np.count_nonzero(np.any(aligned + translation != indices, axis=1))
                best_mismatches = min(best_mismatches, mismatches)
        self.assertEqual(best_mismatches, 0, "returned index step must equal exactly one physical square")

        lookup = {tuple(index): point for index, point in zip(indices, points)}
        for delta in ((1, 0), (0, 1)):
            adjacent_lengths = []
            for index, point in zip(indices, points):
                neighbour = lookup.get(tuple(index + delta))
                if neighbour is not None:
                    adjacent_lengths.append(np.linalg.norm(neighbour - point))
            self.assertGreater(len(adjacent_lengths), 20)
            self.assertGreater(float(np.median(adjacent_lengths)), fixture.pitch * 0.8)
            self.assertLess(float(np.median(adjacent_lengths)), fixture.pitch * 1.2)
        self.assertIsInstance(diagnostics, dict)
        json.dumps(diagnostics, allow_nan=False)
        return points, indices, pattern_size, diagnostics

    def test_regular_cropped_grid_retains_the_single_square_unit(self):
        _, _, _, diagnostics = self.assert_correct_lattice(_CheckerRaster())
        self.assertTrue(diagnostics.get("grid_step_confirmed"), "a clean board should establish one-square pitch")

    def test_partial_board_with_code_block_does_not_select_a_subgrid(self):
        fixture = _CheckerRaster(code=True)
        points, _, _, _ = self.assert_correct_lattice(fixture)
        board = fixture.board_coordinates(points)
        # The useful surrounding grid must survive on both sides of the hole.
        self.assertGreater(np.count_nonzero(board[:, 0] < fixture.code_bounds[0]), 20)
        self.assertGreater(np.count_nonzero(board[:, 0] > fixture.code_bounds[2]), 20)
        left, top, right, bottom = fixture.code_bounds
        hidden_interior = ((board[:, 0] > left + 0.3) & (board[:, 0] < right - 0.3)
                           & (board[:, 1] > top + 0.3) & (board[:, 1] < bottom - 0.3))
        self.assertFalse(hidden_interior.any(), "missing corners inside the code block must not be filled in")

    def test_small_tilt_perspective_and_smooth_bending_preserve_indices(self):
        self.assert_correct_lattice(_CheckerRaster(angle=6.0, perspective=0.025, bend=1.4, code=True))

    def test_inconsistent_corner_clutter_is_excluded_from_the_global_lattice(self):
        self.assert_correct_lattice(_CheckerRaster(code=True, clutter=True))

    def test_large_input_returns_original_resolution_corner_coordinates(self):
        self.assert_correct_lattice(_MagnifiedCheckerRaster())

    def test_blank_and_one_dimensional_stripes_are_rejected(self):
        blank = np.full((544, 768), 127, np.uint8)
        stripes = np.tile(np.where((np.arange(768) // 32) % 2 == 0, 24, 231).astype(np.uint8), (544, 1))
        for label, image in (("blank", blank), ("vertical stripes", stripes), ("horizontal stripes", stripes.T.copy())):
            with self.subTest(image=label):
                with self.assertRaises(ValueError):
                    detect_checker_grid(image)


if __name__ == "__main__":
    unittest.main()
