import numpy as np
import unittest

from algorithms.measurement import (
    OPENING_GAP_MEDIAN_ALGORITHM,
    is_measurement_algorithm,
    measure_opening_gap_from_array,
)
from algorithms.measurement_opening_gap import OpeningGapRegion, measure_opening_gap


def _synthetic_opening() -> np.ndarray:
    image = np.full((110, 110, 3), 220, dtype=np.uint8)
    image[30:70, :] = 30
    image[35:65, 47:51] = 225  # a narrow reflection crosses the dark gap
    return image


class OpeningGapTests(unittest.TestCase):
    def test_dark_plateau_tracks_inner_boundary_at_two_core_brightness_levels(self) -> None:
        for core_gray in (30, 110):
            image = np.full((110, 110, 3), 255, dtype=np.uint8)
            image[30:38] = 140
            image[38:70] = core_gray
            image[70:78] = 140
            result = measure_opening_gap(
                image, OpeningGapRegion("opening", 55, (20, 50), (60, 90)),
                blur_ksize=9, edge_mode="dark_plateau",
            )
            self.assertAlmostEqual(result.upper_y, 38.0, delta=4.0)
            self.assertAlmostEqual(result.lower_y, 70.0, delta=4.0)
            self.assertAlmostEqual(result.distance_px, 32.0, delta=6.0)

    def test_dark_side_edges_ignore_bright_outer_transition(self) -> None:
        image = np.full((110, 110, 3), 255, dtype=np.uint8)
        image[30:38] = 140
        image[38:70] = 30
        image[70:78] = 140
        unfiltered = measure_opening_gap(image, OpeningGapRegion("opening", 55, (20, 50), (60, 90)))
        dark_side = measure_opening_gap(
            image, OpeningGapRegion("opening", 55, (20, 50), (60, 90)),
            edge_dark_max_gray=128,
        )
        self.assertAlmostEqual(unfiltered.distance_px, 48.0, delta=0.5)
        self.assertAlmostEqual(dark_side.distance_px, 32.0, delta=0.5)
        self.assertAlmostEqual(dark_side.lower_y - dark_side.upper_y, dark_side.distance_px, places=4)

    def test_median_gap_with_reflection(self) -> None:
        result = measure_opening_gap(
            _synthetic_opening(),
            OpeningGapRegion("opening", 55, (20, 42), (58, 80)),
            include_column_edges=True,
        )
        self.assertGreaterEqual(result.valid_columns, 40)
        self.assertEqual(len(result.column_edges), result.valid_columns)
        self.assertTrue(all(31 <= x <= 79 for x, _, _ in result.column_edges))
        self.assertTrue(all(lower > upper for _, upper, lower in result.column_edges))
        self.assertAlmostEqual(result.distance_px, 40.0, delta=0.5)
        self.assertEqual(result.to_dict()["type"], OPENING_GAP_MEDIAN_ALGORITHM)

    def test_tight_roi_fallback_and_missing_edge(self) -> None:
        image = _synthetic_opening()
        shape = {"shape_type": "rectangle", "points": [[25, 15], [85, 85]]}
        result = measure_opening_gap_from_array(
            image, shape_by_label={"roi1": shape}, preferred_label="roi1",
        )
        self.assertAlmostEqual(result.distance_px, 40.0, delta=0.5)
        self.assertTrue(is_measurement_algorithm(OPENING_GAP_MEDIAN_ALGORITHM))
        with self.assertRaisesRegex(RuntimeError, "valid columns"):
            measure_opening_gap(
                np.full_like(image, 128),
                OpeningGapRegion("missing", 55, (20, 42), (58, 80)),
            )


if __name__ == "__main__":
    unittest.main()
