"""The physical-to-laboratory mapping must never silently reverse a product."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_lab_opening_lines import MAPPING, physical_identity


class LabLineOrderTests(unittest.TestCase):
    def test_all_fourteen_labels_and_positions(self):
        expected_top = [f"L{i}" for i in range(7, 0, -1)]
        expected_bottom = [f"U{i}" for i in range(7, 0, -1)]
        self.assertEqual(MAPPING, {"top": expected_top, "bottom": expected_bottom})
        for old_row, physical_row, labels in (("U", "top", expected_top), ("L", "bottom", expected_bottom)):
            for column, label in enumerate(labels, 1):
                self.assertEqual(physical_identity(f"{old_row}{column}"), (label, physical_row, column))

    def test_unknown_region_is_rejected(self):
        with self.assertRaises(ValueError):
            physical_identity("U8")


if __name__ == "__main__":
    unittest.main()
