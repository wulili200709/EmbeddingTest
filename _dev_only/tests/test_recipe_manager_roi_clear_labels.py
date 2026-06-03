from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from domain.recipe_manager import clearable_roi_labels


class ClearableRoiLabelsTest(unittest.TestCase):
    def test_prefers_stale_labels_when_only_missing_mode_is_enabled(self) -> None:
        labels, mode = clearable_roi_labels(
            ["roi1"],
            ["roi1", "roi2", "anchor"],
            prefer_stale_only=True,
        )

        self.assertEqual(labels, ["roi2"])
        self.assertEqual(mode, "stale_only")

    def test_full_clear_keeps_current_and_stale_labels(self) -> None:
        labels, mode = clearable_roi_labels(
            ["roi1"],
            ["roi1", "roi2"],
            prefer_stale_only=False,
        )

        self.assertEqual(labels, ["roi1", "roi2"])
        self.assertEqual(mode, "all_existing")


if __name__ == "__main__":
    unittest.main()
