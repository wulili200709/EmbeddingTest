from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from line2dup.core.roi_follow import _best_match


class _NoMatchDetector:
    def match(self, scene_bgr, *, threshold, class_ids, mask=None, backend="original"):
        return []


class Line2DupMatchFailureTest(unittest.TestCase):
    def test_best_match_uses_match_failure_message(self) -> None:
        recipe = SimpleNamespace(
            backend="original",
            threshold=0.5,
            nms_iou=0.3,
            class_id="cls1",
        )

        with self.assertRaisesRegex(RuntimeError, "match failure"):
            _best_match(_NoMatchDetector(), np.zeros((10, 10, 3), dtype=np.uint8), recipe)


if __name__ == "__main__":
    unittest.main()
