from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from ui.roi_overlay_colors import (
    ROI_DEFAULT_COLOR,
    ROI_NG_COLOR,
    ROI_UNLABELED_COLOR,
    color_for_roi_status,
    is_roi_label,
    merge_roi_statuses,
)


class RoiOverlayColorTests(unittest.TestCase):
    def test_color_for_roi_status_uses_blue_for_unlabeled(self) -> None:
        self.assertEqual(color_for_roi_status("").name(), ROI_UNLABELED_COLOR.name())
        self.assertEqual(color_for_roi_status("pending").name(), ROI_UNLABELED_COLOR.name())

    def test_color_for_roi_status_uses_green_for_ok(self) -> None:
        self.assertEqual(color_for_roi_status("ok").name(), ROI_DEFAULT_COLOR.name())

    def test_color_for_roi_status_uses_red_for_ng(self) -> None:
        self.assertEqual(color_for_roi_status("ng").name(), ROI_NG_COLOR.name())
        self.assertEqual(color_for_roi_status("NG").name(), ROI_NG_COLOR.name())

    def test_merge_roi_statuses_filters_non_roi_and_prefers_ng(self) -> None:
        rows = [
            {"camera_id": "cam1", "roi_label": "roi1", "status_kind": "ok"},
            {"camera_id": "cam1", "roi_label": "roi1", "status_kind": "ng"},
            {"camera_id": "cam1", "roi_label": "anchor", "status_kind": "ng"},
            {"camera_id": "cam2", "roi_label": "roi2", "status_kind": "ok"},
        ]

        merged = merge_roi_statuses(rows, camera_id="cam1")

        self.assertEqual(merged, {"roi1": "ng"})
        self.assertTrue(is_roi_label("roi1"))
        self.assertFalse(is_roi_label("anchor"))

    def test_distance_ng_propagates_to_its_helper_rois(self) -> None:
        rows = [
            {
                "item_id": "left_line",
                "camera_id": "cam2",
                "roi_label": "roi2",
                "algorithm_code": "find_line",
                "status_kind": "ok",
            },
            {
                "item_id": "right_line",
                "camera_id": "cam2",
                "roi_label": "roi3",
                "algorithm_code": "find_line",
                "status_kind": "ok",
            },
            {
                "item_id": "width",
                "camera_id": "cam2",
                "roi_label": "",
                "algorithm_code": "line_distance",
                "status_kind": "ng",
                "params": {
                    "line_a_item_id": "left_line",
                    "line_b_item_id": "right_line",
                },
            },
        ]

        self.assertEqual(
            merge_roi_statuses(rows, camera_id="cam2"),
            {"roi2": "ng", "roi3": "ng"},
        )


if __name__ == "__main__":
    unittest.main()
