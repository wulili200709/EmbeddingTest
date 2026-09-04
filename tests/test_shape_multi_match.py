from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from shape.core.recipe import ShapeRecipe
from shape.core.roi_follow import locate_and_follow
from shape.like_matcher import Match


class _FakeDetector:
    def __init__(self, matches: list[Match]) -> None:
        self._matches = list(matches)
        self.class_sources = {}

    def match(self, *_args, **_kwargs) -> list[Match]:
        return list(self._matches)

    def get_templates(self, _class_id: str, _template_id: int, backend: str = "original"):
        del backend
        return [SimpleNamespace(width=19, height=9, tl_x=0, tl_y=0)]

    def get_template_meta(self, _class_id: str, _template_id: int) -> dict[str, float]:
        return {}


class ShapeMultiMatchTests(unittest.TestCase):
    @staticmethod
    def _matches() -> list[Match]:
        return [
            Match(x=80, y=20, similarity=81.0, class_id="product", template_id=0),
            Match(x=10, y=40, similarity=95.0, class_id="product", template_id=0),
            Match(x=140, y=60, similarity=72.0, class_id="product", template_id=0),
        ]

    def test_find_count_two_retains_two_distinct_matches(self) -> None:
        recipe = ShapeRecipe(
            class_id="product",
            backend="original",
            threshold=50.0,
            nms_iou=0.0,
            topk=2,
            follow_mode="match_bbox",
        )

        result = locate_and_follow(
            np.zeros((100, 200, 3), dtype=np.uint8),
            "",
            recipe,
            detector=_FakeDetector(self._matches()),
        )

        self.assertEqual(result.detected_product_count, 2)
        self.assertEqual([match.similarity for match in result.matches], [95.0, 81.0])
        self.assertIs(result.match, result.matches[0])

    def test_find_count_one_preserves_legacy_single_match_behavior(self) -> None:
        recipe = ShapeRecipe(
            class_id="product",
            backend="original",
            threshold=50.0,
            nms_iou=0.0,
            topk=1,
            follow_mode="match_bbox",
        )

        result = locate_and_follow(
            np.zeros((100, 200, 3), dtype=np.uint8),
            "",
            recipe,
            detector=_FakeDetector(self._matches()),
        )

        self.assertEqual(result.detected_product_count, 1)
        self.assertEqual(len(result.matches), 1)

    def test_recipe_rejects_non_positive_find_count(self) -> None:
        self.assertEqual(ShapeRecipe.from_dict({"topk": 0}).topk, 1)


if __name__ == "__main__":
    unittest.main()
