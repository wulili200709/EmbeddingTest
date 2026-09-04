from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ncc.locator import autogen_runtime_roi_shapes_timed
from ncc.model import NccMatchModel, NccMatchOptions


class _FakeCompiledModel:
    def __init__(self, match_count: int) -> None:
        quad = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        self._response = SimpleNamespace(
            matches=tuple(SimpleNamespace(quad=quad) for _index in range(match_count))
        )

    def match(self, *_args, **_kwargs):
        return self._response


class NccMultiMatchTests(unittest.TestCase):
    def test_runtime_result_reports_all_retained_ncc_matches(self) -> None:
        model = NccMatchModel(options=NccMatchOptions(target_num=2))

        with patch("ncc.locator.os.path.exists", return_value=True):
            run = autogen_runtime_roi_shapes_timed(
                np.zeros((20, 20, 3), dtype=np.uint8),
                "product",
                model_path="model.json",
                model=model,
                compiled_model=_FakeCompiledModel(match_count=2),
            )

        self.assertEqual(run.detected_product_count, 2)
        self.assertTrue(run.roi_shapes)


if __name__ == "__main__":
    unittest.main()
