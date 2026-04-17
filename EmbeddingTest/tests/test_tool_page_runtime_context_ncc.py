from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from application.runtime_context import ToolPageRuntimeContext


class _DummyAlgo:
    def is_learning_tool(self, _algorithm_code: str) -> bool:
        return False


class _ToolPageHarness:
    def __init__(self) -> None:
        self.loc_method = "ncc"
        self.algo = _DummyAlgo()
        self.autogen_calls: list[dict[str, object]] = []
        self.invalidated_paths: list[str] = []
        self._line2dup_match_ms_by_image: dict[str, float] = {}
        self._line2dup_autogen_ms_by_image: dict[str, float] = {}

    def current_camera_role(self) -> str:
        return "cam1"

    def _autogen_roi_for_images(self, paths, only_missing: bool, silent: bool):
        self.autogen_calls.append(
            {
                "paths": list(paths),
                "only_missing": bool(only_missing),
                "silent": bool(silent),
            }
        )
        for path in paths:
            self._line2dup_match_ms_by_image[str(path)] = 6.5
            self._line2dup_autogen_ms_by_image[str(path)] = 12.5

    def _invalidate_shape_lookup_cache(self, path: str) -> None:
        self.invalidated_paths.append(str(path))

    def load_embedding_model(self, algorithm: str, model_key: str | None = None) -> None:
        raise AssertionError("no embedding model should be loaded when there are no enabled items")


class ToolPageRuntimeContextNccTest(unittest.TestCase):
    def test_ncc_path_prediction_forces_roi_rematch(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image_path = handle.name

        harness = _ToolPageHarness()
        try:
            rows = ToolPageRuntimeContext(harness).predict_items_batch(image_path, items=[])
        finally:
            os.unlink(image_path)

        self.assertEqual(rows, [])
        self.assertEqual(
            harness.autogen_calls,
            [{"paths": [image_path], "only_missing": False, "silent": True}],
        )
        self.assertEqual(harness.invalidated_paths, [image_path])

    def test_ncc_path_prediction_uses_locate_time_as_match_time(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image_path = handle.name

        harness = _ToolPageHarness()
        try:
            ToolPageRuntimeContext(harness).predict_items_batch(image_path, items=[])
        finally:
            os.unlink(image_path)

        self.assertEqual(harness._line2dup_match_ms_by_image[image_path], 6.5)
        self.assertEqual(harness._line2dup_autogen_ms_by_image[image_path], 12.5)


if __name__ == "__main__":
    unittest.main()
