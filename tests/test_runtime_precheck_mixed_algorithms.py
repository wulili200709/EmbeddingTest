from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from algorithms.registry import (
    SHARED_BACKBONE_ALGORITHM_CODE,
    get_tool_algorithm_spec,
    is_learning_tool_algorithm,
    is_traditional_tool_algorithm,
)
from application.runtime.execution import _precheck
from application.runtime import controller as runtime_controller_module
from domain.inspection_items import InspectionItem


class _FakeFrameGrabService:
    def roles(self):
        return ["cam1"]


class _FakeRuntimeContext:
    def __init__(self, items) -> None:
        self.loc_method = "line2dup"
        self.inspection_items = list(items)
        self.loaded_algorithms: list[tuple[str, str | None]] = []

    def load_embedding_model(self, algorithm: str, model_key: str | None = None) -> None:
        self.loaded_algorithms.append((algorithm, model_key))


class _FakeAlgo:
    def __init__(self) -> None:
        self.product_params = SimpleNamespace(
            traditional_models={"meanintensity::cam1__roi2": {"threshold": 0.5}}
        )
        self.model = SimpleNamespace(backbone="efficientnet_b0", device="cpu")
        self.feat_net_requests: list[str] = []

    def tool_algorithm_spec(self, code):
        return get_tool_algorithm_spec(code)

    def is_learning_tool(self, code) -> bool:
        return is_learning_tool_algorithm(code)

    def is_traditional_tool(self, code) -> bool:
        return is_traditional_tool_algorithm(code)

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, code) -> str:
        if is_learning_tool_algorithm(code):
            return "efficientnet_b0"
        return str(code or "").strip()

    def get_feat_net(self, backbone: str, device=None):
        self.feat_net_requests.append(backbone)
        return object()

    def get_traditional_model_dict(self, algorithm: str, *, model_key: object = ""):
        key = f"{algorithm}::{model_key}" if model_key else algorithm
        model_dict = self.product_params.traditional_models.get(key)
        if isinstance(model_dict, dict):
            return model_dict
        return self.product_params.traditional_models.get(algorithm)


class RuntimePrecheckMixedAlgorithmsTest(unittest.TestCase):
    def test_precheck_accepts_mixed_learning_and_traditional_items(self) -> None:
        items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code=SHARED_BACKBONE_ALGORITHM_CODE,
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam1",
                roi_label="roi2",
                algorithm_code="meanintensity",
            ),
        ]
        runtime_context = _FakeRuntimeContext(items)
        algo = _FakeAlgo()

        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_path = Path(tmpdir) / "recipe.json"
            recipe_path.write_text("{}", encoding="utf-8")
            runtime = SimpleNamespace(
                _frame_grab_service=_FakeFrameGrabService(),
                _runtime_context=runtime_context,
                _session=SimpleNamespace(line2dup_recipe_path=str(recipe_path)),
                _algo=algo,
                _connected_roles=lambda: ["cam1"],
            )
            original_frame_to_bgr = runtime_controller_module.frame_to_bgr_image
            runtime_controller_module.frame_to_bgr_image = object()
            try:
                ok, message = _precheck(runtime)
            finally:
                runtime_controller_module.frame_to_bgr_image = original_frame_to_bgr

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(runtime_context.loaded_algorithms, [("efficientnet_b0", "cam1__roi1")])
        self.assertEqual(algo.feat_net_requests, ["efficientnet_b0"])


if __name__ == "__main__":
    unittest.main()
