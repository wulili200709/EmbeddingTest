from __future__ import annotations

import sys
import tempfile
import unittest
from itertools import combinations
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
from application.runtime import controller as runtime_controller_module
from application.runtime.capture_channels import required_runtime_roles
from application.runtime.execution import _precheck, _precheck_for_roles, _warm_runtime_models
from application.product_session import ProductSession, SessionData
from common.camera_roles import configured_camera_roles
from domain.inspection_items import InspectionItem


class _FakeFrameGrabService:
    def roles(self):
        return ["cam1", "cam2"]


class _FakeRuntimeContext:
    def __init__(self, items) -> None:
        self.loc_method = "shape"
        self.inspection_items = list(items)
        self.loaded_algorithms: list[tuple[str, str | None]] = []

    def load_embedding_model(self, algorithm: str, model_key: str | None = None) -> None:
        self.loaded_algorithms.append((algorithm, model_key))


class _FakeAlgo:
    def __init__(
        self,
        *,
        traditional_models=None,
        backbone: str = "efficientnet_b0",
        backbones=None,
    ) -> None:
        self.product_params = SimpleNamespace(traditional_models=dict(traditional_models or {}))
        self.model = SimpleNamespace(backbone=backbone, device="cpu")
        self.feat_net_requests: list[str] = []
        self.backbones = dict(backbones or {})

    def tool_algorithm_spec(self, code):
        return get_tool_algorithm_spec(code)

    def is_learning_tool(self, code) -> bool:
        return is_learning_tool_algorithm(code)

    def is_traditional_tool(self, code) -> bool:
        return is_traditional_tool_algorithm(code)

    def current_learning_backbone(self, camera_role=None) -> str:
        return self.backbones.get(str(camera_role or "cam1"), "efficientnet_b0")

    def resolve_tool_algorithm(self, code, camera_role=None) -> str:
        if is_learning_tool_algorithm(code):
            return self.current_learning_backbone(camera_role)
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


class RuntimeRolePrecheckTest(unittest.TestCase):
    def test_runtime_warmup_loads_each_camera_backbone(self) -> None:
        items = [
            InspectionItem(
                item_id=f"{role}-roi",
                display_name=f"{role.upper()} ROI",
                camera_id=role,
                roi_label="roi",
                algorithm_code=SHARED_BACKBONE_ALGORITHM_CODE,
            )
            for role in ("cam1", "cam2")
        ]
        runtime_context = _FakeRuntimeContext(items)
        algo = _FakeAlgo(
            backbones={"cam1": "efficientnet_b0", "cam2": "mobilenet_v3_small"}
        )
        runtime = SimpleNamespace(_runtime_context=runtime_context, _algo=algo)

        _warm_runtime_models(runtime, items)

        self.assertEqual(
            runtime_context.loaded_algorithms,
            [
                ("efficientnet_b0", "cam1__cam1-roi"),
                ("mobilenet_v3_small", "cam2__cam2-roi"),
            ],
        )
        self.assertEqual(
            algo.feat_net_requests,
            ["efficientnet_b0", "mobilenet_v3_small"],
        )

    def test_all_nonempty_camera_role_combinations_are_preserved(self) -> None:
        all_roles = ("cam1", "cam2", "cam3")
        for size in range(1, len(all_roles) + 1):
            for roles in combinations(all_roles, size):
                with self.subTest(roles=roles):
                    self.assertEqual(configured_camera_roles(roles), list(roles))

    def test_required_roles_follow_enabled_inspection_items(self) -> None:
        runtime = SimpleNamespace(
            _runtime_context=SimpleNamespace(
                inspection_items=[
                    InspectionItem(
                        item_id="cam2-roi",
                        display_name="CAM2 ROI",
                        camera_id="cam2",
                        roi_label="roi",
                    ),
                    InspectionItem(
                        item_id="cam3-roi",
                        display_name="CAM3 ROI",
                        camera_id="cam3",
                        roi_label="roi",
                    ),
                ]
            ),
            _session=SimpleNamespace(),
            _frame_grab_service=_FakeFrameGrabService(),
        )

        self.assertEqual(required_runtime_roles(runtime), ["cam2", "cam3"])

    def test_saved_product_roles_take_priority_over_connected_cameras_and_items(self) -> None:
        runtime = SimpleNamespace(
            _runtime_context=SimpleNamespace(
                inspection_items=[
                    InspectionItem(
                        item_id="cam1-roi",
                        display_name="CAM1 ROI",
                        camera_id="cam1",
                        roi_label="roi",
                    )
                ]
            ),
            _session=SimpleNamespace(
                load_session=lambda: SimpleNamespace(
                    runtime_camera_roles=["cam2", "cam3"]
                )
            ),
            _frame_grab_service=SimpleNamespace(
                roles=lambda: ["cam1", "cam2", "cam3"]
            ),
        )

        self.assertEqual(required_runtime_roles(runtime), ["cam2", "cam3"])

    def test_product_session_persists_cam2_cam3_without_forcing_cam1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session = ProductSession(tmpdir)
            session.load()
            session.switch_product("Default")
            session.save_session(SessionData(runtime_camera_roles=["cam2", "cam3"]))

            self.assertEqual(
                session.load_session().runtime_camera_roles,
                ["cam2", "cam3"],
            )

    def test_precheck_reports_a_missing_required_camera(self) -> None:
        items = [
            InspectionItem(
                item_id=f"{role}-roi",
                display_name=f"{role.upper()} ROI",
                camera_id=role,
                roi_label="roi",
            )
            for role in ("cam2", "cam3")
        ]
        runtime = SimpleNamespace(
            _frame_grab_service=SimpleNamespace(roles=lambda: ["cam2"]),
            _runtime_context=SimpleNamespace(
                inspection_items=items,
                loc_method_for_role=lambda role: "shape",
            ),
            _algo=_FakeAlgo(),
            _connected_roles=lambda: ["cam2"],
        )

        ok, message = _precheck_for_roles(runtime, ["cam2", "cam3"])

        self.assertFalse(ok)
        self.assertEqual(message, "required cameras are not connected: cam3")

    def _make_runtime(self, items, *, traditional_models=None):
        runtime_context = _FakeRuntimeContext(items)
        algo = _FakeAlgo(traditional_models=traditional_models)
        tmpdir = tempfile.TemporaryDirectory()
        recipe_path = Path(tmpdir.name) / "recipe.json"
        recipe_path.write_text("{}", encoding="utf-8")
        runtime = SimpleNamespace(
            _frame_grab_service=_FakeFrameGrabService(),
            _runtime_context=runtime_context,
            _session=SimpleNamespace(line2dup_recipe_path=str(recipe_path)),
            _algo=algo,
            _connected_roles=lambda: ["cam1", "cam2"],
        )
        return runtime, runtime_context, algo, tmpdir

    def test_role_precheck_requires_items_for_requested_camera(self) -> None:
        items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code=SHARED_BACKBONE_ALGORITHM_CODE,
            ),
        ]
        runtime, _, _, tmpdir = self._make_runtime(items)

        original_frame_to_bgr = runtime_controller_module.frame_to_bgr_image
        runtime_controller_module.frame_to_bgr_image = object()
        try:
            ok, message = _precheck_for_roles(runtime, ["cam2"])
        finally:
            runtime_controller_module.frame_to_bgr_image = original_frame_to_bgr
            tmpdir.cleanup()

        self.assertFalse(ok)
        self.assertEqual(message, "please enable at least one inspection tool for cam2")

    def test_role_precheck_ignores_other_camera_training_gaps(self) -> None:
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
                camera_id="cam2",
                roi_label="roi2",
                algorithm_code="meanintensity",
            ),
        ]
        runtime, runtime_context, algo, tmpdir = self._make_runtime(items)

        original_frame_to_bgr = runtime_controller_module.frame_to_bgr_image
        runtime_controller_module.frame_to_bgr_image = object()
        try:
            role_ok, role_message = _precheck_for_roles(runtime, ["cam1"])
            global_ok, global_message = _precheck(runtime)
        finally:
            runtime_controller_module.frame_to_bgr_image = original_frame_to_bgr
            tmpdir.cleanup()

        self.assertTrue(role_ok)
        self.assertEqual(role_message, "")
        self.assertFalse(global_ok)
        self.assertEqual(global_message, "traditional algorithm meanintensity is not trained yet")
        self.assertEqual(runtime_context.loaded_algorithms, [("efficientnet_b0", "cam1__roi1"), ("efficientnet_b0", "cam1__roi1")])
        self.assertEqual(algo.feat_net_requests, ["efficientnet_b0", "efficientnet_b0"])


if __name__ == "__main__":
    unittest.main()
