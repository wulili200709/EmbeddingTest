from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from algorithms.registry import is_learning_tool_algorithm, is_traditional_tool_algorithm
from application.runtime.execution import (
    _precheck_for_capture_channels, _precheck_for_roles, _warm_runtime_models,
)
from domain import InspectionItem


def shared_item(label, *, source="library", algorithm="shared_backbone_register", camera="cam1"):
    return InspectionItem(
        label, label, camera, label, algorithm_code=algorithm,
        model_source=source, task_group="pin group" if source == "group" else "",
        library_model_id="a" * 32 if source == "library" else "",
        library_version="v000001" if source == "library" else "",
    )


def runtime_for(items, *, ready=True, message=""):
    context = SimpleNamespace(
        inspection_items=items, loc_method_for_role=lambda _role: "shape",
        load_embedding_model=Mock(),
    )
    algo = SimpleNamespace(
        is_learning_tool=is_learning_tool_algorithm,
        is_traditional_tool=is_traditional_tool_algorithm,
        resolve_item_algorithm=Mock(side_effect=lambda item: (
            "mobilenet_v3_small" if is_learning_tool_algorithm(item.algorithm_code) else item.algorithm_code
        )),
        current_learning_backbone=Mock(side_effect=AssertionError("must use published algorithm")),
        get_feat_net=Mock(return_value=object()),
        get_traditional_model_dict=Mock(return_value={"threshold": 10}),
        describe_model_binding=Mock(return_value={"ready": ready, "message": message}),
    )
    return SimpleNamespace(
        _algo=algo, _runtime_context=context,
        _session=SimpleNamespace(product_dir="order-b"),
        _frame_grab_service=SimpleNamespace(roles=lambda: ["cam1", "cam2"]),
        _connected_roles=lambda: ["cam1", "cam2"],
    )


class RuntimeSharedModelPrecheckTests(unittest.TestCase):
    def test_library_model_is_warmed_by_reference_with_its_own_algorithm(self):
        first, second = shared_item("roi1"), shared_item("roi2")
        runtime = runtime_for([first, second])
        with patch("application.runtime.controller.frame_to_bgr_image", object()):
            ok, message = _precheck_for_roles(runtime, ["cam1"])
        self.assertTrue(ok, message)
        runtime._runtime_context.load_embedding_model.assert_called_once_with(
            "mobilenet_v3_small", model_key=first.effective_model_key,
        )
        runtime._algo.current_learning_backbone.assert_not_called()
        runtime._algo.describe_model_binding.assert_called_once_with(first, "order-b")

    def test_missing_library_model_blocks_production_before_warmup(self):
        runtime = runtime_for([shared_item("roi1")], ready=False, message="published version missing")
        with patch("application.runtime.controller.frame_to_bgr_image", object()):
            ok, message = _precheck_for_roles(runtime, ["cam1"])
        self.assertFalse(ok)
        self.assertIn("published version missing", message)
        runtime._runtime_context.load_embedding_model.assert_not_called()

    def test_library_model_load_error_is_not_silently_ignored(self):
        runtime = runtime_for([shared_item("roi1")])
        runtime._runtime_context.load_embedding_model.side_effect = ValueError("invalid model data")
        with patch("application.runtime.controller.frame_to_bgr_image", object()):
            ok, message = _precheck_for_roles(runtime, ["cam1"])
        self.assertFalse(ok)
        self.assertIn("invalid model data", message)

    def test_unavailable_reference_blocks_multi_light_capture_precheck_too(self):
        runtime = runtime_for([shared_item("roi1")], ready=False, message="checksum mismatch")
        with patch("application.runtime.controller.frame_to_bgr_image", object()):
            ok, message = _precheck_for_capture_channels(runtime, [{"role": "cam1", "physical_role": "cam1"}])
        self.assertFalse(ok)
        self.assertIn("checksum mismatch", message)

    def test_other_camera_reference_does_not_block_requested_camera(self):
        runtime = runtime_for([shared_item("roi1"), shared_item("roi2", camera="cam2")])
        runtime._algo.describe_model_binding.side_effect = lambda item, _order: {
            "ready": item.camera_id == "cam1", "message": "cam2 missing",
        }
        with patch("application.runtime.controller.frame_to_bgr_image", object()):
            ok, message = _precheck_for_roles(runtime, ["cam1"])
        self.assertTrue(ok, message)
        self.assertEqual(runtime._algo.describe_model_binding.call_count, 1)

    def test_conflicting_group_algorithms_block_startup(self):
        runtime = runtime_for([
            shared_item("roi1", source="group"),
            shared_item("roi2", source="group", algorithm="meanintensity"),
        ])
        with patch("application.runtime.controller.frame_to_bgr_image", object()):
            ok, message = _precheck_for_roles(runtime, ["cam1"])
        self.assertFalse(ok)
        self.assertIn("incompatible algorithms", message)

    def test_traditional_warmup_uses_group_key(self):
        item = shared_item("roi1", source="group", algorithm="meanintensity")
        runtime = runtime_for([item])
        _warm_runtime_models(runtime, [item])
        runtime._algo.get_traditional_model_dict.assert_called_once_with(
            "meanintensity", model_key=item.effective_model_key,
        )


if __name__ == "__main__":
    unittest.main()
