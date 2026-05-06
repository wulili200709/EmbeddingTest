from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from algorithms.registry import (
    DEFAULT_LEARNING_BACKBONE,
    SHARED_BACKBONE_ALGORITHM_CODE,
    algorithm_display_name,
    get_tool_algorithm_spec,
    is_learning_tool_algorithm,
    is_measurement_tool_algorithm,
    is_traditional_tool_algorithm,
    normalize_tool_algorithm_code,
    storage_code_backbone,
)
from application.algorithm_controller import AlgorithmController
from infrastructure.product_params import ProductRuntimeParams
from ui.i18n import language_code, set_language


class ToolAlgorithmRegistryTest(unittest.TestCase):
    def test_legacy_shared_backbone_codes_normalize_to_single_code(self) -> None:
        self.assertEqual(
            normalize_tool_algorithm_code("inherit_product"),
            SHARED_BACKBONE_ALGORITHM_CODE,
        )
        self.assertEqual(
            normalize_tool_algorithm_code("efficientnet_b0"),
            SHARED_BACKBONE_ALGORITHM_CODE,
        )
        self.assertEqual(
            normalize_tool_algorithm_code("b0"),
            SHARED_BACKBONE_ALGORITHM_CODE,
        )
        self.assertEqual(
            normalize_tool_algorithm_code("lt_b0"),
            SHARED_BACKBONE_ALGORITHM_CODE,
        )

    def test_storage_backbone_codes_normalize_to_backbone_names(self) -> None:
        self.assertEqual(storage_code_backbone("b0"), "efficientnet_b0")
        self.assertEqual(storage_code_backbone("b1"), "mobilenet_v3_small")
        self.assertEqual(storage_code_backbone("b2"), "mobilenet_v3_large")
        self.assertEqual(storage_code_backbone("lt_b0"), "efficientnet_b0")

    def test_registry_knows_learning_and_traditional_families(self) -> None:
        self.assertTrue(is_learning_tool_algorithm(SHARED_BACKBONE_ALGORITHM_CODE))
        self.assertTrue(is_traditional_tool_algorithm("meanintensity"))
        self.assertEqual(get_tool_algorithm_spec("meanhsv_h").family, "traditional")
        self.assertTrue(is_measurement_tool_algorithm("find_line"))
        self.assertEqual(get_tool_algorithm_spec("find_line").family, "measurement")
        self.assertTrue(is_measurement_tool_algorithm("line_distance"))
        self.assertEqual(get_tool_algorithm_spec("line_distance").family, "measurement")

    def test_product_params_accept_learning_backbone_field(self) -> None:
        params = ProductRuntimeParams.from_dict(
            {
                "algorithm": "",
                "learning_backbone": "mobilenet_v3_small",
                "score_mode": "proto",
                "margin": 0.02,
                "topk": 3,
            }
        )

        self.assertEqual(params.learning_backbone, "mobilenet_v3_small")
        self.assertEqual(params.to_dict()["learning_backbone"], "b1")
        self.assertEqual(ProductRuntimeParams.from_dict({"learning_backbone": "b1"}).learning_backbone, "mobilenet_v3_small")
        fallback = ProductRuntimeParams.from_dict({})
        self.assertEqual(fallback.learning_backbone, "")
        self.assertEqual(DEFAULT_LEARNING_BACKBONE, "efficientnet_b0")

    def test_algorithm_controller_keeps_learning_tool_unselected_when_params_empty(self) -> None:
        controller = AlgorithmController()
        with tempfile.TemporaryDirectory() as tmpdir:
            controller.load_params(str(Path(tmpdir) / "product_params.json"))
        self.assertEqual(controller.product_params.algorithm, "")
        self.assertEqual(controller.current_learning_backbone(), "")

    def test_algorithm_display_name_follows_current_language(self) -> None:
        previous = language_code()
        try:
            set_language("en_US", persist=False)
            self.assertEqual(algorithm_display_name(SHARED_BACKBONE_ALGORITHM_CODE), "Learning Tools")
            self.assertEqual(algorithm_display_name("efficientnet_b0"), "High-Accuracy Learning Tool")
            self.assertEqual(algorithm_display_name("line_distance"), "Line Distance")

            set_language("zh_CN", persist=False)
            self.assertEqual(algorithm_display_name(SHARED_BACKBONE_ALGORITHM_CODE), "学习工具")
            self.assertEqual(algorithm_display_name("efficientnet_b0"), "高精度学习工具")
            self.assertEqual(algorithm_display_name("line_distance"), "距离测量")
        finally:
            set_language(previous, persist=False)


if __name__ == "__main__":
    unittest.main()
