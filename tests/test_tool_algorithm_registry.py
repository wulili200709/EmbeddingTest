from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from algorithms.registry import (
    DEFAULT_LEARNING_BACKBONE,
    SHARED_BACKBONE_ALGORITHM_CODE,
    get_tool_algorithm_spec,
    is_learning_tool_algorithm,
    is_traditional_tool_algorithm,
    normalize_tool_algorithm_code,
)
from infrastructure.product_params import ProductRuntimeParams


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

    def test_registry_knows_learning_and_traditional_families(self) -> None:
        self.assertTrue(is_learning_tool_algorithm(SHARED_BACKBONE_ALGORITHM_CODE))
        self.assertTrue(is_traditional_tool_algorithm("meanintensity"))
        self.assertEqual(get_tool_algorithm_spec("meanhsv_h").family, "traditional")

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
        fallback = ProductRuntimeParams.from_dict({})
        self.assertEqual(fallback.learning_backbone, "")
        self.assertEqual(DEFAULT_LEARNING_BACKBONE, "efficientnet_b0")


if __name__ == "__main__":
    unittest.main()
