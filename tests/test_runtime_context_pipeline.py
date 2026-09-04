from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from application.runtime_context import (
    ProductRuntimeContext,
    _group_inspection_items,
    _predict_grouped_items_from_path,
)
from domain import InspectionItem


class _FakeAlgorithmController:
    model = None

    @staticmethod
    def is_learning_tool(code: object) -> bool:
        return str(code) == "learning_tool"

    @staticmethod
    def is_measurement_tool(code: object) -> bool:
        return str(code) == "bright_block_center"


class RuntimeContextPipelineTests(unittest.TestCase):
    def test_grouping_has_one_shared_definition_for_debug_and_runtime(self) -> None:
        algo = _FakeAlgorithmController()
        items = [
            InspectionItem("traditional", "Traditional", "cam1", "roi1", "meanintensity"),
            InspectionItem("measurement", "Measurement", "cam1", "roi2", "bright_block_center"),
            InspectionItem("disabled", "Disabled", "cam1", "roi3", "meanstd", enabled=False),
        ]

        groups = _group_inspection_items(items, algo)

        self.assertEqual([item.item_id for item in groups.enabled], ["traditional", "measurement"])
        self.assertEqual([item.item_id for item in groups.traditional], ["traditional"])
        self.assertEqual([item.item_id for item in groups.measurement], ["measurement"])
        self.assertEqual(groups.learning, [])

    def test_shared_path_pipeline_preserves_order_and_measurement_params(self) -> None:
        algo = _FakeAlgorithmController()
        items = [
            InspectionItem("traditional", "Traditional", "cam1", "roi1", "meanintensity"),
            InspectionItem(
                "measurement",
                "Measurement",
                "cam1",
                "roi2",
                "bright_block_center",
                params={"threshold": 12},
            ),
        ]
        groups = _group_inspection_items(items, algo)
        calls: list[dict[str, object]] = []

        def predict_image(path: str, **kwargs) -> dict[str, object]:
            calls.append({"path": path, **kwargs})
            return {"pred": "OK", "algorithm": kwargs["algorithm_override"]}

        rows = _predict_grouped_items_from_path(
            path="sample.bmp",
            groups=groups,
            match_ms=4.0,
            algo=algo,
            predict_image=predict_image,
            load_embedding_model=lambda *_args, **_kwargs: None,
        )

        self.assertEqual([row["algorithm"] for row in rows], ["meanintensity", "bright_block_center"])
        self.assertNotIn("params_override", calls[0])
        self.assertEqual(calls[1]["params_override"], {"threshold": 12})

    def test_multiple_shape_matches_skip_normal_detection(self) -> None:
        context = ProductRuntimeContext.__new__(ProductRuntimeContext)
        context.algo = _FakeAlgorithmController()
        context.session = SimpleNamespace(product_dir="product")
        context._loc_method = "shape"
        context._loc_methods_by_role = {"cam1": "shape"}
        context._ensure_recipe_loaded = lambda _role: object()
        context._reference_image = lambda _recipe: "reference.png"
        shape_run = SimpleNamespace(
            locate_ms=4.0,
            result=SimpleNamespace(detected_product_count=2),
            roi_shapes=(),
        )
        item = InspectionItem("traditional", "Traditional", "cam1", "roi1", "meanintensity")

        with (
            patch("application.runtime_context.os.path.exists", return_value=True),
            patch(
                "application.runtime_context.shape_locator.autogen_runtime_roi_shapes_timed",
                return_value=shape_run,
            ),
        ):
            batch = context.predict_items_batch_from_frame(
                [[0]],
                camera_role="cam1",
                items=[item],
            )

        self.assertEqual(batch.rows[0]["detected_product_count"], 2)
        self.assertEqual(batch.rows[0]["pred"], "NG")
        self.assertEqual(batch.rows[0]["infer_ms"], 0.0)

    def test_multiple_ncc_matches_skip_normal_detection(self) -> None:
        context = ProductRuntimeContext.__new__(ProductRuntimeContext)
        context.algo = _FakeAlgorithmController()
        context.session = SimpleNamespace(product_dir="product")
        context._loc_method = "ncc"
        context._loc_methods_by_role = {"cam1": "ncc"}
        ncc_run = SimpleNamespace(
            locate_ms=5.0,
            detected_product_count=2,
            roi_shapes=(),
        )
        item = InspectionItem("traditional", "Traditional", "cam1", "roi1", "meanintensity")

        with patch(
            "application.runtime_context.ncc_locator.autogen_runtime_roi_shapes_timed",
            return_value=ncc_run,
        ):
            batch = context.predict_items_batch_from_frame(
                [[0]],
                camera_role="cam1",
                items=[item],
            )

        self.assertEqual(batch.rows[0]["detected_product_count"], 2)
        self.assertEqual(batch.rows[0]["pred"], "NG")
        self.assertEqual(batch.rows[0]["infer_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
