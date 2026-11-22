from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from application import AlgorithmController, SUPPORTED_LEARNING_ALGORITHMS
from algorithms.yolo_classifier import classifier_paths, prepared_roi_from_path
from common.algorithm_codes import learning_backbone_storage_code, storage_code_backbone
from infrastructure.product_params import ProductRuntimeParams


class YoloClassifierIntegrationTests(unittest.TestCase):
    def test_yolo_is_learning_but_not_embedding(self) -> None:
        controller = AlgorithmController()
        self.assertIn("yolov8n_cls", SUPPORTED_LEARNING_ALGORITHMS)
        self.assertTrue(controller.is_yolo_algorithm("yolov8n_cls"))
        self.assertFalse(controller.is_embedding_algorithm("yolov8n_cls"))
        self.assertEqual(learning_backbone_storage_code("yolov8n_cls"), "y8ncls")
        self.assertEqual(storage_code_backbone("y8ncls"), "yolov8n_cls")

    def test_classifier_paths_are_scoped_to_tool_model_key(self) -> None:
        paths = classifier_paths("C:/product", "cam1__part")
        self.assertEqual(paths.onnx.name, "cam1__part_y8ncls.onnx")
        self.assertEqual(paths.pt.name, "cam1__part_y8ncls.pt")
        self.assertEqual(paths.metadata.name, "cam1__part_y8ncls.json")

    def test_roi_preparation_masks_polygon_and_letterboxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (80, 40), (255, 255, 255)).save(image_path)
            image_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "shapes": [
                            {
                                "label": "roi1",
                                "shape_type": "polygon",
                                "points": [[10, 5], [70, 5], [40, 35]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            prepared = prepared_roi_from_path(image_path, "roi1")
            array = np.asarray(prepared)
            self.assertEqual(prepared.size, (224, 224))
            self.assertTrue(np.any(np.all(array == 0, axis=2)))
            self.assertTrue(np.any(np.all(array == 114, axis=2)))

    def test_yolo_parameters_round_trip_in_product_payload(self) -> None:
        params = ProductRuntimeParams(
            yolo_epochs=12,
            yolo_batch=4,
            yolo_ng_threshold=0.65,
        )
        restored = ProductRuntimeParams.from_dict(params.to_dict())
        self.assertEqual(restored.yolo_epochs, 12)
        self.assertEqual(restored.yolo_batch, 4)
        self.assertAlmostEqual(restored.yolo_ng_threshold, 0.65)


if __name__ == "__main__":
    unittest.main()
