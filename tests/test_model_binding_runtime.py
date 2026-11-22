"""Behavior checks for published models across orders and ROI positions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from algorithms.embedding import RegisterModel, save_register_model_npz
from algorithms.embedding_core import compute_prototypes
from application.algorithm_controller import AlgorithmController
from application.runtime_context import (
    ProductRuntimeContext,
    _predict_learning_items_batch_rows,
    _predict_learning_items_batch_rows_from_frame,
)
from common.model_library import ModelLibrary
from domain import InspectionItem
from application.runtime.preview_frame import RuntimePreviewShape


class ModelBindingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.order = self.root / "new-order"
        self.order.mkdir()
        self.controller = AlgorithmController()
        self.controller.model_library = ModelLibrary(self.root / "library")
        self.ok = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.ng = np.array([0., 1., 0., 0.], dtype=np.float32)

    def publish_embedding(self):
        bank_ok = np.stack([self.ok, self.ok])
        bank_ng = np.stack([self.ng, self.ng])
        ok_proto, ng_proto = compute_prototypes(bank_ok, bank_ng)
        source = self.root / "source-model.npz"
        save_register_model_npz(RegisterModel(
            backbone="efficientnet_b0", score_mode="proto", margin=.02, topk=1,
            label_name="source_roi", ok_proto=ok_proto, ng_proto=ng_proto,
            ok_bank=bank_ok, ng_bank=bank_ng,
        ), str(source))
        publication = self.controller.model_library.publish(
            "同类目标", "efficientnet_b0", model_file=source,
            parameters={"score_mode": "proto", "margin": .02, "topk": 1},
        )
        source.unlink()
        return publication

    def library_item(self, label, publication, algorithm="shared_backbone_register"):
        return InspectionItem(label, label, "cam1", label, algorithm,
                              model_source="library",
                              library_model_id=publication["model_id"],
                              library_version=publication["version"])

    def test_published_embedding_predicts_other_rois_without_order_samples_or_backbone_setting(self):
        publication = self.publish_embedding()
        items = [self.library_item("roi2", publication), self.library_item("roi3", publication)]
        # These deliberately disagree with the published decision parameters.
        self.controller.product_params.score_mode = "topk"
        self.controller.product_params.margin = 99.
        self.controller.product_params.topk = 17
        loader = lambda algorithm, model_key: self.controller.load_model_for_algorithm(
            algorithm, str(self.order), model_key=model_key
        )
        with patch("application.runtime_context.qr_core.embed_batch", return_value=np.stack([self.ok, self.ng])) as embed:
            rows = _predict_learning_items_batch_rows(
                path="new-order-image.png", items=items, match_ms=0., algo=self.controller,
                load_embedding_model=loader, feat_net=object(),
            )
        self.assertEqual([rows[item.model_key]["pred"] for item in items], ["OK", "NG"])
        self.assertEqual(embed.call_args.args[2], ["roi2", "roi3"])
        self.assertEqual(self.controller.model.score_mode, "proto")
        self.assertAlmostEqual(self.controller.model.margin, .02)
        self.assertEqual(self.controller.model.topk, 1)
        self.assertEqual(self.controller._active_product_dir, str(self.order))
        self.assertEqual(list(self.order.iterdir()), [])

    def test_live_frame_uses_shared_model_but_keeps_independent_roi_results(self):
        publication = self.publish_embedding()
        items = [self.library_item("left", publication), self.library_item("right", publication)]
        shapes = tuple(RuntimePreviewShape(label=item.roi_label, shape_type="rectangle",
                                          points=((0., 0.), (8., 8.))) for item in items)
        loader = lambda algorithm, model_key: self.controller.load_model_for_algorithm(
            algorithm, str(self.order), model_key=model_key
        )
        with patch("algorithms.embedding.embed_batch_from_array", return_value=np.stack([self.ng, self.ok])) as embed:
            rows = _predict_learning_items_batch_rows_from_frame(
                image_bgr=np.zeros((10, 10, 3), np.uint8), roi_shapes=shapes,
                items=items, match_ms=0., algo=self.controller,
                load_embedding_model=loader, feat_net=object(),
            )
        self.assertEqual([rows[item.model_key]["pred"] for item in items], ["NG", "OK"])
        self.assertEqual(embed.call_args.args[2], ["left", "right"])

    def test_missing_publication_never_falls_back_to_local_model(self):
        publication = self.publish_embedding()
        item = self.library_item("new_roi", publication)
        Path(publication["model_file"]).unlink()
        # A local file exists, but is not a replacement for the bound version.
        (self.order / "register_model_b0.npz").write_bytes(b"wrong-model")
        self.assertFalse(self.controller.describe_model_binding(item, str(self.order))["ready"])
        with self.assertRaises((OSError, ValueError)):
            self.controller.load_model_for_algorithm("efficientnet_b0", str(self.order),
                                                     model_key=item.effective_model_key)

    def test_switching_orders_does_not_treat_independent_legacy_model_as_previous_orders_group(self):
        previous = InspectionItem("roi2", "Shared", "cam1", "roi2", "meanintensity", task_group="roi1")
        self.controller.resolve_item_algorithm(previous)
        threshold = {"algorithm": "meanintensity", "threshold": 128., "ok_when": "greater_equal"}
        self.controller.product_params.traditional_models = {"meanintensity": threshold}
        params_path = str(self.order / "product_params.json")
        self.controller.save_params(params_path)
        # Unkeyed model assets were supported by old independent orders.
        model_path = self.order / "register_model_b0.npz"
        model_path.write_bytes(b"legacy independent model")
        self.assertIsNone(self.controller.get_traditional_model_dict("meanintensity", model_key=previous.effective_model_key))
        self.assertEqual(self.controller._local_embedding_file("efficientnet_b0", str(self.order), previous.effective_model_key), "")
        self.controller.load_params(params_path)
        current = InspectionItem("roi1", "Independent", "cam1", "roi1", "meanintensity", model_source="independent")
        self.assertEqual(self.controller.get_traditional_model_dict("meanintensity", model_key=current.effective_model_key), threshold)
        self.assertEqual(self.controller._local_embedding_file("efficientnet_b0", str(self.order), current.effective_model_key), str(model_path))

    def test_public_model_cannot_be_overwritten_by_training(self):
        item = self.library_item("roi2", self.publish_embedding())
        with self.assertRaisesRegex(ValueError, "只读"):
            self.controller.train([], [], algorithm="efficientnet_b0",
                                  product_dir=str(self.order), label_names=["roi2"],
                                  model_key=item.effective_model_key)

    def test_shared_traditional_model_reads_target_roi_in_file_and_live_frame(self):
        model = dict(algorithm="meanintensity", threshold=128., ok_when="greater_equal",
                     ok_mean=255., ng_mean=0., accuracy=1., roi_label="source_roi")
        publication = self.controller.model_library.publish("亮度模型", "meanintensity", traditional_model=model)
        items = [self.library_item("dark", publication, "meanintensity"),
                 self.library_item("bright", publication, "meanintensity")]
        image = np.zeros((20, 60, 3), dtype=np.uint8)
        image[:, 40:] = 255
        image[:, :20] = 255  # The source ROI is OK; target dark ROI must still be NG.
        image_path = self.order / "different-regions.png"
        cv2.imwrite(str(image_path), image)
        shapes = [dict(label="source_roi", shape_type="rectangle", points=[[1, 1], [18, 18]]),
                  dict(label="dark", shape_type="rectangle", points=[[21, 1], [38, 18]]),
                  dict(label="bright", shape_type="rectangle", points=[[41, 1], [58, 18]])]
        image_path.with_suffix(".json").write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
        results = [self.controller.predict_image(str(image_path), labels=[item.roi_label],
                                                algorithm_override=item.algorithm_code,
                                                model_key_override=item.effective_model_key).pred
                   for item in items]
        self.assertEqual(results, ["NG", "OK"])
        context = ProductRuntimeContext.__new__(ProductRuntimeContext)
        context.algo = self.controller
        context.session = SimpleNamespace(product_dir=str(self.order))
        context._loc_method = "ncc"
        context._loc_methods_by_role = {"cam1": "ncc"}
        located = SimpleNamespace(locate_ms=0., detected_product_count=1, roi_shapes=tuple(
            SimpleNamespace(label_name=shape["label"], shape_type=shape["shape_type"],
                            points=shape["points"]) for shape in shapes))
        with patch("application.runtime_context.ncc_locator.autogen_runtime_roi_shapes_timed", return_value=located):
            batch = context.predict_items_batch_from_frame(image, camera_role="cam1", items=items)
        self.assertEqual([row["pred"] for row in batch.rows], ["NG", "OK"])


if __name__ == "__main__":
    unittest.main()
