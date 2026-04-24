from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from algorithms.embedding import RegisterModel
from domain.inspection_items import InspectionItem, save_inspection_items
from tools.visualize_embeddings import (
    list_available_embedding_models,
    load_product_analysis,
)
import tools.visualize_embeddings as visualize_embeddings
from ui.i18n import language_code, set_language


class VisualizeEmbeddingsToolModelsTest(unittest.TestCase):
    def test_list_available_embedding_models_uses_tool_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            product_dir = Path(tmpdir) / "demo_product"
            product_dir.mkdir(parents=True, exist_ok=True)
            save_inspection_items(
                [
                    InspectionItem(
                        item_id="roi1",
                        display_name="螺丝1",
                        camera_id="cam1",
                        roi_label="roi1",
                        algorithm_code="shared_backbone_register",
                    )
                ],
                str(product_dir / "inspection_items.json"),
            )
            (product_dir / "cam1__roi1_register_model_lt01.npz").write_bytes(b"npz")

            entries = list_available_embedding_models(str(product_dir))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model_key, "cam1__roi1")
        self.assertEqual(entries[0].backbone, "efficientnet_b0")
        self.assertIn("螺丝1", entries[0].display_name)

    def test_load_product_analysis_prefers_tool_scoped_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_root = Path(tmpdir)
            product_dir = session_root / "demo_product"
            product_dir.mkdir(parents=True, exist_ok=True)

            ok_file = product_dir / "ok_1.png"
            ng_file = product_dir / "ng_1.png"
            ok_file.write_bytes(b"ok")
            ng_file.write_bytes(b"ng")
            (product_dir / "session.json").write_text(
                json.dumps(
                    {
                        "ok_files": [str(ok_file)],
                        "ng_files": [str(ng_file)],
                        "test_files": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (product_dir / "product_params.json").write_text(
                json.dumps(
                    {
                        "algorithm": "efficientnet_b0",
                        "learning_backbone": "efficientnet_b0",
                        "score_mode": "proto",
                        "margin": 0.02,
                        "topk": 3,
                        "traditional_models": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            save_inspection_items(
                [
                    InspectionItem(
                        item_id="roi1",
                        display_name="螺丝1",
                        camera_id="cam1",
                        roi_label="roi1",
                        algorithm_code="shared_backbone_register",
                    )
                ],
                str(product_dir / "inspection_items.json"),
            )
            model_path = product_dir / "cam1__roi1_register_model_lt01.npz"
            model_path.write_bytes(b"npz")

            model = RegisterModel(
                backbone="efficientnet_b0",
                score_mode="proto",
                margin=0.02,
                topk=3,
                label_name="roi1",
                label_names=["roi1"],
                device="cpu",
                ok_proto=np.array([[1.0, 0.0]], dtype=np.float32),
                ng_proto=np.array([[0.0, 1.0]], dtype=np.float32),
                ok_bank=np.array([[1.0, 0.0]], dtype=np.float32),
                ng_bank=np.array([[0.0, 1.0]], dtype=np.float32),
            )

            with mock.patch.object(
                visualize_embeddings.qr_core,
                "load_register_model_npz",
                return_value=model,
                create=True,
            ), mock.patch.object(
                visualize_embeddings.qr_core,
                "predict_one_with_model",
                side_effect=[
                    ("OK", 0.8, 0.9, 0.1),
                    ("NG", -0.7, 0.2, 0.9),
                ],
                create=True,
            ):
                result = load_product_analysis(
                    session_root=str(session_root),
                    product_name="demo_product",
                    backbone="efficientnet_b0",
                    model_key="cam1__roi1",
                    projection_method="pca",
                )

        self.assertTrue(result.model_path.endswith("cam1__roi1_register_model_lt01.npz"))
        self.assertEqual(result.tool_name, "螺丝1 (cam1/roi1)")
        self.assertEqual(result.label_names, ["roi1"])
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.score_mode, "proto")
        self.assertAlmostEqual(result.margin, 0.02, places=6)
        self.assertEqual(result.topk, 3)
        self.assertAlmostEqual(result.metrics["ok_diff_min"], 0.8, places=6)
        self.assertAlmostEqual(result.metrics["ng_diff_max"], -0.7, places=6)
        self.assertAlmostEqual(result.metrics["diff_gap"], 1.5, places=6)
        self.assertAlmostEqual(result.metrics["safe_margin_low"], -0.7, places=6)
        self.assertAlmostEqual(result.metrics["safe_margin_high"], 0.8, places=6)
        self.assertAlmostEqual(result.metrics["suggested_margin"], 0.05, places=6)
        self.assertAlmostEqual(result.metrics["suggested_accuracy"], 1.0, places=6)
        self.assertTrue(any("fully separated" in note for note in result.notes))

    def test_list_available_embedding_models_localizes_shared_model_name(self) -> None:
        previous = language_code()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                product_dir = Path(tmpdir) / "demo_product"
                product_dir.mkdir(parents=True, exist_ok=True)
                save_inspection_items([], str(product_dir / "inspection_items.json"))
                (product_dir / "register_model_lt01.npz").write_bytes(b"npz")

                set_language("en_US", persist=False)
                english_entries = list_available_embedding_models(str(product_dir))

                set_language("zh_CN", persist=False)
                chinese_entries = list_available_embedding_models(str(product_dir))

            self.assertEqual(english_entries[0].tool_name, "Shared model")
            self.assertIn("Shared model / High-Accuracy Learning Tool", english_entries[0].display_name)
            self.assertEqual(chinese_entries[0].tool_name, "共享模型")
            self.assertIn("共享模型 / 高精度学习工具", chinese_entries[0].display_name)
        finally:
            set_language(previous, persist=False)


if __name__ == "__main__":
    unittest.main()
