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
    compact_plot_label,
    list_available_embedding_models,
    load_product_analysis,
)
import tools.visualize_embeddings as visualize_embeddings


class VisualizeEmbeddingsToolModelsTest(unittest.TestCase):
    def test_compact_plot_label_uses_last_underscore_token_and_roi_suffix(self) -> None:
        self.assertEqual(
            compact_plot_label("cam1_debug_cam_20260403_163003_752215.png [roi16]"),
            "752215[r16]",
        )
        self.assertEqual(compact_plot_label("sample_ok.png [roi2]"), "ok[r2]")
        self.assertEqual(compact_plot_label("OK proto"), "OK proto")

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

    def test_list_available_embedding_models_uses_group_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            product_dir = Path(tmpdir) / "demo_product"
            product_dir.mkdir(parents=True, exist_ok=True)
            save_inspection_items(
                [
                    InspectionItem(
                        item_id="roi1",
                        display_name="推块1",
                        camera_id="cam1",
                        roi_label="roi1",
                        task_group="pusher",
                        algorithm_code="shared_backbone_register",
                    ),
                    InspectionItem(
                        item_id="roi2",
                        display_name="推块2",
                        camera_id="cam1",
                        roi_label="roi2",
                        task_group="pusher",
                        algorithm_code="shared_backbone_register",
                    ),
                ],
                str(product_dir / "inspection_items.json"),
            )
            (product_dir / "cam1__pusher_register_model_lt01.npz").write_bytes(b"npz")

            entries = list_available_embedding_models(str(product_dir))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model_key, "cam1__pusher")
        self.assertIn("pusher", entries[0].display_name)

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


    def test_load_product_analysis_group_proto_model_shows_all_analysis_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_root = Path(tmpdir)
            product_dir = session_root / "demo_product"
            product_dir.mkdir(parents=True, exist_ok=True)

            ok_files = []
            ng_files = []
            for name in ("ok_1.png", "ok_2.png"):
                path = product_dir / name
                path.write_bytes(b"ok")
                ok_files.append(str(path))
            for name in ("ng_1.png", "ng_2.png"):
                path = product_dir / name
                path.write_bytes(b"ng")
                ng_files.append(str(path))

            (product_dir / "session.json").write_text(
                json.dumps(
                    {
                        "ok_files": ok_files,
                        "ng_files": ng_files,
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
                        display_name="推块1",
                        camera_id="cam1",
                        roi_label="roi1",
                        task_group="pusher",
                        algorithm_code="shared_backbone_register",
                    )
                ],
                str(product_dir / "inspection_items.json"),
            )
            model_path = product_dir / "cam1__pusher_register_model_lt01.npz"
            model_path.write_bytes(b"npz")

            model = RegisterModel(
                backbone="efficientnet_b0",
                score_mode="proto",
                margin=0.02,
                topk=3,
                label_name="pusher",
                label_names=["pusher"],
                device="cpu",
                ok_proto=np.array([[1.0, 0.0]], dtype=np.float32),
                ng_proto=np.array([[0.0, 1.0]], dtype=np.float32),
                ok_bank=np.array([[1.0, 0.0]], dtype=np.float32),
                ng_bank=np.array([[0.0, 1.0]], dtype=np.float32),
                ok_analysis_bank=np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
                ng_analysis_bank=np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
                ok_analysis_names=["ok_1.png [roi1]", "ok_2.png [roi2]"],
                ng_analysis_names=["ng_1.png [roi1]", "ng_2.png [roi2]"],
                ok_analysis_paths=ok_files,
                ng_analysis_paths=ng_files,
                grouped_proto_only=True,
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
                    ("OK", 0.9, 0.9, 0.1),
                    ("OK", 0.8, 0.85, 0.05),
                    ("NG", -0.8, 0.2, 0.9),
                    ("NG", -0.7, 0.3, 0.95),
                ],
                create=True,
            ):
                result = load_product_analysis(
                    session_root=str(session_root),
                    product_name="demo_product",
                    backbone="efficientnet_b0",
                    model_key="cam1__pusher",
                    projection_method="pca",
                )

        self.assertEqual(result.tool_name, "pusher (cam1)")
        self.assertEqual(
            result.point_names,
            ["ok_1.png [roi1]", "ok_2.png [roi2]", "ng_1.png [roi1]", "ng_2.png [roi2]"],
        )
        self.assertEqual(
            [row.file_name for row in result.rows],
            ["ok_1.png [roi1]", "ok_2.png [roi2]", "ng_1.png [roi1]", "ng_2.png [roi2]"],
        )
        self.assertEqual(result.notes, [])

    def test_load_product_analysis_group_proto_model_recovers_points_from_session_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_root = Path(tmpdir)
            product_dir = session_root / "demo_product"
            product_dir.mkdir(parents=True, exist_ok=True)

            train_files = []
            for name in ("sample_ok.png", "sample_ng.png"):
                path = product_dir / name
                path.write_bytes(b"img")
                train_files.append(f"debug_capture/{name}")
                payload = {
                    "version": "5.0.0",
                    "flags": {},
                    "shapes": [
                        {
                            "label": "roi1",
                            "points": [[0, 0], [10, 10]],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {},
                        },
                        {
                            "label": "roi2",
                            "points": [[0, 0], [10, 10]],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {},
                        },
                    ],
                    "imagePath": name,
                    "imageData": None,
                    "imageHeight": 10,
                    "imageWidth": 10,
                }
                (product_dir / "debug_capture").mkdir(exist_ok=True)
                (product_dir / "debug_capture" / name).write_bytes(b"img")
                (product_dir / "debug_capture" / Path(name).with_suffix(".json")).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )

            (product_dir / "session.json").write_text(
                json.dumps(
                    {
                        "train_files": train_files,
                        "ok_files": [],
                        "ng_files": [],
                        "test_files": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (product_dir / "sample_annotations.json").write_text(
                json.dumps(
                    {
                        "images": {
                            "debug_capture/sample_ok.png": {
                                "roi_status": {
                                    "cam1::roi1": "OK",
                                    "cam1::roi2": "OK",
                                }
                            },
                            "debug_capture/sample_ng.png": {
                                "roi_status": {
                                    "cam1::roi1": "NG",
                                    "cam1::roi2": "NG",
                                }
                            },
                        }
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
                        display_name="hole1",
                        camera_id="cam1",
                        roi_label="roi1",
                        task_group="hole",
                        algorithm_code="shared_backbone_register",
                    ),
                    InspectionItem(
                        item_id="roi2",
                        display_name="hole2",
                        camera_id="cam1",
                        roi_label="roi2",
                        task_group="hole",
                        algorithm_code="shared_backbone_register",
                    ),
                ],
                str(product_dir / "inspection_items.json"),
            )
            model_path = product_dir / "cam1__hole_register_model_lt01.npz"
            model_path.write_bytes(b"npz")

            model = RegisterModel(
                backbone="efficientnet_b0",
                score_mode="proto",
                margin=0.02,
                topk=3,
                label_name="hole",
                label_names=["hole"],
                device="cpu",
                ok_proto=np.array([[1.0, 0.0]], dtype=np.float32),
                ng_proto=np.array([[0.0, 1.0]], dtype=np.float32),
                ok_bank=np.array([[1.0, 0.0]], dtype=np.float32),
                ng_bank=np.array([[0.0, 1.0]], dtype=np.float32),
                grouped_proto_only=True,
            )

            with mock.patch.object(
                visualize_embeddings.qr_core,
                "load_register_model_npz",
                return_value=model,
                create=True,
            ), mock.patch.object(
                visualize_embeddings.qr_core,
                "load_backbone",
                return_value=(object(), 2),
                create=True,
            ), mock.patch.object(
                visualize_embeddings.qr_core,
                "embed_one",
                side_effect=[
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.9, 0.1], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                    np.array([0.1, 0.9], dtype=np.float32),
                ],
                create=True,
            ), mock.patch.object(
                visualize_embeddings.qr_core,
                "predict_one_with_model",
                side_effect=[
                    ("OK", 0.9, 0.9, 0.1),
                    ("OK", 0.8, 0.85, 0.05),
                    ("NG", -0.8, 0.2, 0.9),
                    ("NG", -0.7, 0.3, 0.95),
                ],
                create=True,
            ):
                result = load_product_analysis(
                    session_root=str(session_root),
                    product_name="demo_product",
                    backbone="efficientnet_b0",
                    model_key="cam1__hole",
                    projection_method="pca",
                )

        self.assertEqual(
            result.point_names,
            [
                "sample_ok.png [roi1]",
                "sample_ok.png [roi2]",
                "sample_ng.png [roi1]",
                "sample_ng.png [roi2]",
            ],
        )
        self.assertIn("Recovered grouped analysis points from session annotations.", result.notes)


if __name__ == "__main__":
    unittest.main()
