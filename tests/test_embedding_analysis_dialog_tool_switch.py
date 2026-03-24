import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PySide6 import QtTest, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from tools.visualize_embeddings import EmbeddingAnalysisResult, EmbeddingModelEntry
from ui.debug.embedding_analysis_dialog import EmbeddingAnalysisDialog


def _analysis_result(tool_name: str, model_path: str, model_key: str) -> EmbeddingAnalysisResult:
    return EmbeddingAnalysisResult(
        product_name="demo_product",
        backbone="efficientnet_b0",
        model_path=model_path,
        session_file="session.json",
        projection_method="pca",
        feature_dim=2,
        point_coords=np.zeros((2, 2), dtype=np.float32),
        point_labels=["OK", "NG"],
        point_names=["ok.png", "ng.png"],
        rows=[],
        metrics={
            "ok_count": 1.0,
            "ng_count": 1.0,
            "train_accuracy": 1.0,
            "ok_intra_mean": 1.0,
            "ng_intra_mean": 1.0,
            "ok_ng_cross_mean": 0.0,
            "ok_to_ok_proto": 1.0,
            "ng_to_ng_proto": 1.0,
            "ok_to_ng_proto": 0.0,
            "ng_to_ok_proto": 0.0,
            "proto_similarity": 0.0,
        },
        notes=[],
        model_key=model_key,
        tool_name=tool_name,
        label_names=[model_key.split("__")[-1] if "__" in model_key else "roi"],
    )


class EmbeddingAnalysisDialogToolSwitchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_switching_learning_tool_refreshes_analysis_result(self) -> None:
        entries = [
            EmbeddingModelEntry(
                model_key="cam1__roi1",
                backbone="efficientnet_b0",
                model_path="cam1__roi1_register_model_lt01.npz",
                display_name="密封圈1 (cam1/roi1) / 高精度学习工具",
                tool_name="密封圈1 (cam1/roi1)",
            ),
            EmbeddingModelEntry(
                model_key="cam1__roi2",
                backbone="efficientnet_b0",
                model_path="cam1__roi2_register_model_lt01.npz",
                display_name="密封圈2 (cam1/roi2) / 高精度学习工具",
                tool_name="密封圈2 (cam1/roi2)",
            ),
        ]
        load_calls: list[tuple[str, str, str]] = []

        def _fake_load(*, session_root: str, product_name: str, backbone: str, model_key: str, projection_method: str):
            load_calls.append((product_name, model_key, projection_method))
            entry = next(item for item in entries if item.model_key == model_key)
            return _analysis_result(entry.tool_name, entry.model_path, entry.model_key)

        with mock.patch("ui.debug.embedding_analysis_dialog.list_product_names", return_value=["demo_product"]), \
            mock.patch("ui.debug.embedding_analysis_dialog.list_available_embedding_models", return_value=entries), \
            mock.patch("ui.debug.embedding_analysis_dialog.load_product_analysis", side_effect=_fake_load), \
            mock.patch("ui.debug.embedding_analysis_dialog.os.path.getmtime", return_value=1.0), \
            mock.patch("ui.debug.embedding_analysis_dialog.os.path.exists", return_value=True):
            dialog = EmbeddingAnalysisDialog(
                session_root="dummy_root",
                initial_product="demo_product",
                initial_backbone="efficientnet_b0",
                initial_model_key="cam1__roi1",
            )
            try:
                QtTest.QTest.qWait(120)
                self.app.processEvents()
                self.assertIn("密封圈1", dialog.txt_summary.toPlainText())
                self.assertEqual(dialog.lbl_model_path.text(), entries[0].model_path)

                dialog.cmb_model.setCurrentIndex(1)
                QtTest.QTest.qWait(120)
                self.app.processEvents()

                self.assertIn("密封圈2", dialog.txt_summary.toPlainText())
                self.assertEqual(dialog.lbl_model_path.text(), entries[1].model_path)
                self.assertIn(("demo_product", "cam1__roi2", "tsne"), load_calls)
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
