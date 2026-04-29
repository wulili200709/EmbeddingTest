from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import algorithms.embedding as embedding


class _FakeSessionIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def get_inputs(self):
        return [_FakeSessionIO("input")]

    def get_outputs(self):
        return [_FakeSessionIO("features")]

    def run(self, output_names, inputs):
        array = np.asarray(inputs["input"], dtype=np.float32)
        return [array[:, :1, :, :]]


class EmbeddingOnnxRuntimeTest(unittest.TestCase):
    def test_ort_backbone_path_uses_project_session_cache_and_storage_code(self) -> None:
        expected = {
            "b0": "lt01_features_opset17.onnx",
            "b1": "lt02_features_opset17.onnx",
            "b2": "lt03_features_opset17.onnx",
        }
        for backbone, file_name in expected.items():
            with self.subTest(backbone=backbone):
                path = Path(embedding._ort_backbone_path(backbone))
                self.assertEqual(path.name, file_name)
                self.assertEqual(path.parent.name, "_onnx_cache")
                self.assertEqual(path.parent.parent.name, ".qr_session")

    def test_onnx_wrapper_returns_torch_tensor(self) -> None:
        wrapper = embedding._OnnxRuntimeFeatureNet(_FakeSession())
        batch = torch.randn(2, 3, 16, 16, dtype=torch.float32)
        out = wrapper(batch)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2, 1, 16, 16))
        self.assertEqual(out.dtype, torch.float32)

    def test_load_backbone_prefers_onnxruntime_for_learning_backbones_on_cpu(self) -> None:
        expected_out_ch = {
            "b0": 1280,
            "b1": 1280,
            "b2": 1408,
        }
        for backbone, out_ch_expected in expected_out_ch.items():
            with self.subTest(backbone=backbone):
                fake_torch_feat = mock.Mock()
                fake_ort_feat = object()
                with mock.patch.object(
                    embedding,
                    "_build_torch_backbone",
                    return_value=(fake_torch_feat, out_ch_expected),
                ), mock.patch.object(
                    embedding,
                    "_maybe_load_ort_backbone",
                    return_value=fake_ort_feat,
                ):
                    feat, out_ch = embedding.load_backbone(backbone, device="cpu")

                self.assertIs(feat, fake_ort_feat)
                self.assertEqual(out_ch, out_ch_expected)

    def test_load_backbone_falls_back_to_torch_when_ort_unavailable(self) -> None:
        expected_out_ch = {
            "b0": 1280,
            "b1": 1280,
            "b2": 1408,
        }
        for backbone, out_ch_expected in expected_out_ch.items():
            with self.subTest(backbone=backbone):
                fake_torch_feat = mock.Mock()
                fake_torch_feat.eval.return_value = fake_torch_feat
                with mock.patch.object(
                    embedding,
                    "_build_torch_backbone",
                    return_value=(fake_torch_feat, out_ch_expected),
                ), mock.patch.object(
                    embedding,
                    "_maybe_load_ort_backbone",
                    return_value=None,
                ):
                    feat, out_ch = embedding.load_backbone(backbone, device="cpu")

                self.assertIs(feat, fake_torch_feat)
                self.assertEqual(out_ch, out_ch_expected)
                fake_torch_feat.eval.assert_called_once()
                fake_torch_feat.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
