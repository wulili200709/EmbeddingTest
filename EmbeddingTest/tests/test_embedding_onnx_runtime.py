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
        path = Path(embedding._ort_backbone_path("mobilenet_v3_large"))
        self.assertEqual(path.name, "lt03_features_opset17.onnx")
        self.assertEqual(path.parent.name, "_onnx_cache")
        self.assertEqual(path.parent.parent.name, ".qr_session")

    def test_onnx_wrapper_returns_torch_tensor(self) -> None:
        wrapper = embedding._OnnxRuntimeFeatureNet(_FakeSession())
        batch = torch.randn(2, 3, 16, 16, dtype=torch.float32)
        out = wrapper(batch)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2, 1, 16, 16))
        self.assertEqual(out.dtype, torch.float32)

    def test_load_backbone_prefers_onnxruntime_for_lt03_cpu(self) -> None:
        fake_torch_feat = mock.Mock()
        fake_ort_feat = object()
        with mock.patch.object(
            embedding,
            "_build_torch_backbone",
            return_value=(fake_torch_feat, 960),
        ), mock.patch.object(
            embedding,
            "_maybe_load_ort_backbone",
            return_value=fake_ort_feat,
        ):
            feat, out_ch = embedding.load_backbone("mobilenet_v3_large", device="cpu")

        self.assertIs(feat, fake_ort_feat)
        self.assertEqual(out_ch, 960)

    def test_load_backbone_falls_back_to_torch_when_ort_unavailable(self) -> None:
        fake_torch_feat = mock.Mock()
        fake_torch_feat.eval.return_value = fake_torch_feat
        with mock.patch.object(
            embedding,
            "_build_torch_backbone",
            return_value=(fake_torch_feat, 960),
        ), mock.patch.object(
            embedding,
            "_maybe_load_ort_backbone",
            return_value=None,
        ):
            feat, out_ch = embedding.load_backbone("mobilenet_v3_large", device="cpu")

        self.assertIs(feat, fake_torch_feat)
        self.assertEqual(out_ch, 960)
        fake_torch_feat.eval.assert_called_once()
        fake_torch_feat.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
