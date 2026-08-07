from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class _FakeTensor:
    def __init__(self, array=None) -> None:
        self._array = np.asarray(array if array is not None else np.zeros((1, 3, 224, 224), dtype=np.float32))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._array, dtype=np.float32)

    def to(self, _device):
        return self


class _FakeModuleBase:
    def eval(self):
        return self

    def cpu(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, batch):
        if isinstance(batch, _FakeTensor):
            batch_np = batch.numpy()
            return _FakeTensor(np.zeros((len(batch_np), 4), dtype=np.float32))
        return _FakeTensor(np.zeros((1, 4), dtype=np.float32))


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.graph_optimization_level = None
        self.optimized_model_filepath = ""
        self._config: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self._config[key] = value


class _FakeInferenceSession:
    def __init__(self, path: str, sess_options=None, providers=None) -> None:
        self.path = str(path)
        self.providers = list(providers or [])
        if (
            sess_options is not None
            and getattr(sess_options, "optimized_model_filepath", "")
            and getattr(sess_options, "_config", {}).get("session.save_model_format") == "ORT"
        ):
            Path(sess_options.optimized_model_filepath).write_bytes(b"ort")
        self.batch_sizes: list[int] = []
        self._inputs = [types.SimpleNamespace(name="input", shape=["batch_size", 3, 224, 224])]
        self._outputs = [types.SimpleNamespace(name="embedding")]

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def run(self, _output_names, inputs):
        batch = np.asarray(next(iter(inputs.values())), dtype=np.float32)
        self.batch_sizes.append(int(len(batch)))
        return [np.zeros((len(batch), 4), dtype=np.float32)]


class _FakePilImageContext:
    size = (224, 224)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def convert(self, _mode):
        return self


def _fake_dependency_modules() -> dict[str, object]:
    fake_embedding_core = types.ModuleType("algorithms.embedding_core")
    fake_embedding_core.compute_prototypes = lambda *args, **kwargs: None
    fake_embedding_core.predict_one = lambda *args, **kwargs: None
    fake_embedding_core.score_topk = lambda *args, **kwargs: None

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = _FakeTensor
    fake_torch.float32 = np.float32
    fake_torch.randn = lambda *shape, **_kwargs: _FakeTensor(np.zeros(shape, dtype=np.float32))
    fake_torch.stack = lambda tensors, dim=0: _FakeTensor(
        np.stack([tensor.numpy() if isinstance(tensor, _FakeTensor) else np.asarray(tensor) for tensor in tensors], axis=dim)
    )
    fake_torch.no_grad = lambda: (lambda fn: fn)
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.onnx = types.SimpleNamespace(
        export=lambda _model, _dummy, target, **_kwargs: Path(target).write_bytes(b"onnx")
    )

    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_nn.Module = _FakeModuleBase
    fake_torch.nn = fake_torch_nn

    fake_torch_f = types.ModuleType("torch.nn.functional")
    fake_torch_f.adaptive_avg_pool2d = lambda tensor, _size: tensor
    fake_torch_f.normalize = lambda tensor, dim=1: tensor
    fake_torch_nn.functional = fake_torch_f

    fake_models = types.ModuleType("torchvision.models")
    fake_models.EfficientNet_B0_Weights = types.SimpleNamespace(DEFAULT=object())
    fake_models.MobileNet_V3_Small_Weights = types.SimpleNamespace(DEFAULT=object())
    fake_models.MobileNet_V3_Large_Weights = types.SimpleNamespace(DEFAULT=object())
    fake_models.efficientnet_b0 = lambda weights=None: types.SimpleNamespace(features=_FakeModuleBase())
    fake_models.mobilenet_v3_small = lambda weights=None: types.SimpleNamespace(features=_FakeModuleBase())
    fake_models.mobilenet_v3_large = lambda weights=None: types.SimpleNamespace(features=_FakeModuleBase())

    fake_transforms = types.ModuleType("torchvision.transforms")
    fake_transforms.Resize = lambda *_args, **_kwargs: (lambda image: image)
    fake_transforms.ToTensor = lambda: (lambda image: _FakeTensor(np.zeros((3, 224, 224), dtype=np.float32)))
    fake_transforms.Normalize = lambda *_args, **_kwargs: (lambda tensor: tensor)

    class _FakeCompose:
        def __init__(self, items) -> None:
            self._items = list(items)

        def __call__(self, image):
            result = image
            for item in self._items:
                result = item(result)
            return result

    fake_transforms.Compose = _FakeCompose

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = fake_models
    fake_torchvision.transforms = fake_transforms

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.SessionOptions = _FakeSessionOptions
    fake_ort.InferenceSession = _FakeInferenceSession
    fake_ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]

    fake_onnx = types.ModuleType("onnx")

    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.open = lambda *_args, **_kwargs: _FakePilImageContext()
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_pil_image

    return {
        "algorithms.embedding_core": fake_embedding_core,
        "torch": fake_torch,
        "torch.nn": fake_torch_nn,
        "torch.nn.functional": fake_torch_f,
        "torchvision": fake_torchvision,
        "torchvision.models": fake_models,
        "torchvision.transforms": fake_transforms,
        "onnxruntime": fake_ort,
        "onnx": fake_onnx,
        "PIL": fake_pil,
        "PIL.Image": fake_pil_image,
    }


class EmbeddingOrtBackboneTest(unittest.TestCase):
    def test_all_learning_backbones_prefer_ort_runner_when_available(self) -> None:
        module_name = "algorithms.embedding"
        fake_modules = _fake_dependency_modules()
        sys.modules.pop(module_name, None)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(sys.modules, fake_modules, clear=False):
            embedding = importlib.import_module(module_name)
            try:
                with mock.patch.object(
                    embedding,
                    "writable_embedding_test_root",
                    return_value=Path(tmpdir),
                ):
                    for backbone_name, out_ch in {
                        "efficientnet_b0": 1280,
                        "mobilenet_v3_small": 576,
                        "mobilenet_v3_large": 960,
                    }.items():
                        runner, actual_out_ch = embedding.load_backbone(backbone_name, device="cpu")
                        self.assertEqual(getattr(runner, "runtime_backend", ""), "ort")
                        self.assertEqual(getattr(runner, "model_format", ""), "ort")
                        self.assertTrue(str(getattr(runner, "model_path", "")).endswith(".ort"))
                        self.assertEqual(actual_out_ch, out_ch)
            finally:
                sys.modules.pop(module_name, None)

    def test_cpu_dynamic_batch_uses_configured_chunks_and_reports_details(self) -> None:
        module_name = "algorithms.embedding"
        fake_modules = _fake_dependency_modules()
        sys.modules.pop(module_name, None)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(sys.modules, fake_modules, clear=False):
            embedding = importlib.import_module(module_name)
            try:
                with mock.patch.object(
                    embedding,
                    "writable_embedding_test_root",
                    return_value=Path(tmpdir),
                ):
                    runner, _out_ch = embedding.load_backbone("efficientnet_b0", device="cpu")
                    timing: dict[str, object] = {}
                    result = runner.run(
                        np.zeros((5, 3, 224, 224), dtype=np.float32),
                        timing_breakdown=timing,
                    )

                self.assertEqual(result.shape, (5, 4))
                self.assertEqual(runner.session.batch_sizes, [2, 2, 1])
                self.assertEqual(timing["backbone_backend"], "ort")
                self.assertEqual(timing["backbone_provider"], "CPUExecutionProvider")
                self.assertEqual(timing["backbone_batch_size"], 5)
                self.assertEqual(timing["backbone_chunk_size"], 2)
                self.assertEqual(timing["backbone_chunk_count"], 3)
            finally:
                sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
