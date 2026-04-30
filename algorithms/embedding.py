from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnx  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    onnx = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

from app_paths import writable_embedding_test_root
from .labelme import (
    clamp_roi_xywh,
    labelme_json_of_image,
)


_BACKBONE_OUTPUT_CHANNELS = {
    "efficientnet_b0": 1280,
    "mobilenet_v3_small": 576,
    "mobilenet_v3_large": 960,
}
_ORT_EXPORT_VERSION = "v1"
_ORT_EXPORT_LOCK = RLock()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class _NormalizedEmbeddingBackbone(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(batch)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        feat = F.normalize(feat, dim=1)
        return feat


@dataclass(frozen=True)
class OrtBackboneRunner:
    session: Any
    input_name: str
    output_name: str
    providers: Tuple[str, ...]
    device: str
    model_path: str
    model_format: str = "ort"
    runtime_backend: str = "ort"

    def run(self, batch: object) -> np.ndarray:
        if isinstance(batch, torch.Tensor):
            batch_np = batch.detach().cpu().numpy()
        else:
            batch_np = np.asarray(batch, dtype=np.float32)
        batch_np = np.ascontiguousarray(batch_np.astype(np.float32, copy=False))
        outputs = self.session.run([self.output_name], {self.input_name: batch_np})
        return np.asarray(outputs[0], dtype=np.float32)


def _build_torch_backbone_model(name: str) -> tuple[torch.nn.Module, int]:
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat = model.features
        out_ch = 1280
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feat = model.features
        out_ch = 576
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        feat = model.features
        out_ch = 960
    else:
        raise ValueError(f"Unknown backbone: {name}")

    wrapper = _NormalizedEmbeddingBackbone(feat)
    wrapper.eval()
    setattr(wrapper, "runtime_backend", "torch")
    return wrapper, out_ch


def _normalized_device(device: Optional[str] = None) -> str:
    normalized = str(device or get_device()).strip()
    return normalized or "cpu"


def _device_kind(device: Optional[str] = None) -> str:
    normalized = _normalized_device(device).lower()
    return "cuda" if normalized.startswith("cuda") else "cpu"


def _normalized_backbone_backend(preferred_backend: Optional[str] = None) -> str:
    normalized = str(preferred_backend or "auto").strip().lower() or "auto"
    if normalized not in {"auto", "ort", "torch"}:
        raise ValueError(f"unsupported backbone backend: {preferred_backend}")
    return normalized


def _ort_cache_root() -> Path:
    return writable_embedding_test_root(__file__) / ".cache" / "ort_backbones"


def _backbone_onnx_path(name: str) -> Path:
    return _ort_cache_root() / f"{name}_{_ORT_EXPORT_VERSION}.onnx"


def _backbone_ort_path(name: str, *, device: str) -> Path:
    return _ort_cache_root() / f"{name}_{_device_kind(device)}_{_ORT_EXPORT_VERSION}.ort"


def _ort_providers_for_device(device: Optional[str] = None) -> tuple[str, ...]:
    if ort is None:
        return ()
    try:
        available = set(ort.get_available_providers())
    except Exception:
        return ()
    if _device_kind(device) == "cuda":
        if "CUDAExecutionProvider" in available:
            return ("CUDAExecutionProvider", "CPUExecutionProvider")
        return ()
    if "CPUExecutionProvider" in available:
        return ("CPUExecutionProvider",)
    return ()


def _export_backbone_onnx(name: str, onnx_path: Path) -> None:
    if onnx is None:
        raise RuntimeError("onnx is required to export ORT backbone models")
    model, _ = _build_torch_backbone_model(name)
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    os.makedirs(onnx_path.parent, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        input_names=["input"],
        output_names=["embedding"],
        dynamo=False,
        report=False,
        verify=False,
        profile=False,
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=17,
    )


def _ensure_ort_model(name: str, *, device: str) -> Path:
    ort_path = _backbone_ort_path(name, device=device)
    if ort_path.exists():
        return ort_path

    onnx_path = _backbone_onnx_path(name)
    with _ORT_EXPORT_LOCK:
        if ort_path.exists():
            return ort_path
        if not onnx_path.exists():
            _export_backbone_onnx(name, onnx_path)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        can_save_ort = False
        try:
            session_options.optimized_model_filepath = str(ort_path)
            session_options.add_session_config_entry("session.save_model_format", "ORT")
            can_save_ort = True
        except Exception:
            can_save_ort = False
        if not can_save_ort:
            return onnx_path
        ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=list(_ort_providers_for_device(device)),
        )
    if ort_path.exists():
        return ort_path
    return onnx_path


def _load_ort_backbone(name: str, *, device: str) -> Optional[OrtBackboneRunner]:
    if ort is None:
        return None
    providers = _ort_providers_for_device(device)
    if not providers:
        return None
    try:
        model_path = _ensure_ort_model(name, device=device)
        try:
            session = ort.InferenceSession(str(model_path), providers=list(providers))
        except Exception:
            if str(model_path).lower().endswith(".ort"):
                try:
                    os.remove(model_path)
                except OSError:
                    pass
                model_path = _ensure_ort_model(name, device=device)
                session = ort.InferenceSession(str(model_path), providers=list(providers))
            else:
                raise
    except Exception:
        return None
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        return None
    return OrtBackboneRunner(
        session=session,
        input_name=str(inputs[0].name),
        output_name=str(outputs[0].name),
        providers=tuple(str(provider) for provider in providers),
        device=device,
        model_path=str(model_path),
        model_format=model_path.suffix.lstrip(".") or "ort",
    )


def describe_backbone_runner(feat_net: object) -> dict[str, object]:
    backend = str(getattr(feat_net, "runtime_backend", "") or "torch").strip().lower() or "torch"
    model_format = str(getattr(feat_net, "model_format", "") or "").strip().lower()
    if not model_format and backend == "torch":
        model_format = "torch"
    model_path = str(getattr(feat_net, "model_path", "") or "").strip()
    providers = tuple(
        str(provider).strip()
        for provider in tuple(getattr(feat_net, "providers", ()) or ())
        if str(provider).strip()
    )
    return {
        "backend": backend,
        "model_format": model_format,
        "model_path": model_path,
        "providers": providers,
    }


def load_backbone(
    name: str,
    device: Optional[str] = None,
    *,
    preferred_backend: Optional[str] = None,
):
    normalized_name = str(name or "").strip()
    normalized_device = _normalized_device(device)
    backend_choice = _normalized_backbone_backend(preferred_backend)
    if backend_choice in {"auto", "ort"}:
        ort_runner = _load_ort_backbone(normalized_name, device=normalized_device)
        if ort_runner is not None:
            return ort_runner, _BACKBONE_OUTPUT_CHANNELS[normalized_name]
    model, out_ch = _build_torch_backbone_model(normalized_name)
    model.eval().to(normalized_device)
    return model, out_ch


TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _run_embedding_batch(feat_net, batch: torch.Tensor, *, device: Optional[str] = None) -> np.ndarray:
    if getattr(feat_net, "runtime_backend", "") == "ort":
        return feat_net.run(batch)
    batch = batch.to(_normalized_device(device))
    feat = feat_net(batch)
    if isinstance(feat, torch.Tensor):
        return feat.detach().cpu().numpy()
    return np.asarray(feat, dtype=np.float32)


def load_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for ext in exts:
        files += glob.glob(os.path.join(folder, ext))
    return sorted(files)


def _load_image_rgb(img_path: str) -> Image.Image:
    with Image.open(img_path) as image_raw:
        return image_raw.convert("RGB")


@dataclass
class _ImageRoiContext:
    img_path: str
    image: Image.Image
    width: int
    height: int
    image_np: Optional[np.ndarray]
    shape_by_label: dict[str, dict]


def _build_image_roi_context(img_path: str, *, require_shapes: bool) -> _ImageRoiContext:
    image = _load_image_rgb(img_path)
    width, height = image.size
    shape_by_label: dict[str, dict] = {}
    if require_shapes:
        jpath = labelme_json_of_image(img_path)
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"Missing labelme json: {jpath}")
        with open(jpath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        shape_by_label = {
            str(shape.get("label", "")): shape
            for shape in data.get("shapes", [])
            if isinstance(shape, dict) and str(shape.get("label", "")).strip()
        }
    return _ImageRoiContext(
        img_path=img_path,
        image=image,
        width=width,
        height=height,
        image_np=None,
        shape_by_label=shape_by_label,
    )


def _image_from_bgr_array(image_bgr: np.ndarray) -> Image.Image:
    image = np.asarray(image_bgr)
    if image.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    if image.ndim != 3:
        raise ValueError(f"unsupported image shape: {image.shape!r}")
    image = np.ascontiguousarray(image[:, :, :3])
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    return Image.fromarray(rgb).convert("RGB")


def _build_array_roi_context(
    image_bgr: np.ndarray,
    *,
    shape_by_label: dict[str, dict],
) -> _ImageRoiContext:
    image = _image_from_bgr_array(image_bgr)
    width, height = image.size
    normalized_shapes = {
        str(label or ""): dict(shape)
        for label, shape in dict(shape_by_label or {}).items()
        if str(label or "").strip()
    }
    return _ImageRoiContext(
        img_path="",
        image=image,
        width=width,
        height=height,
        image_np=None,
        shape_by_label=normalized_shapes,
    )


def _shape_bbox_xywh(shape: dict, *, width: int, height: int) -> Tuple[int, int, int, int]:
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.size == 0:
        raise RuntimeError("ROI points empty")
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x = int(round(float(x_min)))
    y = int(round(float(y_min)))
    w = int(round(float(x_max - x_min)))
    h = int(round(float(y_max - y_min)))
    x, y, w, h = clamp_roi_xywh(x, y, w, h, W=width, H=height)
    return x, y, w, h


def _resolve_roi_image(
    context: _ImageRoiContext,
    *,
    label_name: str = "roi",
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    roi_img: Optional[Image.Image] = None
    if roi_xywh is None:
        shape = context.shape_by_label.get(label_name)
        if shape and str(shape.get("shape_type", "rectangle")) == "polygon":
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if points.shape[0] >= 3 and cv2 is not None:
                x, y, w, h = _shape_bbox_xywh(shape, width=context.width, height=context.height)
                if context.image_np is None:
                    context.image_np = np.array(context.image)
                crop = context.image_np[y : y + h, x : x + w].copy()
                rel_points = np.round(points - np.array([[x, y]], dtype=np.float32)).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [rel_points], 255)
                crop[mask == 0] = 0
                roi_img = Image.fromarray(crop)
            else:
                x, y, w, h = _shape_bbox_xywh(shape, width=context.width, height=context.height)
        else:
            if shape is None:
                if context.img_path:
                    raise RuntimeError(f"Label '{label_name}' not found in {labelme_json_of_image(context.img_path)}")
                raise RuntimeError(f"Label '{label_name}' not found in runtime ROI shapes")
            x, y, w, h = _shape_bbox_xywh(shape, width=context.width, height=context.height)
    else:
        x, y, w, h = roi_xywh

    if roi_img is None:
        x, y, w, h = clamp_roi_xywh(x, y, w, h, W=context.width, H=context.height)
        roi_img = context.image.crop((x, y, x + w, y + h))
    return roi_img


@torch.no_grad()
def embed_batch(
    img_path: str,
    feat_net,
    label_names: Sequence[str],
    device: Optional[str] = None,
    roi_xywhs: Optional[Sequence[Optional[Tuple[int, int, int, int]]]] = None,
) -> np.ndarray:
    labels = [str(name or "roi").strip() or "roi" for name in label_names]
    if not labels:
        raise ValueError("label_names cannot be empty")
    if roi_xywhs is not None and len(roi_xywhs) != len(labels):
        raise ValueError("roi_xywhs must have the same length as label_names")

    require_shapes = roi_xywhs is None or any(roi_xywh is None for roi_xywh in roi_xywhs)
    context = _build_image_roi_context(img_path, require_shapes=require_shapes)
    tensors = []
    for index, label_name in enumerate(labels):
        roi_xywh = roi_xywhs[index] if roi_xywhs is not None else None
        roi_img = _resolve_roi_image(
            context,
            label_name=label_name,
            roi_xywh=roi_xywh,
        )
        tensors.append(TF(roi_img))

    batch = torch.stack(tensors, dim=0)
    return _run_embedding_batch(feat_net, batch, device=device)


@torch.no_grad()
def embed_one(
    img_path: str,
    feat_net,
    label_name: str = "roi",
    device: Optional[str] = None,
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    embeddings = embed_batch(
        img_path,
        feat_net,
        [label_name],
        device=device,
        roi_xywhs=[roi_xywh],
    )
    return embeddings[0]


def embed_many(
    img_path: str,
    feat_net,
    label_names: Sequence[str],
    device: Optional[str] = None,
) -> np.ndarray:
    labels = [str(name) for name in label_names if str(name).strip()]
    if not labels:
        raise ValueError("label_names cannot be empty")
    parts = [emb for emb in embed_batch(img_path, feat_net, labels, device=device)]
    if len(parts) == 1:
        return parts[0]
    vector = np.concatenate(parts, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


@torch.no_grad()
def embed_batch_from_array(
    image_bgr: np.ndarray,
    feat_net,
    label_names: Sequence[str],
    *,
    shape_by_label: dict[str, dict],
    device: Optional[str] = None,
    roi_xywhs: Optional[Sequence[Optional[Tuple[int, int, int, int]]]] = None,
) -> np.ndarray:
    labels = [str(name or "roi").strip() or "roi" for name in label_names]
    if not labels:
        raise ValueError("label_names cannot be empty")
    if roi_xywhs is not None and len(roi_xywhs) != len(labels):
        raise ValueError("roi_xywhs must have the same length as label_names")

    context = _build_array_roi_context(image_bgr, shape_by_label=shape_by_label)
    tensors = []
    for index, label_name in enumerate(labels):
        roi_xywh = roi_xywhs[index] if roi_xywhs is not None else None
        roi_img = _resolve_roi_image(
            context,
            label_name=label_name,
            roi_xywh=roi_xywh,
        )
        tensors.append(TF(roi_img))

    batch = torch.stack(tensors, dim=0)
    return _run_embedding_batch(feat_net, batch, device=device)


@torch.no_grad()
def embed_one_from_array(
    image_bgr: np.ndarray,
    feat_net,
    *,
    shape_by_label: dict[str, dict],
    label_name: str = "roi",
    device: Optional[str] = None,
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    embeddings = embed_batch_from_array(
        image_bgr,
        feat_net,
        [label_name],
        shape_by_label=shape_by_label,
        device=device,
        roi_xywhs=[roi_xywh],
    )
    return embeddings[0]


def score_topk(e: np.ndarray, bank: np.ndarray, k: int = 3) -> float:
    sims = bank @ e
    if sims.size == 0:
        return float("-inf")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k >= sims.size:
        return float(np.mean(sims))
    return float(np.mean(np.sort(sims)[-k:]))


@dataclass
class RegisterModel:
    backbone: str
    score_mode: str
    margin: float
    topk: int
    label_name: str = "roi"
    label_names: Optional[List[str]] = None
    device: str = "cpu"
    ok_proto: Optional[np.ndarray] = None
    ng_proto: Optional[np.ndarray] = None
    ok_bank: Optional[np.ndarray] = None
    ng_bank: Optional[np.ndarray] = None

    def is_ready(self) -> bool:
        return (
            self.ok_bank is not None
            and self.ng_bank is not None
            and self.ok_bank.size > 0
            and self.ng_bank.size > 0
        )

    def effective_label_names(self) -> List[str]:
        labels = [str(name) for name in (self.label_names or []) if str(name).strip()]
        if labels:
            return labels
        return [str(self.label_name or "roi")]


def save_register_model_npz(model: RegisterModel, npz_path: str) -> None:
    if not model.is_ready():
        raise RuntimeError("Model is not ready")
    ok_proto = model.ok_proto
    ng_proto = model.ng_proto
    ok_bank = model.ok_bank
    ng_bank = model.ng_bank
    assert ok_proto is not None and ng_proto is not None and ok_bank is not None and ng_bank is not None

    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez_compressed(
        npz_path,
        backbone=np.array([model.backbone]),
        score_mode=np.array([model.score_mode]),
        margin=np.array([float(model.margin)], dtype=np.float32),
        topk=np.array([int(model.topk)], dtype=np.int32),
        label_name=np.array([model.label_name]),
        label_names=np.array(model.effective_label_names()),
        device=np.array([model.device]),
        ok_proto=ok_proto.astype(np.float32),
        ng_proto=ng_proto.astype(np.float32),
        ok_bank=ok_bank.astype(np.float32),
        ng_bank=ng_bank.astype(np.float32),
    )


def load_register_model_npz(npz_path: str) -> RegisterModel:
    data = np.load(npz_path, allow_pickle=False)
    return RegisterModel(
        backbone=str(data["backbone"][0]),
        score_mode=str(data["score_mode"][0]),
        margin=float(data["margin"][0]),
        topk=int(data["topk"][0]),
        label_name=str(data["label_name"][0]),
        label_names=[str(value) for value in data["label_names"]]
        if "label_names" in data.files
        else [str(data["label_name"][0])],
        device=str(data["device"][0]),
        ok_proto=data["ok_proto"],
        ng_proto=data["ng_proto"],
        ok_bank=data["ok_bank"],
        ng_bank=data["ng_bank"],
    )


def train_register_model(
    ok_files: Sequence[str],
    ng_files: Sequence[str],
    backbone: str = "efficientnet_b0",
    score_mode: str = "proto",
    margin: float = 0.02,
    topk: int = 3,
    label_name: str = "roi",
    label_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> RegisterModel:
    if not ok_files or not ng_files:
        raise RuntimeError("Both OK and NG samples are required")
    device = device or get_device()
    feat_net, _ = load_backbone(backbone, device=device)
    labels = [str(name) for name in (label_names or [label_name]) if str(name).strip()]
    if not labels:
        labels = ["roi"]

    ok_emb = np.stack([embed_many(path, feat_net, labels, device=device) for path in ok_files])
    ng_emb = np.stack([embed_many(path, feat_net, labels, device=device) for path in ng_files])

    ok_proto = ok_emb.mean(axis=0, keepdims=True)
    ok_proto = ok_proto / np.linalg.norm(ok_proto, axis=1, keepdims=True)
    ng_proto = ng_emb.mean(axis=0, keepdims=True)
    ng_proto = ng_proto / np.linalg.norm(ng_proto, axis=1, keepdims=True)

    return RegisterModel(
        backbone=backbone,
        score_mode=score_mode,
        margin=float(margin),
        topk=int(topk),
        label_name=labels[0],
        label_names=labels,
        device=device,
        ok_proto=ok_proto,
        ng_proto=ng_proto,
        ok_bank=ok_emb,
        ng_bank=ng_emb,
    )


def predict_one_with_model(
    e: np.ndarray,
    model: RegisterModel,
) -> Tuple[str, float, float, float]:
    if not model.is_ready():
        raise RuntimeError("Model is not ready")
    ok_proto = model.ok_proto
    ng_proto = model.ng_proto
    ok_bank = model.ok_bank
    ng_bank = model.ng_bank
    assert ok_proto is not None and ng_proto is not None and ok_bank is not None and ng_bank is not None

    if model.score_mode == "proto":
        sim_ok = float(e @ ok_proto[0])
        sim_ng = float(e @ ng_proto[0])
    elif model.score_mode == "topk":
        sim_ok = score_topk(e, ok_bank, k=min(model.topk, len(ok_bank)))
        sim_ng = score_topk(e, ng_bank, k=min(model.topk, len(ng_bank)))
    else:
        raise ValueError(f"Unknown score mode: {model.score_mode}")

    diff = sim_ok - sim_ng
    pred = "OK" if diff >= model.margin else "NG"
    return pred, float(diff), sim_ok, sim_ng


def predict_one(
    e: np.ndarray,
    ok_proto: np.ndarray,
    ng_proto: np.ndarray,
    ok_bank: np.ndarray,
    ng_bank: np.ndarray,
    mode: str,
    margin: float,
    topk: int,
) -> Tuple[str, float, float, float]:
    if mode == "proto":
        sim_ok = float(e @ ok_proto[0])
        sim_ng = float(e @ ng_proto[0])
    elif mode == "topk":
        sim_ok = score_topk(e, ok_bank, k=min(topk, len(ok_bank)))
        sim_ng = score_topk(e, ng_bank, k=min(topk, len(ng_bank)))
    else:
        raise ValueError(f"Unknown score mode: {mode}")

    diff = sim_ok - sim_ng
    pred = "OK" if diff >= margin else "NG"
    return pred, float(diff), sim_ok, sim_ng


__all__ = [
    "RegisterModel",
    "embed_batch",
    "embed_batch_from_array",
    "embed_many",
    "embed_one",
    "embed_one_from_array",
    "describe_backbone_runner",
    "get_device",
    "load_backbone",
    "load_images",
    "load_register_model_npz",
    "predict_one",
    "predict_one_with_model",
    "save_register_model_npz",
    "score_topk",
    "train_register_model",
]
