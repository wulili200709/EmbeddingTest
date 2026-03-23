from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from .labelme import (
    clamp_roi_xywh,
    labelme_json_of_image,
    read_roi_from_labelme,
    read_shape_from_labelme,
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(name: str, device: Optional[str] = None):
    device = device or get_device()
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

    feat.eval().to(device)
    return feat, out_ch


TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for ext in exts:
        files += glob.glob(os.path.join(folder, ext))
    return sorted(files)


@torch.no_grad()
def embed_one(
    img_path: str,
    feat_net,
    label_name: str = "roi",
    device: Optional[str] = None,
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    device = device or get_device()
    with Image.open(img_path) as image_raw:
        image = image_raw.convert("RGB")
    width, height = image.size

    roi_img: Optional[Image.Image] = None
    if roi_xywh is None:
        jpath = labelme_json_of_image(img_path)
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"Missing labelme json: {jpath}")
        shape = read_shape_from_labelme(jpath, label_name=label_name)
        if shape and str(shape.get("shape_type", "rectangle")) == "polygon":
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if points.shape[0] >= 3 and cv2 is not None:
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                x = int(round(float(x_min)))
                y = int(round(float(y_min)))
                w = int(round(float(x_max - x_min)))
                h = int(round(float(y_max - y_min)))
                x, y, w, h = clamp_roi_xywh(x, y, w, h, W=width, H=height)
                image_np = np.array(image)
                crop = image_np[y : y + h, x : x + w].copy()
                rel_points = np.round(points - np.array([[x, y]], dtype=np.float32)).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [rel_points], 255)
                crop[mask == 0] = 0
                roi_img = Image.fromarray(crop)
            else:
                x, y, w, h = read_roi_from_labelme(jpath, label_name=label_name)
        else:
            x, y, w, h = read_roi_from_labelme(jpath, label_name=label_name)
    else:
        x, y, w, h = roi_xywh

    if roi_img is None:
        x, y, w, h = clamp_roi_xywh(x, y, w, h, W=width, H=height)
        roi_img = image.crop((x, y, x + w, y + h))

    tensor = TF(roi_img).unsqueeze(0).to(device)
    feat = feat_net(tensor)
    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
    feat = F.normalize(feat, dim=1)
    return feat[0].detach().cpu().numpy()


def embed_many(
    img_path: str,
    feat_net,
    label_names: Sequence[str],
    device: Optional[str] = None,
) -> np.ndarray:
    labels = [str(name) for name in label_names if str(name).strip()]
    if not labels:
        raise ValueError("label_names cannot be empty")
    parts = [embed_one(img_path, feat_net, label_name=label, device=device) for label in labels]
    if len(parts) == 1:
        return parts[0]
    vector = np.concatenate(parts, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


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
    "embed_many",
    "embed_one",
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
