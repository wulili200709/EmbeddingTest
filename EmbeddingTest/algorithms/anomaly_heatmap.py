from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .anomaly import AnomalyModel, predict_one_with_anomaly_model
from .image_io import imread


def _get_device() -> str:
    from .embedding import get_device

    return get_device()


def _load_backbone(name: str, *, device: Optional[str] = None):
    from .embedding import load_backbone

    return load_backbone(name, device=device)


def _extract_roi_feature_map(*args, **kwargs):
    from .embedding import extract_roi_feature_map

    return extract_roi_feature_map(*args, **kwargs)


@dataclass
class AnomalyHeatmapResult:
    image_path: str
    roi_label: str
    roi_xywh: Tuple[int, int, int, int]
    pred: str
    score: float
    threshold: float
    diff: float
    topk: int
    ok_image_count: int
    ok_patch_count: int
    patch_max: float
    patch_mean: float
    coarse_patch_scores: np.ndarray
    heatmap_scores: np.ndarray
    heatmap_display: np.ndarray
    full_bgr: np.ndarray
    roi_bgr: np.ndarray
    overlay_bgr: np.ndarray
    roi_overlay_bgr: np.ndarray
    roi_heatmap_bgr: np.ndarray


def pool_feature_map_embedding(feature_map: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature_map, dtype=np.float32)
    if feature.ndim == 4:
        if feature.shape[0] != 1:
            raise ValueError(f"expected single-batch feature map, got {feature.shape!r}")
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"expected [C,H,W] feature map, got {feature.shape!r}")
    vector = feature.mean(axis=(1, 2)).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def flatten_patch_embeddings(feature_map: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    feature = np.asarray(feature_map, dtype=np.float32)
    if feature.ndim == 4:
        if feature.shape[0] != 1:
            raise ValueError(f"expected single-batch feature map, got {feature.shape!r}")
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"expected [C,H,W] feature map, got {feature.shape!r}")
    channels, height, width = feature.shape
    patches = feature.reshape(channels, height * width).T.astype(np.float32)
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches = patches / np.maximum(norms, 1e-12)
    return patches, (height, width)


def score_patch_embeddings_against_bank(
    patch_embeddings: np.ndarray,
    patch_bank: np.ndarray,
    *,
    topk: int,
) -> np.ndarray:
    patches = np.asarray(patch_embeddings, dtype=np.float32)
    bank = np.asarray(patch_bank, dtype=np.float32)
    if patches.ndim != 2:
        raise ValueError(f"expected patch embeddings [N,C], got {patches.shape!r}")
    if bank.ndim != 2:
        raise ValueError(f"expected patch bank [M,C], got {bank.shape!r}")
    if patches.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if bank.shape[0] <= 0:
        raise RuntimeError("patch bank is empty")
    if bank.shape[1] != patches.shape[1]:
        raise ValueError(f"patch bank dim mismatch: {bank.shape[1]} != {patches.shape[1]}")

    patch_norms = np.linalg.norm(patches, axis=1, keepdims=True)
    bank_norms = np.linalg.norm(bank, axis=1, keepdims=True)
    patches = patches / np.maximum(patch_norms, 1e-12)
    bank = bank / np.maximum(bank_norms, 1e-12)

    normalized_topk = min(max(1, int(topk)), int(bank.shape[0]))
    similarities = patches @ bank.T
    if normalized_topk >= int(bank.shape[0]):
        topk_similarity = similarities.mean(axis=1)
    else:
        topk_values = np.partition(similarities, int(bank.shape[0]) - normalized_topk, axis=1)[:, -normalized_topk:]
        topk_similarity = topk_values.mean(axis=1)
    topk_similarity = np.clip(topk_similarity, -1.0, 1.0)
    return (1.0 - topk_similarity).astype(np.float32)


def build_ok_patch_bank(
    ok_files: Sequence[str],
    feat_net,
    *,
    label_name: str,
    device: Optional[str] = None,
) -> np.ndarray:
    patch_parts = []
    normalized_device = str(device or _get_device()).strip() or "cpu"
    for path in ok_files:
        feature_map, _roi_rgb, _roi_xywh = _extract_roi_feature_map(
            path,
            feat_net,
            label_name=label_name,
            device=normalized_device,
        )
        patch_vectors, _grid_shape = flatten_patch_embeddings(feature_map)
        if patch_vectors.size:
            patch_parts.append(patch_vectors)
    if not patch_parts:
        raise RuntimeError("No OK patch features could be extracted for heatmap")
    return np.concatenate(patch_parts, axis=0).astype(np.float32)


def normalize_heatmap_for_display(heatmap_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(heatmap_scores, dtype=np.float32)
    if scores.size == 0:
        return np.zeros_like(scores, dtype=np.float32)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores, dtype=np.float32)
    low = float(finite.min())
    high = float(finite.max())
    if high - low <= 1e-6:
        return np.zeros_like(scores, dtype=np.float32)
    return np.clip((scores - low) / (high - low), 0.0, 1.0).astype(np.float32)


def render_heatmap_bgr(display_heatmap: np.ndarray) -> np.ndarray:
    normalized = np.asarray(display_heatmap, dtype=np.float32)
    heat_u8 = np.clip(np.round(normalized * 255.0), 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)


def overlay_heatmap_on_bgr(
    image_bgr: np.ndarray,
    display_heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    base = np.ascontiguousarray(np.asarray(image_bgr))
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError(f"expected BGR image [H,W,3], got {base.shape!r}")
    overlay = render_heatmap_bgr(display_heatmap)
    if overlay.shape[:2] != base.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    blend_alpha = min(max(float(alpha), 0.0), 1.0)
    return cv2.addWeighted(base[:, :, :3], 1.0 - blend_alpha, overlay[:, :, :3], blend_alpha, 0.0)


def generate_anomaly_heatmap(
    image_path: str,
    *,
    ok_files: Sequence[str],
    model: AnomalyModel,
    label_name: str,
    feat_net=None,
    patch_bank: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> AnomalyHeatmapResult:
    if not model.is_ready():
        raise RuntimeError("Anomaly model is not ready")
    normalized_device = str(device or getattr(model, "device", "") or _get_device()).strip() or "cpu"
    feature_net = feat_net
    if feature_net is None:
        feature_net, _out_ch = _load_backbone(model.backbone, device=normalized_device)

    full_bgr = imread(image_path, cv2.IMREAD_COLOR)
    if full_bgr is None:
        raise FileNotFoundError(image_path)

    effective_patch_bank = np.asarray(patch_bank, dtype=np.float32) if patch_bank is not None else None
    if effective_patch_bank is None or effective_patch_bank.size == 0:
        effective_patch_bank = build_ok_patch_bank(
            ok_files,
            feature_net,
            label_name=label_name,
            device=normalized_device,
        )

    feature_map, roi_rgb, roi_xywh = _extract_roi_feature_map(
        image_path,
        feature_net,
        label_name=label_name,
        device=normalized_device,
    )
    query_embedding = pool_feature_map_embedding(feature_map)
    pred, diff, score = predict_one_with_anomaly_model(query_embedding, model)

    patch_vectors, grid_shape = flatten_patch_embeddings(feature_map)
    coarse_scores = score_patch_embeddings_against_bank(
        patch_vectors,
        effective_patch_bank,
        topk=int(model.topk),
    ).reshape(grid_shape).astype(np.float32)

    x, y, w, h = roi_xywh
    heatmap_scores = cv2.resize(coarse_scores, (int(w), int(h)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    heatmap_display = normalize_heatmap_for_display(heatmap_scores)
    roi_bgr = np.ascontiguousarray(roi_rgb[:, :, ::-1])
    roi_overlay_bgr = overlay_heatmap_on_bgr(roi_bgr, heatmap_display, alpha=0.42)
    roi_heatmap_bgr = render_heatmap_bgr(heatmap_display)

    overlay_bgr = full_bgr.copy()
    overlay_bgr[y : y + h, x : x + w] = roi_overlay_bgr
    cv2.rectangle(overlay_bgr, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    return AnomalyHeatmapResult(
        image_path=image_path,
        roi_label=str(label_name or "roi"),
        roi_xywh=(int(x), int(y), int(w), int(h)),
        pred=str(pred),
        score=float(score),
        threshold=float(model.threshold),
        diff=float(diff),
        topk=int(model.topk),
        ok_image_count=int(len(ok_files)),
        ok_patch_count=int(effective_patch_bank.shape[0]),
        patch_max=float(np.max(heatmap_scores)) if heatmap_scores.size else 0.0,
        patch_mean=float(np.mean(heatmap_scores)) if heatmap_scores.size else 0.0,
        coarse_patch_scores=coarse_scores,
        heatmap_scores=heatmap_scores,
        heatmap_display=heatmap_display,
        full_bgr=full_bgr,
        roi_bgr=roi_bgr,
        overlay_bgr=overlay_bgr,
        roi_overlay_bgr=roi_overlay_bgr,
        roi_heatmap_bgr=roi_heatmap_bgr,
    )


__all__ = [
    "AnomalyHeatmapResult",
    "build_ok_patch_bank",
    "flatten_patch_embeddings",
    "generate_anomaly_heatmap",
    "normalize_heatmap_for_display",
    "overlay_heatmap_on_bgr",
    "pool_feature_map_embedding",
    "render_heatmap_bgr",
    "score_patch_embeddings_against_bank",
]
