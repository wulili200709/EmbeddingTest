from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np



PATCHCORE_LITE_ALGORITHM = "patchcore_lite"
ANOMALY_ALGORITHMS = [PATCHCORE_LITE_ALGORITHM]
_ANOMALY_BACKBONES = {
    PATCHCORE_LITE_ALGORITHM: "b0",
}


def anomaly_backbone_for_algorithm(algorithm: str) -> str:
    normalized = str(algorithm or PATCHCORE_LITE_ALGORITHM).strip() or PATCHCORE_LITE_ALGORITHM
    return _ANOMALY_BACKBONES.get(normalized, "b0")


def _normalize_labels(label_name: str, label_names: Optional[Sequence[str]]) -> List[str]:
    labels = [str(name) for name in (label_names or [label_name]) if str(name).strip()]
    if labels:
        return labels
    return ["roi"]


def anomaly_score_topk(e: np.ndarray, bank: np.ndarray, k: int = 3) -> float:
    from .embedding import score_topk

    ref_bank = np.asarray(bank, dtype=np.float32)
    if ref_bank.ndim != 2 or ref_bank.shape[0] <= 0:
        raise RuntimeError("Anomaly memory bank is empty")
    similarity = score_topk(np.asarray(e, dtype=np.float32), ref_bank, k=min(max(1, int(k)), len(ref_bank)))
    similarity = max(-1.0, min(1.0, float(similarity)))
    return float(1.0 - similarity)


def _score_embeddings_against_bank(embeddings: np.ndarray, bank: np.ndarray, topk: int) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    scores = [anomaly_score_topk(embedding, bank, k=topk) for embedding in embeddings]
    return np.asarray(scores, dtype=np.float32)


def _leave_one_out_ok_scores(ok_bank: np.ndarray, topk: int) -> np.ndarray:
    if ok_bank.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(ok_bank) == 1:
        return np.asarray([anomaly_score_topk(ok_bank[0], ok_bank, k=1)], dtype=np.float32)
    scores = []
    for index in range(len(ok_bank)):
        ref_bank = np.concatenate([ok_bank[:index], ok_bank[index + 1 :]], axis=0)
        scores.append(anomaly_score_topk(ok_bank[index], ref_bank, k=min(max(1, int(topk)), len(ref_bank))))
    return np.asarray(scores, dtype=np.float32)


def _threshold_candidates(ok_scores: np.ndarray, ng_scores: np.ndarray) -> List[float]:
    combined = [float(value) for value in np.concatenate([ok_scores, ng_scores]) if np.isfinite(value)]
    if not combined:
        return [0.02]
    values = sorted(set(combined))
    candidates = [max(0.0, values[0] - 1e-4)]
    candidates.extend(values)
    candidates.extend((left + right) * 0.5 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-4)
    return sorted(set(float(value) for value in candidates))


def _select_threshold(ok_scores: np.ndarray, ng_scores: np.ndarray) -> float:
    if ok_scores.size == 0:
        raise RuntimeError("OK anomaly scores are empty")
    if ng_scores.size <= 0:
        ok_mean = float(np.mean(ok_scores))
        ok_std = float(np.std(ok_scores))
        ok_max = float(np.max(ok_scores))
        fallback = max(ok_max + 0.005, ok_mean + 3.0 * ok_std)
        return float(max(0.0, min(2.0, fallback)))

    best_threshold = float(np.max(ok_scores))
    best_accuracy = -1.0
    total_count = float(len(ok_scores) + len(ng_scores))
    for threshold in _threshold_candidates(ok_scores, ng_scores):
        ok_correct = float(np.sum(ok_scores <= threshold))
        ng_correct = float(np.sum(ng_scores > threshold))
        accuracy = (ok_correct + ng_correct) / total_count
        if accuracy > best_accuracy + 1e-9:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return float(max(0.0, min(2.0, best_threshold)))


@dataclass
class AnomalyModel:
    algorithm: str
    backbone: str
    threshold: float
    topk: int
    label_name: str = "roi"
    label_names: Optional[List[str]] = None
    device: str = "cpu"
    ok_bank: Optional[np.ndarray] = None
    ok_scores: Optional[np.ndarray] = None
    ng_scores: Optional[np.ndarray] = None

    def is_ready(self) -> bool:
        return self.ok_bank is not None and self.ok_bank.size > 0

    def effective_label_names(self) -> List[str]:
        labels = [str(name) for name in (self.label_names or []) if str(name).strip()]
        if labels:
            return labels
        return [str(self.label_name or "roi")]


def train_patchcore_lite_model(
    ok_files: Sequence[str],
    ng_files: Sequence[str],
    *,
    algorithm: str = PATCHCORE_LITE_ALGORITHM,
    topk: int = 3,
    label_name: str = "roi",
    label_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> AnomalyModel:
    if not ok_files:
        raise RuntimeError("At least one OK sample is required")
    normalized_algorithm = str(algorithm or PATCHCORE_LITE_ALGORITHM).strip() or PATCHCORE_LITE_ALGORITHM
    normalized_topk = max(1, int(topk))
    from .embedding import embed_many, get_device, load_backbone

    normalized_device = str(device or get_device()).strip() or "cpu"
    backbone = anomaly_backbone_for_algorithm(normalized_algorithm)
    labels = _normalize_labels(label_name, label_names)
    feat_net, _ = load_backbone(backbone, device=normalized_device)

    ok_embeddings = np.stack([
        embed_many(path, feat_net, labels, device=normalized_device)
        for path in ok_files
    ]).astype(np.float32)
    if ng_files:
        ng_embeddings = np.stack([
            embed_many(path, feat_net, labels, device=normalized_device)
            for path in ng_files
        ]).astype(np.float32)
    else:
        ng_embeddings = np.zeros((0, ok_embeddings.shape[1]), dtype=np.float32)

    ok_scores = _leave_one_out_ok_scores(ok_embeddings, normalized_topk)
    ng_scores = _score_embeddings_against_bank(ng_embeddings, ok_embeddings, normalized_topk)
    threshold = _select_threshold(ok_scores, ng_scores)
    return AnomalyModel(
        algorithm=normalized_algorithm,
        backbone=backbone,
        threshold=threshold,
        topk=normalized_topk,
        label_name=labels[0],
        label_names=labels,
        device=normalized_device,
        ok_bank=ok_embeddings,
        ok_scores=ok_scores,
        ng_scores=ng_scores,
    )


def train_patchcore_lite_model_from_samples(
    ok_samples: Sequence[tuple[str, str]],
    ng_samples: Optional[Sequence[tuple[str, str]]] = None,
    *,
    algorithm: str = PATCHCORE_LITE_ALGORITHM,
    topk: int = 3,
    label_name: str = "roi",
    label_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> AnomalyModel:
    normalized_ok_samples = [
        (str(path or "").strip(), str(label or "").strip() or str(label_name or "roi").strip() or "roi")
        for path, label in list(ok_samples or [])
        if str(path or "").strip()
    ]
    normalized_ng_samples = [
        (str(path or "").strip(), str(label or "").strip() or str(label_name or "roi").strip() or "roi")
        for path, label in list(ng_samples or [])
        if str(path or "").strip()
    ]
    if not normalized_ok_samples:
        raise RuntimeError("At least one OK sample is required")

    normalized_algorithm = str(algorithm or PATCHCORE_LITE_ALGORITHM).strip() or PATCHCORE_LITE_ALGORITHM
    normalized_topk = max(1, int(topk))
    from .embedding import embed_one, get_device, load_backbone

    normalized_device = str(device or get_device()).strip() or "cpu"
    backbone = anomaly_backbone_for_algorithm(normalized_algorithm)
    labels = _normalize_labels(label_name, label_names)
    feat_net, _ = load_backbone(backbone, device=normalized_device)

    ok_embeddings = np.stack([
        embed_one(path, feat_net, label_name=label, device=normalized_device)
        for path, label in normalized_ok_samples
    ]).astype(np.float32)
    if normalized_ng_samples:
        ng_embeddings = np.stack([
            embed_one(path, feat_net, label_name=label, device=normalized_device)
            for path, label in normalized_ng_samples
        ]).astype(np.float32)
    else:
        ng_embeddings = np.zeros((0, ok_embeddings.shape[1]), dtype=np.float32)

    ok_scores = _leave_one_out_ok_scores(ok_embeddings, normalized_topk)
    ng_scores = _score_embeddings_against_bank(ng_embeddings, ok_embeddings, normalized_topk)
    threshold = _select_threshold(ok_scores, ng_scores)
    return AnomalyModel(
        algorithm=normalized_algorithm,
        backbone=backbone,
        threshold=threshold,
        topk=normalized_topk,
        label_name=labels[0],
        label_names=labels,
        device=normalized_device,
        ok_bank=ok_embeddings,
        ok_scores=ok_scores,
        ng_scores=ng_scores,
    )


def predict_one_with_anomaly_model(
    e: np.ndarray,
    model: AnomalyModel,
) -> Tuple[str, float, float]:
    if not model.is_ready() or model.ok_bank is None:
        raise RuntimeError("Anomaly model is not ready")
    score = anomaly_score_topk(e, model.ok_bank, k=max(1, int(model.topk)))
    diff = float(model.threshold) - float(score)
    pred = "OK" if diff >= 0.0 else "NG"
    return pred, diff, float(score)


def save_anomaly_model_npz(model: AnomalyModel, npz_path: str) -> None:
    if not model.is_ready() or model.ok_bank is None:
        raise RuntimeError("Anomaly model is not ready")
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez_compressed(
        npz_path,
        algorithm=np.array([model.algorithm]),
        backbone=np.array([model.backbone]),
        threshold=np.array([float(model.threshold)], dtype=np.float32),
        topk=np.array([int(model.topk)], dtype=np.int32),
        label_name=np.array([model.label_name]),
        label_names=np.array(model.effective_label_names()),
        device=np.array([model.device]),
        ok_bank=model.ok_bank.astype(np.float32),
        ok_scores=(model.ok_scores if model.ok_scores is not None else np.zeros((0,), dtype=np.float32)).astype(np.float32),
        ng_scores=(model.ng_scores if model.ng_scores is not None else np.zeros((0,), dtype=np.float32)).astype(np.float32),
    )


def load_anomaly_model_npz(npz_path: str) -> AnomalyModel:
    data = np.load(npz_path, allow_pickle=False)
    return AnomalyModel(
        algorithm=str(data["algorithm"][0]) if "algorithm" in data.files else PATCHCORE_LITE_ALGORITHM,
        backbone=str(data["backbone"][0]),
        threshold=float(data["threshold"][0]),
        topk=int(data["topk"][0]),
        label_name=str(data["label_name"][0]),
        label_names=[str(value) for value in data["label_names"]] if "label_names" in data.files else [str(data["label_name"][0])],
        device=str(data["device"][0]),
        ok_bank=data["ok_bank"],
        ok_scores=data["ok_scores"] if "ok_scores" in data.files else np.zeros((0,), dtype=np.float32),
        ng_scores=data["ng_scores"] if "ng_scores" in data.files else np.zeros((0,), dtype=np.float32),
    )


__all__ = [
    "ANOMALY_ALGORITHMS",
    "AnomalyModel",
    "PATCHCORE_LITE_ALGORITHM",
    "anomaly_backbone_for_algorithm",
    "anomaly_score_topk",
    "load_anomaly_model_npz",
    "predict_one_with_anomaly_model",
    "save_anomaly_model_npz",
    "train_patchcore_lite_model",
    "train_patchcore_lite_model_from_samples",
]
