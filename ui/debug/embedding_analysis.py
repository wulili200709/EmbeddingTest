from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - fallback to PCA
    TSNE = None

if __package__ in (None, ""):
    root_str = str(Path(__file__).resolve().parents[2])
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import algorithms.proxy as qr_core
from algorithms.registry import (
    algorithm_display_name,
    learning_backbone_storage_code,
    storage_code_backbone,
)
from domain import load_inspection_items
from infrastructure.product_params import load_product_params
from ui.i18n import tr


@dataclass
class EmbeddingAnalysisRow:
    gt_label: str
    file_path: str
    file_name: str
    pred_label: str
    diff: float
    sim_ok: float
    sim_ng: float


@dataclass
class EmbeddingAnalysisResult:
    product_name: str
    backbone: str
    model_path: str
    session_file: str
    projection_method: str
    feature_dim: int
    point_coords: np.ndarray
    point_labels: List[str]
    point_names: List[str]
    rows: List[EmbeddingAnalysisRow]
    metrics: Dict[str, float]
    notes: List[str]
    model_key: str = ""
    tool_name: str = ""
    label_names: List[str] = field(default_factory=list)
    score_mode: str = ""
    margin: float = 0.0
    topk: int = 0


@dataclass(frozen=True)
class EmbeddingModelEntry:
    model_key: str
    backbone: str
    model_path: str
    display_name: str
    tool_name: str


def list_product_names(session_root: str) -> List[str]:
    if not os.path.isdir(session_root):
        return []
    names = [
        name
        for name in sorted(os.listdir(session_root))
        if os.path.isdir(os.path.join(session_root, name))
    ]
    return names


def _parse_register_model_filename(name: str) -> Optional[Tuple[str, str]]:
    if not str(name or "").endswith(".npz"):
        return None
    stem = str(name)[:-4]
    if stem.startswith("register_model_"):
        backbone = storage_code_backbone(stem[len("register_model_") :])
        return ("", backbone) if backbone else None
    marker = "_register_model_"
    if marker not in stem:
        return None
    model_key, backbone = stem.rsplit(marker, 1)
    backbone = storage_code_backbone(backbone)
    if not backbone:
        return None
    return model_key, backbone


def register_model_path(product_dir: str, backbone: str, *, model_key: str = "") -> str:
    normalized_key = str(model_key or "").strip()
    storage_code = learning_backbone_storage_code(backbone)
    if normalized_key:
        return os.path.join(product_dir, f"{normalized_key}_register_model_{storage_code}.npz")
    return os.path.join(product_dir, f"register_model_{storage_code}.npz")


def list_available_embedding_models(product_dir: str) -> List[EmbeddingModelEntry]:
    entries: List[EmbeddingModelEntry] = []
    if not os.path.isdir(product_dir):
        return entries

    inspection_items_path = os.path.join(product_dir, "inspection_items.json")
    items_by_key = {
        item.model_key: item
        for item in load_inspection_items(inspection_items_path)
    }
    shared_model_name = tr("debug.embedding.shared_model")

    for name in sorted(os.listdir(product_dir)):
        parsed = _parse_register_model_filename(name)
        if parsed is None:
            continue
        model_key, backbone = parsed
        item = items_by_key.get(model_key) if model_key else None
        tool_name = shared_model_name
        if item is not None:
            base_name = str(item.display_name or item.roi_label or item.item_id or model_key).strip()
            roi_hint = f"{item.camera_id}/{item.roi_label}".strip("/")
            tool_name = f"{base_name} ({roi_hint})" if roi_hint and base_name != roi_hint else (base_name or roi_hint or model_key)
        elif model_key:
            tool_name = model_key
        learning_name = algorithm_display_name(backbone) or backbone
        entries.append(
            EmbeddingModelEntry(
                model_key=model_key,
                backbone=backbone,
                model_path=os.path.join(product_dir, name),
                display_name=f"{tool_name} / {learning_name}",
                tool_name=tool_name,
            )
        )
    return entries


def list_available_backbones(product_dir: str) -> List[str]:
    return sorted({entry.backbone for entry in list_available_embedding_models(product_dir)})


def _find_embedding_model_entry(
    product_dir: str,
    *,
    backbone: str = "",
    model_key: str = "",
) -> Optional[EmbeddingModelEntry]:
    normalized_backbone = str(backbone or "").strip()
    normalized_model_key = str(model_key or "").strip()
    entries = list_available_embedding_models(product_dir)
    for entry in entries:
        if normalized_backbone and entry.backbone != normalized_backbone:
            continue
        if normalized_model_key and entry.model_key != normalized_model_key:
            continue
        return entry
    if normalized_backbone:
        legacy_path = register_model_path(product_dir, normalized_backbone)
        if os.path.exists(legacy_path):
            shared_model_name = tr("debug.embedding.shared_model")
            return EmbeddingModelEntry(
                model_key="",
                backbone=normalized_backbone,
                model_path=legacy_path,
                display_name=f"{shared_model_name} / {algorithm_display_name(normalized_backbone) or normalized_backbone}",
                tool_name=shared_model_name,
            )
    return None


def load_session_lists(session_file: str) -> Tuple[List[str], List[str], List[str]]:
    if not os.path.exists(session_file):
        return [], [], []
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _filter_existing(values: Sequence[str]) -> List[str]:
        return [value for value in values if isinstance(value, str) and os.path.exists(value)]

    return (
        _filter_existing(data.get("ok_files", [])),
        _filter_existing(data.get("ng_files", [])),
        _filter_existing(data.get("test_files", [])),
    )


def _safe_norm_row(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def _pairwise_mean_cosine(bank: np.ndarray) -> float:
    if bank.ndim != 2 or bank.shape[0] < 2:
        return float("nan")
    sims = bank @ bank.T
    iu = np.triu_indices(bank.shape[0], k=1)
    return float(np.mean(sims[iu]))


def _project_pca(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    coords = centered @ basis
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords.astype(np.float32)


def project_embeddings(features: np.ndarray, method: str = "tsne") -> Tuple[np.ndarray, str]:
    method_key = str(method or "tsne").strip().lower()
    n = int(features.shape[0]) if features.ndim == 2 else 0
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), "empty"
    if n < 4 or method_key == "pca" or TSNE is None:
        return _project_pca(features), "pca"

    perplexity = min(30, max(2, n // 3), n - 1)
    if perplexity >= n:
        perplexity = max(1, n - 1)
    if perplexity < 2:
        return _project_pca(features), "pca"

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(features)
    return coords.astype(np.float32), "tsne"


def _extend_names(files: Sequence[str], expected_count: int, prefix: str) -> List[str]:
    names = [os.path.basename(path) for path in files[:expected_count]]
    if len(names) < expected_count:
        names.extend([f"{prefix}_{idx + 1}" for idx in range(len(names), expected_count)])
    return names


def _build_rows(
    model: qr_core.RegisterModel,
    ok_files: Sequence[str],
    ng_files: Sequence[str],
) -> List[EmbeddingAnalysisRow]:
    rows: List[EmbeddingAnalysisRow] = []
    ok_bank = np.asarray(model.ok_bank if model.ok_bank is not None else np.zeros((0, 0), dtype=np.float32))
    ng_bank = np.asarray(model.ng_bank if model.ng_bank is not None else np.zeros((0, 0), dtype=np.float32))

    ok_names = _extend_names(ok_files, ok_bank.shape[0], "ok")
    ng_names = _extend_names(ng_files, ng_bank.shape[0], "ng")

    for idx, vec in enumerate(ok_bank):
        pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(vec, model)
        file_path = ok_files[idx] if idx < len(ok_files) else ""
        rows.append(
            EmbeddingAnalysisRow(
                gt_label="OK",
                file_path=file_path,
                file_name=ok_names[idx],
                pred_label=pred,
                diff=float(diff),
                sim_ok=float(sim_ok),
                sim_ng=float(sim_ng),
            )
        )

    for idx, vec in enumerate(ng_bank):
        pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(vec, model)
        file_path = ng_files[idx] if idx < len(ng_files) else ""
        rows.append(
            EmbeddingAnalysisRow(
                gt_label="NG",
                file_path=file_path,
                file_name=ng_names[idx],
                pred_label=pred,
                diff=float(diff),
                sim_ok=float(sim_ok),
                sim_ng=float(sim_ng),
            )
        )
    return rows


def _margin_summary(rows: Sequence[EmbeddingAnalysisRow], current_margin: float) -> Dict[str, float]:
    ok_diffs = [float(row.diff) for row in rows if row.gt_label == "OK"]
    ng_diffs = [float(row.diff) for row in rows if row.gt_label == "NG"]

    summary: Dict[str, float] = {
        "current_margin": float(current_margin),
        "suggested_margin": float(current_margin),
        "suggested_accuracy": float("nan"),
        "ok_diff_min": float(min(ok_diffs)) if ok_diffs else float("nan"),
        "ok_diff_max": float(max(ok_diffs)) if ok_diffs else float("nan"),
        "ng_diff_min": float(min(ng_diffs)) if ng_diffs else float("nan"),
        "ng_diff_max": float(max(ng_diffs)) if ng_diffs else float("nan"),
        "diff_gap": float("nan"),
        "safe_margin_low": float("nan"),
        "safe_margin_high": float("nan"),
        "safe_margin_width": float("nan"),
    }

    labeled = [row for row in rows if row.gt_label in {"OK", "NG"}]
    if not labeled:
        return summary

    diffs = sorted({float(row.diff) for row in labeled})
    candidates: List[float] = []
    if diffs:
        candidates.append(diffs[0] - 1e-6)
        candidates.extend(diffs)
        candidates.extend((a + b) * 0.5 for a, b in zip(diffs, diffs[1:]))
        candidates.append(diffs[-1] + 1e-6)

    def _accuracy(threshold: float) -> float:
        matched = 0
        for row in labeled:
            pred = "OK" if float(row.diff) >= threshold else "NG"
            if pred == row.gt_label:
                matched += 1
        return float(matched) / float(len(labeled))

    best_margin = float(current_margin)
    best_acc = _accuracy(best_margin)
    for candidate in candidates:
        acc = _accuracy(candidate)
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12
            and abs(candidate - current_margin) < abs(best_margin - current_margin)
        ):
            best_margin = float(candidate)
            best_acc = float(acc)

    summary["suggested_margin"] = best_margin
    summary["suggested_accuracy"] = float(best_acc)

    if ok_diffs and ng_diffs:
        safe_low = float(max(ng_diffs))
        safe_high = float(min(ok_diffs))
        gap = safe_high - safe_low
        summary["diff_gap"] = float(gap)
        summary["safe_margin_low"] = safe_low
        summary["safe_margin_high"] = safe_high
        summary["safe_margin_width"] = float(gap)
        if gap > 0:
            summary["suggested_margin"] = float((safe_low + safe_high) * 0.5)
            summary["suggested_accuracy"] = float(_accuracy(summary["suggested_margin"]))

    return summary


def load_product_analysis(
    session_root: str,
    product_name: str,
    backbone: str,
    model_key: str = "",
    projection_method: str = "tsne",
) -> EmbeddingAnalysisResult:
    product_dir = os.path.join(session_root, product_name)
    if not os.path.isdir(product_dir):
        raise FileNotFoundError(f"product dir not found: {product_dir}")

    entry = _find_embedding_model_entry(product_dir, backbone=backbone, model_key=model_key)
    if entry is None:
        raise FileNotFoundError(
            f"register model not found: subtype={learning_backbone_storage_code(backbone)!r}, model_key={model_key!r}"
        )
    model_path = entry.model_path

    session_file = os.path.join(product_dir, "session.json")
    ok_files, ng_files, _ = load_session_lists(session_file)
    model = qr_core.load_register_model_npz(model_path)
    params_path = os.path.join(product_dir, "product_params.json")
    params = load_product_params(params_path)
    if params.learning_backbone == entry.backbone or params.algorithm == entry.backbone:
        model.score_mode = params.score_mode
        model.margin = float(params.margin)
        model.topk = int(params.topk)
    if not model.is_ready():
        raise RuntimeError("register model is not ready")

    ok_bank = np.asarray(model.ok_bank if model.ok_bank is not None else np.zeros((0, 0), dtype=np.float32), dtype=np.float32)
    ng_bank = np.asarray(model.ng_bank if model.ng_bank is not None else np.zeros((0, 0), dtype=np.float32), dtype=np.float32)
    all_features = np.vstack([ok_bank, ng_bank]).astype(np.float32)
    coords, used_method = project_embeddings(all_features, method=projection_method)

    point_labels = ["OK"] * ok_bank.shape[0] + ["NG"] * ng_bank.shape[0]
    point_names = _extend_names(ok_files, ok_bank.shape[0], "ok") + _extend_names(ng_files, ng_bank.shape[0], "ng")
    rows = _build_rows(model, ok_files, ng_files)
    margin_metrics = _margin_summary(rows, float(model.margin))

    ok_proto = _safe_norm_row(np.asarray(model.ok_proto[0], dtype=np.float32))
    ng_proto = _safe_norm_row(np.asarray(model.ng_proto[0], dtype=np.float32))
    row_matches = sum(1 for row in rows if row.gt_label == row.pred_label)
    metrics = {
        "ok_count": float(ok_bank.shape[0]),
        "ng_count": float(ng_bank.shape[0]),
        "feature_dim": float(all_features.shape[1] if all_features.ndim == 2 and all_features.size else 0),
        "train_accuracy": (float(row_matches) / float(len(rows))) if rows else float("nan"),
        "ok_intra_mean": _pairwise_mean_cosine(ok_bank),
        "ng_intra_mean": _pairwise_mean_cosine(ng_bank),
        "ok_ng_cross_mean": float(np.mean(ok_bank @ ng_bank.T)) if ok_bank.size and ng_bank.size else float("nan"),
        "ok_to_ok_proto": float(np.mean(ok_bank @ ok_proto)) if ok_bank.size else float("nan"),
        "ng_to_ng_proto": float(np.mean(ng_bank @ ng_proto)) if ng_bank.size else float("nan"),
        "ok_to_ng_proto": float(np.mean(ok_bank @ ng_proto)) if ok_bank.size else float("nan"),
        "ng_to_ok_proto": float(np.mean(ng_bank @ ok_proto)) if ng_bank.size else float("nan"),
        "proto_similarity": float(ok_proto @ ng_proto),
    }
    metrics.update(margin_metrics)

    notes: List[str] = []
    if session_file and not os.path.exists(session_file):
        notes.append("session.json missing; file names unavailable.")
    if len(ok_files) != ok_bank.shape[0]:
        notes.append(f"OK file count ({len(ok_files)}) != OK feature count ({ok_bank.shape[0]}).")
    if len(ng_files) != ng_bank.shape[0]:
        notes.append(f"NG file count ({len(ng_files)}) != NG feature count ({ng_bank.shape[0]}).")
    if used_method != projection_method.lower():
        notes.append(f"Projection fallback: requested {projection_method}, used {used_method}.")
    gap = metrics.get("diff_gap", float("nan"))
    if np.isfinite(gap):
        if gap > 0:
            notes.append(
                "Training diffs are fully separated; any margin inside the safe range keeps OK/NG apart on the training set."
            )
        else:
            notes.append(
                "Training diffs overlap; rely on margin validation and a held-out test set instead of the TSNE plot alone."
            )

    return EmbeddingAnalysisResult(
        product_name=product_name,
        backbone=entry.backbone,
        model_path=model_path,
        session_file=session_file,
        projection_method=used_method,
        feature_dim=int(metrics["feature_dim"]),
        point_coords=coords,
        point_labels=point_labels,
        point_names=point_names,
        rows=rows,
        metrics=metrics,
        notes=notes,
        model_key=entry.model_key,
        tool_name=entry.tool_name,
        label_names=list(model.effective_label_names()),
        score_mode=str(model.score_mode),
        margin=float(model.margin),
        topk=int(model.topk),
    )


def visualize_analysis_matplotlib(result: EmbeddingAnalysisResult):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional for GUI-only usage
        raise RuntimeError("matplotlib is not available") from exc
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    coords = result.point_coords
    labels = result.point_labels
    if coords.shape[0] == 0:
        raise RuntimeError("no embeddings to visualize")

    ok_mask = np.array([label == "OK" for label in labels], dtype=bool)
    ng_mask = ~ok_mask

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[ok_mask, 0], coords[ok_mask, 1], c="tab:blue", s=70, alpha=0.8, label="OK")
    plt.scatter(coords[ng_mask, 0], coords[ng_mask, 1], c="tab:red", s=70, alpha=0.8, label="NG")

    if np.any(ok_mask):
        ok_center = coords[ok_mask].mean(axis=0)
        plt.scatter([ok_center[0]], [ok_center[1]], c="navy", marker="*", s=280, label="OK center")
    if np.any(ng_mask):
        ng_center = coords[ng_mask].mean(axis=0)
        plt.scatter([ng_center[0]], [ng_center[1]], c="darkred", marker="*", s=280, label="NG center")

    for idx, name in enumerate(result.point_names):
        plt.annotate(name, (coords[idx, 0], coords[idx, 1]), fontsize=8, alpha=0.75)

    plt.title(
        f"Embedding analysis: {result.product_name} / {algorithm_display_name(result.backbone) or learning_backbone_storage_code(result.backbone)} / {result.projection_method.upper()}",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SESSION_ROOT = ".qr_session"
    PRODUCT_NAME = "Screw"
    BACKBONE = "b0"

    analysis = load_product_analysis(
        session_root=SESSION_ROOT,
        product_name=PRODUCT_NAME,
        backbone=BACKBONE,
        projection_method="tsne",
    )
    visualize_analysis_matplotlib(analysis)
