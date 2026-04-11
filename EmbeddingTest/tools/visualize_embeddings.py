from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional for GUI-only usage
    plt = None

try:
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - fallback to PCA
    TSNE = None

if __package__ in (None, ""):
    root_str = str(Path(__file__).resolve().parents[1])
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
    items_by_key: Dict[str, object] = {}
    for item in load_inspection_items(inspection_items_path):
        item_key = str(getattr(item, "effective_model_key", getattr(item, "model_key", "")) or "").strip()
        if item_key and item_key not in items_by_key:
            items_by_key[item_key] = item

    for name in sorted(os.listdir(product_dir)):
        parsed = _parse_register_model_filename(name)
        if parsed is None:
            continue
        model_key, backbone = parsed
        item = items_by_key.get(model_key) if model_key else None
        tool_name = "共享模型"
        if item is not None:
            base_name = str(item.display_name or item.roi_label or item.item_id or model_key).strip()
            roi_hint = f"{item.camera_id}/{item.roi_label}".strip("/")
            tool_name = f"{base_name} ({roi_hint})" if roi_hint and base_name != roi_hint else (base_name or roi_hint or model_key)
        elif model_key:
            tool_name = model_key
        if item is not None:
            group_name = str(getattr(item, "task_group", "") or "").strip()
            if group_name:
                camera_hint = str(getattr(item, "camera_id", "") or "").strip()
                tool_name = f"{group_name} ({camera_hint})" if camera_hint else group_name
        elif not tool_name:
            tool_name = "shared model"
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
            return EmbeddingModelEntry(
                model_key="",
                backbone=normalized_backbone,
                model_path=legacy_path,
                display_name=f"共享模型 / {algorithm_display_name(normalized_backbone) or normalized_backbone}",
                tool_name="共享模型",
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


def _load_session_payload(session_file: str) -> dict:
    if not os.path.exists(session_file):
        return {}
    with open(session_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve_session_paths(
    session_payload: dict,
    *,
    product_dir: str,
) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    raw_paths = list(session_payload.get("train_files", []) or [])
    if not raw_paths:
        raw_paths = list(session_payload.get("ok_files", []) or []) + list(session_payload.get("ng_files", []) or [])
    for raw_path in raw_paths:
        text = str(raw_path or "").strip()
        if not text:
            continue
        resolved = text
        if not os.path.isabs(resolved):
            resolved = os.path.normpath(os.path.join(product_dir, text))
        if not os.path.exists(resolved):
            continue
        entries.append((resolved, text.replace("\\", "/")))
    return entries


def _load_sample_annotations_payload(product_dir: str) -> Dict[str, Dict[str, str]]:
    store_path = os.path.join(product_dir, "sample_annotations.json")
    if not os.path.exists(store_path):
        return {}
    try:
        with open(store_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    images_payload = payload.get("images", payload) if isinstance(payload, dict) else {}
    if not isinstance(images_payload, dict):
        return {}
    normalized: Dict[str, Dict[str, str]] = {}
    for path_key, raw_value in images_payload.items():
        if not isinstance(raw_value, dict):
            continue
        roi_status = raw_value.get("roi_status", raw_value)
        if not isinstance(roi_status, dict):
            continue
        normalized[str(path_key or "").replace("\\", "/")] = {
            str(key or "").strip(): str(value or "").strip().upper()
            for key, value in roi_status.items()
            if str(key or "").strip() and str(value or "").strip().upper() in {"OK", "NG"}
        }
    return normalized


def _analysis_sample_name(path: str, label_name: str) -> str:
    base = os.path.basename(str(path or "").strip())
    label = str(label_name or "").strip()
    if base and label:
        return f"{base} [{label}]"
    return base or label or "sample"


def compact_plot_label(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    roi_label = ""
    match = re.search(r"\[([^\]]+)\]\s*$", text)
    if match:
        roi_label = str(match.group(1) or "").strip()
        text = text[: match.start()].strip()
    base = os.path.basename(text)
    stem, _ext = os.path.splitext(base)
    token = stem.rsplit("_", 1)[-1] if stem else base
    token = str(token or stem or base or name).strip()
    if roi_label:
        roi_short = re.sub(r"^roi", "r", roi_label, flags=re.IGNORECASE)
        return f"{token}[{roi_short}]"
    return token


def _resolve_group_analysis_samples(
    product_dir: str,
    *,
    model_key: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not model_key:
        return [], []
    inspection_items = load_inspection_items(os.path.join(product_dir, "inspection_items.json"))
    group_items = [
        item
        for item in inspection_items
        if str(getattr(item, "effective_model_key", getattr(item, "model_key", "")) or "").strip() == str(model_key).strip()
    ]
    if not group_items:
        return [], []
    label_names = list(
        dict.fromkeys(
            str(getattr(item, "roi_label", "") or "").strip()
            for item in group_items
            if str(getattr(item, "roi_label", "") or "").strip()
        )
    )
    camera_role = str(getattr(group_items[0], "camera_id", "") or "cam1").strip() or "cam1"
    session_payload = _load_session_payload(os.path.join(product_dir, "session.json"))
    sample_entries = _resolve_session_paths(session_payload, product_dir=product_dir)
    annotations_by_path = _load_sample_annotations_payload(product_dir)
    ok_samples: List[Tuple[str, str]] = []
    ng_samples: List[Tuple[str, str]] = []
    for resolved_path, session_key in sample_entries:
        annotation_payload = annotations_by_path.get(session_key, {})
        if not annotation_payload:
            continue
        json_path = qr_core.labelme_json_of_image(resolved_path)
        if not os.path.exists(json_path):
            continue
        for label_name in label_names:
            try:
                if qr_core.read_shape_from_labelme(json_path, label_name) is None:
                    continue
            except Exception:
                continue
            status = str(annotation_payload.get(f"{camera_role}::{label_name}", "") or "").strip().upper()
            if status == "OK":
                ok_samples.append((resolved_path, label_name))
            elif status == "NG":
                ng_samples.append((resolved_path, label_name))
    return ok_samples, ng_samples


def _analysis_payload_from_model(
    model: qr_core.RegisterModel,
    *,
    ok_files: Sequence[str],
    ng_files: Sequence[str],
    product_dir: str,
    backbone: str,
    model_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[str], List[str]]:
    ok_analysis_bank = getattr(model, "ok_analysis_bank", None)
    ng_analysis_bank = getattr(model, "ng_analysis_bank", None)
    ok_bank = np.asarray(
        ok_analysis_bank if ok_analysis_bank is not None else model.ok_bank if model.ok_bank is not None else np.zeros((0, 0), dtype=np.float32),
        dtype=np.float32,
    )
    ng_bank = np.asarray(
        ng_analysis_bank if ng_analysis_bank is not None else model.ng_bank if model.ng_bank is not None else np.zeros((0, 0), dtype=np.float32),
        dtype=np.float32,
    )
    ok_names = list(getattr(model, "ok_analysis_names", None) or [])
    ng_names = list(getattr(model, "ng_analysis_names", None) or [])
    ok_paths = list(getattr(model, "ok_analysis_paths", None) or [])
    ng_paths = list(getattr(model, "ng_analysis_paths", None) or [])
    notes: List[str] = []

    explicit_analysis = (
        ok_analysis_bank is not None
        and ng_analysis_bank is not None
        and ok_bank.shape[0] > 0
        and ng_bank.shape[0] > 0
    )
    if not explicit_analysis and bool(getattr(model, "grouped_proto_only", False)) and model_key:
        ok_samples, ng_samples = _resolve_group_analysis_samples(product_dir, model_key=model_key)
        if ok_samples and ng_samples:
            device = str(getattr(model, "device", "") or "") or "cpu"
            feat_net, _ = qr_core.load_backbone(backbone, device=device)
            ok_bank = np.stack(
                [qr_core.embed_one(path, feat_net, label_name=label, device=device) for path, label in ok_samples]
            ).astype(np.float32)
            ng_bank = np.stack(
                [qr_core.embed_one(path, feat_net, label_name=label, device=device) for path, label in ng_samples]
            ).astype(np.float32)
            ok_names = [_analysis_sample_name(path, label) for path, label in ok_samples]
            ng_names = [_analysis_sample_name(path, label) for path, label in ng_samples]
            ok_paths = [str(path) for path, _label in ok_samples]
            ng_paths = [str(path) for path, _label in ng_samples]
            notes.append("Recovered grouped analysis points from session annotations.")

    if len(ok_names) != ok_bank.shape[0]:
        ok_names = _bank_display_names(model, ok_files, ok_bank.shape[0], "ok", gt_label="OK")
    if len(ng_names) != ng_bank.shape[0]:
        ng_names = _bank_display_names(model, ng_files, ng_bank.shape[0], "ng", gt_label="NG")
    if len(ok_paths) != ok_bank.shape[0]:
        ok_paths = _bank_display_paths(model, ok_files, ok_bank.shape[0])
    if len(ng_paths) != ng_bank.shape[0]:
        ng_paths = _bank_display_paths(model, ng_files, ng_bank.shape[0])

    return ok_bank, ng_bank, ok_names, ng_names, ok_paths, ng_paths, notes


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


def _bank_display_names(
    model: qr_core.RegisterModel,
    files: Sequence[str],
    expected_count: int,
    prefix: str,
    *,
    gt_label: str,
) -> List[str]:
    if bool(getattr(model, "grouped_proto_only", False)) and expected_count == 1:
        return [f"{gt_label} proto"]
    return _extend_names(files, expected_count, prefix)


def _bank_display_paths(
    model: qr_core.RegisterModel,
    files: Sequence[str],
    expected_count: int,
) -> List[str]:
    if bool(getattr(model, "grouped_proto_only", False)) and expected_count == 1:
        return [""]
    return [str(files[idx]) if idx < len(files) else "" for idx in range(expected_count)]


def _build_rows(
    model: qr_core.RegisterModel,
    ok_bank: np.ndarray,
    ng_bank: np.ndarray,
    ok_names: Sequence[str],
    ng_names: Sequence[str],
    ok_paths: Sequence[str],
    ng_paths: Sequence[str],
) -> List[EmbeddingAnalysisRow]:
    rows: List[EmbeddingAnalysisRow] = []

    for idx, vec in enumerate(ok_bank):
        pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(vec, model)
        rows.append(
            EmbeddingAnalysisRow(
                gt_label="OK",
                file_path=str(ok_paths[idx]) if idx < len(ok_paths) else "",
                file_name=str(ok_names[idx]) if idx < len(ok_names) else f"ok_{idx + 1}",
                pred_label=pred,
                diff=float(diff),
                sim_ok=float(sim_ok),
                sim_ng=float(sim_ng),
            )
        )

    for idx, vec in enumerate(ng_bank):
        pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(vec, model)
        rows.append(
            EmbeddingAnalysisRow(
                gt_label="NG",
                file_path=str(ng_paths[idx]) if idx < len(ng_paths) else "",
                file_name=str(ng_names[idx]) if idx < len(ng_names) else f"ng_{idx + 1}",
                pred_label=pred,
                diff=float(diff),
                sim_ok=float(sim_ok),
                sim_ng=float(sim_ng),
            )
        )
    return rows


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
            f"register model not found: backbone={backbone!r}, model_key={model_key!r}"
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

    ok_bank, ng_bank, ok_names, ng_names, ok_paths, ng_paths, analysis_notes = _analysis_payload_from_model(
        model,
        ok_files=ok_files,
        ng_files=ng_files,
        product_dir=product_dir,
        backbone=entry.backbone,
        model_key=entry.model_key,
    )
    all_features = np.vstack([ok_bank, ng_bank]).astype(np.float32)
    coords, used_method = project_embeddings(all_features, method=projection_method)

    point_labels = ["OK"] * ok_bank.shape[0] + ["NG"] * ng_bank.shape[0]
    point_names = list(ok_names) + list(ng_names)
    rows = _build_rows(
        model,
        ok_bank,
        ng_bank,
        ok_names,
        ng_names,
        ok_paths,
        ng_paths,
    )

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

    notes: List[str] = []
    notes.extend(analysis_notes)
    if session_file and not os.path.exists(session_file):
        notes.append("session.json missing; file names unavailable.")
    if not bool(getattr(model, "grouped_proto_only", False)) and len(ok_files) != ok_bank.shape[0]:
        notes.append(f"OK file count ({len(ok_files)}) != OK feature count ({ok_bank.shape[0]}).")
    if not bool(getattr(model, "grouped_proto_only", False)) and len(ng_files) != ng_bank.shape[0]:
        notes.append(f"NG file count ({len(ng_files)}) != NG feature count ({ng_bank.shape[0]}).")
    if used_method != projection_method.lower():
        notes.append(f"Projection fallback: requested {projection_method}, used {used_method}.")

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
    )


def visualize_analysis_matplotlib(result: EmbeddingAnalysisResult):
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
        plt.annotate(compact_plot_label(name), (coords[idx, 0], coords[idx, 1]), fontsize=8, alpha=0.75)

    plt.title(
        f"Embedding analysis: {result.product_name} / {result.backbone} / {result.projection_method.upper()}",
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
    BACKBONE = "efficientnet_b0"

    analysis = load_product_analysis(
        session_root=SESSION_ROOT,
        product_name=PRODUCT_NAME,
        backbone=BACKBONE,
        projection_method="tsne",
    )
    visualize_analysis_matplotlib(analysis)
