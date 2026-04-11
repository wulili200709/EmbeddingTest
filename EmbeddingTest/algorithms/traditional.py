from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .image_io import imread
from .labelme import labelme_json_of_image, list_shapes_from_labelme, read_shape_from_labelme


TRADITIONAL_ALGORITHMS = ["meanintensity", "meanstd", "meanhsv_h", "meanhsv_v", "meanhsv_s"]


@dataclass
class TraditionalThresholdModel:
    algorithm: str
    threshold: float
    ok_when: str
    ok_mean: float
    ng_mean: float
    accuracy: float
    roi_label: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraditionalThresholdModel":
        return cls(
            algorithm=str(data.get("algorithm", "")),
            threshold=float(data.get("threshold", 0.0)),
            ok_when=str(data.get("ok_when", "greater_equal")),
            ok_mean=float(data.get("ok_mean", 0.0)),
            ng_mean=float(data.get("ng_mean", 0.0)),
            accuracy=float(data.get("accuracy", 0.0)),
            roi_label=str(data.get("roi_label", "roi")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def signed_diff(self, value: float) -> float:
        if self.ok_when == "less_equal":
            return float(self.threshold - value)
        return float(value - self.threshold)

    def predict(self, value: float) -> Tuple[str, float]:
        diff = self.signed_diff(value)
        return ("OK" if diff >= 0.0 else "NG"), diff


def is_traditional_algorithm(name: str) -> bool:
    return str(name or "").strip().lower() in TRADITIONAL_ALGORITHMS


def _normalized_bgr_array(image_bgr: np.ndarray) -> np.ndarray:
    img_bgr = np.ascontiguousarray(np.asarray(image_bgr))
    if img_bgr.ndim != 3 or img_bgr.shape[2] < 3:
        raise ValueError(f"unsupported image shape: {img_bgr.shape!r}")
    return img_bgr[:, :, :3]


def _resolve_roi_shape(
    shape_by_label: Dict[str, dict],
    preferred_label: str = "roi1",
) -> tuple[str, dict]:
    label_name = str(preferred_label or "").strip() or "roi1"
    shape = dict(shape_by_label or {}).get(label_name)
    if shape is None:
        label_name = "roi"
        shape = dict(shape_by_label or {}).get(label_name)
    if shape is None:
        raise RuntimeError(f"memory image missing {preferred_label}/roi")
    return label_name, shape


def _roi_bbox_and_mask(
    *,
    shape: Dict[str, Any],
    image_height: int,
    image_width: int,
) -> Dict[str, Any]:
    pts = np.asarray(shape.get("points", []), dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError("memory ROI points empty")
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x = max(0, int(np.floor(float(x_min))))
    y = max(0, int(np.floor(float(y_min))))
    x2 = min(image_width, int(np.ceil(float(x_max))))
    y2 = min(image_height, int(np.ceil(float(y_max))))
    if x2 <= x or y2 <= y:
        raise RuntimeError("memory ROI bbox invalid")

    mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    rel_pts = pts - np.array([[x, y]], dtype=np.float32)
    if str(shape.get("shape_type", "rectangle")) == "polygon" and len(rel_pts) >= 3:
        cv2.fillPoly(mask, [np.round(rel_pts).astype(np.int32)], 255)
    else:
        p0 = rel_pts.min(axis=0)
        p1 = rel_pts.max(axis=0)
        rx = max(0, int(np.floor(float(p0[0]))))
        ry = max(0, int(np.floor(float(p0[1]))))
        rx2 = min(mask.shape[1], int(np.ceil(float(p1[0]))))
        ry2 = min(mask.shape[0], int(np.ceil(float(p1[1]))))
        mask[ry:ry2, rx:rx2] = 255

    return {
        "mask": mask,
        "bbox_xywh": (x, y, x2 - x, y2 - y),
    }


def _load_roi_mask_crop(img_path: str, preferred_label: str = "roi1") -> Dict[str, Any]:
    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")

    label_name = preferred_label
    shape = read_shape_from_labelme(jpath, preferred_label)
    if shape is None:
        label_name = "roi"
        shape = read_shape_from_labelme(jpath, label_name)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")

    img_bgr = imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    h_img, w_img = img_bgr.shape[:2]

    pts = np.asarray(shape.get("points", []), dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError(f"{img_path} ROI points empty")
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x = max(0, int(np.floor(float(x_min))))
    y = max(0, int(np.floor(float(y_min))))
    x2 = min(w_img, int(np.ceil(float(x_max))))
    y2 = min(h_img, int(np.ceil(float(y_max))))
    if x2 <= x or y2 <= y:
        raise RuntimeError(f"{img_path} ROI bbox invalid")

    crop_bgr = img_bgr[y:y2, x:x2].copy()
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    rel_pts = pts - np.array([[x, y]], dtype=np.float32)
    if str(shape.get("shape_type", "rectangle")) == "polygon" and len(rel_pts) >= 3:
        cv2.fillPoly(mask, [np.round(rel_pts).astype(np.int32)], 255)
    else:
        p0 = rel_pts.min(axis=0)
        p1 = rel_pts.max(axis=0)
        rx = max(0, int(np.floor(float(p0[0]))))
        ry = max(0, int(np.floor(float(p0[1]))))
        rx2 = min(mask.shape[1], int(np.ceil(float(p1[0]))))
        ry2 = min(mask.shape[0], int(np.ceil(float(p1[1]))))
        mask[ry:ry2, rx:rx2] = 255

    return {
        "label_name": label_name,
        "crop_bgr": crop_bgr,
        "crop_gray": crop_gray,
        "mask": mask,
        "bbox_xywh": (x, y, x2 - x, y2 - y),
    }


def _load_roi_mask_crop_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Dict[str, dict],
    preferred_label: str = "roi1",
) -> Dict[str, Any]:
    label_name, shape = _resolve_roi_shape(shape_by_label, preferred_label=preferred_label)
    img_bgr = _normalized_bgr_array(image_bgr)
    h_img, w_img = img_bgr.shape[:2]
    roi_geometry = _roi_bbox_and_mask(shape=shape, image_height=h_img, image_width=w_img)
    x, y, width, height = roi_geometry["bbox_xywh"]
    x2 = x + width
    y2 = y + height
    crop_bgr = img_bgr[y:y2, x:x2].copy()
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "label_name": label_name,
        "crop_bgr": crop_bgr,
        "crop_gray": crop_gray,
        "mask": roi_geometry["mask"],
        "bbox_xywh": roi_geometry["bbox_xywh"],
    }


def compute_roi_metrics(img_path: str, preferred_label: str = "roi1") -> Dict[str, Any]:
    roi = _load_roi_mask_crop(img_path, preferred_label=preferred_label)
    crop_gray = np.asarray(roi["crop_gray"], dtype=np.float32)
    crop_bgr = np.asarray(roi["crop_bgr"], dtype=np.uint8)
    mask = np.asarray(roi["mask"], dtype=np.uint8)
    valid_gray = crop_gray[mask > 0]
    if valid_gray.size == 0:
        raise RuntimeError(f"{img_path} ROI valid pixels empty")

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    valid_hsv = np.asarray(hsv[mask > 0], dtype=np.float32)
    if valid_hsv.size == 0:
        raise RuntimeError(f"{img_path} ROI HSV pixels empty")

    return {
        "file_path": img_path,
        "file_name": os.path.basename(img_path),
        "roi_label": str(roi["label_name"]),
        "bbox_xywh": list(roi["bbox_xywh"]),
        "meanintensity": float(np.mean(valid_gray)),
        "mean_intensity": float(np.mean(valid_gray)),
        "meanstd": float(np.std(valid_gray)),
        "mean_std": float(np.std(valid_gray)),
        "meanhsv_h": float(np.mean(valid_hsv[:, 0])),
        "meanhsv_s": float(np.mean(valid_hsv[:, 1])),
        "meanhsv_v": float(np.mean(valid_hsv[:, 2])),
        "roi_area": int(valid_gray.size),
    }


def compute_roi_metrics_batch(
    img_path: str,
    *,
    preferred_labels: Sequence[str],
    required_algorithms: Sequence[str] | None = None,
    json_path: str | None = None,
) -> List[Dict[str, Any]]:
    path_text = str(img_path or "").strip()
    if not path_text:
        raise FileNotFoundError(path_text)
    img_bgr = imread(path_text, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path_text)

    jpath = str(json_path or labelme_json_of_image(path_text))
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")

    shape_by_label = {
        str(shape.get("label", "")).strip(): shape
        for shape in list_shapes_from_labelme(jpath)
        if str(shape.get("label", "")).strip()
    }
    rows = compute_roi_metrics_batch_from_array(
        img_bgr,
        shape_by_label=shape_by_label,
        preferred_labels=preferred_labels,
        required_algorithms=required_algorithms,
    )
    for row in rows:
        row["file_path"] = path_text
        row["file_name"] = os.path.basename(path_text)
    return rows


def compute_roi_metrics_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Dict[str, dict],
    preferred_label: str = "roi1",
) -> Dict[str, Any]:
    return compute_roi_metrics_batch_from_array(
        image_bgr,
        shape_by_label=shape_by_label,
        preferred_labels=[preferred_label],
    )[0]


def compute_roi_metrics_batch_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Dict[str, dict],
    preferred_labels: Sequence[str],
    required_algorithms: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    labels = [str(label or "").strip() or "roi1" for label in list(preferred_labels or [])]
    if not labels:
        return []

    img_bgr = _normalized_bgr_array(image_bgr)
    need_gray = True
    need_hsv = True
    normalized_algorithms = {
        str(name or "").strip().lower()
        for name in list(required_algorithms or [])
        if is_traditional_algorithm(name)
    }
    if normalized_algorithms:
        need_gray = any(name in {"meanintensity", "meanstd"} for name in normalized_algorithms)
        need_hsv = any(name in {"meanhsv_h", "meanhsv_s", "meanhsv_v"} for name in normalized_algorithms)

    gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if need_gray else None
    hsv_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) if need_hsv else None
    h_img, w_img = img_bgr.shape[:2]

    rows: List[Dict[str, Any]] = []
    for preferred_label in labels:
        label_name, shape = _resolve_roi_shape(shape_by_label, preferred_label=preferred_label)
        roi_geometry = _roi_bbox_and_mask(shape=shape, image_height=h_img, image_width=w_img)
        x, y, width, height = roi_geometry["bbox_xywh"]
        x2 = x + width
        y2 = y + height
        mask = np.asarray(roi_geometry["mask"], dtype=np.uint8)
        valid_pixels = mask > 0

        row: Dict[str, Any] = {
            "roi_label": str(label_name),
            "bbox_xywh": list(roi_geometry["bbox_xywh"]),
        }

        valid_gray = None
        if gray_image is not None:
            crop_gray = np.asarray(gray_image[y:y2, x:x2], dtype=np.float32)
            valid_gray = crop_gray[valid_pixels]
            if valid_gray.size == 0:
                raise RuntimeError("memory ROI valid pixels empty")
            row["meanintensity"] = float(np.mean(valid_gray))
            row["mean_intensity"] = float(np.mean(valid_gray))
            row["meanstd"] = float(np.std(valid_gray))
            row["mean_std"] = float(np.std(valid_gray))
            row["roi_area"] = int(valid_gray.size)

        if hsv_image is not None:
            crop_hsv = np.asarray(hsv_image[y:y2, x:x2], dtype=np.float32)
            valid_hsv = crop_hsv[valid_pixels]
            if valid_hsv.size == 0:
                raise RuntimeError("memory ROI HSV pixels empty")
            row["meanhsv_h"] = float(np.mean(valid_hsv[:, 0]))
            row["meanhsv_s"] = float(np.mean(valid_hsv[:, 1]))
            row["meanhsv_v"] = float(np.mean(valid_hsv[:, 2]))
            if "roi_area" not in row:
                row["roi_area"] = int(valid_hsv.shape[0])

        rows.append(row)
    return rows


def metric_value(metrics: Dict[str, Any], algorithm: str) -> float:
    key = str(algorithm or "").strip().lower()
    if key not in TRADITIONAL_ALGORITHMS:
        raise ValueError(f"Unsupported traditional algorithm: {algorithm}")
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"Metric {key} missing")
    return float(value)


def _iter_thresholds(values: Iterable[float]) -> List[float]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return [0.0]
    candidates = [ordered[0] - 1e-6]
    candidates.extend(ordered)
    candidates.extend((a + b) * 0.5 for a, b in zip(ordered, ordered[1:]))
    candidates.append(ordered[-1] + 1e-6)
    return candidates


def _target_threshold_center(
    ok_values: Sequence[float],
    ng_values: Sequence[float],
    ok_when: str,
) -> float:
    ok_mean = float(np.mean(ok_values))
    ng_mean = float(np.mean(ng_values))
    fallback_mid = (ok_mean + ng_mean) * 0.5

    if ok_when == "greater_equal":
        lower = float(max(ng_values))
        upper = float(min(ok_values))
    else:
        lower = float(max(ok_values))
        upper = float(min(ng_values))

    if lower < upper:
        return (lower + upper) * 0.5
    return fallback_mid


def train_threshold_model(
    ok_files: Sequence[str],
    ng_files: Sequence[str],
    algorithm: str,
    *,
    preferred_label: str = "roi1",
) -> Tuple[TraditionalThresholdModel, List[Dict[str, Any]]]:
    if not ok_files or not ng_files:
        raise RuntimeError("Traditional algorithm needs both OK and NG samples")
    if not is_traditional_algorithm(algorithm):
        raise ValueError(f"Unsupported traditional algorithm: {algorithm}")

    rows: List[Dict[str, Any]] = []
    ok_values: List[float] = []
    ng_values: List[float] = []

    for path in ok_files:
        metrics = compute_roi_metrics(path, preferred_label=preferred_label)
        value = metric_value(metrics, algorithm)
        ok_values.append(value)
        rows.append({"gt": "OK", "value": value, **metrics})

    for path in ng_files:
        metrics = compute_roi_metrics(path, preferred_label=preferred_label)
        value = metric_value(metrics, algorithm)
        ng_values.append(value)
        rows.append({"gt": "NG", "value": value, **metrics})

    best_acc = -1.0
    best_candidates: List[Tuple[str, float]] = []
    all_values = ok_values + ng_values
    for ok_when in ["greater_equal", "less_equal"]:
        for threshold in _iter_thresholds(all_values):
            correct = 0
            for value in ok_values:
                pred_ok = value >= threshold if ok_when == "greater_equal" else value <= threshold
                correct += 1 if pred_ok else 0
            for value in ng_values:
                pred_ng = value < threshold if ok_when == "greater_equal" else value > threshold
                correct += 1 if pred_ng else 0
            acc = float(correct) / float(len(ok_values) + len(ng_values))
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_candidates = [(ok_when, float(threshold))]
            elif abs(acc - best_acc) <= 1e-12:
                best_candidates.append((ok_when, float(threshold)))

    assert best_candidates
    best_ok_when, best_threshold = min(
        best_candidates,
        key=lambda item: (
            abs(item[1] - _target_threshold_center(ok_values, ng_values, item[0])),
            item[1],
        ),
    )
    best_model = TraditionalThresholdModel(
        algorithm=str(algorithm),
        threshold=float(best_threshold),
        ok_when=best_ok_when,
        ok_mean=float(np.mean(ok_values)),
        ng_mean=float(np.mean(ng_values)),
        accuracy=best_acc,
        roi_label=preferred_label,
    )
    return best_model, rows


def train_threshold_model_from_samples(
    ok_samples: Sequence[tuple[str, str]],
    ng_samples: Sequence[tuple[str, str]],
    algorithm: str,
    *,
    preferred_label: str = "roi1",
) -> Tuple[TraditionalThresholdModel, List[Dict[str, Any]]]:
    if not ok_samples or not ng_samples:
        raise RuntimeError("Traditional algorithm needs both OK and NG samples")
    if not is_traditional_algorithm(algorithm):
        raise ValueError(f"Unsupported traditional algorithm: {algorithm}")

    rows: List[Dict[str, Any]] = []
    ok_values: List[float] = []
    ng_values: List[float] = []

    def _normalized_entries(samples: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        fallback = str(preferred_label or "roi1").strip() or "roi1"
        for entry in list(samples or []):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            path = str(entry[0] or "").strip()
            label = str(entry[1] or "").strip() or fallback
            if not path:
                continue
            entries.append((path, label))
        return entries

    for path, label in _normalized_entries(ok_samples):
        metrics = compute_roi_metrics(path, preferred_label=label)
        value = metric_value(metrics, algorithm)
        ok_values.append(value)
        rows.append({"gt": "OK", "value": value, **metrics})

    for path, label in _normalized_entries(ng_samples):
        metrics = compute_roi_metrics(path, preferred_label=label)
        value = metric_value(metrics, algorithm)
        ng_values.append(value)
        rows.append({"gt": "NG", "value": value, **metrics})

    best_acc = -1.0
    best_candidates: List[Tuple[str, float]] = []
    all_values = ok_values + ng_values
    for ok_when in ["greater_equal", "less_equal"]:
        for threshold in _iter_thresholds(all_values):
            correct = 0
            for value in ok_values:
                pred_ok = value >= threshold if ok_when == "greater_equal" else value <= threshold
                correct += 1 if pred_ok else 0
            for value in ng_values:
                pred_ng = value < threshold if ok_when == "greater_equal" else value > threshold
                correct += 1 if pred_ng else 0
            acc = float(correct) / float(len(ok_values) + len(ng_values))
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_candidates = [(ok_when, float(threshold))]
            elif abs(acc - best_acc) <= 1e-12:
                best_candidates.append((ok_when, float(threshold)))

    assert best_candidates
    best_ok_when, best_threshold = min(
        best_candidates,
        key=lambda item: (
            abs(item[1] - _target_threshold_center(ok_values, ng_values, item[0])),
            item[1],
        ),
    )
    labels = [
        str(label or "").strip()
        for _path, label in _normalized_entries(ok_samples) + _normalized_entries(ng_samples)
        if str(label or "").strip()
    ]
    best_model = TraditionalThresholdModel(
        algorithm=str(algorithm),
        threshold=float(best_threshold),
        ok_when=best_ok_when,
        ok_mean=float(np.mean(ok_values)),
        ng_mean=float(np.mean(ng_values)),
        accuracy=best_acc,
        roi_label=str(preferred_label or (labels[0] if labels else "roi1")).strip() or "roi1",
    )
    return best_model, rows


__all__ = [
    "TRADITIONAL_ALGORITHMS",
    "TraditionalThresholdModel",
    "compute_roi_metrics_batch",
    "compute_roi_metrics",
    "compute_roi_metrics_batch_from_array",
    "compute_roi_metrics_from_array",
    "is_traditional_algorithm",
    "metric_value",
    "train_threshold_model",
    "train_threshold_model_from_samples",
]
