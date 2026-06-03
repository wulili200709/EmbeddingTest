from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping

import cv2
import numpy as np


FIND_LINE_ALGORITHM = "find_line"
FIND_LINE_SUBPIX_ALGORITHM = "find_line_subpix"
FIND_LINE_ALGORITHMS = {FIND_LINE_ALGORITHM, FIND_LINE_SUBPIX_ALGORITHM}
LINE_DISTANCE_ALGORITHM = "line_distance"
LINE_DISTANCE_REF_NORMAL_ALGORITHM = "line_distance_ref_normal"
LINE_DISTANCE_ALGORITHMS = {LINE_DISTANCE_ALGORITHM, LINE_DISTANCE_REF_NORMAL_ALGORITHM}
MEASUREMENT_ALGORITHMS = [
    FIND_LINE_ALGORITHM,
    FIND_LINE_SUBPIX_ALGORITHM,
    LINE_DISTANCE_ALGORITHM,
    LINE_DISTANCE_REF_NORMAL_ALGORITHM,
]

_DIRECTIONS = {"left_right", "right_left", "top_down", "bottom_up"}
_POLARITIES = {"any", "dark_to_bright", "bright_to_dark"}
_EDGE_DETECTORS = {"canny", "subpix_shen"}
_PEAK_SELECTIONS = {"first", "strongest", "dominant"}


@dataclass(frozen=True)
class FindLineConfig:
    direction: str = "left_right"
    polarity: str = "any"
    scan_step: int = 2
    edge_threshold: float = 10.0
    blur_ksize: int = 3
    min_points: int = 10
    edge_detector: str = "canny"
    peak_selection: str = "dominant"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, defaults: "FindLineConfig" | None = None) -> "FindLineConfig":
        base = defaults or cls()
        payload = dict(data or {})
        direction = str(payload.get("direction", base.direction) or base.direction).strip()
        if direction not in _DIRECTIONS:
            direction = base.direction
        polarity = str(payload.get("polarity", base.polarity) or base.polarity).strip()
        if polarity not in _POLARITIES:
            polarity = base.polarity
        edge_detector = str(payload.get("edge_detector", base.edge_detector) or base.edge_detector).strip()
        if edge_detector not in _EDGE_DETECTORS:
            edge_detector = base.edge_detector
        peak_selection = str(payload.get("peak_selection", base.peak_selection) or base.peak_selection).strip()
        if peak_selection not in _PEAK_SELECTIONS:
            peak_selection = base.peak_selection
        blur_ksize = int(payload.get("blur_ksize", base.blur_ksize) or 0)
        if blur_ksize > 0 and blur_ksize % 2 == 0:
            blur_ksize += 1
        return cls(
            direction=direction,
            polarity=polarity,
            scan_step=max(1, int(payload.get("scan_step", base.scan_step) or 1)),
            edge_threshold=max(0.0, float(payload.get("edge_threshold", base.edge_threshold) or 0.0)),
            blur_ksize=max(0, blur_ksize),
            min_points=max(2, int(payload.get("min_points", base.min_points) or 2)),
            edge_detector=edge_detector,
            peak_selection=peak_selection,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeDistanceConfig:
    line_a: FindLineConfig = field(default_factory=lambda: FindLineConfig(direction="left_right"))
    line_b: FindLineConfig = field(default_factory=lambda: FindLineConfig(direction="right_left"))
    pixel_size_mm: float = 0.0
    roi_label: str = ""
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: str = "px"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None, *, roi_label: str = "") -> "EdgeDistanceConfig":
        payload = dict(params or {})
        default_a = FindLineConfig(direction="left_right")
        default_b = FindLineConfig(direction="right_left")
        limit_unit = str(payload.get("limit_unit", "") or "").strip().lower()
        if not limit_unit:
            limit_unit = "mm" if ("lower_limit_mm" in payload or "upper_limit_mm" in payload) else "px"
        if limit_unit not in {"px", "mm"}:
            limit_unit = "px"
        lower_limit = _optional_float(
            payload.get("lower_limit", payload.get(f"lower_limit_{limit_unit}"))
        )
        upper_limit = _optional_float(
            payload.get("upper_limit", payload.get(f"upper_limit_{limit_unit}"))
        )
        return cls(
            line_a=FindLineConfig.from_dict(payload.get("line_a"), defaults=default_a),
            line_b=FindLineConfig.from_dict(payload.get("line_b"), defaults=default_b),
            pixel_size_mm=max(0.0, float(payload.get("pixel_size_mm", 0.0) or 0.0)),
            roi_label=str(payload.get("roi_label", roi_label) or roi_label or "").strip(),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            limit_unit=limit_unit,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_a": self.line_a.to_dict(),
            "line_b": self.line_b.to_dict(),
            "pixel_size_mm": float(self.pixel_size_mm),
            "roi_label": self.roi_label,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "limit_unit": self.limit_unit,
        }


@dataclass(frozen=True)
class FindLineMeasurementConfig:
    line: FindLineConfig = field(default_factory=lambda: FindLineConfig(direction="left_right"))
    pixel_size_mm: float = 0.0
    roi_label: str = ""
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: str = "px"
    value_mode: str = "position"

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None,
        *,
        roi_label: str = "",
        algorithm: object = FIND_LINE_ALGORITHM,
    ) -> "FindLineMeasurementConfig":
        payload = dict(params or {})
        line_payload = payload.get("line")
        if not isinstance(line_payload, Mapping):
            line_payload = payload.get("line_a")
        algorithm_key = str(algorithm or "").strip().lower()
        default_detector = "subpix_shen" if algorithm_key == FIND_LINE_SUBPIX_ALGORITHM else "canny"
        line = FindLineConfig.from_dict(
            line_payload,
            defaults=FindLineConfig(direction="left_right", edge_detector=default_detector),
        )
        limit_unit = str(payload.get("limit_unit", "") or "").strip().lower()
        if not limit_unit:
            limit_unit = "mm" if ("lower_limit_mm" in payload or "upper_limit_mm" in payload) else "px"
        if limit_unit not in {"px", "mm"}:
            limit_unit = "px"
        lower_limit = _optional_float(
            payload.get("lower_limit", payload.get(f"lower_limit_{limit_unit}"))
        )
        upper_limit = _optional_float(
            payload.get("upper_limit", payload.get(f"upper_limit_{limit_unit}"))
        )
        value_mode = str(payload.get("value_mode", "position") or "position").strip().lower()
        if value_mode not in {"position", "angle", "residual"}:
            value_mode = "position"
        return cls(
            line=line,
            pixel_size_mm=max(0.0, float(payload.get("pixel_size_mm", 0.0) or 0.0)),
            roi_label=str(payload.get("roi_label", roi_label) or roi_label or "").strip(),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            limit_unit=limit_unit,
            value_mode=value_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line.to_dict(),
            "pixel_size_mm": float(self.pixel_size_mm),
            "roi_label": self.roi_label,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "limit_unit": self.limit_unit,
            "value_mode": self.value_mode,
        }


@dataclass(frozen=True)
class FittedLine:
    vx: float
    vy: float
    x0: float
    y0: float
    residual: float
    point_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vx": float(self.vx),
            "vy": float(self.vy),
            "x0": float(self.x0),
            "y0": float(self.y0),
            "residual": float(self.residual),
            "point_count": int(self.point_count),
        }


@dataclass(frozen=True)
class EdgeDistanceResult:
    roi_label: str
    distance_px: float
    distance_mm: float | None
    line_a: FittedLine
    line_b: FittedLine
    angle_delta_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_label": self.roi_label,
            "distance_px": float(self.distance_px),
            "distance_mm": float(self.distance_mm) if self.distance_mm is not None else None,
            "line_a": self.line_a.to_dict(),
            "line_b": self.line_b.to_dict(),
            "angle_delta_deg": float(self.angle_delta_deg),
        }


@dataclass(frozen=True)
class FindLineMeasurementResult:
    roi_label: str
    line: FittedLine
    position_px: float
    position_mm: float | None
    angle_deg: float
    value_mode: str
    roi_xywh: tuple[int, int, int, int] = (0, 0, 0, 0)
    edge_points: tuple[tuple[float, float], ...] = ()
    line_segment: tuple[tuple[float, float], tuple[float, float]] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_label": self.roi_label,
            "line": self.line.to_dict(),
            "position_px": float(self.position_px),
            "position_mm": float(self.position_mm) if self.position_mm is not None else None,
            "angle_deg": float(self.angle_deg),
            "value_mode": self.value_mode,
            "roi_xywh": [int(v) for v in self.roi_xywh],
            "edge_points": [[float(x), float(y)] for x, y in self.edge_points],
            "line_segment": (
                [[float(x), float(y)] for x, y in self.line_segment]
                if self.line_segment is not None
                else None
            ),
        }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def is_measurement_algorithm(name: object) -> bool:
    return str(name or "").strip().lower() in MEASUREMENT_ALGORITHMS


def _shape_from_labels(shape_by_label: Mapping[str, dict], preferred_label: str) -> tuple[str, dict]:
    label = str(preferred_label or "").strip() or "roi1"
    shape = dict(shape_by_label or {}).get(label)
    if shape is None:
        label = "roi"
        shape = dict(shape_by_label or {}).get(label)
    if shape is None:
        raise RuntimeError(f"measurement ROI missing: {preferred_label or 'roi1'}")
    return label, shape


def _crop_from_shape(image_bgr: np.ndarray, shape: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = np.ascontiguousarray(np.asarray(image_bgr))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"unsupported image shape: {image.shape!r}")
    image = image[:, :, :3]
    h_img, w_img = image.shape[:2]
    pts = np.asarray(shape.get("points", []), dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError("measurement ROI points empty")
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x = max(0, int(np.floor(float(x_min))))
    y = max(0, int(np.floor(float(y_min))))
    x2 = min(w_img, int(np.ceil(float(x_max))))
    y2 = min(h_img, int(np.ceil(float(y_max))))
    if x2 <= x or y2 <= y:
        raise RuntimeError("measurement ROI bbox invalid")

    crop = image[y:y2, x:x2].copy()
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
    return crop, mask, (x, y)


def _edge_response(delta: np.ndarray, polarity: str, *, direction: str = "left_right") -> np.ndarray:
    if direction in {"right_left", "bottom_up"}:
        delta = -delta
    if polarity == "dark_to_bright":
        return delta
    if polarity == "bright_to_dark":
        return -delta
    return np.abs(delta)


def _canny_thresholds(config: FindLineConfig) -> tuple[float, float]:
    high = max(1.0, float(config.edge_threshold))
    low = max(0.0, high * 0.5)
    return low, high


def _parabolic_peak_offset(left: float, center: float, right: float) -> float:
    denominator = float(left) - 2.0 * float(center) + float(right)
    if abs(denominator) <= 1e-12:
        return 0.0
    offset = 0.5 * (float(left) - float(right)) / denominator
    return float(max(-1.0, min(1.0, offset)))


def _refine_horizontal_edge_x(gray: np.ndarray, y: int, x: int, config: FindLineConfig) -> float:
    h, w = gray.shape[:2]
    if w < 2:
        return float(x)
    delta = gray[int(y), 1:] - gray[int(y), :-1]
    response = _edge_response(delta, config.polarity, direction=config.direction)
    lo = max(0, int(x) - 2)
    hi = min(w - 2, int(x) + 2)
    if hi < lo:
        return float(x)
    local = response[lo:hi + 1]
    if local.size == 0:
        return float(x)
    best = int(lo + int(np.argmax(local)))
    offset = 0.0
    if 0 < best < response.shape[0] - 1:
        offset = _parabolic_peak_offset(response[best - 1], response[best], response[best + 1])
    return float(best + 1 + offset)


def _refine_vertical_edge_y(gray: np.ndarray, x: int, y: int, config: FindLineConfig) -> float:
    h, w = gray.shape[:2]
    if h < 2:
        return float(y)
    delta = gray[1:, int(x)] - gray[:-1, int(x)]
    response = _edge_response(delta, config.polarity, direction=config.direction)
    lo = max(0, int(y) - 2)
    hi = min(h - 2, int(y) + 2)
    if hi < lo:
        return float(y)
    local = response[lo:hi + 1]
    if local.size == 0:
        return float(y)
    best = int(lo + int(np.argmax(local)))
    offset = 0.0
    if 0 < best < response.shape[0] - 1:
        offset = _parabolic_peak_offset(response[best - 1], response[best], response[best + 1])
    return float(best + 1 + offset)


def _smooth_subpixel_derivative(delta: np.ndarray, *, horizontal: bool) -> np.ndarray:
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    kernel = kernel.reshape(1, -1) if horizontal else kernel.reshape(-1, 1)
    return cv2.filter2D(
        np.asarray(delta, dtype=np.float32),
        cv2.CV_32F,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )


def _is_response_peak(response: np.ndarray, index: int, threshold: float) -> bool:
    center = float(response[int(index)])
    if center < float(threshold):
        return False
    left = float(response[int(index) - 1]) if int(index) > 0 else -math.inf
    right = float(response[int(index) + 1]) if int(index) < len(response) - 1 else -math.inf
    return center >= left and center >= right


def _subpixel_peak_coordinate(response: np.ndarray, index: int) -> float:
    idx = int(index)
    offset = 0.0
    if 0 < idx < len(response) - 1:
        offset = _parabolic_peak_offset(response[idx - 1], response[idx], response[idx + 1])
    return float(idx + 1 + offset)


def _filter_subpixel_edge_runs(points: np.ndarray, config: FindLineConfig) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= int(config.min_points):
        return pts
    if config.direction in {"left_right", "right_left"}:
        primary = pts[:, 1]
        secondary = pts[:, 0]
    else:
        primary = pts[:, 0]
        secondary = pts[:, 1]
    runs: list[np.ndarray] = []
    start = 0
    max_primary_gap = max(1.0, float(config.scan_step) * 2.5)
    max_secondary_jump = max(8.0, float(config.scan_step) * 8.0)
    for idx in range(1, len(pts)):
        if (
            abs(float(primary[idx] - primary[idx - 1])) > max_primary_gap
            or abs(float(secondary[idx] - secondary[idx - 1])) > max_secondary_jump
        ):
            runs.append(pts[start:idx])
            start = idx
    runs.append(pts[start:])
    best = max(runs, key=len)
    if len(best) >= int(config.min_points):
        return np.asarray(best, dtype=np.float32)
    return pts


def _subpixel_line_secondary_at_primary(
    line: FittedLine,
    primary: float,
    *,
    horizontal: bool,
) -> float:
    if horizontal:
        if abs(float(line.vy)) <= 1e-12:
            return float(line.x0)
        t = (float(primary) - float(line.y0)) / float(line.vy)
        return float(line.x0) + t * float(line.vx)
    if abs(float(line.vx)) <= 1e-12:
        return float(line.y0)
    t = (float(primary) - float(line.x0)) / float(line.vx)
    return float(line.y0) + t * float(line.vy)


def _subpixel_point_from_candidate(
    primary: float,
    secondary: float,
    *,
    horizontal: bool,
) -> tuple[float, float]:
    if horizontal:
        return float(secondary), float(primary)
    return float(primary), float(secondary)


def _select_subpixel_edge_points(
    candidates_by_scan: list[tuple[float, list[tuple[float, float]]]],
    *,
    horizontal: bool,
    config: FindLineConfig,
) -> np.ndarray:
    if not candidates_by_scan:
        return np.empty((0, 2), dtype=np.float32)

    def first_candidate(candidates: list[tuple[float, float]]) -> tuple[float, float]:
        return candidates[0]

    def strongest_candidate(candidates: list[tuple[float, float]]) -> tuple[float, float]:
        return max(candidates, key=lambda item: float(item[1]))

    selector = first_candidate if config.peak_selection == "first" else strongest_candidate
    seed_points = np.asarray(
        [
            _subpixel_point_from_candidate(primary, selector(candidates)[0], horizontal=horizontal)
            for primary, candidates in candidates_by_scan
            if candidates
        ],
        dtype=np.float32,
    ).reshape(-1, 2)
    seed_points = _filter_subpixel_edge_runs(seed_points, config)
    if config.peak_selection != "dominant" or len(seed_points) < int(config.min_points):
        return seed_points

    try:
        seed_line, seed_fit_points = fit_line_filtered(
            seed_points,
            min_points=min(int(config.min_points), int(len(seed_points))),
            context="subpixel dominant edge seed",
        )
    except Exception:
        return seed_points

    seed_residual = max(float(seed_line.residual), 0.0)
    gate_px = max(
        2.5,
        seed_residual * 3.0 + 1.0,
        float(config.scan_step) * 2.0,
        float(config.blur_ksize) * 0.75 if config.blur_ksize > 0 else 0.0,
    )
    selected: list[tuple[float, float]] = []
    for primary, candidates in candidates_by_scan:
        if not candidates:
            continue
        predicted = _subpixel_line_secondary_at_primary(seed_line, primary, horizontal=horizontal)
        nearest = min(candidates, key=lambda item: abs(float(item[0]) - predicted))
        nearest_distance = abs(float(nearest[0]) - predicted)
        if nearest_distance <= gate_px:
            selected.append(_subpixel_point_from_candidate(primary, nearest[0], horizontal=horizontal))

    refined_points = np.asarray(selected, dtype=np.float32).reshape(-1, 2)
    if len(refined_points) < int(config.min_points):
        return np.asarray(seed_fit_points, dtype=np.float32).reshape(-1, 2)
    return _filter_subpixel_edge_runs(refined_points, config)


def _find_subpixel_edge_points(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    config: FindLineConfig,
) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    h, w = gray.shape[:2]
    points: list[tuple[float, float]] = []

    if config.direction in {"left_right", "right_left"}:
        if w < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[:, 1:] - gray[:, :-1]
        filtered_delta = _smooth_subpixel_derivative(delta, horizontal=True)
        response = _edge_response(filtered_delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        x_indexes = range(0, w - 1) if config.direction == "left_right" else range(w - 2, -1, -1)
        candidates_by_scan: list[tuple[float, list[tuple[float, float]]]] = []
        for y in range(0, h, config.scan_step):
            row_response = response[y]
            row_valid = adjacent_valid[y]
            candidates: list[tuple[float, float]] = []
            for x in x_indexes:
                if row_valid[int(x)] and _is_response_peak(row_response, int(x), config.edge_threshold):
                    candidates.append(
                        (
                            _subpixel_peak_coordinate(row_response, int(x)),
                            float(row_response[int(x)]),
                        )
                    )
            if candidates:
                candidates_by_scan.append((float(y), candidates))
        points = _select_subpixel_edge_points(candidates_by_scan, horizontal=True, config=config)
    else:
        if h < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[1:, :] - gray[:-1, :]
        filtered_delta = _smooth_subpixel_derivative(delta, horizontal=False)
        response = _edge_response(filtered_delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[1:, :] & valid_mask[:-1, :]
        y_indexes = range(0, h - 1) if config.direction == "top_down" else range(h - 2, -1, -1)
        candidates_by_scan: list[tuple[float, list[tuple[float, float]]]] = []
        for x in range(0, w, config.scan_step):
            col_response = response[:, x]
            col_valid = adjacent_valid[:, x]
            candidates: list[tuple[float, float]] = []
            for y in y_indexes:
                if col_valid[int(y)] and _is_response_peak(col_response, int(y), config.edge_threshold):
                    candidates.append(
                        (
                            _subpixel_peak_coordinate(col_response, int(y)),
                            float(col_response[int(y)]),
                        )
                    )
            if candidates:
                candidates_by_scan.append((float(x), candidates))
        points = _select_subpixel_edge_points(candidates_by_scan, horizontal=False, config=config)

    return _filter_subpixel_edge_runs(np.asarray(points, dtype=np.float32), config)


def find_edge_points(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    config: FindLineConfig,
) -> np.ndarray:
    if config.edge_detector == "subpix_shen":
        return _find_subpixel_edge_points(crop_bgr, mask, config)

    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    canny_low, canny_high = _canny_thresholds(config)
    edges = cv2.Canny(gray_u8, canny_low, canny_high, L2gradient=True)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    h, w = gray.shape[:2]
    points: list[tuple[float, float]] = []

    if config.direction in {"left_right", "right_left"}:
        if w < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[:, 1:] - gray[:, :-1]
        response = _edge_response(delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        x_indexes = range(w) if config.direction == "left_right" else range(w - 1, -1, -1)
        for y in range(0, h, config.scan_step):
            for x in x_indexes:
                left = max(0, int(x) - 1)
                right = min(w - 2, int(x))
                if (
                    valid_mask[y, x]
                    and edges[y, x] > 0
                    and adjacent_valid[y, left:right + 1].any()
                    and response[y, left:right + 1].max(initial=0.0) >= config.edge_threshold
                ):
                    points.append((_refine_horizontal_edge_x(gray, y, x, config), float(y)))
                    break
    else:
        if h < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[1:, :] - gray[:-1, :]
        response = _edge_response(delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[1:, :] & valid_mask[:-1, :]
        y_indexes = range(h) if config.direction == "top_down" else range(h - 1, -1, -1)
        for x in range(0, w, config.scan_step):
            for y in y_indexes:
                top = max(0, int(y) - 1)
                bottom = min(h - 2, int(y))
                if (
                    valid_mask[y, x]
                    and edges[y, x] > 0
                    and adjacent_valid[top:bottom + 1, x].any()
                    and response[top:bottom + 1, x].max(initial=0.0) >= config.edge_threshold
                ):
                    points.append((float(x), _refine_vertical_edge_y(gray, x, y, config)))
                    break

    return np.asarray(points, dtype=np.float32)


def fit_line(points: np.ndarray, *, min_points: int, context: str = "") -> FittedLine:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < int(min_points):
        prefix = f"{context}: " if str(context or "").strip() else ""
        raise RuntimeError(f"{prefix}find line points not enough: {len(pts)}/{int(min_points)}")
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(pts, cv2.DIST_WELSCH, 0, 0.01, 0.01).reshape(-1)]
    norm = math.hypot(vx, vy)
    if norm <= 1e-12:
        raise RuntimeError("fit line direction invalid")
    vx /= norm
    vy /= norm
    distances = np.abs(vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0))
    return FittedLine(
        vx=float(vx),
        vy=float(vy),
        x0=float(x0),
        y0=float(y0),
        residual=float(np.mean(distances)) if distances.size else 0.0,
        point_count=int(len(pts)),
    )


def _point_line_distances(points: np.ndarray, line: FittedLine) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return np.abs(float(line.vy) * (pts[:, 0] - float(line.x0)) - float(line.vx) * (pts[:, 1] - float(line.y0)))


def filter_line_points(
    points: np.ndarray,
    line: FittedLine,
    *,
    min_points: int,
    min_distance_px: float = 2.0,
    sigma: float = 3.0,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= int(min_points):
        return pts
    distances = _point_line_distances(pts, line)
    if distances.size == 0:
        return pts
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(float(min_distance_px), median + float(sigma) * robust_sigma)
    keep = distances <= threshold
    if int(np.count_nonzero(keep)) < int(min_points):
        return pts
    return pts[keep]


def fit_line_filtered(points: np.ndarray, *, min_points: int, context: str = "") -> tuple[FittedLine, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    initial = fit_line(pts, min_points=min_points, context=context)
    filtered = filter_line_points(pts, initial, min_points=min_points)
    if len(filtered) == len(pts):
        return initial, pts
    refined = fit_line(filtered, min_points=min_points, context=context)
    return refined, filtered


def _line_distance_px(line_a: FittedLine, line_b: FittedLine) -> float:
    a = -float(line_a.vy)
    b = float(line_a.vx)
    c = float(line_a.vy) * float(line_a.x0) - float(line_a.vx) * float(line_a.y0)
    norm = math.hypot(a, b)
    if norm <= 1e-12:
        raise RuntimeError("line normal invalid")
    return abs((a * float(line_b.x0) + b * float(line_b.y0) + c) / norm)


def _angle_delta_deg(line_a: FittedLine, line_b: FittedLine) -> float:
    dot = abs(float(line_a.vx) * float(line_b.vx) + float(line_a.vy) * float(line_b.vy))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _line_angle_deg(line: FittedLine) -> float:
    angle = math.degrees(math.atan2(float(line.vy), float(line.vx)))
    if angle < 0.0:
        angle += 180.0
    return angle


def _line_position_px(line: FittedLine, direction: str) -> float:
    if direction in {"left_right", "right_left"}:
        return float(line.x0)
    return float(line.y0)


def _line_segment_in_crop(
    line: FittedLine,
    *,
    crop_width: int,
    crop_height: int,
    origin: tuple[int, int] = (0, 0),
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    w = max(0, int(crop_width))
    h = max(0, int(crop_height))
    if w <= 0 or h <= 0:
        return None
    vx = float(line.vx)
    vy = float(line.vy)
    x0 = float(line.x0)
    y0 = float(line.y0)
    candidates: list[tuple[float, float]] = []
    if abs(vx) > 1e-12:
        for x in (0.0, float(w - 1)):
            t = (x - x0) / vx
            y = y0 + t * vy
            if -0.5 <= y <= float(h - 1) + 0.5:
                candidates.append((x, min(float(h - 1), max(0.0, y))))
    if abs(vy) > 1e-12:
        for y in (0.0, float(h - 1)):
            t = (y - y0) / vy
            x = x0 + t * vx
            if -0.5 <= x <= float(w - 1) + 0.5:
                candidates.append((min(float(w - 1), max(0.0, x)), y))

    unique: list[tuple[float, float]] = []
    for point in candidates:
        if not any(math.hypot(point[0] - old[0], point[1] - old[1]) < 1e-6 for old in unique):
            unique.append(point)
    if len(unique) < 2:
        half_span = max(w, h) * 0.5
        unique = [
            (x0 - vx * half_span, y0 - vy * half_span),
            (x0 + vx * half_span, y0 + vy * half_span),
        ]

    ox, oy = origin
    p0 = unique[0]
    p1 = max(unique[1:], key=lambda p: (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2)
    return (
        (float(p0[0] + ox), float(p0[1] + oy)),
        (float(p1[0] + ox), float(p1[1] + oy)),
    )


def measure_find_line_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
    algorithm: object = FIND_LINE_ALGORITHM,
) -> FindLineMeasurementResult:
    config = FindLineMeasurementConfig.from_params(params, roi_label=preferred_label, algorithm=algorithm)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    points = find_edge_points(crop, mask, config.line)
    context = (
        f"roi={roi_label} line direction={config.line.direction} "
        f"threshold={config.line.edge_threshold:.3f}"
    )
    line, fit_points = fit_line_filtered(points, min_points=config.line.min_points, context=context)
    position_px = _line_position_px(line, config.line.direction)
    position_mm = position_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    ox, oy = origin
    absolute_points = tuple(
        (float(x + ox), float(y + oy))
        for x, y in np.asarray(fit_points, dtype=np.float32).reshape(-1, 2)
    )
    return FindLineMeasurementResult(
        roi_label=roi_label,
        line=line,
        position_px=float(position_px),
        position_mm=float(position_mm) if position_mm is not None else None,
        angle_deg=float(_line_angle_deg(line)),
        value_mode=config.value_mode,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
        edge_points=absolute_points,
        line_segment=_line_segment_in_crop(
            line,
            crop_width=int(crop.shape[1]),
            crop_height=int(crop.shape[0]),
            origin=origin,
        ),
    )


def measure_edge_distance_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> EdgeDistanceResult:
    config = EdgeDistanceConfig.from_params(params, roi_label=preferred_label)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, _origin = _crop_from_shape(image_bgr, shape)
    points_a = find_edge_points(crop, mask, config.line_a)
    points_b = find_edge_points(crop, mask, config.line_b)
    context_a = (
        f"roi={roi_label} line_a direction={config.line_a.direction} "
        f"threshold={config.line_a.edge_threshold:.3f}"
    )
    context_b = (
        f"roi={roi_label} line_b direction={config.line_b.direction} "
        f"threshold={config.line_b.edge_threshold:.3f}"
    )
    line_a, _fit_points_a = fit_line_filtered(points_a, min_points=config.line_a.min_points, context=context_a)
    line_b, _fit_points_b = fit_line_filtered(points_b, min_points=config.line_b.min_points, context=context_b)
    distance_px = _line_distance_px(line_a, line_b)
    distance_mm = distance_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    return EdgeDistanceResult(
        roi_label=roi_label,
        distance_px=float(distance_px),
        distance_mm=float(distance_mm) if distance_mm is not None else None,
        line_a=line_a,
        line_b=line_b,
        angle_delta_deg=float(_angle_delta_deg(line_a, line_b)),
    )


def measure_edge_distance(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> EdgeDistanceResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_edge_distance_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_find_line(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
    algorithm: object = FIND_LINE_ALGORITHM,
) -> FindLineMeasurementResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_find_line_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
        algorithm=algorithm,
    )


def measurement_value(result: EdgeDistanceResult | FindLineMeasurementResult, algorithm: object) -> float:
    key = str(algorithm or "").strip().lower()
    if key == "edge_distance" and isinstance(result, EdgeDistanceResult):
        return float(result.distance_px)
    if key in FIND_LINE_ALGORITHMS and isinstance(result, FindLineMeasurementResult):
        return float(result.position_px)
    raise ValueError(f"Unsupported measurement algorithm: {algorithm}")


def judge_find_line(
    result: FindLineMeasurementResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    config = FindLineMeasurementConfig.from_params(params, roi_label=result.roi_label)
    unit = "deg" if config.value_mode == "angle" else "px"
    if config.value_mode == "angle":
        value = float(result.angle_deg)
    elif config.value_mode == "residual":
        value = float(result.line.residual)
    elif config.limit_unit == "mm":
        distance_mm = result.position_mm
        if distance_mm is None and config.pixel_size_mm > 0.0:
            distance_mm = float(result.position_px) * float(config.pixel_size_mm)
        if distance_mm is None:
            raise RuntimeError("pixel_size_mm is required when find-line limits use mm")
        value = float(distance_mm)
        unit = "mm"
    else:
        value = float(result.position_px)
    lower = config.lower_limit
    upper = config.upper_limit
    ok = True
    if lower is not None and value < lower:
        ok = False
    if upper is not None and value > upper:
        ok = False
    return ("OK" if ok else "NG"), value, lower, upper, unit


def judge_edge_distance(
    result: EdgeDistanceResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    config = EdgeDistanceConfig.from_params(params, roi_label=result.roi_label)
    unit = config.limit_unit
    if unit == "mm":
        distance_mm = result.distance_mm
        if distance_mm is None and config.pixel_size_mm > 0.0:
            distance_mm = float(result.distance_px) * float(config.pixel_size_mm)
        if distance_mm is None:
            raise RuntimeError("pixel_size_mm is required when measurement limits use mm")
        value = float(distance_mm)
    else:
        unit = "px"
        value = float(result.distance_px)
    lower = config.lower_limit
    upper = config.upper_limit
    ok = True
    if lower is not None and value < lower:
        ok = False
    if upper is not None and value > upper:
        ok = False
    return ("OK" if ok else "NG"), value, lower, upper, unit


__all__ = [
    "MEASUREMENT_ALGORITHMS",
    "FIND_LINE_ALGORITHM",
    "FIND_LINE_ALGORITHMS",
    "FIND_LINE_SUBPIX_ALGORITHM",
    "LINE_DISTANCE_ALGORITHM",
    "LINE_DISTANCE_ALGORITHMS",
    "LINE_DISTANCE_REF_NORMAL_ALGORITHM",
    "EdgeDistanceConfig",
    "EdgeDistanceResult",
    "FindLineConfig",
    "FindLineMeasurementConfig",
    "FindLineMeasurementResult",
    "FittedLine",
    "find_edge_points",
    "filter_line_points",
    "fit_line",
    "fit_line_filtered",
    "is_measurement_algorithm",
    "judge_edge_distance",
    "judge_find_line",
    "measure_edge_distance",
    "measure_edge_distance_from_array",
    "measure_find_line",
    "measure_find_line_from_array",
    "measurement_value",
]
