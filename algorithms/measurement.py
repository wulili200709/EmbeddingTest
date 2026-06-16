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
PIN_CENTER_DISTANCE_ALGORITHM = "pin_center_distance"
BRIGHT_BLOCK_CENTER_ALGORITHM = "bright_block_center"
BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM = "bright_block_y_distance"
LINE_DISTANCE_ALGORITHM = "line_distance"
LINE_DISTANCE_REF_NORMAL_ALGORITHM = "line_distance_ref_normal"
LINE_DISTANCE_ALGORITHMS = {LINE_DISTANCE_ALGORITHM, LINE_DISTANCE_REF_NORMAL_ALGORITHM}
CENTER_DISTANCE_ALGORITHM = "center_distance"
CENTER_DISTANCE_ALGORITHMS = {CENTER_DISTANCE_ALGORITHM}
MEASUREMENT_ALGORITHMS = [
    FIND_LINE_ALGORITHM,
    FIND_LINE_SUBPIX_ALGORITHM,
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    PIN_CENTER_DISTANCE_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    LINE_DISTANCE_ALGORITHM,
    LINE_DISTANCE_REF_NORMAL_ALGORITHM,
    CENTER_DISTANCE_ALGORITHM,
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
class PinCenterDistanceConfig:
    pixel_size_mm: float = 0.0
    roi_label: str = ""
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: str = "px"
    threshold: float = 0.0
    min_area_px: float = 12.0
    max_area_px: float = 0.0
    min_aspect_ratio: float = 1.6
    min_width_px: float = 4.0
    min_height_px: float = 1.0
    max_height_px: float = 0.0
    target_orientation: str = "horizontal"
    distance_mode: str = "euclidean"
    sort_axis: str = "y"
    blur_ksize: int = 3
    morph_open_width: int = 5
    morph_open_height: int = 1
    morph_close_size: int = 3
    center_target: str = "inner_bright_strip"
    refine_center: bool = True
    refine_expand_x_ratio: float = 0.25
    refine_expand_y_ratio: float = 2.5
    refine_min_fill_ratio: float = 0.25
    inner_strip_min_width_ratio: float = 0.45
    inner_strip_y_bias: float = 0.92
    min_pair_separation_ratio: float = 0.12
    min_pair_separation_size_ratio: float = 1.2

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None,
        *,
        roi_label: str = "",
    ) -> "PinCenterDistanceConfig":
        payload = dict(params or {})
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
        target_orientation = str(payload.get("target_orientation", "horizontal") or "horizontal").strip().lower()
        if target_orientation not in {"horizontal", "vertical", "any"}:
            target_orientation = "horizontal"
        distance_mode = str(payload.get("distance_mode", "euclidean") or "euclidean").strip().lower()
        if distance_mode not in {"euclidean", "vertical", "horizontal"}:
            distance_mode = "euclidean"
        center_target = str(payload.get("center_target", "inner_bright_strip") or "inner_bright_strip").strip().lower()
        if center_target not in {"inner_bright_strip", "metal_body"}:
            center_target = "inner_bright_strip"
        sort_axis = str(payload.get("sort_axis", "") or "").strip().lower()
        if sort_axis not in {"x", "y"}:
            sort_axis = "x" if distance_mode == "horizontal" else "y"
        blur_ksize = int(payload.get("blur_ksize", 3) or 0)
        if blur_ksize > 0 and blur_ksize % 2 == 0:
            blur_ksize += 1
        morph_open_width = max(0, int(payload.get("morph_open_width", 5) or 0))
        morph_open_height = max(0, int(payload.get("morph_open_height", 1) or 0))
        morph_close_size = max(0, int(payload.get("morph_close_size", 3) or 0))
        if morph_close_size > 0 and morph_close_size % 2 == 0:
            morph_close_size += 1
        return cls(
            pixel_size_mm=max(0.0, float(payload.get("pixel_size_mm", 0.0) or 0.0)),
            roi_label=str(payload.get("roi_label", roi_label) or roi_label or "").strip(),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            limit_unit=limit_unit,
            threshold=max(0.0, float(payload.get("threshold", 0.0) or 0.0)),
            min_area_px=max(0.0, float(payload.get("min_area_px", 12.0) or 0.0)),
            max_area_px=max(0.0, float(payload.get("max_area_px", 0.0) or 0.0)),
            min_aspect_ratio=max(1.0, float(payload.get("min_aspect_ratio", 1.6) or 1.0)),
            min_width_px=max(0.0, float(payload.get("min_width_px", 4.0) or 0.0)),
            min_height_px=max(0.0, float(payload.get("min_height_px", 1.0) or 0.0)),
            max_height_px=max(0.0, float(payload.get("max_height_px", 0.0) or 0.0)),
            target_orientation=target_orientation,
            distance_mode=distance_mode,
            sort_axis=sort_axis,
            blur_ksize=max(0, blur_ksize),
            morph_open_width=morph_open_width,
            morph_open_height=morph_open_height,
            morph_close_size=morph_close_size,
            center_target=center_target,
            refine_center=_bool_param(payload.get("refine_center"), default=True),
            refine_expand_x_ratio=max(0.0, float(payload.get("refine_expand_x_ratio", 0.25) or 0.0)),
            refine_expand_y_ratio=max(0.0, float(payload.get("refine_expand_y_ratio", 2.5) or 0.0)),
            refine_min_fill_ratio=max(0.05, min(0.95, float(payload.get("refine_min_fill_ratio", 0.25) or 0.25))),
            inner_strip_min_width_ratio=max(
                0.05,
                min(1.0, float(payload.get("inner_strip_min_width_ratio", 0.45) or 0.45)),
            ),
            inner_strip_y_bias=max(
                0.0,
                min(1.0, float(payload.get("inner_strip_y_bias", 0.92) or 0.92)),
            ),
            min_pair_separation_ratio=max(
                0.0,
                min(0.9, float(payload.get("min_pair_separation_ratio", 0.12) or 0.0)),
            ),
            min_pair_separation_size_ratio=max(
                0.0,
                min(10.0, float(payload.get("min_pair_separation_size_ratio", 1.2) or 0.0)),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


@dataclass(frozen=True)
class PinCenterCandidate:
    center_xy: tuple[float, float]
    box_points: tuple[tuple[float, float], ...]
    area_px: float
    bbox_xywh: tuple[int, int, int, int]
    width_px: float
    height_px: float
    aspect_ratio: float
    angle_deg: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "box_points": [[float(x), float(y)] for x, y in self.box_points],
            "area_px": float(self.area_px),
            "bbox_xywh": [int(v) for v in self.bbox_xywh],
            "width_px": float(self.width_px),
            "height_px": float(self.height_px),
            "aspect_ratio": float(self.aspect_ratio),
            "angle_deg": float(self.angle_deg),
            "score": float(self.score),
        }


@dataclass(frozen=True)
class PinCenterDistanceResult:
    roi_label: str
    distance_px: float
    distance_mm: float | None
    center_a: tuple[float, float]
    center_b: tuple[float, float]
    candidates: tuple[PinCenterCandidate, ...]
    threshold: float
    distance_mode: str
    roi_xywh: tuple[int, int, int, int] = (0, 0, 0, 0)
    measurement_type: str = PIN_CENTER_DISTANCE_ALGORITHM
    dimension_segment: tuple[tuple[float, float], tuple[float, float]] | None = None

    def to_dict(self) -> Dict[str, Any]:
        raw_dimension = self.dimension_segment or (self.center_a, self.center_b)
        dimension_segment = [[float(x), float(y)] for x, y in raw_dimension]
        center_points = [
            [float(self.center_a[0]), float(self.center_a[1])],
            [float(self.center_b[0]), float(self.center_b[1])],
        ]
        return {
            "type": str(self.measurement_type or PIN_CENTER_DISTANCE_ALGORITHM),
            "roi_label": self.roi_label,
            "distance_px": float(self.distance_px),
            "distance_mm": float(self.distance_mm) if self.distance_mm is not None else None,
            "center_a": [float(self.center_a[0]), float(self.center_a[1])],
            "center_b": [float(self.center_b[0]), float(self.center_b[1])],
            "center_points": center_points,
            "edge_points": center_points,
            "dimension_segment": dimension_segment,
            "line_segment": dimension_segment,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "threshold": float(self.threshold),
            "distance_mode": self.distance_mode,
            "roi_xywh": [int(v) for v in self.roi_xywh],
        }


@dataclass(frozen=True)
class BrightBlockCenterResult:
    roi_label: str
    center_xy: tuple[float, float]
    candidate: PinCenterCandidate
    candidates: tuple[PinCenterCandidate, ...]
    threshold: float
    orientation: str
    roi_xywh: tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> Dict[str, Any]:
        center = [float(self.center_xy[0]), float(self.center_xy[1])]
        return {
            "type": BRIGHT_BLOCK_CENTER_ALGORITHM,
            "roi_label": self.roi_label,
            "center": center,
            "center_xy": center,
            "center_points": [center],
            "edge_points": [center],
            "candidate": self.candidate.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "threshold": float(self.threshold),
            "orientation": self.orientation,
            "roi_xywh": [int(v) for v in self.roi_xywh],
        }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _bool_param(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "y", "on", "enable", "enabled"}


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


def _bright_threshold(gray: np.ndarray, valid_mask: np.ndarray, configured_threshold: float) -> float:
    if float(configured_threshold) > 0.0:
        return float(configured_threshold)
    values = np.asarray(gray, dtype=np.float32)[np.asarray(valid_mask, dtype=bool)]
    if values.size == 0:
        raise RuntimeError("pin center ROI mask empty")
    if float(values.max() - values.min()) < 3.0:
        raise RuntimeError("pin center ROI contrast too low")
    values_u8 = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _binary = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(otsu) <= 1.0:
        otsu = float(np.percentile(values, 85.0))
    return float(max(1.0, min(254.0, otsu)))


def _pin_center_binary(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    config: PinCenterDistanceConfig,
) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    threshold = _bright_threshold(gray, valid_mask, config.threshold)
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[(gray >= float(threshold)) & valid_mask] = 255
    if config.morph_open_width > 0 and config.morph_open_height > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(config.morph_open_width), int(config.morph_open_height)),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if config.morph_close_size >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(config.morph_close_size), int(config.morph_close_size)),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary[~valid_mask] = 0
    return binary, threshold


def _pin_candidate_from_contour(
    contour: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
) -> PinCenterCandidate | None:
    area = float(abs(cv2.contourArea(contour)))
    if area < float(config.min_area_px):
        return None
    if config.max_area_px > 0.0 and area > float(config.max_area_px):
        return None
    x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
    if w <= 0 or h <= 0:
        return None
    width = float(w)
    height = float(h)
    if width < float(config.min_width_px) or height < float(config.min_height_px):
        return None
    if config.target_orientation == "horizontal":
        if width / max(height, 1.0) < float(config.min_aspect_ratio):
            return None
        if config.max_height_px > 0.0 and height > float(config.max_height_px):
            return None
        aspect = width / max(height, 1.0)
    elif config.target_orientation == "vertical":
        if height / max(width, 1.0) < float(config.min_aspect_ratio):
            return None
        if config.max_height_px > 0.0 and width > float(config.max_height_px):
            return None
        aspect = height / max(width, 1.0)
    else:
        aspect = max(width, height) / max(min(width, height), 1.0)
        if aspect < float(config.min_aspect_ratio):
            return None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if float(rw) <= 0.0 or float(rh) <= 0.0:
        return None
    box = cv2.boxPoints(rect)
    ox, oy = origin
    absolute_box = tuple((float(px + ox), float(py + oy)) for px, py in np.asarray(box).reshape(-1, 2))
    clipped_aspect = min(float(aspect), 20.0)
    score = float(area * clipped_aspect)
    return PinCenterCandidate(
        center_xy=(float(cx + ox), float(cy + oy)),
        box_points=absolute_box,
        area_px=area,
        bbox_xywh=(int(x + ox), int(y + oy), int(w), int(h)),
        width_px=width,
        height_px=height,
        aspect_ratio=float(aspect),
        angle_deg=float(angle),
        score=score,
    )


def _find_pin_center_candidates(
    binary: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
) -> tuple[PinCenterCandidate, ...]:
    contours, _hierarchy = cv2.findContours(
        np.asarray(binary, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[PinCenterCandidate] = []
    for contour in contours:
        candidate = _pin_candidate_from_contour(contour, origin=origin, config=config)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lx2 = lx + max(0, lw)
    ly2 = ly + max(0, lh)
    rx2 = rx + max(0, rw)
    ry2 = ry + max(0, rh)
    ix0 = max(lx, rx)
    iy0 = max(ly, ry)
    ix1 = min(lx2, rx2)
    iy1 = min(ly2, ry2)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    intersection = float(iw * ih)
    union = float(max(0, lw) * max(0, lh) + max(0, rw) * max(0, rh)) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _dedupe_pin_candidates(
    candidates: list[PinCenterCandidate],
) -> tuple[PinCenterCandidate, ...]:
    selected: list[PinCenterCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for existing in selected:
            same_center = (
                abs(float(candidate.center_xy[0]) - float(existing.center_xy[0]))
                <= max(6.0, min(float(candidate.width_px), float(existing.width_px)) * 0.25)
                and abs(float(candidate.center_xy[1]) - float(existing.center_xy[1]))
                <= max(4.0, min(float(candidate.height_px), float(existing.height_px)) * 0.75)
            )
            if same_center or _bbox_iou(candidate.bbox_xywh, existing.bbox_xywh) >= 0.35:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return tuple(selected)


def _find_pin_inner_strip_candidates(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> tuple[PinCenterCandidate, ...]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    values = gray[valid_mask]
    if values.size == 0:
        return ()

    crop_h, crop_w = gray.shape[:2]
    min_width = max(float(config.min_width_px), min(80.0, max(18.0, float(crop_w) * 0.12)))
    if crop_h <= 180:
        max_height = max(8.0, min(52.0, float(crop_h) * 0.50))
    else:
        max_height = max(8.0, min(42.0, float(crop_h) * 0.35))
    min_area = max(float(config.min_area_px), min_width * 3.0)
    open_w = max(7, int(round(min_width * 0.35)))
    if open_w % 2 == 0:
        open_w += 1
    close_w = max(3, int(round(float(crop_w) * 0.012)))
    if close_w % 2 == 0:
        close_w += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))

    thresholds: list[float] = []
    for percentile in (99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0):
        thresholds.append(float(np.percentile(values, percentile)))
    thresholds.append(float(rough_threshold))
    unique_thresholds = sorted(
        {
            float(max(1.0, min(254.0, threshold)))
            for threshold in thresholds
            if math.isfinite(float(threshold))
        },
        reverse=True,
    )

    ox, oy = origin
    candidates: list[PinCenterCandidate] = []
    for threshold in unique_thresholds:
        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(gray >= threshold) & valid_mask] = 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary[~valid_mask] = 0
        contours, _hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
            if w <= 1 or h <= 1:
                continue
            width = float(w)
            height = float(h)
            if width < min_width or height < float(config.min_height_px):
                continue
            if height > max_height:
                continue
            if width > float(crop_w) * 0.90:
                continue
            aspect = width / max(1.0, height)
            if aspect < max(1.8, float(config.min_aspect_ratio) * 0.85):
                continue
            area = max(float(abs(cv2.contourArea(contour))), width * height * 0.35)
            if area < min_area:
                continue
            fill = area / max(1.0, width * height)
            if fill < 0.22:
                continue
            abs_left = float(x + ox)
            abs_right = float(x + w - 1 + ox)
            abs_top = float(y + oy)
            abs_bottom = float(y + h - 1 + oy)
            center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
            score = area * min(aspect, 20.0) * max(0.25, fill) * (width / max(1.0, min_width))
            candidates.append(
                PinCenterCandidate(
                    center_xy=(float(center[0]), float(center[1])),
                    box_points=(
                        (abs_left, abs_top),
                        (abs_right, abs_top),
                        (abs_right, abs_bottom),
                        (abs_left, abs_bottom),
                    ),
                    area_px=float(area),
                    bbox_xywh=(int(round(abs_left)), int(round(abs_top)), w, h),
                    width_px=width,
                    height_px=height,
                    aspect_ratio=float(aspect),
                    angle_deg=0.0,
                    score=float(score),
                )
            )
    return _dedupe_pin_candidates(candidates)


def _find_bright_vertical_block_candidates(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> tuple[PinCenterCandidate, ...]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    values = gray[valid_mask]
    if values.size == 0:
        return ()

    crop_h, crop_w = gray.shape[:2]
    small_roi = crop_w <= 180 and crop_h <= 320
    min_height = max(float(config.min_height_px), min(90.0, max(14.0, float(crop_h) * 0.055)))
    if small_roi:
        max_height = max(min_height + 4.0, min(220.0, float(crop_h) * 0.95))
        max_width = max(8.0, min(80.0, float(crop_w) * 0.75))
    else:
        max_height = max(min_height + 4.0, min(110.0, float(crop_h) * 0.28))
        max_width = max(8.0, min(55.0, float(crop_w) * 0.12))
    min_area = max(float(config.min_area_px), min_height * 3.0)
    open_h = max(7, int(round(min_height * 0.35)))
    if open_h % 2 == 0:
        open_h += 1
    close_h = max(3, int(round(float(crop_h) * 0.010)))
    if close_h % 2 == 0:
        close_h += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_h))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))

    thresholds: list[float] = []
    for percentile in (99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0):
        thresholds.append(float(np.percentile(values, percentile)))
    thresholds.append(float(rough_threshold))
    unique_thresholds = sorted(
        {
            float(max(1.0, min(254.0, threshold)))
            for threshold in thresholds
            if math.isfinite(float(threshold))
        },
        reverse=True,
    )

    ox, oy = origin
    candidates: list[PinCenterCandidate] = []
    for threshold in unique_thresholds:
        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(gray >= threshold) & valid_mask] = 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary[~valid_mask] = 0
        contours, _hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
            if w <= 1 or h <= 1:
                continue
            width = float(w)
            height = float(h)
            if height < min_height or width < 2.0:
                continue
            if height > max_height or width > max_width:
                continue
            aspect = height / max(1.0, width)
            if aspect < max(1.25, float(config.min_aspect_ratio) * 0.70):
                continue
            area = max(float(abs(cv2.contourArea(contour))), width * height * 0.35)
            if area < min_area:
                continue
            fill = area / max(1.0, width * height)
            if fill < 0.18:
                continue
            abs_left = float(x + ox)
            abs_right = float(x + w - 1 + ox)
            abs_top = float(y + oy)
            abs_bottom = float(y + h - 1 + oy)
            center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
            score = area * min(aspect, 20.0) * max(0.25, fill) * (height / max(1.0, min_height))
            candidates.append(
                PinCenterCandidate(
                    center_xy=(float(center[0]), float(center[1])),
                    box_points=(
                        (abs_left, abs_top),
                        (abs_right, abs_top),
                        (abs_right, abs_bottom),
                        (abs_left, abs_bottom),
                    ),
                    area_px=float(area),
                    bbox_xywh=(int(round(abs_left)), int(round(abs_top)), w, h),
                    width_px=width,
                    height_px=height,
                    aspect_ratio=float(aspect),
                    angle_deg=90.0,
                    score=float(score),
                )
            )
        col_counts = np.count_nonzero(binary > 0, axis=0)
        min_col_hits = max(4, int(round(min_height * 0.45)))
        for col_left, col_right in _runs_from_indices(np.where(col_counts >= min_col_hits)[0]):
            if col_right < col_left:
                continue
            segments: list[tuple[int, int]] = []
            run_width = col_right - col_left + 1
            if float(run_width) <= max_width:
                segments.append((col_left, col_right))
            else:
                edge_width = max(3, int(round(max_width)))
                segments.append((col_left, min(col_right, col_left + edge_width - 1)))
                segments.append((max(col_left, col_right - edge_width + 1), col_right))
            for seg_left, seg_right in segments:
                if seg_right <= seg_left:
                    continue
                segment = binary[:, seg_left:seg_right + 1] > 0
                seg_width = float(seg_right - seg_left + 1)
                min_row_hits = max(2, int(round(seg_width * 0.35)))
                row_counts = np.count_nonzero(segment, axis=1)
                row_indices = np.where(row_counts >= min_row_hits)[0]
                for row_top, row_bottom in _runs_from_indices(row_indices):
                    height = float(row_bottom - row_top + 1)
                    if height < min_height or height > max_height:
                        continue
                    local_points = np.argwhere(segment[row_top:row_bottom + 1, :])
                    if local_points.size == 0:
                        continue
                    ys = local_points[:, 0] + row_top
                    xs = local_points[:, 1] + seg_left
                    x0_i = int(xs.min())
                    x1_i = int(xs.max())
                    y0_i = int(ys.min())
                    y1_i = int(ys.max())
                    width = float(x1_i - x0_i + 1)
                    height = float(y1_i - y0_i + 1)
                    if width > max_width or height < min_height or height > max_height:
                        continue
                    aspect = height / max(1.0, width)
                    if aspect < max(1.15, float(config.min_aspect_ratio) * 0.65):
                        continue
                    area = float(local_points.shape[0])
                    fill = area / max(1.0, width * height)
                    if fill < 0.18:
                        continue
                    abs_left = float(x0_i + ox)
                    abs_right = float(x1_i + ox)
                    abs_top = float(y0_i + oy)
                    abs_bottom = float(y1_i + oy)
                    center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
                    score = area * min(aspect, 20.0) * max(0.25, fill) * 0.75
                    candidates.append(
                        PinCenterCandidate(
                            center_xy=(float(center[0]), float(center[1])),
                            box_points=(
                                (abs_left, abs_top),
                                (abs_right, abs_top),
                                (abs_right, abs_bottom),
                                (abs_left, abs_bottom),
                            ),
                            area_px=float(area),
                            bbox_xywh=(
                                int(round(abs_left)),
                                int(round(abs_top)),
                                int(round(width)),
                                int(round(height)),
                            ),
                            width_px=width,
                            height_px=height,
                            aspect_ratio=float(aspect),
                            angle_deg=90.0,
                            score=float(score),
                        )
                    )
    return _dedupe_pin_candidates(candidates)


def _run_containing_or_nearest(indices: np.ndarray, target: float) -> tuple[int, int] | None:
    values = [int(v) for v in np.asarray(indices, dtype=np.int32).reshape(-1)]
    if not values:
        return None
    runs: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if int(value) != prev + 1:
            runs.append((start, prev))
            start = int(value)
        prev = int(value)
    runs.append((start, prev))
    target_value = float(target)
    for lo, hi in runs:
        if float(lo) <= target_value <= float(hi):
            return lo, hi
    return min(
        runs,
        key=lambda run: min(abs(float(run[0]) - target_value), abs(float(run[1]) - target_value)),
    )


def _runs_from_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    values = [int(v) for v in np.asarray(indices, dtype=np.int32).reshape(-1)]
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if int(value) != prev + 1:
            runs.append((start, prev))
            start = int(value)
        prev = int(value)
    runs.append((start, prev))
    return runs


def _pin_local_threshold(gray: np.ndarray, valid: np.ndarray, rough_threshold: float) -> float:
    values = np.asarray(gray, dtype=np.float32)[np.asarray(valid, dtype=bool)]
    if values.size == 0:
        return max(1.0, float(rough_threshold) * 0.7)
    if float(values.max() - values.min()) < 3.0:
        return max(1.0, min(254.0, float(values.mean())))
    low = float(np.percentile(values, 20.0))
    high = float(np.percentile(values, 95.0))
    contrast_threshold = low + (high - low) * 0.35
    values_u8 = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _binary = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = min(float(otsu), float(contrast_threshold))
    if float(rough_threshold) > 1.0:
        threshold = min(threshold, float(rough_threshold) * 0.5)
    return float(max(1.0, min(254.0, threshold)))


def _refine_pin_candidate_from_local_edges(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> PinCenterCandidate:
    if not config.refine_center:
        return candidate
    crop_h, crop_w = np.asarray(mask).shape[:2]
    if crop_w <= 1 or crop_h <= 1:
        return candidate

    ox, oy = origin
    bx_abs, by_abs, bw, bh = candidate.bbox_xywh
    bx = int(round(float(bx_abs) - float(ox)))
    by = int(round(float(by_abs) - float(oy)))
    bw = max(1, int(bw))
    bh = max(1, int(bh))
    pad_x = int(round(max(4.0, float(bw) * float(config.refine_expand_x_ratio))))
    pad_y = int(round(max(6.0, float(bh) * float(config.refine_expand_y_ratio))))
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(crop_w, bx + bw + pad_x)
    y1 = min(crop_h, by + bh + pad_y)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return candidate

    local_bgr = np.asarray(crop_bgr, dtype=np.uint8)[y0:y1, x0:x1]
    local_mask = np.asarray(mask, dtype=np.uint8)[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(local_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    threshold = _pin_local_threshold(gray, local_mask, rough_threshold)
    local_binary = np.zeros(gray.shape, dtype=np.uint8)
    local_binary[(gray >= threshold) & local_mask] = 255
    if config.morph_close_size >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(config.morph_close_size), int(config.morph_close_size)),
        )
        local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_CLOSE, kernel)
    local_binary[~local_mask] = 0

    rough_cx = float(candidate.center_xy[0]) - float(ox) - float(x0)
    rough_cy = float(candidate.center_xy[1]) - float(oy) - float(y0)
    core_x0 = max(0, bx - x0)
    core_x1 = min(local_binary.shape[1], bx + bw - x0)
    if core_x1 <= core_x0 + 1:
        core_x0, core_x1 = 0, local_binary.shape[1]
    core = local_binary[:, core_x0:core_x1] > 0
    core_valid = local_mask[:, core_x0:core_x1]
    valid_counts = np.count_nonzero(core_valid, axis=1)
    hit_counts = np.count_nonzero(core & core_valid, axis=1)
    row_ratios = hit_counts / np.maximum(valid_counts, 1)
    min_row_hits = max(2, int(round(max(1, core_x1 - core_x0) * float(config.refine_min_fill_ratio))))
    row_indices = np.where((hit_counts >= min_row_hits) | (row_ratios >= float(config.refine_min_fill_ratio)))[0]
    row_run = _run_containing_or_nearest(row_indices, rough_cy)
    if row_run is None:
        return candidate
    top, bottom = row_run
    if bottom <= top:
        return candidate

    y_slice = slice(max(0, top), min(local_binary.shape[0], bottom + 1))
    body = local_binary[y_slice, :] > 0
    body_valid = local_mask[y_slice, :]
    valid_col_counts = np.count_nonzero(body_valid, axis=0)
    hit_col_counts = np.count_nonzero(body & body_valid, axis=0)
    col_ratios = hit_col_counts / np.maximum(valid_col_counts, 1)
    min_col_hits = max(2, int(round(max(1, bottom - top + 1) * float(config.refine_min_fill_ratio))))
    col_indices = np.where((hit_col_counts >= min_col_hits) | (col_ratios >= float(config.refine_min_fill_ratio)))[0]
    col_run = _run_containing_or_nearest(col_indices, rough_cx)
    if col_run is None:
        return candidate
    left, right = col_run
    if right <= left:
        return candidate

    width = float(right - left + 1)
    height = float(bottom - top + 1)
    if width < float(config.min_width_px) or height < float(config.min_height_px):
        return candidate
    aspect = max(width, height) / max(1.0, min(width, height))
    if config.target_orientation == "horizontal" and width / max(height, 1.0) < max(1.0, config.min_aspect_ratio * 0.65):
        return candidate
    if config.target_orientation == "vertical" and height / max(width, 1.0) < max(1.0, config.min_aspect_ratio * 0.65):
        return candidate

    abs_left = float(x0 + left + ox)
    abs_right = float(x0 + right + ox)
    abs_top = float(y0 + top + oy)
    abs_bottom = float(y0 + bottom + oy)
    center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
    box_points = (
        (abs_left, abs_top),
        (abs_right, abs_top),
        (abs_right, abs_bottom),
        (abs_left, abs_bottom),
    )
    area = float(width * height)
    return PinCenterCandidate(
        center_xy=(float(center[0]), float(center[1])),
        box_points=box_points,
        area_px=area,
        bbox_xywh=(
            int(round(abs_left)),
            int(round(abs_top)),
            max(1, int(round(width))),
            max(1, int(round(height))),
        ),
        width_px=float(width),
        height_px=float(height),
        aspect_ratio=float(aspect),
        angle_deg=0.0,
        score=float(max(candidate.score, area * min(aspect, 20.0))),
    )


def _refine_pin_candidate_to_inner_bright_strip(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> PinCenterCandidate:
    if not config.refine_center:
        return candidate
    crop_h, crop_w = np.asarray(mask).shape[:2]
    if crop_w <= 1 or crop_h <= 1:
        return candidate

    ox, oy = origin
    bx_abs, by_abs, bw, bh = candidate.bbox_xywh
    bx = int(round(float(bx_abs) - float(ox)))
    by = int(round(float(by_abs) - float(oy)))
    bw = max(1, int(bw))
    bh = max(1, int(bh))
    pad_x = int(round(max(4.0, float(bw) * max(0.15, float(config.refine_expand_x_ratio)))))
    pad_y = int(round(max(8.0, float(bh) * max(1.5, float(config.refine_expand_y_ratio)))))
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(crop_w, bx + bw + pad_x)
    y1 = min(crop_h, by + bh + pad_y)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return candidate

    local_bgr = np.asarray(crop_bgr, dtype=np.uint8)[y0:y1, x0:x1]
    local_mask = np.asarray(mask, dtype=np.uint8)[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(local_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = gray[local_mask]
    if values.size == 0:
        return candidate
    rough_cx = float(candidate.center_xy[0]) - float(ox) - float(x0)
    rough_cy = float(candidate.center_xy[1]) - float(oy) - float(y0)
    preferred_cy = max(
        rough_cy,
        float(by - y0) + (float(bh) - 1.0) * float(config.inner_strip_y_bias),
    )
    min_width = max(float(config.min_width_px), float(bw) * float(config.inner_strip_min_width_ratio))
    local_h, local_w = gray.shape[:2]
    core_x0 = max(0, bx - x0)
    core_x1 = min(local_w, bx + bw - x0)
    if core_x1 <= core_x0 + 1:
        core_x0, core_x1 = 0, local_w

    def make_local_binary(threshold: float) -> np.ndarray:
        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(gray >= float(threshold)) & local_mask] = 255
        close_w = max(3, int(round(float(bw) * 0.035)))
        if close_w % 2 == 0:
            close_w += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary[~local_mask] = 0
        return binary

    lower_cy = float(by - y0) + float(bh) * 0.30
    upper_cy = float(by - y0) + float(bh) * 1.35
    min_overlap = max(4.0, min_width * 0.35)
    threshold_candidates: list[float] = []
    for percentile in (98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0):
        threshold_candidates.append(float(np.percentile(values, percentile)))
    threshold_candidates.append(float(rough_threshold))
    unique_thresholds = sorted(
        {
            float(max(1.0, min(254.0, threshold)))
            for threshold in threshold_candidates
            if math.isfinite(float(threshold))
        },
        reverse=True,
    )

    for threshold in unique_thresholds:
        local_binary = make_local_binary(threshold)
        contours, _hierarchy = cv2.findContours(
            local_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        best: PinCenterCandidate | None = None
        best_score = -math.inf
        for contour in contours:
            left, top, width_i, height_i = [int(v) for v in cv2.boundingRect(contour)]
            if width_i <= 1 or height_i <= 1:
                continue
            width = float(width_i)
            height = float(height_i)
            if width < min_width or height < float(config.min_height_px):
                continue
            if height > max(42.0, float(bh) * 0.85):
                continue
            aspect = width / max(height, 1.0)
            if aspect < max(1.4, float(config.min_aspect_ratio) * 0.75):
                continue
            cx = float(left) + (width - 1.0) * 0.5
            cy = float(top) + (height - 1.0) * 0.5
            if cy < lower_cy or cy > upper_cy:
                continue
            overlap = float(max(0, min(left + width_i, core_x1) - max(left, core_x0)))
            if overlap < min(min_overlap, width * 0.7):
                continue
            area = max(float(abs(cv2.contourArea(contour))), width * height * 0.35)
            dx_penalty = abs(cx - rough_cx) / max(1.0, float(local_w))
            dy_penalty = abs(cy - preferred_cy) / max(1.0, float(local_h))
            fill = area / max(1.0, width * height)
            score = area * min(aspect, 20.0) * max(0.25, fill)
            score *= 1.0 - min(0.9, dx_penalty + dy_penalty * 1.75)
            score *= 1.0 + min(1.0, overlap / max(1.0, width)) * 0.35
            if score <= best_score:
                continue
            abs_left = float(x0 + left + ox)
            abs_right = float(x0 + left + width_i - 1 + ox)
            abs_top = float(y0 + top + oy)
            abs_bottom = float(y0 + top + height_i - 1 + oy)
            center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
            best = PinCenterCandidate(
                center_xy=(float(center[0]), float(center[1])),
                box_points=(
                    (abs_left, abs_top),
                    (abs_right, abs_top),
                    (abs_right, abs_bottom),
                    (abs_left, abs_bottom),
                ),
                area_px=float(area),
                bbox_xywh=(
                    int(round(abs_left)),
                    int(round(abs_top)),
                    max(1, int(round(width))),
                    max(1, int(round(height))),
                ),
                width_px=float(width),
                height_px=float(height),
                aspect_ratio=float(aspect),
                angle_deg=0.0,
                score=float(max(candidate.score, score)),
            )
            best_score = score
        if best is not None:
            return best

    threshold = max(1.0, min(254.0, float(np.percentile(values, 90.0))))
    local_binary = make_local_binary(threshold)

    best: PinCenterCandidate | None = None
    best_score = -math.inf
    core = local_binary[:, core_x0:core_x1] > 0
    core_valid = local_mask[:, core_x0:core_x1]
    valid_counts = np.count_nonzero(core_valid, axis=1)
    hit_counts = np.count_nonzero(core & core_valid, axis=1)
    min_row_hits = max(2, int(round(min_width * 0.72)))
    row_indices = np.where(hit_counts >= min_row_hits)[0]

    for top, bottom in _runs_from_indices(row_indices):
        if bottom < top:
            continue
        y_slice = slice(max(0, top), min(local_h, bottom + 1))
        body = local_binary[y_slice, :] > 0
        body_valid = local_mask[y_slice, :]
        valid_col_counts = np.count_nonzero(body_valid, axis=0)
        hit_col_counts = np.count_nonzero(body & body_valid, axis=0)
        min_col_hits = max(1, int(round(max(1, bottom - top + 1) * 0.45)))
        col_indices = np.where((hit_col_counts >= min_col_hits) & (valid_col_counts > 0))[0]
        col_run = _run_containing_or_nearest(col_indices, rough_cx)
        if col_run is None:
            continue
        left, right = col_run
        if right <= left:
            continue
        width = float(right - left + 1)
        height = float(bottom - top + 1)
        if width < min_width:
            continue
        aspect = width / max(height, 1.0)
        if aspect < max(1.2, float(config.min_aspect_ratio) * 0.65):
            continue
        if height > max(float(bh) * 1.8, 18.0):
            continue
        cx = float(left) + (width - 1.0) * 0.5
        cy = float(top) + (height - 1.0) * 0.5
        dx_penalty = abs(cx - rough_cx) / max(1.0, float(local_w))
        dy_penalty = abs(cy - preferred_cy) / max(1.0, float(local_h))
        upper_penalty = max(0.0, preferred_cy - cy) / max(1.0, float(local_h))
        lower_bonus = 1.0 + 6.0 * max(0.0, min(1.0, cy / max(1.0, float(local_h - 1))))
        area = float(width * height)
        score = area * min(aspect, 20.0) * lower_bonus
        score *= 1.0 - min(0.95, dx_penalty + dy_penalty * 2.0 + upper_penalty * 5.0)
        if score <= best_score:
            continue
        abs_left = float(x0 + left + ox)
        abs_right = float(x0 + right + ox)
        abs_top = float(y0 + top + oy)
        abs_bottom = float(y0 + bottom + oy)
        center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
        best = PinCenterCandidate(
            center_xy=(float(center[0]), float(center[1])),
            box_points=(
                (abs_left, abs_top),
                (abs_right, abs_top),
                (abs_right, abs_bottom),
                (abs_left, abs_bottom),
            ),
            area_px=area,
            bbox_xywh=(
                int(round(abs_left)),
                int(round(abs_top)),
                max(1, int(round(width))),
                max(1, int(round(height))),
            ),
            width_px=width,
            height_px=height,
            aspect_ratio=float(aspect),
            angle_deg=0.0,
            score=float(max(candidate.score, score)),
        )
        best_score = score

    return best if best is not None else candidate


def _select_pin_center_pair(
    candidates: tuple[PinCenterCandidate, ...],
    *,
    config: PinCenterDistanceConfig,
    roi_size: tuple[int, int],
) -> tuple[PinCenterCandidate, PinCenterCandidate]:
    if len(candidates) < 2:
        raise RuntimeError(f"pin centers not found: {len(candidates)}/2")
    shortlist = tuple(candidates[: min(len(candidates), 24)])
    axis_index = 0 if config.sort_axis == "x" else 1
    axis_span = float(max(1, roi_size[0] if axis_index == 0 else roi_size[1]))
    best_pair: tuple[PinCenterCandidate, PinCenterCandidate] | None = None
    best_score = -math.inf
    for idx, left in enumerate(shortlist):
        for right in shortlist[idx + 1:]:
            separation = abs(float(right.center_xy[axis_index]) - float(left.center_xy[axis_index]))
            min_separation = _pin_pair_min_separation(
                left,
                right,
                axis_span=axis_span,
                axis_index=axis_index,
                config=config,
            )
            if separation < min_separation:
                continue
            normalized_separation = separation / axis_span
            pair_score = (float(left.score) + float(right.score)) * (
                1.0 + normalized_separation * normalized_separation * 4.0
            )
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (left, right)
    if best_pair is None:
        min_separation = axis_span * float(config.min_pair_separation_ratio)
        raise RuntimeError(
            f"pin center pair separation too small: need >= {min_separation:.1f}px"
        )
    return tuple(sorted(best_pair, key=lambda item: item.center_xy[axis_index]))  # type: ignore[return-value]


def _pin_pair_min_separation(
    left: PinCenterCandidate,
    right: PinCenterCandidate,
    *,
    axis_span: float,
    axis_index: int,
    config: PinCenterDistanceConfig,
) -> float:
    ratio_separation = float(axis_span) * float(config.min_pair_separation_ratio)
    if axis_index == 1:
        size_reference = max(float(left.width_px), float(right.width_px))
    else:
        size_reference = max(float(left.height_px), float(right.height_px))
    size_separation = size_reference * float(config.min_pair_separation_size_ratio)
    return max(float(ratio_separation), float(size_separation))


def _pin_center_distance_px(
    center_a: tuple[float, float],
    center_b: tuple[float, float],
    distance_mode: str,
) -> float:
    dx = float(center_b[0]) - float(center_a[0])
    dy = float(center_b[1]) - float(center_a[1])
    if distance_mode == "horizontal":
        return abs(dx)
    if distance_mode == "vertical":
        return abs(dy)
    return float(math.hypot(dx, dy))


def _select_bright_block_y_pair(
    vertical_candidates: tuple[PinCenterCandidate, ...],
    horizontal_candidates: tuple[PinCenterCandidate, ...],
    *,
    roi_size: tuple[int, int],
) -> tuple[PinCenterCandidate, PinCenterCandidate]:
    if not vertical_candidates:
        raise RuntimeError("bright vertical block not found")
    if not horizontal_candidates:
        raise RuntimeError("bright horizontal block not found")
    roi_w = float(max(1, int(roi_size[0])))
    min_x_separation = max(12.0, roi_w * 0.12)
    best_pair: tuple[PinCenterCandidate, PinCenterCandidate] | None = None
    best_score = -math.inf
    for vertical in vertical_candidates[:16]:
        for horizontal in horizontal_candidates[:16]:
            x_separation = float(horizontal.center_xy[0]) - float(vertical.center_xy[0])
            if x_separation < min_x_separation:
                continue
            y_delta = abs(float(horizontal.center_xy[1]) - float(vertical.center_xy[1]))
            score = (float(vertical.score) + float(horizontal.score)) * (1.0 + x_separation / roi_w)
            score *= 1.0 + min(1.0, y_delta / max(1.0, float(roi_size[1]))) * 0.25
            if score > best_score:
                best_score = score
                best_pair = (vertical, horizontal)
    if best_pair is None:
        raise RuntimeError(f"bright block pair x separation too small: need >= {min_x_separation:.1f}px")
    return best_pair


def _normalized_block_orientation(value: object) -> str:
    orientation = str(value or "auto").strip().lower()
    if orientation in {"h", "x", "strip"}:
        orientation = "horizontal"
    elif orientation in {"v", "y", "block"}:
        orientation = "vertical"
    if orientation not in {"auto", "horizontal", "vertical"}:
        orientation = "auto"
    return orientation


def _center_candidate_score(
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    roi_size: tuple[int, int],
) -> float:
    ox, oy = origin
    roi_w, roi_h = max(1.0, float(roi_size[0])), max(1.0, float(roi_size[1]))
    roi_cx = float(ox) + roi_w * 0.5
    roi_cy = float(oy) + roi_h * 0.5
    dx = abs(float(candidate.center_xy[0]) - roi_cx) / max(1.0, roi_w * 0.5)
    dy = abs(float(candidate.center_xy[1]) - roi_cy) / max(1.0, roi_h * 0.5)
    center_penalty = min(0.88, (dx + dy) * 0.28)
    return float(candidate.score) * (1.0 - center_penalty)


def _select_bright_block_center_candidate(
    horizontal_candidates: tuple[PinCenterCandidate, ...],
    vertical_candidates: tuple[PinCenterCandidate, ...],
    *,
    orientation: str,
    origin: tuple[int, int],
    roi_size: tuple[int, int],
) -> tuple[PinCenterCandidate, tuple[PinCenterCandidate, ...]]:
    if orientation == "horizontal":
        candidates = tuple(horizontal_candidates)
    elif orientation == "vertical":
        candidates = tuple(vertical_candidates)
    else:
        candidates = _dedupe_pin_candidates([*horizontal_candidates, *vertical_candidates])
    if not candidates:
        raise RuntimeError("bright block center not found")
    selected = max(
        candidates,
        key=lambda candidate: _center_candidate_score(candidate, origin=origin, roi_size=roi_size),
    )
    return selected, candidates


def _bright_block_center_has_adjacent_body(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    params: Mapping[str, Any] | None = None,
) -> bool:
    payload = dict(params or {})
    if not _bool_param(payload.get("require_adjacent_body"), default=True):
        return True
    if float(candidate.height_px) <= float(candidate.width_px) * 1.6:
        return True

    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    crop_h, crop_w = gray.shape[:2]
    valid_area = int(np.count_nonzero(valid_mask))
    if (
        valid_area > 0
        and float(candidate.height_px) >= max(40.0, float(crop_h) * 0.45)
        and float(candidate.width_px) >= max(10.0, float(crop_w) * 0.20)
        and float(candidate.area_px) >= max(900.0, float(valid_area) * 0.18)
    ):
        return True

    ox, oy = origin
    bx = int(round(float(candidate.bbox_xywh[0]) - float(ox)))
    by = int(round(float(candidate.bbox_xywh[1]) - float(oy)))
    bw = max(1, int(candidate.bbox_xywh[2]))
    bh = max(1, int(candidate.bbox_xywh[3]))
    x0 = max(0, bx + bw + 2)
    x1 = min(crop_w, int(round(float(crop_w))))
    y0 = max(0, int(round(float(by) + float(bh) * 0.20)))
    y1 = min(crop_h, int(round(float(by) + float(bh) * 0.80)))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return True

    patch = gray[y0:y1, x0:x1]
    patch_valid = valid_mask[y0:y1, x0:x1]
    values = patch[patch_valid]
    if values.size < 30:
        return True

    bright_threshold = int(max(1, min(255, float(payload.get("adjacent_body_threshold", 200.0) or 200.0))))
    min_pixels = int(max(1, float(payload.get("adjacent_body_min_pixels", 24.0) or 24.0)))
    min_ratio = max(0.0, min(1.0, float(payload.get("adjacent_body_min_ratio", 0.012) or 0.012)))
    bright_count = int(np.count_nonzero(values >= bright_threshold))
    required = max(min_pixels, int(round(float(values.size) * min_ratio)))
    return bright_count >= required


def measure_bright_block_center_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> BrightBlockCenterResult:
    payload = dict(params or {})
    orientation = _normalized_block_orientation(
        payload.get("block_orientation", payload.get("target_orientation", "auto"))
    )
    config = PinCenterDistanceConfig.from_params(
        {
            **payload,
            "target_orientation": "any",
            "distance_mode": "euclidean",
        },
        roi_label=preferred_label,
    )
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    _binary, threshold = _pin_center_binary(crop, mask, config)
    horizontal_candidates = (
        _find_pin_inner_strip_candidates(
            crop,
            mask,
            origin=origin,
            config=config,
            rough_threshold=threshold,
        )
        if orientation in {"auto", "horizontal"}
        else ()
    )
    vertical_candidates = (
        _find_bright_vertical_block_candidates(
            crop,
            mask,
            origin=origin,
            config=config,
            rough_threshold=threshold,
        )
        if orientation in {"auto", "vertical"}
        else ()
    )
    candidate, candidates = _select_bright_block_center_candidate(
        horizontal_candidates,
        vertical_candidates,
        orientation=orientation,
        origin=origin,
        roi_size=(int(crop.shape[1]), int(crop.shape[0])),
    )
    if not _bright_block_center_has_adjacent_body(
        crop,
        mask,
        candidate,
        origin=origin,
        params=payload,
    ):
        raise RuntimeError("bright block adjacent metal body not found")
    ox, oy = origin
    return BrightBlockCenterResult(
        roi_label=roi_label,
        center_xy=candidate.center_xy,
        candidate=candidate,
        candidates=candidates,
        threshold=float(threshold),
        orientation=orientation,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
    )


def measure_pin_center_distance_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
    config = PinCenterDistanceConfig.from_params(params, roi_label=preferred_label)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    binary, threshold = _pin_center_binary(crop, mask, config)
    direct_candidates = (
        _find_pin_inner_strip_candidates(
            crop,
            mask,
            origin=origin,
            config=config,
            rough_threshold=threshold,
        )
        if config.center_target == "inner_bright_strip"
        else ()
    )
    if len(direct_candidates) >= 2:
        candidates = direct_candidates
        pin_a, pin_b = _select_pin_center_pair(
            candidates,
            config=config,
            roi_size=(int(crop.shape[1]), int(crop.shape[0])),
        )
    else:
        candidates = _find_pin_center_candidates(binary, origin=origin, config=config)
        pin_a, pin_b = _select_pin_center_pair(
            candidates,
            config=config,
            roi_size=(int(crop.shape[1]), int(crop.shape[0])),
        )
        refine = (
            _refine_pin_candidate_from_local_edges
            if config.center_target == "metal_body"
            else _refine_pin_candidate_to_inner_bright_strip
        )
        pin_a = refine(crop, mask, pin_a, origin=origin, config=config, rough_threshold=threshold)
        pin_b = refine(crop, mask, pin_b, origin=origin, config=config, rough_threshold=threshold)
    axis_index = 0 if config.sort_axis == "x" else 1
    axis_span = float(max(1, int(crop.shape[1]) if axis_index == 0 else int(crop.shape[0])))
    final_separation = abs(float(pin_b.center_xy[axis_index]) - float(pin_a.center_xy[axis_index]))
    min_separation = _pin_pair_min_separation(
        pin_a,
        pin_b,
        axis_span=axis_span,
        axis_index=axis_index,
        config=config,
    )
    if final_separation < min_separation:
        raise RuntimeError(
            f"pin center pair separation too small after refinement: need >= {min_separation:.1f}px"
        )
    pin_a, pin_b = tuple(sorted((pin_a, pin_b), key=lambda item: item.center_xy[axis_index]))
    distance_px = _pin_center_distance_px(pin_a.center_xy, pin_b.center_xy, config.distance_mode)
    distance_mm = distance_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    ox, oy = origin
    return PinCenterDistanceResult(
        roi_label=roi_label,
        distance_px=float(distance_px),
        distance_mm=float(distance_mm) if distance_mm is not None else None,
        center_a=pin_a.center_xy,
        center_b=pin_b.center_xy,
        candidates=(pin_a, pin_b),
        threshold=float(threshold),
        distance_mode=config.distance_mode,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
    )


def measure_bright_block_y_distance_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
    config = PinCenterDistanceConfig.from_params(
        {
            **dict(params or {}),
            "distance_mode": "vertical",
            "sort_axis": "y",
            "target_orientation": "any",
        },
        roi_label=preferred_label,
    )
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    _binary, threshold = _pin_center_binary(crop, mask, config)
    horizontal_candidates = _find_pin_inner_strip_candidates(
        crop,
        mask,
        origin=origin,
        config=config,
        rough_threshold=threshold,
    )
    vertical_candidates = _find_bright_vertical_block_candidates(
        crop,
        mask,
        origin=origin,
        config=config,
        rough_threshold=threshold,
    )
    vertical, horizontal = _select_bright_block_y_pair(
        vertical_candidates,
        horizontal_candidates,
        roi_size=(int(crop.shape[1]), int(crop.shape[0])),
    )
    distance_px = abs(float(horizontal.center_xy[1]) - float(vertical.center_xy[1]))
    distance_mm = distance_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    dimension_x = float(horizontal.center_xy[0])
    dimension_segment = (
        (dimension_x, float(vertical.center_xy[1])),
        (dimension_x, float(horizontal.center_xy[1])),
    )
    ox, oy = origin
    return PinCenterDistanceResult(
        roi_label=roi_label,
        distance_px=float(distance_px),
        distance_mm=float(distance_mm) if distance_mm is not None else None,
        center_a=vertical.center_xy,
        center_b=horizontal.center_xy,
        candidates=(vertical, horizontal),
        threshold=float(threshold),
        distance_mode="vertical",
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
        measurement_type=BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
        dimension_segment=dimension_segment,
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


def measure_pin_center_distance(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
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
    return measure_pin_center_distance_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_bright_block_center(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> BrightBlockCenterResult:
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
    return measure_bright_block_center_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_bright_block_y_distance(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
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
    return measure_bright_block_y_distance_from_array(
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


def measurement_value(
    result: EdgeDistanceResult | FindLineMeasurementResult | PinCenterDistanceResult | BrightBlockCenterResult,
    algorithm: object,
) -> float:
    key = str(algorithm or "").strip().lower()
    if key == "edge_distance" and isinstance(result, EdgeDistanceResult):
        return float(result.distance_px)
    if key in FIND_LINE_ALGORITHMS and isinstance(result, FindLineMeasurementResult):
        return float(result.position_px)
    if key == PIN_CENTER_DISTANCE_ALGORITHM and isinstance(result, PinCenterDistanceResult):
        return float(result.distance_px)
    if key == BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM and isinstance(result, PinCenterDistanceResult):
        return float(result.distance_px)
    if key == BRIGHT_BLOCK_CENTER_ALGORITHM and isinstance(result, BrightBlockCenterResult):
        return float(result.center_xy[1])
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


def judge_pin_center_distance(
    result: PinCenterDistanceResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    config = PinCenterDistanceConfig.from_params(params, roi_label=result.roi_label)
    unit = config.limit_unit
    if unit == "mm":
        distance_mm = result.distance_mm
        if distance_mm is None and config.pixel_size_mm > 0.0:
            distance_mm = float(result.distance_px) * float(config.pixel_size_mm)
        if distance_mm is None:
            raise RuntimeError("pixel_size_mm is required when pin-center limits use mm")
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


def judge_bright_block_y_distance(
    result: PinCenterDistanceResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    return judge_pin_center_distance(result, params)


__all__ = [
    "MEASUREMENT_ALGORITHMS",
    "BRIGHT_BLOCK_CENTER_ALGORITHM",
    "BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM",
    "CENTER_DISTANCE_ALGORITHM",
    "CENTER_DISTANCE_ALGORITHMS",
    "FIND_LINE_ALGORITHM",
    "FIND_LINE_ALGORITHMS",
    "FIND_LINE_SUBPIX_ALGORITHM",
    "PIN_CENTER_DISTANCE_ALGORITHM",
    "LINE_DISTANCE_ALGORITHM",
    "LINE_DISTANCE_ALGORITHMS",
    "LINE_DISTANCE_REF_NORMAL_ALGORITHM",
    "EdgeDistanceConfig",
    "EdgeDistanceResult",
    "FindLineConfig",
    "FindLineMeasurementConfig",
    "FindLineMeasurementResult",
    "FittedLine",
    "BrightBlockCenterResult",
    "PinCenterCandidate",
    "PinCenterDistanceConfig",
    "PinCenterDistanceResult",
    "find_edge_points",
    "filter_line_points",
    "fit_line",
    "fit_line_filtered",
    "is_measurement_algorithm",
    "judge_edge_distance",
    "judge_bright_block_y_distance",
    "judge_find_line",
    "judge_pin_center_distance",
    "measure_bright_block_center",
    "measure_bright_block_center_from_array",
    "measure_bright_block_y_distance",
    "measure_bright_block_y_distance_from_array",
    "measure_edge_distance",
    "measure_edge_distance_from_array",
    "measure_find_line",
    "measure_find_line_from_array",
    "measure_pin_center_distance",
    "measure_pin_center_distance_from_array",
    "measurement_value",
]
