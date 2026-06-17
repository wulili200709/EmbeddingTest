from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


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
    "BrightBlockCenterResult",
    "EdgeDistanceConfig",
    "EdgeDistanceResult",
    "FindLineConfig",
    "FindLineMeasurementConfig",
    "FindLineMeasurementResult",
    "FittedLine",
    "PinCenterCandidate",
    "PinCenterDistanceConfig",
    "PinCenterDistanceResult",
    "_bool_param",
    "_optional_float",
    "is_measurement_algorithm",
]
