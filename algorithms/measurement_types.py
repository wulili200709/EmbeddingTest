from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


FIND_LINE_ALGORITHM = "find_line"
FIND_LINE_SUBPIX_ALGORITHM = "find_line_subpix"
FIND_LINE_ALGORITHMS = {FIND_LINE_ALGORITHM, FIND_LINE_SUBPIX_ALGORITHM}
PIN_CENTER_DISTANCE_ALGORITHM = "pin_center_distance"
BRIGHT_BLOCK_CENTER_ALGORITHM = "bright_block_center"
PIN_TIP_POINT_ALGORITHM = "pin_tip_point"
MULTI_PIN_TIP_HEIGHT_ALGORITHM = "multi_pin_tip_height"
BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM = "bright_block_y_distance"
LINE_DISTANCE_ALGORITHM = "line_distance"
LINE_DISTANCE_REF_NORMAL_ALGORITHM = "line_distance_ref_normal"
LINE_DISTANCE_ALGORITHMS = {LINE_DISTANCE_ALGORITHM, LINE_DISTANCE_REF_NORMAL_ALGORITHM}
POINT_LINE_DISTANCE_ALGORITHM = "point_line_distance"
CENTER_DISTANCE_ALGORITHM = "center_distance"
CENTER_DISTANCE_ALGORITHMS = {CENTER_DISTANCE_ALGORITHM}
MEASUREMENT_ALGORITHMS = [
    FIND_LINE_ALGORITHM,
    FIND_LINE_SUBPIX_ALGORITHM,
    PIN_TIP_POINT_ALGORITHM,
    MULTI_PIN_TIP_HEIGHT_ALGORITHM,
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    PIN_CENTER_DISTANCE_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    LINE_DISTANCE_ALGORITHM,
    LINE_DISTANCE_REF_NORMAL_ALGORITHM,
    POINT_LINE_DISTANCE_ALGORITHM,
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
class PinTipPointConfig:
    roi_label: str = ""
    threshold: float = 0.0
    blur_ksize: int = 3
    morph_open_size: int = 0
    morph_close_size: int = 3
    min_area_px: float = 80.0
    min_width_px: float = 4.0
    min_height_px: float = 20.0
    border_margin_px: int = 2
    tip_band_ratio: float = 0.22
    arc_depth_ratio: float = 0.55
    min_arc_points: int = 8

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None,
        *,
        roi_label: str = "",
    ) -> "PinTipPointConfig":
        payload = dict(params or {})
        blur_ksize = max(0, int(payload.get("blur_ksize", 3) or 0))
        if blur_ksize > 0 and blur_ksize % 2 == 0:
            blur_ksize += 1
        morph_open_size = max(0, int(payload.get("morph_open_size", 0) or 0))
        morph_close_size = max(0, int(payload.get("morph_close_size", 3) or 0))
        if morph_open_size > 0 and morph_open_size % 2 == 0:
            morph_open_size += 1
        if morph_close_size > 0 and morph_close_size % 2 == 0:
            morph_close_size += 1
        return cls(
            roi_label=str(payload.get("roi_label", roi_label) or roi_label or "").strip(),
            threshold=max(0.0, min(255.0, float(payload.get("threshold", 0.0) or 0.0))),
            blur_ksize=blur_ksize,
            morph_open_size=morph_open_size,
            morph_close_size=morph_close_size,
            min_area_px=max(1.0, float(payload.get("min_area_px", 80.0) or 1.0)),
            min_width_px=max(1.0, float(payload.get("min_width_px", 4.0) or 1.0)),
            min_height_px=max(2.0, float(payload.get("min_height_px", 20.0) or 2.0)),
            border_margin_px=max(0, int(payload.get("border_margin_px", 2) or 0)),
            tip_band_ratio=max(0.05, min(0.75, float(payload.get("tip_band_ratio", 0.22) or 0.22))),
            arc_depth_ratio=max(0.15, min(1.0, float(payload.get("arc_depth_ratio", 0.55) or 0.55))),
            min_arc_points=max(5, int(payload.get("min_arc_points", 8) or 5)),
        )


@dataclass(frozen=True)
class MultiPinTipHeightConfig:
    roi_label: str = ""
    expected_pin_count: int = 20
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: str = "px"
    pixel_size_mm: float = 0.0
    threshold: float = 0.0
    blur_ksize: int = 3
    morph_open_size: int = 0
    morph_close_size: int = 3
    min_area_px: float = 80.0
    min_width_px: float = 4.0
    min_height_px: float = 20.0
    border_margin_px: int = 2
    tip_band_ratio: float = 0.22
    arc_depth_ratio: float = 0.55
    min_arc_points: int = 8
    reference_edge_threshold: float = 18.0
    reference_scan_step: int = 2
    reference_min_points: int = 10
    reference_search_ratio: float = 0.65
    reference_cut_margin_px: float = 4.0

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None,
        *,
        roi_label: str = "",
    ) -> "MultiPinTipHeightConfig":
        payload = dict(params or {})
        point_config = PinTipPointConfig.from_params(payload, roi_label=roi_label)
        limit_unit = str(payload.get("limit_unit", "px") or "px").strip().lower()
        if limit_unit not in {"px", "mm"}:
            limit_unit = "px"
        return cls(
            roi_label=point_config.roi_label,
            expected_pin_count=max(1, min(200, int(payload.get("expected_pin_count", 20) or 20))),
            lower_limit=_optional_float(
                payload.get("lower_limit", payload.get(f"lower_limit_{limit_unit}"))
            ),
            upper_limit=_optional_float(
                payload.get("upper_limit", payload.get(f"upper_limit_{limit_unit}"))
            ),
            limit_unit=limit_unit,
            pixel_size_mm=max(0.0, float(payload.get("pixel_size_mm", 0.0) or 0.0)),
            threshold=point_config.threshold,
            blur_ksize=point_config.blur_ksize,
            morph_open_size=point_config.morph_open_size,
            morph_close_size=point_config.morph_close_size,
            min_area_px=point_config.min_area_px,
            min_width_px=point_config.min_width_px,
            min_height_px=point_config.min_height_px,
            border_margin_px=point_config.border_margin_px,
            tip_band_ratio=point_config.tip_band_ratio,
            arc_depth_ratio=point_config.arc_depth_ratio,
            min_arc_points=point_config.min_arc_points,
            reference_edge_threshold=max(
                1.0, float(payload.get("reference_edge_threshold", 18.0) or 18.0)
            ),
            reference_scan_step=max(1, int(payload.get("reference_scan_step", 2) or 2)),
            reference_min_points=max(5, int(payload.get("reference_min_points", 10) or 10)),
            reference_search_ratio=max(
                0.2, min(0.9, float(payload.get("reference_search_ratio", 0.65) or 0.65))
            ),
            reference_cut_margin_px=max(
                1.0, float(payload.get("reference_cut_margin_px", 4.0) or 4.0)
            ),
        )


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


@dataclass(frozen=True)
class PinTipPointResult:
    roi_label: str
    point_xy: tuple[float, float]
    axis_direction: tuple[float, float]
    threshold: float
    confidence: float
    fit_residual: float
    component_area_px: float
    component_bbox_xywh: tuple[int, int, int, int]
    edge_points: tuple[tuple[float, float], ...] = ()
    roi_xywh: tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> Dict[str, Any]:
        point = [float(self.point_xy[0]), float(self.point_xy[1])]
        x, y, width, height = self.component_bbox_xywh
        box_points = [
            [float(x), float(y)],
            [float(x + width), float(y)],
            [float(x + width), float(y + height)],
            [float(x), float(y + height)],
        ]
        return {
            "type": PIN_TIP_POINT_ALGORITHM,
            "roi_label": self.roi_label,
            "point": point,
            "point_xy": point,
            "center": point,
            "center_xy": point,
            "center_points": [point],
            "axis_direction": [float(self.axis_direction[0]), float(self.axis_direction[1])],
            "threshold": float(self.threshold),
            "confidence": float(self.confidence),
            "fit_residual": float(self.fit_residual),
            "component_area_px": float(self.component_area_px),
            "component_bbox_xywh": [int(v) for v in self.component_bbox_xywh],
            "box_points": box_points,
            "edge_points": [[float(px), float(py)] for px, py in self.edge_points],
            "roi_xywh": [int(v) for v in self.roi_xywh],
        }


@dataclass(frozen=True)
class MultiPinTipHeightResult:
    roi_label: str
    expected_pin_count: int
    tip_points: tuple[tuple[float, float], ...]
    distances_px: tuple[float, ...]
    reference_line: FittedLine
    reference_line_segment: tuple[tuple[float, float], tuple[float, float]]
    threshold: float
    fit_residuals: tuple[float, ...] = ()
    component_boxes_xywh: tuple[tuple[int, int, int, int], ...] = ()
    roi_xywh: tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> Dict[str, Any]:
        points = [[float(x), float(y)] for x, y in self.tip_points]
        return {
            "type": MULTI_PIN_TIP_HEIGHT_ALGORITHM,
            "roi_label": self.roi_label,
            "expected_pin_count": int(self.expected_pin_count),
            "detected_pin_count": int(len(self.tip_points)),
            "center_points": points,
            "tip_points": points,
            "distances_px": [float(value) for value in self.distances_px],
            "reference_line": self.reference_line.to_dict(),
            "reference_line_segment": [
                [float(x), float(y)] for x, y in self.reference_line_segment
            ],
            "line_segment": [
                [float(x), float(y)] for x, y in self.reference_line_segment
            ],
            "threshold": float(self.threshold),
            "fit_residuals": [float(value) for value in self.fit_residuals],
            "component_boxes_xywh": [
                [int(value) for value in box] for box in self.component_boxes_xywh
            ],
            "roi_xywh": [int(value) for value in self.roi_xywh],
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
    "PIN_TIP_POINT_ALGORITHM",
    "MULTI_PIN_TIP_HEIGHT_ALGORITHM",
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
    "POINT_LINE_DISTANCE_ALGORITHM",
    "BrightBlockCenterResult",
    "PinTipPointConfig",
    "PinTipPointResult",
    "MultiPinTipHeightConfig",
    "MultiPinTipHeightResult",
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
