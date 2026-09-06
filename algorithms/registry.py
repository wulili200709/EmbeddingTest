from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.algorithm_codes import (
    DEFAULT_LEARNING_BACKBONE,
    LEARNING_BACKBONES,
    LEGACY_SHARED_BACKBONE_ALGORITHM_CODES,
    SHARED_BACKBONE_ALGORITHM_CODE,
    is_learning_backbone_code,
    learning_backbone_storage_code,
    learning_backbone_storage_codes,
    normalize_tool_algorithm_code,
    storage_code_backbone,
)

from .measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHMS,
    FIND_LINE_ALGORITHMS,
    FIND_LINE_SUBPIX_ALGORITHM,
    LINE_DISTANCE_ALGORITHMS,
    MEASUREMENT_ALGORITHMS,
    MULTI_PIN_TIP_HEIGHT_ALGORITHM,
    PIN_CENTER_DISTANCE_ALGORITHM,
    PIN_TIP_POINT_ALGORITHM,
    POINT_LINE_DISTANCE_ALGORITHM,
)
from .traditional import TRADITIONAL_ALGORITHMS


@dataclass(frozen=True)
class ToolAlgorithmSpec:
    code: str
    display_name: str
    family: str
    fit_mode: str
    default_params: Dict[str, Any] = field(default_factory=dict)


_TOOL_ALGORITHM_SPECS: Dict[str, ToolAlgorithmSpec] = {
    SHARED_BACKBONE_ALGORITHM_CODE: ToolAlgorithmSpec(
        code=SHARED_BACKBONE_ALGORITHM_CODE,
        display_name="learning",
        family="learning",
        fit_mode="register",
        default_params={},
    ),
}
for _code in TRADITIONAL_ALGORITHMS:
    _TOOL_ALGORITHM_SPECS[_code] = ToolAlgorithmSpec(
        code=_code,
        display_name=_code,
        family="traditional",
        fit_mode="calibrate",
        default_params={},
    )
for _code in MEASUREMENT_ALGORITHMS:
    _TOOL_ALGORITHM_SPECS[_code] = ToolAlgorithmSpec(
        code=_code,
        display_name=_code,
        family="measurement",
        fit_mode="measure",
        default_params=(
            {
                "line": {
                    "direction": "left_right",
                    "edge_detector": "subpix_shen" if _code == FIND_LINE_SUBPIX_ALGORITHM else "canny",
                }
            }
            if _code in FIND_LINE_ALGORITHMS
            else {
                "line_a_item_id": "",
                "line_b_item_id": "",
                "limit_unit": "px",
            }
            if _code in LINE_DISTANCE_ALGORITHMS
            else {
                "point_item_id": "",
                "line_item_id": "",
                "limit_unit": "px",
            }
            if _code == POINT_LINE_DISTANCE_ALGORITHM
            else {
                "center_a_item_id": "",
                "center_b_item_id": "",
                "distance_mode": "vertical",
                "limit_unit": "px",
            }
            if _code in CENTER_DISTANCE_ALGORITHMS
            else {
                "threshold": 0.0,
                "blur_ksize": 3,
                "morph_open_size": 0,
                "morph_close_size": 3,
                "min_area_px": 80.0,
                "min_width_px": 4.0,
                "min_height_px": 20.0,
                "border_margin_px": 2,
                "tip_band_ratio": 0.22,
                "arc_depth_ratio": 0.55,
                "min_arc_points": 8,
            }
            if _code == PIN_TIP_POINT_ALGORITHM
            else {
                "expected_pin_count": 20,
                "height_check_enabled": True,
                "spacing_check_enabled": False,
                "spacing_specs": [],
                "reference_line_item_id": "",
                "limit_unit": "px",
                "threshold": 0.0,
                "blur_ksize": 3,
                "morph_open_size": 0,
                "morph_close_size": 3,
                "min_area_px": 80.0,
                "min_width_px": 4.0,
                "min_height_px": 20.0,
                "border_margin_px": 2,
                "tip_band_ratio": 0.22,
                "arc_depth_ratio": 0.55,
                "min_arc_points": 8,
                "reference_edge_threshold": 18.0,
                "reference_scan_step": 2,
                "reference_min_points": 10,
                "reference_search_ratio": 0.65,
                "reference_cut_margin_px": 4.0,
            }
            if _code == MULTI_PIN_TIP_HEIGHT_ALGORITHM
            else {
                "block_orientation": "auto",
                "limit_unit": "px",
                "threshold": 0.0,
                "min_area_px": 12.0,
                "min_aspect_ratio": 1.6,
                "target_orientation": "any",
                "center_target": "inner_bright_strip",
                "refine_center": True,
                "inner_strip_min_width_ratio": 0.45,
                "inner_strip_y_bias": 0.92,
                "require_adjacent_body": True,
                "adjacent_body_threshold": 200.0,
                "adjacent_body_min_pixels": 24.0,
                "adjacent_body_min_ratio": 0.012,
            }
            if _code == BRIGHT_BLOCK_CENTER_ALGORITHM
            else {
                "limit_unit": "px",
                "threshold": 0.0,
                "min_area_px": 12.0,
                "min_aspect_ratio": 1.6,
                "target_orientation": "horizontal",
                "distance_mode": "euclidean",
                "sort_axis": "y",
                "center_target": "inner_bright_strip",
                "refine_center": True,
                "refine_expand_y_ratio": 2.5,
                "inner_strip_min_width_ratio": 0.45,
                "inner_strip_y_bias": 0.92,
                "min_pair_separation_ratio": 0.12,
                "min_pair_separation_size_ratio": 1.2,
            }
            if _code in {PIN_CENTER_DISTANCE_ALGORITHM, BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM}
            else {
                "line_a": {"direction": "left_right"},
                "line_b": {"direction": "right_left"},
            }
        ),
    )

def get_tool_algorithm_spec(code: object) -> Optional[ToolAlgorithmSpec]:
    normalized = normalize_tool_algorithm_code(code)
    return _TOOL_ALGORITHM_SPECS.get(normalized)


def list_tool_algorithm_specs() -> List[ToolAlgorithmSpec]:
    return [
        _TOOL_ALGORITHM_SPECS[SHARED_BACKBONE_ALGORITHM_CODE],
        *[
            _TOOL_ALGORITHM_SPECS[code]
            for code in TRADITIONAL_ALGORITHMS
            if code in _TOOL_ALGORITHM_SPECS
        ],
        *[
            _TOOL_ALGORITHM_SPECS[code]
            for code in MEASUREMENT_ALGORITHMS
            if code in _TOOL_ALGORITHM_SPECS
        ],
    ]


def list_tool_algorithm_codes() -> List[str]:
    return [spec.code for spec in list_tool_algorithm_specs()]

def is_learning_tool_algorithm(code: object) -> bool:
    spec = get_tool_algorithm_spec(code)
    return bool(spec is not None and spec.family == "learning")


def is_traditional_tool_algorithm(code: object) -> bool:
    spec = get_tool_algorithm_spec(code)
    return bool(spec is not None and spec.family == "traditional")


def is_measurement_tool_algorithm(code: object) -> bool:
    spec = get_tool_algorithm_spec(code)
    return bool(spec is not None and spec.family == "measurement")


__all__ = [
    "DEFAULT_LEARNING_BACKBONE",
    "LEARNING_BACKBONES",
    "LEGACY_SHARED_BACKBONE_ALGORITHM_CODES",
    "SHARED_BACKBONE_ALGORITHM_CODE",
    "ToolAlgorithmSpec",
    "get_tool_algorithm_spec",
    "is_learning_tool_algorithm",
    "is_learning_backbone_code",
    "is_measurement_tool_algorithm",
    "is_traditional_tool_algorithm",
    "learning_backbone_storage_code",
    "learning_backbone_storage_codes",
    "list_tool_algorithm_codes",
    "list_tool_algorithm_specs",
    "normalize_tool_algorithm_code",
    "storage_code_backbone",
]
