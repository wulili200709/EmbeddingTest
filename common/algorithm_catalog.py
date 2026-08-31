from __future__ import annotations

from algorithms.registry import get_tool_algorithm_spec
from common.algorithm_codes import (
    SHARED_BACKBONE_ALGORITHM_CODE,
    normalize_tool_algorithm_code,
    storage_code_backbone,
)


ALGORITHM_DISPLAY_KEYS = {
    SHARED_BACKBONE_ALGORITHM_CODE: "debug.algorithm_group.learning",
    "efficientnet_b0": "debug.algorithm.efficientnet_b0",
    "mobilenet_v3_small": "debug.algorithm.mobilenet_v3_small",
    "mobilenet_v3_large": "debug.algorithm.mobilenet_v3_large",
    "meanintensity": "debug.algorithm.meanintensity",
    "meanstd": "debug.algorithm.meanstd",
    "meanhsv_h": "debug.algorithm.meanhsv_h",
    "meanhsv_v": "debug.algorithm.meanhsv_v",
    "meanhsv_s": "debug.algorithm.meanhsv_s",
    "find_circle": "debug.algorithm.find_circle",
    "find_line": "debug.algorithm.find_line",
    "find_line_subpix": "debug.algorithm.find_line_subpix",
    "bright_block_center": "debug.algorithm.bright_block_center",
    "pin_center_distance": "debug.algorithm.pin_center_distance",
    "bright_block_y_distance": "debug.algorithm.bright_block_y_distance",
    "center_distance": "debug.algorithm.center_distance",
    "line_distance": "debug.algorithm.line_distance",
    "line_distance_ref_normal": "debug.algorithm.line_distance_ref_normal",
}

DEFAULT_ALGORITHM_DISPLAY_NAMES = {
    SHARED_BACKBONE_ALGORITHM_CODE: "Learning Tool",
    "efficientnet_b0": "High Accuracy Learning Tool",
    "mobilenet_v3_small": "Lightweight Learning Tool",
    "mobilenet_v3_large": "Balanced Learning Tool",
    "meanintensity": "Grayscale Tool",
    "meanstd": "Deviation Tool",
    "meanhsv_h": "Hue Tool",
    "meanhsv_v": "Brightness Tool",
    "meanhsv_s": "Saturation Tool",
    "find_line": "Find Line",
    "find_line_subpix": "Subpixel Find Line",
    "bright_block_center": "Bright Block Center",
    "pin_center_distance": "Pin Center Distance",
    "bright_block_y_distance": "Bright Block Y Distance",
    "center_distance": "Center Distance",
    "line_distance": "Line Distance",
    "line_distance_ref_normal": "Reference Normal Distance",
}


def algorithm_display_key(code: object) -> str:
    normalized = storage_code_backbone(code)
    if normalized in ALGORITHM_DISPLAY_KEYS:
        return ALGORITHM_DISPLAY_KEYS[normalized]
    return ALGORITHM_DISPLAY_KEYS.get(normalize_tool_algorithm_code(normalized), "")


def default_algorithm_display_name(code: object) -> str:
    normalized = storage_code_backbone(code)
    if not normalized:
        return ""
    for candidate in (normalized, normalize_tool_algorithm_code(normalized)):
        if candidate in DEFAULT_ALGORITHM_DISPLAY_NAMES:
            return DEFAULT_ALGORITHM_DISPLAY_NAMES[candidate]
        spec = get_tool_algorithm_spec(candidate)
        if spec is not None:
            return spec.display_name or candidate
    return normalized


__all__ = [
    "ALGORITHM_DISPLAY_KEYS",
    "DEFAULT_ALGORITHM_DISPLAY_NAMES",
    "algorithm_display_key",
    "default_algorithm_display_name",
]
