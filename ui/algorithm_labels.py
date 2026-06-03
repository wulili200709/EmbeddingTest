from __future__ import annotations

from common.algorithm_codes import (
    SHARED_BACKBONE_ALGORITHM_CODE,
    normalize_tool_algorithm_code,
    storage_code_backbone,
)
from algorithms.registry import get_tool_algorithm_spec


_ALGORITHM_DISPLAY_KEYS = {
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
    "line_distance": "debug.algorithm.line_distance",
    "line_distance_ref_normal": "debug.algorithm.line_distance_ref_normal",
}

_DEFAULT_DISPLAY_NAMES = {
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
    "line_distance": "Line Distance",
    "line_distance_ref_normal": "Reference Normal Distance",
}


def _default_algorithm_display_name(code: str) -> str:
    if code in _DEFAULT_DISPLAY_NAMES:
        return _DEFAULT_DISPLAY_NAMES[code]
    spec = get_tool_algorithm_spec(code)
    if spec is not None:
        return spec.display_name or code
    return code


def _translated_display_name(code: str, fallback: str) -> str:
    key = _ALGORITHM_DISPLAY_KEYS.get(code, "")
    if not key:
        return fallback
    try:
        from ui.i18n import tr
    except Exception:
        return fallback
    return str(tr(key) or fallback)


def algorithm_display_name(code: object) -> str:
    normalized = storage_code_backbone(code)
    if not normalized:
        return ""

    direct_name = _default_algorithm_display_name(normalized)
    if direct_name != normalized or normalized in _ALGORITHM_DISPLAY_KEYS:
        return _translated_display_name(normalized, direct_name)

    normalized_tool = normalize_tool_algorithm_code(normalized)
    translated_name = _default_algorithm_display_name(normalized_tool)
    if translated_name != normalized_tool or normalized_tool in _ALGORITHM_DISPLAY_KEYS:
        return _translated_display_name(normalized_tool, translated_name)
    return normalized


__all__ = ["algorithm_display_name"]
