from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .measurement import (
    FIND_LINE_ALGORITHMS,
    FIND_LINE_SUBPIX_ALGORITHM,
    LINE_DISTANCE_ALGORITHMS,
    MEASUREMENT_ALGORITHMS,
)
from .traditional import TRADITIONAL_ALGORITHMS


LEARNING_BACKBONES = [
    "efficientnet_b0",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]
DEFAULT_LEARNING_BACKBONE = LEARNING_BACKBONES[0]
SHARED_BACKBONE_ALGORITHM_CODE = "shared_backbone_register"
_LEARNING_STORAGE_CODES = {
    "efficientnet_b0": "b0",
    "mobilenet_v3_small": "b1",
    "mobilenet_v3_large": "b2",
}
_LEGACY_LEARNING_STORAGE_CODES = {
    "efficientnet_b0": ("lt_b0", "lt01"),
    "mobilenet_v3_small": ("lt_b1", "lt02"),
    "mobilenet_v3_large": ("lt_b2", "lt03"),
}
_STORAGE_CODE_TO_BACKBONE = {
    code: key
    for key, codes in {
        key: (storage_code, *_LEGACY_LEARNING_STORAGE_CODES.get(key, ()))
        for key, storage_code in _LEARNING_STORAGE_CODES.items()
    }.items()
    for code in codes
}
_BACKBONE_INPUT_ALIASES = {
    "efficinet_b0": "efficientnet_b0",
    "mobilenetv3_small": "mobilenet_v3_small",
    "mobilenetv3_large": "mobilenet_v3_large",
}
LEGACY_SHARED_BACKBONE_ALGORITHM_CODES = {
    "",
    "inherit_product",
    "inherit_product_backbone",
    SHARED_BACKBONE_ALGORITHM_CODE,
    *LEARNING_BACKBONES,
    *_LEARNING_STORAGE_CODES.values(),
    *[
        legacy_code
        for legacy_codes in _LEGACY_LEARNING_STORAGE_CODES.values()
        for legacy_code in legacy_codes
    ],
}

_LEARNING_DISPLAY_NAMES = {
    "efficientnet_b0": "高精度学习工具",
    "mobilenet_v3_small": "轻量学习工具",
    "mobilenet_v3_large": "均衡学习工具",
}

_TRADITIONAL_DISPLAY_NAMES = {
    "meanintensity": "灰度工具",
    "meanstd": "偏差工具",
    "meanhsv_h": "色相工具",
    "meanhsv_v": "明度工具",
    "meanhsv_s": "饱和度工具",
}

_MEASUREMENT_DISPLAY_NAMES = {
    "find_line": "Find Line",
    "find_line_subpix": "Subpixel Find Line",
    "line_distance": "Line Distance",
    "line_distance_ref_normal": "Reference Normal Distance",
}

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
        display_name="学习工具",
        family="learning",
        fit_mode="register",
        default_params={},
    ),
}
for _code in TRADITIONAL_ALGORITHMS:
    _TOOL_ALGORITHM_SPECS[_code] = ToolAlgorithmSpec(
        code=_code,
        display_name=_TRADITIONAL_DISPLAY_NAMES.get(_code, _code),
        family="traditional",
        fit_mode="calibrate",
        default_params={},
    )
for _code in MEASUREMENT_ALGORITHMS:
    _TOOL_ALGORITHM_SPECS[_code] = ToolAlgorithmSpec(
        code=_code,
        display_name=_MEASUREMENT_DISPLAY_NAMES.get(_code, _code),
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
                "line_a": {"direction": "left_right"},
                "line_b": {"direction": "right_left"},
            }
        ),
    )


def normalize_tool_algorithm_code(code: object) -> str:
    normalized = str(code or "").strip()
    normalized_backbone = storage_code_backbone(normalized)
    if (
        not normalized
        or normalized in LEGACY_SHARED_BACKBONE_ALGORITHM_CODES
        or normalized_backbone in LEARNING_BACKBONES
    ):
        return SHARED_BACKBONE_ALGORITHM_CODE
    if normalized == "edge_distance":
        return "find_line"
    return normalized


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


def _default_algorithm_display_name(code: str) -> str:
    if code in _LEARNING_DISPLAY_NAMES:
        return _LEARNING_DISPLAY_NAMES[code]
    if code in _TRADITIONAL_DISPLAY_NAMES:
        return _TRADITIONAL_DISPLAY_NAMES[code]
    if code in _MEASUREMENT_DISPLAY_NAMES:
        return _MEASUREMENT_DISPLAY_NAMES[code]
    spec = _TOOL_ALGORITHM_SPECS.get(code)
    if spec is not None:
        return spec.display_name
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


def learning_backbone_storage_code(code: object) -> str:
    normalized = storage_code_backbone(code)
    if not normalized:
        return ""
    return _LEARNING_STORAGE_CODES.get(normalized, normalized)


def learning_backbone_storage_codes(code: object) -> List[str]:
    normalized = storage_code_backbone(code)
    if not normalized:
        return []
    primary = _LEARNING_STORAGE_CODES.get(normalized)
    legacy = list(_LEGACY_LEARNING_STORAGE_CODES.get(normalized, ()))
    codes = [value for value in [primary, *legacy] if value]
    return list(dict.fromkeys(codes)) or [normalized]


def storage_code_backbone(code: object) -> str:
    normalized = str(code or "").strip()
    if not normalized:
        return ""
    normalized_lower = normalized.lower()
    if normalized_lower in _BACKBONE_INPUT_ALIASES:
        return _BACKBONE_INPUT_ALIASES[normalized_lower]
    return _STORAGE_CODE_TO_BACKBONE.get(normalized_lower, normalized)


def is_learning_backbone_code(code: object) -> bool:
    return storage_code_backbone(code) in LEARNING_BACKBONES


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
    "algorithm_display_name",
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
