from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .measurement import MEASUREMENT_ALGORITHMS
from .traditional import TRADITIONAL_ALGORITHMS


LEARNING_BACKBONES = [
    "efficientnet_b0",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]
DEFAULT_LEARNING_BACKBONE = LEARNING_BACKBONES[0]
SHARED_BACKBONE_ALGORITHM_CODE = "shared_backbone_register"
LEGACY_SHARED_BACKBONE_ALGORITHM_CODES = {
    "",
    "inherit_product",
    "inherit_product_backbone",
    SHARED_BACKBONE_ALGORITHM_CODE,
    *LEARNING_BACKBONES,
}

_LEARNING_DISPLAY_NAMES = {
    "efficientnet_b0": "高精度学习工具",
    "mobilenet_v3_small": "轻量学习工具",
    "mobilenet_v3_large": "均衡学习工具",
}

_LEARNING_STORAGE_CODES = {
    "efficientnet_b0": "lt01",
    "mobilenet_v3_small": "lt02",
    "mobilenet_v3_large": "lt03",
}
_STORAGE_CODE_TO_BACKBONE = {
    value: key
    for key, value in _LEARNING_STORAGE_CODES.items()
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
    "line_distance": "Line Distance",
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
    "line_distance": "debug.algorithm.line_distance",
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
            {"line": {"direction": "left_right"}}
            if _code == "find_line"
            else {
                "line_a_item_id": "",
                "line_b_item_id": "",
                "limit_unit": "px",
            }
            if _code == "line_distance"
            else {
                "line_a": {"direction": "left_right"},
                "line_b": {"direction": "right_left"},
            }
        ),
    )


def normalize_tool_algorithm_code(code: object) -> str:
    normalized = str(code or "").strip()
    if not normalized or normalized in LEGACY_SHARED_BACKBONE_ALGORITHM_CODES:
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
    normalized = str(code or "").strip()
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
    normalized = str(code or "").strip()
    if not normalized:
        return ""
    return _LEARNING_STORAGE_CODES.get(normalized, normalized)


def storage_code_backbone(code: object) -> str:
    normalized = str(code or "").strip()
    if not normalized:
        return ""
    return _STORAGE_CODE_TO_BACKBONE.get(normalized, normalized)


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
    "is_measurement_tool_algorithm",
    "is_traditional_tool_algorithm",
    "learning_backbone_storage_code",
    "list_tool_algorithm_codes",
    "list_tool_algorithm_specs",
    "normalize_tool_algorithm_code",
    "storage_code_backbone",
]
