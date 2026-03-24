from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


def normalize_tool_algorithm_code(code: object) -> str:
    normalized = str(code or "").strip()
    if not normalized or normalized in LEGACY_SHARED_BACKBONE_ALGORITHM_CODES:
        return SHARED_BACKBONE_ALGORITHM_CODE
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
    ]


def list_tool_algorithm_codes() -> List[str]:
    return [spec.code for spec in list_tool_algorithm_specs()]


def algorithm_display_name(code: object) -> str:
    normalized = str(code or "").strip()
    if not normalized:
        return ""
    if normalized in _LEARNING_DISPLAY_NAMES:
        return _LEARNING_DISPLAY_NAMES[normalized]
    if normalized in _TRADITIONAL_DISPLAY_NAMES:
        return _TRADITIONAL_DISPLAY_NAMES[normalized]
    spec = get_tool_algorithm_spec(normalized)
    if spec is not None:
        return spec.display_name
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


__all__ = [
    "DEFAULT_LEARNING_BACKBONE",
    "LEARNING_BACKBONES",
    "LEGACY_SHARED_BACKBONE_ALGORITHM_CODES",
    "SHARED_BACKBONE_ALGORITHM_CODE",
    "ToolAlgorithmSpec",
    "algorithm_display_name",
    "get_tool_algorithm_spec",
    "is_learning_tool_algorithm",
    "is_traditional_tool_algorithm",
    "learning_backbone_storage_code",
    "list_tool_algorithm_codes",
    "list_tool_algorithm_specs",
    "normalize_tool_algorithm_code",
    "storage_code_backbone",
]
