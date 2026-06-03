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
    FIND_LINE_ALGORITHMS,
    FIND_LINE_SUBPIX_ALGORITHM,
    LINE_DISTANCE_ALGORITHMS,
    MEASUREMENT_ALGORITHMS,
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
