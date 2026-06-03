from __future__ import annotations

from typing import List


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


def storage_code_backbone(code: object) -> str:
    normalized = str(code or "").strip()
    if not normalized:
        return ""
    normalized_lower = normalized.lower()
    if normalized_lower in _BACKBONE_INPUT_ALIASES:
        return _BACKBONE_INPUT_ALIASES[normalized_lower]
    return _STORAGE_CODE_TO_BACKBONE.get(normalized_lower, normalized)


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


def is_learning_backbone_code(code: object) -> bool:
    return storage_code_backbone(code) in LEARNING_BACKBONES


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


__all__ = [
    "DEFAULT_LEARNING_BACKBONE",
    "LEARNING_BACKBONES",
    "LEGACY_SHARED_BACKBONE_ALGORITHM_CODES",
    "SHARED_BACKBONE_ALGORITHM_CODE",
    "is_learning_backbone_code",
    "learning_backbone_storage_code",
    "learning_backbone_storage_codes",
    "normalize_tool_algorithm_code",
    "storage_code_backbone",
]
