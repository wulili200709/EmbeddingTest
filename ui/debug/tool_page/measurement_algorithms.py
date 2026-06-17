from __future__ import annotations

from algorithms.measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHMS,
    FIND_LINE_ALGORITHM,
    FIND_LINE_ALGORITHMS,
    FIND_LINE_SUBPIX_ALGORITHM,
    LINE_DISTANCE_ALGORITHMS,
    PIN_CENTER_DISTANCE_ALGORITHM,
)
from common.algorithm_codes import normalize_tool_algorithm_code


def is_line_distance_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() in LINE_DISTANCE_ALGORITHMS


def is_find_line_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() in FIND_LINE_ALGORITHMS


def is_pin_center_distance_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() == PIN_CENTER_DISTANCE_ALGORITHM


def is_bright_block_y_distance_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() == BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM


def is_bright_block_center_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() == BRIGHT_BLOCK_CENTER_ALGORITHM


def is_center_distance_algorithm(algorithm: object) -> bool:
    return str(algorithm or "").strip() in CENTER_DISTANCE_ALGORITHMS


def is_single_roi_distance_algorithm(algorithm: object) -> bool:
    return is_pin_center_distance_algorithm(algorithm) or is_bright_block_y_distance_algorithm(algorithm)


def public_algorithm_code(algorithm: object) -> str:
    raw = str(algorithm or "").strip()
    if raw == FIND_LINE_SUBPIX_ALGORITHM:
        return FIND_LINE_ALGORITHM
    normalized = normalize_tool_algorithm_code(raw)
    if normalized == FIND_LINE_SUBPIX_ALGORITHM:
        return FIND_LINE_ALGORITHM
    if raw in {"", "edge_distance"}:
        return normalized
    if normalized == FIND_LINE_ALGORITHM and raw not in FIND_LINE_ALGORITHMS:
        return normalized
    if normalized == PIN_CENTER_DISTANCE_ALGORITHM:
        return normalized
    if normalized == BRIGHT_BLOCK_CENTER_ALGORITHM:
        return normalized
    if normalized == BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM:
        return normalized
    if normalized == CENTER_DISTANCE_ALGORITHM:
        return normalized
    if normalized in LINE_DISTANCE_ALGORITHMS:
        return normalized
    return raw


def hide_from_algorithm_picker(algorithm: object) -> bool:
    normalized = normalize_tool_algorithm_code(algorithm)
    return (
        normalized == FIND_LINE_SUBPIX_ALGORITHM
        or normalized in LINE_DISTANCE_ALGORITHMS
        or normalized in {
            PIN_CENTER_DISTANCE_ALGORITHM,
            BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
        }
    )
