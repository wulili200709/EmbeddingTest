from __future__ import annotations

import math

import cv2
import numpy as np

from algorithms.measurement_types import FindLineConfig, FittedLine


def _edge_response(delta: np.ndarray, polarity: str, *, direction: str = "left_right") -> np.ndarray:
    if direction in {"right_left", "bottom_up"}:
        delta = -delta
    if polarity == "dark_to_bright":
        return delta
    if polarity == "bright_to_dark":
        return -delta
    return np.abs(delta)


def _canny_thresholds(config: FindLineConfig) -> tuple[float, float]:
    high = max(1.0, float(config.edge_threshold))
    low = max(0.0, high * 0.5)
    return low, high


def _parabolic_peak_offset(left: float, center: float, right: float) -> float:
    denominator = float(left) - 2.0 * float(center) + float(right)
    if abs(denominator) <= 1e-12:
        return 0.0
    offset = 0.5 * (float(left) - float(right)) / denominator
    return float(max(-1.0, min(1.0, offset)))


def _refine_horizontal_edge_x(gray: np.ndarray, y: int, x: int, config: FindLineConfig) -> float:
    h, w = gray.shape[:2]
    if w < 2:
        return float(x)
    delta = gray[int(y), 1:] - gray[int(y), :-1]
    response = _edge_response(delta, config.polarity, direction=config.direction)
    lo = max(0, int(x) - 2)
    hi = min(w - 2, int(x) + 2)
    if hi < lo:
        return float(x)
    local = response[lo:hi + 1]
    if local.size == 0:
        return float(x)
    best = int(lo + int(np.argmax(local)))
    offset = 0.0
    if 0 < best < response.shape[0] - 1:
        offset = _parabolic_peak_offset(response[best - 1], response[best], response[best + 1])
    return float(best + 1 + offset)


def _refine_vertical_edge_y(gray: np.ndarray, x: int, y: int, config: FindLineConfig) -> float:
    h, w = gray.shape[:2]
    if h < 2:
        return float(y)
    delta = gray[1:, int(x)] - gray[:-1, int(x)]
    response = _edge_response(delta, config.polarity, direction=config.direction)
    lo = max(0, int(y) - 2)
    hi = min(h - 2, int(y) + 2)
    if hi < lo:
        return float(y)
    local = response[lo:hi + 1]
    if local.size == 0:
        return float(y)
    best = int(lo + int(np.argmax(local)))
    offset = 0.0
    if 0 < best < response.shape[0] - 1:
        offset = _parabolic_peak_offset(response[best - 1], response[best], response[best + 1])
    return float(best + 1 + offset)


def _smooth_subpixel_derivative(delta: np.ndarray, *, horizontal: bool) -> np.ndarray:
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    kernel = kernel.reshape(1, -1) if horizontal else kernel.reshape(-1, 1)
    return cv2.filter2D(
        np.asarray(delta, dtype=np.float32),
        cv2.CV_32F,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )


def _is_response_peak(response: np.ndarray, index: int, threshold: float) -> bool:
    center = float(response[int(index)])
    if center < float(threshold):
        return False
    left = float(response[int(index) - 1]) if int(index) > 0 else -math.inf
    right = float(response[int(index) + 1]) if int(index) < len(response) - 1 else -math.inf
    return center >= left and center >= right


def _subpixel_peak_coordinate(response: np.ndarray, index: int) -> float:
    idx = int(index)
    offset = 0.0
    if 0 < idx < len(response) - 1:
        offset = _parabolic_peak_offset(response[idx - 1], response[idx], response[idx + 1])
    return float(idx + 1 + offset)


def _filter_subpixel_edge_runs(points: np.ndarray, config: FindLineConfig) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= int(config.min_points):
        return pts
    if config.direction in {"left_right", "right_left"}:
        primary = pts[:, 1]
        secondary = pts[:, 0]
    else:
        primary = pts[:, 0]
        secondary = pts[:, 1]
    runs: list[np.ndarray] = []
    start = 0
    max_primary_gap = max(1.0, float(config.scan_step) * 2.5)
    max_secondary_jump = max(8.0, float(config.scan_step) * 8.0)
    for idx in range(1, len(pts)):
        if (
            abs(float(primary[idx] - primary[idx - 1])) > max_primary_gap
            or abs(float(secondary[idx] - secondary[idx - 1])) > max_secondary_jump
        ):
            runs.append(pts[start:idx])
            start = idx
    runs.append(pts[start:])
    best = max(runs, key=len)
    if len(best) >= int(config.min_points):
        return np.asarray(best, dtype=np.float32)
    return pts


def _subpixel_line_secondary_at_primary(
    line: FittedLine,
    primary: float,
    *,
    horizontal: bool,
) -> float:
    if horizontal:
        if abs(float(line.vy)) <= 1e-12:
            return float(line.x0)
        t = (float(primary) - float(line.y0)) / float(line.vy)
        return float(line.x0) + t * float(line.vx)
    if abs(float(line.vx)) <= 1e-12:
        return float(line.y0)
    t = (float(primary) - float(line.x0)) / float(line.vx)
    return float(line.y0) + t * float(line.vy)


def _subpixel_point_from_candidate(
    primary: float,
    secondary: float,
    *,
    horizontal: bool,
) -> tuple[float, float]:
    if horizontal:
        return float(secondary), float(primary)
    return float(primary), float(secondary)


def _select_subpixel_edge_points(
    candidates_by_scan: list[tuple[float, list[tuple[float, float]]]],
    *,
    horizontal: bool,
    config: FindLineConfig,
) -> np.ndarray:
    if not candidates_by_scan:
        return np.empty((0, 2), dtype=np.float32)

    def first_candidate(candidates: list[tuple[float, float]]) -> tuple[float, float]:
        return candidates[0]

    def strongest_candidate(candidates: list[tuple[float, float]]) -> tuple[float, float]:
        return max(candidates, key=lambda item: float(item[1]))

    selector = first_candidate if config.peak_selection == "first" else strongest_candidate
    seed_points = np.asarray(
        [
            _subpixel_point_from_candidate(primary, selector(candidates)[0], horizontal=horizontal)
            for primary, candidates in candidates_by_scan
            if candidates
        ],
        dtype=np.float32,
    ).reshape(-1, 2)
    seed_points = _filter_subpixel_edge_runs(seed_points, config)
    if config.peak_selection != "dominant" or len(seed_points) < int(config.min_points):
        return seed_points

    try:
        seed_line, seed_fit_points = fit_line_filtered(
            seed_points,
            min_points=min(int(config.min_points), int(len(seed_points))),
            context="subpixel dominant edge seed",
        )
    except Exception:
        return seed_points

    seed_residual = max(float(seed_line.residual), 0.0)
    gate_px = max(
        2.5,
        seed_residual * 3.0 + 1.0,
        float(config.scan_step) * 2.0,
        float(config.blur_ksize) * 0.75 if config.blur_ksize > 0 else 0.0,
    )
    selected: list[tuple[float, float]] = []
    for primary, candidates in candidates_by_scan:
        if not candidates:
            continue
        predicted = _subpixel_line_secondary_at_primary(seed_line, primary, horizontal=horizontal)
        nearest = min(candidates, key=lambda item: abs(float(item[0]) - predicted))
        nearest_distance = abs(float(nearest[0]) - predicted)
        if nearest_distance <= gate_px:
            selected.append(_subpixel_point_from_candidate(primary, nearest[0], horizontal=horizontal))

    refined_points = np.asarray(selected, dtype=np.float32).reshape(-1, 2)
    if len(refined_points) < int(config.min_points):
        return np.asarray(seed_fit_points, dtype=np.float32).reshape(-1, 2)
    return _filter_subpixel_edge_runs(refined_points, config)


def _find_subpixel_edge_points(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    config: FindLineConfig,
) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    h, w = gray.shape[:2]
    points: list[tuple[float, float]] = []

    if config.direction in {"left_right", "right_left"}:
        if w < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[:, 1:] - gray[:, :-1]
        filtered_delta = _smooth_subpixel_derivative(delta, horizontal=True)
        response = _edge_response(filtered_delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        x_indexes = range(0, w - 1) if config.direction == "left_right" else range(w - 2, -1, -1)
        candidates_by_scan: list[tuple[float, list[tuple[float, float]]]] = []
        for y in range(0, h, config.scan_step):
            row_response = response[y]
            row_valid = adjacent_valid[y]
            candidates: list[tuple[float, float]] = []
            for x in x_indexes:
                if row_valid[int(x)] and _is_response_peak(row_response, int(x), config.edge_threshold):
                    candidates.append(
                        (
                            _subpixel_peak_coordinate(row_response, int(x)),
                            float(row_response[int(x)]),
                        )
                    )
            if candidates:
                candidates_by_scan.append((float(y), candidates))
        points = _select_subpixel_edge_points(candidates_by_scan, horizontal=True, config=config)
    else:
        if h < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[1:, :] - gray[:-1, :]
        filtered_delta = _smooth_subpixel_derivative(delta, horizontal=False)
        response = _edge_response(filtered_delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[1:, :] & valid_mask[:-1, :]
        y_indexes = range(0, h - 1) if config.direction == "top_down" else range(h - 2, -1, -1)
        candidates_by_scan: list[tuple[float, list[tuple[float, float]]]] = []
        for x in range(0, w, config.scan_step):
            col_response = response[:, x]
            col_valid = adjacent_valid[:, x]
            candidates: list[tuple[float, float]] = []
            for y in y_indexes:
                if col_valid[int(y)] and _is_response_peak(col_response, int(y), config.edge_threshold):
                    candidates.append(
                        (
                            _subpixel_peak_coordinate(col_response, int(y)),
                            float(col_response[int(y)]),
                        )
                    )
            if candidates:
                candidates_by_scan.append((float(x), candidates))
        points = _select_subpixel_edge_points(candidates_by_scan, horizontal=False, config=config)

    return _filter_subpixel_edge_runs(np.asarray(points, dtype=np.float32), config)


def find_edge_points(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    config: FindLineConfig,
) -> np.ndarray:
    if config.edge_detector == "subpix_shen":
        return _find_subpixel_edge_points(crop_bgr, mask, config)

    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    canny_low, canny_high = _canny_thresholds(config)
    edges = cv2.Canny(gray_u8, canny_low, canny_high, L2gradient=True)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    h, w = gray.shape[:2]
    points: list[tuple[float, float]] = []

    if config.direction in {"left_right", "right_left"}:
        if w < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[:, 1:] - gray[:, :-1]
        response = _edge_response(delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        x_indexes = range(w) if config.direction == "left_right" else range(w - 1, -1, -1)
        for y in range(0, h, config.scan_step):
            for x in x_indexes:
                left = max(0, int(x) - 1)
                right = min(w - 2, int(x))
                if (
                    valid_mask[y, x]
                    and edges[y, x] > 0
                    and adjacent_valid[y, left:right + 1].any()
                    and response[y, left:right + 1].max(initial=0.0) >= config.edge_threshold
                ):
                    points.append((_refine_horizontal_edge_x(gray, y, x, config), float(y)))
                    break
    else:
        if h < 2:
            return np.empty((0, 2), dtype=np.float32)
        delta = gray[1:, :] - gray[:-1, :]
        response = _edge_response(delta, config.polarity, direction=config.direction)
        adjacent_valid = valid_mask[1:, :] & valid_mask[:-1, :]
        y_indexes = range(h) if config.direction == "top_down" else range(h - 1, -1, -1)
        for x in range(0, w, config.scan_step):
            for y in y_indexes:
                top = max(0, int(y) - 1)
                bottom = min(h - 2, int(y))
                if (
                    valid_mask[y, x]
                    and edges[y, x] > 0
                    and adjacent_valid[top:bottom + 1, x].any()
                    and response[top:bottom + 1, x].max(initial=0.0) >= config.edge_threshold
                ):
                    points.append((float(x), _refine_vertical_edge_y(gray, x, y, config)))
                    break

    return np.asarray(points, dtype=np.float32)


def fit_line(points: np.ndarray, *, min_points: int, context: str = "") -> FittedLine:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < int(min_points):
        prefix = f"{context}: " if str(context or "").strip() else ""
        raise RuntimeError(f"{prefix}find line points not enough: {len(pts)}/{int(min_points)}")
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(pts, cv2.DIST_WELSCH, 0, 0.01, 0.01).reshape(-1)]
    norm = math.hypot(vx, vy)
    if norm <= 1e-12:
        raise RuntimeError("fit line direction invalid")
    vx /= norm
    vy /= norm
    distances = np.abs(vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0))
    return FittedLine(
        vx=float(vx),
        vy=float(vy),
        x0=float(x0),
        y0=float(y0),
        residual=float(np.mean(distances)) if distances.size else 0.0,
        point_count=int(len(pts)),
    )


def _point_line_distances(points: np.ndarray, line: FittedLine) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return np.abs(float(line.vy) * (pts[:, 0] - float(line.x0)) - float(line.vx) * (pts[:, 1] - float(line.y0)))


def filter_line_points(
    points: np.ndarray,
    line: FittedLine,
    *,
    min_points: int,
    min_distance_px: float = 2.0,
    sigma: float = 3.0,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= int(min_points):
        return pts
    distances = _point_line_distances(pts, line)
    if distances.size == 0:
        return pts
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(float(min_distance_px), median + float(sigma) * robust_sigma)
    keep = distances <= threshold
    if int(np.count_nonzero(keep)) < int(min_points):
        return pts
    return pts[keep]


def fit_line_filtered(points: np.ndarray, *, min_points: int, context: str = "") -> tuple[FittedLine, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    initial = fit_line(pts, min_points=min_points, context=context)
    filtered = filter_line_points(pts, initial, min_points=min_points)
    if len(filtered) == len(pts):
        return initial, pts
    refined = fit_line(filtered, min_points=min_points, context=context)
    return refined, filtered


def _line_distance_px(line_a: FittedLine, line_b: FittedLine) -> float:
    a = -float(line_a.vy)
    b = float(line_a.vx)
    c = float(line_a.vy) * float(line_a.x0) - float(line_a.vx) * float(line_a.y0)
    norm = math.hypot(a, b)
    if norm <= 1e-12:
        raise RuntimeError("line normal invalid")
    return abs((a * float(line_b.x0) + b * float(line_b.y0) + c) / norm)


def _angle_delta_deg(line_a: FittedLine, line_b: FittedLine) -> float:
    dot = abs(float(line_a.vx) * float(line_b.vx) + float(line_a.vy) * float(line_b.vy))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _line_angle_deg(line: FittedLine) -> float:
    angle = math.degrees(math.atan2(float(line.vy), float(line.vx)))
    if angle < 0.0:
        angle += 180.0
    return angle


def _line_position_px(line: FittedLine, direction: str) -> float:
    if direction in {"left_right", "right_left"}:
        return float(line.x0)
    return float(line.y0)


def _line_segment_in_crop(
    line: FittedLine,
    *,
    crop_width: int,
    crop_height: int,
    origin: tuple[int, int] = (0, 0),
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    w = max(0, int(crop_width))
    h = max(0, int(crop_height))
    if w <= 0 or h <= 0:
        return None
    vx = float(line.vx)
    vy = float(line.vy)
    x0 = float(line.x0)
    y0 = float(line.y0)
    candidates: list[tuple[float, float]] = []
    if abs(vx) > 1e-12:
        for x in (0.0, float(w - 1)):
            t = (x - x0) / vx
            y = y0 + t * vy
            if -0.5 <= y <= float(h - 1) + 0.5:
                candidates.append((x, min(float(h - 1), max(0.0, y))))
    if abs(vy) > 1e-12:
        for y in (0.0, float(h - 1)):
            t = (y - y0) / vy
            x = x0 + t * vx
            if -0.5 <= x <= float(w - 1) + 0.5:
                candidates.append((min(float(w - 1), max(0.0, x)), y))

    unique: list[tuple[float, float]] = []
    for point in candidates:
        if not any(math.hypot(point[0] - old[0], point[1] - old[1]) < 1e-6 for old in unique):
            unique.append(point)
    if len(unique) < 2:
        half_span = max(w, h) * 0.5
        unique = [
            (x0 - vx * half_span, y0 - vy * half_span),
            (x0 + vx * half_span, y0 + vy * half_span),
        ]

    ox, oy = origin
    p0 = unique[0]
    p1 = max(unique[1:], key=lambda p: (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2)
    return (
        (float(p0[0] + ox), float(p0[1] + oy)),
        (float(p1[0] + ox), float(p1[1] + oy)),
    )


__all__ = [
    "_angle_delta_deg",
    "_line_angle_deg",
    "_line_distance_px",
    "_line_position_px",
    "_line_segment_in_crop",
    "filter_line_points",
    "find_edge_points",
    "fit_line",
    "fit_line_filtered",
]
