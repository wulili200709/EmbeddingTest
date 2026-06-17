from __future__ import annotations

import math
from typing import Any, Mapping

import cv2
import numpy as np

from algorithms.measurement_types import PinCenterCandidate, PinCenterDistanceConfig, _bool_param


def _bright_threshold(gray: np.ndarray, valid_mask: np.ndarray, configured_threshold: float) -> float:
    if float(configured_threshold) > 0.0:
        return float(configured_threshold)
    values = np.asarray(gray, dtype=np.float32)[np.asarray(valid_mask, dtype=bool)]
    if values.size == 0:
        raise RuntimeError("pin center ROI mask empty")
    if float(values.max() - values.min()) < 3.0:
        raise RuntimeError("pin center ROI contrast too low")
    values_u8 = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _binary = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(otsu) <= 1.0:
        otsu = float(np.percentile(values, 85.0))
    return float(max(1.0, min(254.0, otsu)))


def _pin_center_binary(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    config: PinCenterDistanceConfig,
) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    threshold = _bright_threshold(gray, valid_mask, config.threshold)
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[(gray >= float(threshold)) & valid_mask] = 255
    if config.morph_open_width > 0 and config.morph_open_height > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(config.morph_open_width), int(config.morph_open_height)),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if config.morph_close_size >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(config.morph_close_size), int(config.morph_close_size)),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary[~valid_mask] = 0
    return binary, threshold


def _pin_candidate_from_contour(
    contour: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
) -> PinCenterCandidate | None:
    area = float(abs(cv2.contourArea(contour)))
    if area < float(config.min_area_px):
        return None
    if config.max_area_px > 0.0 and area > float(config.max_area_px):
        return None
    x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
    if w <= 0 or h <= 0:
        return None
    width = float(w)
    height = float(h)
    if width < float(config.min_width_px) or height < float(config.min_height_px):
        return None
    if config.target_orientation == "horizontal":
        if width / max(height, 1.0) < float(config.min_aspect_ratio):
            return None
        if config.max_height_px > 0.0 and height > float(config.max_height_px):
            return None
        aspect = width / max(height, 1.0)
    elif config.target_orientation == "vertical":
        if height / max(width, 1.0) < float(config.min_aspect_ratio):
            return None
        if config.max_height_px > 0.0 and width > float(config.max_height_px):
            return None
        aspect = height / max(width, 1.0)
    else:
        aspect = max(width, height) / max(min(width, height), 1.0)
        if aspect < float(config.min_aspect_ratio):
            return None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if float(rw) <= 0.0 or float(rh) <= 0.0:
        return None
    box = cv2.boxPoints(rect)
    ox, oy = origin
    absolute_box = tuple((float(px + ox), float(py + oy)) for px, py in np.asarray(box).reshape(-1, 2))
    clipped_aspect = min(float(aspect), 20.0)
    score = float(area * clipped_aspect)
    return PinCenterCandidate(
        center_xy=(float(cx + ox), float(cy + oy)),
        box_points=absolute_box,
        area_px=area,
        bbox_xywh=(int(x + ox), int(y + oy), int(w), int(h)),
        width_px=width,
        height_px=height,
        aspect_ratio=float(aspect),
        angle_deg=float(angle),
        score=score,
    )


def _find_pin_center_candidates(
    binary: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
) -> tuple[PinCenterCandidate, ...]:
    contours, _hierarchy = cv2.findContours(
        np.asarray(binary, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[PinCenterCandidate] = []
    for contour in contours:
        candidate = _pin_candidate_from_contour(contour, origin=origin, config=config)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lx2 = lx + max(0, lw)
    ly2 = ly + max(0, lh)
    rx2 = rx + max(0, rw)
    ry2 = ry + max(0, rh)
    ix0 = max(lx, rx)
    iy0 = max(ly, ry)
    ix1 = min(lx2, rx2)
    iy1 = min(ly2, ry2)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    intersection = float(iw * ih)
    union = float(max(0, lw) * max(0, lh) + max(0, rw) * max(0, rh)) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _dedupe_pin_candidates(
    candidates: list[PinCenterCandidate],
) -> tuple[PinCenterCandidate, ...]:
    selected: list[PinCenterCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for existing in selected:
            same_center = (
                abs(float(candidate.center_xy[0]) - float(existing.center_xy[0]))
                <= max(6.0, min(float(candidate.width_px), float(existing.width_px)) * 0.25)
                and abs(float(candidate.center_xy[1]) - float(existing.center_xy[1]))
                <= max(4.0, min(float(candidate.height_px), float(existing.height_px)) * 0.75)
            )
            if same_center or _bbox_iou(candidate.bbox_xywh, existing.bbox_xywh) >= 0.35:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return tuple(selected)


def _find_pin_inner_strip_candidates(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> tuple[PinCenterCandidate, ...]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    values = gray[valid_mask]
    if values.size == 0:
        return ()

    crop_h, crop_w = gray.shape[:2]
    min_width = max(float(config.min_width_px), min(80.0, max(18.0, float(crop_w) * 0.12)))
    if crop_h <= 180:
        max_height = max(8.0, min(52.0, float(crop_h) * 0.50))
    else:
        max_height = max(8.0, min(42.0, float(crop_h) * 0.35))
    min_area = max(float(config.min_area_px), min_width * 3.0)
    open_w = max(7, int(round(min_width * 0.35)))
    if open_w % 2 == 0:
        open_w += 1
    close_w = max(3, int(round(float(crop_w) * 0.012)))
    if close_w % 2 == 0:
        close_w += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))

    thresholds: list[float] = []
    for percentile in (99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0):
        thresholds.append(float(np.percentile(values, percentile)))
    thresholds.append(float(rough_threshold))
    unique_thresholds = sorted(
        {
            float(max(1.0, min(254.0, threshold)))
            for threshold in thresholds
            if math.isfinite(float(threshold))
        },
        reverse=True,
    )

    ox, oy = origin
    candidates: list[PinCenterCandidate] = []
    for threshold in unique_thresholds:
        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(gray >= threshold) & valid_mask] = 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary[~valid_mask] = 0
        contours, _hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
            if w <= 1 or h <= 1:
                continue
            width = float(w)
            height = float(h)
            if width < min_width or height < float(config.min_height_px):
                continue
            if height > max_height:
                continue
            if width > float(crop_w) * 0.90:
                continue
            aspect = width / max(1.0, height)
            if aspect < max(1.8, float(config.min_aspect_ratio) * 0.85):
                continue
            area = max(float(abs(cv2.contourArea(contour))), width * height * 0.35)
            if area < min_area:
                continue
            fill = area / max(1.0, width * height)
            if fill < 0.22:
                continue
            abs_left = float(x + ox)
            abs_right = float(x + w - 1 + ox)
            abs_top = float(y + oy)
            abs_bottom = float(y + h - 1 + oy)
            center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
            score = area * min(aspect, 20.0) * max(0.25, fill) * (width / max(1.0, min_width))
            candidates.append(
                PinCenterCandidate(
                    center_xy=(float(center[0]), float(center[1])),
                    box_points=(
                        (abs_left, abs_top),
                        (abs_right, abs_top),
                        (abs_right, abs_bottom),
                        (abs_left, abs_bottom),
                    ),
                    area_px=float(area),
                    bbox_xywh=(int(round(abs_left)), int(round(abs_top)), w, h),
                    width_px=width,
                    height_px=height,
                    aspect_ratio=float(aspect),
                    angle_deg=0.0,
                    score=float(score),
                )
            )
    return _dedupe_pin_candidates(candidates)


def _find_bright_vertical_block_candidates(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> tuple[PinCenterCandidate, ...]:
    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if config.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    values = gray[valid_mask]
    if values.size == 0:
        return ()

    crop_h, crop_w = gray.shape[:2]
    small_roi = crop_w <= 180 and crop_h <= 320
    min_height = max(float(config.min_height_px), min(90.0, max(14.0, float(crop_h) * 0.055)))
    if small_roi:
        max_height = max(min_height + 4.0, min(220.0, float(crop_h) * 0.95))
        max_width = max(8.0, min(80.0, float(crop_w) * 0.75))
    else:
        max_height = max(min_height + 4.0, min(110.0, float(crop_h) * 0.28))
        max_width = max(8.0, min(55.0, float(crop_w) * 0.12))
    min_area = max(float(config.min_area_px), min_height * 3.0)
    open_h = max(7, int(round(min_height * 0.35)))
    if open_h % 2 == 0:
        open_h += 1
    close_h = max(3, int(round(float(crop_h) * 0.010)))
    if close_h % 2 == 0:
        close_h += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_h))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))

    thresholds: list[float] = []
    for percentile in (99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0):
        thresholds.append(float(np.percentile(values, percentile)))
    thresholds.append(float(rough_threshold))
    unique_thresholds = sorted(
        {
            float(max(1.0, min(254.0, threshold)))
            for threshold in thresholds
            if math.isfinite(float(threshold))
        },
        reverse=True,
    )

    ox, oy = origin
    candidates: list[PinCenterCandidate] = []
    for threshold in unique_thresholds:
        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(gray >= threshold) & valid_mask] = 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary[~valid_mask] = 0
        contours, _hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
            if w <= 1 or h <= 1:
                continue
            width = float(w)
            height = float(h)
            if height < min_height or width < 2.0:
                continue
            if height > max_height or width > max_width:
                continue
            aspect = height / max(1.0, width)
            if aspect < max(1.25, float(config.min_aspect_ratio) * 0.70):
                continue
            area = max(float(abs(cv2.contourArea(contour))), width * height * 0.35)
            if area < min_area:
                continue
            fill = area / max(1.0, width * height)
            if fill < 0.18:
                continue
            abs_left = float(x + ox)
            abs_right = float(x + w - 1 + ox)
            abs_top = float(y + oy)
            abs_bottom = float(y + h - 1 + oy)
            center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
            score = area * min(aspect, 20.0) * max(0.25, fill) * (height / max(1.0, min_height))
            candidates.append(
                PinCenterCandidate(
                    center_xy=(float(center[0]), float(center[1])),
                    box_points=(
                        (abs_left, abs_top),
                        (abs_right, abs_top),
                        (abs_right, abs_bottom),
                        (abs_left, abs_bottom),
                    ),
                    area_px=float(area),
                    bbox_xywh=(int(round(abs_left)), int(round(abs_top)), w, h),
                    width_px=width,
                    height_px=height,
                    aspect_ratio=float(aspect),
                    angle_deg=90.0,
                    score=float(score),
                )
            )
        col_counts = np.count_nonzero(binary > 0, axis=0)
        min_col_hits = max(4, int(round(min_height * 0.45)))
        for col_left, col_right in _runs_from_indices(np.where(col_counts >= min_col_hits)[0]):
            if col_right < col_left:
                continue
            segments: list[tuple[int, int]] = []
            run_width = col_right - col_left + 1
            if float(run_width) <= max_width:
                segments.append((col_left, col_right))
            else:
                edge_width = max(3, int(round(max_width)))
                segments.append((col_left, min(col_right, col_left + edge_width - 1)))
                segments.append((max(col_left, col_right - edge_width + 1), col_right))
            for seg_left, seg_right in segments:
                if seg_right <= seg_left:
                    continue
                segment = binary[:, seg_left:seg_right + 1] > 0
                seg_width = float(seg_right - seg_left + 1)
                min_row_hits = max(2, int(round(seg_width * 0.35)))
                row_counts = np.count_nonzero(segment, axis=1)
                row_indices = np.where(row_counts >= min_row_hits)[0]
                for row_top, row_bottom in _runs_from_indices(row_indices):
                    height = float(row_bottom - row_top + 1)
                    if height < min_height or height > max_height:
                        continue
                    local_points = np.argwhere(segment[row_top:row_bottom + 1, :])
                    if local_points.size == 0:
                        continue
                    ys = local_points[:, 0] + row_top
                    xs = local_points[:, 1] + seg_left
                    x0_i = int(xs.min())
                    x1_i = int(xs.max())
                    y0_i = int(ys.min())
                    y1_i = int(ys.max())
                    width = float(x1_i - x0_i + 1)
                    height = float(y1_i - y0_i + 1)
                    if width > max_width or height < min_height or height > max_height:
                        continue
                    aspect = height / max(1.0, width)
                    if aspect < max(1.15, float(config.min_aspect_ratio) * 0.65):
                        continue
                    area = float(local_points.shape[0])
                    fill = area / max(1.0, width * height)
                    if fill < 0.18:
                        continue
                    abs_left = float(x0_i + ox)
                    abs_right = float(x1_i + ox)
                    abs_top = float(y0_i + oy)
                    abs_bottom = float(y1_i + oy)
                    center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
                    score = area * min(aspect, 20.0) * max(0.25, fill) * 0.75
                    candidates.append(
                        PinCenterCandidate(
                            center_xy=(float(center[0]), float(center[1])),
                            box_points=(
                                (abs_left, abs_top),
                                (abs_right, abs_top),
                                (abs_right, abs_bottom),
                                (abs_left, abs_bottom),
                            ),
                            area_px=float(area),
                            bbox_xywh=(
                                int(round(abs_left)),
                                int(round(abs_top)),
                                int(round(width)),
                                int(round(height)),
                            ),
                            width_px=width,
                            height_px=height,
                            aspect_ratio=float(aspect),
                            angle_deg=90.0,
                            score=float(score),
                        )
                    )
    return _dedupe_pin_candidates(candidates)


def _run_containing_or_nearest(indices: np.ndarray, target: float) -> tuple[int, int] | None:
    values = [int(v) for v in np.asarray(indices, dtype=np.int32).reshape(-1)]
    if not values:
        return None
    runs: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if int(value) != prev + 1:
            runs.append((start, prev))
            start = int(value)
        prev = int(value)
    runs.append((start, prev))
    target_value = float(target)
    for lo, hi in runs:
        if float(lo) <= target_value <= float(hi):
            return lo, hi
    return min(
        runs,
        key=lambda run: min(abs(float(run[0]) - target_value), abs(float(run[1]) - target_value)),
    )


def _runs_from_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    values = [int(v) for v in np.asarray(indices, dtype=np.int32).reshape(-1)]
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if int(value) != prev + 1:
            runs.append((start, prev))
            start = int(value)
        prev = int(value)
    runs.append((start, prev))
    return runs


def _pin_local_threshold(gray: np.ndarray, valid: np.ndarray, rough_threshold: float) -> float:
    values = np.asarray(gray, dtype=np.float32)[np.asarray(valid, dtype=bool)]
    if values.size == 0:
        return max(1.0, float(rough_threshold) * 0.7)
    if float(values.max() - values.min()) < 3.0:
        return max(1.0, min(254.0, float(values.mean())))
    low = float(np.percentile(values, 20.0))
    high = float(np.percentile(values, 95.0))
    contrast_threshold = low + (high - low) * 0.35
    values_u8 = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _binary = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = min(float(otsu), float(contrast_threshold))
    if float(rough_threshold) > 1.0:
        threshold = min(threshold, float(rough_threshold) * 0.5)
    return float(max(1.0, min(254.0, threshold)))


def _refine_pin_candidate_from_local_edges(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> PinCenterCandidate:
    if not config.refine_center:
        return candidate
    crop_h, crop_w = np.asarray(mask).shape[:2]
    if crop_w <= 1 or crop_h <= 1:
        return candidate

    ox, oy = origin
    bx_abs, by_abs, bw, bh = candidate.bbox_xywh
    bx = int(round(float(bx_abs) - float(ox)))
    by = int(round(float(by_abs) - float(oy)))
    bw = max(1, int(bw))
    bh = max(1, int(bh))
    pad_x = int(round(max(4.0, float(bw) * float(config.refine_expand_x_ratio))))
    pad_y = int(round(max(6.0, float(bh) * float(config.refine_expand_y_ratio))))
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(crop_w, bx + bw + pad_x)
    y1 = min(crop_h, by + bh + pad_y)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return candidate

    local_bgr = np.asarray(crop_bgr, dtype=np.uint8)[y0:y1, x0:x1]
    local_mask = np.asarray(mask, dtype=np.uint8)[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(local_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    threshold = _pin_local_threshold(gray, local_mask, rough_threshold)
    local_binary = np.zeros(gray.shape, dtype=np.uint8)
    local_binary[(gray >= threshold) & local_mask] = 255
    if config.morph_close_size >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(config.morph_close_size), int(config.morph_close_size)),
        )
        local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_CLOSE, kernel)
    local_binary[~local_mask] = 0

    rough_cx = float(candidate.center_xy[0]) - float(ox) - float(x0)
    rough_cy = float(candidate.center_xy[1]) - float(oy) - float(y0)
    core_x0 = max(0, bx - x0)
    core_x1 = min(local_binary.shape[1], bx + bw - x0)
    if core_x1 <= core_x0 + 1:
        core_x0, core_x1 = 0, local_binary.shape[1]
    core = local_binary[:, core_x0:core_x1] > 0
    core_valid = local_mask[:, core_x0:core_x1]
    valid_counts = np.count_nonzero(core_valid, axis=1)
    hit_counts = np.count_nonzero(core & core_valid, axis=1)
    row_ratios = hit_counts / np.maximum(valid_counts, 1)
    min_row_hits = max(2, int(round(max(1, core_x1 - core_x0) * float(config.refine_min_fill_ratio))))
    row_indices = np.where((hit_counts >= min_row_hits) | (row_ratios >= float(config.refine_min_fill_ratio)))[0]
    row_run = _run_containing_or_nearest(row_indices, rough_cy)
    if row_run is None:
        return candidate
    top, bottom = row_run
    if bottom <= top:
        return candidate

    y_slice = slice(max(0, top), min(local_binary.shape[0], bottom + 1))
    body = local_binary[y_slice, :] > 0
    body_valid = local_mask[y_slice, :]
    valid_col_counts = np.count_nonzero(body_valid, axis=0)
    hit_col_counts = np.count_nonzero(body & body_valid, axis=0)
    col_ratios = hit_col_counts / np.maximum(valid_col_counts, 1)
    min_col_hits = max(2, int(round(max(1, bottom - top + 1) * float(config.refine_min_fill_ratio))))
    col_indices = np.where((hit_col_counts >= min_col_hits) | (col_ratios >= float(config.refine_min_fill_ratio)))[0]
    col_run = _run_containing_or_nearest(col_indices, rough_cx)
    if col_run is None:
        return candidate
    left, right = col_run
    if right <= left:
        return candidate

    width = float(right - left + 1)
    height = float(bottom - top + 1)
    if width < float(config.min_width_px) or height < float(config.min_height_px):
        return candidate
    aspect = max(width, height) / max(1.0, min(width, height))
    if config.target_orientation == "horizontal" and width / max(height, 1.0) < max(1.0, config.min_aspect_ratio * 0.65):
        return candidate
    if config.target_orientation == "vertical" and height / max(width, 1.0) < max(1.0, config.min_aspect_ratio * 0.65):
        return candidate

    abs_left = float(x0 + left + ox)
    abs_right = float(x0 + right + ox)
    abs_top = float(y0 + top + oy)
    abs_bottom = float(y0 + bottom + oy)
    center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
    box_points = (
        (abs_left, abs_top),
        (abs_right, abs_top),
        (abs_right, abs_bottom),
        (abs_left, abs_bottom),
    )
    area = float(width * height)
    return PinCenterCandidate(
        center_xy=(float(center[0]), float(center[1])),
        box_points=box_points,
        area_px=area,
        bbox_xywh=(
            int(round(abs_left)),
            int(round(abs_top)),
            max(1, int(round(width))),
            max(1, int(round(height))),
        ),
        width_px=float(width),
        height_px=float(height),
        aspect_ratio=float(aspect),
        angle_deg=0.0,
        score=float(max(candidate.score, area * min(aspect, 20.0))),
    )


def _refine_pin_candidate_to_inner_bright_strip(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    config: PinCenterDistanceConfig,
    rough_threshold: float,
) -> PinCenterCandidate:
    if not config.refine_center:
        return candidate
    crop_h, crop_w = np.asarray(mask).shape[:2]
    if crop_w <= 1 or crop_h <= 1:
        return candidate

    ox, oy = origin
    bx_abs, by_abs, bw, bh = candidate.bbox_xywh
    bx = int(round(float(bx_abs) - float(ox)))
    by = int(round(float(by_abs) - float(oy)))
    bw = max(1, int(bw))
    bh = max(1, int(bh))
    pad_x = int(round(max(4.0, float(bw) * max(0.15, float(config.refine_expand_x_ratio)))))
    pad_y = int(round(max(8.0, float(bh) * max(1.5, float(config.refine_expand_y_ratio)))))
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(crop_w, bx + bw + pad_x)
    y1 = min(crop_h, by + bh + pad_y)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return candidate

    local_bgr = np.asarray(crop_bgr, dtype=np.uint8)[y0:y1, x0:x1]
    local_mask = np.asarray(mask, dtype=np.uint8)[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(local_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = gray[local_mask]
    if values.size == 0:
        return candidate
    rough_cx = float(candidate.center_xy[0]) - float(ox) - float(x0)
    rough_cy = float(candidate.center_xy[1]) - float(oy) - float(y0)
    preferred_cy = max(
        rough_cy,
        float(by - y0) + (float(bh) - 1.0) * float(config.inner_strip_y_bias),
    )
    min_width = max(float(config.min_width_px), float(bw) * float(config.inner_strip_min_width_ratio))
    local_h, local_w = gray.shape[:2]
    core_x0 = max(0, bx - x0)
    core_x1 = min(local_w, bx + bw - x0)
    if core_x1 <= core_x0 + 1:
        core_x0, core_x1 = 0, local_w

    def make_local_binary(threshold: float) -> np.ndarray:
        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(gray >= float(threshold)) & local_mask] = 255
        close_w = max(3, int(round(float(bw) * 0.035)))
        if close_w % 2 == 0:
            close_w += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary[~local_mask] = 0
        return binary

    lower_cy = float(by - y0) + float(bh) * 0.30
    upper_cy = float(by - y0) + float(bh) * 1.35
    min_overlap = max(4.0, min_width * 0.35)
    threshold_candidates: list[float] = []
    for percentile in (98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0):
        threshold_candidates.append(float(np.percentile(values, percentile)))
    threshold_candidates.append(float(rough_threshold))
    unique_thresholds = sorted(
        {
            float(max(1.0, min(254.0, threshold)))
            for threshold in threshold_candidates
            if math.isfinite(float(threshold))
        },
        reverse=True,
    )

    for threshold in unique_thresholds:
        local_binary = make_local_binary(threshold)
        contours, _hierarchy = cv2.findContours(
            local_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        best: PinCenterCandidate | None = None
        best_score = -math.inf
        for contour in contours:
            left, top, width_i, height_i = [int(v) for v in cv2.boundingRect(contour)]
            if width_i <= 1 or height_i <= 1:
                continue
            width = float(width_i)
            height = float(height_i)
            if width < min_width or height < float(config.min_height_px):
                continue
            if height > max(42.0, float(bh) * 0.85):
                continue
            aspect = width / max(height, 1.0)
            if aspect < max(1.4, float(config.min_aspect_ratio) * 0.75):
                continue
            cx = float(left) + (width - 1.0) * 0.5
            cy = float(top) + (height - 1.0) * 0.5
            if cy < lower_cy or cy > upper_cy:
                continue
            overlap = float(max(0, min(left + width_i, core_x1) - max(left, core_x0)))
            if overlap < min(min_overlap, width * 0.7):
                continue
            area = max(float(abs(cv2.contourArea(contour))), width * height * 0.35)
            dx_penalty = abs(cx - rough_cx) / max(1.0, float(local_w))
            dy_penalty = abs(cy - preferred_cy) / max(1.0, float(local_h))
            fill = area / max(1.0, width * height)
            score = area * min(aspect, 20.0) * max(0.25, fill)
            score *= 1.0 - min(0.9, dx_penalty + dy_penalty * 1.75)
            score *= 1.0 + min(1.0, overlap / max(1.0, width)) * 0.35
            if score <= best_score:
                continue
            abs_left = float(x0 + left + ox)
            abs_right = float(x0 + left + width_i - 1 + ox)
            abs_top = float(y0 + top + oy)
            abs_bottom = float(y0 + top + height_i - 1 + oy)
            center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
            best = PinCenterCandidate(
                center_xy=(float(center[0]), float(center[1])),
                box_points=(
                    (abs_left, abs_top),
                    (abs_right, abs_top),
                    (abs_right, abs_bottom),
                    (abs_left, abs_bottom),
                ),
                area_px=float(area),
                bbox_xywh=(
                    int(round(abs_left)),
                    int(round(abs_top)),
                    max(1, int(round(width))),
                    max(1, int(round(height))),
                ),
                width_px=float(width),
                height_px=float(height),
                aspect_ratio=float(aspect),
                angle_deg=0.0,
                score=float(max(candidate.score, score)),
            )
            best_score = score
        if best is not None:
            return best

    threshold = max(1.0, min(254.0, float(np.percentile(values, 90.0))))
    local_binary = make_local_binary(threshold)

    best: PinCenterCandidate | None = None
    best_score = -math.inf
    core = local_binary[:, core_x0:core_x1] > 0
    core_valid = local_mask[:, core_x0:core_x1]
    valid_counts = np.count_nonzero(core_valid, axis=1)
    hit_counts = np.count_nonzero(core & core_valid, axis=1)
    min_row_hits = max(2, int(round(min_width * 0.72)))
    row_indices = np.where(hit_counts >= min_row_hits)[0]

    for top, bottom in _runs_from_indices(row_indices):
        if bottom < top:
            continue
        y_slice = slice(max(0, top), min(local_h, bottom + 1))
        body = local_binary[y_slice, :] > 0
        body_valid = local_mask[y_slice, :]
        valid_col_counts = np.count_nonzero(body_valid, axis=0)
        hit_col_counts = np.count_nonzero(body & body_valid, axis=0)
        min_col_hits = max(1, int(round(max(1, bottom - top + 1) * 0.45)))
        col_indices = np.where((hit_col_counts >= min_col_hits) & (valid_col_counts > 0))[0]
        col_run = _run_containing_or_nearest(col_indices, rough_cx)
        if col_run is None:
            continue
        left, right = col_run
        if right <= left:
            continue
        width = float(right - left + 1)
        height = float(bottom - top + 1)
        if width < min_width:
            continue
        aspect = width / max(height, 1.0)
        if aspect < max(1.2, float(config.min_aspect_ratio) * 0.65):
            continue
        if height > max(float(bh) * 1.8, 18.0):
            continue
        cx = float(left) + (width - 1.0) * 0.5
        cy = float(top) + (height - 1.0) * 0.5
        dx_penalty = abs(cx - rough_cx) / max(1.0, float(local_w))
        dy_penalty = abs(cy - preferred_cy) / max(1.0, float(local_h))
        upper_penalty = max(0.0, preferred_cy - cy) / max(1.0, float(local_h))
        lower_bonus = 1.0 + 6.0 * max(0.0, min(1.0, cy / max(1.0, float(local_h - 1))))
        area = float(width * height)
        score = area * min(aspect, 20.0) * lower_bonus
        score *= 1.0 - min(0.95, dx_penalty + dy_penalty * 2.0 + upper_penalty * 5.0)
        if score <= best_score:
            continue
        abs_left = float(x0 + left + ox)
        abs_right = float(x0 + right + ox)
        abs_top = float(y0 + top + oy)
        abs_bottom = float(y0 + bottom + oy)
        center = ((abs_left + abs_right) * 0.5, (abs_top + abs_bottom) * 0.5)
        best = PinCenterCandidate(
            center_xy=(float(center[0]), float(center[1])),
            box_points=(
                (abs_left, abs_top),
                (abs_right, abs_top),
                (abs_right, abs_bottom),
                (abs_left, abs_bottom),
            ),
            area_px=area,
            bbox_xywh=(
                int(round(abs_left)),
                int(round(abs_top)),
                max(1, int(round(width))),
                max(1, int(round(height))),
            ),
            width_px=width,
            height_px=height,
            aspect_ratio=float(aspect),
            angle_deg=0.0,
            score=float(max(candidate.score, score)),
        )
        best_score = score

    return best if best is not None else candidate


def _select_pin_center_pair(
    candidates: tuple[PinCenterCandidate, ...],
    *,
    config: PinCenterDistanceConfig,
    roi_size: tuple[int, int],
) -> tuple[PinCenterCandidate, PinCenterCandidate]:
    if len(candidates) < 2:
        raise RuntimeError(f"pin centers not found: {len(candidates)}/2")
    shortlist = tuple(candidates[: min(len(candidates), 24)])
    axis_index = 0 if config.sort_axis == "x" else 1
    axis_span = float(max(1, roi_size[0] if axis_index == 0 else roi_size[1]))
    best_pair: tuple[PinCenterCandidate, PinCenterCandidate] | None = None
    best_score = -math.inf
    for idx, left in enumerate(shortlist):
        for right in shortlist[idx + 1:]:
            separation = abs(float(right.center_xy[axis_index]) - float(left.center_xy[axis_index]))
            min_separation = _pin_pair_min_separation(
                left,
                right,
                axis_span=axis_span,
                axis_index=axis_index,
                config=config,
            )
            if separation < min_separation:
                continue
            normalized_separation = separation / axis_span
            pair_score = (float(left.score) + float(right.score)) * (
                1.0 + normalized_separation * normalized_separation * 4.0
            )
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (left, right)
    if best_pair is None:
        min_separation = axis_span * float(config.min_pair_separation_ratio)
        raise RuntimeError(
            f"pin center pair separation too small: need >= {min_separation:.1f}px"
        )
    return tuple(sorted(best_pair, key=lambda item: item.center_xy[axis_index]))  # type: ignore[return-value]


def _pin_pair_min_separation(
    left: PinCenterCandidate,
    right: PinCenterCandidate,
    *,
    axis_span: float,
    axis_index: int,
    config: PinCenterDistanceConfig,
) -> float:
    ratio_separation = float(axis_span) * float(config.min_pair_separation_ratio)
    if axis_index == 1:
        size_reference = max(float(left.width_px), float(right.width_px))
    else:
        size_reference = max(float(left.height_px), float(right.height_px))
    size_separation = size_reference * float(config.min_pair_separation_size_ratio)
    return max(float(ratio_separation), float(size_separation))


def _pin_center_distance_px(
    center_a: tuple[float, float],
    center_b: tuple[float, float],
    distance_mode: str,
) -> float:
    dx = float(center_b[0]) - float(center_a[0])
    dy = float(center_b[1]) - float(center_a[1])
    if distance_mode == "horizontal":
        return abs(dx)
    if distance_mode == "vertical":
        return abs(dy)
    return float(math.hypot(dx, dy))


def _select_bright_block_y_pair(
    vertical_candidates: tuple[PinCenterCandidate, ...],
    horizontal_candidates: tuple[PinCenterCandidate, ...],
    *,
    roi_size: tuple[int, int],
) -> tuple[PinCenterCandidate, PinCenterCandidate]:
    if not vertical_candidates:
        raise RuntimeError("bright vertical block not found")
    if not horizontal_candidates:
        raise RuntimeError("bright horizontal block not found")
    roi_w = float(max(1, int(roi_size[0])))
    min_x_separation = max(12.0, roi_w * 0.12)
    best_pair: tuple[PinCenterCandidate, PinCenterCandidate] | None = None
    best_score = -math.inf
    for vertical in vertical_candidates[:16]:
        for horizontal in horizontal_candidates[:16]:
            x_separation = float(horizontal.center_xy[0]) - float(vertical.center_xy[0])
            if x_separation < min_x_separation:
                continue
            y_delta = abs(float(horizontal.center_xy[1]) - float(vertical.center_xy[1]))
            score = (float(vertical.score) + float(horizontal.score)) * (1.0 + x_separation / roi_w)
            score *= 1.0 + min(1.0, y_delta / max(1.0, float(roi_size[1]))) * 0.25
            if score > best_score:
                best_score = score
                best_pair = (vertical, horizontal)
    if best_pair is None:
        raise RuntimeError(f"bright block pair x separation too small: need >= {min_x_separation:.1f}px")
    return best_pair


def _normalized_block_orientation(value: object) -> str:
    orientation = str(value or "auto").strip().lower()
    if orientation in {"h", "x", "strip"}:
        orientation = "horizontal"
    elif orientation in {"v", "y", "block"}:
        orientation = "vertical"
    if orientation not in {"auto", "horizontal", "vertical"}:
        orientation = "auto"
    return orientation


def _center_candidate_score(
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    roi_size: tuple[int, int],
) -> float:
    ox, oy = origin
    roi_w, roi_h = max(1.0, float(roi_size[0])), max(1.0, float(roi_size[1]))
    roi_cx = float(ox) + roi_w * 0.5
    roi_cy = float(oy) + roi_h * 0.5
    dx = abs(float(candidate.center_xy[0]) - roi_cx) / max(1.0, roi_w * 0.5)
    dy = abs(float(candidate.center_xy[1]) - roi_cy) / max(1.0, roi_h * 0.5)
    center_penalty = min(0.88, (dx + dy) * 0.28)
    return float(candidate.score) * (1.0 - center_penalty)


def _select_bright_block_center_candidate(
    horizontal_candidates: tuple[PinCenterCandidate, ...],
    vertical_candidates: tuple[PinCenterCandidate, ...],
    *,
    orientation: str,
    origin: tuple[int, int],
    roi_size: tuple[int, int],
) -> tuple[PinCenterCandidate, tuple[PinCenterCandidate, ...]]:
    if orientation == "horizontal":
        candidates = tuple(horizontal_candidates)
    elif orientation == "vertical":
        candidates = tuple(vertical_candidates)
    else:
        candidates = _dedupe_pin_candidates([*horizontal_candidates, *vertical_candidates])
    if not candidates:
        raise RuntimeError("bright block center not found")
    selected = max(
        candidates,
        key=lambda candidate: _center_candidate_score(candidate, origin=origin, roi_size=roi_size),
    )
    return selected, candidates


def _bright_block_center_has_adjacent_body(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    candidate: PinCenterCandidate,
    *,
    origin: tuple[int, int],
    params: Mapping[str, Any] | None = None,
) -> bool:
    payload = dict(params or {})
    if not _bool_param(payload.get("require_adjacent_body"), default=True):
        return True
    if float(candidate.height_px) <= float(candidate.width_px) * 1.6:
        return True

    gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    valid_mask = np.asarray(mask, dtype=np.uint8) > 0
    crop_h, crop_w = gray.shape[:2]
    valid_area = int(np.count_nonzero(valid_mask))
    if (
        valid_area > 0
        and float(candidate.height_px) >= max(40.0, float(crop_h) * 0.45)
        and float(candidate.width_px) >= max(10.0, float(crop_w) * 0.20)
        and float(candidate.area_px) >= max(900.0, float(valid_area) * 0.18)
    ):
        return True

    ox, oy = origin
    bx = int(round(float(candidate.bbox_xywh[0]) - float(ox)))
    by = int(round(float(candidate.bbox_xywh[1]) - float(oy)))
    bw = max(1, int(candidate.bbox_xywh[2]))
    bh = max(1, int(candidate.bbox_xywh[3]))
    x0 = max(0, bx + bw + 2)
    x1 = min(crop_w, int(round(float(crop_w))))
    y0 = max(0, int(round(float(by) + float(bh) * 0.20)))
    y1 = min(crop_h, int(round(float(by) + float(bh) * 0.80)))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return True

    patch = gray[y0:y1, x0:x1]
    patch_valid = valid_mask[y0:y1, x0:x1]
    values = patch[patch_valid]
    if values.size < 30:
        return True

    bright_threshold = int(max(1, min(255, float(payload.get("adjacent_body_threshold", 200.0) or 200.0))))
    min_pixels = int(max(1, float(payload.get("adjacent_body_min_pixels", 24.0) or 24.0)))
    min_ratio = max(0.0, min(1.0, float(payload.get("adjacent_body_min_ratio", 0.012) or 0.012)))
    bright_count = int(np.count_nonzero(values >= bright_threshold))
    required = max(min_pixels, int(round(float(values.size) * min_ratio)))
    return bright_count >= required


__all__ = [
    "_bright_block_center_has_adjacent_body",
    "_find_bright_vertical_block_candidates",
    "_find_pin_inner_strip_candidates",
    "_normalized_block_orientation",
    "_pin_center_binary",
    "_pin_center_distance_px",
    "_pin_pair_min_separation",
    "_select_bright_block_center_candidate",
    "_select_bright_block_y_pair",
    "_select_pin_center_pair",
]
