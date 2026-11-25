"""Detect observed checkerboard corners and their sparse integer grid topology.

This detector supports approximately horizontal/vertical square checkerboards,
including boards with small coded areas and cropped outer edges. It neither
decodes the codes nor creates observations where corners are missing. Strong
rotation, severe foreshortening and very small squares are rejected explicitly.
"""
from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree


def _gray8(image):
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("自动格点检测需要单通道灰度原图。")
    if min(image.shape) < 48 or image.dtype.kind not in "uif":
        raise ValueError("灰度图太小或数据类型不受支持。")
    if not np.isfinite(image).all():
        raise ValueError("灰度图包含非有限数值。")
    if image.dtype == np.uint8:
        gray = np.ascontiguousarray(image)
    else:
        low, high = np.percentile(image, [0.5, 99.5])
        if high <= low:
            raise ValueError("图像没有足够明暗变化，无法检测棋盘。")
        gray = np.clip((image.astype(np.float64) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    low, high = np.percentile(gray, [2, 98])
    if high - low < 35:
        raise ValueError("图像对比度不足，未发现可靠棋盘。")
    return gray, float(high - low)


def _periodic_steps(gray):
    """Estimate one-square pitch from the checkerboard's two-axis fundamental."""
    ratio = min(1.0, 1024.0 / max(gray.shape))
    width = max(1, int(round(gray.shape[1] * ratio)))
    height = max(1, int(round(gray.shape[0] * ratio)))
    small = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float64)
    small -= small.mean()
    small *= np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fft2(small)) ** 2
    steps, strengths = [], []
    for axis, size, scale in ((0, width, gray.shape[1] / width), (1, height, gray.shape[0] / height)):
        spectrum = power.sum(axis=axis)
        frequencies = np.abs(np.fft.fftfreq(size))
        usable = (frequencies > 2.0 / size) & (frequencies < 1.0 / 6.0)
        if not usable.any():
            raise ValueError("棋盘格数量或图像尺寸不足，无法确认横纵周期。")
        peak = int(np.argmax(np.where(usable, spectrum, -1.0)))
        strength = float(spectrum[peak] / max(float(np.median(spectrum[usable])), 1e-12))
        steps.append(float(0.5 / frequencies[peak] * scale))
        strengths.append(strength)
    if min(strengths) < 8:
        raise ValueError("未发现可靠的双向棋盘周期；空图、条纹和随机纹理不能用于棋盘标定。")
    if min(steps) < 7:
        raise ValueError("棋盘格在检测图中不足约 7 像素，无法可靠区分真实角点；请提高格子成像尺寸。")
    if max(steps) / min(steps) > 1.35:
        raise ValueError("横纵格距差异过大；自动检测仅支持接近正视、格轴接近水平或垂直的方格棋盘。")
    return np.array(steps), strengths


def _sample(gray, locations):
    locations = np.asarray(locations, np.float32)
    shape = locations.shape[:-1]
    flattened = locations.reshape(-1, 2)
    # OpenCV remap requires each output dimension to be below SHRT_MAX. A
    # dense board can have far more samples than corners, so bound each call.
    values = np.empty(len(flattened), np.float64)
    for begin in range(0, len(flattened), 16000):
        end = min(begin + 16000, len(flattened))
        batch = flattened[begin:end]
        values[begin:end] = cv2.remap(
            gray, batch[:, 0].reshape(-1, 1), batch[:, 1].reshape(-1, 1),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        ).ravel()
    return values.reshape(shape)


def _saddles(gray, pitch, contrast):
    minimum = float(min(pitch))
    block = min(9, max(3, int(round(minimum * 0.18)) | 1))
    expected = gray.size / float(np.prod(pitch))
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=min(30000, max(200, int(expected * 3 + 100))),
        qualityLevel=0.018, minDistance=max(2.0, minimum * 0.34), blockSize=block,
    )
    if corners is None:
        raise ValueError("未检测到棋盘交点。")
    corners = corners.reshape(-1, 2)
    window = min(7, max(2, int(round(minimum * 0.15))))
    cv2.cornerSubPix(gray, corners, (window, window), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 35, 0.001))
    margin = minimum * 0.34 + 2
    inside = ((corners[:, 0] > margin) & (corners[:, 0] < gray.shape[1] - 1 - margin)
              & (corners[:, 1] > margin) & (corners[:, 1] < gray.shape[0] - 1 - margin))
    corners = corners[inside]
    filtered = cv2.GaussianBlur(gray, (3, 3), 0.55)
    keep = np.ones(len(corners), bool)
    phase = None
    signs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float64)
    for fraction in (0.18, 0.28):
        values = _sample(filtered, corners[:, None, :] + signs[None] * minimum * fraction)
        diagonal = values[:, 0] + values[:, 3] - values[:, 1] - values[:, 2]
        disagreement = abs(values[:, 0] - values[:, 3]) + abs(values[:, 1] - values[:, 2])
        current_phase = np.sign(diagonal).astype(np.int8)
        keep &= (abs(diagonal) > contrast * 1.05) & (disagreement < contrast * 0.52)
        if phase is not None:
            keep &= phase == current_phase
        phase = current_phase
    return corners[keep], phase[keep], int(len(corners))


def _unit_edges(gray, points, phase, pitch, contrast):
    """Only connect actual one-square edges, with opposite saddle phases.

    Across a single checker edge, opposite-side intensities keep their sign
    throughout its interior. Skipping three squares would flip that sign; a
    two-square jump already fails the opposite-corner-phase requirement.
    """
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=min(36, len(points)))
    candidates = {}
    for index, nearby in enumerate(neighbors):
        delta = points[nearby] - points[index]
        for axis in (0, 1):
            major = abs(delta[:, axis])
            minor = abs(delta[:, 1 - axis])
            valid = ((major > pitch[axis] * 0.65) & (major < pitch[axis] * 1.35)
                     & (minor < major * 0.325) & (phase[nearby] == -phase[index]))
            for slot in np.where(valid)[0]:
                other = int(nearby[slot])
                pair = (min(index, other), max(index, other))
                candidates[pair] = axis
    if not candidates:
        raise ValueError("角点不能组成接近水平/垂直的单格邻接关系；请检查棋盘角度和清晰度。")
    pairs = np.asarray(list(candidates), np.int32)
    axes = np.asarray(list(candidates.values()), np.int8)
    start, end = points[pairs[:, 0]], points[pairs[:, 1]]
    vectors = end - start
    lengths = np.linalg.norm(vectors, axis=1)
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0])) / lengths[:, None]
    parameters = np.array([0.19, 0.34, 0.50, 0.66, 0.81])
    positions = start[:, None] + vectors[:, None] * parameters[None, :, None]
    offset = normals[:, None] * np.minimum(lengths * 0.14, 7.0)[:, None, None]
    difference = _sample(gray, positions + offset) - _sample(gray, positions - offset)
    valid = ((np.abs(difference) > contrast * 0.30).all(axis=1)
             & ((difference > 0).all(axis=1) | (difference < 0).all(axis=1)))
    pairs, axes, vectors, lengths = pairs[valid], axes[valid], vectors[valid], lengths[valid]
    if len(pairs) < 25:
        raise ValueError("能确认的一格棋盘边太少；编码、模糊或裁切使实际格距不可靠。")
    actual_steps = np.array([np.median(lengths[axes == axis]) if np.any(axes == axis) else np.nan for axis in (0, 1)])
    if not np.isfinite(actual_steps).all() or max(actual_steps) / min(actual_steps) > 1.30:
        raise ValueError("未确认相近的横纵单格步长；请使用接近正视的方格棋盘。")
    # Reject competing corners occupying one direction before assigning indices.
    best = {}
    for edge, (pair, axis, vector, length) in enumerate(zip(pairs, axes, vectors, lengths)):
        for source, target, direction in ((pair[0], pair[1], vector), (pair[1], pair[0], -vector)):
            sign = 1 if direction[axis] > 0 else -1
            key = (int(source), int(axis), sign)
            cost = abs(length / actual_steps[axis] - 1) + abs(direction[1 - axis]) / actual_steps[axis]
            if key not in best or cost < best[key][0]:
                best[key] = (cost, edge)
    selected = set(value[1] for value in best.values())
    accepted = []
    for edge in selected:
        pair, axis, vector = pairs[edge], int(axes[edge]), vectors[edge]
        step = np.zeros(2, np.int32)
        step[axis] = 1 if vector[axis] > 0 else -1
        accepted.append((int(pair[0]), int(pair[1]), step))
    return accepted, actual_steps, int(len(candidates)), int(valid.sum())


def _components(count, edges, allowed):
    adjacency = [[] for _ in range(count)]
    for source, target, step in edges:
        if allowed[source] and allowed[target]:
            adjacency[source].append((target, step))
            adjacency[target].append((source, -step))
    seen = {}
    components = []
    for seed in np.flatnonzero(allowed):
        if int(seed) in seen:
            continue
        seen[int(seed)] = np.zeros(2, np.int32)
        queue, members, conflicts = deque([int(seed)]), [], 0
        while queue:
            source = queue.popleft()
            members.append(source)
            for target, step in adjacency[source]:
                expected = seen[source] + step
                if target in seen:
                    conflicts += int(np.any(seen[target] != expected))
                else:
                    seen[target] = expected
                    queue.append(target)
        components.append((members, np.asarray([seen[index] for index in members]), conflicts))
    components.sort(key=lambda item: len(item[0]), reverse=True)
    return components, adjacency


def _local_geometry_mask(points, indices, step):
    """Exclude isolated corner structures inconsistent with their local lattice.

    A small, smoothly deformed checker neighborhood is locally affine. Predict
    each corner from other observed neighbors, never from itself or a global
    lens model. This separates displaced code fragments from genuine saddles.
    """
    lookup = {tuple(value): index for index, value in enumerate(indices)}
    result = np.ones(len(points), bool)
    for index, cell in enumerate(indices):
        neighbors = [lookup[tuple(cell + [dx, dy])]
                     for dy in range(-2, 3) for dx in range(-2, 3)
                     if (dx or dy) and tuple(cell + [dx, dy]) in lookup]
        if len(neighbors) < 6:
            continue
        relative = indices[neighbors] - cell
        design = np.column_stack((relative, np.ones(len(relative))))
        if np.linalg.matrix_rank(design) < 3:
            continue
        observed = points[neighbors]
        spatial_weights = 1.0 / (1.0 + np.sum(relative ** 2, axis=1))
        weights = spatial_weights.copy()
        for _ in range(3):
            square_root = np.sqrt(weights)
            affine = np.linalg.lstsq(design * square_root[:, None], observed * square_root[:, None], rcond=None)[0]
            error = np.linalg.norm(design @ affine - observed, axis=1)
            weights = spatial_weights * np.minimum(1.0, max(0.6, step * 0.025) / np.maximum(error, 1e-6))
        result[index] = np.linalg.norm(points[index] - affine[2]) <= max(0.8, step * 0.07)
    return result


def detect_checker_grid(gray, progress=None):
    """Return real full-resolution corners, sparse (column,row) indices and report.

    ``progress`` may accept ``(step, total, message)``. Coordinates use the input
    image's pixel-center convention, including after internal resizing. Indices
    have minimum zero on each axis; ``pattern_size`` is their span, not a count
    of a complete visible rectangle. Missing coded/occluded points are omitted.
    A ``ValueError`` explains unsupported or untrustworthy observations.
    """
    started = time.perf_counter()
    original, contrast = _gray8(gray)
    original_height, original_width = original.shape
    ratio = min(1.0, 2600.0 / max(original.shape))
    width, height = int(round(original_width * ratio)), int(round(original_height * ratio))
    work = original if ratio == 1 else cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
    scale_xy = np.array([original_width / width, original_height / height], np.float64)
    if progress:
        progress(0, 4, "正在确认横纵棋盘周期和真实鞍点…")
    pitch, periodic_strength = _periodic_steps(work)
    points, phase, candidate_count = _saddles(work, pitch, contrast)
    if len(points) < 20:
        raise ValueError("可靠棋盘鞍点不足 20 个；请检查格子大小、曝光和图案类型。")
    if progress:
        progress(1, 4, f"找到 {len(points)} 个候选鞍点，正在验证单格邻接与黑白交替…")
    edges, actual_steps, edge_candidates, confirmed_edges = _unit_edges(work, points, phase, pitch, contrast)
    allowed = np.ones(len(points), bool)
    duplicate_removed = 0
    for _ in range(3):
        components, adjacency = _components(len(points), edges, allowed)
        if not components or len(components[0][0]) < 20:
            raise ValueError("候选点不能组成至少 20 点的单一二维棋盘网格。")
        members, indices, conflicts = components[0]
        if conflicts:
            raise ValueError("棋盘邻接环路出现索引冲突，可能存在错点或跳格；已拒绝自动标定。")
        _, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
        duplicate = counts[inverse] > 1
        if not duplicate.any():
            break
        rejected = np.asarray(members)[duplicate]
        allowed[rejected] = False
        duplicate_removed += int(len(rejected))
    else:
        raise ValueError("棋盘格点索引重复，无法可靠排除编码区域的干扰。")
    if len(components) > 1 and len(components[1][0]) >= max(20, int(len(members) * 0.10)):
        raise ValueError("发现多个较大的棋盘连通区域，无法确认属于同一张标定板；请减少干扰或遮挡。")
    # Every returned point must have support in both grid directions. This drops
    # one-dimensional tails and code-like local fragments, using no fitted model.
    member_set = set(members)
    supported = []
    for index in members:
        axes = {int(np.flatnonzero(step)[0]) for neighbor, step in adjacency[index] if neighbor in member_set}
        if len(axes) == 2:
            supported.append(index)
    supported_set = set(supported)
    allowed = np.array([index in supported_set for index in range(len(points))])
    components, adjacency = _components(len(points), edges, allowed)
    if not components or len(components[0][0]) < 20:
        raise ValueError("缺少同时受横纵邻接支持的真实棋盘角点。")
    members, indices, conflicts = components[0]
    if conflicts or len(np.unique(indices, axis=0)) != len(indices):
        raise ValueError("局部筛选后的格点索引不一致，已拒绝自动标定。")
    if len(components) > 1 and len(components[1][0]) >= max(20, int(len(members) * 0.10)):
        raise ValueError("双向邻接筛选后出现多个较大网格，无法可靠确认同一棋盘。")
    # Refine before the local topology checks so filtering is evaluated on the
    # actual returned observations, including for internally downsampled input.
    refined_points = ((points.astype(np.float64) + 0.5) * scale_xy - 0.5).astype(np.float32)
    before_refine = refined_points.copy()
    full_step = float(min(actual_steps * scale_xy))
    window = min(9, max(2, int(round(full_step * 0.15))))
    cv2.cornerSubPix(original, refined_points, (window, window), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 35, 0.001))
    refinement_shift = np.linalg.norm(refined_points - before_refine, axis=1)
    if np.any(refinement_shift[members] > full_step * 0.18):
        raise ValueError("回到原始分辨率后部分角点移动过大，局部结构不可靠；请提高图像清晰度。")
    # Evaluate once against the original observed neighborhood. Repeatedly
    # rebuilding this neighborhood after deletion can erode genuine local
    # curvature in a flexible board, which this detector must not straighten.
    locally_consistent = _local_geometry_mask(refined_points[members], indices, full_step)
    local_structure_removed = int((~locally_consistent).sum())
    if local_structure_removed:
        if local_structure_removed > max(10, int(len(members) * 0.10)):
            raise ValueError("大量局部角点与相邻格距不一致，请检查编码干扰、图像清晰度或板面折痕。")
        allowed[np.asarray(members)[~locally_consistent]] = False
        components, adjacency = _components(len(points), edges, allowed)
        if not components or len(components[0][0]) < 20:
            raise ValueError("局部格距和角点结构不一致，无法确认可靠的二维棋盘。")
        members, indices, conflicts = components[0]
        if conflicts or len(np.unique(indices, axis=0)) != len(indices):
            raise ValueError("局部结构检查后的棋盘索引不一致。")
        if len(components) > 1 and len(components[1][0]) >= max(20, int(len(members) * 0.10)):
            raise ValueError("局部结构检查后棋盘分成多个较大区域，请检查遮挡或板面折痕。")
    indices -= indices.min(axis=0)
    pattern_size = tuple(int(value) for value in indices.max(axis=0) + 1)
    if min(pattern_size) < 4 or min(len(np.unique(indices[:, axis])) for axis in (0, 1)) < 4:
        raise ValueError("自动标定至少需要四行、四列实际观测到的棋盘交点。")
    density = len(indices) / float(np.prod(pattern_size))
    if density < 0.35:
        raise ValueError("棋盘网格过于稀疏，无法可靠区分缺点与跨格连接。")
    selected = refined_points[members]
    order = np.lexsort((indices[:, 0], indices[:, 1]))
    selected, indices = selected[order], indices[order].astype(np.int32)
    final_members = set(members)
    unit_counts = [0, 0]
    for source, target, step in edges:
        if source in final_members and target in final_members:
            unit_counts[int(np.flatnonzero(step)[0])] += 1
    step_confirmed = min(unit_counts) >= 8
    hull = cv2.convexHull(selected)
    diagnostics = {
        "detector": "saddle-grid-v1", "coordinate_order": "column,row",
        "grid_step_confirmed": bool(step_confirmed),
        "grid_step_evidence": "opposite saddle phase, constant edge-side contrast, local unit pitch and consistent grid loops",
        "candidate_count": candidate_count, "saddle_count": int(len(points)),
        "point_count": int(len(selected)), "pattern_size": list(pattern_size),
        "density": float(density), "duplicate_index_points_removed": duplicate_removed,
        "local_structure_points_removed": local_structure_removed,
        "local_structure_filter_iterations": 1,
        "discarded_point_count": int(len(points) - len(selected)),
        "component_sizes": [int(len(component[0])) for component in components[:8]],
        "loop_conflicts": int(conflicts), "unit_edge_counts": unit_counts,
        "edge_candidates": edge_candidates, "confirmed_edge_count": confirmed_edges,
        "step_x_px": float(actual_steps[0] * scale_xy[0]),
        "step_y_px": float(actual_steps[1] * scale_xy[1]),
        "periodic_peak_strength": periodic_strength,
        "coverage_fraction": float(cv2.contourArea(hull) / original.size),
        "internal_size": [width, height], "input_size": [original_width, original_height],
        "maximum_axis_angle_degrees": float(np.degrees(np.arctan(0.325))),
        "coded_regions_decoded": False, "missing_points_filled": False,
        "elapsed_ms": float((time.perf_counter() - started) * 1000),
    }
    if progress:
        progress(4, 4, f"已确认 {len(selected)} 个真实角点，格网跨度 {pattern_size[0]}×{pattern_size[1]}，未补缺点。")
    return selected, indices, pattern_size, diagnostics


__all__ = ["detect_checker_grid"]
