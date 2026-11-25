"""Bounded local translation in reference coordinates after a global pose.

Only small search windows are resampled. A successful result is a measured
translation of the saved reference anchor, not a defect assessment.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import os
from pathlib import Path
import threading
from typing import Any, Iterable

import cv2
import numpy as np

from algorithms import image_io


class GroupDiagnosticError(ValueError):
    """Language-neutral diagnostic identity, with a legacy log message."""

    def __init__(self, code: str, message: str, **params: Any) -> None:
        self.reason_code = code
        self.reason_params = params
        super().__init__(message)


def diagnostic_fields(error: Exception, fallback_code: str = "unexpected_error") -> dict[str, Any]:
    """Keep core/runtime independent of the UI and its selected language."""
    return {"reason": str(error), "reason_code": getattr(error, "reason_code", fallback_code),
            "reason_params": dict(getattr(error, "reason_params", {}))}


class LocalizationGroupError(RuntimeError):
    """A bound group failed; preview_result must never be used for inspection."""

    def __init__(self, group_results: list[dict[str, Any]], preview_result: Any = None) -> None:
        self.group_results = group_results
        self.preview_result = preview_result
        failures = [r for r in group_results if r.get("used", True) and not r.get("valid", False)]
        details = "; ".join(f"{r.get('name') or r.get('id')}: {r.get('reason', 'failed')}" for r in failures)
        super().__init__(f"定位组失败 / localization group failed: {details}")


# Cache only immutable reference data, never scene results or previous offsets.
# Both the filename and file fingerprint matter; replacing a reference cannot
# silently retain patches from the old image. Group rectangle changes key patches.
_CACHE_LOCK = threading.RLock()
_REFERENCE_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_PATCH_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()


def clear_reference_cache() -> None:
    with _CACHE_LOCK:
        _REFERENCE_CACHE.clear()
        _PATCH_CACHE.clear()


def _fingerprint(path: str) -> tuple:
    absolute = os.path.normcase(os.path.abspath(os.fspath(path)))
    stat = Path(absolute).stat()
    return absolute, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _reference_image(path: str) -> tuple[np.ndarray, tuple]:
    key = _fingerprint(path)
    with _CACHE_LOCK:
        cached = _REFERENCE_CACHE.get(key)
        if cached is not None:
            _REFERENCE_CACHE.move_to_end(key)
            return cached, key
    for _ in range(2):
        before = _fingerprint(path)
        gray = image_io.imread(path, cv2.IMREAD_GRAYSCALE)
        after = _fingerprint(path)
        if before == after:
            key = after
            break
    else:
        raise GroupDiagnosticError("reference_changed_during_read", "参考图在读取时发生变化")
    if gray is None or gray.ndim != 2 or gray.size == 0:
        raise GroupDiagnosticError("reference_read_failed", "无法读取定位组参考图")
    gray = np.ascontiguousarray(gray)
    gray.setflags(write=False)
    with _CACHE_LOCK:
        _REFERENCE_CACHE[key] = gray
        _REFERENCE_CACHE.move_to_end(key)
        while len(_REFERENCE_CACHE) > 4:
            old, _ = _REFERENCE_CACHE.popitem(last=False)
            for patch_key in list(_PATCH_CACHE):
                if patch_key[0] == old:
                    del _PATCH_CACHE[patch_key]
    return gray, key


def _anchor_rect(group: dict[str, Any], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    points = np.asarray(group.get("points", []), dtype=np.float64)
    if points.shape not in ((2, 2), (4, 2)) or not np.isfinite(points).all():
        raise GroupDiagnosticError("invalid_anchor", "请先画出有效的矩形定位区域")
    lo, hi = points.min(axis=0), points.max(axis=0)
    if len(points) == 4:
        expected = {(float(x), float(y)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])}
        if set(map(tuple, points)) != expected:
            raise GroupDiagnosticError("anchor_not_axis_aligned", "定位区域必须是轴对齐矩形")
    x0, y0 = np.floor(lo).astype(int)
    x1, y1 = np.ceil(hi).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise GroupDiagnosticError("anchor_too_small", "定位区域至少需要 8 × 8 像素", minimum=8)
    if x0 < 0 or y0 < 0 or x1 > shape[1] or y1 > shape[0]:
        raise GroupDiagnosticError("anchor_outside_reference", "定位区域超出参考图范围，请重新画区")
    return int(x0), int(y0), int(x1), int(y1)


def _reference_patch(gray: np.ndarray, key: tuple, rect: tuple[int, ...]) -> np.ndarray:
    cache_key = key, rect
    with _CACHE_LOCK:
        patch = _PATCH_CACHE.get(cache_key)
        if patch is not None:
            _PATCH_CACHE.move_to_end(cache_key)
            return patch
    x0, y0, x1, y1 = rect
    patch = np.ascontiguousarray(gray[y0:y1, x0:x1])
    if float(np.std(patch)) < 3.0:
        raise GroupDiagnosticError("low_texture", "定位区域纹理不足，请选择清晰且稳定的结构")
    patch.setflags(write=False)
    with _CACHE_LOCK:
        _PATCH_CACHE[cache_key] = patch
        _PATCH_CACHE.move_to_end(cache_key)
        while len(_PATCH_CACHE) > 32:
            _PATCH_CACHE.popitem(last=False)
    return patch


def _subpixel_peak(scores: np.ndarray, x: int, y: int) -> np.ndarray:
    point = np.array([x, y], dtype=np.float64)
    for axis in (0, 1):
        values = scores[y, x - 1:x + 2] if axis == 0 else scores[y - 1:y + 2, x]
        if len(values) != 3 or not np.isfinite(values).all():
            continue
        a, b, c = map(float, values)
        denominator = a - 2.0 * b + c
        if denominator < -1e-8:
            point[axis] += float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
    return point


def _sample_search(
    scene_gray: np.ndarray, transform: np.ndarray, rect: tuple[int, ...], radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = rect
    xx, yy = np.meshgrid(
        np.arange(x0 - radius, x1 + radius, dtype=np.float32),
        np.arange(y0 - radius, y1 + radius, dtype=np.float32),
    )
    map_x = (transform[0, 0] * xx + transform[0, 1] * yy + transform[0, 2]).astype(np.float32)
    map_y = (transform[1, 0] * xx + transform[1, 1] * yy + transform[1, 2]).astype(np.float32)
    valid = ((map_x >= 0) & (map_x <= scene_gray.shape[1] - 1) &
             (map_y >= 0) & (map_y <= scene_gray.shape[0] - 1))
    search = cv2.remap(scene_gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Every tested anchor must be fully inside the image. Padding is never
    # evidence for a match; the search border itself may extend outside.
    integral = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_64F)
    width, height = x1 - x0, y1 - y0
    counts = (integral[height:, width:] - integral[:-height, width:] -
              integral[height:, :-width] + integral[:-height, :-width])
    return search, counts / float(width * height)


def _locate_one(
    scene_gray: np.ndarray, reference_gray: np.ndarray, reference_key: tuple,
    group: dict[str, Any], transform: np.ndarray, result: dict[str, Any],
) -> None:
    if group.get("needs_rebuild", False):
        raise GroupDiagnosticError("reference_changed", "参考图已更换，请重新确认并保存此组定位区域")
    radius_value = float(group.get("search_radius", 64))
    minimum = float(group.get("min_score", 0.65))
    margin_min = float(group.get("min_peak_margin", 0.03))
    if not math.isfinite(radius_value) or radius_value != int(radius_value) or not 1 <= radius_value <= 512:
        raise GroupDiagnosticError("invalid_radius", "搜索半径必须是 1 到 512 的整数")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (minimum, margin_min)):
        raise GroupDiagnosticError("invalid_thresholds", "匹配分数和峰值差阈值必须在 0 到 1 之间")
    radius = int(radius_value)
    rect = _anchor_rect(group, reference_gray.shape)
    template = _reference_patch(reference_gray, reference_key, rect)
    search, fractions = _sample_search(scene_gray, transform, rect, radius)
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    valid = (fractions >= 1.0 - 1e-9) & np.isfinite(scores)
    if not valid.any():
        raise GroupDiagnosticError("no_valid_candidate", "预测定位区域在图像外，没有完整有效的搜索候选")
    candidates = np.where(valid, scores, -np.inf)
    y, x = np.unravel_index(int(np.argmax(candidates)), candidates.shape)
    score = float(candidates[y, x])
    result["score"] = score
    result["valid_fraction"] = float(fractions[y, x])
    result["delta_ref"] = [float(x - radius), float(y - radius)]
    result["delta_scene"] = (transform[:, :2] @ np.asarray(result["delta_ref"])).tolist()
    if score < minimum:
        raise GroupDiagnosticError("low_score", f"匹配分数不足 ({score:.3f} < {minimum:.3f})", score=score, minimum=minimum)
    if x == 0 or y == 0 or x == candidates.shape[1] - 1 or y == candidates.shape[0] - 1:
        raise GroupDiagnosticError("search_boundary", "最佳位置到达搜索边界，请检查定位区域或搜索半径")
    if not valid[y - 1:y + 2, x - 1:x + 2].all():
        raise GroupDiagnosticError("image_boundary", "最佳位置贴近有效图像边界，无法可靠细化")
    # Compare distinct local maxima. A point on the shoulder of a single broad
    # correlation peak is not a competing location. Flat/repeated maxima still
    # compete and therefore fail the margin check.
    exclusion = max(2, min(6, int(min(template.shape) * 0.08)))
    yy, xx = np.indices(candidates.shape)
    local_maxima = candidates >= cv2.dilate(candidates.astype(np.float32), np.ones((3, 3), np.uint8)) - 1e-6
    other_peak = local_maxima & ((xx - x) ** 2 + (yy - y) ** 2 > exclusion ** 2)
    alternative = np.where(other_peak, candidates, -np.inf)
    second = float(np.max(alternative))
    if not math.isfinite(second):
        second = -1.0
    margin = score - second
    result["second_score"] = second
    result["peak_margin"] = margin
    if margin < margin_min:
        raise GroupDiagnosticError("ambiguous_peak", f"存在相似候选，定位不唯一 (峰值差 {margin:.3f} < {margin_min:.3f})", margin=margin, minimum=margin_min)
    delta = _subpixel_peak(candidates, x, y) - radius
    result["delta_ref"] = delta.tolist()
    result["delta_scene"] = (transform[:, :2] @ delta).tolist()
    x0, y0, x1, y1 = rect
    corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float) + delta
    result["anchor_scene_points"] = (corners @ transform[:, :2].T + transform[:, 2]).tolist()
    result["valid"] = True
    result["reason"] = ""
    result["reason_code"] = "ok"
    result["reason_params"] = {}


def localize_groups(
    scene_bgr: np.ndarray, reference_path: str, groups: Iterable[dict[str, Any]],
    active_group_ids: Iterable[str], reference_to_scene: np.ndarray,
) -> list[dict[str, Any]]:
    """Return diagnostics for every group; callers must reject used invalid ones.

    ``reference_to_scene`` is a forward 2x3 affine matrix. A returned delta is
    in reference pixels; final points are T_global * (P_reference + delta_ref).
    Unbound groups are reported as unused and incur no image matching.
    """
    active = {str(value).strip() for value in active_group_ids if str(value).strip() not in ("", "global")}
    groups = list(groups)
    results: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        gid = str(group.get("id", "")).strip()
        if gid in by_id or not gid or gid == "global":
            raise GroupDiagnosticError("invalid_group_id", "定位组 ID 必须唯一且不能为空或 global")
        by_id[gid] = group
        results.append({"id": gid, "name": str(group.get("name") or gid), "used": gid in active,
                        "valid": gid not in active, "score": None, "peak_margin": None,
                        "delta_ref": [0.0, 0.0], "delta_scene": [0.0, 0.0],
                        "reason": "未绑定 ROI" if gid not in active else "",
                        "reason_code": "unused" if gid not in active else "", "reason_params": {}})
    for missing in sorted(active - set(by_id)):
        results.append({"id": missing, "name": missing, "used": True, "valid": False,
                        "score": None, "peak_margin": None, "delta_ref": [0.0, 0.0],
                        "delta_scene": [0.0, 0.0], "reason": "ROI 绑定的定位组不存在",
                        "reason_code": "missing_group", "reason_params": {}})
    if not active:
        return results
    try:
        transform = np.asarray(reference_to_scene, dtype=np.float64)
        if transform.shape != (2, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform[:, :2])) < 1e-8:
            raise GroupDiagnosticError("invalid_transform", "全局定位矩阵无效")
        if scene_bgr is None or scene_bgr.dtype != np.uint8:
            raise GroupDiagnosticError("invalid_scene", "定位组需要有效的 8 位检测图")
        if scene_bgr.ndim == 2:
            gray = scene_bgr
        elif scene_bgr.ndim == 3 and scene_bgr.shape[2] in (3, 4):
            gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGRA2GRAY if scene_bgr.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        else:
            raise GroupDiagnosticError("invalid_scene_channels", "检测图通道数无效")
        reference, key = _reference_image(reference_path)
    except (OSError, ValueError, cv2.error) as error:
        for result in results:
            if result["used"] and result["id"] in by_id:
                result.update(diagnostic_fields(error, "reference_read_failed" if isinstance(error, OSError) else "unexpected_error"))
        return results
    for result in results:
        if not result["used"] or result["id"] not in by_id:
            continue
        try:
            _locate_one(gray, reference, key, by_id[result["id"]], transform, result)
        except (ValueError, TypeError, cv2.error) as error:
            result["valid"] = False
            result.update(diagnostic_fields(error))
    return results


__all__ = ["GroupDiagnosticError", "LocalizationGroupError", "diagnostic_fields", "clear_reference_cache", "localize_groups"]
