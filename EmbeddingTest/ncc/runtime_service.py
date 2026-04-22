from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from algorithms.image_io import imread

from .interop.native_api import NccNativeApi, NccNativeMatcher
from .model import (
    NccAngleRange,
    NccMatchBoundingBox,
    NccMatchModel,
    NccMatchOptions,
    NccMatchResult,
    load_model,
    resolve_asset_path,
)


@dataclass(frozen=True)
class NccMatchResponse:
    matches: Tuple[NccMatchResult, ...]
    elapsed_ms: float
    backend_name: str


def _prepare_gray_image(image_bgr: np.ndarray, *, bitwise_not: bool = False) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is required")
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray.astype(np.uint8, copy=False))
    if bitwise_not:
        gray = cv2.bitwise_not(gray)
    return gray


def _clamp_xywh(xywh: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in xywh]
    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _crop_bgr(scene_bgr: np.ndarray, search_roi: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if not search_roi:
        return scene_bgr, (0, 0, int(scene_bgr.shape[1]), int(scene_bgr.shape[0]))
    x, y, w, h = _clamp_xywh(search_roi, int(scene_bgr.shape[1]), int(scene_bgr.shape[0]))
    return scene_bgr[y : y + h, x : x + w].copy(), (x, y, w, h)


def _build_angle_candidates(search: object) -> List[float]:
    if search is None:
        return [0.0]

    mode = str(getattr(search, "mode", "ranges") or "ranges").strip().lower()
    tolerance = max(0.0, float(getattr(search, "tolerance_angle", 0.0) or 0.0))
    ranges = list(getattr(search, "ranges", []) or [])
    if mode == "symmetric":
        ranges = [NccAngleRange(-tolerance, tolerance)]
    elif not ranges:
        ranges = [NccAngleRange(-180.0, 180.0)]

    angles: List[float] = []
    for item in ranges:
        start = float(getattr(item, "start", -180.0))
        end = float(getattr(item, "end", 180.0))
        if end < start:
            start, end = end, start
        span = end - start
        if span <= 0.0:
            angles.append(round(start, 4))
            continue
        if span <= 12.0:
            step = 1.0
        elif span <= 45.0:
            step = 2.0
        elif span <= 120.0:
            step = 3.0
        else:
            step = 5.0
        count = max(1, int(math.floor(span / step)))
        for index in range(count + 1):
            value = start + min(span, index * step)
            angles.append(round(value, 4))
        angles.append(round(end, 4))
    deduped = sorted({round(angle, 4) for angle in angles})
    if 0.0 not in deduped:
        deduped.append(0.0)
        deduped.sort()
    return deduped


def _rotate_template(template_gray: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    height, width = template_gray.shape[:2]
    if abs(float(angle_deg)) < 1e-6:
        quad = np.array(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )
        return template_gray, quad

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    bound_w = int(math.ceil(height * abs_sin + width * abs_cos))
    bound_h = int(math.ceil(height * abs_cos + width * abs_sin))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]

    border_value = 255 if float(template_gray.mean()) < 128.0 else 0
    rotated = cv2.warpAffine(
        template_gray,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width - 1), 0.0, 1.0],
            [float(width - 1), float(height - 1), 1.0],
            [0.0, float(height - 1), 1.0],
        ],
        dtype=np.float32,
    )
    transformed = corners @ matrix.T
    return rotated, transformed.astype(np.float32)


def _rotate_template_with_mask(
    template_gray: np.ndarray,
    mask_gray: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = template_gray.shape[:2]
    if abs(float(angle_deg)) < 1e-6:
        quad = np.array(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )
        return template_gray, mask_gray, quad

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    bound_w = int(math.ceil(height * abs_sin + width * abs_cos))
    bound_h = int(math.ceil(height * abs_cos + width * abs_sin))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]

    border_value = 255 if float(template_gray.mean()) < 128.0 else 0
    rotated = cv2.warpAffine(
        template_gray,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    rotated_mask = cv2.warpAffine(
        mask_gray,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width - 1), 0.0, 1.0],
            [float(width - 1), float(height - 1), 1.0],
            [0.0, float(height - 1), 1.0],
        ],
        dtype=np.float32,
    )
    transformed = corners @ matrix.T
    return rotated, rotated_mask, transformed.astype(np.float32)


def _bounding_box_from_quad(quad: Sequence[Tuple[float, float]] | np.ndarray) -> NccMatchBoundingBox:
    pts = np.asarray(quad, dtype=np.float32)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    return NccMatchBoundingBox(
        x=x_min,
        y=y_min,
        width=max(1.0, x_max - x_min),
        height=max(1.0, y_max - y_min),
    )


def _bbox_iou(a: NccMatchBoundingBox, b: NccMatchBoundingBox) -> float:
    ax1 = a.x + a.width
    ay1 = a.y + a.height
    bx1 = b.x + b.width
    by1 = b.y + b.height
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _nms_candidates(candidates: Iterable[NccMatchResult], max_overlap: float, limit: int) -> Tuple[NccMatchResult, ...]:
    selected: List[NccMatchResult] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_bbox_iou(candidate.bbox, current.bbox) > max_overlap for current in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _offset_match(match: NccMatchResult, dx: float, dy: float) -> NccMatchResult:
    quad = tuple((float(x + dx), float(y + dy)) for x, y in match.quad)
    bbox = NccMatchBoundingBox(
        x=match.bbox.x + dx,
        y=match.bbox.y + dy,
        width=match.bbox.width,
        height=match.bbox.height,
    )
    return NccMatchResult(
        score=match.score,
        angle=match.angle,
        center=(match.center[0] + dx, match.center[1] + dy),
        quad=quad,
        bbox=bbox,
    )


def _native_payload(options: NccMatchOptions) -> Dict[str, object]:
    normalized = options.normalized()
    return {
        "targetNum": normalized.target_num,
        "maxOverlap": normalized.max_overlap,
        "scoreThreshold": normalized.score_threshold,
        "useSimd": normalized.use_simd,
        "useSubpixel": normalized.use_subpixel,
        "bitwiseNot": normalized.bitwise_not,
        "stopLayer1": normalized.stop_layer1,
        "angleSearch": {
            "mode": str(normalized.angle_search.mode or "ranges"),
            "toleranceAngle": normalized.angle_search.tolerance_angle,
            "ranges": [
                {"start": item.start, "end": item.end}
                for item in normalized.angle_search.ranges
            ],
        },
    }


def _parse_native_response(payload: dict) -> Tuple[NccMatchResult, ...]:
    matches: List[NccMatchResult] = []
    for item in list(payload.get("matches", []) or []):
        if not isinstance(item, dict):
            continue
        quad_raw = list(item.get("quad", []) or [])
        quad: List[Tuple[float, float]] = []
        for point in quad_raw:
            if not isinstance(point, dict):
                continue
            quad.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
        if len(quad) < 4:
            continue
        bbox_raw = item.get("bbox", {}) or {}
        bbox = NccMatchBoundingBox(
            x=float(bbox_raw.get("x", 0.0)),
            y=float(bbox_raw.get("y", 0.0)),
            width=float(bbox_raw.get("width", 1.0)),
            height=float(bbox_raw.get("height", 1.0)),
        )
        center_raw = item.get("center", {}) or {}
        matches.append(
            NccMatchResult(
                score=float(item.get("score", 0.0)),
                angle=float(item.get("angle", 0.0)),
                center=(
                    float(center_raw.get("x", item.get("centerX", 0.0))),
                    float(center_raw.get("y", item.get("centerY", 0.0))),
                ),
                quad=tuple(quad),
                bbox=bbox,
            )
        )
    return tuple(matches)


def _match_python(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    options: NccMatchOptions,
    *,
    mask_gray: Optional[np.ndarray] = None,
) -> Tuple[NccMatchResult, ...]:
    prepared_scene = _prepare_gray_image(scene_gray, bitwise_not=options.bitwise_not)
    prepared_template = _prepare_gray_image(template_gray, bitwise_not=options.bitwise_not)
    if prepared_scene.shape[0] < prepared_template.shape[0] or prepared_scene.shape[1] < prepared_template.shape[1]:
        return tuple()
    prepared_mask: Optional[np.ndarray] = None
    if mask_gray is not None:
        prepared_mask = np.ascontiguousarray(mask_gray.astype(np.uint8, copy=False))
        if prepared_mask.shape[:2] != prepared_template.shape[:2]:
            raise ValueError("template mask shape must match template image")
        if int(cv2.countNonZero(prepared_mask)) <= 0:
            prepared_mask = None

    candidates: List[NccMatchResult] = []
    for angle in _build_angle_candidates(options.angle_search):
        rotated_mask: Optional[np.ndarray] = None
        if prepared_mask is not None:
            rotated_template, rotated_mask, quad_local = _rotate_template_with_mask(prepared_template, prepared_mask, angle)
            if int(cv2.countNonZero(rotated_mask)) <= 0:
                continue
        else:
            rotated_template, quad_local = _rotate_template(prepared_template, angle)
        if (
            rotated_template.shape[0] < 2
            or rotated_template.shape[1] < 2
            or rotated_template.shape[0] > prepared_scene.shape[0]
            or rotated_template.shape[1] > prepared_scene.shape[1]
        ):
            continue
        if rotated_mask is not None:
            response = cv2.matchTemplate(
                prepared_scene,
                rotated_template,
                cv2.TM_CCORR_NORMED,
                mask=rotated_mask,
            )
        else:
            response = cv2.matchTemplate(prepared_scene, rotated_template, cv2.TM_CCOEFF_NORMED)
        work = response.copy()
        per_angle_limit = max(options.target_num * 4, options.target_num)
        for _ in range(per_angle_limit):
            _, score, _, max_loc = cv2.minMaxLoc(work)
            if float(score) < float(options.score_threshold):
                break
            x, y = int(max_loc[0]), int(max_loc[1])
            quad = tuple((float(px + x), float(py + y)) for px, py in quad_local)
            bbox = _bounding_box_from_quad(quad)
            candidates.append(
                NccMatchResult(
                    score=float(score),
                    angle=float(angle),
                    center=(
                        bbox.x + bbox.width / 2.0,
                        bbox.y + bbox.height / 2.0,
                    ),
                    quad=quad,
                    bbox=bbox,
                )
            )
            suppress_w = max(1, int(round(rotated_template.shape[1] * (1.0 - options.max_overlap * 0.5))))
            suppress_h = max(1, int(round(rotated_template.shape[0] * (1.0 - options.max_overlap * 0.5))))
            x0 = max(0, x - suppress_w // 2)
            y0 = max(0, y - suppress_h // 2)
            x1 = min(work.shape[1], x + suppress_w)
            y1 = min(work.shape[0], y + suppress_h)
            work[y0:y1, x0:x1] = -1.0

    return _nms_candidates(candidates, options.max_overlap, options.target_num)


class NccCompiledModel:
    def __init__(self, model_path: str, model: Optional[NccMatchModel] = None) -> None:
        self.model_path = str(model_path)
        self.model = (model or load_model(model_path)).normalized()
        self._template_last_write_ns: int = -1
        self._template_gray: Optional[np.ndarray] = None
        self._template_mask_last_write_ns: int = -2
        self._template_mask_gray: Optional[np.ndarray] = None
        self._native_matcher_cache: Dict[int, NccNativeMatcher] = {}

    def close(self) -> None:
        for matcher in self._native_matcher_cache.values():
            matcher.close()
        self._native_matcher_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _template_file(self) -> Path:
        return resolve_asset_path(self.model_path, self.model.template_image_path)

    def _template_mask_file(self) -> Path:
        return resolve_asset_path(self.model_path, self.model.mask_image_path)

    def _load_template_gray(self) -> np.ndarray:
        template_path = self._template_file()
        if not template_path.exists():
            raise FileNotFoundError(template_path)
        last_write_ns = template_path.stat().st_mtime_ns
        if self._template_gray is not None and self._template_last_write_ns == last_write_ns:
            return self._template_gray
        image = imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read template image: {template_path}")
        self._template_gray = np.ascontiguousarray(image)
        self._template_last_write_ns = last_write_ns
        if self._native_matcher_cache:
            self.close()
        return self._template_gray

    def _load_template_mask_gray(self) -> Optional[np.ndarray]:
        if not bool(getattr(self.model, "template_mask_enabled", False)):
            self._template_mask_last_write_ns = -1
            self._template_mask_gray = None
            return None
        mask_path = self._template_mask_file()
        if not mask_path.exists():
            self._template_mask_last_write_ns = -1
            self._template_mask_gray = None
            return None
        last_write_ns = mask_path.stat().st_mtime_ns
        if self._template_mask_gray is not None and self._template_mask_last_write_ns == last_write_ns:
            return self._template_mask_gray
        image = imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read template mask image: {mask_path}")
        self._template_mask_gray = np.ascontiguousarray(image.astype(np.uint8, copy=False))
        self._template_mask_last_write_ns = last_write_ns
        return self._template_mask_gray

    def _get_native_matcher(self, min_reduced_area: int) -> NccNativeMatcher:
        template_gray = self._load_template_gray()
        key = max(16, int(min_reduced_area))
        matcher = self._native_matcher_cache.get(key)
        if matcher is not None:
            return matcher
        matcher = NccNativeMatcher(template_gray, min_reduced_area=key)
        self._native_matcher_cache[key] = matcher
        return matcher

    def match(
        self,
        scene_bgr: np.ndarray,
        options: Optional[NccMatchOptions] = None,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        *,
        prefer_native: bool = True,
    ) -> NccMatchResponse:
        if scene_bgr is None or scene_bgr.size == 0:
            raise ValueError("scene_bgr is required")
        normalized_options = (options or self.model.options).normalized()
        scene_crop, crop_xywh = _crop_bgr(scene_bgr, search_roi)
        template_gray = self._load_template_gray()
        template_mask_gray = self._load_template_mask_gray()

        t0 = time.perf_counter()
        backend_name = "python-ncc-mask" if template_mask_gray is not None else "python-ncc"
        matches: Tuple[NccMatchResult, ...]

        if template_mask_gray is None and prefer_native and NccNativeApi.is_available():
            try:
                matcher = self._get_native_matcher(normalized_options.min_reduced_area)
                source_gray = _prepare_gray_image(scene_crop, bitwise_not=normalized_options.bitwise_not)
                native_payload = matcher.match(source_gray, _native_payload(normalized_options))
                matches = _parse_native_response(native_payload)
                backend_name = "native-ncc"
            except Exception:
                matches = _match_python(scene_crop, template_gray, normalized_options)
        else:
            matches = _match_python(
                scene_crop,
                template_gray,
                normalized_options,
                mask_gray=template_mask_gray,
            )

        if crop_xywh[0] or crop_xywh[1]:
            matches = tuple(_offset_match(item, crop_xywh[0], crop_xywh[1]) for item in matches)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return NccMatchResponse(matches=matches, elapsed_ms=elapsed_ms, backend_name=backend_name)

    @staticmethod
    def render_matches(scene_bgr: np.ndarray, matches: Sequence[NccMatchResult], label: str = "") -> np.ndarray:
        canvas = np.ascontiguousarray(scene_bgr.copy())
        for index, item in enumerate(matches, start=1):
            points = np.array(item.quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
            text = f"{label}{index}: {item.score:.3f} / {item.angle:.1f}deg" if label else f"{item.score:.3f} / {item.angle:.1f}deg"
            anchor = (max(0, int(item.bbox.x)), max(18, int(item.bbox.y) - 6))
            cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas


__all__ = ["NccCompiledModel", "NccMatchResponse"]
