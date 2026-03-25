#!/usr/bin/env python3
"""
line2Dup/shape_based_matching style matcher in Python (OpenCV + NumPy).

This follows the core ideas from:
https://github.com/meiqua/shape_based_matching
"""

from __future__ import annotations

import argparse
import base64
import copy
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Feature:
    x: int
    y: int
    label: int
    theta: float = 0.0


@dataclass
class TemplateLevel:
    width: int = -1
    height: int = -1
    tl_x: int = 0
    tl_y: int = 0
    pyramid_level: int = 0
    features: List[Feature] = None

    def __post_init__(self) -> None:
        if self.features is None:
            self.features = []


@dataclass
class Match:
    x: int
    y: int
    similarity: float
    class_id: str
    template_id: int
    backend: str = "original"
    refined_transform: Optional[np.ndarray] = None
    refined_scale: Optional[float] = None
    refined_angle_deg: Optional[float] = None
    refined_fitness: Optional[float] = None
    refined_rmse: Optional[float] = None
    refined_quad: Optional[List[Tuple[float, float]]] = None


@dataclass
class ShapeInfo:
    angle: float
    scale: float


@dataclass
class SceneLevelData:
    width: int
    height: int
    T: int
    response_maps: List[np.ndarray]
    linear_memories: List[np.ndarray]
    memory_width: int
    memory_height: int


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_mask(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask is None:
        return np.full((h, w), 255, dtype=np.uint8)
    if mask.ndim == 3:
        mask = to_gray(mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def get_label(one_hot: int) -> int:
    if one_hot <= 0:
        raise ValueError("one_hot must be positive")
    return int(int(one_hot).bit_length() - 1)


def theta_deg_to_label(theta_deg: float) -> int:
    """Map angle (deg) to the same 8-bin label style used by auto extraction."""
    if not np.isfinite(theta_deg):
        return 0
    a = float(theta_deg) % 360.0
    q16 = int(a * (16.0 / 360.0))
    q16 = max(0, min(15, q16))
    return int(q16 & 7)


def label_to_theta_deg(label: int) -> float:
    """Return a canonical angle (deg) near the center of the folded 8-bin class."""
    return (float(int(label) & 7) + 0.5) * 22.5


def _fallback_similarity_lut() -> np.ndarray:
    lut = np.zeros((256,), dtype=np.uint8)
    for ori in range(8):
        base = ori * 32
        for nibble in range(16):
            best = 0
            for bit in range(4):
                if nibble & (1 << bit):
                    d = abs((ori % 4) - bit)
                    d = min(d, 4 - d)
                    score = 4 if d == 0 else (3 if d == 1 else 0)
                    best = max(best, score)
            lut[base + nibble] = np.uint8(best)

        for nibble in range(16):
            best = 0
            for bit in range(4):
                if nibble & (1 << bit):
                    label = bit + 4
                    d = abs(ori - label)
                    d = min(d, 8 - d)
                    score = 4 if d == 0 else (3 if d == 1 else 0)
                    best = max(best, score)
            lut[base + 16 + nibble] = np.uint8(best)
    return lut


def _load_similarity_lut() -> np.ndarray:
    cpp_path = Path(__file__).resolve().parents[1] / "vendor" / "_third_party_shape_based_matching" / "line2Dup.cpp"
    if not cpp_path.exists():
        return _fallback_similarity_lut()

    text = cpp_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"SIMILARITY_LUT\[256\]\s*=\s*\{([^}]*)\};", text, flags=re.S)
    if not match:
        return _fallback_similarity_lut()

    raw = match.group(1).replace("LUT3", "3")
    vals: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(int(token))
        except ValueError:
            return _fallback_similarity_lut()
    if len(vals) != 256:
        return _fallback_similarity_lut()
    return np.array(vals, dtype=np.uint8)


SIMILARITY_LUT = _load_similarity_lut()

_DEFAULT_OPENCV_BUILD_ROOT = Path(r"C:\Users\ADMIN\tools\opencv\build")
NATIVE_BACKEND_TO_MODULE = {
    "original": "line2dup_native",
    "fusion": "line2dup_fusion_native",
    "fusionv2": "line2dup_fusionv2_native",
    "sim3": "line2dup_sim3_native",
}
_NATIVE_MODULES: Dict[str, Any] = {}
_OPENCV_DLL_HANDLE: Any = None
_NATIVE_MODULE_ERRORS: Dict[str, BaseException] = {}
_NATIVE_FALLBACK_WARNED: set[str] = set()


def _opencv_build_root() -> Path:
    override = os.environ.get("LINE2DUP_OPENCV_BUILD", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_OPENCV_BUILD_ROOT


def _native_build_instructions() -> str:
    return (
        "To enable the OpenCV-backed accelerators:\n"
        "py -3 -m pip install -U setuptools wheel pybind11\n"
        "py -3 EmbeddingTest\\setup.py build_ext --inplace\n"
        f"Set LINE2DUP_OPENCV_BUILD if OpenCV is not installed at {_DEFAULT_OPENCV_BUILD_ROOT}.\n"
        "Optional: set LINE2DUP_OPENCV_WORLD_LIB when your OpenCV world library name is not auto-detected."
    )


def _native_fallback_warn_enabled() -> bool:
    value = os.environ.get("LINE2DUP_WARN_NATIVE_FALLBACK", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _normalize_backend_name(backend: str) -> str:
    key = str(backend or "original").strip().lower()
    if key in {"orig", "original", "native"}:
        return "original"
    if key in {"fusion", "fused"}:
        return "fusion"
    if key in {"fusionv2", "fusion_v2", "fused_v2"}:
        return "fusionv2"
    if key in {"sim3", "icp", "icp(sim3)", "sim3_icp"}:
        return "sim3"
    raise ValueError(f"Unsupported backend: {backend}")


def _warn_native_fallback(backend: str, exc: BaseException) -> None:
    backend = _normalize_backend_name(backend)
    if backend in _NATIVE_FALLBACK_WARNED:
        return
    if not _native_fallback_warn_enabled():
        _NATIVE_FALLBACK_WARNED.add(backend)
        return
    print(
        f"{NATIVE_BACKEND_TO_MODULE[backend]} is unavailable; falling back to the slower Python matcher.\n"
        f"{_native_build_instructions()}\n"
        f"Original import error: {exc}",
        file=sys.stderr,
    )
    _NATIVE_FALLBACK_WARNED.add(backend)


def _load_native_matcher(backend: str = "original", required: bool = True) -> Any:
    backend = _normalize_backend_name(backend)
    module_name = NATIVE_BACKEND_TO_MODULE[backend]
    global _OPENCV_DLL_HANDLE
    if backend in _NATIVE_MODULES:
        return _NATIVE_MODULES[backend]
    if backend in _NATIVE_MODULE_ERRORS:
        if required:
            raise RuntimeError(
                f"{module_name} is unavailable.\n{_native_build_instructions()}\n"
                f"Original import error: {_NATIVE_MODULE_ERRORS[backend]}"
            )
        return None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_dir = _opencv_build_root() / "x64" / "vc16" / "bin"
        if dll_dir.exists() and _OPENCV_DLL_HANDLE is None:
            _OPENCV_DLL_HANDLE = os.add_dll_directory(str(dll_dir))
    try:
        _NATIVE_MODULES[backend] = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised via runtime import failures
        _NATIVE_MODULE_ERRORS[backend] = exc
        if required:
            raise RuntimeError(f"{module_name} is unavailable.\n{_native_build_instructions()}\nOriginal import error: {exc}") from exc
        return None
    return _NATIVE_MODULES[backend]


def ensure_native_backends_available(backends: Sequence[str] = ("original", "fusion", "fusionv2", "sim3")) -> None:
    for backend in backends:
        _load_native_matcher(backend=backend, required=True)


def create_native_detector(
    num_features: int,
    T_levels: Sequence[int],
    weak_threshold: float,
    strong_threshold: float,
    *,
    backend: str = "original",
) -> Any:
    native_matcher = _load_native_matcher(backend=backend, required=True)
    return native_matcher.NativeDetector(
        int(num_features),
        [int(t) for t in T_levels],
        float(weak_threshold),
        float(strong_threshold),
    )


def hysteresis_gradient(
    magnitude: np.ndarray,
    angle_deg: np.ndarray,
    threshold_sq: float,
) -> np.ndarray:
    quant_unfiltered = np.clip((angle_deg * (16.0 / 360.0)).astype(np.int32), 0, 15).astype(np.uint8)

    quant_unfiltered[0, :] = 0
    quant_unfiltered[-1, :] = 0
    quant_unfiltered[:, 0] = 0
    quant_unfiltered[:, -1] = 0

    quant_unfiltered &= np.uint8(7)

    counts = []
    for i in range(8):
        mask_i = (quant_unfiltered == i).astype(np.uint8)
        cnt_i = cv2.boxFilter(mask_i, ddepth=cv2.CV_8U, ksize=(3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        counts.append(cnt_i)
    counts_stacked = np.stack(counts, axis=-1)
    max_votes = counts_stacked.max(axis=-1)
    max_idx = counts_stacked.argmax(axis=-1).astype(np.uint8)

    valid = magnitude > threshold_sq
    valid[0, :] = False
    valid[-1, :] = False
    valid[:, 0] = False
    valid[:, -1] = False
    valid &= max_votes >= 5

    out = np.zeros_like(quant_unfiltered, dtype=np.uint8)
    out[valid] = np.left_shift(np.uint8(1), max_idx[valid])
    return out


def quantized_orientations(src: np.ndarray, weak_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = cv2.GaussianBlur(src, (7, 7), 0, 0, borderType=cv2.BORDER_REPLICATE)

    if smoothed.ndim == 2:
        dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
        dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        magnitude = dx * dx + dy * dy
        angle_ori = cv2.phase(dx, dy, angleInDegrees=True)
    else:
        channels = cv2.split(smoothed)
        dxs = [cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE) for ch in channels]
        dys = [cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE) for ch in channels]
        mags = [dx * dx + dy * dy for dx, dy in zip(dxs, dys)]
        mags_stacked = np.stack(mags, axis=-1)
        idx = mags_stacked.argmax(axis=-1)
        magnitude = mags_stacked.max(axis=-1)

        dx_stack = np.stack(dxs, axis=-1)
        dy_stack = np.stack(dys, axis=-1)
        dx = np.take_along_axis(dx_stack, idx[..., None], axis=-1)[..., 0]
        dy = np.take_along_axis(dy_stack, idx[..., None], axis=-1)[..., 0]
        angle_ori = cv2.phase(dx, dy, angleInDegrees=True)

    quantized = hysteresis_gradient(magnitude, angle_ori, weak_threshold * weak_threshold)
    return magnitude, quantized, angle_ori


def spread_or(src: np.ndarray, T: int) -> np.ndarray:
    h, w = src.shape[:2]
    dst = np.zeros((h, w), dtype=np.uint8)
    for r in range(T):
        for c in range(T):
            dst[: h - r, : w - c] |= src[r:, c:]
    return dst


def compute_response_maps(spread_img: np.ndarray) -> List[np.ndarray]:
    low = spread_img & np.uint8(15)
    high = np.right_shift(spread_img, 4)

    response_maps: List[np.ndarray] = []
    for ori in range(8):
        base = ori * 32
        lut_local = SIMILARITY_LUT[base : base + 32]
        low_res = lut_local[low]
        high_res = lut_local[16 + high]
        resp = np.maximum(low_res, high_res).astype(np.uint8)
        response_maps.append(resp)
    return response_maps


def linearize_response_map(response_map: np.ndarray, T: int) -> np.ndarray:
    h, w = response_map.shape[:2]
    if h % T != 0 or w % T != 0:
        raise ValueError("response_map shape must be divisible by T")
    mem_h = h // T
    mem_w = w // T
    linearized = response_map.reshape(mem_h, T, mem_w, T).transpose(1, 3, 0, 2).reshape(T * T, mem_h * mem_w)
    return np.ascontiguousarray(linearized)


def access_linear_memory(
    linear_memories: Sequence[np.ndarray],
    label: int,
    x: int,
    y: int,
    T: int,
    memory_width: int,
) -> Tuple[np.ndarray, int]:
    memory_grid = linear_memories[label]
    grid_index = (y % T) * T + (x % T)
    memory = memory_grid[grid_index]
    lm_index = (y // T) * memory_width + (x // T)
    return memory, lm_index


def memory_patch_view(
    memory: np.ndarray,
    base_index: int,
    rows: int,
    cols: int,
    row_stride: int,
) -> Optional[np.ndarray]:
    if rows <= 0 or cols <= 0 or base_index < 0:
        return None
    required = base_index + (rows - 1) * row_stride + cols
    if required > memory.size:
        return None
    base = memory[base_index:required]
    itemsize = base.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        base,
        shape=(rows, cols),
        strides=(row_stride * itemsize, itemsize),
        writeable=False,
    )


def accumulator_dtype(num_features: int) -> np.dtype:
    return np.uint8 if int(num_features) < 64 else np.uint16


def raw_similarity_threshold(threshold: float, num_features: int) -> float:
    return float(threshold) * (4.0 * max(1, int(num_features))) / 100.0


def similarity_from_raw(raw_score: float, num_features: int) -> float:
    return float(raw_score) * (100.0 / (4.0 * max(1, int(num_features))))


def select_scattered_features(
    candidates: List[Tuple[float, Feature]],
    num_features: int,
    distance: float,
) -> List[Feature]:
    if not candidates:
        return []
    features: List[Feature] = []
    distance_sq = distance * distance
    i = 0
    first_select = True

    while True:
        score, cand = candidates[i]
        keep = True
        for f in features:
            dx = cand.x - f.x
            dy = cand.y - f.y
            if dx * dx + dy * dy < distance_sq:
                keep = False
                break
        if keep:
            features.append(cand)

        i += 1
        if i == len(candidates):
            num_ok = len(features) >= num_features
            if first_select:
                if num_ok:
                    features = []
                    i = 0
                    distance += 1.0
                    distance_sq = distance * distance
                    continue
                first_select = False

            i = 0
            distance -= 1.0
            distance_sq = distance * distance
            if num_ok or distance < 3.0:
                break
    return features


def crop_templates(template_levels: List[TemplateLevel]) -> None:
    min_x = 10**9
    min_y = 10**9
    max_x = -10**9
    max_y = -10**9

    for templ in template_levels:
        for f in templ.features:
            x = f.x << templ.pyramid_level
            y = f.y << templ.pyramid_level
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if min_x % 2 == 1:
        min_x -= 1
    if min_y % 2 == 1:
        min_y -= 1

    for templ in template_levels:
        shift = 1 << templ.pyramid_level
        templ.width = (max_x - min_x) // shift
        templ.height = (max_y - min_y) // shift
        templ.tl_x = min_x // shift
        templ.tl_y = min_y // shift
        for f in templ.features:
            f.x -= templ.tl_x
            f.y -= templ.tl_y


def rotate_point(in_point: Tuple[float, float], center: Tuple[float, float], ang_rad: float) -> Tuple[float, float]:
    px, py = in_point
    cx, cy = center
    qx = px - cx
    qy = py - cy
    rx = math.cos(ang_rad) * qx - math.sin(ang_rad) * qy
    ry = math.sin(ang_rad) * qx + math.cos(ang_rad) * qy
    return rx + cx, ry + cy


def offset_from_T(T: int) -> int:
    return T // 2 + (T % 2 - 1)


class ColorGradientPyramid:
    def __init__(
        self,
        src: np.ndarray,
        mask: Optional[np.ndarray],
        weak_threshold: float,
        num_features: int,
        strong_threshold: float,
    ) -> None:
        self.src = src
        self.mask = ensure_mask(mask, src.shape[:2])
        self.weak_threshold = weak_threshold
        self.num_features = int(num_features)
        self.strong_threshold = strong_threshold
        self.pyramid_level = 0
        self.angle: np.ndarray = np.empty((0, 0), dtype=np.uint8)
        self.magnitude: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.angle_ori: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.update()

    def update(self) -> None:
        magnitude, quantized_angle, angle_ori = quantized_orientations(self.src, self.weak_threshold)
        self.magnitude = magnitude
        self.angle = quantized_angle
        self.angle_ori = angle_ori

    def pyr_down(self) -> None:
        self.num_features = max(1, self.num_features // 2)
        self.src = cv2.pyrDown(self.src)
        self.mask = cv2.resize(self.mask, (self.src.shape[1], self.src.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.pyramid_level += 1
        self.update()

    def quantize(self) -> np.ndarray:
        dst = np.zeros_like(self.angle, dtype=np.uint8)
        dst[self.mask > 0] = self.angle[self.mask > 0]
        return dst

    def extract_template(self) -> Optional[TemplateLevel]:
        local_mask: Optional[np.ndarray] = None
        if self.mask is not None and self.mask.size > 0:
            local_mask = cv2.erode(self.mask, np.ones((3, 3), dtype=np.uint8), iterations=1, borderType=cv2.BORDER_REPLICATE)

        no_mask = local_mask is None
        threshold_sq = self.strong_threshold * self.strong_threshold

        nms_kernel = np.ones((5, 5), dtype=np.uint8)
        local_max = cv2.dilate(self.magnitude, nms_kernel)
        is_peak = self.magnitude >= local_max

        valid = is_peak & (self.magnitude > threshold_sq) & (self.angle > 0)
        if not no_mask and local_mask is not None:
            valid &= local_mask > 0

        ys, xs = np.where(valid)
        candidates: List[Tuple[float, Feature]] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            one_hot = int(self.angle[y, x])
            if one_hot <= 0:
                continue
            label = get_label(one_hot)
            score = float(self.magnitude[y, x])
            theta = float(self.angle_ori[y, x])
            candidates.append((score, Feature(x=x, y=y, label=label, theta=theta)))

        if len(candidates) < self.num_features:
            if len(candidates) <= 4:
                return None

        candidates.sort(key=lambda s: s[0], reverse=True)
        distance = float(len(candidates) / max(1, self.num_features) + 1)
        features = select_scattered_features(candidates, self.num_features, distance)
        if len(features) == 0:
            return None

        templ = TemplateLevel(
            width=-1,
            height=-1,
            tl_x=0,
            tl_y=0,
            pyramid_level=self.pyramid_level,
            features=features,
        )
        return templ


def similarity_full(
    scene_level: SceneLevelData,
    templ: TemplateLevel,
) -> np.ndarray:
    if not templ.features:
        return np.empty((0, 0), dtype=np.uint16)

    T = scene_level.T
    W = scene_level.memory_width
    H = scene_level.memory_height
    wf = (templ.width - 1) // T + 1
    hf = (templ.height - 1) // T + 1
    span_x = W - wf
    span_y = H - hf
    if span_x < 0 or span_y < 0:
        return np.empty((0, 0), dtype=np.uint16)

    dst = np.zeros((span_y + 1, span_x + 1), dtype=accumulator_dtype(len(templ.features)))

    for f in templ.features:
        if f.label < 0 or f.label >= len(scene_level.linear_memories):
            continue
        if f.x < 0 or f.y < 0 or f.x >= scene_level.width or f.y >= scene_level.height:
            continue
        memory, base_index = access_linear_memory(
            scene_level.linear_memories,
            f.label,
            f.x,
            f.y,
            T,
            W,
        )
        patch = memory_patch_view(memory, base_index, span_y + 1, span_x + 1, W)
        if patch is None:
            continue
        np.add(dst, patch, out=dst, casting="unsafe")
    return dst


def similarity_local(
    scene_level: SceneLevelData,
    templ: TemplateLevel,
    center_xy: Tuple[int, int],
) -> np.ndarray:
    if not templ.features:
        return np.empty((0, 0), dtype=np.uint16)

    T = scene_level.T
    center_x, center_y = center_xy
    offset_x = (center_x // T - 8) * T
    offset_y = (center_y // T - 8) * T
    dst = np.zeros((16, 16), dtype=accumulator_dtype(len(templ.features)))

    for f in templ.features:
        if f.label < 0 or f.label >= len(scene_level.linear_memories):
            continue
        x0 = f.x + offset_x
        y0 = f.y + offset_y
        if x0 < 0 or y0 < 0 or x0 >= scene_level.width or y0 >= scene_level.height:
            continue
        memory, base_index = access_linear_memory(
            scene_level.linear_memories,
            f.label,
            x0,
            y0,
            T,
            scene_level.memory_width,
        )
        patch = memory_patch_view(memory, base_index, 16, 16, scene_level.memory_width)
        if patch is None:
            continue
        np.add(dst, patch, out=dst, casting="unsafe")
    return dst


class Line2DupLikeDetector:
    def __init__(
        self,
        num_features: int = 128,
        T_levels: Sequence[int] = (4, 8),
        weak_threshold: float = 30.0,
        strong_threshold: float = 60.0,
    ) -> None:
        self.num_features = int(num_features)
        self.T_at_level = [int(t) for t in T_levels]
        self.pyramid_levels = len(self.T_at_level)
        self.weak_threshold = float(weak_threshold)
        self.strong_threshold = float(strong_threshold)
        self.backend_templates: Dict[str, Dict[str, List[List[TemplateLevel]]]] = {
            backend: {} for backend in NATIVE_BACKEND_TO_MODULE
        }
        self.class_templates = self.backend_templates["original"]
        self.class_meta: Dict[str, List[Dict[str, float]]] = {}
        self.class_sources: Dict[str, Dict[str, Any]] = {}
        self.original_editor_levels: Dict[str, List[TemplateLevel]] = {}
        self._native_detectors: Dict[str, Any] = {}
        self._native_dirty = True

    def _reset_template_storage(self) -> None:
        self.backend_templates = {backend: {} for backend in NATIVE_BACKEND_TO_MODULE}
        self.class_templates = self.backend_templates["original"]

    def invalidate_native_cache(self, class_id: Optional[str] = None) -> None:
        self._native_dirty = True
        self._native_detectors.clear()

    def _create_native_detector(self, backend: str = "original", required: bool = True) -> Any:
        native_matcher = _load_native_matcher(backend=backend, required=required)
        if native_matcher is None:
            return None
        return native_matcher.NativeDetector(
            int(self.num_features),
            [int(t) for t in self.T_at_level],
            float(self.weak_threshold),
            float(self.strong_threshold),
        )

    @staticmethod
    def _template_pyramid_to_native(tp: Sequence[TemplateLevel]) -> List[Dict[str, Any]]:
        return [_template_level_to_dict(level) for level in tp]

    @staticmethod
    def _template_pyramid_from_native(levels_raw: Sequence[Dict[str, Any]]) -> List[TemplateLevel]:
        return [_template_level_from_dict(level) for level in levels_raw]

    def _ensure_class_containers(self, class_id: str) -> None:
        for backend_store in self.backend_templates.values():
            if class_id not in backend_store:
                backend_store[class_id] = []
        if class_id not in self.class_meta:
            self.class_meta[class_id] = []

    def _backend_store(self, backend: str) -> Dict[str, List[List[TemplateLevel]]]:
        return self.backend_templates[_normalize_backend_name(backend)]

    def _ensure_class_meta_length(self, class_id: str, size: int) -> None:
        self._ensure_class_containers(class_id)
        while len(self.class_meta[class_id]) < size:
            self.class_meta[class_id].append({})

    def _set_template_meta(self, class_id: str, template_id: int, metadata: Optional[Dict[str, float]]) -> None:
        self._ensure_class_meta_length(class_id, template_id + 1)
        self.class_meta[class_id][template_id] = dict(metadata) if isinstance(metadata, dict) else {}

    def set_backend_templates(
        self,
        class_id: str,
        template_pyramids: Sequence[Sequence[TemplateLevel]],
        *,
        backend: str = "original",
    ) -> None:
        backend = _normalize_backend_name(backend)
        self._ensure_class_containers(class_id)
        self.backend_templates[backend][class_id] = [clone_template_levels(tp) for tp in template_pyramids]
        self.invalidate_native_cache(class_id)

    def set_class_source(self, class_id: str, source_info: Dict[str, Any]) -> None:
        self._ensure_class_containers(class_id)
        self.class_sources[class_id] = copy.deepcopy(source_info)

    def get_class_source(self, class_id: str) -> Dict[str, Any]:
        info = self.class_sources.get(class_id, {})
        return copy.deepcopy(info)

    def set_original_editor_levels(self, class_id: str, levels: Sequence[TemplateLevel]) -> None:
        self._ensure_class_containers(class_id)
        self.original_editor_levels[class_id] = clone_template_levels(levels)

    def get_original_editor_levels(self, class_id: str) -> List[TemplateLevel]:
        levels = self.original_editor_levels.get(class_id, [])
        return clone_template_levels(levels)

    def _import_python_templates_to_native(self, backend: str = "original") -> Any:
        backend = _normalize_backend_name(backend)
        native_detector = self._create_native_detector(backend=backend)
        native_detector.clear_classes()
        for class_id, template_pyramids in self.backend_templates[backend].items():
            native_detector.replace_class_templates(
                class_id,
                [self._template_pyramid_to_native(tp) for tp in template_pyramids],
            )
        self._native_detectors[backend] = native_detector
        self._native_dirty = False
        return native_detector

    def _ensure_native_detector_synced(self, backend: str = "original", required: bool = True) -> Any:
        backend = _normalize_backend_name(backend)
        native_detector = self._native_detectors.get(backend)
        if native_detector is not None and not self._native_dirty:
            return native_detector
        if native_detector is None and not required:
            native_matcher = _load_native_matcher(backend=backend, required=False)
            if native_matcher is None:
                return None
        if native_detector is None or self._native_dirty:
            return self._import_python_templates_to_native(backend=backend)
        return native_detector

    def _append_template_from_native(self, native_detector: Any, class_id: str, template_id: int, *, backend: str = "original") -> None:
        backend = _normalize_backend_name(backend)
        levels_raw = native_detector.export_template_pyramid(class_id, int(template_id))
        if not levels_raw:
            raise RuntimeError(f"Native detector did not return template pyramid: class={class_id}, id={template_id}")
        self._ensure_class_containers(class_id)
        template_pyramid = self._template_pyramid_from_native(levels_raw)
        store = self.backend_templates[backend][class_id]
        if template_id == len(store):
            store.append(template_pyramid)
            return
        while len(store) <= template_id:
            store.append([])
        store[template_id] = template_pyramid

    def add_template(
        self,
        source: np.ndarray,
        class_id: str,
        object_mask: Optional[np.ndarray] = None,
        num_features: Optional[int] = None,
        metadata: Optional[Dict[str, float]] = None,
        backend: str = "original",
    ) -> int:
        backend = _normalize_backend_name(backend)
        nfeat = int(num_features) if num_features is not None and num_features > 0 else self.num_features
        self._ensure_class_containers(class_id)
        native_detector = self._ensure_native_detector_synced(backend=backend, required=(backend != "original"))
        if native_detector is not None:
            mask = ensure_mask(object_mask, source.shape[:2]) if object_mask is not None else None
            template_id = int(native_detector.add_template(source, class_id, mask, nfeat))
            if template_id < 0:
                return -1
            self._append_template_from_native(native_detector, class_id, template_id, backend=backend)
            self._set_template_meta(class_id, template_id, metadata)
            self._native_dirty = False
            return template_id

        if backend != "original":
            raise RuntimeError(f"{backend} backend is unavailable.\n{_native_build_instructions()}")
        if "original" in _NATIVE_MODULE_ERRORS:
            _warn_native_fallback("original", _NATIVE_MODULE_ERRORS["original"])
        return self._add_template_python(source, class_id, object_mask, nfeat, metadata)

    def _add_template_python(
        self,
        source: np.ndarray,
        class_id: str,
        object_mask: Optional[np.ndarray],
        num_features: int,
        metadata: Optional[Dict[str, float]],
    ) -> int:
        qp = ColorGradientPyramid(
            source,
            object_mask,
            weak_threshold=self.weak_threshold,
            num_features=num_features,
            strong_threshold=self.strong_threshold,
        )

        tp: List[TemplateLevel] = []
        for l in range(self.pyramid_levels):
            if l > 0:
                qp.pyr_down()
            templ = qp.extract_template()
            if templ is None:
                return -1
            tp.append(templ)

        crop_templates(tp)
        template_id = len(self.class_templates[class_id])
        self.class_templates[class_id].append(tp)
        self._set_template_meta(class_id, template_id, metadata)
        self._native_dirty = True
        self._native_detectors.clear()
        return template_id

    def add_template_rotate(
        self,
        class_id: str,
        zero_id: int,
        theta_deg: float,
        center: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> int:
        if class_id not in self.class_templates:
            return -1
        if zero_id < 0 or zero_id >= len(self.class_templates[class_id]):
            return -1

        base_tp = self.class_templates[class_id][zero_id]
        if center is None:
            tl = base_tp[0].tl_x
            tt = base_tp[0].tl_y
            center = (tl + base_tp[0].width * 0.5, tt + base_tp[0].height * 0.5)

        native_detector = self._ensure_native_detector_synced(backend="original", required=False)
        if native_detector is not None:
            template_id = int(
                native_detector.add_template_rotate(
                    class_id,
                    int(zero_id),
                    float(theta_deg),
                    float(center[0]),
                    float(center[1]),
                )
            )
            if template_id < 0:
                return -1
            self._append_template_from_native(native_detector, class_id, template_id, backend="original")
            self._set_template_meta(class_id, template_id, metadata)
            self._native_dirty = False
            return template_id

        if "original" in _NATIVE_MODULE_ERRORS:
            _warn_native_fallback("original", _NATIVE_MODULE_ERRORS["original"])
        return self._add_template_rotate_python(class_id, zero_id, theta_deg, center, metadata)

    def _add_template_rotate_python(
        self,
        class_id: str,
        zero_id: int,
        theta_deg: float,
        center: Tuple[float, float],
        metadata: Optional[Dict[str, float]],
    ) -> int:
        base_tp = self.class_templates[class_id][zero_id]
        tp: List[TemplateLevel] = []
        c_x, c_y = center
        for l in range(self.pyramid_levels):
            if l > 0:
                c_x *= 0.5
                c_y *= 0.5
            rotated_features: List[Feature] = []
            level_base = base_tp[l]
            ang_rad = -theta_deg / 180.0 * math.pi
            for f in level_base.features:
                px = f.x + level_base.tl_x
                py = f.y + level_base.tl_y
                rx, ry = rotate_point((px, py), (c_x, c_y), ang_rad)
                theta_new = (f.theta - theta_deg) % 360.0
                label = int(theta_new * 16.0 / 360.0 + 0.5) & 7
                rotated_features.append(Feature(x=int(round(rx)), y=int(round(ry)), label=label, theta=theta_new))
            tp.append(
                TemplateLevel(
                    width=-1,
                    height=-1,
                    tl_x=0,
                    tl_y=0,
                    pyramid_level=l,
                    features=rotated_features,
                )
            )

        crop_templates(tp)
        template_id = len(self.class_templates[class_id])
        self.class_templates[class_id].append(tp)
        self._set_template_meta(class_id, template_id, metadata)
        self._native_dirty = True
        self._native_detectors.clear()
        return template_id

    def get_templates(self, class_id: str, template_id: int, backend: str = "original") -> List[TemplateLevel]:
        backend = _normalize_backend_name(backend)
        return self.backend_templates[backend][class_id][template_id]

    def get_template_meta(self, class_id: str, template_id: int) -> Dict[str, float]:
        return self.class_meta[class_id][template_id]

    def class_ids(self) -> List[str]:
        out: List[str] = []
        seen = set()
        for mapping in [self.class_sources, self.class_meta, *self.backend_templates.values()]:
            for class_id in mapping.keys():
                if class_id in seen:
                    continue
                seen.add(class_id)
                out.append(class_id)
        return out

    def match(
        self,
        source: np.ndarray,
        threshold: float,
        class_ids: Optional[Sequence[str]] = None,
        mask: Optional[np.ndarray] = None,
        backend: str = "original",
    ) -> List[Match]:
        backend = _normalize_backend_name(backend)
        backend_store = self.backend_templates[backend]
        if not backend_store:
            return []
        if class_ids is None or len(class_ids) == 0:
            class_ids = [class_id for class_id in self.class_ids() if backend_store.get(class_id)]
        if not class_ids:
            return []
        class_ids = [class_id for class_id in class_ids if backend_store.get(class_id)]
        if not class_ids:
            return []

        native_detector = self._ensure_native_detector_synced(backend=backend, required=(backend != "original"))
        if native_detector is not None:
            align = max((int(t) * (1 << idx)) for idx, t in enumerate(self.T_at_level)) if self.T_at_level else 1
            h, w = source.shape[:2]
            h2 = (h // align) * align
            w2 = (w // align) * align
            if h2 <= 0 or w2 <= 0:
                return []
            scene = source if (h2 == h and w2 == w) else source[:h2, :w2].copy()
            scene_mask = None
            if mask is not None:
                full_mask = ensure_mask(mask, source.shape[:2])
                scene_mask = full_mask if (h2 == h and w2 == w) else full_mask[:h2, :w2].copy()

            matches = [
                Match(
                    x=int(x),
                    y=int(y),
                    similarity=float(similarity),
                    class_id=str(native_class_id),
                    template_id=int(template_id),
                    backend=backend,
                )
                for x, y, similarity, native_class_id, template_id in native_detector.match(
                    scene,
                    float(threshold),
                    list(class_ids),
                    scene_mask,
                )
            ]
        else:
            if backend != "original":
                raise RuntimeError(f"{backend} backend is unavailable.\n{_native_build_instructions()}")
            if "original" in _NATIVE_MODULE_ERRORS:
                _warn_native_fallback("original", _NATIVE_MODULE_ERRORS["original"])
            matches = self._match_python(source, float(threshold), class_ids, mask)

        matches.sort(key=lambda m: m.similarity, reverse=True)

        deduped: List[Match] = []
        seen = set()
        for m in matches:
            key = (m.class_id, m.template_id, m.x, m.y, int(round(m.similarity * 100.0)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(m)
        return deduped

    def _match_python(
        self,
        source: np.ndarray,
        threshold: float,
        class_ids: Sequence[str],
        mask: Optional[np.ndarray],
    ) -> List[Match]:
        quantizer = ColorGradientPyramid(
            source,
            mask,
            weak_threshold=self.weak_threshold,
            num_features=self.num_features,
            strong_threshold=self.strong_threshold,
        )

        scene_levels: List[SceneLevelData] = []
        for l in range(self.pyramid_levels):
            if l > 0:
                quantizer.pyr_down()
            T = self.T_at_level[l]
            quant = quantizer.quantize()
            h, w = quant.shape[:2]
            h = (h // T) * T
            w = (w // T) * T
            if h <= 0 or w <= 0:
                continue
            quant = quant[:h, :w]
            spread = spread_or(quant, T)
            response_maps = compute_response_maps(spread)
            linear_memories = [linearize_response_map(rm, T) for rm in response_maps]
            scene_levels.append(
                SceneLevelData(
                    width=w,
                    height=h,
                    T=T,
                    response_maps=response_maps,
                    linear_memories=linear_memories,
                    memory_width=w // T,
                    memory_height=h // T,
                )
            )

        if len(scene_levels) != self.pyramid_levels:
            return []

        matches: List[Match] = []
        for class_id in class_ids:
            if class_id not in self.class_templates:
                continue
            self._match_class(scene_levels, threshold, class_id, self.class_templates[class_id], matches)
        return matches

    def _match_class(
        self,
        scene_levels: Sequence[SceneLevelData],
        threshold: float,
        class_id: str,
        template_pyramids: Sequence[List[TemplateLevel]],
        out_matches: List[Match],
    ) -> None:
        final_threshold = float(threshold)
        coarse_threshold = float(final_threshold)

        for template_id, tp in enumerate(template_pyramids):
            lowest_idx = self.pyramid_levels - 1
            lowest_scene = scene_levels[lowest_idx]
            lowest_templ = tp[lowest_idx]
            lowest_num_features = len(lowest_templ.features)

            sim_map = similarity_full(lowest_scene, lowest_templ)
            if sim_map.size == 0:
                continue

            candidates: List[Match] = []
            low_T = lowest_scene.T
            off = offset_from_T(low_T)
            coarse_raw_threshold = raw_similarity_threshold(coarse_threshold, lowest_num_features)
            ys, xs = np.where(sim_map > coarse_raw_threshold)
            for r, c in zip(ys.tolist(), xs.tolist()):
                x = int(c * low_T + off)
                y = int(r * low_T + off)
                score = similarity_from_raw(sim_map[r, c], lowest_num_features)
                candidates.append(Match(x=x, y=y, similarity=score, class_id=class_id, template_id=template_id))

            if not candidates:
                continue

            for l in range(self.pyramid_levels - 2, -1, -1):
                scene = scene_levels[l]
                templ = tp[l]
                T = scene.T
                border = 8 * T
                off = offset_from_T(T)
                max_x = scene.width - templ.width - border
                max_y = scene.height - templ.height - border
                if max_x <= border or max_y <= border:
                    max_tl_x = max(0, int(scene.width - templ.width - 1))
                    max_tl_y = max(0, int(scene.height - templ.height - 1))
                    propagated: List[Match] = []
                    for m in candidates:
                        x = int(np.clip(m.x * 2 + 1, 0, max_tl_x))
                        y = int(np.clip(m.y * 2 + 1, 0, max_tl_y))
                        propagated.append(
                            Match(
                                x=x,
                                y=y,
                                similarity=float(m.similarity),
                                class_id=class_id,
                                template_id=template_id,
                            )
                        )
                    candidates = propagated
                    if not candidates:
                        break
                    continue

                refined: List[Match] = []
                num_features = len(templ.features)
                coarse_raw_threshold = raw_similarity_threshold(coarse_threshold, num_features)
                for m in candidates:
                    x = m.x * 2 + 1
                    y = m.y * 2 + 1
                    x = max(x, border)
                    y = max(y, border)
                    x = min(x, max_x)
                    y = min(y, max_y)

                    local = similarity_local(scene, templ, (x, y))
                    if local.size == 0:
                        continue
                    best_idx = int(np.argmax(local))
                    best_r, best_c = divmod(best_idx, local.shape[1])
                    best_raw_score = float(local[best_r, best_c])
                    if best_raw_score < coarse_raw_threshold:
                        continue

                    best_score = similarity_from_raw(best_raw_score, num_features)
                    new_x = (x // T - 8 + best_c) * T + off
                    new_y = (y // T - 8 + best_r) * T + off
                    refined.append(
                        Match(
                            x=int(new_x),
                            y=int(new_y),
                            similarity=best_score,
                            class_id=class_id,
                            template_id=template_id,
                        )
                    )
                candidates = refined
                if not candidates:
                    break

            for m in candidates:
                if float(m.similarity) >= final_threshold:
                    out_matches.append(m)


def _apply_affine_transform_point(transform: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    tx = float(transform[0, 0] * x + transform[0, 1] * y + transform[0, 2])
    ty = float(transform[1, 0] * x + transform[1, 1] * y + transform[1, 2])
    tw = 1.0
    if transform.shape[0] >= 3 and transform.shape[1] >= 3:
        tw = float(transform[2, 0] * x + transform[2, 1] * y + transform[2, 2])
    if abs(tw) > 1e-9 and abs(tw - 1.0) > 1e-9:
        tx /= tw
        ty /= tw
    return tx, ty


def _display_canvas_size_for_template(
    detector: Line2DupLikeDetector,
    class_id: str,
    template_id: int,
    backend: str,
) -> Tuple[int, int]:
    backend = _normalize_backend_name(backend)
    templ0 = detector.get_templates(class_id, template_id, backend=backend)[0]
    meta = detector.get_template_meta(class_id, template_id)
    source_info = detector.class_sources.get(class_id, {})
    original_mode = str(source_info.get("original_mode", "auto")) if isinstance(source_info, dict) else "auto"

    if backend == "original" and original_mode == "manual_points":
        return max(1, int(templ0.width) + 1), max(1, int(templ0.height) + 1)

    canvas_w = int(meta.get("canvas_w", meta.get("roi_w", int(templ0.width) + 1)))
    canvas_h = int(meta.get("canvas_h", meta.get("roi_h", int(templ0.height) + 1)))
    canvas_w = max(canvas_w, int(templ0.width) + 1)
    canvas_h = max(canvas_h, int(templ0.height) + 1)
    return canvas_w, canvas_h


def _template_display_quad(
    detector: Line2DupLikeDetector,
    class_id: str,
    template_id: int,
    backend: str,
    match_x: int,
    match_y: int,
) -> List[Tuple[float, float]]:
    backend = _normalize_backend_name(backend)
    templ0 = detector.get_templates(class_id, template_id, backend=backend)[0]
    canvas_w, canvas_h = _display_canvas_size_for_template(detector, class_id, template_id, backend)
    x0 = float(match_x - int(templ0.tl_x))
    y0 = float(match_y - int(templ0.tl_y))
    return [
        (x0, y0),
        (x0 + float(canvas_w - 1), y0),
        (x0 + float(canvas_w - 1), y0 + float(canvas_h - 1)),
        (x0, y0 + float(canvas_h - 1)),
    ]


def refine_matches_sim3(
    detector: Line2DupLikeDetector,
    source: np.ndarray,
    matches: Sequence[Match],
) -> float:
    if not matches:
        return 0.0
    native_detector = detector._ensure_native_detector_synced(backend="sim3", required=True)
    if not hasattr(native_detector, "refine_match"):
        raise RuntimeError("line2dup_sim3_native does not expose refine_match(). Rebuild native extensions.")

    started = time.perf_counter()
    for match in matches:
        info = native_detector.refine_match(source, match.class_id, int(match.template_id), int(match.x), int(match.y))
        transform = np.array(info.get("transform", np.eye(3, dtype=np.float32)), dtype=np.float32)
        meta = detector.get_template_meta(match.class_id, match.template_id)
        raw_quad = _template_display_quad(
            detector,
            match.class_id,
            int(match.template_id),
            "sim3",
            int(match.x),
            int(match.y),
        )
        base_scale = float(meta.get("scale", 1.0))
        base_angle = float(meta.get("angle", 0.0))
        match.backend = "sim3"
        match.refined_transform = transform
        match.refined_scale = base_scale * float(info.get("delta_scale", 1.0))
        match.refined_angle_deg = base_angle + float(info.get("delta_angle_deg", 0.0))
        match.refined_fitness = float(info.get("fitness", 0.0))
        match.refined_rmse = float(info.get("rmse", 0.0))
        match.refined_quad = [_apply_affine_transform_point(transform, x, y) for x, y in raw_quad]
    return time.perf_counter() - started


class ShapeInfoProducer:
    def __init__(self, src: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        self.src = src
        self.mask = ensure_mask(mask, src.shape[:2])
        self.angle_range: List[float] = [0.0]
        self.scale_range: List[float] = [1.0]
        self.angle_step: float = 15.0
        self.scale_step: float = 0.5

    @staticmethod
    def transform(src: np.ndarray, angle: float, scale: float) -> np.ndarray:
        center = (src.shape[1] / 2.0, src.shape[0] / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(src, mat, (src.shape[1], src.shape[0]))

    def produce_infos(self) -> List[ShapeInfo]:
        eps = 1e-5
        angles = self._expand_range(self.angle_range, self.angle_step, eps)
        scales = self._expand_range(self.scale_range, self.scale_step, eps)
        infos: List[ShapeInfo] = []
        for s in scales:
            for a in angles:
                infos.append(ShapeInfo(angle=float(a), scale=float(s)))
        return infos

    def src_of(self, info: ShapeInfo) -> np.ndarray:
        return self.transform(self.src, info.angle, info.scale)

    def mask_of(self, info: ShapeInfo) -> np.ndarray:
        m = self.transform(self.mask, info.angle, info.scale)
        return (m > 0).astype(np.uint8) * 255

    @staticmethod
    def _expand_range(vals: Sequence[float], step: float, eps: float) -> List[float]:
        if len(vals) == 0:
            return [0.0]
        if len(vals) == 1:
            return [float(vals[0])]
        lo = float(vals[0])
        hi = float(vals[1])
        if hi < lo:
            lo, hi = hi, lo
        out = []
        cur = lo
        while cur <= hi + eps:
            out.append(float(cur))
            cur += max(step, eps * 10.0)
        return out


def _feature_to_dict(f: Feature) -> Dict[str, Any]:
    return {
        "x": int(f.x),
        "y": int(f.y),
        "label": int(f.label),
        "theta": float(f.theta),
    }


def _feature_from_dict(data: Dict[str, Any]) -> Feature:
    return Feature(
        x=int(data.get("x", 0)),
        y=int(data.get("y", 0)),
        label=int(data.get("label", 0)),
        theta=float(data.get("theta", 0.0)),
    )


def _template_level_to_dict(t: TemplateLevel) -> Dict[str, Any]:
    return {
        "width": int(t.width),
        "height": int(t.height),
        "tl_x": int(t.tl_x),
        "tl_y": int(t.tl_y),
        "pyramid_level": int(t.pyramid_level),
        "features": [_feature_to_dict(f) for f in t.features],
    }


def _template_level_from_dict(data: Dict[str, Any]) -> TemplateLevel:
    return TemplateLevel(
        width=int(data.get("width", -1)),
        height=int(data.get("height", -1)),
        tl_x=int(data.get("tl_x", 0)),
        tl_y=int(data.get("tl_y", 0)),
        pyramid_level=int(data.get("pyramid_level", 0)),
        features=[_feature_from_dict(x) for x in data.get("features", [])],
    )


def clone_template_levels(levels: Sequence[TemplateLevel]) -> List[TemplateLevel]:
    cloned: List[TemplateLevel] = []
    for level in levels:
        cloned.append(
            TemplateLevel(
                width=int(level.width),
                height=int(level.height),
                tl_x=int(level.tl_x),
                tl_y=int(level.tl_y),
                pyramid_level=int(level.pyramid_level),
                features=[
                    Feature(
                        x=int(feature.x),
                        y=int(feature.y),
                        label=int(feature.label),
                        theta=float(feature.theta),
                    )
                    for feature in level.features
                ],
            )
        )
    return cloned


def encode_png_base64(image: Optional[np.ndarray]) -> str:
    if image is None:
        return ""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_png_base64(data: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    if not data:
        return None
    raw = base64.b64decode(data.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, flags)
    if image is None:
        raise ValueError("Failed to decode PNG data from model.")
    return image


def detector_to_dict(detector: Line2DupLikeDetector) -> Dict[str, Any]:
    classes: Dict[str, Any] = {}
    for class_id in detector.class_ids():
        metas = detector.class_meta.get(class_id, [])
        source_info = detector.class_sources.get(class_id, {})
        pose_infos = copy.deepcopy(source_info.get("pose_infos", {})) if isinstance(source_info, dict) else {}
        if not isinstance(pose_infos, dict):
            pose_infos = {}
        if "items" not in pose_infos or not isinstance(pose_infos.get("items"), list):
            pose_infos["items"] = [
                {
                    "angle": float(meta.get("angle", 0.0)),
                    "scale": float(meta.get("scale", 1.0)),
                }
                for meta in metas
            ]
        if "ui" not in pose_infos or not isinstance(pose_infos.get("ui"), dict):
            pose_infos["ui"] = {}

        packed_backends: Dict[str, List[Dict[str, Any]]] = {}
        for backend in NATIVE_BACKEND_TO_MODULE:
            template_pyramids = detector.backend_templates.get(backend, {}).get(class_id, [])
            packed_backends[backend] = [
                {
                    "template_id": int(tid),
                    "levels": [_template_level_to_dict(lv) for lv in tp],
                }
                for tid, tp in enumerate(template_pyramids)
            ]

        classes[class_id] = {
            "source": copy.deepcopy(source_info.get("source", {})) if isinstance(source_info, dict) else {},
            "pose_infos": pose_infos,
            "original_mode": str(source_info.get("original_mode", "auto")) if isinstance(source_info, dict) else "auto",
            "meta": [dict(meta) if isinstance(meta, dict) else {} for meta in metas],
            "backends": packed_backends,
            "original_editor_levels": [
                _template_level_to_dict(level)
                for level in detector.original_editor_levels.get(class_id, [])
            ],
        }

    return {
        "format": "line2dup_like_model_v2",
        "params": {
            "num_features": int(detector.num_features),
            "T_levels": [int(t) for t in detector.T_at_level],
            "weak_threshold": float(detector.weak_threshold),
            "strong_threshold": float(detector.strong_threshold),
        },
        "classes": classes,
    }


def _source_level_shapes(width: int, height: int, total_levels: int) -> List[Tuple[int, int]]:
    w = max(1, int(width))
    h = max(1, int(height))
    shapes: List[Tuple[int, int]] = []
    for _ in range(max(1, int(total_levels))):
        shapes.append((w, h))
        if w > 1:
            w = max(1, (w + 1) // 2)
        if h > 1:
            h = max(1, (h + 1) // 2)
    return shapes


def _expand_single_level_template(
    level0: TemplateLevel,
    *,
    width: int,
    height: int,
    total_levels: int,
) -> List[TemplateLevel]:
    shapes = _source_level_shapes(width, height, total_levels)
    if not shapes:
        return []

    w0, h0 = shapes[0]
    max_x0 = max(0, int(w0) - 1)
    max_y0 = max(0, int(h0) - 1)
    l0_feats: List[Feature] = []
    seen_l0 = set()
    for feature in level0.features:
        x0 = int(np.clip(int(feature.x) + int(level0.tl_x), 0, max_x0))
        y0 = int(np.clip(int(feature.y) + int(level0.tl_y), 0, max_y0))
        key = (x0, y0, int(feature.label) & 7)
        if key in seen_l0:
            continue
        seen_l0.add(key)
        l0_feats.append(Feature(x=x0, y=y0, label=int(feature.label) & 7, theta=float(feature.theta)))

    out: List[TemplateLevel] = [
        TemplateLevel(
            width=max_x0,
            height=max_y0,
            tl_x=0,
            tl_y=0,
            pyramid_level=0,
            features=l0_feats,
        )
    ]
    for level_index in range(1, len(shapes)):
        w, h = shapes[level_index]
        max_x = max(0, int(w) - 1)
        max_y = max(0, int(h) - 1)
        div = float(1 << level_index)
        feats: List[Feature] = []
        seen = set()
        for feature in l0_feats:
            x = int(np.clip(int(round(float(feature.x) / div)), 0, max_x))
            y = int(np.clip(int(round(float(feature.y) / div)), 0, max_y))
            key = (x, y, int(feature.label) & 7)
            if key in seen:
                continue
            seen.add(key)
            feats.append(Feature(x=x, y=y, label=int(feature.label) & 7, theta=float(feature.theta)))
        out.append(
            TemplateLevel(
                width=max_x,
                height=max_y,
                tl_x=0,
                tl_y=0,
                pyramid_level=level_index,
                features=feats,
            )
        )
    return out


def detector_from_dict(data: Dict[str, Any]) -> Line2DupLikeDetector:
    model_format = str(data.get("format", "")).strip()
    if model_format == "line2dup_like_model_v1":
        raise ValueError("Model format line2dup_like_model_v1 is no longer supported. Recreate the model as v2.")
    if model_format != "line2dup_like_model_v2":
        raise ValueError(f"Unsupported model format: {model_format or '(missing format)'}")

    params = data.get("params", {})
    detector = Line2DupLikeDetector(
        num_features=int(params.get("num_features", 128)),
        T_levels=[int(x) for x in params.get("T_levels", [4, 8])],
        weak_threshold=float(params.get("weak_threshold", 30.0)),
        strong_threshold=float(params.get("strong_threshold", 60.0)),
    )

    classes = data.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("Invalid model format: classes must be a dict.")

    detector._reset_template_storage()
    detector.class_meta = {}
    detector.class_sources = {}
    detector.original_editor_levels = {}
    detector.invalidate_native_cache()

    for class_id, class_entry in classes.items():
        if not isinstance(class_entry, dict):
            raise ValueError(f"Invalid class entry for '{class_id}'.")
        detector._ensure_class_containers(class_id)

        source_block = class_entry.get("source", {})
        pose_block = class_entry.get("pose_infos", {})
        if not isinstance(source_block, dict):
            source_block = {}
        if not isinstance(pose_block, dict):
            pose_block = {}
        detector.class_sources[class_id] = {
            "source": copy.deepcopy(source_block),
            "pose_infos": {
                "items": copy.deepcopy(pose_block.get("items", [])) if isinstance(pose_block.get("items", []), list) else [],
                "ui": copy.deepcopy(pose_block.get("ui", {})) if isinstance(pose_block.get("ui", {}), dict) else {},
            },
            "original_mode": str(class_entry.get("original_mode", "auto")),
        }

        backends_raw = class_entry.get("backends", {})
        if not isinstance(backends_raw, dict):
            raise ValueError(f"Invalid backends block for class '{class_id}'.")

        expected_templates = 0
        for backend in NATIVE_BACKEND_TO_MODULE:
            packed_templates = backends_raw.get(backend, [])
            if not isinstance(packed_templates, list):
                raise ValueError(f"Backend '{backend}' templates for class '{class_id}' must be a list.")
            parsed_templates: List[List[TemplateLevel]] = []
            for item in packed_templates:
                levels_raw = item.get("levels", []) if isinstance(item, dict) else []
                levels = [_template_level_from_dict(x) for x in levels_raw]
                if (
                    backend == "original"
                    and len(levels) == 1
                    and detector.pyramid_levels > 1
                ):
                    roi_w = int(source_block.get("roi_w", 0)) if isinstance(source_block, dict) else 0
                    roi_h = int(source_block.get("roi_h", 0)) if isinstance(source_block, dict) else 0
                    if roi_w > 0 and roi_h > 0:
                        levels = _expand_single_level_template(
                            levels[0],
                            width=roi_w,
                            height=roi_h,
                            total_levels=detector.pyramid_levels,
                        )
                if len(levels) != detector.pyramid_levels:
                    raise ValueError(
                        f"Template levels mismatch for class '{class_id}' backend '{backend}': "
                        f"expected {detector.pyramid_levels}, got {len(levels)}"
                    )
                parsed_templates.append(levels)
            detector.backend_templates[backend][class_id] = parsed_templates
            expected_templates = max(expected_templates, len(parsed_templates))

        meta_items = class_entry.get("meta", [])
        if not isinstance(meta_items, list):
            raise ValueError(f"Meta block for class '{class_id}' must be a list.")
        detector.class_meta[class_id] = [dict(item) if isinstance(item, dict) else {} for item in meta_items]
        if expected_templates > 0 and len(detector.class_meta[class_id]) != expected_templates:
            raise ValueError(
                f"Meta/template count mismatch for class '{class_id}': "
                f"meta={len(detector.class_meta[class_id])}, templates={expected_templates}"
            )

        editor_levels_raw = class_entry.get("original_editor_levels", [])
        if editor_levels_raw:
            editor_levels = [_template_level_from_dict(x) for x in editor_levels_raw]
            if len(editor_levels) == 1 and detector.pyramid_levels > 1:
                roi_w = int(source_block.get("roi_w", 0)) if isinstance(source_block, dict) else 0
                roi_h = int(source_block.get("roi_h", 0)) if isinstance(source_block, dict) else 0
                if roi_w > 0 and roi_h > 0:
                    editor_levels = _expand_single_level_template(
                        editor_levels[0],
                        width=roi_w,
                        height=roi_h,
                        total_levels=detector.pyramid_levels,
                    )
            if len(editor_levels) != detector.pyramid_levels:
                raise ValueError(
                    f"original_editor_levels mismatch for class '{class_id}': "
                    f"expected {detector.pyramid_levels}, got {len(editor_levels)}"
                )
            detector.original_editor_levels[class_id] = editor_levels
        elif detector.backend_templates["original"].get(class_id):
            detector.original_editor_levels[class_id] = clone_template_levels(
                detector.backend_templates["original"][class_id][0]
            )

    return detector


def save_detector_model(detector: Line2DupLikeDetector, model_path: str) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = detector_to_dict(detector)
    text = json.dumps(data, indent=2, ensure_ascii=True)
    path.write_text(text, encoding="utf-8")


def load_detector_model(model_path: str) -> Line2DupLikeDetector:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid model content.")
    return detector_from_dict(data)


def match_quad(detector: Line2DupLikeDetector, match: Match) -> List[Tuple[float, float]]:
    corners = _template_display_quad(
        detector,
        match.class_id,
        int(match.template_id),
        match.backend,
        int(match.x),
        int(match.y),
    )
    if match.refined_quad:
        return [(float(x), float(y)) for x, y in match.refined_quad]
    if match.refined_transform is not None:
        return [_apply_affine_transform_point(match.refined_transform, x, y) for x, y in corners]
    return corners


def match_bbox(detector: Line2DupLikeDetector, match: Match) -> List[int]:
    quad = match_quad(detector, match)
    xs = [float(x) for x, _y in quad]
    ys = [float(y) for _x, y in quad]
    x1 = int(math.floor(min(xs)))
    y1 = int(math.floor(min(ys)))
    x2 = int(math.ceil(max(xs)))
    y2 = int(math.ceil(max(ys)))
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def nms_matches(
    detector: Line2DupLikeDetector,
    matches: Sequence[Match],
    iou_threshold: float,
    score_threshold: float = 0.0,
) -> List[Match]:
    if not matches or iou_threshold <= 0:
        return list(matches)

    boxes: List[List[int]] = []
    scores: List[float] = []
    for m in matches:
        boxes.append(match_bbox(detector, m))
        scores.append(float(m.similarity))

    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=score_threshold, nms_threshold=iou_threshold)
    if keep is None or len(keep) == 0:
        return []

    indices: List[int] = []
    for idx in keep:
        if isinstance(idx, (list, tuple, np.ndarray)):
            indices.append(int(idx[0]))
        else:
            indices.append(int(idx))
    indices = sorted(set(indices))
    return [matches[i] for i in indices]


def draw_matches(
    detector: Line2DupLikeDetector,
    image_bgr: np.ndarray,
    matches: Sequence[Match],
    topk: int,
) -> np.ndarray:
    out = image_bgr.copy()
    palette = [(0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0)]
    draw_n = min(topk, len(matches))
    for i in range(draw_n):
        m = matches[i]
        t0 = detector.get_templates(m.class_id, m.template_id, backend=m.backend)[0]
        meta = detector.get_template_meta(m.class_id, m.template_id)
        color = palette[i % len(palette)]
        refined_transform = m.refined_transform

        # Draw matched template feature points in scene coordinates.
        pts: List[Tuple[float, float]] = []
        for f in t0.features:
            px = float(f.x + m.x)
            py = float(f.y + m.y)
            if refined_transform is not None:
                px, py = _apply_affine_transform_point(refined_transform, px, py)
            pts.append((px, py))
            pxi = int(round(px))
            pyi = int(round(py))
            theta = float(f.theta)
            if not np.isfinite(theta):
                theta = label_to_theta_deg(int(f.label))
            rad = np.deg2rad(theta)
            raw_p2x = float(f.x + m.x + 7.0 * float(np.cos(rad)))
            raw_p2y = float(f.y + m.y + 7.0 * float(np.sin(rad)))
            if refined_transform is not None:
                raw_p2x, raw_p2y = _apply_affine_transform_point(refined_transform, raw_p2x, raw_p2y)
            p2x = int(round(raw_p2x))
            p2y = int(round(raw_p2y))
            # Draw dark outline then color line for visibility.
            cv2.arrowedLine(out, (pxi, pyi), (p2x, p2y), (0, 0, 0), 3, cv2.LINE_AA, 0, 0.35)
            cv2.arrowedLine(out, (pxi, pyi), (p2x, p2y), color, 1, cv2.LINE_AA, 0, 0.35)

        # Draw stable template bbox (width/height from matched template).
        # Using minAreaRect on sparse/one-sided feature clouds may degenerate to a near-line.
        corners = match_quad(detector, m)
        if m.refined_quad or refined_transform is not None:
            pts_i = np.array([[int(round(x)), int(round(y))] for x, y in corners], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            tx = int(round(corners[0][0]))
            ty = int(round(corners[0][1]))
        else:
            x1 = int(m.x)
            y1 = int(m.y)
            x2 = int(m.x + t0.width)
            y2 = int(m.y + t0.height)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            tx = x1
            ty = y1

        ang = float(m.refined_angle_deg) if m.refined_angle_deg is not None else float(meta.get("angle", 0.0))
        label = f"#{i+1} {m.similarity:.1f} a={ang:.1f}"
        cv2.putText(
            out,
            label,
            (int(tx), max(18, int(ty) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def parse_levels(arg: str) -> List[int]:
    vals = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("T levels cannot be empty.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="line2Dup/shape_based_matching style matcher (Python).")
    parser.add_argument("--template", required=True, help="Template image path.")
    parser.add_argument("--scene", required=True, help="Scene image path.")
    parser.add_argument("--template-mask", default="", help="Template mask path (optional).")
    parser.add_argument("--scene-mask", default="", help="Scene mask path (optional).")
    parser.add_argument("--out", default="line2dup_like_result.png", help="Output visualization path.")
    parser.add_argument("--threshold", type=float, default=90.0, help="Similarity threshold in [0, 100].")
    parser.add_argument("--num-features", type=int, default=128, help="Requested feature count per template.")
    parser.add_argument("--weak-thresh", type=float, default=30.0, help="Weak threshold for quantization.")
    parser.add_argument("--strong-thresh", type=float, default=60.0, help="Strong threshold for template candidate features.")
    parser.add_argument("--levels", default="4,8", help="Pyramid T levels, e.g. 4,8")
    parser.add_argument("--angle-start", type=float, default=0.0, help="Training angle start.")
    parser.add_argument("--angle-end", type=float, default=0.0, help="Training angle end.")
    parser.add_argument("--angle-step", type=float, default=10.0, help="Training angle step.")
    parser.add_argument("--scale-start", type=float, default=1.0, help="Training scale start.")
    parser.add_argument("--scale-end", type=float, default=1.0, help="Training scale end.")
    parser.add_argument("--scale-step", type=float, default=0.1, help="Training scale step.")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="NMS IoU threshold on final matches.")
    parser.add_argument("--topk", type=int, default=20, help="How many matches to print and draw.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    template = cv2.imread(args.template, cv2.IMREAD_COLOR)
    scene = cv2.imread(args.scene, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Failed to read template: {args.template}")
    if scene is None:
        raise FileNotFoundError(f"Failed to read scene: {args.scene}")

    templ_mask = None
    if args.template_mask:
        templ_mask = cv2.imread(args.template_mask, cv2.IMREAD_GRAYSCALE)
        if templ_mask is None:
            raise FileNotFoundError(f"Failed to read template mask: {args.template_mask}")

    scene_mask = None
    if args.scene_mask:
        scene_mask = cv2.imread(args.scene_mask, cv2.IMREAD_GRAYSCALE)
        if scene_mask is None:
            raise FileNotFoundError(f"Failed to read scene mask: {args.scene_mask}")

    levels = parse_levels(args.levels)
    detector = Line2DupLikeDetector(
        num_features=args.num_features,
        T_levels=levels,
        weak_threshold=args.weak_thresh,
        strong_threshold=args.strong_thresh,
    )

    producer = ShapeInfoProducer(template, templ_mask)
    producer.angle_range = [args.angle_start, args.angle_end] if args.angle_start != args.angle_end else [args.angle_start]
    producer.scale_range = [args.scale_start, args.scale_end] if args.scale_start != args.scale_end else [args.scale_start]
    producer.angle_step = args.angle_step
    producer.scale_step = args.scale_step
    infos = producer.produce_infos()

    class_id = "object"
    success = 0
    for info in infos:
        src_i = producer.src_of(info)
        mask_i = producer.mask_of(info)
        nfeat = max(16, int(round(args.num_features * info.scale)))
        templ_id = detector.add_template(
            src_i,
            class_id=class_id,
            object_mask=mask_i,
            num_features=nfeat,
            metadata={"angle": info.angle, "scale": info.scale},
        )
        if templ_id >= 0:
            success += 1

    if success == 0:
        print("No template could be extracted.")
        return 2

    matches = detector.match(scene, threshold=args.threshold, class_ids=[class_id], mask=scene_mask)
    matches = nms_matches(detector, matches, iou_threshold=args.nms_iou)
    matches.sort(key=lambda m: m.similarity, reverse=True)

    topk = min(args.topk, len(matches))
    for i in range(topk):
        m = matches[i]
        meta = detector.get_template_meta(m.class_id, m.template_id)
        t0 = detector.get_templates(m.class_id, m.template_id)[0]
        angle = float(meta.get("angle", 0.0))
        scale = float(meta.get("scale", 1.0))
        print(
            f"[{i+1}] sim={m.similarity:.2f}, x={m.x}, y={m.y}, "
            f"templ_id={m.template_id}, angle={angle:.2f}, scale={scale:.3f}, "
            f"w={t0.width}, h={t0.height}"
        )

    overlay = draw_matches(detector, scene, matches, topk=args.topk)
    ok = cv2.imwrite(args.out, overlay)
    if not ok:
        raise RuntimeError(f"Failed to write output: {args.out}")
    print(f"templates_loaded={success}, raw_matches={len(matches)}, saved={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
