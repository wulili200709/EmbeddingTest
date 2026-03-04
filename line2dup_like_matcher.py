#!/usr/bin/env python3
"""
line2Dup/shape_based_matching style matcher in Python (OpenCV + NumPy).

This follows the core ideas from:
https://github.com/meiqua/shape_based_matching
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
    cpp_path = Path(__file__).resolve().parent / "_third_party_shape_based_matching" / "line2Dup.cpp"
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
    response_maps: Sequence[np.ndarray],
    templ: TemplateLevel,
    width: int,
    height: int,
    T: int,
) -> np.ndarray:
    if not templ.features:
        return np.empty((0, 0), dtype=np.float32)

    W = width // T
    H = height // T
    wf = (templ.width - 1) // T + 1
    hf = (templ.height - 1) // T + 1
    span_x = W - wf
    span_y = H - hf
    if span_x < 0 or span_y < 0:
        return np.empty((0, 0), dtype=np.float32)

    dst = np.zeros((span_y + 1, span_x + 1), dtype=np.float32)
    y_len = (span_y + 1) * T
    x_len = (span_x + 1) * T

    for f in templ.features:
        if f.label < 0 or f.label >= len(response_maps):
            continue
        rm = response_maps[f.label]
        y0 = f.y
        x0 = f.x
        y1 = y0 + y_len
        x1 = x0 + x_len
        if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
            continue
        patch = rm[y0:y1:T, x0:x1:T]
        if patch.shape != dst.shape:
            continue
        dst += patch.astype(np.float32)

    dst *= (100.0 / (4.0 * max(1, len(templ.features))))
    return dst


def similarity_local(
    response_maps: Sequence[np.ndarray],
    templ: TemplateLevel,
    width: int,
    height: int,
    T: int,
    center_xy: Tuple[int, int],
) -> np.ndarray:
    if not templ.features:
        return np.empty((0, 0), dtype=np.float32)

    center_x, center_y = center_xy
    offset_x = (center_x // T - 8) * T
    offset_y = (center_y // T - 8) * T
    dst = np.zeros((16, 16), dtype=np.float32)

    for f in templ.features:
        if f.label < 0 or f.label >= len(response_maps):
            continue
        rm = response_maps[f.label]
        x0 = f.x + offset_x
        y0 = f.y + offset_y
        x1 = x0 + 16 * T
        y1 = y0 + 16 * T
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue
        patch = rm[y0:y1:T, x0:x1:T]
        if patch.shape != dst.shape:
            continue
        dst += patch.astype(np.float32)

    dst *= (100.0 / (4.0 * max(1, len(templ.features))))
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
        self.class_templates: Dict[str, List[List[TemplateLevel]]] = {}
        self.class_meta: Dict[str, List[Dict[str, float]]] = {}

    def add_template(
        self,
        source: np.ndarray,
        class_id: str,
        object_mask: Optional[np.ndarray] = None,
        num_features: Optional[int] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> int:
        if class_id not in self.class_templates:
            self.class_templates[class_id] = []
            self.class_meta[class_id] = []

        nfeat = int(num_features) if num_features is not None and num_features > 0 else self.num_features
        qp = ColorGradientPyramid(
            source,
            object_mask,
            weak_threshold=self.weak_threshold,
            num_features=nfeat,
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
        self.class_meta[class_id].append(metadata or {})
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
        self.class_meta[class_id].append(metadata or {})
        return template_id

    def get_templates(self, class_id: str, template_id: int) -> List[TemplateLevel]:
        return self.class_templates[class_id][template_id]

    def get_template_meta(self, class_id: str, template_id: int) -> Dict[str, float]:
        return self.class_meta[class_id][template_id]

    def class_ids(self) -> List[str]:
        return list(self.class_templates.keys())

    def match(
        self,
        source: np.ndarray,
        threshold: float,
        class_ids: Optional[Sequence[str]] = None,
        mask: Optional[np.ndarray] = None,
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
            scene_levels.append(SceneLevelData(width=w, height=h, T=T, response_maps=response_maps))

        if len(scene_levels) != self.pyramid_levels:
            return []

        if class_ids is None or len(class_ids) == 0:
            class_ids = self.class_ids()

        matches: List[Match] = []
        for class_id in class_ids:
            if class_id not in self.class_templates:
                continue
            self._match_class(scene_levels, float(threshold), class_id, self.class_templates[class_id], matches)

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

    def _match_class(
        self,
        scene_levels: Sequence[SceneLevelData],
        threshold: float,
        class_id: str,
        template_pyramids: Sequence[List[TemplateLevel]],
        out_matches: List[Match],
    ) -> None:
        # In line2Dup-style coarse-to-fine search, using the same high threshold at every
        # pyramid stage can drop valid matches too early. Use a lower internal threshold for
        # candidate propagation, then keep the user threshold as final filtering criterion.
        final_threshold = float(threshold)
        coarse_threshold = float(final_threshold)
        if final_threshold >= 35.0:
            # Empirically, valid instances can score much lower on the coarsest stage than on
            # the final stage. Keep internal threshold around [35, final], with 90 -> 45.
            coarse_threshold = min(final_threshold, max(35.0, final_threshold - 45.0))

        for template_id, tp in enumerate(template_pyramids):
            lowest_idx = self.pyramid_levels - 1
            lowest_scene = scene_levels[lowest_idx]
            lowest_templ = tp[lowest_idx]

            sim_map = similarity_full(
                lowest_scene.response_maps,
                lowest_templ,
                lowest_scene.width,
                lowest_scene.height,
                lowest_scene.T,
            )
            if sim_map.size == 0:
                continue

            candidates: List[Match] = []
            low_T = lowest_scene.T
            off = offset_from_T(low_T)
            ys, xs = np.where(sim_map > coarse_threshold)
            for r, c in zip(ys.tolist(), xs.tolist()):
                x = int(c * low_T + off)
                y = int(r * low_T + off)
                score = float(sim_map[r, c])
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
                    # Fallback for large templates / small scenes: local 16x16 refinement window
                    # cannot fit near borders, so we only propagate coarse positions to finer level.
                    # This keeps same-image and near-full-frame matching usable.
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
                for m in candidates:
                    x = m.x * 2 + 1
                    y = m.y * 2 + 1
                    x = max(x, border)
                    y = max(y, border)
                    x = min(x, max_x)
                    y = min(y, max_y)

                    local = similarity_local(
                        scene.response_maps,
                        templ,
                        scene.width,
                        scene.height,
                        T,
                        (x, y),
                    )
                    if local.size == 0:
                        continue
                    best_idx = int(np.argmax(local))
                    best_r, best_c = divmod(best_idx, local.shape[1])
                    best_score = float(local[best_r, best_c])
                    if best_score < coarse_threshold:
                        continue

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


def detector_to_dict(detector: Line2DupLikeDetector) -> Dict[str, Any]:
    classes: Dict[str, Any] = {}
    for class_id, template_pyramids in detector.class_templates.items():
        metas = detector.class_meta.get(class_id, [])
        packed_templates: List[Dict[str, Any]] = []
        for tid, tp in enumerate(template_pyramids):
            meta = metas[tid] if tid < len(metas) else {}
            packed_templates.append(
                {
                    "template_id": int(tid),
                    "meta": dict(meta) if isinstance(meta, dict) else {},
                    "levels": [_template_level_to_dict(lv) for lv in tp],
                }
            )
        classes[class_id] = packed_templates

    return {
        "format": "line2dup_like_model_v1",
        "params": {
            "num_features": int(detector.num_features),
            "T_levels": [int(t) for t in detector.T_at_level],
            "weak_threshold": float(detector.weak_threshold),
            "strong_threshold": float(detector.strong_threshold),
        },
        "classes": classes,
    }


def detector_from_dict(data: Dict[str, Any]) -> Line2DupLikeDetector:
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

    detector.class_templates = {}
    detector.class_meta = {}

    for class_id, packed_templates in classes.items():
        detector.class_templates[class_id] = []
        detector.class_meta[class_id] = []
        if not isinstance(packed_templates, list):
            continue
        for item in packed_templates:
            levels_raw = item.get("levels", []) if isinstance(item, dict) else []
            levels = [_template_level_from_dict(x) for x in levels_raw]
            if len(levels) != detector.pyramid_levels:
                raise ValueError(
                    f"Template levels mismatch for class '{class_id}': "
                    f"expected {detector.pyramid_levels}, got {len(levels)}"
                )
            meta = item.get("meta", {}) if isinstance(item, dict) else {}
            detector.class_templates[class_id].append(levels)
            detector.class_meta[class_id].append(dict(meta) if isinstance(meta, dict) else {})

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
        templ0 = detector.get_templates(m.class_id, m.template_id)[0]
        boxes.append([int(m.x), int(m.y), int(max(1, templ0.width)), int(max(1, templ0.height))])
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


def strict_orientation_score(
    scene_quant: np.ndarray,
    templ: TemplateLevel,
    match: Match,
    window: int = 1,
    neighbor_half_credit: bool = True,
) -> float:
    h, w = scene_quant.shape[:2]
    win = max(0, int(window))
    exact = 0.0
    soft = 0.0
    valid = 0

    for f in templ.features:
        x = int(match.x + f.x)
        y = int(match.y + f.y)
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        x0 = max(0, x - win)
        y0 = max(0, y - win)
        x1 = min(w, x + win + 1)
        y1 = min(h, y + win + 1)
        patch = scene_quant[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        valid += 1

        target = np.uint8(1 << int(f.label))
        if np.any((patch & target) != 0):
            exact += 1.0
            continue

        if neighbor_half_credit:
            left = np.uint8(1 << ((int(f.label) + 7) % 8))
            right = np.uint8(1 << ((int(f.label) + 1) % 8))
            if np.any((patch & left) != 0) or np.any((patch & right) != 0):
                soft += 0.5

    if valid <= 0:
        return 0.0
    return float((exact + soft) * 100.0 / valid)


def verify_matches_strict_orientation(
    detector: Line2DupLikeDetector,
    scene_bgr: np.ndarray,
    matches: Sequence[Match],
    verify_window: int = 1,
    min_verify_score: float = 0.0,
    blend_with_line2dup: float = 0.35,
) -> List[Match]:
    if not matches:
        return []

    qp = ColorGradientPyramid(
        scene_bgr,
        None,
        weak_threshold=detector.weak_threshold,
        num_features=detector.num_features,
        strong_threshold=detector.strong_threshold,
    )
    scene_quant = qp.quantize()

    out: List[Match] = []
    alpha = float(np.clip(blend_with_line2dup, 0.0, 1.0))
    for m in matches:
        t0 = detector.get_templates(m.class_id, m.template_id)[0]
        strict = strict_orientation_score(
            scene_quant=scene_quant,
            templ=t0,
            match=m,
            window=verify_window,
            neighbor_half_credit=True,
        )
        if strict < float(min_verify_score):
            continue
        # Keep line2Dup ranking signal, but calibrate with stricter score.
        calibrated = alpha * float(m.similarity) + (1.0 - alpha) * strict
        out.append(
            Match(
                x=m.x,
                y=m.y,
                similarity=float(calibrated),
                class_id=m.class_id,
                template_id=m.template_id,
            )
        )

    out.sort(key=lambda mm: mm.similarity, reverse=True)
    return out


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
        t0 = detector.get_templates(m.class_id, m.template_id)[0]
        meta = detector.get_template_meta(m.class_id, m.template_id)
        color = palette[i % len(palette)]

        # Draw matched template feature points in scene coordinates.
        pts: List[Tuple[float, float]] = []
        for f in t0.features:
            px = float(f.x + m.x)
            py = float(f.y + m.y)
            pts.append((px, py))
            pxi = int(round(px))
            pyi = int(round(py))
            theta = float(f.theta)
            if not np.isfinite(theta):
                theta = float((int(f.label) % 8) * 45.0)
            rad = np.deg2rad(theta)
            p2x = int(round(pxi + 7.0 * float(np.cos(rad))))
            p2y = int(round(pyi + 7.0 * float(np.sin(rad))))
            # Draw dark outline then color line for visibility.
            cv2.arrowedLine(out, (pxi, pyi), (p2x, p2y), (0, 0, 0), 3, cv2.LINE_AA, 0, 0.35)
            cv2.arrowedLine(out, (pxi, pyi), (p2x, p2y), color, 1, cv2.LINE_AA, 0, 0.35)

        # Draw stable template bbox (width/height from matched template).
        # Using minAreaRect on sparse/one-sided feature clouds may degenerate to a near-line.
        x1 = int(m.x)
        y1 = int(m.y)
        x2 = int(m.x + t0.width)
        y2 = int(m.y + t0.height)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        tx = x1
        ty = y1

        ang = float(meta.get("angle", 0.0))
        label = f"#{i+1} {m.similarity:.1f} a={ang:.0f}"
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
    parser.add_argument(
        "--verify-strict",
        action="store_true",
        help="Apply strict orientation verification to suppress false high scores.",
    )
    parser.add_argument(
        "--verify-window",
        type=int,
        default=1,
        help="Neighborhood radius (in pixels) for strict orientation verification.",
    )
    parser.add_argument(
        "--verify-min",
        type=float,
        default=0.0,
        help="Drop matches with strict verification score below this value.",
    )
    parser.add_argument(
        "--verify-blend",
        type=float,
        default=0.35,
        help="Final score = blend*line2dup + (1-blend)*strict_verify.",
    )
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
    if args.verify_strict:
        matches = verify_matches_strict_orientation(
            detector=detector,
            scene_bgr=scene,
            matches=matches,
            verify_window=args.verify_window,
            min_verify_score=args.verify_min,
            blend_with_line2dup=args.verify_blend,
        )
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
