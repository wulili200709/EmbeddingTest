from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


SubPixelMode = Literal["none", "interpolation"]


def _ensure_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _angle_wrap_pi_scalar(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _quantize_angle_0_2pi(phi: np.ndarray, nbins: int) -> np.ndarray:
    phi_0 = np.mod(phi, 2.0 * np.pi)
    bins = np.floor(phi_0 * (nbins / (2.0 * np.pi))).astype(np.int32)
    return np.clip(bins, 0, nbins - 1).astype(np.int16)


def _undirected_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(_angle_wrap_pi(a - b))
    return np.minimum(d, np.pi - d)


def _edges_and_orientation(
    gray_u8: np.ndarray,
    canny1: int,
    canny2: int,
    sobel_ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = cv2.Canny(gray_u8, canny1, canny2)
    edges_bool = edges.astype(bool)
    dx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    dy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    phi = np.arctan2(dy, dx)
    phi = np.mod(phi, 2.0 * np.pi).astype(np.float32)
    mag = cv2.magnitude(dx, dy).astype(np.float32)
    return edges_bool, phi, mag


def _centroid_from_mask(mask_u8: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(mask_u8, binaryImage=True)
    if abs(m["m00"]) < 1e-6:
        ys, xs = np.nonzero(mask_u8)
        if len(xs) == 0:
            raise ValueError("Empty mask; cannot compute centroid.")
        return float(np.mean(ys)), float(np.mean(xs))
    cy = m["m01"] / m["m00"]
    cx = m["m10"] / m["m00"]
    return float(cy), float(cx)


def _auto_pyramid_levels(h: int, w: int, min_size: int = 80, max_levels: int = 4) -> int:
    levels = 1
    while levels < max_levels and min(h, w) / (2 ** (levels - 1)) >= min_size * 2:
        levels += 1
    return levels


def _parabolic_subpixel(fm1: float, f0: float, fp1: float) -> float:
    denom = fm1 - 2.0 * f0 + fp1
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (fm1 - fp1) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def _rotated_rect_iou(rect1, rect2) -> float:
    try:
        ret, inter = cv2.rotatedRectangleIntersection(rect1, rect2)
    except cv2.error:
        return 0.0
    if ret == 0 or inter is None:
        return 0.0
    inter = np.asarray(inter, dtype=np.float32)
    if inter.ndim != 3 or inter.shape[1:] != (1, 2):
        inter = inter.reshape(-1, 1, 2)
    area_inter = float(abs(cv2.contourArea(inter)))
    area1 = float(rect1[1][0] * rect1[1][1])
    area2 = float(rect2[1][0] * rect2[1][1])
    denom = area1 + area2 - area_inter
    if denom <= 0:
        return 0.0
    return float(np.clip(area_inter / denom, 0.0, 1.0))


def _resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    inter = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, (nw, nh), interpolation=inter)


def _resize_mask(mask_u8: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    if mask_u8 is None:
        return None
    if abs(scale - 1.0) < 1e-6:
        return mask_u8
    h, w = mask_u8.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    out = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8) * 255


@dataclass(frozen=True)
class Match:
    row: float
    col: float
    angle: float
    scale: float
    score: float


@dataclass(frozen=True)
class _ModelLevel:
    scale_factor: float
    rel_xy: np.ndarray
    phi: np.ndarray
    pre_xy: np.ndarray
    pre_phi: np.ndarray


class ScaledShapeModel:
    """
    HALCON-like shape model matcher:
    - create(): build model points from edge features
    - find(): fast candidate generation (GeneralizedHoughGuil) + soft chamfer refinement
    - save()/load(): persist model data
    """

    def __init__(
        self,
        *,
        nbins: int,
        origin_rc: Tuple[float, float],
        r_table: Sequence[np.ndarray],
        model_rel_xy: np.ndarray,
        model_phi: np.ndarray,
        base_size_wh: Tuple[float, float],
        canny1: int,
        canny2: int,
        template_u8: Optional[np.ndarray] = None,
        template_mask_u8: Optional[np.ndarray] = None,
    ) -> None:
        self.nbins = int(nbins)
        self.origin_rc = (float(origin_rc[0]), float(origin_rc[1]))  # (row, col)
        self.r_table = [np.asarray(v, dtype=np.float32).reshape(-1, 2) for v in r_table]
        self.model_rel_xy = np.asarray(model_rel_xy, dtype=np.float32).reshape(-1, 2)  # (dcol, drow)
        self.model_phi = np.asarray(model_phi, dtype=np.float32).reshape(-1)
        self.base_size_wh = (float(base_size_wh[0]), float(base_size_wh[1]))
        self.canny1 = int(canny1)
        self.canny2 = int(canny2)

        if len(self.r_table) != self.nbins:
            self.r_table = [np.zeros((0, 2), dtype=np.float32) for _ in range(self.nbins)]
        if self.model_rel_xy.shape[0] != self.model_phi.shape[0]:
            raise ValueError("model_rel_xy and model_phi size mismatch")

        self.template_u8 = _ensure_gray_u8(template_u8) if template_u8 is not None else None
        if template_mask_u8 is None:
            self.template_mask_u8 = None
        else:
            if template_mask_u8.ndim == 3:
                template_mask_u8 = cv2.cvtColor(template_mask_u8, cv2.COLOR_BGR2GRAY)
            self.template_mask_u8 = (template_mask_u8 > 0).astype(np.uint8) * 255

        self._template_cache: dict[int, Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float]]] = {}
        self._synthetic_template: Optional[np.ndarray] = None
        self._synthetic_mask: Optional[np.ndarray] = None
        self._synthetic_origin_rc: Optional[Tuple[float, float]] = None
        self._model_pyramid: list[_ModelLevel] = []
        self._build_model_pyramid(max_levels=5)

    @classmethod
    def create(
        cls,
        template_image: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        nbins: int = 30,
        canny1: int = 50,
        canny2: int = 150,
        max_r_vectors_per_bin: int = 80,
        max_model_points: int = 4000,
        rng_seed: int = 0,
    ) -> "ScaledShapeModel":
        gray = _ensure_gray_u8(template_image)
        if mask is not None:
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_u8 = (mask > 0).astype(np.uint8) * 255
        else:
            mask_u8 = None

        edges, phi, mag = _edges_and_orientation(gray, canny1, canny2)
        if mask_u8 is not None:
            edges &= mask_u8.astype(bool)

        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            raise ValueError("No edges found in template (after mask). Adjust ROI/Canny.")

        if mask_u8 is not None:
            origin_r, origin_c = _centroid_from_mask(mask_u8)
        else:
            origin_r = float(np.mean(ys))
            origin_c = float(np.mean(xs))

        edge_mag = mag[ys, xs].astype(np.float32)
        order = np.argsort(-edge_mag)
        rng = np.random.default_rng(rng_seed)
        if order.size > max_model_points:
            keep_pool = order[: min(order.size, max_model_points * 3)]
            chosen = rng.choice(keep_pool.size, size=max_model_points, replace=False)
            order = keep_pool[chosen]

        sel_x = xs[order].astype(np.float32)
        sel_y = ys[order].astype(np.float32)
        sel_phi = phi[ys[order], xs[order]].astype(np.float32)
        rel = np.stack([sel_x - origin_c, sel_y - origin_r], axis=1).astype(np.float32)

        # Keep r_table for backward compatibility.
        phi_bins = _quantize_angle_0_2pi(phi, nbins)
        r_table: list[list[Tuple[float, float]]] = [[] for _ in range(nbins)]
        for x_i, y_i in zip(xs.tolist(), ys.tolist()):
            b = int(phi_bins[y_i, x_i])
            vec = (float(origin_c - x_i), float(origin_r - y_i))
            r_table[b].append(vec)
            b_opp = (b + (nbins // 2)) % nbins
            if b_opp != b:
                r_table[b_opp].append(vec)

        r_table_np: list[np.ndarray] = []
        for b in range(nbins):
            vecs = np.asarray(r_table[b], dtype=np.float32).reshape(-1, 2)
            if vecs.shape[0] > max_r_vectors_per_bin:
                take = rng.choice(vecs.shape[0], size=max_r_vectors_per_bin, replace=False)
                vecs = vecs[take]
            r_table_np.append(vecs)

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        if w <= 1e-3 or h <= 1e-3:
            w = float(np.max(xs) - np.min(xs) + 1)
            h = float(np.max(ys) - np.min(ys) + 1)

        return cls(
            nbins=nbins,
            origin_rc=(origin_r, origin_c),
            r_table=r_table_np,
            model_rel_xy=rel,
            model_phi=sel_phi,
            base_size_wh=(float(w), float(h)),
            canny1=canny1,
            canny2=canny2,
            # Use synthetic sparse template from sampled model points by default.
            # This avoids extremely slow GHT on very large dense templates.
            template_u8=None,
            template_mask_u8=None,
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        r_obj = np.empty((self.nbins,), dtype=object)
        for i, v in enumerate(self.r_table):
            r_obj[i] = np.asarray(v, dtype=np.float32)

        has_template = 1 if self.template_u8 is not None else 0
        has_mask = 1 if self.template_mask_u8 is not None else 0
        template_u8 = self.template_u8 if self.template_u8 is not None else np.zeros((0, 0), dtype=np.uint8)
        template_mask_u8 = (
            self.template_mask_u8 if self.template_mask_u8 is not None else np.zeros((0, 0), dtype=np.uint8)
        )

        np.savez_compressed(
            path,
            version=np.int32(2),
            nbins=np.int32(self.nbins),
            origin_rc=np.asarray(self.origin_rc, dtype=np.float32),
            r_table=r_obj,
            model_rel_xy=self.model_rel_xy.astype(np.float32),
            model_phi=self.model_phi.astype(np.float32),
            base_size_wh=np.asarray(self.base_size_wh, dtype=np.float32),
            canny1=np.int32(self.canny1),
            canny2=np.int32(self.canny2),
            has_template=np.int32(has_template),
            has_mask=np.int32(has_mask),
            template_u8=template_u8,
            template_mask_u8=template_mask_u8,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScaledShapeModel":
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        nbins = int(data["nbins"])
        origin_rc = tuple(float(x) for x in data["origin_rc"].tolist())

        if "r_table" in data:
            r_obj = data["r_table"]
            r_table = [np.asarray(r_obj[i], dtype=np.float32).reshape(-1, 2) for i in range(nbins)]
        else:
            r_table = [np.zeros((0, 2), dtype=np.float32) for _ in range(nbins)]

        model_rel_xy = np.asarray(data["model_rel_xy"], dtype=np.float32).reshape(-1, 2)
        model_phi = np.asarray(data["model_phi"], dtype=np.float32).reshape(-1)
        base_size_wh = tuple(float(x) for x in data["base_size_wh"].tolist())
        canny1 = int(data["canny1"])
        canny2 = int(data["canny2"])

        template_u8 = None
        template_mask_u8 = None
        if "has_template" in data and int(data["has_template"]) > 0 and "template_u8" in data:
            arr = np.asarray(data["template_u8"], dtype=np.uint8)
            if arr.size > 0:
                template_u8 = arr
        elif "template_u8" in data:
            arr = np.asarray(data["template_u8"], dtype=np.uint8)
            if arr.size > 0:
                template_u8 = arr

        if "has_mask" in data and int(data["has_mask"]) > 0 and "template_mask_u8" in data:
            arr = np.asarray(data["template_mask_u8"], dtype=np.uint8)
            if arr.size > 0:
                template_mask_u8 = arr
        elif "template_mask_u8" in data:
            arr = np.asarray(data["template_mask_u8"], dtype=np.uint8)
            if arr.size > 0:
                template_mask_u8 = arr

        return cls(
            nbins=nbins,
            origin_rc=origin_rc,
            r_table=r_table,
            model_rel_xy=model_rel_xy,
            model_phi=model_phi,
            base_size_wh=base_size_wh,
            canny1=canny1,
            canny2=canny2,
            template_u8=template_u8,
            template_mask_u8=template_mask_u8,
        )

    @staticmethod
    def _pick_spread_indices(xy: np.ndarray, target_n: int) -> np.ndarray:
        n = int(xy.shape[0])
        if n <= target_n:
            return np.arange(n, dtype=np.int32)
        if target_n <= 0:
            return np.zeros((0,), dtype=np.int32)

        mins = np.min(xy, axis=0)
        maxs = np.max(xy, axis=0)
        spans = np.maximum(maxs - mins, 1e-3)
        grid = int(max(2, round(math.sqrt(target_n))))
        gx = np.clip(((xy[:, 0] - mins[0]) / spans[0] * (grid - 1)).astype(np.int32), 0, grid - 1)
        gy = np.clip(((xy[:, 1] - mins[1]) / spans[1] * (grid - 1)).astype(np.int32), 0, grid - 1)
        cell = gy * grid + gx

        order = np.argsort(cell, kind="stable")
        cell_sorted = cell[order]
        chosen: list[int] = []
        i = 0
        while i < order.size and len(chosen) < target_n:
            j = i + 1
            while j < order.size and cell_sorted[j] == cell_sorted[i]:
                j += 1
            chosen.append(int(order[(i + j - 1) // 2]))
            i = j

        if len(chosen) < target_n:
            used = np.zeros((n,), dtype=bool)
            if chosen:
                used[np.asarray(chosen, dtype=np.int32)] = True
            rest = np.nonzero(~used)[0]
            need = target_n - len(chosen)
            if rest.size > 0 and need > 0:
                step = max(1, int(round(rest.size / need)))
                chosen.extend(rest[::step][:need].tolist())

        return np.asarray(chosen[:target_n], dtype=np.int32)

    def _build_model_pyramid(self, max_levels: int = 5) -> None:
        self._model_pyramid = []
        rel0 = self.model_rel_xy.astype(np.float32)
        phi0 = self.model_phi.astype(np.float32)
        if rel0.shape[0] == 0:
            self._model_pyramid.append(
                _ModelLevel(
                    scale_factor=1.0,
                    rel_xy=np.zeros((0, 2), dtype=np.float32),
                    phi=np.zeros((0,), dtype=np.float32),
                    pre_xy=np.zeros((0, 2), dtype=np.float32),
                    pre_phi=np.zeros((0,), dtype=np.float32),
                )
            )
            return

        # Coarser levels keep fewer points for fast pre-pruning.
        level_caps = [2400, 1400, 850, 480, 300]
        total_n = int(rel0.shape[0])
        for lv in range(max(1, int(max_levels))):
            sf = 1.0 / float(2**lv)
            pts = (rel0 * sf).astype(np.float32)
            ph = phi0

            # Deduplicate near-overlapping points after downsampling.
            key = np.rint(pts).astype(np.int32)
            _, idx = np.unique(key, axis=0, return_index=True)
            idx = np.sort(idx.astype(np.int32))
            pts = pts[idx]
            ph = ph[idx]

            cap = level_caps[min(lv, len(level_caps) - 1)]
            cap = int(min(cap, total_n))
            if pts.shape[0] > cap:
                keep = self._pick_spread_indices(pts, cap)
                pts = pts[keep]
                ph = ph[keep]

            pre_n = int(min(max(64, pts.shape[0] // 5), 220, pts.shape[0]))
            pre_idx = self._pick_spread_indices(pts, pre_n)
            pre_xy = pts[pre_idx]
            pre_phi = ph[pre_idx]

            self._model_pyramid.append(
                _ModelLevel(
                    scale_factor=sf,
                    rel_xy=pts.astype(np.float32),
                    phi=ph.astype(np.float32),
                    pre_xy=pre_xy.astype(np.float32),
                    pre_phi=pre_phi.astype(np.float32),
                )
            )

            if min(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) < 6.0 and lv >= 2:
                break

    def _build_synthetic_template_from_points(self) -> None:
        if self._synthetic_template is not None:
            return
        if self.model_rel_xy.shape[0] == 0:
            self._synthetic_template = np.zeros((32, 32), dtype=np.uint8)
            self._synthetic_mask = None
            self._synthetic_origin_rc = (16.0, 16.0)
            return

        dcol = self.model_rel_xy[:, 0]
        drow = self.model_rel_xy[:, 1]
        min_dc, max_dc = float(np.min(dcol)), float(np.max(dcol))
        min_dr, max_dr = float(np.min(drow)), float(np.max(drow))

        margin = 8.0
        w = int(max(16, math.ceil(max_dc - min_dc + 1.0 + 2.0 * margin)))
        h = int(max(16, math.ceil(max_dr - min_dr + 1.0 + 2.0 * margin)))
        origin_c = -min_dc + margin
        origin_r = -min_dr + margin

        tpl = np.zeros((h, w), dtype=np.uint8)
        xs = np.rint(origin_c + dcol).astype(np.int32)
        ys = np.rint(origin_r + drow).astype(np.int32)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        tpl[ys, xs] = 255
        tpl = cv2.dilate(tpl, np.ones((3, 3), dtype=np.uint8), iterations=1)

        self._synthetic_template = tpl
        self._synthetic_mask = (tpl > 0).astype(np.uint8) * 255
        self._synthetic_origin_rc = (float(origin_r), float(origin_c))

    def _get_template_pack(
        self,
        work_scale: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float]]:
        key = int(round(float(work_scale) * 1000.0))
        cached = self._template_cache.get(key)
        if cached is not None:
            return cached

        if self.template_u8 is not None:
            base_tpl = self.template_u8
            base_mask = self.template_mask_u8
            base_origin_rc = self.origin_rc
        else:
            self._build_synthetic_template_from_points()
            assert self._synthetic_template is not None
            assert self._synthetic_origin_rc is not None
            base_tpl = self._synthetic_template
            base_mask = self._synthetic_mask
            base_origin_rc = self._synthetic_origin_rc

        tpl = _resize_gray(base_tpl, work_scale)
        msk = _resize_mask(base_mask, work_scale)
        origin = (float(base_origin_rc[0] * work_scale), float(base_origin_rc[1] * work_scale))

        if min(tpl.shape[:2]) < 12 and work_scale < 1.0:
            tpl = base_tpl
            msk = base_mask
            origin = (float(base_origin_rc[0]), float(base_origin_rc[1]))

        self._template_cache[key] = (tpl, msk, origin)
        return self._template_cache[key]

    @staticmethod
    def _angle_segments_deg(
        angle_start: float,
        angle_extent: float,
        fallback_extent_deg: float,
    ) -> list[Tuple[float, float]]:
        ext_deg = abs(math.degrees(angle_extent))
        if ext_deg >= 360.0 - 1e-6:
            return [(0.0, 360.0)]
        if ext_deg <= 1e-6:
            ext_deg = max(0.5, float(fallback_extent_deg))

        s = math.degrees(angle_start) % 360.0
        e = s + ext_deg
        if e <= 360.0:
            hi = max(s + 1e-3, e)
            return [(float(s), float(hi))]
        hi1 = 360.0
        hi2 = e - 360.0
        out: list[Tuple[float, float]] = []
        if hi1 - s > 1e-3:
            out.append((float(s), float(hi1)))
        if hi2 > 1e-3:
            out.append((0.0, float(hi2)))
        return out

    @staticmethod
    def _angle_in_query(angle: float, angle_start: float, angle_extent: float) -> bool:
        if angle_extent >= 2.0 * math.pi - 1e-6:
            return True
        if angle_extent <= 1e-9:
            return abs(_angle_wrap_pi_scalar(angle - angle_start)) <= math.radians(1.0)
        d = (angle - angle_start) % (2.0 * math.pi)
        return bool(d <= angle_extent + 1e-6)

    @staticmethod
    def _choose_work_scale(
        h: int,
        w: int,
        *,
        num_levels: int,
        greediness: float,
        angle_count: int,
        scale_count: int,
    ) -> float:
        if num_levels > 0:
            levels = max(1, int(num_levels))
        else:
            levels = _auto_pyramid_levels(h, w)
        scale = 1.0 / float(2 ** (levels - 1))

        complexity = max(1, int(angle_count) * int(scale_count))
        if complexity > 700:
            scale = min(scale, 0.5)
        if complexity > 1400:
            scale = min(scale, 0.4)
        if greediness >= 0.9 and complexity > 500:
            scale = min(scale, 0.5)

        area = int(h) * int(w)
        if area > 1_200_000 and complexity > 500:
            scale = min(scale, 0.4)
        if area > 2_000_000 and complexity > 500:
            scale = min(scale, 0.33)

        scale = float(np.clip(scale, 0.25, 1.0))
        return scale

    def _collect_candidates_ght(
        self,
        gray0: np.ndarray,
        *,
        angle_start: float,
        angle_extent: float,
        scale_min: float,
        scale_max: float,
        angle_step_deg: float,
        scale_step: float,
        greediness: float,
        work_scale: float,
        max_candidates: int,
    ) -> list[Tuple[float, float, float, float, float]]:
        img = _resize_gray(gray0, work_scale)
        if img.size == 0:
            return []

        templ, templ_mask, templ_center = self._get_template_pack(work_scale)
        if min(templ.shape[:2]) < 8:
            return []

        edges_i = cv2.Canny(img, self.canny1, self.canny2)
        if np.count_nonzero(edges_i) < 20:
            return []

        templ_for_ght = templ.copy()
        if templ_mask is not None:
            templ_for_ght[templ_mask == 0] = 0

        edges_t = cv2.Canny(templ_for_ght, self.canny1, self.canny2)
        if np.count_nonzero(edges_t) < 15:
            return []

        segs = self._angle_segments_deg(angle_start, angle_extent, angle_step_deg)
        all_cands: list[Tuple[float, float, float, float, float]] = []
        edge_count_t = int(np.count_nonzero(edges_t))
        base_vote = max(6, int(edge_count_t * 0.012))

        for min_a, max_a in segs:
            if max_a - min_a <= 1e-3:
                continue
            seg_found = False
            for relax in (1.0, 0.55, 0.25):
                gh = cv2.createGeneralizedHoughGuil()
                gh.setTemplate(
                    templ_for_ght,
                    (int(round(float(templ_center[1]))), int(round(float(templ_center[0])))),
                )
                gh.setCannyLowThresh(int(self.canny1))
                gh.setCannyHighThresh(int(self.canny2))

                gh.setMinAngle(float(max(0.0, min_a)))
                gh.setMaxAngle(float(min(360.0, max_a)))
                gh.setAngleStep(float(max(0.5, angle_step_deg)))

                smin = float(max(1e-3, scale_min))
                smax = float(max(smin + 1e-3, scale_max))
                gh.setMinScale(smin)
                gh.setMaxScale(smax)
                gh.setScaleStep(float(max(0.005, scale_step)))

                if work_scale >= 0.7:
                    dp = 1.5
                elif work_scale >= 0.5:
                    dp = 2.0
                elif work_scale >= 0.35:
                    dp = 2.5
                else:
                    dp = 3.0
                dp = float(min(4.0, dp + 0.3 * float(np.clip(greediness, 0.0, 1.0))))
                gh.setDp(dp)
                gh.setMinDist(float(max(4.0, min(templ.shape[:2]) * 0.12)))

                thr = max(2, int(round(base_vote * relax)))
                gh.setPosThresh(thr)
                gh.setAngleThresh(thr)
                gh.setScaleThresh(thr)

                try:
                    positions, votes = gh.detect(img)
                except cv2.error:
                    continue
                if positions is None:
                    continue

                pos = np.asarray(positions, dtype=np.float32).reshape(-1, positions.shape[-1])
                if pos.shape[0] == 0:
                    continue

                v_arr = None
                if votes is not None:
                    v_arr = np.asarray(votes, dtype=np.float32).reshape(-1, votes.shape[-1])

                for i in range(pos.shape[0]):
                    x, y, sc, ang_deg = pos[i, :4].tolist()
                    angle = _angle_wrap_pi_scalar(math.radians(float(ang_deg)))
                    if not self._angle_in_query(angle, angle_start, angle_extent):
                        continue
                    row = float(y / work_scale)
                    col = float(x / work_scale)
                    vote = 0.0
                    if v_arr is not None and i < v_arr.shape[0]:
                        if v_arr.shape[1] >= 3:
                            vote = float(v_arr[i, 0] + 0.1 * v_arr[i, 1] + 0.1 * v_arr[i, 2])
                        else:
                            vote = float(v_arr[i, 0])
                    all_cands.append((row, col, float(angle), float(sc), vote))
                seg_found = seg_found or (pos.shape[0] > 0)
                if seg_found:
                    break

        if not all_cands:
            return []

        all_cands.sort(key=lambda t: t[4], reverse=True)
        uniq: list[Tuple[float, float, float, float, float]] = []
        seen: set[Tuple[int, int, int, int]] = set()
        angle_q = max(math.radians(2.0), math.radians(float(max(0.5, angle_step_deg))))
        scale_q = max(0.01, float(scale_step))
        for row, col, ang, sc, vote in all_cands:
            key = (
                int(round(row / 4.0)),
                int(round(col / 4.0)),
                int(round(ang / angle_q)),
                int(round(sc / scale_q)),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append((row, col, ang, sc, vote))
            if len(uniq) >= max(80, int(max_candidates) * 4):
                break
        return uniq

    def _score_pose_points(
        self,
        dist: np.ndarray,
        phi_img: np.ndarray,
        row: float,
        col: float,
        angle: float,
        scale: float,
        *,
        rel_xy: np.ndarray,
        model_phi: np.ndarray,
        max_dist: float,
        max_ori_diff: float,
        reject_below: Optional[float] = None,
    ) -> float:
        h, w = dist.shape[:2]
        if rel_xy.shape[0] == 0:
            return 0.0

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dcol = rel_xy[:, 0]
        drow = rel_xy[:, 1]
        col_f = col + scale * (cos_a * dcol - sin_a * drow)
        row_f = row + scale * (sin_a * dcol + cos_a * drow)

        col_i = np.rint(col_f).astype(np.int32)
        row_i = np.rint(row_f).astype(np.int32)
        inside = (row_i >= 0) & (row_i < h) & (col_i >= 0) & (col_i < w)
        inside_count = int(np.sum(inside))
        n = int(rel_xy.shape[0])
        if inside_count <= 0:
            return 0.0
        if inside_count < max(8, int(0.15 * n)):
            return 0.0

        max_possible = float(inside_count) / float(max(1, n))
        if reject_below is not None and max_possible < float(reject_below):
            return max_possible

        ri = row_i[inside]
        ci = col_i[inside]
        d_vals = dist[ri, ci]
        exp_phi = np.mod(model_phi[inside] + angle, 2.0 * np.pi)
        img_phi = phi_img[ri, ci]
        diff = _undirected_angle_diff(img_phi, exp_phi)

        d_sigma = max(1e-3, float(max_dist))
        o_sigma = max(math.radians(3.0), float(max_ori_diff))
        d_score = np.exp(-(d_vals * d_vals) / (2.0 * d_sigma * d_sigma))
        o_score = np.exp(-(diff * diff) / (2.0 * o_sigma * o_sigma))
        in_score = float(np.sum(d_score * o_score))
        return in_score / float(max(1, n))

    def _score_pose(
        self,
        dist: np.ndarray,
        phi_img: np.ndarray,
        row: float,
        col: float,
        angle: float,
        scale: float,
        *,
        max_dist: float,
        max_ori_diff: float,
    ) -> float:
        lvl0 = self._model_pyramid[0]
        return self._score_pose_points(
            dist,
            phi_img,
            row,
            col,
            angle,
            scale,
            rel_xy=lvl0.rel_xy,
            model_phi=lvl0.phi,
            max_dist=max_dist,
            max_ori_diff=max_ori_diff,
            reject_below=None,
        )

    def _build_image_feature_pyramid(
        self,
        gray0: np.ndarray,
        levels: int,
    ) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        pyr: list[np.ndarray] = [gray0]
        for _ in range(1, levels):
            pyr.append(cv2.pyrDown(pyr[-1]))

        dist_list: list[np.ndarray] = []
        phi_list: list[np.ndarray] = []
        for g in pyr:
            edges, phi, _ = _edges_and_orientation(g, self.canny1, self.canny2)
            inv = np.where(edges, 0, 255).astype(np.uint8)
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
            dist_list.append(dist)
            phi_list.append(phi.astype(np.float32))
        return pyr, dist_list, phi_list

    def _beam_refine_candidates(
        self,
        seeds: list[Tuple[float, float, float, float, float]],
        *,
        dist_list: list[np.ndarray],
        phi_list: list[np.ndarray],
        levels: int,
        angle_start: float,
        angle_extent: float,
        scale_min: float,
        scale_max: float,
        angle_step: float,
        scale_step: float,
        min_score: float,
        max_dist: float,
        max_ori_diff: float,
        max_candidates: int,
    ) -> list[Match]:
        if not seeds:
            return []

        max_lv = max(0, levels - 1)
        factor0 = float(2**max_lv)
        idx_lv = min(max_lv, len(self._model_pyramid) - 1)
        lvl = self._model_pyramid[idx_lv]

        beam: list[Match] = []
        pre_gate = max(0.02, float(min_score) * 0.35)
        full_gate = max(0.03, float(min_score) * 0.45)
        seed_cap = max(20, int(max_candidates))
        for row, col, ang, sc, vote in seeds[:seed_cap]:
            if sc < scale_min or sc > scale_max:
                continue
            if not self._angle_in_query(float(ang), angle_start, angle_extent):
                continue
            rc = float(row / factor0)
            cc = float(col / factor0)
            s_pre = self._score_pose_points(
                dist_list[max_lv],
                phi_list[max_lv],
                rc,
                cc,
                float(ang),
                float(sc),
                rel_xy=lvl.pre_xy,
                model_phi=lvl.pre_phi,
                max_dist=max_dist / factor0,
                max_ori_diff=max_ori_diff,
                reject_below=pre_gate,
            )
            if s_pre < pre_gate:
                continue
            s_full = self._score_pose_points(
                dist_list[max_lv],
                phi_list[max_lv],
                rc,
                cc,
                float(ang),
                float(sc),
                rel_xy=lvl.rel_xy,
                model_phi=lvl.phi,
                max_dist=max_dist / factor0,
                max_ori_diff=max_ori_diff,
                reject_below=full_gate,
            )
            if s_full >= full_gate:
                beam.append(Match(row=rc, col=cc, angle=float(ang), scale=float(sc), score=float(s_full + 1e-6 * vote)))

        if not beam:
            return []
        beam.sort(key=lambda m: m.score, reverse=True)
        beam = beam[: max(12, int(max_candidates * 0.4))]

        pos_offsets = [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
        for lv in range(max_lv - 1, -1, -1):
            idx_lv = min(lv, len(self._model_pyramid) - 1)
            lvl = self._model_pyramid[idx_lv]
            factor = float(2**lv)
            stage = float(max_lv - lv + 1) / float(max_lv + 1)
            da = float(angle_step) * (0.7 ** stage)
            ds = float(scale_step) * (0.7 ** stage)
            da_set = (0.0, da, -da)
            ds_set = (0.0, ds, -ds)
            pre_gate = max(0.025, float(min_score) * (0.35 + 0.35 * stage))
            full_gate = max(0.03, float(min_score) * (0.45 + 0.45 * stage))

            next_beam: list[Match] = []
            max_dist_lv = max_dist / factor
            for b in beam:
                base_r = b.row * 2.0
                base_c = b.col * 2.0
                for dr, dc in pos_offsets:
                    r2 = base_r + dr
                    c2 = base_c + dc
                    for d_a in da_set:
                        a2 = float(b.angle + d_a)
                        if not self._angle_in_query(a2, angle_start, angle_extent):
                            continue
                        for d_s in ds_set:
                            s2 = float(max(1e-4, b.scale + d_s))
                            if s2 < scale_min or s2 > scale_max:
                                continue
                            s_pre = self._score_pose_points(
                                dist_list[lv],
                                phi_list[lv],
                                r2,
                                c2,
                                a2,
                                s2,
                                rel_xy=lvl.pre_xy,
                                model_phi=lvl.pre_phi,
                                max_dist=max_dist_lv,
                                max_ori_diff=max_ori_diff,
                                reject_below=pre_gate,
                            )
                            if s_pre < pre_gate:
                                continue
                            s_full = self._score_pose_points(
                                dist_list[lv],
                                phi_list[lv],
                                r2,
                                c2,
                                a2,
                                s2,
                                rel_xy=lvl.rel_xy,
                                model_phi=lvl.phi,
                                max_dist=max_dist_lv,
                                max_ori_diff=max_ori_diff,
                                reject_below=full_gate,
                            )
                            if s_full >= full_gate:
                                next_beam.append(Match(row=r2, col=c2, angle=a2, scale=s2, score=float(s_full)))

            if not next_beam:
                return []
            next_beam.sort(key=lambda m: m.score, reverse=True)
            # Keep compact beam: this is the branch-and-bound style pruning stage.
            beam = next_beam[: max(10, int(max_candidates * (0.35 + 0.15 * (lv == 0))))]

        return beam

    def find(
        self,
        image: np.ndarray,
        *,
        angle_start: float = -0.39,
        angle_extent: float = 0.78,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        min_score: float = 0.5,
        num_matches: int = 1,
        max_overlap: float = 0.5,
        subpixel: SubPixelMode = "interpolation",
        num_levels: int = 0,
        greediness: float = 0.9,
        angle_step: Optional[float] = None,
        scale_step: Optional[float] = None,
        max_dist: float = 2.0,
        max_ori_diff: float = math.radians(25.0),
        top_k_per_pose: int = 10,
        max_candidates: int = 400,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if angle_extent < 0:
            raise ValueError("angle_extent must be >= 0")
        if scale_min <= 0:
            raise ValueError("scale_min must be > 0")
        if scale_max < scale_min:
            raise ValueError("scale_max must be >= scale_min")
        if not (0.0 <= min_score <= 1.0):
            raise ValueError("min_score must be within [0,1]")
        if num_matches < 0:
            raise ValueError("num_matches must be >= 0")
        if not (0.0 <= max_overlap <= 1.0):
            raise ValueError("max_overlap must be within [0,1]")
        if self.model_rel_xy.shape[0] == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        gray0 = _ensure_gray_u8(image)
        h0, w0 = gray0.shape[:2]

        if angle_step is None:
            if angle_extent <= 1e-9:
                angle_step = math.radians(2.0)
            else:
                angle_step = max(math.radians(1.5), float(angle_extent) / 90.0)
        if scale_step is None:
            rng = max(0.0, float(scale_max - scale_min))
            scale_step = 0.02 if rng <= 1e-9 else max(0.01, rng / 20.0)

        angle_step = float(max(1e-4, angle_step))
        scale_step = float(max(1e-4, scale_step))
        angle_step_deg = math.degrees(angle_step)

        if angle_extent <= 1e-9:
            angle_count = 1
        else:
            angle_count = int(math.floor(angle_extent / angle_step)) + 1
        if abs(scale_max - scale_min) <= 1e-12:
            scale_count = 1
        else:
            scale_count = int(math.floor((scale_max - scale_min) / scale_step)) + 1

        work_scale = self._choose_work_scale(
            h0,
            w0,
            num_levels=int(num_levels),
            greediness=float(np.clip(greediness, 0.0, 1.0)),
            angle_count=angle_count,
            scale_count=scale_count,
        )

        search_complexity = max(1, angle_count * scale_count)
        angle_step_ght = angle_step
        scale_step_ght = scale_step
        if search_complexity > 700:
            angle_step_ght = max(angle_step_ght, math.radians(4.0))
            scale_step_ght = max(scale_step_ght, 0.03)
        if search_complexity > 1400 or (h0 * w0 > 1_200_000 and search_complexity > 500):
            angle_step_ght = max(angle_step_ght, math.radians(5.0))
            scale_step_ght = max(scale_step_ght, 0.035)

        candidates = self._collect_candidates_ght(
            gray0,
            angle_start=angle_start,
            angle_extent=angle_extent,
            scale_min=scale_min,
            scale_max=scale_max,
            angle_step_deg=math.degrees(angle_step_ght),
            scale_step=scale_step_ght,
            greediness=float(np.clip(greediness, 0.0, 1.0)),
            work_scale=work_scale,
            max_candidates=max_candidates,
        )
        if not candidates:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        levels = int(num_levels) if int(num_levels) > 0 else _auto_pyramid_levels(h0, w0)
        levels = int(np.clip(levels, 1, max(1, min(5, len(self._model_pyramid)))))
        _pyr, dist_list, phi_list = self._build_image_feature_pyramid(gray0, levels)

        beam = self._beam_refine_candidates(
            candidates,
            dist_list=dist_list,
            phi_list=phi_list,
            levels=levels,
            angle_start=angle_start,
            angle_extent=angle_extent,
            scale_min=scale_min,
            scale_max=scale_max,
            angle_step=angle_step,
            scale_step=scale_step,
            min_score=min_score,
            max_dist=max_dist,
            max_ori_diff=max_ori_diff,
            max_candidates=max_candidates,
        )
        if not beam:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        beam.sort(key=lambda m: m.score, reverse=True)
        refine_limit = min(len(beam), max(12, int(max_candidates // 3), int(top_k_per_pose) * 4))
        dist0 = dist_list[0]
        phi0 = phi_list[0]

        scored: list[Match] = []
        half_da = 0.4 * angle_step
        half_ds = 0.4 * scale_step

        for m in beam[:refine_limit]:
            best = (float(m.row), float(m.col), float(m.angle), float(m.scale))
            best_score = self._score_pose(
                dist0,
                phi0,
                best[0],
                best[1],
                best[2],
                best[3],
                max_dist=max_dist,
                max_ori_diff=max_ori_diff,
            )

            for da in (0.0, half_da, -half_da):
                for ds in (0.0, half_ds, -half_ds):
                    if da == 0.0 and ds == 0.0:
                        continue
                    a2 = best[2] + da
                    if not self._angle_in_query(a2, angle_start, angle_extent):
                        continue
                    s2 = best[3] + ds
                    if s2 < scale_min or s2 > scale_max:
                        continue
                    s = self._score_pose(
                        dist0,
                        phi0,
                        best[0],
                        best[1],
                        a2,
                        s2,
                        max_dist=max_dist,
                        max_ori_diff=max_ori_diff,
                    )
                    if s > best_score:
                        best_score = s
                        best = (best[0], best[1], a2, s2)

            for dr, dc in ((0.0, 0.0), (-0.8, 0.0), (0.8, 0.0), (0.0, -0.8), (0.0, 0.8)):
                if dr == 0.0 and dc == 0.0:
                    continue
                r2 = best[0] + dr
                c2 = best[1] + dc
                s = self._score_pose(
                    dist0,
                    phi0,
                    r2,
                    c2,
                    best[2],
                    best[3],
                    max_dist=max_dist,
                    max_ori_diff=max_ori_diff,
                )
                if s > best_score:
                    best_score = s
                    best = (r2, c2, best[2], best[3])

            if best_score < float(min_score):
                continue

            if subpixel == "interpolation":
                r0, c0, a0, s0 = best
                f0 = best_score
                fl = self._score_pose(dist0, phi0, r0, c0 - 1.0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                fr = self._score_pose(dist0, phi0, r0, c0 + 1.0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                dx = _parabolic_subpixel(fl, f0, fr)
                fu = self._score_pose(dist0, phi0, r0 - 1.0, c0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                fd = self._score_pose(dist0, phi0, r0 + 1.0, c0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                dy = _parabolic_subpixel(fu, f0, fd)
                best = (r0 + dy, c0 + dx, a0, s0)

            scored.append(Match(row=best[0], col=best[1], angle=best[2], scale=best[3], score=float(best_score)))

        if not scored:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        scored.sort(key=lambda m: m.score, reverse=True)

        kept: list[Match] = []
        w0, h0 = self.base_size_wh
        for m in scored:
            rect_m = ((m.col, m.row), (w0 * m.scale, h0 * m.scale), math.degrees(m.angle))
            accept = True
            for k in kept:
                rect_k = ((k.col, k.row), (w0 * k.scale, h0 * k.scale), math.degrees(k.angle))
                if _rotated_rect_iou(rect_m, rect_k) > float(max_overlap):
                    accept = False
                    break
            if accept:
                kept.append(m)
            if num_matches > 0 and len(kept) >= int(num_matches):
                break

        rows = np.asarray([m.row for m in kept], dtype=np.float32)
        cols = np.asarray([m.col for m in kept], dtype=np.float32)
        angs = np.asarray([m.angle for m in kept], dtype=np.float32)
        scs = np.asarray([m.scale for m in kept], dtype=np.float32)
        scr = np.asarray([m.score for m in kept], dtype=np.float32)
        return rows, cols, angs, scs, scr
