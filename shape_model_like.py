from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

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


def _angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _quantize_angle_0_2pi(phi: np.ndarray, nbins: int) -> np.ndarray:
    phi_0 = np.mod(phi, 2 * np.pi)
    bins = np.floor(phi_0 * (nbins / (2 * np.pi))).astype(np.int32)
    return np.clip(bins, 0, nbins - 1).astype(np.int16)


def _undirected_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the smallest difference between angles assuming edge direction is undirected,
    i.e. angle and angle+pi are equivalent.
    Output range: [0, pi/2] (in practice [0, pi]).
    """
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
    phi = np.arctan2(dy, dx)  # [-pi, pi]
    phi = np.mod(phi, 2 * np.pi).astype(np.float32)  # [0, 2pi)
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


def _auto_pyramid_levels(h: int, w: int, min_size: int = 64, max_levels: int = 6) -> int:
    levels = 1
    while levels < max_levels and min(h, w) / (2 ** (levels - 1)) >= min_size * 2:
        levels += 1
    return levels


def _parabolic_subpixel(fm1: float, f0: float, fp1: float) -> float:
    denom = (fm1 - 2.0 * f0 + fp1)
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (fm1 - fp1) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def _rotated_rect_iou(rect1, rect2) -> float:
    # rect: ((cx, cy), (w, h), angle_deg)
    try:
        ret, inter = cv2.rotatedRectangleIntersection(rect1, rect2)
    except cv2.error:
        return 0.0
    if ret == 0 or inter is None:
        return 0.0
    inter = np.array(inter, dtype=np.float32)
    if inter.ndim != 3 or inter.shape[1:] != (1, 2):
        inter = inter.reshape(-1, 1, 2)
    area_inter = float(abs(cv2.contourArea(inter)))
    area1 = float(rect1[1][0] * rect1[1][1])
    area2 = float(rect2[1][0] * rect2[1][1])
    denom = area1 + area2 - area_inter
    if denom <= 0:
        return 0.0
    return float(np.clip(area_inter / denom, 0.0, 1.0))


@dataclass(frozen=True)
class Match:
    row: float
    col: float
    angle: float
    scale: float
    score: float


class ScaledShapeModel:
    """
    一个“类 HALCON find_scaled_shape_model”的核心版实现：
    - 旋转 + 等比例缩放 + 平移
    - Top-N
    - MaxOverlap 类似的重叠抑制（近似 IoU NMS）
    - SubPixel='interpolation' 的位置亚像素细化
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
    ) -> None:
        self.nbins = int(nbins)
        self.origin_rc = (float(origin_rc[0]), float(origin_rc[1]))  # (row, col)
        self.r_table = [np.asarray(v, dtype=np.float32) for v in r_table]  # each: (N,2) (dcol,drow)
        self.model_rel_xy = np.asarray(model_rel_xy, dtype=np.float32)  # (N,2) (dcol,drow) relative to origin
        self.model_phi = np.asarray(model_phi, dtype=np.float32)  # (N,) in [0,2pi)
        self.base_size_wh = (float(base_size_wh[0]), float(base_size_wh[1]))
        self.canny1 = int(canny1)
        self.canny2 = int(canny2)

        if len(self.r_table) != self.nbins:
            raise ValueError("r_table length must equal nbins")
        if self.model_rel_xy.ndim != 2 or self.model_rel_xy.shape[1] != 2:
            raise ValueError("model_rel_xy must be (N,2)")
        if self.model_phi.ndim != 1 or self.model_phi.shape[0] != self.model_rel_xy.shape[0]:
            raise ValueError("model_phi shape mismatch")

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
            edges = edges & (mask_u8.astype(bool))

        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            raise ValueError("No edges found in template (after mask). Try adjusting Canny thresholds.")

        if mask_u8 is not None:
            origin_r, origin_c = _centroid_from_mask(mask_u8)
        else:
            origin_r = float(np.mean(ys))
            origin_c = float(np.mean(xs))

        # Quantize orientations for edge pixels
        phi_bins = _quantize_angle_0_2pi(phi, nbins)

        # Collect model points and orientations: prefer strong edges for robustness
        rng = np.random.default_rng(rng_seed)
        edge_mag = mag[ys, xs].astype(np.float32)
        order = np.argsort(-edge_mag)
        if order.size > max_model_points:
            order = order[:max_model_points]
        # Randomly shuffle within the selected strong-edge set to avoid over-concentration
        if order.size > 0:
            order = order[rng.permutation(order.size)]

        sel_x = xs[order].astype(np.float32)
        sel_y = ys[order].astype(np.float32)
        sel_phi = phi[ys[order], xs[order]].astype(np.float32)

        # Model points relative to origin (dcol, drow)
        rel = np.stack([sel_x - origin_c, sel_y - origin_r], axis=1).astype(np.float32)

        # Build R-table: r = origin - p = (-rel)
        r_table: list[list[Tuple[float, float]]] = [[] for _ in range(nbins)]
        for x_i, y_i in zip(xs, ys):
            b = int(phi_bins[y_i, x_i])
            vec = (float(origin_c - x_i), float(origin_r - y_i))  # (dcol, drow)
            r_table[b].append(vec)
            # Edge direction is often ambiguous (phi and phi+pi). Register to opposite bin too.
            b_opp = (b + (nbins // 2)) % nbins
            if b_opp != b:
                r_table[b_opp].append(vec)

        # Downsample per bin to cap runtime
        r_table_np: list[np.ndarray] = []
        for b in range(nbins):
            vecs = np.array(r_table[b], dtype=np.float32)
            if vecs.size == 0:
                r_table_np.append(vecs.reshape(0, 2))
                continue
            if vecs.shape[0] > max_r_vectors_per_bin:
                take = rng.choice(vecs.shape[0], size=max_r_vectors_per_bin, replace=False)
                vecs = vecs[take]
            r_table_np.append(vecs)

        # Base size for NMS overlap (min area rect on edge points)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle_deg)
        (w, h) = rect[1]
        if w <= 1e-3 or h <= 1e-3:
            # fallback
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
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        # Save r_table as a single object array for simplicity
        r_obj = np.empty((self.nbins,), dtype=object)
        for i, v in enumerate(self.r_table):
            r_obj[i] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(
            path,
            nbins=np.int32(self.nbins),
            origin_rc=np.array(self.origin_rc, dtype=np.float32),
            r_table=r_obj,
            model_rel_xy=self.model_rel_xy.astype(np.float32),
            model_phi=self.model_phi.astype(np.float32),
            base_size_wh=np.array(self.base_size_wh, dtype=np.float32),
            canny1=np.int32(self.canny1),
            canny2=np.int32(self.canny2),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScaledShapeModel":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        nbins = int(data["nbins"])
        origin_rc = tuple(float(x) for x in data["origin_rc"].tolist())
        r_obj = data["r_table"]
        r_table = [np.asarray(r_obj[i], dtype=np.float32).reshape(-1, 2) for i in range(nbins)]
        model_rel_xy = np.asarray(data["model_rel_xy"], dtype=np.float32)
        model_phi = np.asarray(data["model_phi"], dtype=np.float32)
        base_size_wh = tuple(float(x) for x in data["base_size_wh"].tolist())
        canny1 = int(data["canny1"])
        canny2 = int(data["canny2"])
        return cls(
            nbins=nbins,
            origin_rc=origin_rc,
            r_table=r_table,
            model_rel_xy=model_rel_xy,
            model_phi=model_phi,
            base_size_wh=base_size_wh,
            canny1=canny1,
            canny2=canny2,
        )

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
        h, w = dist.shape[:2]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        rel = self.model_rel_xy  # (N,2) (dcol,drow)
        dcol = rel[:, 0]
        drow = rel[:, 1]
        col_f = col + scale * (cos_a * dcol - sin_a * drow)
        row_f = row + scale * (sin_a * dcol + cos_a * drow)

        col_i = np.rint(col_f).astype(np.int32)
        row_i = np.rint(row_f).astype(np.int32)

        inside = (row_i >= 0) & (row_i < h) & (col_i >= 0) & (col_i < w)
        if not np.any(inside):
            return 0.0

        # Distance-to-edge check
        d_ok = np.zeros_like(inside, dtype=bool)
        d_ok[inside] = dist[row_i[inside], col_i[inside]] <= max_dist

        # Orientation agreement check
        exp_phi = np.mod(self.model_phi + angle, 2 * np.pi)
        a_phi = np.zeros_like(exp_phi, dtype=np.float32)
        a_phi[inside] = phi_img[row_i[inside], col_i[inside]]
        diff = _undirected_angle_diff(a_phi, exp_phi)
        o_ok = diff <= max_ori_diff

        inliers = int(np.sum(d_ok & o_ok))
        total = int(self.model_rel_xy.shape[0])
        if total <= 0:
            return 0.0
        return float(inliers / total)

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

        gray0 = _ensure_gray_u8(image)
        h0, w0 = gray0.shape[:2]
        levels = int(num_levels) if int(num_levels) > 0 else _auto_pyramid_levels(h0, w0)

        pyr: list[np.ndarray] = [gray0]
        for _ in range(1, levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        gray_coarse = pyr[-1]

        # Prepare coarse edges and orientation bins
        edges_c, phi_c, _ = _edges_and_orientation(gray_coarse, self.canny1, self.canny2)
        phi_bins_c = _quantize_angle_0_2pi(phi_c, self.nbins)
        ys, xs = np.nonzero(edges_c)
        edge_pts = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)  # (N,2) (x,y)
        if edge_pts.shape[0] == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        # Sampling controlled by greediness (higher -> fewer points)
        greediness = float(np.clip(greediness, 0.0, 1.0))
        rng = np.random.default_rng(12345)
        sample_ratio = float(np.clip(1.0 - 0.7 * greediness, 0.1, 1.0))
        if sample_ratio < 1.0 and edge_pts.shape[0] > 2000:
            take_n = int(max(2000, sample_ratio * edge_pts.shape[0]))
            take = rng.choice(edge_pts.shape[0], size=take_n, replace=False)
            edge_pts = edge_pts[take]

        # Angle/scale sampling
        if angle_step is None:
            angle_step = 0.05  # ~2.9 deg
        if scale_step is None:
            scale_step = 0.02
        angle_end = angle_start + angle_extent
        if angle_extent == 0:
            angles = np.array([angle_start], dtype=np.float32)
        else:
            n = int(math.floor((angle_end - angle_start) / angle_step)) + 1
            angles = (angle_start + np.arange(n, dtype=np.float32) * angle_step).astype(np.float32)
        if scale_max == scale_min:
            scales = np.array([scale_min], dtype=np.float32)
        else:
            n = int(math.floor((scale_max - scale_min) / scale_step)) + 1
            scales = (scale_min + np.arange(n, dtype=np.float32) * scale_step).astype(np.float32)

        # Generate candidates on coarse level using voting
        hc, wc = gray_coarse.shape[:2]
        candidates: list[Tuple[int, int, float, float, int]] = []  # (row, col, angle, scale, votes)

        for angle in angles.tolist():
            cos_a = math.cos(float(angle))
            sin_a = math.sin(float(angle))
            # For a given rotation, an image edge with orientation phi_img corresponds to
            # a model edge with orientation phi_model ≈ phi_img - angle.
            bin_shift = int(round(float(angle) * (self.nbins / (2.0 * math.pi))))
            for scale in scales.tolist():
                acc = np.zeros((hc, wc), dtype=np.uint16)
                sc = float(scale)
                for x, y in edge_pts:
                    b_img = int(phi_bins_c[y, x])
                    b = (b_img - bin_shift) % self.nbins
                    vecs = self.r_table[b]
                    if vecs.size == 0:
                        continue
                    rc = vecs[:, 0]
                    rr = vecs[:, 1]
                    dc = sc * (cos_a * rc - sin_a * rr)
                    dr = sc * (sin_a * rc + cos_a * rr)
                    cc = np.rint(x + dc).astype(np.int32)
                    rr_i = np.rint(y + dr).astype(np.int32)
                    ok = (rr_i >= 0) & (rr_i < hc) & (cc >= 0) & (cc < wc)
                    if not np.any(ok):
                        continue
                    # Increment votes (small arrays; loop is acceptable)
                    for r_i, c_i in zip(rr_i[ok].tolist(), cc[ok].tolist()):
                        acc[r_i, c_i] += 1

                if top_k_per_pose <= 0:
                    continue
                k = int(min(top_k_per_pose, acc.size))
                flat = acc.reshape(-1)
                idx = np.argpartition(flat, -k)[-k:]
                # keep only meaningful peaks
                for ii in idx.tolist():
                    v = int(flat[ii])
                    if v <= 0:
                        continue
                    r = int(ii // wc)
                    c = int(ii % wc)
                    candidates.append((r, c, float(angle), float(scale), v))

        if not candidates:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        candidates.sort(key=lambda t: t[4], reverse=True)
        candidates = candidates[: int(max_candidates)]

        # Prepare full-res edge distance and orientation for scoring
        edges0, phi0, _ = _edges_and_orientation(gray0, self.canny1, self.canny2)
        inv = (~edges0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)

        factor = 2 ** (levels - 1)
        scored: list[Match] = []

        for r_c, c_c, ang, sc, _votes in candidates:
            row = float(r_c * factor)
            col = float(c_c * factor)

            # Local refine over neighboring angle/scale (very small budget)
            best = (row, col, float(ang), float(sc))
            best_score = self._score_pose(
                dist, phi0, best[0], best[1], best[2], best[3], max_dist=max_dist, max_ori_diff=max_ori_diff
            )

            # try 4 neighbors
            for dang in (0.0, float(angle_step), -float(angle_step)):
                for dsc in (0.0, float(scale_step), -float(scale_step)):
                    if dang == 0.0 and dsc == 0.0:
                        continue
                    a2 = float(ang + dang)
                    s2 = float(sc + dsc)
                    if s2 < scale_min or s2 > scale_max:
                        continue
                    s = self._score_pose(dist, phi0, row, col, a2, s2, max_dist=max_dist, max_ori_diff=max_ori_diff)
                    if s > best_score:
                        best_score = s
                        best = (row, col, a2, s2)

            if best_score < float(min_score):
                continue

            # Subpixel refinement on position (only)
            if subpixel == "interpolation":
                r0, c0, a0, s0 = best
                f0 = best_score
                # x direction
                fl = self._score_pose(dist, phi0, r0, c0 - 1.0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                fr = self._score_pose(dist, phi0, r0, c0 + 1.0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                dx = _parabolic_subpixel(fl, f0, fr)
                # y direction
                fu = self._score_pose(dist, phi0, r0 - 1.0, c0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
                fd = self._score_pose(dist, phi0, r0 + 1.0, c0, a0, s0, max_dist=max_dist, max_ori_diff=max_ori_diff)
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

        # NMS by rotated-rect IoU (approximate MaxOverlap)
        kept: list[Match] = []
        w_base, h_base = self.base_size_wh
        for m in scored:
            rect_m = ((m.col, m.row), (w_base * m.scale, h_base * m.scale), math.degrees(m.angle))
            ok = True
            for k in kept:
                rect_k = ((k.col, k.row), (w_base * k.scale, h_base * k.scale), math.degrees(k.angle))
                if _rotated_rect_iou(rect_m, rect_k) > float(max_overlap):
                    ok = False
                    break
            if ok:
                kept.append(m)
            if num_matches > 0 and len(kept) >= int(num_matches):
                break

        rows = np.array([m.row for m in kept], dtype=np.float32)
        cols = np.array([m.col for m in kept], dtype=np.float32)
        angs = np.array([m.angle for m in kept], dtype=np.float32)
        scs = np.array([m.scale for m in kept], dtype=np.float32)
        scr = np.array([m.score for m in kept], dtype=np.float32)
        return rows, cols, angs, scs, scr

