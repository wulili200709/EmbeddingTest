#!/usr/bin/env python3
"""
Robust template matcher for partial occlusion scenes.

Pipeline:
1) Candidate generation from ORB+RANSAC and part-based voting.
2) Robust scoring via truncated + trimmed chamfer distance.
3) Lightweight translation-only refinement.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class TemplateBlock:
    image: np.ndarray
    offset_x: int
    offset_y: int
    width: int
    height: int


@dataclass
class CandidatePose:
    matrix: np.ndarray  # 2x3, template -> scene
    source: str
    confidence: float


@dataclass
class MatchResult:
    matrix: np.ndarray
    score: float
    source: str
    trimmed_mean: float
    inlier_ratio: float
    orientation_error: float


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    two_pi = 2.0 * np.pi
    diff = np.mod(np.abs(a - b), two_pi)
    return np.minimum(diff, two_pi - diff)


def extract_edge_orientations(gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    orientations = np.arctan2(gy, gx)
    yy, xx = np.where(edges > 0)
    return orientations[yy, xx]


def compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    outer_3x3 = np.vstack([outer, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    inner_3x3 = np.vstack([inner, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    return (outer_3x3 @ inner_3x3)[:2, :].astype(np.float32)


class RobustTemplateMatcher:
    def __init__(
        self,
        occlusion_tolerance: float = 0.30,
        canny_low: int = 80,
        canny_high: int = 180,
        grid_rows: int = 3,
        grid_cols: int = 3,
        part_topk: int = 10,
        part_score_threshold: float = 0.35,
        vote_bin: int = 6,
        max_template_points: int = 2400,
        truncated_distance: float = 12.0,
        inlier_distance: float = 4.0,
        orientation_weight: float = 0.15,
        max_candidates: int = 120,
        max_candidates_pre_score: int = 2000,
        orb_features: int = 1200,
        rotation_step_deg: float = 0.0,
        post_spatial_radius: float = 24.0,
        post_angle_radius_deg: float = 14.0,
        spatial_cell_size: int = 100,
        max_per_spatial_cell: int = 10,
        quality_min_score: float = 0.0,
        quality_min_inlier_ratio: float = 0.0,
        quality_max_trimmed_mean: float = 1e9,
        instance_cluster_radius: float = 52.0,
    ) -> None:
        if not (0.0 <= occlusion_tolerance < 1.0):
            raise ValueError("occlusion_tolerance must be in [0, 1).")
        self.keep_ratio = 1.0 - occlusion_tolerance
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.part_topk = part_topk
        self.part_score_threshold = part_score_threshold
        self.vote_bin = vote_bin
        self.max_template_points = max_template_points
        self.truncated_distance = truncated_distance
        self.inlier_distance = inlier_distance
        self.orientation_weight = orientation_weight
        self.max_candidates = max_candidates
        self.max_candidates_pre_score = max_candidates_pre_score
        self.rotation_step_deg = rotation_step_deg
        self.post_spatial_radius = post_spatial_radius
        self.post_angle_radius_deg = post_angle_radius_deg
        self.spatial_cell_size = max(1, spatial_cell_size)
        self.max_per_spatial_cell = max(1, max_per_spatial_cell)
        self.quality_min_score = quality_min_score
        self.quality_min_inlier_ratio = quality_min_inlier_ratio
        self.quality_max_trimmed_mean = quality_max_trimmed_mean
        self.instance_cluster_radius = max(1.0, instance_cluster_radius)
        self.orb = cv2.ORB_create(nfeatures=orb_features)

        self.template_gray: Optional[np.ndarray] = None
        self.template_edges: Optional[np.ndarray] = None
        self.template_shape: Optional[Tuple[int, int]] = None  # h, w
        self.template_points: Optional[np.ndarray] = None  # Nx2 (x, y), float32
        self.template_orientations: Optional[np.ndarray] = None  # N, radians
        self.template_kps: Optional[Sequence[cv2.KeyPoint]] = None
        self.template_desc: Optional[np.ndarray] = None
        self.blocks: List[TemplateBlock] = []

    def build_template(self, template_image: np.ndarray) -> None:
        gray = to_gray(template_image)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        yy, xx = np.where(edges > 0)
        if len(xx) == 0:
            raise RuntimeError("Template has no edge points after Canny.")

        pts = np.column_stack([xx, yy]).astype(np.float32)
        step = 1
        if pts.shape[0] > self.max_template_points:
            step = max(1, pts.shape[0] // self.max_template_points)
            pts = pts[::step]

        # Re-index orientations with the same down-sampling step.
        full_orients = extract_edge_orientations(gray, edges)
        if pts.shape[0] < full_orients.shape[0]:
            full_orients = full_orients[::step]
            full_orients = full_orients[: pts.shape[0]]

        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None:
            desc = np.empty((0, 32), dtype=np.uint8)
            kps = []

        self.template_gray = gray
        self.template_edges = edges
        self.template_shape = gray.shape[:2]
        self.template_points = pts
        self.template_orientations = full_orients.astype(np.float32)
        self.template_kps = kps
        self.template_desc = desc
        self.blocks = self._build_blocks(edges)

    def match(
        self,
        scene_image: np.ndarray,
        refine_radius: int = 2,
        max_results: int = 3,
    ) -> List[MatchResult]:
        self._ensure_template()

        scene_gray = to_gray(scene_image)
        scene_edges = cv2.Canny(scene_gray, self.canny_low, self.canny_high)
        scene_dist = cv2.distanceTransform(255 - scene_edges, cv2.DIST_L2, 3)
        scene_orient = self._compute_scene_orientation(scene_gray)

        candidates = []
        candidates.extend(self._candidates_from_orb(scene_gray))
        candidates.extend(self._candidates_from_part_voting(scene_edges))
        candidates = self._expand_candidates_with_rotation(candidates)
        candidates = self._deduplicate_candidates(candidates)

        if not candidates:
            return []

        scored: List[MatchResult] = []
        for cand in candidates:
            matrix = cand.matrix.copy()
            if refine_radius > 0:
                matrix = self._refine_xy(matrix, scene_dist, scene_orient, refine_radius)

            result = self._score_pose(
                matrix,
                scene_dist=scene_dist,
                scene_orient=scene_orient,
                source=cand.source,
            )
            scored.append(result)

        scored.sort(key=lambda x: x.score, reverse=True)
        scored = self._deduplicate_scored_results(scored)
        filtered = self._filter_scored_results(scored)
        if not filtered:
            filtered = scored
        clustered = self._cluster_round_robin(filtered)
        return self._select_diverse_results(clustered, max_results=max_results)

    def _ensure_template(self) -> None:
        if self.template_points is None:
            raise RuntimeError("Call build_template(...) first.")

    def _build_blocks(self, edge_image: np.ndarray) -> List[TemplateBlock]:
        h, w = edge_image.shape[:2]
        blocks: List[TemplateBlock] = []
        row_edges = np.linspace(0, h, self.grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, w, self.grid_cols + 1, dtype=int)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y0, y1 = row_edges[r], row_edges[r + 1]
                x0, x1 = col_edges[c], col_edges[c + 1]
                patch = edge_image[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                edge_ratio = float(np.count_nonzero(patch)) / float(patch.size)
                if edge_ratio < 0.02:
                    continue
                blocks.append(
                    TemplateBlock(
                        image=patch,
                        offset_x=x0,
                        offset_y=y0,
                        width=x1 - x0,
                        height=y1 - y0,
                    )
                )
        return blocks

    def _compute_scene_orientation(self, gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.arctan2(gy, gx)

    def _candidates_from_orb(self, scene_gray: np.ndarray) -> List[CandidatePose]:
        assert self.template_kps is not None
        assert self.template_desc is not None
        if self.template_desc.shape[0] < 8:
            return []

        scene_kps, scene_desc = self.orb.detectAndCompute(scene_gray, None)
        if scene_desc is None or len(scene_kps) < 8:
            return []

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(self.template_desc, scene_desc, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < 0.78 * n.distance:
                good.append(m)

        if len(good) < 8:
            return []

        src = np.float32([self.template_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([scene_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=3000,
            confidence=0.995,
        )
        if matrix is None or inliers is None:
            return []

        inlier_ratio = float(np.count_nonzero(inliers)) / float(len(good))
        if inlier_ratio < 0.25:
            return []

        return [CandidatePose(matrix=matrix.astype(np.float32), source="orb_ransac", confidence=inlier_ratio)]

    def _candidates_from_part_voting(self, scene_edges: np.ndarray) -> List[CandidatePose]:
        if not self.blocks:
            return []

        votes: dict[Tuple[int, int], List[float]] = {}
        scene_float = scene_edges.astype(np.float32) / 255.0

        for block in self.blocks:
            block_img = block.image.astype(np.float32) / 255.0
            sh, sw = scene_float.shape[:2]
            if block.height > sh or block.width > sw:
                continue
            score_map = cv2.matchTemplate(scene_float, block_img, cv2.TM_CCOEFF_NORMED)
            peaks = self._topk_peaks(score_map, self.part_topk, self.part_score_threshold)
            for px, py, val in peaks:
                tx = px - block.offset_x
                ty = py - block.offset_y
                bx = int(round(tx / self.vote_bin))
                by = int(round(ty / self.vote_bin))
                votes.setdefault((bx, by), []).append(float(val))

        if not votes:
            return []

        ranked = sorted(
            votes.items(),
            key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))),
            reverse=True,
        )

        candidates: List[CandidatePose] = []
        total_blocks = max(1, len(self.blocks))
        max_keep = min(self.max_candidates, max(4, len(ranked)))
        for (bx, by), block_scores in ranked[:max_keep]:
            tx = bx * self.vote_bin
            ty = by * self.vote_bin
            matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
            confidence = float(len(block_scores)) / float(total_blocks)
            candidates.append(
                CandidatePose(
                    matrix=matrix,
                    source="part_vote",
                    confidence=confidence,
                )
            )
        return candidates

    def _topk_peaks(self, score_map: np.ndarray, k: int, min_score: float) -> List[Tuple[int, int, float]]:
        if score_map.size == 0 or k <= 0:
            return []

        flat = score_map.ravel()
        probe_k = min(k * 10, flat.shape[0])
        if probe_k <= 0:
            return []

        indices = np.argpartition(flat, -probe_k)[-probe_k:]
        indices = indices[np.argsort(flat[indices])[::-1]]

        peaks: List[Tuple[int, int, float]] = []
        h, w = score_map.shape[:2]
        min_dist = max(4, min(h, w) // 30)
        for idx in indices:
            y, x = divmod(int(idx), w)
            val = float(score_map[y, x])
            if val < min_score:
                continue
            if peaks:
                ok = True
                for px, py, _ in peaks:
                    if (px - x) * (px - x) + (py - y) * (py - y) < min_dist * min_dist:
                        ok = False
                        break
                if not ok:
                    continue
            peaks.append((x, y, val))
            if len(peaks) >= k:
                break
        return peaks

    def _deduplicate_candidates(self, candidates: List[CandidatePose]) -> List[CandidatePose]:
        if not candidates:
            return []

        best_by_sig: dict[Tuple[int, int, int, int], CandidatePose] = {}
        for cand in candidates:
            sig = self._matrix_signature(cand.matrix, t_bin=4.0, s_bin=0.08, a_bin=6.0)
            cur = best_by_sig.get(sig)
            if cur is None or cand.confidence > cur.confidence:
                best_by_sig[sig] = cand

        unique = list(best_by_sig.values())
        unique.sort(key=lambda c: c.confidence, reverse=True)
        return unique[: self.max_candidates_pre_score]

    def _matrix_signature(
        self,
        matrix: np.ndarray,
        t_bin: float,
        s_bin: float,
        a_bin: float,
    ) -> Tuple[int, int, int, int]:
        tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
        sx = math.sqrt(float(matrix[0, 0]) ** 2 + float(matrix[1, 0]) ** 2)
        angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        return (
            int(round(tx / t_bin)),
            int(round(ty / t_bin)),
            int(round((sx - 1.0) / s_bin)),
            int(round(angle / a_bin)),
        )

    def _deduplicate_scored_results(self, ranked: Sequence[MatchResult]) -> List[MatchResult]:
        if not ranked:
            return []
        best_by_sig: Dict[Tuple[int, int, int, int], MatchResult] = {}
        for res in ranked:
            sig = self._matrix_signature(res.matrix, t_bin=3.0, s_bin=0.04, a_bin=4.0)
            cur = best_by_sig.get(sig)
            if cur is None or res.score > cur.score:
                best_by_sig[sig] = res
        out = list(best_by_sig.values())
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    def _expand_candidates_with_rotation(self, candidates: List[CandidatePose]) -> List[CandidatePose]:
        if not candidates or self.rotation_step_deg <= 0.0:
            return candidates
        if self.template_shape is None:
            return candidates

        h, w = self.template_shape
        center = ((w - 1.0) * 0.5, (h - 1.0) * 0.5)
        angles = np.arange(0.0, 360.0, self.rotation_step_deg, dtype=np.float32)
        if angles.size <= 1:
            return candidates

        expanded: List[CandidatePose] = []
        for cand in candidates:
            expanded.append(cand)
            for deg in angles:
                if abs(float(deg)) < 1e-6:
                    continue
                rot = cv2.getRotationMatrix2D(center, float(deg), 1.0).astype(np.float32)
                mat = compose_affine(cand.matrix, rot)
                expanded.append(
                    CandidatePose(
                        matrix=mat.astype(np.float32),
                        source=f"{cand.source}_rot",
                        confidence=cand.confidence,
                    )
                )
        return expanded

    def _pose_center_and_angle_deg(self, matrix: np.ndarray) -> Tuple[float, float, float]:
        if self.template_shape is None:
            return float(matrix[0, 2]), float(matrix[1, 2]), 0.0
        h, w = self.template_shape
        corners = np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.transform(corners, matrix).reshape(-1, 2)
        cx = float(np.mean(warped[:, 0]))
        cy = float(np.mean(warped[:, 1]))
        angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        if angle < 0.0:
            angle += 360.0
        return cx, cy, angle

    def _filter_scored_results(self, ranked: Sequence[MatchResult]) -> List[MatchResult]:
        filtered: List[MatchResult] = []
        for res in ranked:
            if res.score < self.quality_min_score:
                continue
            if res.inlier_ratio < self.quality_min_inlier_ratio:
                continue
            if res.trimmed_mean > self.quality_max_trimmed_mean:
                continue
            filtered.append(res)
        return filtered

    def _cluster_results_by_center(self, ranked: Sequence[MatchResult]) -> List[List[MatchResult]]:
        clusters: List[Dict[str, object]] = []
        radius_sq = self.instance_cluster_radius * self.instance_cluster_radius

        for res in ranked:
            cx, cy, _ = self._pose_center_and_angle_deg(res.matrix)
            best_idx = -1
            best_dist = float("inf")

            for idx, cluster in enumerate(clusters):
                center = cluster["center"]
                if not isinstance(center, tuple):
                    continue
                dx = cx - float(center[0])
                dy = cy - float(center[1])
                dist = dx * dx + dy * dy
                if dist <= radius_sq and dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx < 0:
                clusters.append(
                    {
                        "center": (cx, cy),
                        "members": [res],
                    }
                )
            else:
                cluster = clusters[best_idx]
                members = cluster["members"]
                if isinstance(members, list):
                    members.append(res)
                    m = len(members)
                    old_center = cluster["center"]
                    if isinstance(old_center, tuple):
                        ncx = (float(old_center[0]) * (m - 1) + cx) / float(m)
                        ncy = (float(old_center[1]) * (m - 1) + cy) / float(m)
                        cluster["center"] = (ncx, ncy)

        out: List[List[MatchResult]] = []
        for cluster in clusters:
            members = cluster["members"]
            if isinstance(members, list) and members:
                members.sort(key=lambda r: r.score, reverse=True)
                out.append(members)

        out.sort(key=lambda c: c[0].score if c else -1.0, reverse=True)
        return out

    def _cluster_round_robin(self, ranked: Sequence[MatchResult]) -> List[MatchResult]:
        if not ranked:
            return []
        clusters = self._cluster_results_by_center(ranked)
        if not clusters:
            return list(ranked)

        ordered: List[MatchResult] = []
        layer = 0
        while True:
            added = False
            for cluster in clusters:
                if layer < len(cluster):
                    ordered.append(cluster[layer])
                    added = True
            if not added:
                break
            layer += 1
        return ordered

    def _select_diverse_results(self, ranked: Sequence[MatchResult], max_results: int) -> List[MatchResult]:
        if max_results <= 0 or not ranked:
            return []

        selected: List[MatchResult] = []
        selected_meta: List[Tuple[float, float, float]] = []
        selected_indices: set[int] = set()
        cell_counts: Dict[Tuple[int, int], int] = {}
        radius_sq = self.post_spatial_radius * self.post_spatial_radius

        # First pass: enforce both local NMS and per-cell quota.
        for idx, res in enumerate(ranked):
            cx, cy, ang = self._pose_center_and_angle_deg(res.matrix)
            cell = (int(math.floor(cx / self.spatial_cell_size)), int(math.floor(cy / self.spatial_cell_size)))
            if cell_counts.get(cell, 0) >= self.max_per_spatial_cell:
                continue

            too_close = False
            for sx, sy, sa in selected_meta:
                dx = cx - sx
                dy = cy - sy
                if dx * dx + dy * dy <= radius_sq:
                    da = abs(ang - sa)
                    da = min(da, 360.0 - da)
                    if da < self.post_angle_radius_deg:
                        too_close = True
                        break
            if too_close:
                continue

            selected.append(res)
            selected_meta.append((cx, cy, ang))
            selected_indices.add(idx)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if len(selected) >= max_results:
                selected.sort(key=lambda r: r.score, reverse=True)
                return selected[:max_results]

        # Second pass: if still not enough, relax local NMS but keep a softer per-cell quota.
        relaxed_quota = self.max_per_spatial_cell * 3
        relaxed_radius_sq = (self.post_spatial_radius * 0.65) * (self.post_spatial_radius * 0.65)
        relaxed_angle = self.post_angle_radius_deg * 0.65
        for idx, res in enumerate(ranked):
            if idx in selected_indices:
                continue
            cx, cy, ang = self._pose_center_and_angle_deg(res.matrix)
            cell = (int(math.floor(cx / self.spatial_cell_size)), int(math.floor(cy / self.spatial_cell_size)))
            if cell_counts.get(cell, 0) >= relaxed_quota:
                continue

            too_close = False
            for sx, sy, sa in selected_meta:
                dx = cx - sx
                dy = cy - sy
                if dx * dx + dy * dy <= relaxed_radius_sq:
                    da = abs(ang - sa)
                    da = min(da, 360.0 - da)
                    if da < relaxed_angle:
                        too_close = True
                        break
            if too_close:
                continue

            selected.append(res)
            selected_indices.add(idx)
            selected_meta.append((cx, cy, ang))
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if len(selected) >= max_results:
                break

        selected.sort(key=lambda r: r.score, reverse=True)
        return selected[:max_results]

    def _refine_xy(
        self,
        matrix: np.ndarray,
        scene_dist: np.ndarray,
        scene_orient: np.ndarray,
        radius: int,
    ) -> np.ndarray:
        best_matrix = matrix.copy()
        best_score = -1.0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                test = matrix.copy()
                test[0, 2] += dx
                test[1, 2] += dy
                result = self._score_pose(test, scene_dist, scene_orient, source="refine")
                if result.score > best_score:
                    best_score = result.score
                    best_matrix = test
        return best_matrix

    def _score_pose(
        self,
        matrix: np.ndarray,
        scene_dist: np.ndarray,
        scene_orient: np.ndarray,
        source: str,
    ) -> MatchResult:
        assert self.template_points is not None
        assert self.template_orientations is not None

        pts = cv2.transform(self.template_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        dists, orient_err = self._sample_scene_metrics(
            pts,
            scene_dist=scene_dist,
            scene_orient=scene_orient,
            matrix=matrix,
        )

        dists = np.minimum(dists, self.truncated_distance)
        keep_n = max(1, min(dists.shape[0], int(round(self.keep_ratio * dists.shape[0]))))
        trimmed = np.partition(dists, keep_n - 1)[:keep_n]
        trimmed_mean = float(np.mean(trimmed))
        inlier_ratio = float(np.mean(dists < self.inlier_distance))

        if orient_err.size > 0:
            keep_orient_n = min(keep_n, orient_err.shape[0])
            orient_trim = np.partition(orient_err, keep_orient_n - 1)[:keep_orient_n]
            orientation_error = float(np.mean(orient_trim))
        else:
            orientation_error = float(np.pi)

        dist_score = 1.0 - min(1.0, trimmed_mean / self.truncated_distance)
        orient_score = 1.0 - min(1.0, orientation_error / np.pi)
        score = 0.55 * dist_score + 0.35 * inlier_ratio + self.orientation_weight * orient_score
        score = float(np.clip(score, 0.0, 1.0))

        return MatchResult(
            matrix=matrix.astype(np.float32),
            score=score,
            source=source,
            trimmed_mean=trimmed_mean,
            inlier_ratio=inlier_ratio,
            orientation_error=orientation_error,
        )

    def _sample_scene_metrics(
        self,
        points_xy: np.ndarray,
        scene_dist: np.ndarray,
        scene_orient: np.ndarray,
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.template_orientations is not None
        h, w = scene_dist.shape[:2]
        x = np.rint(points_xy[:, 0]).astype(np.int32)
        y = np.rint(points_xy[:, 1]).astype(np.int32)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

        dists = np.full((points_xy.shape[0],), self.truncated_distance, dtype=np.float32)
        if np.any(valid):
            xv = x[valid]
            yv = y[valid]
            dists[valid] = scene_dist[yv, xv]

        # Estimate rotation from affine matrix and align template edge orientations.
        rotation = float(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        warped_template_orient = self.template_orientations + rotation

        orient_errors = np.empty((0,), dtype=np.float32)
        if np.any(valid):
            scene_vals = scene_orient[y[valid], x[valid]]
            template_vals = warped_template_orient[valid]
            orient_errors = angle_diff_rad(scene_vals, template_vals).astype(np.float32)

        return dists, orient_errors


def _matrix_to_warped_corners(template_shape: Tuple[int, int], matrix: np.ndarray) -> np.ndarray:
    h, w = template_shape
    corners = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.transform(corners, matrix).reshape(-1, 2).astype(np.int32)


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def select_results_for_drawing(
    results: Sequence[MatchResult],
    template_shape: Tuple[int, int],
    max_draw: int,
    iou_threshold: float,
    center_threshold: float,
    angle_threshold_deg: float = 35.0,
) -> List[Tuple[MatchResult, np.ndarray]]:
    if max_draw <= 0:
        max_draw = len(results)

    kept: List[Tuple[MatchResult, np.ndarray]] = []
    meta: List[Tuple[Tuple[float, float, float, float], Tuple[float, float], float]] = []
    center_threshold_sq = center_threshold * center_threshold

    for res in results:
        warped = _matrix_to_warped_corners(template_shape, res.matrix)
        x1 = float(np.min(warped[:, 0]))
        y1 = float(np.min(warped[:, 1]))
        x2 = float(np.max(warped[:, 0]))
        y2 = float(np.max(warped[:, 1]))
        center = (float(np.mean(warped[:, 0])), float(np.mean(warped[:, 1])))
        angle = math.degrees(math.atan2(float(res.matrix[1, 0]), float(res.matrix[0, 0])))
        if angle < 0.0:
            angle += 360.0

        drop = False
        for (kb, kc, ka) in meta:
            iou = _bbox_iou((x1, y1, x2, y2), kb)
            dx = center[0] - kc[0]
            dy = center[1] - kc[1]
            da = abs(angle - ka)
            da = min(da, 360.0 - da)
            if iou >= iou_threshold or (dx * dx + dy * dy <= center_threshold_sq and da <= angle_threshold_deg):
                drop = True
                break
        if drop:
            continue

        kept.append((res, warped))
        meta.append(((x1, y1, x2, y2), center, angle))
        if len(kept) >= max_draw:
            break
    return kept


def draw_matches_overlay(
    scene_bgr: np.ndarray,
    template_shape: Tuple[int, int],
    results: Sequence[MatchResult],
    draw_top: int = 20,
    legend_top: int = 20,
    draw_nms_iou: float = 0.40,
    draw_nms_center: float = 18.0,
) -> np.ndarray:
    out = scene_bgr.copy()
    if not results:
        return out

    palette = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 80, 0),
        (255, 0, 255),
        (255, 255, 0),
    ]

    draw_items = select_results_for_drawing(
        results=results,
        template_shape=template_shape,
        max_draw=draw_top,
        iou_threshold=draw_nms_iou,
        center_threshold=draw_nms_center,
    )
    draw_count = len(draw_items)
    for idx, item in enumerate(draw_items, start=1):
        res, warped = item
        color = palette[(idx - 1) % len(palette)]
        cv2.polylines(out, [warped], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        center = np.mean(warped, axis=0)
        label = f"#{idx}"
        cv2.putText(
            out,
            label,
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    list_count = min(len(results), max(1, legend_top))
    header = f"Top {len(results)} matches (draw {draw_count})"
    lines = [header]
    for idx, res in enumerate(results[:list_count], start=1):
        angle_deg = math.degrees(math.atan2(float(res.matrix[1, 0]), float(res.matrix[0, 0])))
        lines.append(f"#{idx:02d} s={res.score:.3f} a={angle_deg:6.1f} {res.source}")

    if list_count < len(results):
        lines.append(f"... {len(results) - list_count} more")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.44
    thickness = 1
    pad = 8
    line_h = 15
    panel_h = pad * 2 + line_h * len(lines)
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)
    panel_w = max_w + pad * 2

    overlay = out.copy()
    cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    y = 6 + pad + 11
    for i, line in enumerate(lines):
        color = (20, 230, 20) if i == 0 else (240, 240, 240)
        cv2.putText(out, line, (6 + pad, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_h
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust template matching with occlusion tolerance.")
    parser.add_argument("--template", required=True, help="Path to template image.")
    parser.add_argument("--scene", required=True, help="Path to scene image.")
    parser.add_argument("--out", default="match_result.png", help="Path to output visualization image.")
    parser.add_argument(
        "--occlusion",
        type=float,
        default=0.30,
        help="Allowed occlusion ratio in [0, 1). Default: 0.30",
    )
    parser.add_argument("--max-results", type=int, default=3, help="Number of top matches to print.")
    parser.add_argument("--no-refine", action="store_true", help="Disable XY lightweight refinement.")
    parser.add_argument("--grid-rows", type=int, default=3, help="Template block rows for part voting.")
    parser.add_argument("--grid-cols", type=int, default=3, help="Template block cols for part voting.")
    parser.add_argument("--part-topk", type=int, default=10, help="Top-K local hits per block.")
    parser.add_argument(
        "--part-score-thresh",
        type=float,
        default=0.35,
        help="Minimum normalized score for block-level local peaks.",
    )
    parser.add_argument(
        "--part-vote-max",
        type=int,
        default=120,
        help="Maximum translation vote hypotheses kept before scoring.",
    )
    parser.add_argument("--vote-bin", type=int, default=6, help="Translation vote quantization size.")
    parser.add_argument("--trunc-dist", type=float, default=12.0, help="Chamfer distance truncation.")
    parser.add_argument("--inlier-dist", type=float, default=4.0, help="Distance threshold for inlier ratio.")
    parser.add_argument("--canny-low", type=int, default=80, help="Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=180, help="Canny high threshold.")
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=0.0,
        help="Rotation sweep step in degrees for candidate expansion. 0 disables sweep.",
    )
    parser.add_argument(
        "--spatial-nms-radius",
        type=float,
        default=24.0,
        help="Post-score center-distance NMS radius in pixels.",
    )
    parser.add_argument(
        "--angle-nms-deg",
        type=float,
        default=14.0,
        help="Post-score angle NMS threshold in degrees.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=100,
        help="Spatial cell size for diversity quota.",
    )
    parser.add_argument(
        "--max-per-cell",
        type=int,
        default=10,
        help="Max kept matches per spatial cell in first pass.",
    )
    parser.add_argument(
        "--quality-min-score",
        type=float,
        default=0.0,
        help="Drop matches with score lower than this value.",
    )
    parser.add_argument(
        "--quality-min-inlier",
        type=float,
        default=0.0,
        help="Drop matches with inlier ratio lower than this value.",
    )
    parser.add_argument(
        "--quality-max-trimmed",
        type=float,
        default=1e9,
        help="Drop matches with trimmed mean larger than this value.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=52.0,
        help="Spatial radius (pixels) for instance cluster round-robin ordering.",
    )
    parser.add_argument(
        "--draw-top",
        type=int,
        default=20,
        help="How many top matches to draw as boxes. <=0 draws all.",
    )
    parser.add_argument(
        "--legend-top",
        type=int,
        default=25,
        help="How many top matches to list in text panel.",
    )
    parser.add_argument(
        "--draw-nms-iou",
        type=float,
        default=0.40,
        help="IoU threshold used by visualization-only box dedup.",
    )
    parser.add_argument(
        "--draw-nms-center",
        type=float,
        default=18.0,
        help="Center-distance threshold used by visualization-only box dedup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    template = cv2.imread(args.template, cv2.IMREAD_COLOR)
    scene = cv2.imread(args.scene, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Failed to read template image: {args.template}")
    if scene is None:
        raise FileNotFoundError(f"Failed to read scene image: {args.scene}")

    matcher = RobustTemplateMatcher(
        occlusion_tolerance=args.occlusion,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        part_topk=args.part_topk,
        part_score_threshold=args.part_score_thresh,
        vote_bin=args.vote_bin,
        max_candidates=args.part_vote_max,
        truncated_distance=args.trunc_dist,
        inlier_distance=args.inlier_dist,
        rotation_step_deg=args.rotation_step,
        post_spatial_radius=args.spatial_nms_radius,
        post_angle_radius_deg=args.angle_nms_deg,
        spatial_cell_size=args.cell_size,
        max_per_spatial_cell=args.max_per_cell,
        quality_min_score=args.quality_min_score,
        quality_min_inlier_ratio=args.quality_min_inlier,
        quality_max_trimmed_mean=args.quality_max_trimmed,
        instance_cluster_radius=args.cluster_radius,
    )
    matcher.build_template(template)

    refine_radius = 0 if args.no_refine else 2
    results = matcher.match(scene, refine_radius=refine_radius, max_results=args.max_results)
    if not results:
        print("No valid candidate pose found.")
        return 2

    for idx, res in enumerate(results, start=1):
        m = res.matrix
        print(
            f"[{idx}] score={res.score:.4f}, source={res.source}, "
            f"trimmed_mean={res.trimmed_mean:.3f}, inlier_ratio={res.inlier_ratio:.3f}, "
            f"orientation_error={math.degrees(res.orientation_error):.2f}deg, "
            f"matrix=[[{m[0,0]:.4f},{m[0,1]:.4f},{m[0,2]:.2f}],[{m[1,0]:.4f},{m[1,1]:.4f},{m[1,2]:.2f}]]"
        )

    overlay = draw_matches_overlay(
        scene_bgr=scene,
        template_shape=matcher.template_shape,  # type: ignore[arg-type]
        results=results,
        draw_top=args.draw_top,
        legend_top=args.legend_top,
        draw_nms_iou=args.draw_nms_iou,
        draw_nms_center=args.draw_nms_center,
    )
    ok = cv2.imwrite(args.out, overlay)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.out}")
    print(f"Saved top-{len(results)} match visualization to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
