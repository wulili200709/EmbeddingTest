#!/usr/bin/env python3
"""
Batch-run robust template matching on test cases, using case-specific YAML infos.

Outputs are written to each case under:
  test/<case>/result_robust/
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.template_matching.robust_template_match import (
    MatchResult,
    RobustTemplateMatcher,
    select_results_for_drawing,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


Image = np.ndarray


@dataclass
class SceneRunResult:
    scene: str
    output_image: str
    output_yaml: str
    success: bool
    message: str
    best_score: Optional[float] = None
    best_scale: Optional[float] = None


@dataclass
class Case0MatchEntry:
    result: MatchResult
    scale: float
    template_bgr: Image
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    angle_deg: float


def load_infos(info_yaml: Path) -> List[Tuple[float, float]]:
    fs = cv2.FileStorage(str(info_yaml), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Failed to open info yaml: {info_yaml}")
    node = fs.getNode("infos")
    if node.empty() or not node.isSeq():
        fs.release()
        return []
    out: List[Tuple[float, float]] = []
    for i in range(node.size()):
        item = node.at(i)
        angle = float(item.getNode("angle").real())
        scale = float(item.getNode("scale").real())
        out.append((angle, scale))
    fs.release()
    return out


def unique_sorted(values: Sequence[float], decimals: int = 6) -> List[float]:
    return sorted({round(float(v), decimals) for v in values})


def infer_angle_step_deg(infos: Sequence[Tuple[float, float]]) -> float:
    angles = unique_sorted([x[0] for x in infos], decimals=6)
    if len(angles) < 2:
        return 0.0
    diffs: List[float] = []
    for i in range(1, len(angles)):
        d = angles[i] - angles[i - 1]
        if d > 1e-6:
            diffs.append(d)
    if not diffs:
        return 0.0
    return max(0.0, min(diffs))


def read_image(path: Path) -> Image:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_template(base: Image, scale: float) -> Image:
    h, w = base.shape[:2]
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(base, (nw, nh), interpolation=interp)


def build_matcher(rotation_step_deg: float, occlusion: float, max_candidates: int) -> RobustTemplateMatcher:
    return RobustTemplateMatcher(
        occlusion_tolerance=occlusion,
        rotation_step_deg=rotation_step_deg,
        max_candidates=max_candidates,
    )


def match_once(
    template: Image,
    scene: Image,
    rotation_step_deg: float,
    occlusion: float,
    max_results: int,
    max_candidates: int,
) -> List[MatchResult]:
    matcher = build_matcher(
        rotation_step_deg=rotation_step_deg,
        occlusion=occlusion,
        max_candidates=max_candidates,
    )
    matcher.build_template(template)
    return matcher.match(scene, max_results=max_results)


def result_to_dict(res: MatchResult) -> Dict[str, Any]:
    angle_deg = math.degrees(math.atan2(float(res.matrix[1, 0]), float(res.matrix[0, 0])))
    return {
        "score": float(res.score),
        "source": res.source,
        "trimmed_mean": float(res.trimmed_mean),
        "inlier_ratio": float(res.inlier_ratio),
        "orientation_error_deg": float(math.degrees(res.orientation_error)),
        "angle_deg": float(angle_deg),
        "matrix": [
            [float(res.matrix[0, 0]), float(res.matrix[0, 1]), float(res.matrix[0, 2])],
            [float(res.matrix[1, 0]), float(res.matrix[1, 1]), float(res.matrix[1, 2])],
        ],
    }


def case0_entry_to_dict(entry: Case0MatchEntry) -> Dict[str, Any]:
    out = result_to_dict(entry.result)
    out["scale"] = float(entry.scale)
    out["bbox_xyxy"] = [
        float(entry.bbox[0]),
        float(entry.bbox[1]),
        float(entry.bbox[2]),
        float(entry.bbox[3]),
    ]
    out["center_xy"] = [float(entry.center[0]), float(entry.center[1])]
    return out


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    else:
        import json

        text = json.dumps(data, indent=2, ensure_ascii=True)
    path.write_text(text, encoding="utf-8")


def _warped_corners(template_shape: Tuple[int, int], matrix: np.ndarray) -> np.ndarray:
    h, w = template_shape
    corners = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.transform(corners, matrix).reshape(-1, 2)


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
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def _build_case0_entry(res: MatchResult, scale: float, template_bgr: Image) -> Case0MatchEntry:
    warped = _warped_corners(template_bgr.shape[:2], res.matrix)
    x1 = float(np.min(warped[:, 0]))
    y1 = float(np.min(warped[:, 1]))
    x2 = float(np.max(warped[:, 0]))
    y2 = float(np.max(warped[:, 1]))
    center = (float(np.mean(warped[:, 0])), float(np.mean(warped[:, 1])))
    angle = math.degrees(math.atan2(float(res.matrix[1, 0]), float(res.matrix[0, 0])))
    if angle < 0.0:
        angle += 360.0
    return Case0MatchEntry(
        result=res,
        scale=float(scale),
        template_bgr=template_bgr,
        bbox=(x1, y1, x2, y2),
        center=center,
        angle_deg=float(angle),
    )


def _select_case0_entries(
    entries: Sequence[Case0MatchEntry],
    max_keep: int,
    iou_threshold: float = 0.45,
    center_threshold: float = 14.0,
    angle_threshold_deg: float = 35.0,
) -> List[Case0MatchEntry]:
    ranked = sorted(entries, key=lambda e: e.result.score, reverse=True)
    if max_keep <= 0:
        max_keep = len(ranked)

    kept: List[Case0MatchEntry] = []
    center_threshold_sq = center_threshold * center_threshold
    for cur in ranked:
        drop = False
        for old in kept:
            iou = _bbox_iou(cur.bbox, old.bbox)
            dx = cur.center[0] - old.center[0]
            dy = cur.center[1] - old.center[1]
            da = abs(cur.angle_deg - old.angle_deg)
            da = min(da, 360.0 - da)
            if iou >= iou_threshold or (dx * dx + dy * dy <= center_threshold_sq and da <= angle_threshold_deg):
                drop = True
                break
        if drop:
            continue
        kept.append(cur)
        if len(kept) >= max_keep:
            break
    return kept


def _filter_case0_entries(
    entries: Sequence[Case0MatchEntry],
    min_score: float,
    min_inlier_ratio: float,
    max_trimmed_mean: float,
) -> List[Case0MatchEntry]:
    out: List[Case0MatchEntry] = []
    for entry in entries:
        r = entry.result
        if r.score < min_score:
            continue
        if r.inlier_ratio < min_inlier_ratio:
            continue
        if r.trimmed_mean > max_trimmed_mean:
            continue
        out.append(entry)
    return out


def _crop_to_stride(image: Image, stride: int) -> Image:
    h, w = image.shape[:2]
    n = h // stride
    m = w // stride
    if n <= 0 or m <= 0:
        return image.copy()
    return image[: n * stride, : m * stride].copy()


def _dedup_case2_results(
    results: Sequence[MatchResult],
    template_shape: Tuple[int, int],
    max_keep: int = 4,
    center_radius: float = 150.0,
    iou_threshold: float = 0.28,
) -> List[MatchResult]:
    if not results:
        return []
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    radius_sq = center_radius * center_radius
    kept: List[MatchResult] = []
    kept_meta: List[Tuple[Tuple[float, float, float, float], Tuple[float, float]]] = []

    for res in ranked:
        warped = _warped_corners(template_shape, res.matrix)
        x1 = float(np.min(warped[:, 0]))
        y1 = float(np.min(warped[:, 1]))
        x2 = float(np.max(warped[:, 0]))
        y2 = float(np.max(warped[:, 1]))
        center = (float(np.mean(warped[:, 0])), float(np.mean(warped[:, 1])))
        bbox = (x1, y1, x2, y2)

        dup = False
        for old_bbox, old_center in kept_meta:
            iou = _bbox_iou(bbox, old_bbox)
            dx = center[0] - old_center[0]
            dy = center[1] - old_center[1]
            if iou >= iou_threshold or (dx * dx + dy * dy <= radius_sq):
                dup = True
                break
        if dup:
            continue

        kept.append(res)
        kept_meta.append((bbox, center))
        if len(kept) >= max_keep:
            break

    if kept:
        return kept
    return ranked[:max_keep]


def _template_contours(template_bgr: Image, canny_low: int = 80, canny_high: int = 180) -> List[np.ndarray]:
    gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: List[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 8:
            continue
        if cv2.contourArea(contour) < 4.0:
            continue
        out.append(contour.astype(np.float32))
    if out:
        return out

    # Fallback to template rectangle when edge contour is unavailable.
    h, w = gray.shape[:2]
    rect = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.float32)
    return [rect]


def draw_matches_contour_overlay(
    scene_bgr: Image,
    template_bgr: Image,
    results: Sequence[MatchResult],
    draw_top: int = 20,
    legend_top: int = 20,
    draw_nms_iou: float = 0.40,
    draw_nms_center: float = 18.0,
) -> Image:
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

    template_shape = template_bgr.shape[:2]
    base_contours = _template_contours(template_bgr)
    draw_items = select_results_for_drawing(
        results=results,
        template_shape=template_shape,
        max_draw=draw_top,
        iou_threshold=draw_nms_iou,
        center_threshold=draw_nms_center,
    )

    draw_count = len(draw_items)
    for idx, item in enumerate(draw_items, start=1):
        res, warped_quad = item
        color = palette[(idx - 1) % len(palette)]
        for contour in base_contours:
            warped = cv2.transform(contour, res.matrix).astype(np.int32)
            if warped.shape[0] >= 2:
                cv2.polylines(out, [warped], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        center = np.mean(warped_quad, axis=0)
        cv2.putText(
            out,
            f"#{idx}",
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    list_count = min(len(results), max(1, legend_top))
    lines = [f"Top {len(results)} matches (contour draw {draw_count})"]
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


def draw_case0_overlay_multi(
    scene_bgr: Image,
    entries: Sequence[Case0MatchEntry],
    draw_top: int = 5,
    legend_top: int = 5,
) -> Image:
    out = scene_bgr.copy()
    if not entries:
        return out

    palette = [
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
        (0, 180, 255),
        (255, 80, 0),
    ]
    top = entries if draw_top <= 0 else entries[:draw_top]
    for idx, entry in enumerate(top, start=1):
        color = palette[(idx - 1) % len(palette)]
        base_contours = _template_contours(entry.template_bgr)
        for contour in base_contours:
            warped = cv2.transform(contour, entry.result.matrix).astype(np.int32)
            if warped.shape[0] >= 2:
                cv2.polylines(out, [warped], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        cx, cy = entry.center
        cv2.putText(
            out,
            f"#{idx}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    list_count = min(len(entries), max(1, legend_top))
    lines = [f"Top {len(entries)} matches (cross-scale draw {len(top)})"]
    for idx, entry in enumerate(entries[:list_count], start=1):
        lines.append(
            f"#{idx:02d} s={entry.result.score:.3f} sc={entry.scale:.2f} a={entry.angle_deg:6.1f} {entry.result.source}"
        )
    if list_count < len(entries):
        lines.append(f"... {len(entries) - list_count} more")

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


def run_case0(
    root: Path,
    out_dir: Path,
    occlusion: float,
    max_results: int,
    draw_top: int,
    legend_top: int,
    max_candidates: int,
) -> Dict[str, Any]:
    case_dir = root / "test" / "case0"
    info_yaml = case_dir / "circle_info.yaml"
    templ_path = case_dir / "templ" / "circle.png"
    infos = load_infos(info_yaml)
    scales = unique_sorted([x[1] for x in infos], decimals=5)
    angles = unique_sorted([x[0] for x in infos], decimals=5)
    angle_step = infer_angle_step_deg(infos)
    rotation_step = angle_step if angle_step > 0 else 0.0

    template_base = read_image(templ_path)
    scenes = sorted(
        p
        for p in case_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        and not p.name.startswith("_")
    )

    scene_results: List[SceneRunResult] = []
    for scene_path in scenes:
        scene = read_image(scene_path)
        per_scale_keep = max(3, min(12, max_results))
        all_entries: List[Case0MatchEntry] = []

        for scale in scales:
            scaled_template = resize_template(template_base, scale)
            try:
                results = match_once(
                    template=scaled_template,
                    scene=scene,
                    rotation_step_deg=rotation_step,
                    occlusion=occlusion,
                    max_results=per_scale_keep,
                    max_candidates=max_candidates,
                )
            except Exception:
                continue
            if not results:
                continue
            for res in results[:per_scale_keep]:
                all_entries.append(_build_case0_entry(res, float(scale), scaled_template))

        strict_entries = _filter_case0_entries(
            all_entries,
            min_score=0.60,
            min_inlier_ratio=0.50,
            max_trimmed_mean=4.80,
        )
        entry_pool = strict_entries if len(strict_entries) >= 5 else all_entries

        selected_entries = _select_case0_entries(
            entry_pool,
            max_keep=max_results,
            iou_threshold=0.35,
            center_threshold=45.0,
            angle_threshold_deg=35.0,
        )

        out_img_path = out_dir / f"{scene_path.stem}_robust.png"
        out_yaml_path = out_dir / f"{scene_path.stem}_robust.yaml"
        if not selected_entries:
            cv2.imwrite(str(out_img_path), scene)
            result_data: Dict[str, Any] = {
                "scene": str(scene_path),
                "success": False,
                "message": "No valid candidate pose found across all scales.",
                "used_scale_count": len(scales),
                "rotation_step_deg": float(rotation_step),
                "draw_mode": "contour",
                "match_mode": "cross_scale_global_nms",
                "quality_filter": {
                    "min_score": 0.60,
                    "min_inlier_ratio": 0.50,
                    "max_trimmed_mean": 4.80,
                    "used_strict_pool": bool(len(strict_entries) >= 5),
                },
            }
            dump_yaml(out_yaml_path, result_data)
            scene_results.append(
                SceneRunResult(
                    scene=str(scene_path),
                    output_image=str(out_img_path),
                    output_yaml=str(out_yaml_path),
                    success=False,
                    message="No valid candidate pose found.",
                )
            )
            continue

        case0_draw_top = min(8, draw_top) if draw_top > 0 else 8
        case0_legend_top = min(8, legend_top) if legend_top > 0 else 8
        overlay = draw_case0_overlay_multi(
            scene_bgr=scene,
            entries=selected_entries,
            draw_top=case0_draw_top,
            legend_top=case0_legend_top,
        )
        cv2.imwrite(str(out_img_path), overlay)
        result_data = {
            "scene": str(scene_path),
            "success": True,
            "best_scale": float(selected_entries[0].scale),
            "rotation_step_deg": float(rotation_step),
            "draw_mode": "contour",
            "match_mode": "cross_scale_global_nms",
            "input_candidates": len(all_entries),
            "strict_candidates": len(strict_entries),
            "used_strict_pool": bool(len(strict_entries) >= 5),
            "selected_candidates": len(selected_entries),
            "top_matches": [case0_entry_to_dict(e) for e in selected_entries],
        }
        dump_yaml(out_yaml_path, result_data)
        scene_results.append(
            SceneRunResult(
                scene=str(scene_path),
                output_image=str(out_img_path),
                output_yaml=str(out_yaml_path),
                success=True,
                message="ok",
                best_score=float(selected_entries[0].result.score),
                best_scale=float(selected_entries[0].scale),
            )
        )

    return {
        "case": "case0",
        "info_yaml": str(info_yaml),
        "template": str(templ_path),
        "scale_count": len(scales),
        "angle_values_in_info": angles[:8] + (["..."] if len(angles) > 8 else []),
        "rotation_step_deg": float(rotation_step),
        "scenes": [vars(x) for x in scene_results],
    }


def make_case1_template(train_img: Image) -> Image:
    # Match the upstream C++ test setup: ROI(130,110,270,270) + padding 100.
    roi = train_img[110 : 110 + 270, 130 : 130 + 270].copy()
    pad = 100
    out = np.zeros((roi.shape[0] + 2 * pad, roi.shape[1] + 2 * pad, 3), dtype=roi.dtype)
    out[pad : pad + roi.shape[0], pad : pad + roi.shape[1]] = roi
    return out


def run_case_single_scene(
    case_name: str,
    case_dir: Path,
    info_yaml_name: str,
    template: Image,
    scene_path: Path,
    out_dir: Path,
    occlusion: float,
    max_results: int,
    draw_top: int,
    legend_top: int,
    max_candidates: int,
) -> Dict[str, Any]:
    info_yaml = case_dir / info_yaml_name
    infos = load_infos(info_yaml)
    angle_step = infer_angle_step_deg(infos)
    rotation_step = angle_step if angle_step > 0 else 0.0

    scene_raw = read_image(scene_path)
    if case_name == "case2":
        scene = _crop_to_stride(scene_raw, stride=16)
    else:
        scene = scene_raw
    out_img_path = out_dir / f"{scene_path.stem}_robust.png"
    out_yaml_path = out_dir / f"{scene_path.stem}_robust.yaml"

    try:
        results = match_once(
            template=template,
            scene=scene,
            rotation_step_deg=rotation_step,
            occlusion=occlusion,
            max_results=max_results,
            max_candidates=max_candidates,
        )
    except Exception as exc:
        results = []
        err_msg = f"{type(exc).__name__}: {exc}"
    else:
        err_msg = ""

    if case_name == "case2" and results:
        # Keep one hypothesis per cross instance (reduce rotation-symmetric duplicates).
        results = _dedup_case2_results(
            results=results,
            template_shape=template.shape[:2],
            max_keep=4,
            center_radius=150.0,
            iou_threshold=0.28,
        )

    if not results:
        cv2.imwrite(str(out_img_path), scene)
        result_yaml = {
            "scene": str(scene_path),
            "success": False,
            "message": err_msg or "No valid candidate pose found.",
            "rotation_step_deg": float(rotation_step),
            "preprocess": {
                "case2_stride_crop": bool(case_name == "case2"),
                "source_shape_hw": [int(scene_raw.shape[0]), int(scene_raw.shape[1])],
                "match_shape_hw": [int(scene.shape[0]), int(scene.shape[1])],
            },
        }
        dump_yaml(out_yaml_path, result_yaml)
        return {
            "case": case_name,
            "info_yaml": str(info_yaml),
            "rotation_step_deg": float(rotation_step),
            "scenes": [
                vars(
                    SceneRunResult(
                        scene=str(scene_path),
                        output_image=str(out_img_path),
                        output_yaml=str(out_yaml_path),
                        success=False,
                        message=result_yaml["message"],
                    )
                )
            ],
        }

    overlay = draw_matches_contour_overlay(
        scene_bgr=scene,
        template_bgr=template,
        results=results,
        draw_top=draw_top,
        legend_top=legend_top,
    )
    cv2.imwrite(str(out_img_path), overlay)
    result_yaml = {
        "scene": str(scene_path),
        "success": True,
        "rotation_step_deg": float(rotation_step),
        "draw_mode": "contour",
        "preprocess": {
            "case2_stride_crop": bool(case_name == "case2"),
            "source_shape_hw": [int(scene_raw.shape[0]), int(scene_raw.shape[1])],
            "match_shape_hw": [int(scene.shape[0]), int(scene.shape[1])],
        },
        "postprocess": {
            "case2_one_per_cross": bool(case_name == "case2"),
            "max_keep": 4 if case_name == "case2" else max_results,
            "center_radius": 150.0 if case_name == "case2" else None,
            "iou_threshold": 0.28 if case_name == "case2" else None,
        },
        "top_matches": [result_to_dict(r) for r in results],
    }
    dump_yaml(out_yaml_path, result_yaml)
    return {
        "case": case_name,
        "info_yaml": str(info_yaml),
        "rotation_step_deg": float(rotation_step),
        "scenes": [
            vars(
                SceneRunResult(
                    scene=str(scene_path),
                    output_image=str(out_img_path),
                    output_yaml=str(out_yaml_path),
                    success=True,
                    message="ok",
                    best_score=float(results[0].score),
                )
            )
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test robust_template_match on test cases.")
    parser.add_argument("--root", default=".", help="Repo root path.")
    parser.add_argument("--out-name", default="result_robust", help="Per-case output directory name.")
    parser.add_argument("--occlusion", type=float, default=0.30, help="Occlusion tolerance for matcher.")
    parser.add_argument("--max-results", type=int, default=30, help="Top results kept per scene.")
    parser.add_argument("--draw-top", type=int, default=30, help="How many detections to draw.")
    parser.add_argument("--legend-top", type=int, default=30, help="How many detections to list in panel.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum translation vote hypotheses before robust scoring.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not (root / "test").exists():
        raise FileNotFoundError(f"Could not find test directory under: {root}")

    summaries: List[Dict[str, Any]] = []

    case0_out = root / "test" / "case0" / args.out_name
    case0_out.mkdir(parents=True, exist_ok=True)
    summaries.append(
        run_case0(
            root=root,
            out_dir=case0_out,
            occlusion=args.occlusion,
            max_results=args.max_results,
            draw_top=args.draw_top,
            legend_top=args.legend_top,
            max_candidates=args.max_candidates,
        )
    )

    case1_dir = root / "test" / "case1"
    case1_out = case1_dir / args.out_name
    case1_out.mkdir(parents=True, exist_ok=True)
    case1_train = read_image(case1_dir / "train.png")
    case1_template = make_case1_template(case1_train)
    summaries.append(
        run_case_single_scene(
            case_name="case1",
            case_dir=case1_dir,
            info_yaml_name="test_info.yaml",
            template=case1_template,
            scene_path=case1_dir / "test.png",
            out_dir=case1_out,
            occlusion=args.occlusion,
            max_results=args.max_results,
            draw_top=args.draw_top,
            legend_top=args.legend_top,
            max_candidates=args.max_candidates,
        )
    )

    case2_dir = root / "test" / "case2"
    case2_out = case2_dir / args.out_name
    case2_out.mkdir(parents=True, exist_ok=True)
    case2_template = read_image(case2_dir / "train.png")
    summaries.append(
        run_case_single_scene(
            case_name="case2",
            case_dir=case2_dir,
            info_yaml_name="test_info.yaml",
            template=case2_template,
            scene_path=case2_dir / "test.png",
            out_dir=case2_out,
            occlusion=args.occlusion,
            max_results=args.max_results,
            draw_top=args.draw_top,
            legend_top=args.legend_top,
            max_candidates=args.max_candidates,
        )
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "root": str(root),
            "out_name": args.out_name,
            "occlusion": float(args.occlusion),
            "max_results": int(args.max_results),
            "draw_top": int(args.draw_top),
            "legend_top": int(args.legend_top),
            "max_candidates": int(args.max_candidates),
        },
        "cases": summaries,
    }

    for case in ("case0", "case1", "case2"):
        path = root / "test" / case / args.out_name / "summary.yaml"
        case_summary = {
            "generated_at_utc": summary["generated_at_utc"],
            "args": summary["args"],
            "case": next(c for c in summaries if c["case"] == case),
        }
        dump_yaml(path, case_summary)

    root_summary = root / "test" / args.out_name
    root_summary.mkdir(parents=True, exist_ok=True)
    dump_yaml(root_summary / "summary.yaml", summary)

    ok = 0
    total = 0
    for case in summaries:
        for scene in case["scenes"]:
            total += 1
            if scene["success"]:
                ok += 1
    print(f"Batch finished: {ok}/{total} scenes succeeded.")
    print(f"Global summary: {root_summary / 'summary.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
