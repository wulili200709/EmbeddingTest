#!/usr/bin/env python3
"""
Export intermediate visualizations for line2Dup-like matching.

This script is intended for debugging template quality:
- template feature points per pyramid level
- quantized orientation maps
- coarse similarity heatmap on scene
- final match overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from line2dup_like_matcher import (
    ColorGradientPyramid,
    Line2DupLikeDetector,
    Match,
    ShapeInfoProducer,
    compute_response_maps,
    draw_matches,
    nms_matches,
    offset_from_T,
    similarity_full,
    spread_or,
)


def parse_levels(arg: str) -> List[int]:
    vals = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("T levels cannot be empty.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug visualization for line2Dup-like matching.")
    parser.add_argument("--template", required=True, help="Template image path.")
    parser.add_argument("--scene", required=True, help="Scene image path.")
    parser.add_argument("--template-mask", default="", help="Template mask path (optional).")
    parser.add_argument("--scene-mask", default="", help="Scene mask path (optional).")
    parser.add_argument("--out-dir", required=True, help="Directory to save debug images.")
    parser.add_argument("--threshold", type=float, default=55.0, help="Similarity threshold in [0,100].")
    parser.add_argument("--num-features", type=int, default=128, help="Requested feature count per template.")
    parser.add_argument("--weak-thresh", type=float, default=30.0, help="Weak threshold for quantization.")
    parser.add_argument("--strong-thresh", type=float, default=60.0, help="Strong threshold for template features.")
    parser.add_argument("--levels", default="4,8", help="Pyramid T levels, e.g. 4,8")
    parser.add_argument("--nms-iou", type=float, default=0.30, help="NMS IoU threshold on final matches.")
    parser.add_argument("--topk", type=int, default=20, help="How many matches to draw.")
    parser.add_argument("--angle-start", type=float, default=0.0, help="Training angle start.")
    parser.add_argument("--angle-end", type=float, default=0.0, help="Training angle end.")
    parser.add_argument("--angle-step", type=float, default=10.0, help="Training angle step.")
    parser.add_argument("--scale-start", type=float, default=1.0, help="Training scale start.")
    parser.add_argument("--scale-end", type=float, default=1.0, help="Training scale end.")
    parser.add_argument("--scale-step", type=float, default=0.1, help="Training scale step.")
    return parser.parse_args()


def orientation_palette_bgr() -> List[Tuple[int, int, int]]:
    return [
        (0, 0, 255),
        (0, 128, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 128, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]


def quantized_to_index_map(one_hot: np.ndarray) -> np.ndarray:
    out = np.full(one_hot.shape, -1, dtype=np.int16)
    for i in range(8):
        out[(one_hot & np.uint8(1 << i)) != 0] = i
    return out


def render_quantized(one_hot: np.ndarray) -> np.ndarray:
    idx = quantized_to_index_map(one_hot)
    h, w = idx.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    palette = orientation_palette_bgr()
    for i, color in enumerate(palette):
        out[idx == i] = color
    return out


def draw_template_features(
    image_bgr: np.ndarray,
    templ,
    title: str,
) -> np.ndarray:
    out = image_bgr.copy()
    x1 = int(templ.tl_x)
    y1 = int(templ.tl_y)
    x2 = int(templ.tl_x + templ.width)
    y2 = int(templ.tl_y + templ.height)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

    palette = orientation_palette_bgr()
    for f in templ.features:
        color = palette[int(f.label) % len(palette)]
        px = int(f.x + templ.tl_x)
        py = int(f.y + templ.tl_y)
        cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)

    cv2.putText(
        out,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def crop_to_stride(image: np.ndarray, stride: int) -> np.ndarray:
    h, w = image.shape[:2]
    n = h // stride
    m = w // stride
    if n <= 0 or m <= 0:
        return image.copy()
    return image[: n * stride, : m * stride].copy()


def build_scene_levels(
    scene: np.ndarray,
    scene_mask: np.ndarray | None,
    weak_thresh: float,
    strong_thresh: float,
    num_features: int,
    T_levels: Sequence[int],
):
    qp = ColorGradientPyramid(
        scene,
        scene_mask,
        weak_threshold=weak_thresh,
        num_features=num_features,
        strong_threshold=strong_thresh,
    )

    levels = []
    for l, T in enumerate(T_levels):
        if l > 0:
            qp.pyr_down()
        quant = qp.quantize()
        h, w = quant.shape[:2]
        h = (h // T) * T
        w = (w // T) * T
        if h <= 0 or w <= 0:
            continue
        quant = quant[:h, :w]
        spread = spread_or(quant, int(T))
        response_maps = compute_response_maps(spread)
        levels.append(
            {
                "level": l,
                "T": int(T),
                "width": int(w),
                "height": int(h),
                "quant": quant,
                "spread": spread,
                "response_maps": response_maps,
            }
        )
    return levels


def save_similarity_heatmap(
    out_dir: Path,
    scene_bgr: np.ndarray,
    detector: Line2DupLikeDetector,
    class_id: str,
    scene_levels,
) -> None:
    if not scene_levels:
        return

    lowest_idx = len(scene_levels) - 1
    scene_low = scene_levels[lowest_idx]
    width = scene_low["width"]
    height = scene_low["height"]
    T = scene_low["T"]

    best_sim = None
    best_tid = -1
    best_max = -1.0

    for tid in range(len(detector.class_templates.get(class_id, []))):
        templ_low = detector.get_templates(class_id, tid)[lowest_idx]
        sim_map = similarity_full(
            scene_low["response_maps"],
            templ_low,
            width,
            height,
            T,
        )
        if sim_map.size == 0:
            continue
        cur_max = float(np.max(sim_map))
        if cur_max > best_max:
            best_max = cur_max
            best_tid = tid
            best_sim = sim_map

    if best_sim is None:
        return

    sim_norm = cv2.normalize(best_sim, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sim_heat_small = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
    sim_heat = cv2.resize(sim_heat_small, (width, height), interpolation=cv2.INTER_LINEAR)

    scene_crop = scene_bgr[:height, :width].copy()
    overlay = cv2.addWeighted(scene_crop, 0.62, sim_heat, 0.38, 0.0)
    cv2.putText(
        overlay,
        f"coarse similarity heatmap (template_id={best_tid}, max={best_max:.2f})",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_dir / "scene_coarse_similarity_heatmap.png"), overlay)

    flat = best_sim.reshape(-1)
    top_n = min(20, flat.size)
    top_idx = np.argpartition(flat, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

    top_img = scene_crop.copy()
    off = offset_from_T(T)
    for rank, idx in enumerate(top_idx, start=1):
        r, c = np.unravel_index(int(idx), best_sim.shape)
        x = int(c * T + off)
        y = int(r * T + off)
        score = float(best_sim[r, c])
        cv2.circle(top_img, (x, y), 5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            top_img,
            f"{rank}:{score:.1f}",
            (x + 4, max(16, y - 4)),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "scene_coarse_top_candidates.png"), top_img)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    ok_templates = 0
    for info in infos:
        src_i = producer.src_of(info)
        mask_i = producer.mask_of(info)
        nfeat = max(16, int(round(args.num_features * info.scale)))
        tid = detector.add_template(
            src_i,
            class_id=class_id,
            object_mask=mask_i,
            num_features=nfeat,
            metadata={"angle": float(info.angle), "scale": float(info.scale)},
        )
        if tid >= 0:
            ok_templates += 1

    if ok_templates == 0:
        print("No template could be extracted.")
        return 2

    # Save quantized orientation maps and feature overlays for template pyramid levels.
    qp_templ = ColorGradientPyramid(
        template,
        templ_mask,
        weak_threshold=args.weak_thresh,
        num_features=args.num_features,
        strong_threshold=args.strong_thresh,
    )
    tp = detector.get_templates(class_id, 0)
    templ_img_level = template.copy()
    for l in range(len(levels)):
        if l > 0:
            qp_templ.pyr_down()
            templ_img_level = cv2.pyrDown(templ_img_level)

        q = qp_templ.quantize()
        q_vis = render_quantized(q)
        cv2.imwrite(str(out_dir / f"template_level{l}_quantized.png"), q_vis)

        vis_feat = draw_template_features(
            templ_img_level,
            tp[l],
            title=f"template level={l}, T={levels[l]}, feats={len(tp[l].features)}",
        )
        cv2.imwrite(str(out_dir / f"template_level{l}_features.png"), vis_feat)

    # Save scene quantized map on level 0 for quick inspection.
    qp_scene_l0 = ColorGradientPyramid(
        scene,
        scene_mask,
        weak_threshold=args.weak_thresh,
        num_features=args.num_features,
        strong_threshold=args.strong_thresh,
    )
    scene_q0 = qp_scene_l0.quantize()
    scene_q0_vis = render_quantized(scene_q0)
    cv2.imwrite(str(out_dir / "scene_level0_quantized.png"), scene_q0_vis)

    # Coarse similarity map on lowest pyramid.
    max_stride = max(levels)
    scene_for_match = crop_to_stride(scene, stride=max_stride * 2)
    if scene_mask is not None:
        scene_mask_for_match = crop_to_stride(scene_mask, stride=max_stride * 2)
    else:
        scene_mask_for_match = None

    scene_levels = build_scene_levels(
        scene_for_match,
        scene_mask_for_match,
        weak_thresh=args.weak_thresh,
        strong_thresh=args.strong_thresh,
        num_features=args.num_features,
        T_levels=levels,
    )
    save_similarity_heatmap(out_dir, scene_for_match, detector, class_id, scene_levels)

    # Final matches for reference.
    matches = detector.match(scene_for_match, threshold=args.threshold, class_ids=[class_id], mask=scene_mask_for_match)
    raw_cnt = len(matches)
    matches = nms_matches(detector, matches, iou_threshold=args.nms_iou)
    matches.sort(key=lambda m: m.similarity, reverse=True)
    vis_match = draw_matches(detector, scene_for_match, matches, topk=args.topk)
    cv2.imwrite(str(out_dir / "scene_final_matches.png"), vis_match)

    top_lines: List[str] = []
    for i, m in enumerate(matches[: min(args.topk, len(matches))], start=1):
        meta = detector.get_template_meta(m.class_id, m.template_id)
        top_lines.append(
            f"[{i}] sim={m.similarity:.2f}, x={m.x}, y={m.y}, "
            f"templ_id={m.template_id}, angle={meta.get('angle', 0.0):.2f}, scale={meta.get('scale', 1.0):.3f}"
        )

    summary_lines = [
        f"templates_loaded={ok_templates}",
        f"raw_matches={raw_cnt}",
        f"after_nms={len(matches)}",
        f"threshold={args.threshold}",
        f"nms_iou={args.nms_iou}",
        "",
        "top_matches:",
    ]
    summary_lines.extend(top_lines if top_lines else ["(none)"])
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved debug images to: {out_dir}")
    print((out_dir / "summary.txt").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
