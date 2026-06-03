#!/usr/bin/env python3
"""
Build and save shape templates into a reusable model package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shape.like_matcher import (
    ColorGradientPyramid,
    Line2DupLikeDetector,
    ShapeInfoProducer,
    save_detector_model,
)


def parse_levels(arg: str) -> List[int]:
    vals = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("T levels cannot be empty.")
    return vals


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
    out = np.zeros((idx.shape[0], idx.shape[1], 3), dtype=np.uint8)
    palette = orientation_palette_bgr()
    for i, color in enumerate(palette):
        out[idx == i] = color
    return out


def draw_template_features(image_bgr: np.ndarray, templ, title: str) -> np.ndarray:
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
        cv2.drawMarker(
            out,
            (px, py),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            out,
            (px, py),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=7,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shape model from a template image.")
    parser.add_argument("--template", required=True, help="Template image path.")
    parser.add_argument("--template-mask", default="", help="Template mask path (optional).")
    parser.add_argument("--out-model", required=True, help="Output model path (JSON).")
    parser.add_argument("--class-id", default="object", help="Class id name to store in model.")
    parser.add_argument("--num-features", type=int, default=128, help="Requested feature count per template.")
    parser.add_argument("--weak-thresh", type=float, default=30.0, help="Weak threshold for quantization.")
    parser.add_argument("--strong-thresh", type=float, default=60.0, help="Strong threshold for template features.")
    parser.add_argument("--levels", default="4,8", help="Pyramid T levels, e.g. 4,8")
    parser.add_argument("--angle-start", type=float, default=0.0, help="Training angle start.")
    parser.add_argument("--angle-end", type=float, default=0.0, help="Training angle end.")
    parser.add_argument("--angle-step", type=float, default=10.0, help="Training angle step.")
    parser.add_argument("--scale-start", type=float, default=1.0, help="Training scale start.")
    parser.add_argument("--scale-end", type=float, default=1.0, help="Training scale end.")
    parser.add_argument("--scale-step", type=float, default=0.1, help="Training scale step.")
    parser.add_argument(
        "--vis-dir",
        default="",
        help="Directory to save template visualization. Default: <out-model-stem>_vis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = cv2.imread(args.template, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Failed to read template: {args.template}")

    templ_mask = None
    if args.template_mask:
        templ_mask = cv2.imread(args.template_mask, cv2.IMREAD_GRAYSCALE)
        if templ_mask is None:
            raise FileNotFoundError(f"Failed to read template mask: {args.template_mask}")

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

    success = 0
    first_success = None
    first_tid = -1
    for info in infos:
        src_i = producer.src_of(info)
        mask_i = producer.mask_of(info)
        nfeat = max(16, int(round(args.num_features * info.scale)))
        templ_id = detector.add_template(
            src_i,
            class_id=args.class_id,
            object_mask=mask_i,
            num_features=nfeat,
            metadata={"angle": float(info.angle), "scale": float(info.scale)},
        )
        if templ_id >= 0:
            success += 1
            if first_success is None:
                first_success = info
                first_tid = templ_id

    if success == 0:
        print("No template could be extracted.")
        return 2

    out_model = Path(args.out_model)
    save_detector_model(detector, str(out_model))

    if args.vis_dir:
        vis_dir = Path(args.vis_dir)
    else:
        vis_dir = out_model.parent / f"{out_model.stem}_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Save first successful template visualizations.
    assert first_success is not None
    assert first_tid >= 0
    src0 = producer.src_of(first_success)
    mask0 = producer.mask_of(first_success)
    qp = ColorGradientPyramid(
        src0,
        mask0,
        weak_threshold=args.weak_thresh,
        num_features=args.num_features,
        strong_threshold=args.strong_thresh,
    )

    src_level = src0.copy()
    tp = detector.get_templates(args.class_id, first_tid)
    for l in range(detector.pyramid_levels):
        if l > 0:
            qp.pyr_down()
            src_level = cv2.pyrDown(src_level)
        q = qp.quantize()
        q_vis = render_quantized(q)
        cv2.imwrite(str(vis_dir / f"template_level{l}_quantized.png"), q_vis)

        title = f"class={args.class_id} tid={first_tid} L{l} T={levels[l]} feats={len(tp[l].features)}"
        feat_vis = draw_template_features(src_level, tp[l], title)
        cv2.imwrite(str(vis_dir / f"template_level{l}_features.png"), feat_vis)

    summary: Dict[str, object] = {
        "model_path": str(out_model.resolve()),
        "template_path": str(Path(args.template).resolve()),
        "template_mask_path": str(Path(args.template_mask).resolve()) if args.template_mask else "",
        "class_id": args.class_id,
        "detector_params": {
            "num_features": int(args.num_features),
            "weak_thresh": float(args.weak_thresh),
            "strong_thresh": float(args.strong_thresh),
            "levels": levels,
        },
        "train_ranges": {
            "angle_start": float(args.angle_start),
            "angle_end": float(args.angle_end),
            "angle_step": float(args.angle_step),
            "scale_start": float(args.scale_start),
            "scale_end": float(args.scale_end),
            "scale_step": float(args.scale_step),
        },
        "train_infos_total": len(infos),
        "templates_loaded": success,
        "first_template_id": first_tid,
        "first_template_meta": detector.get_template_meta(args.class_id, first_tid),
    }
    (vis_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"templates_loaded={success}")
    print(f"saved_model={out_model}")
    print(f"saved_vis={vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
