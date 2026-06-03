#!/usr/bin/env python3
"""
Find objects in scene using a prebuilt shape template model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shape.like_matcher import (
    draw_matches,
    load_detector_model,
    nms_matches,
)


def parse_class_ids(arg: str) -> List[str]:
    ids = [x.strip() for x in arg.split(",") if x.strip()]
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find with shape model.")
    parser.add_argument("--model", required=True, help="Model path from shape_make_template.py")
    parser.add_argument("--scene", required=True, help="Scene image path.")
    parser.add_argument("--scene-mask", default="", help="Scene mask path (optional).")
    parser.add_argument("--class-ids", default="", help="Comma-separated class ids. Empty means all.")
    parser.add_argument("--out", default="shape_find_result.png", help="Output visualization path.")
    parser.add_argument("--threshold", type=float, default=90.0, help="Similarity threshold in [0,100].")
    parser.add_argument(
        "--auto-sweep",
        action="store_true",
        help="If no match at --threshold, sweep down by --sweep-step until --sweep-min.",
    )
    parser.add_argument("--sweep-min", type=float, default=20.0, help="Minimum threshold for --auto-sweep.")
    parser.add_argument("--sweep-step", type=float, default=5.0, help="Threshold step for --auto-sweep.")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="NMS IoU threshold.")
    parser.add_argument("--topk", type=int, default=20, help="How many matches to print and draw.")
    parser.add_argument(
        "--crop-stride",
        type=int,
        default=0,
        help="If >0, crop scene to (h//stride*stride, w//stride*stride) before matching.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print stage timings for load/read/crop/match/NMS/draw/save/total.",
    )
    return parser.parse_args()


def main() -> int:
    t_total_start = time.perf_counter()
    args = parse_args()

    bench: dict[str, float] = {
        "model_load_s": 0.0,
        "scene_read_s": 0.0,
        "scene_mask_read_s": 0.0,
        "crop_s": 0.0,
        "match_s": 0.0,
        "nms_s": 0.0,
        "draw_s": 0.0,
        "save_s": 0.0,
    }
    match_attempts = 0

    t0 = time.perf_counter()
    detector = load_detector_model(args.model)
    bench["model_load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scene = cv2.imread(args.scene, cv2.IMREAD_COLOR)
    bench["scene_read_s"] = time.perf_counter() - t0
    if scene is None:
        raise FileNotFoundError(f"Failed to read scene: {args.scene}")

    t0 = time.perf_counter()
    if int(args.crop_stride) > 0:
        stride = int(args.crop_stride)
        h, w = scene.shape[:2]
        h2 = (h // stride) * stride
        w2 = (w // stride) * stride
        if h2 > 0 and w2 > 0:
            scene = scene[:h2, :w2].copy()
    bench["crop_s"] = time.perf_counter() - t0

    scene_mask = None
    if args.scene_mask:
        t0 = time.perf_counter()
        scene_mask = cv2.imread(args.scene_mask, cv2.IMREAD_GRAYSCALE)
        bench["scene_mask_read_s"] = time.perf_counter() - t0
        if scene_mask is None:
            raise FileNotFoundError(f"Failed to read scene mask: {args.scene_mask}")

    class_ids = parse_class_ids(args.class_ids)
    if not class_ids:
        class_ids = detector.class_ids()

    def run_once(threshold: float):
        nonlocal match_attempts
        match_attempts += 1
        t0 = time.perf_counter()
        ms = detector.match(scene, threshold=threshold, class_ids=class_ids, mask=scene_mask)
        bench["match_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        ms = nms_matches(detector, ms, iou_threshold=args.nms_iou)
        ms.sort(key=lambda m: m.similarity, reverse=True)
        bench["nms_s"] += time.perf_counter() - t0
        return ms

    used_threshold = float(args.threshold)
    matches = run_once(used_threshold)
    if args.auto_sweep and len(matches) == 0:
        step = max(0.1, float(args.sweep_step))
        sweep_min = float(args.sweep_min)
        th = float(args.threshold) - step
        while th >= sweep_min - 1e-9:
            cand = run_once(th)
            if cand:
                matches = cand
                used_threshold = th
                break
            th -= step

    topk = min(args.topk, len(matches))
    for i in range(topk):
        m = matches[i]
        meta = detector.get_template_meta(m.class_id, m.template_id)
        t0 = detector.get_templates(m.class_id, m.template_id)[0]
        angle = float(meta.get("angle", 0.0))
        scale = float(meta.get("scale", 1.0))
        print(
            f"[{i+1}] sim={m.similarity:.2f}, class={m.class_id}, x={m.x}, y={m.y}, "
            f"templ_id={m.template_id}, angle={angle:.2f}, scale={scale:.3f}, "
            f"w={t0.width}, h={t0.height}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    overlay = draw_matches(detector, scene, matches, topk=args.topk)
    bench["draw_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok = cv2.imwrite(str(out), overlay)
    bench["save_s"] = time.perf_counter() - t0
    if not ok:
        raise RuntimeError(f"Failed to write output: {out}")

    print(f"classes={class_ids}")
    print(f"threshold_used={used_threshold:.2f}")
    print(f"raw_matches={len(matches)}")
    print(f"saved={out}")
    if args.benchmark:
        total_s = time.perf_counter() - t_total_start
        print(f"benchmark.match_attempts={match_attempts}")
        print(f"benchmark.model_load_s={bench['model_load_s']:.6f}")
        print(f"benchmark.scene_read_s={bench['scene_read_s']:.6f}")
        print(f"benchmark.scene_mask_read_s={bench['scene_mask_read_s']:.6f}")
        print(f"benchmark.crop_s={bench['crop_s']:.6f}")
        print(f"benchmark.match_s={bench['match_s']:.6f}")
        print(f"benchmark.nms_s={bench['nms_s']:.6f}")
        print(f"benchmark.draw_s={bench['draw_s']:.6f}")
        print(f"benchmark.save_s={bench['save_s']:.6f}")
        print(f"benchmark.total_s={total_s:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
