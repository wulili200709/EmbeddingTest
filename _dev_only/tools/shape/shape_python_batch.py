#!/usr/bin/env python3
"""
Batch runner for Python shape pipeline, aligned to _third_party test.cpp logic.

Outputs:
  test/case0/result_shape_python/{1,2,3,4}.png
  test/case1/result_shape_python/result.png
  test/case2/result_shape_python/result.png
  test/shape_python/summary.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shape.like_matcher import (
    Line2DupLikeDetector,
    Match,
    ShapeInfoProducer,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class CaseRun:
    case: str
    success: bool
    output: str
    notes: str


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    else:
        import json

        text = json.dumps(data, indent=2, ensure_ascii=True)
    path.write_text(text, encoding="utf-8")


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def crop_to_stride(image: np.ndarray, stride: int) -> np.ndarray:
    h, w = image.shape[:2]
    n = h // stride
    m = w // stride
    if n <= 0 or m <= 0:
        return image.copy()
    return image[: n * stride, : m * stride].copy()


def nms_indices_xywh(boxes: Sequence[List[int]], scores: Sequence[float], iou: float) -> List[int]:
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, list(scores), score_threshold=0.0, nms_threshold=float(iou))
    if keep is None or len(keep) == 0:
        return []
    out: List[int] = []
    for idx in keep:
        if isinstance(idx, (list, tuple, np.ndarray)):
            out.append(int(idx[0]))
        else:
            out.append(int(idx))
    return sorted(set(out))


def put_text_visible(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 1.6,
) -> None:
    # Draw outline first to keep text readable on dark/bright backgrounds.
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_PLAIN, scale, color, 2, cv2.LINE_AA)


def draw_case0(
    detector: Line2DupLikeDetector,
    image: np.ndarray,
    matches: Sequence[Match],
    class_id: str,
    topk: int = 5,
) -> np.ndarray:
    out = image.copy()
    palette = [
        (255, 120, 255),
        (255, 220, 120),
        (120, 220, 255),
        (120, 255, 140),
        (255, 180, 120),
    ]
    draw_n = min(topk, len(matches))
    for i in range(draw_n):
        m = matches[i]
        templ = detector.get_templates(class_id, m.template_id)
        t0 = templ[0]
        x = t0.width // 2 + m.x
        y = t0.height // 2 + m.y
        r = max(1, t0.width // 2)
        color = palette[i % len(palette)]
        tx = int(m.x + r - 10)
        ty = max(22, int(m.y - 4))
        put_text_visible(out, str(int(round(m.similarity))), (tx, ty), color=color, scale=2.0)
        cv2.circle(out, (int(x), int(y)), int(r), color, 2, cv2.LINE_AA)
    return out


def draw_case1(
    detector: Line2DupLikeDetector,
    image: np.ndarray,
    matches: Sequence[Match],
    class_id: str,
    topk: int = 1,
) -> np.ndarray:
    out = image.copy()
    rng = np.random.default_rng(2024)
    draw_n = min(topk, len(matches))
    for i in range(draw_n):
        m = matches[i]
        templ = detector.get_templates(class_id, m.template_id)
        meta = detector.get_template_meta(class_id, m.template_id)
        angle = float(meta.get("angle", 0.0))
        scale = float(meta.get("scale", 1.0))

        r_scaled = 270.0 / 2.0 * scale
        train_img_half_width = 270.0 / 2.0 + 100.0
        train_img_half_height = 270.0 / 2.0 + 100.0
        x = float(m.x - templ[0].tl_x + train_img_half_width)
        y = float(m.y - templ[0].tl_y + train_img_half_height)

        color = tuple(int(x) for x in rng.integers(100, 255, size=3))
        for feat in templ[0].features:
            cv2.circle(out, (int(feat.x + m.x), int(feat.y + m.y)), 3, color, -1, cv2.LINE_AA)

        cv2.putText(
            out,
            str(int(round(m.similarity))),
            (int(m.x + r_scaled - 10), int(m.y - 3)),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            color,
            2,
            cv2.LINE_AA,
        )
        rr = ((x, y), (2.0 * r_scaled, 2.0 * r_scaled), -angle)
        pts = cv2.boxPoints(rr).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    return out


def draw_case2(
    detector: Line2DupLikeDetector,
    image: np.ndarray,
    matches: Sequence[Match],
    class_id: str,
) -> np.ndarray:
    out = image.copy()
    rng = np.random.default_rng(7)
    for m in matches:
        templ = detector.get_templates(class_id, m.template_id)
        t0 = templ[0]
        x2 = int(t0.width + m.x)
        y2 = int(t0.height + m.y)
        r = max(1, t0.width // 2)
        color = tuple(int(x) for x in rng.integers(100, 255, size=3))

        for feat in t0.features:
            cv2.circle(out, (int(feat.x + m.x), int(feat.y + m.y)), 2, color, -1, cv2.LINE_AA)

        cv2.putText(
            out,
            str(int(round(m.similarity))),
            (int(m.x + r - 10), int(m.y - 3)),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(out, (int(m.x), int(m.y)), (x2, y2), color, 2, cv2.LINE_AA)
    return out


def case2_nms_matches(
    detector: Line2DupLikeDetector,
    matches: Sequence[Match],
    class_id: str,
    nms_iou: float,
) -> List[Match]:
    if not matches:
        return []
    boxes: List[List[int]] = []
    scores: List[float] = []
    for m in matches:
        templ = detector.get_templates(class_id, m.template_id)
        boxes.append([int(m.x), int(m.y), int(max(1, templ[0].width)), int(max(1, templ[0].height))])
        scores.append(float(m.similarity))
    keep = nms_indices_xywh(boxes, scores, iou=nms_iou)
    out = [matches[i] for i in keep if 0 <= i < len(matches)]
    out.sort(key=lambda m: m.similarity, reverse=True)
    return out


def run_case0(test_root: Path) -> CaseRun:
    case_dir = test_root / "case0"
    out_dir = case_dir / "result_shape_python"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Line2DupLikeDetector(num_features=150, T_levels=(4, 8))
    class_id = "circle"

    img = read_bgr(case_dir / "templ" / "circle.png")
    shapes = ShapeInfoProducer(img, None)
    shapes.scale_range = [0.1, 1.0]
    shapes.scale_step = 0.01
    shapes.angle_range = [0.0]
    infos = shapes.produce_infos()

    templ_ok = 0
    for info in infos:
        src_i = shapes.src_of(info)
        mask_i = shapes.mask_of(info)
        nfeat = max(1, int(round(150 * info.scale)))
        templ_id = detector.add_template(
            src_i,
            class_id=class_id,
            object_mask=mask_i,
            num_features=nfeat,
            metadata={"angle": float(info.angle), "scale": float(info.scale)},
        )
        if templ_id >= 0:
            templ_ok += 1

    if templ_ok == 0:
        return CaseRun("case0", False, str(out_dir), "no template extracted")

    scenes = ["1.jpg", "2.jpg", "3.png", "4.png"]
    used_thresholds: Dict[str, float] = {}
    for scene_file in scenes:
        test_img = read_bgr(case_dir / scene_file)
        img_crop = crop_to_stride(test_img, stride=32)
        matches: List[Match] = []
        matched_threshold = 90.0
        for th in (90.0, 85.0, 80.0, 75.0, 70.0):
            cand = detector.match(img_crop, threshold=th, class_ids=[class_id])
            if cand:
                matches = cand
                matched_threshold = th
                break
        matches.sort(key=lambda m: m.similarity, reverse=True)
        vis = draw_case0(detector, img_crop, matches, class_id=class_id, topk=5)
        if not matches:
            put_text_visible(vis, "NO MATCH", (12, 28), (180, 180, 255), scale=2.0)
        stem = Path(scene_file).stem
        cv2.imwrite(str(out_dir / f"{stem}.png"), vis)
        used_thresholds[scene_file] = matched_threshold

    return CaseRun("case0", True, str(out_dir), f"templates={templ_ok}, thresholds={used_thresholds}")


def run_case1(test_root: Path) -> CaseRun:
    case_dir = test_root / "case1"
    out_dir = case_dir / "result_shape_python"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Line2DupLikeDetector(num_features=128, T_levels=(4, 8))
    class_id = "test"

    train = read_bgr(case_dir / "train.png")
    roi = train[110 : 110 + 270, 130 : 130 + 270].copy()
    mask = np.full(roi.shape[:2], 255, dtype=np.uint8)

    padding = 100
    padded_img = np.zeros((roi.shape[0] + 2 * padding, roi.shape[1] + 2 * padding, 3), dtype=roi.dtype)
    padded_img[padding : padding + roi.shape[0], padding : padding + roi.shape[1]] = roi
    padded_mask = np.zeros((mask.shape[0] + 2 * padding, mask.shape[1] + 2 * padding), dtype=np.uint8)
    padded_mask[padding : padding + mask.shape[0], padding : padding + mask.shape[1]] = mask

    shapes = ShapeInfoProducer(padded_img, padded_mask)
    shapes.angle_range = [0.0, 360.0]
    shapes.angle_step = 1.0
    shapes.scale_range = [1.0]
    infos = shapes.produce_infos()

    first_id = -1
    first_angle = 0.0
    is_first = True
    templ_ok = 0
    for info in infos:
        if is_first:
            templ_id = detector.add_template(
                shapes.src_of(info),
                class_id=class_id,
                object_mask=shapes.mask_of(info),
                metadata={"angle": float(info.angle), "scale": float(info.scale)},
            )
            if templ_id >= 0:
                first_id = templ_id
                first_angle = float(info.angle)
                is_first = False
                templ_ok += 1
            continue

        templ_id = detector.add_template_rotate(
            class_id=class_id,
            zero_id=first_id,
            theta_deg=float(info.angle - first_angle),
            center=(shapes.src.shape[1] / 2.0, shapes.src.shape[0] / 2.0),
            metadata={"angle": float(info.angle), "scale": float(info.scale)},
        )
        if templ_id >= 0:
            templ_ok += 1

    if templ_ok == 0:
        return CaseRun("case1", False, str(out_dir), "no template extracted")

    test_img = read_bgr(case_dir / "test.png")
    pad = 250
    padded_scene = np.zeros((test_img.shape[0] + 2 * pad, test_img.shape[1] + 2 * pad, 3), dtype=test_img.dtype)
    padded_scene[pad : pad + test_img.shape[0], pad : pad + test_img.shape[1]] = test_img
    scene = crop_to_stride(padded_scene, stride=16)

    matches = detector.match(scene, threshold=90.0, class_ids=[class_id])
    matches.sort(key=lambda m: m.similarity, reverse=True)
    vis = draw_case1(detector, scene, matches, class_id=class_id, topk=1)
    cv2.imwrite(str(out_dir / "result.png"), vis)
    return CaseRun("case1", True, str(out_dir), f"templates={templ_ok}, matches={len(matches)}")


def run_case2(test_root: Path) -> CaseRun:
    case_dir = test_root / "case2"
    out_dir = case_dir / "result_shape_python"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Line2DupLikeDetector(num_features=30, T_levels=(4, 8))
    class_id = "test"

    train = read_bgr(case_dir / "train.png")
    mask = np.full(train.shape[:2], 255, dtype=np.uint8)

    shapes = ShapeInfoProducer(train, mask)
    shapes.angle_range = [0.0, 360.0]
    shapes.angle_step = 1.0
    shapes.scale_range = [1.0]
    infos = shapes.produce_infos()

    templ_ok = 0
    for info in infos:
        templ_id = detector.add_template(
            shapes.src_of(info),
            class_id=class_id,
            object_mask=shapes.mask_of(info),
            metadata={"angle": float(info.angle), "scale": float(info.scale)},
        )
        if templ_id >= 0:
            templ_ok += 1

    if templ_ok == 0:
        return CaseRun("case2", False, str(out_dir), "no template extracted")

    test_img = read_bgr(case_dir / "test.png")
    scene = crop_to_stride(test_img, stride=16)
    threshold_try = [90.0, 88.0, 85.0, 80.0, 75.0]
    used_th = threshold_try[0]
    best_kept: List[Match] = []
    raw_best = 0
    for th in threshold_try:
        raw = detector.match(scene, threshold=th, class_ids=[class_id])
        raw.sort(key=lambda m: m.similarity, reverse=True)
        kept = case2_nms_matches(detector, raw, class_id=class_id, nms_iou=0.5)
        if len(kept) > len(best_kept):
            best_kept = kept
            used_th = th
            raw_best = len(raw)
        if len(kept) >= 4:
            best_kept = kept
            used_th = th
            raw_best = len(raw)
            break

    matches = best_kept[:4] if len(best_kept) >= 4 else best_kept
    vis = draw_case2(detector, scene, matches, class_id=class_id)
    cv2.imwrite(str(out_dir / "result.png"), vis)
    return CaseRun(
        "case2",
        True,
        str(out_dir),
        f"templates={templ_ok}, threshold={used_th}, raw={raw_best}, matches={len(matches)}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Python shape pipeline for case0/1/2.")
    parser.add_argument("--root", default=".", help="Repo root path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    test_root = root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Cannot find test dir: {test_root}")

    runs = [
        run_case0(test_root),
        run_case1(test_root),
        run_case2(test_root),
    ]
    ok = sum(1 for r in runs if r.success)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "python_shape",
        "runs": [r.__dict__ for r in runs],
    }
    dump_yaml(test_root / "shape_python" / "summary.yaml", summary)
    print(f"shape_python_batch done: {ok}/{len(runs)} cases succeeded.")
    print(f"summary: {test_root / 'shape_python' / 'summary.yaml'}")
    return 0 if ok == len(runs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
