from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from shape_model_like import ScaledShapeModel


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _make_synthetic() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # template: a simple shape with strong edges
    tpl = np.zeros((180, 220), np.uint8)
    cv2.rectangle(tpl, (40, 40), (180, 140), 255, -1)
    cv2.circle(tpl, (80, 90), 18, 0, -1)
    cv2.circle(tpl, (140, 90), 10, 0, -1)
    tpl = cv2.GaussianBlur(tpl, (3, 3), 0)

    mask = (tpl > 0).astype(np.uint8) * 255

    # search image: place a rotated+scaled instance
    img = np.zeros((480, 640), np.uint8)
    angle_deg = 27.0
    scale = 1.18
    center = (tpl.shape[1] / 2, tpl.shape[0] / 2)
    # Warp into a larger canvas to avoid clipping when scale>1 and rotation is applied.
    out_w = int(round(tpl.shape[1] * scale * 1.6))
    out_h = int(round(tpl.shape[0] * scale * 1.6))
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += (out_w / 2) - center[0]
    M[1, 2] += (out_h / 2) - center[1]
    warped = cv2.warpAffine(tpl, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
    # paste
    r0, c0 = 80, 160
    rr, cc = warped.shape
    img[r0 : r0 + rr, c0 : c0 + cc] = np.maximum(img[r0 : r0 + rr, c0 : c0 + cc], warped)
    # add mild noise
    noise = (np.random.default_rng(0).normal(0, 6, img.shape)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return tpl, mask, img


def _draw_match(
    out_bgr: np.ndarray,
    row: float,
    col: float,
    angle: float,
    scale: float,
    base_wh: Tuple[float, float],
    color=(0, 255, 0),
) -> None:
    w0, h0 = base_wh
    rect = ((float(col), float(row)), (float(w0 * scale), float(h0 * scale)), float(math.degrees(angle)))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(out_bgr, [box], True, color, 2, cv2.LINE_AA)
    cv2.circle(out_bgr, (int(round(col)), int(round(row))), 4, (0, 0, 255), -1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=str, default=None, help="模板图路径（不填则用合成示例）")
    ap.add_argument("--image", type=str, default=None, help="搜索图路径（不填则用合成示例）")
    ap.add_argument("--mask", type=str, default=None, help="模板 mask/ROI（可选）")
    ap.add_argument("--model", type=str, default=None, help="保存/加载模型的 npz 路径（可选）")
    ap.add_argument("--out", type=str, default="match_result.png", help="输出可视化图片")

    ap.add_argument("--angle_start", type=float, default=-0.8)
    ap.add_argument("--angle_extent", type=float, default=1.6)
    ap.add_argument("--scale_min", type=float, default=0.8)
    ap.add_argument("--scale_max", type=float, default=1.3)
    # 注意：本实现的 score 是“边缘点命中比例”的近似值，通常比 HALCON 的 score 偏低一些。
    ap.add_argument("--min_score", type=float, default=0.18)
    ap.add_argument("--num_matches", type=int, default=3)
    ap.add_argument("--max_overlap", type=float, default=0.3)
    ap.add_argument("--num_levels", type=int, default=0)
    ap.add_argument("--greediness", type=float, default=0.9)
    ap.add_argument("--subpixel", type=str, default="interpolation", choices=["none", "interpolation"])
    args = ap.parse_args()

    if args.template is None or args.image is None:
        print("[1/3] use synthetic sample ...", flush=True)
        tpl, mask, img = _make_synthetic()
        tpl_path = None
        img_path = None
        mask_u8 = mask
    else:
        print("[1/3] load images ...", flush=True)
        tpl_path = Path(args.template)
        img_path = Path(args.image)
        tpl = _read_gray(tpl_path)
        img = _read_gray(img_path)
        mask_u8 = None
        if args.mask:
            mask_u8 = _read_gray(Path(args.mask))

    model_path: Optional[Path] = Path(args.model) if args.model else None

    t0 = time.perf_counter()
    if model_path and model_path.exists():
        print("[2/3] load model ...", flush=True)
        model = ScaledShapeModel.load(model_path)
    else:
        print("[2/3] create model ...", flush=True)
        model = ScaledShapeModel.create(tpl, mask=mask_u8)
        if model_path:
            model.save(model_path)
            print(f"      model saved: {model_path}", flush=True)
    t_model = time.perf_counter() - t0

    print("[3/3] find in scene ...", flush=True)
    t1 = time.perf_counter()
    rows, cols, angs, scs, scores = model.find(
        img,
        angle_start=args.angle_start,
        angle_extent=args.angle_extent,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        min_score=args.min_score,
        num_matches=args.num_matches,
        max_overlap=args.max_overlap,
        subpixel=args.subpixel,
        num_levels=args.num_levels,
        greediness=args.greediness,
    )
    t_find = time.perf_counter() - t1

    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(scores)):
        _draw_match(out, float(rows[i]), float(cols[i]), float(angs[i]), float(scs[i]), model.base_size_wh)
        cv2.putText(
            out,
            f"{i}: score={scores[i]:.3f} a={angs[i]:.3f} s={scs[i]:.3f}",
            (10, 25 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    out_path = Path(args.out)
    cv2.imwrite(str(out_path), out)

    print("matches =", len(scores))
    print(f"time: model={t_model:.3f}s find={t_find:.3f}s total={(t_model+t_find):.3f}s")
    for i in range(len(scores)):
        print(
            f"[{i}] row={rows[i]:.2f} col={cols[i]:.2f} angle(rad)={angs[i]:.4f} scale={scs[i]:.4f} score={scores[i]:.4f}"
        )
    print("saved =", str(out_path))


if __name__ == "__main__":
    main()
