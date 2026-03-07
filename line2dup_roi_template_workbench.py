#!/usr/bin/env python3
"""
ROI-based template workbench for line2Dup-like workflow.

Flow:
1) Open image and select ROI.
2) Auto-generate feature points from current parameters.
3) Tune parameters and edit points (add/delete) visually.
4) Save model JSON for matching.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from line2dup_like_matcher import (
    Feature,
    Line2DupLikeDetector,
    TemplateLevel,
    label_to_theta_deg as matcher_label_to_theta_deg,
    save_detector_model,
    theta_deg_to_label as matcher_theta_deg_to_label,
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


def label_to_angle_deg(label: int) -> float:
    return float(matcher_label_to_theta_deg(label))


def angle_deg_to_label(theta_deg: float) -> int:
    return int(matcher_theta_deg_to_label(theta_deg))


def arrow_endpoint(x: int, y: int, theta_deg: float, length: float) -> Tuple[int, int]:
    rad = np.deg2rad(theta_deg)
    ex = int(round(x + length * float(np.cos(rad))))
    ey = int(round(y + length * float(np.sin(rad))))
    return ex, ey


def clone_levels(levels: List[TemplateLevel]) -> List[TemplateLevel]:
    out: List[TemplateLevel] = []
    for lv in levels:
        feats = [Feature(x=int(f.x), y=int(f.y), label=int(f.label), theta=float(f.theta)) for f in lv.features]
        out.append(
            TemplateLevel(
                width=int(lv.width),
                height=int(lv.height),
                tl_x=int(lv.tl_x),
                tl_y=int(lv.tl_y),
                pyramid_level=int(lv.pyramid_level),
                features=feats,
            )
        )
    return out


@dataclass
class RoiRect:
    x: int
    y: int
    w: int
    h: int


class WorkbenchState:
    def __init__(
        self,
        image_bgr: np.ndarray,
        roi: RoiRect,
        class_id: str,
        levels: List[int],
        out_model: Path,
        num_features: int,
        weak_thresh: float,
        strong_thresh: float,
        zoom: float,
    ) -> None:
        self.image = image_bgr
        self.roi = roi
        self.roi_img = image_bgr[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w].copy()
        self.roi_mask = np.full((roi.h, roi.w), 255, dtype=np.uint8)

        self.class_id = class_id
        self.levels = levels
        self.out_model = out_model

        self.num_features = int(max(8, num_features))
        self.weak_thresh = float(max(1.0, weak_thresh))
        self.strong_thresh = float(max(1.0, strong_thresh))
        self.current_level = 0
        self.current_label = 0
        self.zoom = float(max(1.0, zoom))
        self.delete_radius = 10.0

        self.template_levels: List[TemplateLevel] = []
        self.need_recompute = True
        self.last_error = ""
        self.dirty = False
        self.show_help = False
        self.show_status = False
        self.hover_index: Optional[int] = None
        self.is_dragging = False
        self.drag_start_abs: Optional[Tuple[int, int]] = None
        self.drag_end_abs: Optional[Tuple[int, int]] = None

    def current_level_template(self) -> Optional[TemplateLevel]:
        if not self.template_levels:
            return None
        idx = max(0, min(self.current_level, len(self.template_levels) - 1))
        return self.template_levels[idx]

    def recompute(self) -> bool:
        detector = Line2DupLikeDetector(
            num_features=self.num_features,
            T_levels=self.levels,
            weak_threshold=self.weak_thresh,
            strong_threshold=self.strong_thresh,
        )
        tid = detector.add_template(
            self.roi_img,
            class_id=self.class_id,
            object_mask=self.roi_mask,
            num_features=self.num_features,
            metadata={"angle": 0.0, "scale": 1.0},
        )
        if tid < 0:
            self.last_error = "No template extracted. Try lower strong-thresh or weak-thresh."
            return False

        self.template_levels = clone_levels(detector.get_templates(self.class_id, tid))
        self.current_level = max(0, min(self.current_level, len(self.template_levels) - 1))
        self.hover_index = None
        self.last_error = ""
        self.need_recompute = False
        self.dirty = True
        return True

    def get_background(self) -> np.ndarray:
        img = self.roi_img.copy()
        for _ in range(self.current_level):
            img = cv2.pyrDown(img)
        return img

    def get_abs_points(self) -> List[Tuple[int, int]]:
        tl = self.current_level_template()
        if tl is None:
            return []
        pts: List[Tuple[int, int]] = []
        for f in tl.features:
            pts.append((int(f.x + tl.tl_x), int(f.y + tl.tl_y)))
        return pts

    def update_hover(self, x_abs: int, y_abs: int) -> None:
        tl = self.current_level_template()
        if tl is None or not tl.features:
            self.hover_index = None
            return
        pts = self.get_abs_points()
        best_idx = -1
        best_d2 = 1e18
        for i, (px, py) in enumerate(pts):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx >= 0 and best_d2 <= self.delete_radius * self.delete_radius:
            self.hover_index = best_idx
        else:
            self.hover_index = None

    def add_point(self, x_abs: int, y_abs: int, label: int, theta_deg: Optional[float] = None) -> bool:
        tl = self.current_level_template()
        if tl is None:
            return False
        x_rel = int(round(x_abs - tl.tl_x))
        y_rel = int(round(y_abs - tl.tl_y))
        x_rel = max(0, min(x_rel, int(tl.width)))
        y_rel = max(0, min(y_rel, int(tl.height)))
        lb = int(label) % 8
        theta = label_to_angle_deg(lb) if theta_deg is None else float(theta_deg)
        tl.features.append(Feature(x=x_rel, y=y_rel, label=lb, theta=theta))
        self.dirty = True
        return True

    def delete_point(self, x_abs: int, y_abs: int) -> bool:
        tl = self.current_level_template()
        if tl is None or not tl.features:
            return False
        pts = self.get_abs_points()
        best_idx = -1
        best_d2 = 1e18
        for i, (px, py) in enumerate(pts):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx < 0 or best_d2 > self.delete_radius * self.delete_radius:
            return False
        del tl.features[best_idx]
        self.hover_index = None
        self.dirty = True
        return True

    def set_zoom(self, z: float) -> None:
        self.zoom = max(1.0, min(16.0, float(z)))

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self.zoom * float(factor))

    def save_model(self) -> None:
        if not self.template_levels:
            raise RuntimeError("No template levels to save.")

        detector = Line2DupLikeDetector(
            num_features=self.num_features,
            T_levels=self.levels,
            weak_threshold=self.weak_thresh,
            strong_threshold=self.strong_thresh,
        )
        detector.class_templates = {self.class_id: [clone_levels(self.template_levels)]}
        detector.class_meta = {
            self.class_id: [
                {
                    "angle": 0.0,
                    "scale": 1.0,
                    "roi_x": int(self.roi.x),
                    "roi_y": int(self.roi.y),
                    "roi_w": int(self.roi.w),
                    "roi_h": int(self.roi.h),
                }
            ]
        }
        save_detector_model(detector, str(self.out_model))

        # Save preview image for quick verification.
        preview = self.render()
        preview_path = self.out_model.with_name(f"{self.out_model.stem}_preview.png")
        cv2.imwrite(str(preview_path), preview)
        self.dirty = False

    def render(self) -> np.ndarray:
        bg = self.get_background()
        canvas = bg.copy()
        tl = self.current_level_template()

        if tl is not None:
            x1 = int(tl.tl_x)
            y1 = int(tl.tl_y)
            x2 = int(tl.tl_x + tl.width)
            y2 = int(tl.tl_y + tl.height)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

            palette = orientation_palette_bgr()
            pts = self.get_abs_points()
            for i, (px, py) in enumerate(pts):
                f = tl.features[i]
                color = palette[int(f.label) % len(palette)]
                theta = float(f.theta) if np.isfinite(f.theta) else label_to_angle_deg(int(f.label))
                p2 = arrow_endpoint(int(px), int(py), theta, length=8.0)
                if self.hover_index is not None and i == self.hover_index:
                    cv2.circle(canvas, (int(px), int(py)), 7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.arrowedLine(
                    canvas,
                    (int(px), int(py)),
                    p2,
                    color,
                    1,
                    cv2.LINE_AA,
                    0,
                    0.35,
                )
                cv2.circle(canvas, (int(px), int(py)), 1, color, -1, cv2.LINE_AA)

        if self.is_dragging and self.drag_start_abs is not None and self.drag_end_abs is not None:
            sx, sy = self.drag_start_abs
            ex, ey = self.drag_end_abs
            dx = ex - sx
            dy = ey - sy
            if abs(dx) + abs(dy) > 0:
                theta = float(np.degrees(np.arctan2(float(dy), float(dx))))
                lb = angle_deg_to_label(theta)
            else:
                lb = int(self.current_label) % 8
            color = orientation_palette_bgr()[lb]
            cv2.arrowedLine(canvas, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA, 0, 0.35)

        if self.show_status:
            status = (
                f"roi=({self.roi.x},{self.roi.y},{self.roi.w},{self.roi.h})  "
                f"L={self.current_level}/{max(0, len(self.levels)-1)}  "
                f"points={0 if tl is None else len(tl.features)}  "
                f"label={self.current_label}  zoom={self.zoom:.2f}x  "
                f"nf={self.num_features} weak={self.weak_thresh:.1f} strong={self.strong_thresh:.1f}"
            )
            cv2.putText(canvas, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 240, 80), 1, cv2.LINE_AA)

        if self.show_help:
            lines = [
                "L-drag:add with direction  R-click:delete  Wheel/+- zoom  0 reset",
                "Trackbars auto-recompute. s save, r recompute, h help, i status, q quit",
            ]
            y = 48
            for t in lines:
                cv2.putText(canvas, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                y += 20

        if self.last_error and (self.show_help or self.show_status):
            cv2.putText(canvas, self.last_error, (8, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)

        if self.zoom != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)
        return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI template workbench: select ROI, tune params, edit points, save model.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out-model", required=True, help="Output model path (json).")
    parser.add_argument("--class-id", default="object", help="Class id stored in model.")
    parser.add_argument("--levels", default="4,8", help="Pyramid levels, e.g. 4,8")
    parser.add_argument("--num-features", type=int, default=128, help="Initial feature count.")
    parser.add_argument("--weak-thresh", type=float, default=30.0, help="Initial weak threshold.")
    parser.add_argument("--strong-thresh", type=float, default=60.0, help="Initial strong threshold.")
    parser.add_argument("--zoom", type=float, default=2.0, help="Initial display zoom.")
    parser.add_argument("--window", default="line2dup_roi_workbench", help="Window title.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    select_win = "Select ROI (ENTER to confirm, c to cancel)"
    roi_tuple = cv2.selectROI(select_win, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(select_win)
    x, y, w, h = [int(v) for v in roi_tuple]
    if w <= 0 or h <= 0:
        print("ROI canceled.")
        return 1

    levels = parse_levels(args.levels)
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    state = WorkbenchState(
        image_bgr=image,
        roi=RoiRect(x=x, y=y, w=w, h=h),
        class_id=args.class_id,
        levels=levels,
        out_model=out_model,
        num_features=args.num_features,
        weak_thresh=args.weak_thresh,
        strong_thresh=args.strong_thresh,
        zoom=args.zoom,
    )

    win = args.window
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def change_zoom(factor: float) -> None:
        old = state.zoom
        state.zoom_by(factor)
        if abs(old - state.zoom) > 1e-6:
            print(f"zoom={state.zoom:.2f}x")

    # Trackbars
    cv2.createTrackbar("num_features", win, int(max(8, min(400, state.num_features))), 400, lambda _v: None)
    cv2.createTrackbar("weak_thresh", win, int(max(1, min(100, round(state.weak_thresh)))), 100, lambda _v: None)
    cv2.createTrackbar("strong_thresh", win, int(max(1, min(120, round(state.strong_thresh)))), 120, lambda _v: None)
    cv2.createTrackbar("level", win, int(state.current_level), max(0, len(levels) - 1), lambda _v: None)
    cv2.createTrackbar("label", win, int(state.current_label), 7, lambda _v: None)

    def sync_from_trackbars() -> None:
        nf = max(8, int(cv2.getTrackbarPos("num_features", win)))
        wk = max(1.0, float(cv2.getTrackbarPos("weak_thresh", win)))
        st = max(1.0, float(cv2.getTrackbarPos("strong_thresh", win)))
        lv = int(cv2.getTrackbarPos("level", win))
        lb = int(cv2.getTrackbarPos("label", win)) % 8

        if nf != state.num_features or abs(wk - state.weak_thresh) > 1e-9 or abs(st - state.strong_thresh) > 1e-9:
            state.num_features = nf
            state.weak_thresh = wk
            state.strong_thresh = st
            state.need_recompute = True
        state.current_level = max(0, min(lv, len(levels) - 1))
        state.current_label = lb

    def on_mouse(event: int, xw: int, yw: int, flags: int, _userdata) -> None:
        x_abs = int(round(xw / state.zoom))
        y_abs = int(round(yw / state.zoom))
        if event == cv2.EVENT_MOUSEMOVE:
            state.update_hover(x_abs, y_abs)
            if state.is_dragging:
                state.drag_end_abs = (x_abs, y_abs)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state.is_dragging = True
            state.drag_start_abs = (x_abs, y_abs)
            state.drag_end_abs = (x_abs, y_abs)
        elif event == cv2.EVENT_LBUTTONUP:
            if state.is_dragging and state.drag_start_abs is not None and state.drag_end_abs is not None:
                sx, sy = state.drag_start_abs
                ex, ey = state.drag_end_abs
                dx = ex - sx
                dy = ey - sy
                dist = float(np.hypot(dx, dy))
                if dist >= 2.0:
                    theta = float(np.degrees(np.arctan2(float(dy), float(dx))))
                    lb = angle_deg_to_label(theta)
                    state.current_label = lb
                    cv2.setTrackbarPos("label", win, lb)
                    state.add_point(sx, sy, lb, theta_deg=theta)
                else:
                    lb = int(state.current_label) % 8
                    state.add_point(sx, sy, lb, theta_deg=label_to_angle_deg(lb))
            state.is_dragging = False
            state.drag_start_abs = None
            state.drag_end_abs = None
            state.update_hover(x_abs, y_abs)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.delete_point(x_abs, y_abs)
            state.update_hover(x_abs, y_abs)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 0
            if hasattr(cv2, "getMouseWheelDelta"):
                try:
                    delta = int(cv2.getMouseWheelDelta(flags))
                except Exception:
                    delta = 0
            if delta == 0:
                delta = 1 if flags > 0 else -1
            if delta > 0:
                change_zoom(1.15)
            elif delta < 0:
                change_zoom(1.0 / 1.15)

    cv2.setMouseCallback(win, on_mouse)

    print("Workbench started.")
    print(f"image={args.image}")
    print(f"roi=(x={x}, y={y}, w={w}, h={h})")
    print(f"out_model={out_model}")
    print("Adjust trackbars to auto-recompute points.")
    print("Mouse: left-drag add with direction, right click delete. Press 'h' for help, 'i' for status.")

    while True:
        sync_from_trackbars()
        if state.need_recompute:
            state.recompute()

        canvas = state.render()
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in (27, ord("q")):
            break
        if key == ord("h"):
            state.show_help = not state.show_help
        elif key == ord("i"):
            state.show_status = not state.show_status
        elif key == ord("r"):
            state.need_recompute = True
        elif key == ord("s"):
            state.save_model()
            print(f"saved: {out_model}")
        elif key in (ord("+"), ord("="), 43, 61, 107, 171):
            change_zoom(1.2)
        elif key in (ord("-"), ord("_"), 45, 95, 109, 173):
            change_zoom(1.0 / 1.2)
        elif key == ord("0"):
            old = state.zoom
            state.set_zoom(1.0)
            if abs(state.zoom - old) > 1e-6:
                print(f"zoom={state.zoom:.2f}x")

    cv2.destroyAllWindows()
    if state.dirty:
        print("Unsaved edits exist. Press 's' before quitting next time if you need them.")
    print("Exit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
