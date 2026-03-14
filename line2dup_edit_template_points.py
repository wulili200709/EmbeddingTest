#!/usr/bin/env python3
"""
Interactive visual editor for line2Dup-like template feature points.

Mouse:
- Left drag: add a point and infer direction label from drag angle.
- Right click: delete nearest point within radius.
- Right drag: box-select and delete all points inside the box.
- Wheel: zoom in/out.

Keyboard:
- s: save model
- q / ESC: quit
- c / x: next / prev class
- n / p: next / prev template id
- l / k: next / prev pyramid level
- ] / [: next / prev label for new points
- + / -: zoom in / out
- 0: reset zoom to 1x
- u: undo last edit
- h: toggle help text
- i: toggle status text
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
    ShapeInfoProducer,
    label_to_theta_deg as matcher_label_to_theta_deg,
    load_detector_model,
    save_detector_model,
    theta_deg_to_label as matcher_theta_deg_to_label,
)


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


def drag_is_click(start: Tuple[int, int], end: Tuple[int, int], threshold: int = 3) -> bool:
    return abs(int(end[0]) - int(start[0])) < threshold and abs(int(end[1]) - int(start[1])) < threshold


def normalize_drag_rect(start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[int, int, int, int]:
    return (
        min(int(start[0]), int(end[0])),
        min(int(start[1]), int(end[1])),
        max(int(start[0]), int(end[0])),
        max(int(start[1]), int(end[1])),
    )


@dataclass
class UndoItem:
    class_id: str
    template_id: int
    level: int
    features: List[Feature]


class EditorState:
    def __init__(
        self,
        model_path: Path,
        out_model_path: Path,
        template_image: Optional[np.ndarray],
        start_class: str,
        start_template_id: int,
        start_level: int,
        zoom: float,
    ) -> None:
        self.model_path = model_path
        self.out_model_path = out_model_path
        self.detector = load_detector_model(str(model_path))
        self.template_image = template_image

        self.class_ids = self.detector.class_ids()
        if not self.class_ids:
            raise RuntimeError("No classes in model.")

        if start_class and start_class in self.class_ids:
            self.class_index = self.class_ids.index(start_class)
        else:
            self.class_index = 0

        self.template_id = 0
        self.level = 0
        self.set_template_level(start_template_id, start_level)

        self.current_label = 0
        self.zoom = max(1.0, float(zoom))
        self.delete_radius = 8.0
        self.hover_index: Optional[int] = None
        self.undo_stack: List[UndoItem] = []
        self.dirty = False
        self.show_help = False
        self.show_status = False
        self.is_dragging = False
        self.drag_start_abs: Optional[Tuple[int, int]] = None
        self.drag_end_abs: Optional[Tuple[int, int]] = None
        self.is_erase_dragging = False
        self.erase_start_abs: Optional[Tuple[int, int]] = None
        self.erase_end_abs: Optional[Tuple[int, int]] = None

    @property
    def class_id(self) -> str:
        return self.class_ids[self.class_index]

    def templates_count(self) -> int:
        return len(self.detector.class_templates[self.class_id])

    def levels_count(self) -> int:
        return self.detector.pyramid_levels

    def set_template_level(self, template_id: int, level: int) -> None:
        tcnt = self.templates_count()
        if tcnt <= 0:
            self.template_id = 0
        else:
            self.template_id = max(0, min(int(template_id), tcnt - 1))
        self.level = max(0, min(int(level), self.levels_count() - 1))
        self.hover_index = None

    def next_class(self, step: int) -> None:
        n = len(self.class_ids)
        self.class_index = (self.class_index + step) % n
        self.set_template_level(0, self.level)

    def next_template(self, step: int) -> None:
        tcnt = self.templates_count()
        if tcnt <= 0:
            return
        self.template_id = (self.template_id + step) % tcnt
        self.hover_index = None

    def next_level(self, step: int) -> None:
        cnt = self.levels_count()
        self.level = (self.level + step) % cnt
        self.hover_index = None

    def current_template_level(self):
        return self.detector.get_templates(self.class_id, self.template_id)[self.level]

    def current_meta(self) -> dict:
        return self.detector.get_template_meta(self.class_id, self.template_id)

    def get_feature_abs_points(self) -> List[Tuple[int, int]]:
        tl = self.current_template_level()
        pts: List[Tuple[int, int]] = []
        for f in tl.features:
            pts.append((int(f.x + tl.tl_x), int(f.y + tl.tl_y)))
        return pts

    def push_undo(self) -> None:
        tl = self.current_template_level()
        snapshot = [Feature(x=int(f.x), y=int(f.y), label=int(f.label), theta=float(f.theta)) for f in tl.features]
        self.undo_stack.append(UndoItem(self.class_id, self.template_id, self.level, snapshot))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        item = self.undo_stack.pop()
        # Restore exact slot if still valid.
        if item.class_id not in self.detector.class_templates:
            return False
        if item.template_id < 0 or item.template_id >= len(self.detector.class_templates[item.class_id]):
            return False
        if item.level < 0 or item.level >= self.detector.pyramid_levels:
            return False
        self.detector.class_templates[item.class_id][item.template_id][item.level].features = item.features
        self.detector.invalidate_native_cache(item.class_id)
        if item.class_id in self.class_ids:
            self.class_index = self.class_ids.index(item.class_id)
        self.template_id = item.template_id
        self.level = item.level
        self.hover_index = None
        self.dirty = True
        return True

    def add_feature_abs(self, x_abs: int, y_abs: int, label: int, theta_deg: Optional[float] = None) -> None:
        tl = self.current_template_level()
        x_rel = int(round(x_abs - tl.tl_x))
        y_rel = int(round(y_abs - tl.tl_y))
        x_rel = max(0, min(x_rel, int(tl.width)))
        y_rel = max(0, min(y_rel, int(tl.height)))
        self.push_undo()
        lb = int(label) % 8
        theta = label_to_angle_deg(lb) if theta_deg is None else float(theta_deg)
        tl.features.append(Feature(x=x_rel, y=y_rel, label=lb, theta=theta))
        self.detector.invalidate_native_cache(self.class_id)
        self.dirty = True

    def delete_nearest_abs(self, x_abs: int, y_abs: int) -> bool:
        tl = self.current_template_level()
        if not tl.features:
            return False
        pts = self.get_feature_abs_points()
        best_idx = -1
        best_d2 = 1e18
        for i, (px, py) in enumerate(pts):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx < 0:
            return False
        if best_d2 > self.delete_radius * self.delete_radius:
            return False
        self.push_undo()
        del tl.features[best_idx]
        self.detector.invalidate_native_cache(self.class_id)
        self.hover_index = None
        self.dirty = True
        return True

    def delete_box_abs(self, start_abs: Tuple[int, int], end_abs: Tuple[int, int]) -> int:
        tl = self.current_template_level()
        if not tl.features:
            return 0
        x1, y1, x2, y2 = normalize_drag_rect(start_abs, end_abs)
        pts = self.get_feature_abs_points()
        keep: List[Feature] = []
        deleted = 0
        for i, feat in enumerate(tl.features):
            px, py = pts[i]
            if x1 <= int(px) <= x2 and y1 <= int(py) <= y2:
                deleted += 1
                continue
            keep.append(feat)
        if deleted <= 0:
            return 0
        self.push_undo()
        tl.features = keep
        self.detector.invalidate_native_cache(self.class_id)
        self.hover_index = None
        self.dirty = True
        return deleted

    def update_hover_abs(self, x_abs: int, y_abs: int) -> None:
        tl = self.current_template_level()
        if not tl.features:
            self.hover_index = None
            return
        pts = self.get_feature_abs_points()
        best_idx = -1
        best_d2 = 1e18
        for i, (px, py) in enumerate(pts):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx >= 0 and best_d2 <= (self.delete_radius * self.delete_radius):
            self.hover_index = best_idx
        else:
            self.hover_index = None

    def save(self) -> None:
        save_detector_model(self.detector, str(self.out_model_path))
        self.dirty = False

    def set_zoom(self, z: float) -> None:
        self.zoom = max(1.0, min(16.0, float(z)))

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self.zoom * float(factor))

    def render_background(self) -> np.ndarray:
        tl = self.current_template_level()

        if self.template_image is None:
            # Build a blank canvas large enough to host points and bbox.
            max_x = int(tl.tl_x + tl.width + 20)
            max_y = int(tl.tl_y + tl.height + 20)
            for px, py in self.get_feature_abs_points():
                max_x = max(max_x, px + 20)
                max_y = max(max_y, py + 20)
            max_x = max(max_x, 240)
            max_y = max(max_y, 160)
            bg = np.zeros((max_y, max_x, 3), dtype=np.uint8)
            return bg

        meta = self.current_meta()
        angle = float(meta.get("angle", 0.0))
        scale = float(meta.get("scale", 1.0))
        img = ShapeInfoProducer.transform(self.template_image, angle, scale)
        for _ in range(self.level):
            img = cv2.pyrDown(img)
        return img

    def render(self) -> np.ndarray:
        img = self.render_background().copy()
        tl = self.current_template_level()
        palette = orientation_palette_bgr()

        # Template bbox at current level.
        x1 = int(tl.tl_x)
        y1 = int(tl.tl_y)
        x2 = int(tl.tl_x + tl.width)
        y2 = int(tl.tl_y + tl.height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

        pts = self.get_feature_abs_points()
        for i, (px, py) in enumerate(pts):
            f = tl.features[i]
            color = palette[int(f.label) % len(palette)]
            theta = float(f.theta) if np.isfinite(f.theta) else label_to_angle_deg(int(f.label))
            p2 = arrow_endpoint(int(px), int(py), theta, length=8.0)
            if self.hover_index is not None and i == self.hover_index:
                cv2.circle(img, (int(px), int(py)), 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(
                img,
                (int(px), int(py)),
                p2,
                color,
                1,
                cv2.LINE_AA,
                0,
                0.35,
            )
            cv2.circle(img, (int(px), int(py)), 1, color, -1, cv2.LINE_AA)

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
            color = palette[lb]
            cv2.arrowedLine(img, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA, 0, 0.35)

        if self.is_erase_dragging and self.erase_start_abs is not None and self.erase_end_abs is not None:
            x1, y1, x2, y2 = normalize_drag_rect(self.erase_start_abs, self.erase_end_abs)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 255), 1, cv2.LINE_AA)

        if self.show_status:
            status = (
                f"class={self.class_id} ({self.class_index+1}/{len(self.class_ids)})  "
                f"tid={self.template_id+1}/{self.templates_count()}  "
                f"level={self.level}/{self.levels_count()-1}  "
                f"points={len(tl.features)}  label={self.current_label}  zoom={self.zoom:.2f}x"
            )
            color = (40, 220, 40) if not self.dirty else (0, 180, 255)
            cv2.putText(img, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        if self.show_help:
            lines = [
                "L-drag:add direction  R-click:delete nearest  R-drag:box delete  Wheel/+-:zoom",
                "c/x class, n/p template, l/k level, ]/[ label, s save, u undo, i status, q quit",
            ]
            y = 48
            for t in lines:
                cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                y += 20

        if self.zoom != 1.0:
            img = cv2.resize(img, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)
        return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive point editor for line2Dup-like template model.")
    parser.add_argument("--model", required=True, help="Input model json path.")
    parser.add_argument("--template-image", default="", help="Template source image path for background visualization.")
    parser.add_argument(
        "--out-model",
        default="",
        help="Output model path. Default: overwrite --model.",
    )
    parser.add_argument("--class-id", default="", help="Initial class id.")
    parser.add_argument("--template-id", type=int, default=0, help="Initial template id (0-based).")
    parser.add_argument("--level", type=int, default=0, help="Initial pyramid level.")
    parser.add_argument("--zoom", type=float, default=2.0, help="Display zoom factor.")
    parser.add_argument("--window", default="line2dup_template_editor", help="OpenCV window title.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    out_model_path = Path(args.out_model) if args.out_model else model_path

    template_img = None
    if args.template_image:
        template_img = cv2.imread(args.template_image, cv2.IMREAD_COLOR)
        if template_img is None:
            raise FileNotFoundError(f"Failed to read template image: {args.template_image}")

    state = EditorState(
        model_path=model_path,
        out_model_path=out_model_path,
        template_image=template_img,
        start_class=args.class_id,
        start_template_id=args.template_id,
        start_level=args.level,
        zoom=args.zoom,
    )

    win = args.window
    # Use AUTOSIZE so displayed pixels match the rendered canvas size.
    # Otherwise WINDOW_NORMAL may auto-fit image and hide visual zoom effect.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def change_zoom(factor: float) -> None:
        old = state.zoom
        state.zoom_by(factor)
        if abs(state.zoom - old) > 1e-6:
            print(f"zoom={state.zoom:.2f}x")

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata) -> None:
        x_abs = int(round(x / state.zoom))
        y_abs = int(round(y / state.zoom))
        if event == cv2.EVENT_MOUSEMOVE:
            state.update_hover_abs(x_abs, y_abs)
            if state.is_dragging:
                state.drag_end_abs = (x_abs, y_abs)
            if state.is_erase_dragging:
                state.erase_end_abs = (x_abs, y_abs)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if state.is_erase_dragging:
                return
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
                    state.add_feature_abs(sx, sy, lb, theta_deg=theta)
                else:
                    lb = int(state.current_label) % 8
                    state.add_feature_abs(sx, sy, lb, theta_deg=label_to_angle_deg(lb))
            state.is_dragging = False
            state.drag_start_abs = None
            state.drag_end_abs = None
            state.update_hover_abs(x_abs, y_abs)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state.is_dragging:
                return
            state.is_erase_dragging = True
            state.erase_start_abs = (x_abs, y_abs)
            state.erase_end_abs = (x_abs, y_abs)
        elif event == cv2.EVENT_RBUTTONUP:
            if state.is_erase_dragging and state.erase_start_abs is not None and state.erase_end_abs is not None:
                state.erase_end_abs = (x_abs, y_abs)
                if drag_is_click(state.erase_start_abs, state.erase_end_abs):
                    state.delete_nearest_abs(x_abs, y_abs)
                else:
                    deleted = state.delete_box_abs(state.erase_start_abs, state.erase_end_abs)
                    if deleted > 0:
                        print(f"deleted {deleted} points in selection")
            state.is_erase_dragging = False
            state.erase_start_abs = None
            state.erase_end_abs = None
            state.update_hover_abs(x_abs, y_abs)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 0
            if hasattr(cv2, "getMouseWheelDelta"):
                try:
                    delta = int(cv2.getMouseWheelDelta(_flags))
                except Exception:
                    delta = 0
            if delta == 0:
                # Fallback for builds without getMouseWheelDelta behavior.
                delta = 1 if _flags > 0 else -1
            if delta > 0:
                change_zoom(1.15)
            elif delta < 0:
                change_zoom(1.0 / 1.15)

    cv2.setMouseCallback(win, on_mouse)

    print("Editor started.")
    print(f"model={model_path}")
    print(f"out_model={out_model_path}")
    print("Mouse: left-drag add with direction, right click delete nearest, right-drag box delete. Press 'h' for help, 'i' for status.")

    while True:
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
        elif key == ord("s"):
            state.save()
            print(f"saved: {state.out_model_path}")
        elif key == ord("u"):
            if state.undo():
                print("undo: ok")
        elif key == ord("c"):
            state.next_class(+1)
        elif key == ord("x"):
            state.next_class(-1)
        elif key == ord("n"):
            state.next_template(+1)
        elif key == ord("p"):
            state.next_template(-1)
        elif key == ord("l"):
            state.next_level(+1)
        elif key == ord("k"):
            state.next_level(-1)
        elif key == ord("]"):
            state.current_label = (state.current_label + 1) % 8
        elif key == ord("["):
            state.current_label = (state.current_label + 7) % 8
        elif key in (ord("+"), ord("="), 43, 61, 107, 171):
            change_zoom(1.2)
        elif key in (ord("-"), ord("_"), 45, 95, 109, 173):
            change_zoom(1.0 / 1.2)
        elif key in (ord("0"),):
            old = state.zoom
            state.set_zoom(1.0)
            if abs(state.zoom - old) > 1e-6:
                print(f"zoom={state.zoom:.2f}x")

    cv2.destroyAllWindows()
    if state.dirty:
        print("Exit without save. Press 's' next time before quit to persist edits.")
    else:
        print("Exit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
