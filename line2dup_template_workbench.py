#!/usr/bin/env python3
"""
Unified GUI workbench for line2Dup-like template flow.

One command starts one GUI window:
1) Create template model from an image ROI.
2) Open/edit existing model points.
3) Find objects in scene with a saved model.

All core operations are available via mouse + GUI controls.
No keyboard shortcuts are required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from line2dup_like_matcher import (
    Feature,
    Line2DupLikeDetector,
    ShapeInfoProducer,
    TemplateLevel,
    crop_templates,
    draw_matches,
    label_to_theta_deg as matcher_label_to_theta_deg,
    load_detector_model,
    nms_matches,
    save_detector_model,
    theta_deg_to_label as matcher_theta_deg_to_label,
)

ZOOM_MIN = 0.2
ZOOM_MAX = 16.0


def parse_levels(arg: str) -> List[int]:
    vals = [int(x.strip()) for x in str(arg).split(",") if x.strip()]
    if not vals:
        raise ValueError("levels cannot be empty")
    return vals


def expand_numeric_range(start: float, end: float, step: float, eps: float = 1e-9) -> List[float]:
    s0 = float(start)
    s1 = float(end)
    if abs(s1 - s0) <= eps:
        return [s0]
    st = abs(float(step))
    if st <= eps:
        raise ValueError("step must be > 0 when start != end")
    lo = min(s0, s1)
    hi = max(s0, s1)
    vals: List[float] = []
    cur = lo
    while cur <= hi + eps:
        vals.append(float(cur))
        cur += st
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


def clone_levels(levels: Sequence[TemplateLevel]) -> List[TemplateLevel]:
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


def bgr_to_photo(image_bgr: np.ndarray) -> tk.PhotoImage:
    if image_bgr is None or image_bgr.size == 0:
        image_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    payload = header + rgb.tobytes()
    return tk.PhotoImage(data=payload, format="PPM")


def wheel_delta(event: tk.Event) -> int:
    if hasattr(event, "delta") and int(event.delta) != 0:
        return 1 if event.delta > 0 else -1
    if getattr(event, "num", None) == 4:
        return 1
    if getattr(event, "num", None) == 5:
        return -1
    return 0


def keep_canvas_point_stable_after_zoom(
    canvas: tk.Canvas,
    event: tk.Event,
    old_zoom: float,
    new_zoom: float,
    content_w: int,
    content_h: int,
) -> None:
    if old_zoom <= 1e-9 or new_zoom <= 1e-9:
        return
    img_x = float(canvas.canvasx(event.x)) / old_zoom
    img_y = float(canvas.canvasy(event.y)) / old_zoom
    target_canvas_x = img_x * new_zoom
    target_canvas_y = img_y * new_zoom

    canvas.update_idletasks()
    view_w = max(1, int(canvas.winfo_width()))
    view_h = max(1, int(canvas.winfo_height()))
    total_w = max(1, int(content_w))
    total_h = max(1, int(content_h))

    left = target_canvas_x - float(event.x)
    top = target_canvas_y - float(event.y)
    max_left = max(0.0, float(total_w - view_w))
    max_top = max(0.0, float(total_h - view_h))
    left = min(max(left, 0.0), max_left)
    top = min(max(top, 0.0), max_top)

    if total_w > view_w:
        canvas.xview_moveto(left / float(total_w))
    else:
        canvas.xview_moveto(0.0)
    if total_h > view_h:
        canvas.yview_moveto(top / float(total_h))
    else:
        canvas.yview_moveto(0.0)


@dataclass
class RoiRect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class UndoItem:
    class_id: str
    template_id: int
    level: int
    features: List[Feature]


class CreateTemplateTab(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_path = ""
        self.roi: Optional[RoiRect] = None
        self.template_levels: List[TemplateLevel] = []

        self.drag_kind: Optional[str] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.hover_index: Optional[int] = None

        self._photo: Optional[tk.PhotoImage] = None
        self._canvas_image_id: Optional[int] = None
        self._last_render_w = 320
        self._last_render_h = 240

        self._build_ui()
        self._refresh_canvas()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        control = ttk.Frame(self, padding=8)
        control.grid(row=0, column=0, sticky="ns")
        view = ttk.Frame(self, padding=(0, 8, 8, 8))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Load an image to start.")
        self.tool_var = tk.StringVar(value="roi")
        self.class_id_var = tk.StringVar(value="object")
        self.levels_var = tk.StringVar(value="4,8")
        self.num_features_var = tk.IntVar(value=128)
        self.weak_thresh_var = tk.DoubleVar(value=30.0)
        self.strong_thresh_var = tk.DoubleVar(value=60.0)
        self.level_var = tk.IntVar(value=0)
        self.label_var = tk.IntVar(value=0)
        self.zoom_var = tk.DoubleVar(value=2.0)
        self.out_model_var = tk.StringVar(value="")
        self.angle_start_var = tk.DoubleVar(value=0.0)
        self.angle_end_var = tk.DoubleVar(value=0.0)
        self.angle_step_var = tk.DoubleVar(value=10.0)
        self.scale_start_var = tk.DoubleVar(value=1.0)
        self.scale_end_var = tk.DoubleVar(value=1.0)
        self.scale_step_var = tk.DoubleVar(value=0.05)

        row = 0
        ttk.Label(control, text="Create Template (ROI -> points -> model)").grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Select Image...", command=self.select_image).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1
        self.image_label = ttk.Label(control, text="(no image)", width=36, wraplength=260)
        self.image_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Output Model...", command=self.select_output_model).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        self.out_label = ttk.Label(control, text="(not set)", width=36, wraplength=260)
        self.out_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Separator(control, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=6)
        row += 1

        ttk.Label(control, text="Class ID").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Entry(control, textvariable=self.class_id_var, width=28).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(control, text="Levels (e.g. 4,8)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Entry(control, textvariable=self.levels_var, width=28).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(control, text="Num Features").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=8, to=1024, increment=8, textvariable=self.num_features_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Weak Threshold").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=1, to=255, increment=1, textvariable=self.weak_thresh_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Strong Threshold").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=1, to=255, increment=1, textvariable=self.strong_thresh_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="View Level").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self.level_spin = ttk.Spinbox(control, from_=0, to=0, increment=1, textvariable=self.level_var, width=10, command=self._refresh_canvas)
        self.level_spin.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Label For Short Drag").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=0, to=7, increment=1, textvariable=self.label_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Zoom").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Scale(control, from_=ZOOM_MIN, to=ZOOM_MAX, orient="horizontal", variable=self.zoom_var, command=lambda _e: self._refresh_canvas()).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(control, text="Angle Start / End / Step").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        angle_row = ttk.Frame(control)
        angle_row.grid(row=row, column=0, sticky="w")
        ttk.Spinbox(angle_row, from_=-3600, to=3600, increment=1, textvariable=self.angle_start_var, width=8).pack(side="left")
        ttk.Spinbox(angle_row, from_=-3600, to=3600, increment=1, textvariable=self.angle_end_var, width=8).pack(side="left", padx=(4, 0))
        ttk.Spinbox(angle_row, from_=0.1, to=3600, increment=1, textvariable=self.angle_step_var, width=8).pack(side="left", padx=(4, 0))
        row += 1

        ttk.Label(control, text="Scale Start / End / Step").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        scale_row = ttk.Frame(control)
        scale_row.grid(row=row, column=0, sticky="w")
        ttk.Spinbox(scale_row, from_=0.01, to=100, increment=0.01, textvariable=self.scale_start_var, width=8).pack(side="left")
        ttk.Spinbox(scale_row, from_=0.01, to=100, increment=0.01, textvariable=self.scale_end_var, width=8).pack(side="left", padx=(4, 0))
        ttk.Spinbox(scale_row, from_=0.001, to=100, increment=0.01, textvariable=self.scale_step_var, width=8).pack(side="left", padx=(4, 0))
        row += 1

        ttk.Label(control, text="Tool").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        tool_row = ttk.Frame(control)
        tool_row.grid(row=row, column=0, sticky="w")
        ttk.Radiobutton(tool_row, text="Select ROI", value="roi", variable=self.tool_var, command=self._refresh_canvas).pack(side="left")
        ttk.Radiobutton(tool_row, text="Edit Points", value="point", variable=self.tool_var, command=self._refresh_canvas).pack(side="left", padx=(8, 0))
        row += 1

        ttk.Button(control, text="Extract Points From ROI", command=self.extract_points).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        ttk.Button(control, text="Reset ROI", command=self.reset_roi).grid(row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1
        ttk.Button(control, text="Save Model", command=self.save_model).grid(row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1

        ttk.Separator(control, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=8)
        row += 1
        ttk.Label(control, textvariable=self.status_var, wraplength=260, foreground="#2f7d32").grid(row=row, column=0, sticky="w")

        self.canvas = tk.Canvas(view, bg="#1d1d1d", highlightthickness=0)
        xbar = ttk.Scrollbar(view, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(view, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_motion)

    def select_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Read error", f"Failed to read image:\n{path}")
            return
        self.image_bgr = img
        self.image_path = path
        self.image_label.configure(text=path)
        self.roi = None
        self.template_levels = []
        self.level_var.set(0)
        self.level_spin.configure(from_=0, to=0)
        self.tool_var.set("roi")
        self.status_var.set("Image loaded. Drag left mouse to select ROI.")
        self._refresh_canvas()

    def select_output_model(self) -> None:
        initial = self.out_model_var.get().strip()
        if not initial and self.image_path:
            stem = Path(self.image_path).stem
            initial = str(Path(self.image_path).with_name(f"{stem}_model.json"))
        path = filedialog.asksaveasfilename(
            title="Output model path",
            initialfile=Path(initial).name if initial else "template_model.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.out_model_var.set(path)
        self.out_label.configure(text=path)

    def reset_roi(self) -> None:
        self.roi = None
        self.template_levels = []
        self.level_var.set(0)
        self.level_spin.configure(from_=0, to=0)
        self.tool_var.set("roi")
        self.status_var.set("ROI reset. Drag left mouse to select a new ROI.")
        self._refresh_canvas()

    def extract_points(self) -> None:
        if self.image_bgr is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        if self.roi is None:
            messagebox.showwarning("No ROI", "Please select ROI first.")
            return
        try:
            levels = parse_levels(self.levels_var.get())
        except Exception as exc:
            messagebox.showerror("Invalid levels", str(exc))
            return

        class_id = self.class_id_var.get().strip() or "object"
        num_features = int(max(8, self.num_features_var.get()))
        weak = float(max(1.0, self.weak_thresh_var.get()))
        strong = float(max(1.0, self.strong_thresh_var.get()))

        roi_img = self.image_bgr[self.roi.y : self.roi.y + self.roi.h, self.roi.x : self.roi.x + self.roi.w].copy()
        mask = np.full((roi_img.shape[0], roi_img.shape[1]), 255, dtype=np.uint8)

        detector = Line2DupLikeDetector(
            num_features=num_features,
            T_levels=levels,
            weak_threshold=weak,
            strong_threshold=strong,
        )
        tid = detector.add_template(
            roi_img,
            class_id=class_id,
            object_mask=mask,
            num_features=num_features,
            metadata={"angle": 0.0, "scale": 1.0},
        )
        if tid < 0:
            self.status_var.set("No template extracted. Try lower strong/weak threshold.")
            messagebox.showwarning("No template extracted", "Try lower strong/weak threshold or select a clearer ROI.")
            return

        extracted_levels = clone_levels(detector.get_templates(class_id, tid))
        if not extracted_levels:
            self.status_var.set("No template extracted.")
            return

        # Editing baseline is always level-0 (full-resolution ROI).
        self.template_levels = [extracted_levels[0]]
        self._force_levels_to_roi_extent()
        self._sync_levels_from_level0(total_levels=len(levels))
        max_level = max(0, len(self.template_levels) - 1)
        self.level_spin.configure(from_=0, to=max_level)
        self.level_var.set(0)
        self.tool_var.set("point")
        self.status_var.set(
            f"Extracted {len(self.template_levels[0].features)} points at level0. "
            f"L0 is editable; L1+ are auto-scaled from L0."
        )
        self._refresh_canvas()

    def save_model(self) -> None:
        out_path = self.out_model_var.get().strip()
        if not out_path:
            self.select_output_model()
            out_path = self.out_model_var.get().strip()
            if not out_path:
                return
        if self.roi is None:
            messagebox.showerror("No ROI", "ROI is missing.")
            return
        if self.image_bgr is None:
            messagebox.showerror("No image", "Please select image first.")
            return

        try:
            levels = parse_levels(self.levels_var.get())
        except Exception as exc:
            messagebox.showerror("Invalid levels", str(exc))
            return

        try:
            pose_infos = self._pose_infos_from_ui()
        except Exception as exc:
            messagebox.showerror("Invalid angle/scale range", str(exc))
            return
        scales = [float(s) for _a, s in pose_infos]
        if scales:
            s_min = min(scales)
            s_max = max(scales)
            if s_min < 0.5 or s_max > 2.0:
                ok = messagebox.askyesno(
                    "Scale range warning",
                    "Current scale range is very wide.\n"
                    "This often introduces unsuitable matches.\n\n"
                    f"Current range: [{s_min:.3f}, {s_max:.3f}]\n"
                    "Recommended first try: [0.8, 1.2]\n\n"
                    "Continue anyway?",
                )
                if not ok:
                    return
        if len(pose_infos) > 500:
            ok = messagebox.askyesno(
                "Large template set",
                f"Will generate {len(pose_infos)} templates. Continue?",
            )
            if not ok:
                return

        class_id = self.class_id_var.get().strip() or "object"
        num_features = int(max(8, self.num_features_var.get()))
        weak_threshold = float(max(1.0, self.weak_thresh_var.get()))
        strong_threshold = float(max(1.0, self.strong_thresh_var.get()))
        detector = Line2DupLikeDetector(
            num_features=num_features,
            T_levels=levels,
            weak_threshold=weak_threshold,
            strong_threshold=strong_threshold,
        )
        roi_img = self.image_bgr[self.roi.y : self.roi.y + self.roi.h, self.roi.x : self.roi.x + self.roi.w].copy()
        roi_mask = np.full((roi_img.shape[0], roi_img.shape[1]), 255, dtype=np.uint8)

        kept = 0
        skipped = 0
        first_success_tid = -1
        first_success_src: Optional[np.ndarray] = None
        use_manual_points = len(self.template_levels) > 0

        if use_manual_points and len(self.template_levels) != len(levels):
            messagebox.showerror(
                "Levels mismatch",
                "Current edited points were extracted with a different levels setting.\n"
                "Please click 'Extract Points From ROI' again after changing Levels.",
            )
            return

        if use_manual_points:
            base_levels = clone_levels(self.template_levels)
            detector.class_templates[class_id] = []
            detector.class_meta[class_id] = []
            for angle_deg, scale in pose_infos:
                transformed = self._transform_levels_for_pose(
                    base_levels=base_levels,
                    angle_deg=float(angle_deg),
                    scale=float(scale),
                    # Keep expanded canvas for transformed variants (do not crop back).
                    auto_crop=False,
                    adapt_feature_count=True,
                )
                if (not transformed) or any(len(lv.features) <= 0 for lv in transformed):
                    skipped += 1
                    continue

                detector.class_templates[class_id].append(transformed)
                detector.class_meta[class_id].append(
                    {
                        "angle": float(angle_deg),
                        "scale": float(scale),
                        "roi_x": int(self.roi.x),
                        "roi_y": int(self.roi.y),
                        "roi_w": int(self.roi.w),
                        "roi_h": int(self.roi.h),
                    }
                )
                kept += 1
                if first_success_tid < 0:
                    first_success_tid = len(detector.class_templates[class_id]) - 1
                    l0 = transformed[0]
                    ph = max(1, int(l0.height) + 1)
                    pw = max(1, int(l0.width) + 1)
                    first_success_src = np.zeros((ph, pw, 3), dtype=np.uint8)
        else:
            producer = ShapeInfoProducer(roi_img, roi_mask)
            a0 = float(self.angle_start_var.get())
            a1 = float(self.angle_end_var.get())
            s0 = float(self.scale_start_var.get())
            s1 = float(self.scale_end_var.get())
            producer.angle_range = [a0, a1] if abs(a1 - a0) > 1e-9 else [a0]
            producer.scale_range = [s0, s1] if abs(s1 - s0) > 1e-9 else [s0]
            producer.angle_step = float(max(1e-6, self.angle_step_var.get()))
            producer.scale_step = float(max(1e-6, self.scale_step_var.get()))
            infos = producer.produce_infos()
            for info in infos:
                src_i = producer.src_of(info)
                mask_i = producer.mask_of(info)
                nfeat = max(16, int(round(float(num_features) * float(info.scale))))
                tid = detector.add_template(
                    src_i,
                    class_id=class_id,
                    object_mask=mask_i,
                    num_features=nfeat,
                    metadata={
                        "angle": float(info.angle),
                        "scale": float(info.scale),
                        "roi_x": int(self.roi.x),
                        "roi_y": int(self.roi.y),
                        "roi_w": int(self.roi.w),
                        "roi_h": int(self.roi.h),
                    },
                )
                if tid >= 0:
                    kept += 1
                    if first_success_tid < 0:
                        first_success_tid = int(tid)
                        first_success_src = src_i.copy()
                else:
                    skipped += 1

        if kept <= 0:
            messagebox.showerror(
                "No valid templates",
                "All angle/scale variants became empty.\n"
                "Try smaller angle/scale range, lower thresholds, or add more points.",
            )
            return

        save_detector_model(detector, out_path)

        if first_success_tid >= 0 and first_success_src is not None:
            preview = first_success_src
            tl = detector.get_templates(class_id, first_success_tid)[0]
            x1 = int(tl.tl_x)
            y1 = int(tl.tl_y)
            x2 = int(tl.tl_x + tl.width)
            y2 = int(tl.tl_y + tl.height)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
            palette = orientation_palette_bgr()
            for f in tl.features:
                px = int(f.x + tl.tl_x)
                py = int(f.y + tl.tl_y)
                color = palette[int(f.label) % len(palette)]
                cv2.drawMarker(preview, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1, line_type=cv2.LINE_AA)
        else:
            preview = self._render_image()
        preview_path = str(Path(out_path).with_name(f"{Path(out_path).stem}_preview.png"))
        cv2.imwrite(preview_path, preview)

        mode = "edited-points" if use_manual_points else "auto-extracted"
        self.status_var.set(f"Saved model: {out_path}  templates={kept}  mode={mode}")
        messagebox.showinfo(
            "Saved",
            f"Model saved:\n{out_path}\n"
            f"Templates: {kept} (skipped {skipped})\n"
            f"Source: {mode}\n\n"
            f"Preview:\n{preview_path}",
        )

    def _current_level_template(self) -> Optional[TemplateLevel]:
        if not self.template_levels:
            return None
        idx = max(0, min(int(self.level_var.get()), len(self.template_levels) - 1))
        return self.template_levels[idx]

    def _roi_level_shapes(self, num_levels: Optional[int] = None) -> List[Tuple[int, int]]:
        """Return (w, h) for each pyramid level based on the fixed user ROI."""
        if self.image_bgr is None or self.roi is None:
            return []
        img = self.image_bgr[self.roi.y : self.roi.y + self.roi.h, self.roi.x : self.roi.x + self.roi.w].copy()
        shapes: List[Tuple[int, int]] = []
        nlevels = len(self.template_levels) if num_levels is None else max(1, int(num_levels))
        for _ in range(nlevels):
            h, w = img.shape[:2]
            shapes.append((int(w), int(h)))
            if h < 2 or w < 2:
                continue
            img = cv2.pyrDown(img)
        return shapes

    def _sync_levels_from_level0(self, total_levels: Optional[int] = None) -> None:
        """
        Rebuild L1+ from L0 by pure geometric down-scaling (no per-level re-extraction).
        """
        if not self.template_levels:
            return
        if total_levels is None:
            total_levels = len(self.template_levels)
        total_levels = max(1, int(total_levels))

        shapes = self._roi_level_shapes(num_levels=total_levels)
        if not shapes:
            return

        l0 = self.template_levels[0]
        w0, h0 = shapes[0]
        max_x0 = max(0, int(w0) - 1)
        max_y0 = max(0, int(h0) - 1)

        # Normalize level-0 points into fixed ROI coordinates.
        l0_feats: List[Feature] = []
        l0_seen = set()
        for f in l0.features:
            x0 = int(np.clip(int(f.x) + int(l0.tl_x), 0, max_x0))
            y0 = int(np.clip(int(f.y) + int(l0.tl_y), 0, max_y0))
            key = (x0, y0, int(f.label) & 7)
            if key in l0_seen:
                continue
            l0_seen.add(key)
            l0_feats.append(Feature(x=x0, y=y0, label=int(f.label) & 7, theta=float(f.theta)))

        new_levels: List[TemplateLevel] = [
            TemplateLevel(
                width=max_x0,
                height=max_y0,
                tl_x=0,
                tl_y=0,
                pyramid_level=0,
                features=l0_feats,
            )
        ]

        for lv in range(1, total_levels):
            w, h = shapes[lv]
            max_x = max(0, int(w) - 1)
            max_y = max(0, int(h) - 1)
            div = float(1 << lv)

            feats: List[Feature] = []
            seen = set()
            for f in l0_feats:
                x = int(round(float(f.x) / div))
                y = int(round(float(f.y) / div))
                x = int(np.clip(x, 0, max_x))
                y = int(np.clip(y, 0, max_y))
                key = (x, y, int(f.label) & 7)
                if key in seen:
                    continue
                seen.add(key)
                feats.append(Feature(x=x, y=y, label=int(f.label) & 7, theta=float(f.theta)))

            new_levels.append(
                TemplateLevel(
                    width=max_x,
                    height=max_y,
                    tl_x=0,
                    tl_y=0,
                    pyramid_level=lv,
                    features=feats,
                )
            )

        self.template_levels = new_levels

    def _force_levels_to_roi_extent(self) -> None:
        """
        Keep template extent fixed to the user-selected ROI at each level.
        Convert cropped-template feature coordinates back to ROI coordinates.
        """
        if not self.template_levels:
            return
        shapes = self._roi_level_shapes()
        if not shapes:
            return
        for i, tl in enumerate(self.template_levels):
            if i >= len(shapes):
                break
            w, h = shapes[i]
            max_x = max(0, int(w) - 1)
            max_y = max(0, int(h) - 1)
            old_tlx = int(tl.tl_x)
            old_tly = int(tl.tl_y)

            # Existing points are currently relative to cropped bbox; map back to full ROI level.
            for f in tl.features:
                f.x = int(np.clip(int(f.x) + old_tlx, 0, max_x))
                f.y = int(np.clip(int(f.y) + old_tly, 0, max_y))

            tl.tl_x = 0
            tl.tl_y = 0
            tl.width = max_x
            tl.height = max_y

    def _pose_infos_from_ui(self) -> List[Tuple[float, float]]:
        a0 = float(self.angle_start_var.get())
        a1 = float(self.angle_end_var.get())
        astep = float(self.angle_step_var.get())
        s0 = float(self.scale_start_var.get())
        s1 = float(self.scale_end_var.get())
        sstep = float(self.scale_step_var.get())
        if s0 <= 0.0 or s1 <= 0.0:
            raise ValueError("scale start/end must be > 0")

        angles = expand_numeric_range(a0, a1, astep)
        scales = expand_numeric_range(s0, s1, sstep)

        infos: List[Tuple[float, float]] = []
        seen = set()
        for sc in scales:
            for ang in angles:
                ang_n = float((float(ang) % 360.0 + 360.0) % 360.0)
                if abs(ang_n - 360.0) < 1e-9:
                    ang_n = 0.0
                key = (round(ang_n, 6), round(float(sc), 6))
                if key in seen:
                    continue
                seen.add(key)
                infos.append((ang_n, float(sc)))
        if not infos:
            infos = [(0.0, 1.0)]
        return infos

    def _transform_levels_for_pose(
        self,
        base_levels: Sequence[TemplateLevel],
        angle_deg: float,
        scale: float,
        auto_crop: bool = False,
        adapt_feature_count: bool = False,
    ) -> List[TemplateLevel]:
        out: List[TemplateLevel] = []
        ang_rad = -float(angle_deg) / 180.0 * float(np.pi)
        c = float(np.cos(ang_rad))
        s = float(np.sin(ang_rad))
        sc = float(scale)

        for base in base_levels:
            x_min = int(base.tl_x)
            y_min = int(base.tl_y)
            x_max = int(base.tl_x + base.width)
            y_max = int(base.tl_y + base.height)
            w = int(base.width) + 1
            h = int(base.height) + 1
            cx = float(x_min + w * 0.5)
            cy = float(y_min + h * 0.5)

            def _xf(px: float, py: float) -> Tuple[float, float]:
                dx = (px - cx) * sc
                dy = (py - cy) * sc
                rx = c * dx - s * dy + cx
                ry = s * dx + c * dy + cy
                return rx, ry

            # Auto-expand canvas by transformed ROI rectangle bounds.
            corners = [
                (float(x_min), float(y_min)),
                (float(x_max), float(y_min)),
                (float(x_max), float(y_max)),
                (float(x_min), float(y_max)),
            ]
            tc = [_xf(px, py) for px, py in corners]
            canvas_x_min = int(np.floor(min(p[0] for p in tc)))
            canvas_y_min = int(np.floor(min(p[1] for p in tc)))
            canvas_x_max = int(np.ceil(max(p[0] for p in tc)))
            canvas_y_max = int(np.ceil(max(p[1] for p in tc)))
            if canvas_x_max < canvas_x_min:
                canvas_x_max = canvas_x_min
            if canvas_y_max < canvas_y_min:
                canvas_y_max = canvas_y_min

            feats: List[Feature] = []
            seen = set()
            for f in base.features:
                px = float(f.x + base.tl_x)
                py = float(f.y + base.tl_y)
                rx, ry = _xf(px, py)
                xi = int(round(rx))
                yi = int(round(ry))
                base_theta = float(f.theta)
                if not np.isfinite(base_theta):
                    base_theta = label_to_angle_deg(int(f.label))
                theta = (base_theta - float(angle_deg)) % 360.0
                lb = angle_deg_to_label(theta)
                xr = int(xi - canvas_x_min)
                yr = int(yi - canvas_y_min)
                if xr < 0 or yr < 0:
                    continue
                k = (xr, yr, lb)
                if k in seen:
                    continue
                seen.add(k)
                feats.append(Feature(x=xr, y=yr, label=lb, theta=theta))

            if adapt_feature_count:
                s_clamped = float(np.clip(scale, 0.05, 1.0))
                target = max(8, int(round(len(base.features) * s_clamped)))
                if len(feats) > target:
                    feats = feats[:target]

            out.append(
                TemplateLevel(
                    width=int(canvas_x_max - canvas_x_min),
                    height=int(canvas_y_max - canvas_y_min),
                    tl_x=0,
                    tl_y=0,
                    pyramid_level=int(base.pyramid_level),
                    features=feats,
                )
            )
        if auto_crop and out and all(len(x.features) > 0 for x in out):
            crop_templates(out)
        return out

    def _in_roi_view(self) -> bool:
        return self.roi is not None and self.tool_var.get() == "point"

    def _level_roi_image(self) -> np.ndarray:
        if self.image_bgr is None or self.roi is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        img = self.image_bgr[self.roi.y : self.roi.y + self.roi.h, self.roi.x : self.roi.x + self.roi.w].copy()
        lv = max(0, int(self.level_var.get()))
        for _ in range(lv):
            if img.shape[0] < 2 or img.shape[1] < 2:
                break
            img = cv2.pyrDown(img)
        return img

    def _get_abs_points(self) -> List[Tuple[int, int]]:
        tl = self._current_level_template()
        if tl is None:
            return []
        return [(int(f.x + tl.tl_x), int(f.y + tl.tl_y)) for f in tl.features]

    def _update_hover(self, x_abs: int, y_abs: int) -> None:
        tl = self._current_level_template()
        if tl is None or not tl.features:
            self.hover_index = None
            return
        best_idx = -1
        best_d2 = 1e18
        for i, (px, py) in enumerate(self._get_abs_points()):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx >= 0 and best_d2 <= 10.0 * 10.0:
            self.hover_index = best_idx
        else:
            self.hover_index = None

    def _add_point(self, x_abs: int, y_abs: int, label: int, theta_deg: Optional[float]) -> None:
        tl = self._current_level_template()
        if tl is None:
            return
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch View Level to 0.")
            return
        xr = int(round(x_abs - tl.tl_x))
        yr = int(round(y_abs - tl.tl_y))
        xr = max(0, min(xr, int(tl.width)))
        yr = max(0, min(yr, int(tl.height)))
        lb = int(label) % 8
        theta = label_to_angle_deg(lb) if theta_deg is None else float(theta_deg)
        tl.features.append(Feature(x=xr, y=yr, label=lb, theta=theta))
        self._sync_levels_from_level0(total_levels=len(self.template_levels))

    def _delete_nearest(self, x_abs: int, y_abs: int) -> bool:
        tl = self._current_level_template()
        if tl is None or not tl.features:
            return False
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch View Level to 0.")
            return False
        best_idx = -1
        best_d2 = 1e18
        pts = self._get_abs_points()
        for i, (px, py) in enumerate(pts):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx < 0 or best_d2 > 10.0 * 10.0:
            return False
        del tl.features[best_idx]
        self.hover_index = None
        self._sync_levels_from_level0(total_levels=len(self.template_levels))
        return True

    def _canvas_to_image(self, event: tk.Event) -> Tuple[int, int]:
        z = max(1e-6, float(self.zoom_var.get()))
        xw = float(self.canvas.canvasx(event.x))
        yw = float(self.canvas.canvasy(event.y))
        return int(round(xw / z)), int(round(yw / z))

    def _on_left_down(self, event: tk.Event) -> None:
        if self.image_bgr is None:
            return
        x, y = self._canvas_to_image(event)
        if self.tool_var.get() == "roi":
            self.drag_kind = "roi"
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif self.tool_var.get() == "point" and self.roi is not None:
            if int(self.level_var.get()) != 0:
                self.status_var.set("Only L0 is editable. Switch View Level to 0.")
                return
            self.drag_kind = "point"
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_left_move(self, event: tk.Event) -> None:
        if self.drag_kind is None:
            return
        x, y = self._canvas_to_image(event)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_left_up(self, event: tk.Event) -> None:
        if self.drag_kind is None or self.drag_start is None:
            return
        x, y = self._canvas_to_image(event)
        self.drag_end = (x, y)

        if self.drag_kind == "roi" and self.image_bgr is not None:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_end
            xa = max(0, min(x0, x1))
            ya = max(0, min(y0, y1))
            xb = min(self.image_bgr.shape[1], max(x0, x1))
            yb = min(self.image_bgr.shape[0], max(y0, y1))
            w = max(0, xb - xa)
            h = max(0, yb - ya)
            if w >= 4 and h >= 4:
                self.roi = RoiRect(x=xa, y=ya, w=w, h=h)
                self.template_levels = []
                self.level_var.set(0)
                self.level_spin.configure(from_=0, to=0)
                self.status_var.set(f"ROI set: x={xa}, y={ya}, w={w}, h={h}. Click 'Extract Points From ROI'.")
            else:
                self.status_var.set("ROI too small, please drag a larger region.")

        elif self.drag_kind == "point" and self.roi is not None:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            dx = ex - sx
            dy = ey - sy
            dist = float(np.hypot(dx, dy))
            if dist >= 2.0:
                theta = float(np.degrees(np.arctan2(float(dy), float(dx))))
                lb = angle_deg_to_label(theta)
                self.label_var.set(lb)
                self._add_point(sx, sy, lb, theta_deg=theta)
            else:
                lb = int(self.label_var.get()) % 8
                self._add_point(sx, sy, lb, theta_deg=label_to_angle_deg(lb))

        self.drag_kind = None
        self.drag_start = None
        self.drag_end = None
        self._refresh_canvas()

    def _on_right_click(self, event: tk.Event) -> None:
        if self.tool_var.get() != "point" or self.roi is None:
            return
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch View Level to 0.")
            return
        x, y = self._canvas_to_image(event)
        if self._delete_nearest(x, y):
            self.status_var.set("Point deleted.")
        self._refresh_canvas()

    def _on_motion(self, event: tk.Event) -> None:
        if self.tool_var.get() != "point":
            return
        x, y = self._canvas_to_image(event)
        self._update_hover(x, y)
        if self.drag_kind == "point":
            self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        old_z = float(self.zoom_var.get())
        delta = wheel_delta(event)
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        new_z = max(ZOOM_MIN, min(ZOOM_MAX, old_z * factor))
        if abs(new_z - old_z) < 1e-9:
            return
        self.zoom_var.set(new_z)
        self._refresh_canvas()
        keep_canvas_point_stable_after_zoom(
            self.canvas,
            event,
            old_zoom=old_z,
            new_zoom=new_z,
            content_w=self._last_render_w,
            content_h=self._last_render_h,
        )

    def _render_image(self) -> np.ndarray:
        if self.image_bgr is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "Select image", (80, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
            return blank

        if self._in_roi_view():
            canvas = self._level_roi_image().copy()
            tl = self._current_level_template()
            if tl is not None:
                x1 = int(tl.tl_x)
                y1 = int(tl.tl_y)
                x2 = int(tl.tl_x + tl.width)
                y2 = int(tl.tl_y + tl.height)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
                pts = self._get_abs_points()
                palette = orientation_palette_bgr()
                for i, (px, py) in enumerate(pts):
                    f = tl.features[i]
                    color = palette[int(f.label) % 8]
                    theta = float(f.theta) if np.isfinite(f.theta) else label_to_angle_deg(int(f.label))
                    p2 = arrow_endpoint(int(px), int(py), theta, 8.0)
                    if self.hover_index is not None and i == self.hover_index:
                        cv2.circle(canvas, (int(px), int(py)), 6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.arrowedLine(canvas, (int(px), int(py)), p2, color, 1, cv2.LINE_AA, 0, 0.35)
                    cv2.circle(canvas, (int(px), int(py)), 1, color, -1, cv2.LINE_AA)

            if self.drag_kind == "point" and self.drag_start is not None and self.drag_end is not None:
                sx, sy = self.drag_start
                ex, ey = self.drag_end
                dx = ex - sx
                dy = ey - sy
                if abs(dx) + abs(dy) > 0:
                    lb = angle_deg_to_label(np.degrees(np.arctan2(float(dy), float(dx))))
                else:
                    lb = int(self.label_var.get()) % 8
                color = orientation_palette_bgr()[lb]
                cv2.arrowedLine(canvas, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA, 0, 0.35)
            return canvas

        canvas = self.image_bgr.copy()
        if self.roi is not None:
            cv2.rectangle(
                canvas,
                (int(self.roi.x), int(self.roi.y)),
                (int(self.roi.x + self.roi.w), int(self.roi.y + self.roi.h)),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.drag_kind == "roi" and self.drag_start is not None and self.drag_end is not None:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_end
            cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, cv2.LINE_AA)
        return canvas

    def _refresh_canvas(self) -> None:
        img = self._render_image()
        z = max(ZOOM_MIN, float(self.zoom_var.get()))
        if abs(z - 1.0) > 1e-6:
            img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        self._photo = bgr_to_photo(img)
        w = int(img.shape[1])
        h = int(img.shape[0])
        self._last_render_w = w
        self._last_render_h = h
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))

class EditTemplateTab(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        self.detector: Optional[Line2DupLikeDetector] = None
        self.template_image: Optional[np.ndarray] = None
        self.model_path = ""
        self.undo_stack: List[UndoItem] = []

        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.is_dragging = False
        self.hover_index: Optional[int] = None

        self._photo: Optional[tk.PhotoImage] = None
        self._canvas_image_id: Optional[int] = None
        self._last_render_w = 320
        self._last_render_h = 240

        self._build_ui()
        self._refresh_canvas()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        control = ttk.Frame(self, padding=8)
        control.grid(row=0, column=0, sticky="ns")
        view = ttk.Frame(self, padding=(0, 8, 8, 8))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Load model to start.")
        self.class_var = tk.StringVar(value="")
        self.template_id_var = tk.IntVar(value=0)
        self.level_var = tk.IntVar(value=0)
        self.label_var = tk.IntVar(value=0)
        self.zoom_var = tk.DoubleVar(value=2.0)
        self.out_model_var = tk.StringVar(value="")

        row = 0
        ttk.Label(control, text="Edit Existing Model").grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Open Model...", command=self.open_model).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1
        self.model_label = ttk.Label(control, text="(no model)", width=36, wraplength=260)
        self.model_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Template Image (optional)...", command=self.open_template_image).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        self.template_img_label = ttk.Label(control, text="(none)", width=36, wraplength=260)
        self.template_img_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Output Model...", command=self.select_output_model).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        self.out_label = ttk.Label(control, text="(overwrite input model)", width=36, wraplength=260)
        self.out_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Separator(control, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=6)
        row += 1

        ttk.Label(control, text="Class").grid(row=row, column=0, sticky="w")
        row += 1
        self.class_combo = ttk.Combobox(control, textvariable=self.class_var, values=[], state="readonly", width=24)
        self.class_combo.grid(row=row, column=0, sticky="ew")
        self.class_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_class_changed())
        row += 1

        ttk.Label(control, text="Template ID").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self.template_spin = ttk.Spinbox(
            control,
            from_=0,
            to=0,
            increment=1,
            textvariable=self.template_id_var,
            width=10,
            command=self._refresh_canvas,
        )
        self.template_spin.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Level").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self.level_spin = ttk.Spinbox(control, from_=0, to=0, increment=1, textvariable=self.level_var, width=10, command=self._refresh_canvas)
        self.level_spin.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Label For Short Drag").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=0, to=7, increment=1, textvariable=self.label_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Zoom").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Scale(control, from_=ZOOM_MIN, to=ZOOM_MAX, orient="horizontal", variable=self.zoom_var, command=lambda _e: self._refresh_canvas()).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Button(control, text="Undo", command=self.undo).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        ttk.Button(control, text="Save Model", command=self.save_model).grid(row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1

        ttk.Separator(control, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=8)
        row += 1
        ttk.Label(control, textvariable=self.status_var, wraplength=260, foreground="#2f7d32").grid(row=row, column=0, sticky="w")

        self.canvas = tk.Canvas(view, bg="#1d1d1d", highlightthickness=0)
        xbar = ttk.Scrollbar(view, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(view, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_motion)

    def open_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Open model",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            detector = load_detector_model(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        class_ids = detector.class_ids()
        if not class_ids:
            messagebox.showerror("Invalid model", "No classes found in model.")
            return

        self.detector = detector
        self.model_path = path
        self.model_label.configure(text=path)
        self.out_model_var.set(path)
        self.out_label.configure(text=path)
        self.class_combo.configure(values=class_ids)
        self.class_var.set(class_ids[0])
        self.undo_stack = []
        self._on_class_changed()
        self.status_var.set("Model loaded. Left-drag to add directional point, right click to delete.")
        self._refresh_canvas()

    def open_template_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open template image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Read error", f"Failed to read image:\n{path}")
            return
        self.template_image = img
        self.template_img_label.configure(text=path)
        self.status_var.set("Template image loaded for background preview.")
        self._refresh_canvas()

    def select_output_model(self) -> None:
        initial = self.out_model_var.get().strip() or self.model_path
        path = filedialog.asksaveasfilename(
            title="Output model path",
            initialfile=Path(initial).name if initial else "template_model.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.out_model_var.set(path)
        self.out_label.configure(text=path)

    def _on_class_changed(self) -> None:
        if self.detector is None:
            return
        class_id = self.class_var.get().strip()
        if not class_id:
            return
        if class_id not in self.detector.class_templates:
            return
        tcnt = len(self.detector.class_templates[class_id])
        self.template_spin.configure(from_=0, to=max(0, tcnt - 1))
        self.template_id_var.set(0)
        self.level_spin.configure(from_=0, to=max(0, self.detector.pyramid_levels - 1))
        self.level_var.set(0)
        self.hover_index = None
        self._refresh_canvas()

    def _current_template_level(self) -> Optional[TemplateLevel]:
        if self.detector is None:
            return None
        class_id = self.class_var.get().strip()
        if not class_id or class_id not in self.detector.class_templates:
            return None
        tcnt = len(self.detector.class_templates[class_id])
        if tcnt <= 0:
            return None
        tid = max(0, min(int(self.template_id_var.get()), tcnt - 1))
        self.template_id_var.set(tid)
        lv = max(0, min(int(self.level_var.get()), self.detector.pyramid_levels - 1))
        self.level_var.set(lv)
        return self.detector.get_templates(class_id, tid)[lv]

    def _get_abs_points(self) -> List[Tuple[int, int]]:
        tl = self._current_template_level()
        if tl is None:
            return []
        return [(int(f.x + tl.tl_x), int(f.y + tl.tl_y)) for f in tl.features]

    def _push_undo(self) -> None:
        tl = self._current_template_level()
        if tl is None:
            return
        class_id = self.class_var.get().strip()
        tid = int(self.template_id_var.get())
        lv = int(self.level_var.get())
        snap = [Feature(x=int(f.x), y=int(f.y), label=int(f.label), theta=float(f.theta)) for f in tl.features]
        self.undo_stack.append(UndoItem(class_id=class_id, template_id=tid, level=lv, features=snap))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if self.detector is None or not self.undo_stack:
            return
        item = self.undo_stack.pop()
        if item.class_id not in self.detector.class_templates:
            return
        if item.template_id < 0 or item.template_id >= len(self.detector.class_templates[item.class_id]):
            return
        if item.level < 0 or item.level >= self.detector.pyramid_levels:
            return
        self.detector.class_templates[item.class_id][item.template_id][item.level].features = item.features
        self.class_var.set(item.class_id)
        self._on_class_changed()
        self.template_id_var.set(item.template_id)
        self.level_var.set(item.level)
        self.status_var.set("Undo complete.")
        self._refresh_canvas()

    def _update_hover(self, x_abs: int, y_abs: int) -> None:
        tl = self._current_template_level()
        if tl is None or not tl.features:
            self.hover_index = None
            return
        best_idx = -1
        best_d2 = 1e18
        for i, (px, py) in enumerate(self._get_abs_points()):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx >= 0 and best_d2 <= 8.0 * 8.0:
            self.hover_index = best_idx
        else:
            self.hover_index = None

    def _add_feature(self, x_abs: int, y_abs: int, label: int, theta_deg: Optional[float]) -> None:
        tl = self._current_template_level()
        if tl is None:
            return
        self._push_undo()
        xr = int(round(x_abs - tl.tl_x))
        yr = int(round(y_abs - tl.tl_y))
        xr = max(0, min(xr, int(tl.width)))
        yr = max(0, min(yr, int(tl.height)))
        lb = int(label) % 8
        theta = label_to_angle_deg(lb) if theta_deg is None else float(theta_deg)
        tl.features.append(Feature(x=xr, y=yr, label=lb, theta=theta))

    def _delete_nearest(self, x_abs: int, y_abs: int) -> bool:
        tl = self._current_template_level()
        if tl is None or not tl.features:
            return False
        best_idx = -1
        best_d2 = 1e18
        pts = self._get_abs_points()
        for i, (px, py) in enumerate(pts):
            dx = float(px - x_abs)
            dy = float(py - y_abs)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx < 0 or best_d2 > 8.0 * 8.0:
            return False
        self._push_undo()
        del tl.features[best_idx]
        self.hover_index = None
        return True

    def save_model(self) -> None:
        if self.detector is None:
            messagebox.showwarning("No model", "Please open model first.")
            return
        out_path = self.out_model_var.get().strip() or self.model_path
        if not out_path:
            self.select_output_model()
            out_path = self.out_model_var.get().strip() or self.model_path
            if not out_path:
                return
        save_detector_model(self.detector, out_path)
        self.status_var.set(f"Saved model: {out_path}")
        messagebox.showinfo("Saved", f"Model saved:\n{out_path}")

    def _canvas_to_image(self, event: tk.Event) -> Tuple[int, int]:
        z = max(1e-6, float(self.zoom_var.get()))
        xw = float(self.canvas.canvasx(event.x))
        yw = float(self.canvas.canvasy(event.y))
        return int(round(xw / z)), int(round(yw / z))

    def _on_left_down(self, event: tk.Event) -> None:
        if self.detector is None:
            return
        x, y = self._canvas_to_image(event)
        self.is_dragging = True
        self.drag_start = (x, y)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_left_move(self, event: tk.Event) -> None:
        if not self.is_dragging:
            return
        x, y = self._canvas_to_image(event)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_left_up(self, event: tk.Event) -> None:
        if not self.is_dragging or self.drag_start is None:
            return
        x, y = self._canvas_to_image(event)
        self.drag_end = (x, y)
        sx, sy = self.drag_start
        ex, ey = self.drag_end
        dx = ex - sx
        dy = ey - sy
        dist = float(np.hypot(dx, dy))
        if dist >= 2.0:
            theta = float(np.degrees(np.arctan2(float(dy), float(dx))))
            lb = angle_deg_to_label(theta)
            self.label_var.set(lb)
            self._add_feature(sx, sy, lb, theta_deg=theta)
        else:
            lb = int(self.label_var.get()) % 8
            self._add_feature(sx, sy, lb, theta_deg=label_to_angle_deg(lb))
        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None
        self._refresh_canvas()

    def _on_right_click(self, event: tk.Event) -> None:
        if self.detector is None:
            return
        x, y = self._canvas_to_image(event)
        if self._delete_nearest(x, y):
            self.status_var.set("Point deleted.")
        self._refresh_canvas()

    def _on_motion(self, event: tk.Event) -> None:
        if self.detector is None:
            return
        x, y = self._canvas_to_image(event)
        self._update_hover(x, y)
        if self.is_dragging:
            self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        old_z = float(self.zoom_var.get())
        delta = wheel_delta(event)
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        new_z = max(ZOOM_MIN, min(ZOOM_MAX, old_z * factor))
        if abs(new_z - old_z) < 1e-9:
            return
        self.zoom_var.set(new_z)
        self._refresh_canvas()
        keep_canvas_point_stable_after_zoom(
            self.canvas,
            event,
            old_zoom=old_z,
            new_zoom=new_z,
            content_w=self._last_render_w,
            content_h=self._last_render_h,
        )

    def _render_background(self) -> np.ndarray:
        tl = self._current_template_level()
        if tl is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "Open model", (90, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
            return blank

        if self.template_image is None:
            max_x = int(tl.tl_x + tl.width + 20)
            max_y = int(tl.tl_y + tl.height + 20)
            for px, py in self._get_abs_points():
                max_x = max(max_x, px + 20)
                max_y = max(max_y, py + 20)
            max_x = max(max_x, 240)
            max_y = max(max_y, 160)
            return np.zeros((max_y, max_x, 3), dtype=np.uint8)

        if self.detector is None:
            return self.template_image.copy()
        class_id = self.class_var.get().strip()
        tid = int(self.template_id_var.get())
        meta = self.detector.get_template_meta(class_id, tid)
        angle = float(meta.get("angle", 0.0))
        scale = float(meta.get("scale", 1.0))
        img = ShapeInfoProducer.transform(self.template_image, angle, scale)
        lv = int(self.level_var.get())
        for _ in range(max(0, lv)):
            if img.shape[0] < 2 or img.shape[1] < 2:
                break
            img = cv2.pyrDown(img)
        return img

    def _render_image(self) -> np.ndarray:
        canvas = self._render_background().copy()
        tl = self._current_template_level()
        if tl is None:
            return canvas

        x1 = int(tl.tl_x)
        y1 = int(tl.tl_y)
        x2 = int(tl.tl_x + tl.width)
        y2 = int(tl.tl_y + tl.height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

        pts = self._get_abs_points()
        palette = orientation_palette_bgr()
        for i, (px, py) in enumerate(pts):
            f = tl.features[i]
            color = palette[int(f.label) % 8]
            theta = float(f.theta) if np.isfinite(f.theta) else label_to_angle_deg(int(f.label))
            p2 = arrow_endpoint(int(px), int(py), theta, 8.0)
            if self.hover_index is not None and i == self.hover_index:
                cv2.circle(canvas, (int(px), int(py)), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (int(px), int(py)), p2, color, 1, cv2.LINE_AA, 0, 0.35)
            cv2.circle(canvas, (int(px), int(py)), 1, color, -1, cv2.LINE_AA)

        if self.is_dragging and self.drag_start is not None and self.drag_end is not None:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            dx = ex - sx
            dy = ey - sy
            if abs(dx) + abs(dy) > 0:
                lb = angle_deg_to_label(np.degrees(np.arctan2(float(dy), float(dx))))
            else:
                lb = int(self.label_var.get()) % 8
            color = orientation_palette_bgr()[lb]
            cv2.arrowedLine(canvas, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA, 0, 0.35)
        return canvas

    def _refresh_canvas(self) -> None:
        img = self._render_image()
        z = max(ZOOM_MIN, float(self.zoom_var.get()))
        if abs(z - 1.0) > 1e-6:
            img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        self._photo = bgr_to_photo(img)
        w = int(img.shape[1])
        h = int(img.shape[0])
        self._last_render_w = w
        self._last_render_h = h
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))


class FindTemplateTab(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.detector: Optional[Line2DupLikeDetector] = None
        self.model_path = ""
        self.scene_bgr: Optional[np.ndarray] = None
        self.scene_path = ""
        self.scene_mask: Optional[np.ndarray] = None
        self.scene_mask_path = ""
        self.result_bgr: Optional[np.ndarray] = None

        self._photo: Optional[tk.PhotoImage] = None
        self._canvas_image_id: Optional[int] = None
        self._last_render_w = 320
        self._last_render_h = 240

        self._build_ui()
        self._refresh_canvas()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        control = ttk.Frame(self, padding=8)
        control.grid(row=0, column=0, sticky="ns")
        view = ttk.Frame(self, padding=(0, 8, 8, 8))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Load model and scene, then click Run Match.")
        self.class_choice_var = tk.StringVar(value="ALL")
        self.threshold_var = tk.DoubleVar(value=90.0)
        self.crop_stride_var = tk.IntVar(value=0)
        self.nms_iou_var = tk.DoubleVar(value=0.50)
        self.topk_var = tk.IntVar(value=20)
        self.auto_sweep_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.DoubleVar(value=1.5)
        self.out_image_var = tk.StringVar(value="")

        row = 0
        ttk.Label(control, text="Find With Model").grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Open Model...", command=self.open_model).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1
        self.model_label = ttk.Label(control, text="(no model)", width=36, wraplength=260)
        self.model_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Open Scene...", command=self.open_scene).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        self.scene_label = ttk.Label(control, text="(no scene)", width=36, wraplength=260)
        self.scene_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Open Scene Mask (optional)...", command=self.open_scene_mask).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        self.mask_label = ttk.Label(control, text="(none)", width=36, wraplength=260)
        self.mask_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Output Image...", command=self.select_output_image).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        self.out_label = ttk.Label(control, text="(not set)", width=36, wraplength=260)
        self.out_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Separator(control, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=6)
        row += 1

        ttk.Label(control, text="Class").grid(row=row, column=0, sticky="w")
        row += 1
        self.class_combo = ttk.Combobox(control, textvariable=self.class_choice_var, values=["ALL"], state="readonly", width=24)
        self.class_combo.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(control, text="Threshold").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=0, to=100, increment=1, textvariable=self.threshold_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Crop Stride (0=off)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=0, to=256, increment=1, textvariable=self.crop_stride_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="NMS IoU").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=0.0, to=1.0, increment=0.05, textvariable=self.nms_iou_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(control, text="Top K").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=1, to=500, increment=1, textvariable=self.topk_var, width=10).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Checkbutton(control, text="Auto Threshold Sweep", variable=self.auto_sweep_var).grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1

        ttk.Label(control, text="Zoom").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Scale(control, from_=ZOOM_MIN, to=ZOOM_MAX, orient="horizontal", variable=self.zoom_var, command=lambda _e: self._refresh_canvas()).grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Button(control, text="Run Match", command=self.run_match).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        ttk.Button(control, text="Save Overlay", command=self.save_overlay).grid(row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1

        ttk.Label(control, text="Top Results").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.result_list = tk.Listbox(control, width=44, height=8)
        self.result_list.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Separator(control, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=8)
        row += 1
        ttk.Label(control, textvariable=self.status_var, wraplength=260, foreground="#2f7d32").grid(row=row, column=0, sticky="w")

        self.canvas = tk.Canvas(view, bg="#1d1d1d", highlightthickness=0)
        xbar = ttk.Scrollbar(view, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(view, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

    def _reset_class_combo(self) -> None:
        if self.detector is None:
            self.class_combo.configure(values=["ALL"])
            self.class_choice_var.set("ALL")
            return
        classes = self.detector.class_ids()
        values = ["ALL"] + classes
        self.class_combo.configure(values=values)
        if self.class_choice_var.get() not in values:
            self.class_choice_var.set("ALL")

    def open_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Open model",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_model(path)

    def load_model(self, path: str) -> bool:
        try:
            detector = load_detector_model(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return False
        classes = detector.class_ids()
        if not classes:
            messagebox.showerror("Invalid model", "No classes found in model.")
            return False
        self.detector = detector
        self.model_path = path
        self.model_label.configure(text=path)
        self._reset_class_combo()
        self.status_var.set("Model loaded.")
        return True

    def open_scene(self) -> None:
        path = filedialog.askopenfilename(
            title="Open scene image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Read error", f"Failed to read scene:\n{path}")
            return
        self.scene_bgr = img
        self.scene_path = path
        self.scene_label.configure(text=path)
        self.result_bgr = None
        if not self.out_image_var.get().strip():
            default_out = str(Path(path).with_name(f"{Path(path).stem}_find_result.png"))
            self.out_image_var.set(default_out)
            self.out_label.configure(text=default_out)
        self.status_var.set("Scene loaded. Click Run Match.")
        self._refresh_canvas()

    def open_scene_mask(self) -> None:
        path = filedialog.askopenfilename(
            title="Open scene mask",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            messagebox.showerror("Read error", f"Failed to read mask:\n{path}")
            return
        self.scene_mask = m
        self.scene_mask_path = path
        self.mask_label.configure(text=path)
        self.status_var.set("Scene mask loaded.")

    def select_output_image(self) -> None:
        initial = self.out_image_var.get().strip()
        if not initial and self.scene_path:
            initial = str(Path(self.scene_path).with_name(f"{Path(self.scene_path).stem}_find_result.png"))
        path = filedialog.asksaveasfilename(
            title="Output image path",
            initialfile=Path(initial).name if initial else "find_result.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return
        self.out_image_var.set(path)
        self.out_label.configure(text=path)

    def run_match(self) -> None:
        if self.detector is None:
            messagebox.showwarning("No model", "Please open model first.")
            return
        if self.scene_bgr is None:
            messagebox.showwarning("No scene", "Please open scene image first.")
            return

        threshold = float(np.clip(self.threshold_var.get(), 0.0, 100.0))
        crop_stride = int(max(0, self.crop_stride_var.get()))
        nms_iou = float(np.clip(self.nms_iou_var.get(), 0.0, 1.0))
        topk = int(max(1, self.topk_var.get()))

        scene_for_match = self.scene_bgr
        if crop_stride > 0:
            h, w = scene_for_match.shape[:2]
            h2 = (h // crop_stride) * crop_stride
            w2 = (w // crop_stride) * crop_stride
            if h2 > 0 and w2 > 0:
                scene_for_match = scene_for_match[:h2, :w2].copy()

        class_choice = self.class_choice_var.get().strip()
        if class_choice == "" or class_choice == "ALL":
            class_ids = self.detector.class_ids()
        else:
            class_ids = [class_choice]

        def match_once(th: float):
            ms = self.detector.match(
                scene_for_match,
                threshold=th,
                class_ids=class_ids,
                mask=self.scene_mask,
            )
            ms = nms_matches(self.detector, ms, iou_threshold=nms_iou)
            ms.sort(key=lambda m: m.similarity, reverse=True)
            return ms

        used_threshold = threshold
        matches = match_once(used_threshold)
        if not matches and self.auto_sweep_var.get():
            cur = int(np.floor(used_threshold)) - 5
            while cur >= 20:
                cand = match_once(float(cur))
                if cand:
                    matches = cand
                    used_threshold = float(cur)
                    break
                cur -= 5

        overlay = draw_matches(self.detector, scene_for_match, matches, topk=topk)
        self.result_bgr = overlay
        self._refresh_canvas()

        self.result_list.delete(0, tk.END)
        shown = min(topk, len(matches))
        for i in range(shown):
            m = matches[i]
            meta = self.detector.get_template_meta(m.class_id, m.template_id)
            line = (
                f"#{i+1} s={m.similarity:.1f} cls={m.class_id} "
                f"x={m.x} y={m.y} a={float(meta.get('angle', 0.0)):.0f} "
                f"sc={float(meta.get('scale', 1.0)):.2f}"
            )
            self.result_list.insert(tk.END, line)
        if shown == 0:
            self.result_list.insert(tk.END, "No match.")

        self.status_var.set(
            f"Match done. total={len(matches)}, shown={shown}, threshold_used={used_threshold:.1f}, "
            f"stride={crop_stride}."
        )

    def save_overlay(self) -> None:
        if self.result_bgr is None:
            messagebox.showwarning("No result", "Please run match first.")
            return
        out_path = self.out_image_var.get().strip()
        if not out_path:
            self.select_output_image()
            out_path = self.out_image_var.get().strip()
            if not out_path:
                return
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out), self.result_bgr)
        if not ok:
            messagebox.showerror("Save error", f"Failed to write:\n{out_path}")
            return
        self.status_var.set(f"Overlay saved: {out_path}")
        messagebox.showinfo("Saved", f"Overlay saved:\n{out_path}")

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        old_z = float(self.zoom_var.get())
        delta = wheel_delta(event)
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        new_z = max(ZOOM_MIN, min(ZOOM_MAX, old_z * factor))
        if abs(new_z - old_z) < 1e-9:
            return
        self.zoom_var.set(new_z)
        self._refresh_canvas()
        keep_canvas_point_stable_after_zoom(
            self.canvas,
            event,
            old_zoom=old_z,
            new_zoom=new_z,
            content_w=self._last_render_w,
            content_h=self._last_render_h,
        )

    def _render_image(self) -> np.ndarray:
        if self.result_bgr is not None:
            return self.result_bgr.copy()
        if self.scene_bgr is not None:
            return self.scene_bgr.copy()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "Open model + scene", (52, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        return blank

    def _refresh_canvas(self) -> None:
        img = self._render_image()
        z = max(ZOOM_MIN, float(self.zoom_var.get()))
        if abs(z - 1.0) > 1e-6:
            img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        self._photo = bgr_to_photo(img)
        w = int(img.shape[1])
        h = int(img.shape[0])
        self._last_render_w = w
        self._last_render_h = h
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))


class TemplateWorkbenchApp(tk.Tk):
    def __init__(
        self,
        preload_image: str = "",
        preload_model: str = "",
        preload_template_image: str = "",
        preload_out_model: str = "",
        preload_scene: str = "",
    ) -> None:
        super().__init__()
        self.title("line2dup Template Workbench")
        self.geometry("1450x900")

        self._build_menu()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.create_tab = CreateTemplateTab(notebook)
        self.edit_tab = EditTemplateTab(notebook)
        self.find_tab = FindTemplateTab(notebook)
        notebook.add(self.create_tab, text="Create Template")
        notebook.add(self.edit_tab, text="Edit Model")
        notebook.add(self.find_tab, text="Find")

        if preload_image:
            img = cv2.imread(preload_image, cv2.IMREAD_COLOR)
            if img is not None:
                self.create_tab.image_bgr = img
                self.create_tab.image_path = preload_image
                self.create_tab.image_label.configure(text=preload_image)
                self.create_tab.status_var.set("Image preloaded. Drag left mouse to select ROI.")
                if preload_out_model:
                    self.create_tab.out_model_var.set(preload_out_model)
                    self.create_tab.out_label.configure(text=preload_out_model)
                self.create_tab._refresh_canvas()

        if preload_model:
            try:
                det = load_detector_model(preload_model)
                self.edit_tab.detector = det
                self.edit_tab.model_path = preload_model
                self.edit_tab.model_label.configure(text=preload_model)
                out_path = preload_out_model or preload_model
                self.edit_tab.out_model_var.set(out_path)
                self.edit_tab.out_label.configure(text=out_path)
                class_ids = det.class_ids()
                if class_ids:
                    self.edit_tab.class_combo.configure(values=class_ids)
                    self.edit_tab.class_var.set(class_ids[0])
                    self.edit_tab._on_class_changed()
                self.find_tab.detector = det
                self.find_tab.model_path = preload_model
                self.find_tab.model_label.configure(text=preload_model)
                self.find_tab._reset_class_combo()
            except Exception:
                pass

        if preload_template_image:
            img = cv2.imread(preload_template_image, cv2.IMREAD_COLOR)
            if img is not None:
                self.edit_tab.template_image = img
                self.edit_tab.template_img_label.configure(text=preload_template_image)
                self.edit_tab._refresh_canvas()

        if preload_scene:
            scene = cv2.imread(preload_scene, cv2.IMREAD_COLOR)
            if scene is not None:
                self.find_tab.scene_bgr = scene
                self.find_tab.scene_path = preload_scene
                self.find_tab.scene_label.configure(text=preload_scene)
                if preload_out_model:
                    out_img = str(Path(preload_out_model).with_suffix(".png"))
                else:
                    out_img = str(Path(preload_scene).with_name(f"{Path(preload_scene).stem}_find_result.png"))
                self.find_tab.out_image_var.set(out_img)
                self.find_tab.out_label.configure(text=out_img)
                self.find_tab._refresh_canvas()

    def _build_menu(self) -> None:
        bar = tk.Menu(self)
        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Open Image (Create Tab)", command=self._menu_open_image)
        file_menu.add_command(label="Open Model (Edit Tab)", command=self._menu_open_model)
        file_menu.add_command(label="Open Model (Find Tab)", command=self._menu_open_model_find)
        file_menu.add_command(label="Open Scene (Find Tab)", command=self._menu_open_scene_find)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        bar.add_cascade(label="File", menu=file_menu)
        self.config(menu=bar)

    def _menu_open_image(self) -> None:
        self.create_tab.select_image()

    def _menu_open_model(self) -> None:
        self.edit_tab.open_model()

    def _menu_open_model_find(self) -> None:
        self.find_tab.open_model()

    def _menu_open_scene_find(self) -> None:
        self.find_tab.open_scene()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified GUI workbench for line2Dup-like template making/editing.")
    parser.add_argument("--image", default="", help="Optional preload image for Create tab.")
    parser.add_argument("--model", default="", help="Optional preload model for Edit tab.")
    parser.add_argument("--scene", default="", help="Optional preload scene image for Find tab.")
    parser.add_argument("--template-image", default="", help="Optional preload template image for Edit tab.")
    parser.add_argument("--out-model", default="", help="Optional preload output model path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = TemplateWorkbenchApp(
        preload_image=args.image,
        preload_model=args.model,
        preload_scene=args.scene,
        preload_template_image=args.template_image,
        preload_out_model=args.out_model,
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
