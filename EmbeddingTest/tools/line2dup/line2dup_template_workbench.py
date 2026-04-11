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
import sys
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from line2dup.like_matcher import (
    Feature,
    Line2DupLikeDetector,
    ShapeInfoProducer,
    TemplateLevel,
    clone_template_levels,
    create_native_detector,
    crop_templates,
    decode_png_base64,
    draw_matches,
    encode_png_base64,
    ensure_native_backends_available,
    label_to_theta_deg as matcher_label_to_theta_deg,
    load_detector_model,
    nms_matches,
    refine_matches_sim3,
    save_detector_model,
    theta_deg_to_label as matcher_theta_deg_to_label,
)

ZOOM_MIN = 0.2
ZOOM_MAX = 16.0
SIDEBAR_WIDTH = 320
BACKEND_ITEMS = [
    ("Original", "original"),
    ("Fusion", "fusion"),
    ("Fusion V2", "fusionv2"),
    ("ICP (sim3)", "sim3"),
]
BACKEND_LABEL_TO_KEY = {label: key for label, key in BACKEND_ITEMS}
BACKEND_KEY_TO_LABEL = {key: label for label, key in BACKEND_ITEMS}


class ScrollableSidebar(ttk.Frame):
    def __init__(self, master: tk.Misc, *, width: int = SIDEBAR_WIDTH, padding: int | Tuple[int, int, int, int] = 8) -> None:
        super().__init__(master)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, width=int(width))
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="ns")
        self.vbar.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas, padding=padding)
        self.content.columnconfigure(0, weight=1)
        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=max(1, int(event.width)))


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
    return clone_template_levels(levels)


def build_mask_from_rects(width: int, height: int, mask_rects: Sequence[MaskRect]) -> np.ndarray:
    mask = np.full((max(1, int(height)), max(1, int(width))), 255, dtype=np.uint8)
    for rect in mask_rects:
        x1 = max(0, min(int(rect.x), int(width)))
        y1 = max(0, min(int(rect.y), int(height)))
        x2 = max(x1, min(int(rect.x + rect.w), int(width)))
        y2 = max(y1, min(int(rect.y + rect.h), int(height)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask


def pose_infos_from_ui_values(
    angle_start: float,
    angle_end: float,
    angle_step: float,
    scale_start: float,
    scale_end: float,
    scale_step: float,
) -> List[Tuple[float, float]]:
    if scale_start <= 0.0 or scale_end <= 0.0:
        raise ValueError("scale start/end must be > 0")

    angles = expand_numeric_range(angle_start, angle_end, angle_step)
    scales = expand_numeric_range(scale_start, scale_end, scale_step)
    infos: List[Tuple[float, float]] = []
    seen = set()
    for scale in scales:
        for angle in angles:
            angle_norm = float((float(angle) % 360.0 + 360.0) % 360.0)
            if abs(angle_norm - 360.0) < 1e-9:
                angle_norm = 0.0
            key = (round(angle_norm, 6), round(float(scale), 6))
            if key in seen:
                continue
            seen.add(key)
            infos.append((angle_norm, float(scale)))
    if not infos:
        infos = [(0.0, 1.0)]
    return infos


def roi_level_shapes_from_image(image_bgr: np.ndarray, total_levels: int) -> List[Tuple[int, int]]:
    img = image_bgr.copy()
    shapes: List[Tuple[int, int]] = []
    for _ in range(max(1, int(total_levels))):
        h, w = img.shape[:2]
        shapes.append((int(w), int(h)))
        if h < 2 or w < 2:
            continue
        img = cv2.pyrDown(img)
    return shapes


def sync_levels_from_level0(level0: TemplateLevel, shapes: Sequence[Tuple[int, int]]) -> List[TemplateLevel]:
    if not shapes:
        return []
    w0, h0 = shapes[0]
    max_x0 = max(0, int(w0) - 1)
    max_y0 = max(0, int(h0) - 1)
    l0_feats: List[Feature] = []
    l0_seen = set()
    for feature in level0.features:
        x0 = int(np.clip(int(feature.x) + int(level0.tl_x), 0, max_x0))
        y0 = int(np.clip(int(feature.y) + int(level0.tl_y), 0, max_y0))
        key = (x0, y0, int(feature.label) & 7)
        if key in l0_seen:
            continue
        l0_seen.add(key)
        l0_feats.append(Feature(x=x0, y=y0, label=int(feature.label) & 7, theta=float(feature.theta)))

    out: List[TemplateLevel] = [
        TemplateLevel(
            width=max_x0,
            height=max_y0,
            tl_x=0,
            tl_y=0,
            pyramid_level=0,
            features=l0_feats,
        )
    ]

    for level_index in range(1, len(shapes)):
        w, h = shapes[level_index]
        max_x = max(0, int(w) - 1)
        max_y = max(0, int(h) - 1)
        div = float(1 << level_index)
        feats: List[Feature] = []
        seen = set()
        for feature in l0_feats:
            x = int(round(float(feature.x) / div))
            y = int(round(float(feature.y) / div))
            x = int(np.clip(x, 0, max_x))
            y = int(np.clip(y, 0, max_y))
            key = (x, y, int(feature.label) & 7)
            if key in seen:
                continue
            seen.add(key)
            feats.append(Feature(x=x, y=y, label=int(feature.label) & 7, theta=float(feature.theta)))
        out.append(
            TemplateLevel(
                width=max_x,
                height=max_y,
                tl_x=0,
                tl_y=0,
                pyramid_level=level_index,
                features=feats,
            )
        )
    return out


def normalize_extracted_levels_to_roi(levels: Sequence[TemplateLevel], roi_image_bgr: np.ndarray) -> List[TemplateLevel]:
    if not levels:
        return []
    shapes = roi_level_shapes_from_image(roi_image_bgr, len(levels))
    if not shapes:
        return []
    level0 = clone_levels([levels[0]])[0]
    w0, h0 = shapes[0]
    max_x0 = max(0, int(w0) - 1)
    max_y0 = max(0, int(h0) - 1)
    for feature in level0.features:
        feature.x = int(np.clip(int(feature.x) + int(level0.tl_x), 0, max_x0))
        feature.y = int(np.clip(int(feature.y) + int(level0.tl_y), 0, max_y0))
    level0.tl_x = 0
    level0.tl_y = 0
    level0.width = max_x0
    level0.height = max_y0
    return sync_levels_from_level0(level0, shapes)


def transform_levels_for_pose(
    base_levels: Sequence[TemplateLevel],
    angle_deg: float,
    scale: float,
    *,
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
        for feature in base.features:
            px = float(feature.x + base.tl_x)
            py = float(feature.y + base.tl_y)
            rx, ry = _xf(px, py)
            xi = int(round(rx))
            yi = int(round(ry))
            base_theta = float(feature.theta)
            if not np.isfinite(base_theta):
                base_theta = label_to_angle_deg(int(feature.label))
            theta = (base_theta - float(angle_deg)) % 360.0
            label = angle_deg_to_label(theta)
            xr = int(xi - canvas_x_min)
            yr = int(yi - canvas_y_min)
            if xr < 0 or yr < 0:
                continue
            key = (xr, yr, label)
            if key in seen:
                continue
            seen.add(key)
            feats.append(Feature(x=xr, y=yr, label=label, theta=theta))

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
    if auto_crop and out and all(len(level.features) > 0 for level in out):
        crop_templates(out)
    return out


def transform_image_and_mask_expanded(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    center = (float(w) * 0.5, float(h) * 0.5)
    mat = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))

    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(w - 1), 0.0, 1.0],
            [float(w - 1), float(h - 1), 1.0],
            [0.0, float(h - 1), 1.0],
        ],
        dtype=np.float32,
    )
    transformed = (mat @ corners.T).T
    min_x = float(np.floor(np.min(transformed[:, 0])))
    min_y = float(np.floor(np.min(transformed[:, 1])))
    max_x = float(np.ceil(np.max(transformed[:, 0])))
    max_y = float(np.ceil(np.max(transformed[:, 1])))
    new_w = max(1, int(max_x - min_x + 1.0))
    new_h = max(1, int(max_y - min_y + 1.0))

    mat[0, 2] -= min_x
    mat[1, 2] -= min_y
    out_img = cv2.warpAffine(image_bgr, mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    out_mask = cv2.warpAffine(mask, mat, (new_w, new_h), flags=cv2.INTER_NEAREST)
    out_mask = (out_mask > 0).astype(np.uint8) * 255
    return out_img, out_mask


def make_class_source_payload(
    roi_img: np.ndarray,
    roi_mask: np.ndarray,
    roi_rect: RoiRect,
    mask_rects: Sequence[MaskRect],
    pose_infos: Sequence[Tuple[float, float]],
    pose_ui: dict,
    original_mode: str,
) -> dict:
    return {
        "source": {
            "roi_png": encode_png_base64(roi_img),
            "mask_png": encode_png_base64(roi_mask),
            "roi_x": int(roi_rect.x),
            "roi_y": int(roi_rect.y),
            "roi_w": int(roi_rect.w),
            "roi_h": int(roi_rect.h),
            "mask_rects": [
                {"x": int(rect.x), "y": int(rect.y), "w": int(rect.w), "h": int(rect.h)}
                for rect in mask_rects
            ],
        },
        "pose_infos": {
            "items": [{"angle": float(angle), "scale": float(scale)} for angle, scale in pose_infos],
            "ui": {
                "angle_start": float(pose_ui.get("angle_start", 0.0)),
                "angle_end": float(pose_ui.get("angle_end", 0.0)),
                "angle_step": float(pose_ui.get("angle_step", 10.0)),
                "scale_start": float(pose_ui.get("scale_start", 1.0)),
                "scale_end": float(pose_ui.get("scale_end", 1.0)),
                "scale_step": float(pose_ui.get("scale_step", 0.05)),
            },
        },
        "original_mode": str(original_mode),
    }


def load_class_source_assets(detector: Line2DupLikeDetector, class_id: str) -> Tuple[dict, np.ndarray, np.ndarray, RoiRect, List[MaskRect]]:
    source_info = detector.get_class_source(class_id)
    source_block = source_info.get("source", {}) if isinstance(source_info, dict) else {}
    if not isinstance(source_block, dict):
        raise ValueError(f"Model class '{class_id}' does not contain editable source information.")
    roi_img = decode_png_base64(str(source_block.get("roi_png", "")), cv2.IMREAD_COLOR)
    roi_mask = decode_png_base64(str(source_block.get("mask_png", "")), cv2.IMREAD_GRAYSCALE)
    if roi_img is None or roi_mask is None:
        raise ValueError(f"Model class '{class_id}' is missing embedded ROI image or mask.")
    roi_rect = RoiRect(
        x=int(source_block.get("roi_x", 0)),
        y=int(source_block.get("roi_y", 0)),
        w=int(source_block.get("roi_w", roi_img.shape[1])),
        h=int(source_block.get("roi_h", roi_img.shape[0])),
    )
    mask_rects = [
        MaskRect(
            x=int(item.get("x", 0)),
            y=int(item.get("y", 0)),
            w=int(item.get("w", 0)),
            h=int(item.get("h", 0)),
        )
        for item in source_block.get("mask_rects", [])
        if isinstance(item, dict)
    ]
    return source_info, roi_img, roi_mask, roi_rect, mask_rects


def build_multi_backend_detector(
    *,
    class_id: str,
    roi_img: np.ndarray,
    roi_rect: RoiRect,
    mask_rects: Sequence[MaskRect],
    pose_infos: Sequence[Tuple[float, float]],
    pose_ui: dict,
    levels: Sequence[int],
    num_features: int,
    weak_threshold: float,
    strong_threshold: float,
    original_mode: str,
    original_editor_levels: Optional[Sequence[TemplateLevel]] = None,
) -> Tuple[Line2DupLikeDetector, int, int]:
    ensure_native_backends_available(("original", "fusion", "fusionv2", "sim3"))
    roi_mask = build_mask_from_rects(roi_img.shape[1], roi_img.shape[0], mask_rects)

    detector = Line2DupLikeDetector(
        num_features=num_features,
        T_levels=levels,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
    )
    detector.set_class_source(
        class_id,
        make_class_source_payload(
            roi_img=roi_img,
            roi_mask=roi_mask,
            roi_rect=roi_rect,
            mask_rects=mask_rects,
            pose_infos=pose_infos,
            pose_ui=pose_ui,
            original_mode=original_mode,
        ),
    )

    if original_editor_levels:
        base_editor_levels = clone_levels(original_editor_levels)
    else:
        original_native_for_editor = create_native_detector(
            num_features=num_features,
            T_levels=levels,
            weak_threshold=weak_threshold,
            strong_threshold=strong_threshold,
            backend="original",
        )
        editor_tid = int(original_native_for_editor.add_template(roi_img, class_id, roi_mask, int(num_features)))
        if editor_tid < 0:
            raise RuntimeError("Failed to extract the base Original template from ROI.")
        editor_levels_raw = original_native_for_editor.export_template_pyramid(class_id, editor_tid)
        base_editor_levels = normalize_extracted_levels_to_roi(
            Line2DupLikeDetector._template_pyramid_from_native(editor_levels_raw),
            roi_img,
        )
    detector.set_original_editor_levels(class_id, base_editor_levels)

    original_native = None
    if original_mode != "manual_points":
        original_native = create_native_detector(
            num_features=num_features,
            T_levels=levels,
            weak_threshold=weak_threshold,
            strong_threshold=strong_threshold,
            backend="original",
        )
    fusion_native = create_native_detector(
        num_features=num_features,
        T_levels=levels,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
        backend="fusion",
    )
    fusionv2_native = create_native_detector(
        num_features=num_features,
        T_levels=levels,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
        backend="fusionv2",
    )
    sim3_native = create_native_detector(
        num_features=num_features,
        T_levels=levels,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
        backend="sim3",
    )

    backend_templates = {backend: [] for backend in BACKEND_LABEL_TO_KEY.values()}
    metas: List[dict] = []
    kept = 0
    skipped = 0

    for angle_deg, scale in pose_infos:
        src_i, mask_i = transform_image_and_mask_expanded(roi_img, roi_mask, float(angle_deg), float(scale))
        nfeat = max(16, int(round(float(num_features) * float(scale))))

        if original_mode == "manual_points":
            original_tp = transform_levels_for_pose(
                base_editor_levels,
                angle_deg=float(angle_deg),
                scale=float(scale),
                auto_crop=False,
                adapt_feature_count=True,
            )
            if (not original_tp) or any(len(level.features) <= 0 for level in original_tp):
                skipped += 1
                continue
        else:
            original_tid = int(original_native.add_template(src_i, class_id, mask_i, nfeat))
            if original_tid < 0:
                skipped += 1
                continue
            original_tp = Line2DupLikeDetector._template_pyramid_from_native(
                original_native.export_template_pyramid(class_id, original_tid)
            )

        fusion_tid = int(fusion_native.add_template(src_i, class_id, mask_i, nfeat))
        if fusion_tid < 0:
            skipped += 1
            continue
        fusionv2_tid = int(fusionv2_native.add_template(src_i, class_id, mask_i, nfeat))
        if fusionv2_tid < 0:
            skipped += 1
            continue
        sim3_tid = int(sim3_native.add_template(src_i, class_id, mask_i, nfeat))
        if sim3_tid < 0:
            skipped += 1
            continue

        fusion_tp = Line2DupLikeDetector._template_pyramid_from_native(
            fusion_native.export_template_pyramid(class_id, fusion_tid)
        )
        fusionv2_tp = Line2DupLikeDetector._template_pyramid_from_native(
            fusionv2_native.export_template_pyramid(class_id, fusionv2_tid)
        )
        sim3_tp = Line2DupLikeDetector._template_pyramid_from_native(
            sim3_native.export_template_pyramid(class_id, sim3_tid)
        )

        backend_templates["original"].append(original_tp)
        backend_templates["fusion"].append(fusion_tp)
        backend_templates["fusionv2"].append(fusionv2_tp)
        backend_templates["sim3"].append(sim3_tp)
        metas.append(
            {
                "angle": float(angle_deg),
                "scale": float(scale),
                "roi_x": int(roi_rect.x),
                "roi_y": int(roi_rect.y),
                "roi_w": int(roi_rect.w),
                "roi_h": int(roi_rect.h),
                "canvas_w": int(src_i.shape[1]),
                "canvas_h": int(src_i.shape[0]),
                "mask_rects": [
                    {"x": int(rect.x), "y": int(rect.y), "w": int(rect.w), "h": int(rect.h)}
                    for rect in mask_rects
                ],
            }
        )
        kept += 1

    if kept <= 0:
        raise RuntimeError("All angle/scale variants became empty for at least one backend.")

    for backend_key, templates in backend_templates.items():
        detector.set_backend_templates(class_id, templates, backend=backend_key)
    detector.class_meta[class_id] = metas
    return detector, kept, skipped


def copy_detector_class(src: Line2DupLikeDetector, dst: Line2DupLikeDetector, class_id: str) -> None:
    source_info = src.get_class_source(class_id)
    if source_info:
        dst.set_class_source(class_id, source_info)
    editor_levels = src.get_original_editor_levels(class_id)
    if editor_levels:
        dst.set_original_editor_levels(class_id, editor_levels)
    dst.class_meta[class_id] = [dict(item) if isinstance(item, dict) else {} for item in src.class_meta.get(class_id, [])]
    for backend_key in BACKEND_LABEL_TO_KEY.values():
        dst.set_backend_templates(
            class_id,
            src.backend_templates.get(backend_key, {}).get(class_id, []),
            backend=backend_key,
        )


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
class RoiRect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class MaskRect:
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
        self.mask_rects: List[MaskRect] = []
        self.template_levels: List[TemplateLevel] = []
        self.manual_points_dirty = False

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

        control_sidebar = ScrollableSidebar(self)
        control_sidebar.grid(row=0, column=0, sticky="ns")
        control = control_sidebar.content
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
        ttk.Radiobutton(tool_row, text="Edit Mask", value="mask", variable=self.tool_var, command=self._refresh_canvas).pack(side="left", padx=(8, 0))
        row += 1

        ttk.Button(control, text="Extract Points From ROI", command=self.extract_points).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        ttk.Button(control, text="Clear Masks", command=self.clear_masks).grid(row=row, column=0, sticky="ew", pady=(4, 0))
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
        self.canvas.bind("<ButtonPress-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)
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
        self.mask_rects = []
        self.template_levels = []
        self.manual_points_dirty = False
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
        self.mask_rects = []
        self.template_levels = []
        self.manual_points_dirty = False
        self.level_var.set(0)
        self.level_spin.configure(from_=0, to=0)
        self.tool_var.set("roi")
        self.status_var.set("ROI reset. Drag left mouse to select a new ROI.")
        self._refresh_canvas()

    def clear_masks(self) -> None:
        if not self.mask_rects:
            self.status_var.set("No mask rectangles to clear.")
            return
        self.mask_rects = []
        self._drop_points_inside_masks()
        self.status_var.set("Mask rectangles cleared.")
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
        mask = self._build_roi_mask()

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
        self.manual_points_dirty = False
        self._force_levels_to_roi_extent()
        self._sync_levels_from_level0(total_levels=len(levels))
        max_level = max(0, len(self.template_levels) - 1)
        self.level_spin.configure(from_=0, to=max_level)
        self.level_var.set(0)
        self.tool_var.set("point")
        self.status_var.set(
            f"Extracted {len(self.template_levels[0].features)} points at level0 with {len(self.mask_rects)} mask rects. "
            f"L0 is editable; L1+ are auto-scaled from L0. Right-drag box deletes selected points."
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
        roi_img = self.image_bgr[self.roi.y : self.roi.y + self.roi.h, self.roi.x : self.roi.x + self.roi.w].copy()
        use_manual_points = bool(self.manual_points_dirty and len(self.template_levels) > 0)

        if len(self.template_levels) > 0 and len(self.template_levels) != len(levels):
            messagebox.showerror(
                "Levels mismatch",
                "Current edited points were extracted with a different levels setting.\n"
                "Please click 'Extract Points From ROI' again after changing Levels.",
            )
            return

        original_mode = "manual_points" if use_manual_points else "auto"
        pose_ui = {
            "angle_start": float(self.angle_start_var.get()),
            "angle_end": float(self.angle_end_var.get()),
            "angle_step": float(self.angle_step_var.get()),
            "scale_start": float(self.scale_start_var.get()),
            "scale_end": float(self.scale_end_var.get()),
            "scale_step": float(self.scale_step_var.get()),
        }

        try:
            detector, kept, skipped = build_multi_backend_detector(
                class_id=class_id,
                roi_img=roi_img,
                roi_rect=self.roi,
                mask_rects=self.mask_rects,
                pose_infos=pose_infos,
                pose_ui=pose_ui,
                levels=levels,
                num_features=num_features,
                weak_threshold=weak_threshold,
                strong_threshold=strong_threshold,
                original_mode=original_mode,
                original_editor_levels=self.template_levels if use_manual_points else None,
            )
        except Exception as exc:
            messagebox.showerror("Build error", str(exc))
            return

        if kept <= 0:
            messagebox.showerror(
                "No valid templates",
                "All angle/scale variants became empty.\n"
                "Try smaller angle/scale range, lower thresholds, or add more points.",
            )
            return

        save_detector_model(detector, out_path)

        preview_meta = detector.get_template_meta(class_id, 0)
        preview = ShapeInfoProducer.transform(
            roi_img,
            float(preview_meta.get("angle", 0.0)),
            float(preview_meta.get("scale", 1.0)),
        ).copy()
        tl = detector.get_templates(class_id, 0, backend="original")[0]
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
        return self.roi is not None and self.tool_var.get() in {"point", "mask"}

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

    def _build_roi_mask(self) -> np.ndarray:
        if self.roi is None:
            return np.zeros((1, 1), dtype=np.uint8)
        mask = np.full((self.roi.h, self.roi.w), 255, dtype=np.uint8)
        for rect in self.mask_rects:
            x1 = max(0, min(int(rect.x), self.roi.w))
            y1 = max(0, min(int(rect.y), self.roi.h))
            x2 = max(x1, min(int(rect.x + rect.w), self.roi.w))
            y2 = max(y1, min(int(rect.y + rect.h), self.roi.h))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def _display_mask_rects(self) -> List[MaskRect]:
        rects: List[MaskRect] = []
        lv = max(0, int(self.level_var.get()))
        scale = max(1, 1 << lv)
        for rect in self.mask_rects:
            x1 = int(rect.x // scale)
            y1 = int(rect.y // scale)
            x2 = int((rect.x + rect.w + scale - 1) // scale)
            y2 = int((rect.y + rect.h + scale - 1) // scale)
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            rects.append(MaskRect(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
        return rects

    def _point_is_masked(self, x_abs: int, y_abs: int) -> bool:
        for rect in self._display_mask_rects():
            if rect.x <= x_abs < rect.x + rect.w and rect.y <= y_abs < rect.y + rect.h:
                return True
        return False

    def _drop_points_inside_masks(self) -> None:
        if not self.template_levels or not self.mask_rects:
            return
        tl = self.template_levels[0]
        kept: List[Feature] = []
        for f in tl.features:
            keep = True
            for rect in self.mask_rects:
                if rect.x <= int(f.x) < rect.x + rect.w and rect.y <= int(f.y) < rect.y + rect.h:
                    keep = False
                    break
            if keep:
                kept.append(f)
        if len(kept) != len(tl.features):
            tl.features = kept
            self._sync_levels_from_level0(total_levels=len(self.template_levels))

    def _add_mask_rect(self, x0_abs: int, y0_abs: int, x1_abs: int, y1_abs: int) -> bool:
        if self.roi is None:
            return False
        lv = max(0, int(self.level_var.get()))
        scale = max(1, 1 << lv)
        xa0 = int(round(min(x0_abs, x1_abs) * scale))
        ya0 = int(round(min(y0_abs, y1_abs) * scale))
        xb0 = int(round(max(x0_abs, x1_abs) * scale))
        yb0 = int(round(max(y0_abs, y1_abs) * scale))

        xa0 = max(0, min(xa0, self.roi.w))
        ya0 = max(0, min(ya0, self.roi.h))
        xb0 = max(0, min(xb0, self.roi.w))
        yb0 = max(0, min(yb0, self.roi.h))
        w = max(0, xb0 - xa0)
        h = max(0, yb0 - ya0)
        if w < scale or h < scale:
            self.status_var.set("Mask rectangle too small.")
            return False

        self.mask_rects.append(MaskRect(x=xa0, y=ya0, w=w, h=h))
        self._drop_points_inside_masks()
        self.status_var.set(f"Added mask rectangle. masks={len(self.mask_rects)}")
        return True

    def _delete_nearest_mask(self, x_abs: int, y_abs: int) -> bool:
        if not self.mask_rects:
            return False
        lv = max(0, int(self.level_var.get()))
        scale = max(1, 1 << lv)
        x0 = int(round(x_abs * scale))
        y0 = int(round(y_abs * scale))
        tol = 10.0 * scale
        best_idx = -1
        best_d2 = 1e18
        for i, rect in enumerate(self.mask_rects):
            rx1 = int(rect.x)
            ry1 = int(rect.y)
            rx2 = int(rect.x + rect.w)
            ry2 = int(rect.y + rect.h)
            dx = 0.0
            dy = 0.0
            if x0 < rx1:
                dx = float(rx1 - x0)
            elif x0 > rx2:
                dx = float(x0 - rx2)
            if y0 < ry1:
                dy = float(ry1 - y0)
            elif y0 > ry2:
                dy = float(y0 - ry2)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx < 0 or best_d2 > tol * tol:
            return False
        del self.mask_rects[best_idx]
        self._drop_points_inside_masks()
        self.status_var.set(f"Deleted mask rectangle. masks={len(self.mask_rects)}")
        return True

    def _update_hover(self, x_abs: int, y_abs: int) -> None:
        if self.tool_var.get() == "mask":
            self.hover_index = None
            return
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
        if self._point_is_masked(x_abs, y_abs):
            self.status_var.set("Cannot add point inside masked area.")
            return
        xr = int(round(x_abs - tl.tl_x))
        yr = int(round(y_abs - tl.tl_y))
        xr = max(0, min(xr, int(tl.width)))
        yr = max(0, min(yr, int(tl.height)))
        lb = int(label) % 8
        theta = label_to_angle_deg(lb) if theta_deg is None else float(theta_deg)
        tl.features.append(Feature(x=xr, y=yr, label=lb, theta=theta))
        self.manual_points_dirty = True
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
        self.manual_points_dirty = True
        self._sync_levels_from_level0(total_levels=len(self.template_levels))
        return True

    def _delete_points_in_box(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        tl = self._current_level_template()
        if tl is None or not tl.features:
            return 0
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch View Level to 0.")
            return 0
        x1, y1, x2, y2 = normalize_drag_rect(start, end)
        pts = self._get_abs_points()
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
        tl.features = keep
        self.hover_index = None
        self.manual_points_dirty = True
        self._sync_levels_from_level0(total_levels=len(self.template_levels))
        return deleted

    def _canvas_to_image(self, event: tk.Event) -> Tuple[int, int]:
        z = max(1e-6, float(self.zoom_var.get()))
        xw = float(self.canvas.canvasx(event.x))
        yw = float(self.canvas.canvasy(event.y))
        return int(round(xw / z)), int(round(yw / z))

    def _on_left_down(self, event: tk.Event) -> None:
        if self.image_bgr is None:
            return
        if self.drag_kind == "erase":
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
        elif self.tool_var.get() == "mask" and self.roi is not None:
            self.drag_kind = "mask"
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_left_move(self, event: tk.Event) -> None:
        if self.drag_kind is None:
            return
        if self.drag_kind == "erase":
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
                self.mask_rects = []
                self.template_levels = []
                self.manual_points_dirty = False
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
        elif self.drag_kind == "mask" and self.roi is not None:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            self._add_mask_rect(sx, sy, ex, ey)

        self.drag_kind = None
        self.drag_start = None
        self.drag_end = None
        self._refresh_canvas()

    def _on_right_down(self, event: tk.Event) -> None:
        if self.roi is None:
            return
        if self.tool_var.get() == "mask":
            self.drag_kind = "mask_delete"
            self.drag_start = self._canvas_to_image(event)
            self.drag_end = self.drag_start
            return
        if self.tool_var.get() != "point":
            return
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch View Level to 0.")
            return
        if self.drag_kind is not None:
            return
        x, y = self._canvas_to_image(event)
        self.drag_kind = "erase"
        self.drag_start = (x, y)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_right_move(self, event: tk.Event) -> None:
        if self.drag_kind == "mask_delete":
            return
        if self.drag_kind != "erase":
            return
        x, y = self._canvas_to_image(event)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_right_up(self, event: tk.Event) -> None:
        if self.drag_kind == "mask_delete":
            x, y = self._canvas_to_image(event)
            self._delete_nearest_mask(x, y)
            self.drag_kind = None
            self.drag_start = None
            self.drag_end = None
            self._refresh_canvas()
            return
        if self.drag_kind != "erase" or self.drag_start is None:
            return
        x, y = self._canvas_to_image(event)
        self.drag_end = (x, y)
        if drag_is_click(self.drag_start, self.drag_end):
            if self._delete_nearest(x, y):
                self.status_var.set("Point deleted.")
        else:
            deleted = self._delete_points_in_box(self.drag_start, self.drag_end)
            if deleted > 0:
                self.status_var.set(f"Deleted {deleted} points in selection.")
            else:
                self.status_var.set("No points in selection.")
        self.drag_kind = None
        self.drag_start = None
        self.drag_end = None
        self._refresh_canvas()

    def _on_motion(self, event: tk.Event) -> None:
        if self.tool_var.get() not in {"point", "mask"}:
            return
        x, y = self._canvas_to_image(event)
        if self.tool_var.get() == "point":
            self._update_hover(x, y)
        else:
            self.hover_index = None
        if self.drag_kind in {"point", "mask"}:
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
            mask_rects = self._display_mask_rects()
            if mask_rects:
                overlay = canvas.copy()
                for rect in mask_rects:
                    x1 = int(rect.x)
                    y1 = int(rect.y)
                    x2 = int(rect.x + rect.w)
                    y2 = int(rect.y + rect.h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0)
                for rect in mask_rects:
                    x1 = int(rect.x)
                    y1 = int(rect.y)
                    x2 = int(rect.x + rect.w)
                    y2 = int(rect.y + rect.h)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 220), 1, cv2.LINE_AA)
                    cv2.line(canvas, (x1, y1), (x2, y2), (20, 20, 220), 1, cv2.LINE_AA)
                    cv2.line(canvas, (x2, y1), (x1, y2), (20, 20, 220), 1, cv2.LINE_AA)
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
            elif self.drag_kind == "mask" and self.drag_start is not None and self.drag_end is not None:
                x1, y1, x2, y2 = normalize_drag_rect(self.drag_start, self.drag_end)
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            elif self.drag_kind == "erase" and self.drag_start is not None and self.drag_end is not None:
                x1, y1, x2, y2 = normalize_drag_rect(self.drag_start, self.drag_end)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 1, cv2.LINE_AA)
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
            for rect in self.mask_rects:
                x1 = int(self.roi.x + rect.x)
                y1 = int(self.roi.y + rect.y)
                x2 = int(self.roi.x + rect.x + rect.w)
                y2 = int(self.roi.y + rect.y + rect.h)
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0.0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 220), 1, cv2.LINE_AA)
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
        self.erase_start: Optional[Tuple[int, int]] = None
        self.erase_end: Optional[Tuple[int, int]] = None
        self.is_erasing = False
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

        control_sidebar = ScrollableSidebar(self)
        control_sidebar.grid(row=0, column=0, sticky="ns")
        control = control_sidebar.content
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
        self.canvas.bind("<ButtonPress-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)
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
        self.status_var.set("Model loaded. Left-drag adds point direction; right-drag box deletes selected points.")
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
        self.detector.invalidate_native_cache(item.class_id)
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
        class_id = self.class_var.get().strip()
        if class_id:
            self.detector.invalidate_native_cache(class_id)

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
        class_id = self.class_var.get().strip()
        if class_id:
            self.detector.invalidate_native_cache(class_id)
        self.hover_index = None
        return True

    def _delete_points_in_box(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        tl = self._current_template_level()
        if tl is None or not tl.features:
            return 0
        x1, y1, x2, y2 = normalize_drag_rect(start, end)
        pts = self._get_abs_points()
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
        self._push_undo()
        tl.features = keep
        class_id = self.class_var.get().strip()
        if class_id:
            self.detector.invalidate_native_cache(class_id)
        self.hover_index = None
        return deleted

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
        if self.is_erasing:
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

    def _on_right_down(self, event: tk.Event) -> None:
        if self.detector is None:
            return
        x, y = self._canvas_to_image(event)
        if self.is_dragging:
            return
        self.is_erasing = True
        self.erase_start = (x, y)
        self.erase_end = (x, y)
        self._refresh_canvas()

    def _on_right_move(self, event: tk.Event) -> None:
        if not self.is_erasing:
            return
        x, y = self._canvas_to_image(event)
        self.erase_end = (x, y)
        self._refresh_canvas()

    def _on_right_up(self, event: tk.Event) -> None:
        if self.detector is None or not self.is_erasing or self.erase_start is None:
            return
        x, y = self._canvas_to_image(event)
        self.erase_end = (x, y)
        if drag_is_click(self.erase_start, self.erase_end):
            if self._delete_nearest(x, y):
                self.status_var.set("Point deleted.")
        else:
            deleted = self._delete_points_in_box(self.erase_start, self.erase_end)
            if deleted > 0:
                self.status_var.set(f"Deleted {deleted} points in selection.")
            else:
                self.status_var.set("No points in selection.")
        self.is_erasing = False
        self.erase_start = None
        self.erase_end = None
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
        if self.is_erasing and self.erase_start is not None and self.erase_end is not None:
            x1, y1, x2, y2 = normalize_drag_rect(self.erase_start, self.erase_end)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 1, cv2.LINE_AA)
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

        control_sidebar = ScrollableSidebar(self)
        control_sidebar.grid(row=0, column=0, sticky="ns")
        control = control_sidebar.content
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
        self.icp_candidates_var = tk.IntVar(value=20)
        self.auto_sweep_var = tk.BooleanVar(value=True)
        self.backend_var = tk.StringVar(value="Original")
        self.zoom_var = tk.DoubleVar(value=1.5)
        self.out_image_var = tk.StringVar(value="")
        self.backend_thresholds = {key: 90.0 for _label, key in BACKEND_ITEMS}
        self._last_backend_label = "Original"

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

        ttk.Label(control, text="Backend").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self.backend_combo = ttk.Combobox(
            control,
            textvariable=self.backend_var,
            values=[label for label, _key in BACKEND_ITEMS],
            state="readonly",
            width=24,
        )
        self.backend_combo.grid(row=row, column=0, sticky="ew")
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_backend_changed())
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

        ttk.Label(control, text="ICP Candidates").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Spinbox(control, from_=1, to=500, increment=1, textvariable=self.icp_candidates_var, width=10).grid(row=row, column=0, sticky="w")
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

    def _on_backend_changed(self) -> None:
        previous_key = None
        for label, key in BACKEND_ITEMS:
            if label == getattr(self, "_last_backend_label", self.backend_var.get().strip()):
                previous_key = key
                break
        if previous_key is not None:
            self.backend_thresholds[previous_key] = float(self.threshold_var.get())
        current_label = self.backend_var.get().strip()
        current_key = BACKEND_LABEL_TO_KEY.get(current_label, "original")
        self.threshold_var.set(float(self.backend_thresholds.get(current_key, 90.0)))
        self._last_backend_label = current_label

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

        total_t0 = time.perf_counter()
        threshold = float(np.clip(self.threshold_var.get(), 0.0, 100.0))
        crop_stride = int(max(0, self.crop_stride_var.get()))
        nms_iou = float(np.clip(self.nms_iou_var.get(), 0.0, 1.0))
        topk = int(max(1, self.topk_var.get()))
        icp_candidates = int(max(1, self.icp_candidates_var.get()))
        backend_key = BACKEND_LABEL_TO_KEY.get(self.backend_var.get().strip(), "original")
        backend_label = BACKEND_KEY_TO_LABEL.get(backend_key, "Original")
        self.backend_thresholds[backend_key] = threshold
        match_ms = 0.0
        nms_ms = 0.0
        refine_ms = 0.0
        attempts = 0

        scene_for_match = self.scene_bgr
        scene_mask_for_match = self.scene_mask
        if crop_stride > 0:
            h, w = scene_for_match.shape[:2]
            h2 = (h // crop_stride) * crop_stride
            w2 = (w // crop_stride) * crop_stride
            if h2 > 0 and w2 > 0:
                scene_for_match = scene_for_match[:h2, :w2].copy()
                if scene_mask_for_match is not None:
                    scene_mask_for_match = scene_mask_for_match[:h2, :w2].copy()

        class_choice = self.class_choice_var.get().strip()
        if class_choice == "" or class_choice == "ALL":
            class_ids = self.detector.class_ids()
        else:
            class_ids = [class_choice]

        if backend_key in {"fusion", "fusionv2"} and scene_mask_for_match is not None:
            messagebox.showerror(
                "Fusion backend",
                "Fusion backends do not support scene mask. Clear the scene mask or switch backend.",
            )
            return

        def match_once(th: float):
            nonlocal match_ms, attempts
            attempts += 1
            t0 = time.perf_counter()
            ms = self.detector.match(
                scene_for_match,
                threshold=th,
                class_ids=class_ids,
                mask=None if backend_key in {"fusion", "fusionv2"} else scene_mask_for_match,
                backend=backend_key,
            )
            match_ms += (time.perf_counter() - t0) * 1000.0
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

        if backend_key == "sim3" and matches:
            refine_pool = matches[: min(len(matches), icp_candidates)]
            refine_ms = refine_matches_sim3(self.detector, scene_for_match, refine_pool) * 1000.0
            refine_pool.sort(
                key=lambda m: (
                    -float(m.refined_fitness if m.refined_fitness is not None else -1.0),
                    float(m.refined_rmse if m.refined_rmse is not None else 1e9),
                    -float(m.similarity),
                )
            )
            t1 = time.perf_counter()
            matches = nms_matches(self.detector, refine_pool, iou_threshold=nms_iou)
            matches.sort(
                key=lambda m: (
                    -float(m.refined_fitness if m.refined_fitness is not None else -1.0),
                    float(m.refined_rmse if m.refined_rmse is not None else 1e9),
                    -float(m.similarity),
                )
            )
            nms_ms += (time.perf_counter() - t1) * 1000.0
        else:
            t1 = time.perf_counter()
            matches = nms_matches(self.detector, matches, iou_threshold=nms_iou)
            matches.sort(key=lambda m: m.similarity, reverse=True)
            nms_ms += (time.perf_counter() - t1) * 1000.0

        shown = min(topk, len(matches))

        draw_t0 = time.perf_counter()
        overlay = draw_matches(self.detector, scene_for_match, matches, topk=topk)
        draw_ms = (time.perf_counter() - draw_t0) * 1000.0
        total_ms = (time.perf_counter() - total_t0) * 1000.0
        self.result_bgr = overlay
        self._refresh_canvas()

        self.result_list.delete(0, tk.END)
        self.result_list.insert(
            tk.END,
            (
                f"[time] backend={backend_label} total={total_ms:.0f}ms "
                f"match={match_ms:.0f}ms nms={nms_ms:.0f}ms "
                f"refine={refine_ms:.0f}ms draw={draw_ms:.0f}ms attempts={attempts}"
            ),
        )
        for i in range(shown):
            m = matches[i]
            meta = self.detector.get_template_meta(m.class_id, m.template_id)
            if backend_key == "sim3" and m.refined_scale is not None:
                fit = float(m.refined_fitness) if m.refined_fitness is not None else 0.0
                rmse = float(m.refined_rmse) if m.refined_rmse is not None else 0.0
                line = (
                    f"#{i+1} s={m.similarity:.1f} cls={m.class_id} "
                    f"x={m.x} y={m.y} a={float(m.refined_angle_deg or 0.0):.1f} "
                    f"sc={float(m.refined_scale):.3f} fit={fit:.3f} rmse={rmse:.3f}"
                )
            else:
                line = (
                    f"#{i+1} s={m.similarity:.1f} cls={m.class_id} "
                    f"x={m.x} y={m.y} a={float(meta.get('angle', 0.0)):.0f} "
                    f"sc={float(meta.get('scale', 1.0)):.2f}"
                )
            self.result_list.insert(tk.END, line)
        if shown == 0:
            self.result_list.insert(tk.END, "No match.")

        self.status_var.set(
            f"Match done. backend={backend_label}, total={len(matches)}, shown={shown}, "
            f"threshold_used={used_threshold:.1f}, stride={crop_stride}, time={total_ms:.0f}ms "
            f"(match={match_ms:.0f}, nms={nms_ms:.0f}, refine={refine_ms:.0f}, draw={draw_ms:.0f}, icp_candidates={icp_candidates})."
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


class EditTemplateTabV2(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        self.detector: Optional[Line2DupLikeDetector] = None
        self.model_path = ""
        self.class_states: dict[str, dict] = {}
        self.current_class_id = ""
        self.undo_stack: List[UndoItem] = []

        self.image_bgr: Optional[np.ndarray] = None
        self.roi: Optional[RoiRect] = None
        self.source_roi_rect: Optional[RoiRect] = None
        self.mask_rects: List[MaskRect] = []
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

        control_sidebar = ScrollableSidebar(self)
        control_sidebar.grid(row=0, column=0, sticky="ns")
        control = control_sidebar.content
        view = ttk.Frame(self, padding=(0, 8, 8, 8))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Load a v2 model to edit embedded ROI/mask/pose data.")
        self.class_var = tk.StringVar(value="")
        self.tool_var = tk.StringVar(value="point")
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
        ttk.Label(control, text="Edit Existing Model (rebuild Fusion / Fusion V2 / sim3 on save)").grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(control, text="Open Model...", command=self.open_model).grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1
        self.model_label = ttk.Label(control, text="(no model)", width=36, wraplength=260)
        self.model_label.grid(row=row, column=0, sticky="w")
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
        ttk.Radiobutton(tool_row, text="Edit Points", value="point", variable=self.tool_var, command=self._refresh_canvas).pack(side="left")
        ttk.Radiobutton(tool_row, text="Edit Mask", value="mask", variable=self.tool_var, command=self._refresh_canvas).pack(side="left", padx=(8, 0))
        row += 1

        ttk.Label(control, text="Original points only affect Original backend.", wraplength=260, foreground="#7a5a00").grid(
            row=row, column=0, sticky="w", pady=(6, 0)
        )
        row += 1

        ttk.Button(control, text="Undo", command=self.undo).grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        ttk.Button(control, text="Clear Masks", command=self.clear_masks).grid(row=row, column=0, sticky="ew", pady=(4, 0))
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
        self.canvas.bind("<ButtonPress-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)
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

        class_states: dict[str, dict] = {}
        for class_id in detector.class_ids():
            try:
                source_info, roi_img, _roi_mask, roi_rect, mask_rects = load_class_source_assets(detector, class_id)
            except Exception:
                continue
            pose_ui = source_info.get("pose_infos", {}).get("ui", {}) if isinstance(source_info, dict) else {}
            editor_levels = detector.get_original_editor_levels(class_id)
            if not editor_levels and detector.backend_templates.get("original", {}).get(class_id):
                editor_levels = normalize_extracted_levels_to_roi(
                    detector.get_templates(class_id, 0, backend="original"),
                    roi_img,
                )
            class_states[class_id] = {
                "image_bgr": roi_img,
                "source_roi_rect": roi_rect,
                "mask_rects": [MaskRect(x=int(rect.x), y=int(rect.y), w=int(rect.w), h=int(rect.h)) for rect in mask_rects],
                "template_levels": clone_levels(editor_levels),
                "pose_ui": {
                    "angle_start": float(pose_ui.get("angle_start", 0.0)),
                    "angle_end": float(pose_ui.get("angle_end", 0.0)),
                    "angle_step": float(pose_ui.get("angle_step", 10.0)),
                    "scale_start": float(pose_ui.get("scale_start", 1.0)),
                    "scale_end": float(pose_ui.get("scale_end", 1.0)),
                    "scale_step": float(pose_ui.get("scale_step", 0.05)),
                },
                "original_mode": str(source_info.get("original_mode", "auto")),
                "points_dirty": False,
                "mask_dirty": False,
            }
        if not class_states:
            messagebox.showerror("Invalid model", "No editable v2 classes found in model.")
            return

        self.detector = detector
        self.class_states = class_states
        self.model_path = path
        self.model_label.configure(text=path)
        self.out_model_var.set(path)
        self.out_label.configure(text=path)
        class_ids = list(class_states.keys())
        self.class_combo.configure(values=class_ids)
        self.class_var.set(class_ids[0])
        self.undo_stack = []
        self._on_class_changed()
        self.status_var.set("Model loaded. Edit embedded ROI points or masks, then save to rebuild all backends.")
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

    def _capture_current_state(self) -> None:
        if not self.current_class_id or self.current_class_id not in self.class_states:
            return
        state = self.class_states[self.current_class_id]
        state["image_bgr"] = self.image_bgr.copy() if self.image_bgr is not None else None
        if self.source_roi_rect is not None:
            state["source_roi_rect"] = RoiRect(
                x=int(self.source_roi_rect.x),
                y=int(self.source_roi_rect.y),
                w=int(self.source_roi_rect.w),
                h=int(self.source_roi_rect.h),
            )
        state["mask_rects"] = [MaskRect(x=int(rect.x), y=int(rect.y), w=int(rect.w), h=int(rect.h)) for rect in self.mask_rects]
        state["template_levels"] = clone_levels(self.template_levels)
        state["pose_ui"] = {
            "angle_start": float(self.angle_start_var.get()),
            "angle_end": float(self.angle_end_var.get()),
            "angle_step": float(self.angle_step_var.get()),
            "scale_start": float(self.scale_start_var.get()),
            "scale_end": float(self.scale_end_var.get()),
            "scale_step": float(self.scale_step_var.get()),
        }

    def _on_class_changed(self) -> None:
        if self.detector is None or not self.class_states:
            return
        self._capture_current_state()
        class_id = self.class_var.get().strip()
        if not class_id or class_id not in self.class_states:
            return
        state = self.class_states[class_id]
        self.current_class_id = class_id
        self.image_bgr = state["image_bgr"].copy() if state.get("image_bgr") is not None else None
        self.source_roi_rect = RoiRect(
            x=int(state["source_roi_rect"].x),
            y=int(state["source_roi_rect"].y),
            w=int(state["source_roi_rect"].w),
            h=int(state["source_roi_rect"].h),
        )
        self.roi = RoiRect(0, 0, int(self.image_bgr.shape[1]), int(self.image_bgr.shape[0])) if self.image_bgr is not None else None
        self.mask_rects = [MaskRect(x=int(rect.x), y=int(rect.y), w=int(rect.w), h=int(rect.h)) for rect in state["mask_rects"]]
        self.template_levels = clone_levels(state["template_levels"])
        pose_ui = state["pose_ui"]
        self.angle_start_var.set(float(pose_ui.get("angle_start", 0.0)))
        self.angle_end_var.set(float(pose_ui.get("angle_end", 0.0)))
        self.angle_step_var.set(float(pose_ui.get("angle_step", 10.0)))
        self.scale_start_var.set(float(pose_ui.get("scale_start", 1.0)))
        self.scale_end_var.set(float(pose_ui.get("scale_end", 1.0)))
        self.scale_step_var.set(float(pose_ui.get("scale_step", 0.05)))
        self.level_spin.configure(from_=0, to=max(0, len(self.template_levels) - 1))
        self.level_var.set(0)
        self.tool_var.set("point")
        self.hover_index = None
        self._refresh_canvas()

    def _current_template_level(self) -> Optional[TemplateLevel]:
        if not self.template_levels:
            return None
        level_index = max(0, min(int(self.level_var.get()), len(self.template_levels) - 1))
        self.level_var.set(level_index)
        return self.template_levels[level_index]

    def _get_abs_points(self) -> List[Tuple[int, int]]:
        tl = self._current_template_level()
        if tl is None:
            return []
        return [(int(feature.x + tl.tl_x), int(feature.y + tl.tl_y)) for feature in tl.features]

    def _level_roi_image(self) -> np.ndarray:
        if self.image_bgr is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        img = self.image_bgr.copy()
        for _ in range(max(0, int(self.level_var.get()))):
            if img.shape[0] < 2 or img.shape[1] < 2:
                break
            img = cv2.pyrDown(img)
        return img

    def _build_roi_mask(self) -> np.ndarray:
        if self.roi is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return build_mask_from_rects(self.roi.w, self.roi.h, self.mask_rects)

    def _display_mask_rects(self) -> List[MaskRect]:
        rects: List[MaskRect] = []
        level = max(0, int(self.level_var.get()))
        scale = max(1, 1 << level)
        for rect in self.mask_rects:
            x1 = int(rect.x // scale)
            y1 = int(rect.y // scale)
            x2 = int((rect.x + rect.w + scale - 1) // scale)
            y2 = int((rect.y + rect.h + scale - 1) // scale)
            rects.append(MaskRect(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1)))
        return rects

    def _point_is_masked(self, x_abs: int, y_abs: int) -> bool:
        for rect in self._display_mask_rects():
            if rect.x <= x_abs < rect.x + rect.w and rect.y <= y_abs < rect.y + rect.h:
                return True
        return False

    def _sync_levels_from_level0(self) -> None:
        if not self.template_levels or self.image_bgr is None:
            return
        shapes = roi_level_shapes_from_image(self.image_bgr, len(self.template_levels))
        self.template_levels = sync_levels_from_level0(self.template_levels[0], shapes)

    def _drop_points_inside_masks(self) -> None:
        if not self.template_levels or not self.mask_rects:
            return
        level0 = self.template_levels[0]
        kept: List[Feature] = []
        for feature in level0.features:
            keep = True
            for rect in self.mask_rects:
                if rect.x <= int(feature.x) < rect.x + rect.w and rect.y <= int(feature.y) < rect.y + rect.h:
                    keep = False
                    break
            if keep:
                kept.append(feature)
        if len(kept) != len(level0.features):
            level0.features = kept
            self._sync_levels_from_level0()

    def _add_mask_rect(self, x0_abs: int, y0_abs: int, x1_abs: int, y1_abs: int) -> bool:
        if self.roi is None:
            return False
        level = max(0, int(self.level_var.get()))
        scale = max(1, 1 << level)
        xa0 = int(round(min(x0_abs, x1_abs) * scale))
        ya0 = int(round(min(y0_abs, y1_abs) * scale))
        xb0 = int(round(max(x0_abs, x1_abs) * scale))
        yb0 = int(round(max(y0_abs, y1_abs) * scale))
        xa0 = max(0, min(xa0, self.roi.w))
        ya0 = max(0, min(ya0, self.roi.h))
        xb0 = max(0, min(xb0, self.roi.w))
        yb0 = max(0, min(yb0, self.roi.h))
        w = max(0, xb0 - xa0)
        h = max(0, yb0 - ya0)
        if w < scale or h < scale:
            self.status_var.set("Mask rectangle too small.")
            return False
        self.mask_rects.append(MaskRect(x=xa0, y=ya0, w=w, h=h))
        self._drop_points_inside_masks()
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["mask_dirty"] = True
        self.status_var.set(f"Added mask rectangle. masks={len(self.mask_rects)}")
        return True

    def _delete_nearest_mask(self, x_abs: int, y_abs: int) -> bool:
        if not self.mask_rects:
            return False
        level = max(0, int(self.level_var.get()))
        scale = max(1, 1 << level)
        x0 = int(round(x_abs * scale))
        y0 = int(round(y_abs * scale))
        tol = 10.0 * scale
        best_idx = -1
        best_d2 = 1e18
        for i, rect in enumerate(self.mask_rects):
            rx1 = int(rect.x)
            ry1 = int(rect.y)
            rx2 = int(rect.x + rect.w)
            ry2 = int(rect.y + rect.h)
            dx = 0.0
            dy = 0.0
            if x0 < rx1:
                dx = float(rx1 - x0)
            elif x0 > rx2:
                dx = float(x0 - rx2)
            if y0 < ry1:
                dy = float(ry1 - y0)
            elif y0 > ry2:
                dy = float(y0 - ry2)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx < 0 or best_d2 > tol * tol:
            return False
        del self.mask_rects[best_idx]
        self._drop_points_inside_masks()
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["mask_dirty"] = True
        self.status_var.set(f"Deleted mask rectangle. masks={len(self.mask_rects)}")
        return True

    def _push_undo(self) -> None:
        tl = self._current_template_level()
        if tl is None:
            return
        snapshot = [Feature(x=int(f.x), y=int(f.y), label=int(f.label), theta=float(f.theta)) for f in tl.features]
        self.undo_stack.append(UndoItem(class_id=self.class_var.get().strip(), template_id=0, level=int(self.level_var.get()), features=snapshot))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if self.detector is None or not self.undo_stack:
            return
        item = self.undo_stack.pop()
        if item.class_id != self.class_var.get().strip():
            return
        if item.level < 0 or item.level >= len(self.template_levels):
            return
        self.template_levels[item.level].features = item.features
        self._sync_levels_from_level0()
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["points_dirty"] = True
        self.level_var.set(item.level)
        self.status_var.set("Undo complete.")
        self._refresh_canvas()

    def clear_masks(self) -> None:
        if not self.mask_rects:
            self.status_var.set("No mask rectangles to clear.")
            return
        self.mask_rects = []
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["mask_dirty"] = True
        self.status_var.set("Mask rectangles cleared.")
        self._refresh_canvas()

    def _update_hover(self, x_abs: int, y_abs: int) -> None:
        if self.tool_var.get() == "mask":
            self.hover_index = None
            return
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
        self.hover_index = best_idx if best_idx >= 0 and best_d2 <= 8.0 * 8.0 else None

    def _add_feature(self, x_abs: int, y_abs: int, label: int, theta_deg: Optional[float]) -> None:
        tl = self._current_template_level()
        if tl is None:
            return
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch Level to 0.")
            return
        if self._point_is_masked(x_abs, y_abs):
            self.status_var.set("Cannot add point inside masked area.")
            return
        self._push_undo()
        xr = int(round(x_abs - tl.tl_x))
        yr = int(round(y_abs - tl.tl_y))
        xr = max(0, min(xr, int(tl.width)))
        yr = max(0, min(yr, int(tl.height)))
        lb = int(label) % 8
        theta = label_to_angle_deg(lb) if theta_deg is None else float(theta_deg)
        tl.features.append(Feature(x=xr, y=yr, label=lb, theta=theta))
        self._sync_levels_from_level0()
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["points_dirty"] = True

    def _delete_nearest(self, x_abs: int, y_abs: int) -> bool:
        tl = self._current_template_level()
        if tl is None or not tl.features:
            return False
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch Level to 0.")
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
        self._sync_levels_from_level0()
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["points_dirty"] = True
        self.hover_index = None
        return True

    def _delete_points_in_box(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        tl = self._current_template_level()
        if tl is None or not tl.features:
            return 0
        if int(self.level_var.get()) != 0:
            self.status_var.set("Only L0 is editable. Switch Level to 0.")
            return 0
        x1, y1, x2, y2 = normalize_drag_rect(start, end)
        pts = self._get_abs_points()
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
        self._push_undo()
        tl.features = keep
        self._sync_levels_from_level0()
        if self.current_class_id in self.class_states:
            self.class_states[self.current_class_id]["points_dirty"] = True
        self.hover_index = None
        return deleted

    def save_model(self) -> None:
        if self.detector is None:
            messagebox.showwarning("No model", "Please open model first.")
            return
        self._capture_current_state()
        out_path = self.out_model_var.get().strip() or self.model_path
        if not out_path:
            self.select_output_model()
            out_path = self.out_model_var.get().strip() or self.model_path
            if not out_path:
                return

        final_detector = Line2DupLikeDetector(
            num_features=self.detector.num_features,
            T_levels=self.detector.T_at_level,
            weak_threshold=self.detector.weak_threshold,
            strong_threshold=self.detector.strong_threshold,
        )
        build_lines: List[str] = []

        for class_id in self.detector.class_ids():
            if class_id not in self.class_states:
                copy_detector_class(self.detector, final_detector, class_id)
                continue
            state = self.class_states[class_id]
            pose_ui = state["pose_ui"]
            try:
                pose_infos = pose_infos_from_ui_values(
                    float(pose_ui.get("angle_start", 0.0)),
                    float(pose_ui.get("angle_end", 0.0)),
                    float(pose_ui.get("angle_step", 10.0)),
                    float(pose_ui.get("scale_start", 1.0)),
                    float(pose_ui.get("scale_end", 1.0)),
                    float(pose_ui.get("scale_step", 0.05)),
                )
            except Exception as exc:
                messagebox.showerror("Invalid angle/scale range", f"{class_id}: {exc}")
                return

            original_mode = "manual_points" if state.get("points_dirty") or str(state.get("original_mode", "auto")) == "manual_points" else "auto"
            try:
                rebuilt, kept, skipped = build_multi_backend_detector(
                    class_id=class_id,
                    roi_img=state["image_bgr"],
                    roi_rect=state["source_roi_rect"],
                    mask_rects=state["mask_rects"],
                    pose_infos=pose_infos,
                    pose_ui=pose_ui,
                    levels=self.detector.T_at_level,
                    num_features=self.detector.num_features,
                    weak_threshold=self.detector.weak_threshold,
                    strong_threshold=self.detector.strong_threshold,
                    original_mode=original_mode,
                    original_editor_levels=state["template_levels"] if original_mode == "manual_points" else None,
                )
            except Exception as exc:
                messagebox.showerror("Build error", f"{class_id}: {exc}")
                return
            copy_detector_class(rebuilt, final_detector, class_id)
            build_lines.append(f"{class_id}: templates={kept} skipped={skipped} mode={original_mode}")

        save_detector_model(final_detector, out_path)
        self.detector = final_detector
        self.model_path = out_path
        self.model_label.configure(text=out_path)
        self.out_model_var.set(out_path)
        self.out_label.configure(text=out_path)

        reloaded_states: dict[str, dict] = {}
        for class_id in final_detector.class_ids():
            try:
                source_info, roi_img, _roi_mask, roi_rect, mask_rects = load_class_source_assets(final_detector, class_id)
            except Exception:
                continue
            pose_ui = source_info.get("pose_infos", {}).get("ui", {}) if isinstance(source_info, dict) else {}
            reloaded_states[class_id] = {
                "image_bgr": roi_img,
                "source_roi_rect": roi_rect,
                "mask_rects": [MaskRect(x=int(rect.x), y=int(rect.y), w=int(rect.w), h=int(rect.h)) for rect in mask_rects],
                "template_levels": final_detector.get_original_editor_levels(class_id),
                "pose_ui": {
                    "angle_start": float(pose_ui.get("angle_start", 0.0)),
                    "angle_end": float(pose_ui.get("angle_end", 0.0)),
                    "angle_step": float(pose_ui.get("angle_step", 10.0)),
                    "scale_start": float(pose_ui.get("scale_start", 1.0)),
                    "scale_end": float(pose_ui.get("scale_end", 1.0)),
                    "scale_step": float(pose_ui.get("scale_step", 0.05)),
                },
                "original_mode": str(source_info.get("original_mode", "auto")),
                "points_dirty": False,
                "mask_dirty": False,
            }
        self.class_states = reloaded_states
        self.class_combo.configure(values=list(self.class_states.keys()))
        if self.class_var.get().strip() not in self.class_states:
            self.class_var.set(next(iter(self.class_states.keys()), ""))
        self._on_class_changed()
        self.status_var.set(f"Saved model: {out_path}")
        messagebox.showinfo("Saved", f"Model saved:\n{out_path}\n\n" + "\n".join(build_lines))

    def _canvas_to_image(self, event: tk.Event) -> Tuple[int, int]:
        zoom = max(1e-6, float(self.zoom_var.get()))
        return int(round(float(self.canvas.canvasx(event.x)) / zoom)), int(round(float(self.canvas.canvasy(event.y)) / zoom))

    def _on_left_down(self, event: tk.Event) -> None:
        if self.image_bgr is None:
            return
        x, y = self._canvas_to_image(event)
        if self.tool_var.get() == "point":
            if int(self.level_var.get()) != 0:
                self.status_var.set("Only L0 is editable. Switch Level to 0.")
                return
            self.drag_kind = "point"
        else:
            self.drag_kind = "mask"
        self.drag_start = (x, y)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_left_move(self, event: tk.Event) -> None:
        if self.drag_kind not in {"point", "mask"}:
            return
        self.drag_end = self._canvas_to_image(event)
        self._refresh_canvas()

    def _on_left_up(self, event: tk.Event) -> None:
        if self.drag_kind not in {"point", "mask"} or self.drag_start is None:
            return
        self.drag_end = self._canvas_to_image(event)
        if self.drag_kind == "point":
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            dx = ex - sx
            dy = ey - sy
            dist = float(np.hypot(dx, dy))
            if dist >= 2.0:
                theta = float(np.degrees(np.arctan2(float(dy), float(dx))))
                label = angle_deg_to_label(theta)
                self.label_var.set(label)
                self._add_feature(sx, sy, label, theta_deg=theta)
            else:
                label = int(self.label_var.get()) % 8
                self._add_feature(sx, sy, label, theta_deg=label_to_angle_deg(label))
        else:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            self._add_mask_rect(sx, sy, ex, ey)
        self.drag_kind = None
        self.drag_start = None
        self.drag_end = None
        self._refresh_canvas()

    def _on_right_down(self, event: tk.Event) -> None:
        if self.image_bgr is None:
            return
        x, y = self._canvas_to_image(event)
        if self.tool_var.get() == "mask":
            self.drag_kind = "mask_delete"
        else:
            if int(self.level_var.get()) != 0:
                self.status_var.set("Only L0 is editable. Switch Level to 0.")
                return
            self.drag_kind = "erase"
        self.drag_start = (x, y)
        self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_right_move(self, event: tk.Event) -> None:
        if self.drag_kind != "erase":
            return
        self.drag_end = self._canvas_to_image(event)
        self._refresh_canvas()

    def _on_right_up(self, event: tk.Event) -> None:
        if self.drag_kind == "mask_delete":
            x, y = self._canvas_to_image(event)
            self._delete_nearest_mask(x, y)
            self.drag_kind = None
            self.drag_start = None
            self.drag_end = None
            self._refresh_canvas()
            return
        if self.drag_kind != "erase" or self.drag_start is None:
            return
        self.drag_end = self._canvas_to_image(event)
        if drag_is_click(self.drag_start, self.drag_end):
            if self._delete_nearest(*self.drag_end):
                self.status_var.set("Point deleted.")
        else:
            deleted = self._delete_points_in_box(self.drag_start, self.drag_end)
            self.status_var.set(f"Deleted {deleted} points in selection." if deleted > 0 else "No points in selection.")
        self.drag_kind = None
        self.drag_start = None
        self.drag_end = None
        self._refresh_canvas()

    def _on_motion(self, event: tk.Event) -> None:
        if self.image_bgr is None:
            return
        x, y = self._canvas_to_image(event)
        self._update_hover(x, y)
        if self.drag_kind in {"point", "mask"}:
            self.drag_end = (x, y)
        self._refresh_canvas()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        old_zoom = float(self.zoom_var.get())
        delta = wheel_delta(event)
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        self.zoom_var.set(new_zoom)
        self._refresh_canvas()
        keep_canvas_point_stable_after_zoom(
            self.canvas,
            event,
            old_zoom=old_zoom,
            new_zoom=new_zoom,
            content_w=self._last_render_w,
            content_h=self._last_render_h,
        )

    def _render_image(self) -> np.ndarray:
        if self.image_bgr is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "Open v2 model", (78, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
            return blank
        canvas = self._level_roi_image().copy()
        for rect in self._display_mask_rects():
            overlay = canvas.copy()
            x1 = int(rect.x)
            y1 = int(rect.y)
            x2 = int(rect.x + rect.w)
            y2 = int(rect.y + rect.h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 220), 1, cv2.LINE_AA)
        tl = self._current_template_level()
        if tl is None:
            return canvas
        cv2.rectangle(canvas, (int(tl.tl_x), int(tl.tl_y)), (int(tl.tl_x + tl.width), int(tl.tl_y + tl.height)), (0, 255, 255), 1, cv2.LINE_AA)
        palette = orientation_palette_bgr()
        for i, (px, py) in enumerate(self._get_abs_points()):
            feature = tl.features[i]
            color = palette[int(feature.label) % 8]
            theta = float(feature.theta) if np.isfinite(feature.theta) else label_to_angle_deg(int(feature.label))
            p2 = arrow_endpoint(int(px), int(py), theta, 8.0)
            if self.hover_index is not None and i == self.hover_index:
                cv2.circle(canvas, (int(px), int(py)), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (int(px), int(py)), p2, color, 1, cv2.LINE_AA, 0, 0.35)
            cv2.circle(canvas, (int(px), int(py)), 1, color, -1, cv2.LINE_AA)
        if self.drag_kind == "point" and self.drag_start is not None and self.drag_end is not None:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            label = angle_deg_to_label(np.degrees(np.arctan2(float(ey - sy), float(ex - sx)))) if abs(ex - sx) + abs(ey - sy) > 0 else int(self.label_var.get()) % 8
            cv2.arrowedLine(canvas, (sx, sy), (ex, ey), palette[label], 1, cv2.LINE_AA, 0, 0.35)
        elif self.drag_kind == "mask" and self.drag_start is not None and self.drag_end is not None:
            x1, y1, x2, y2 = normalize_drag_rect(self.drag_start, self.drag_end)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        elif self.drag_kind == "erase" and self.drag_start is not None and self.drag_end is not None:
            x1, y1, x2, y2 = normalize_drag_rect(self.drag_start, self.drag_end)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 1, cv2.LINE_AA)
        return canvas

    def _refresh_canvas(self) -> None:
        image = self._render_image()
        zoom = max(ZOOM_MIN, float(self.zoom_var.get()))
        if abs(zoom - 1.0) > 1e-6:
            image = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        self._photo = bgr_to_photo(image)
        self._last_render_w = int(image.shape[1])
        self._last_render_h = int(image.shape[0])
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, self._last_render_w, self._last_render_h))


EditTemplateTab = EditTemplateTabV2


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
                class_states: dict[str, dict] = {}
                for class_id in det.class_ids():
                    try:
                        source_info, roi_img, _roi_mask, roi_rect, mask_rects = load_class_source_assets(det, class_id)
                    except Exception:
                        continue
                    pose_ui = source_info.get("pose_infos", {}).get("ui", {}) if isinstance(source_info, dict) else {}
                    class_states[class_id] = {
                        "image_bgr": roi_img,
                        "source_roi_rect": roi_rect,
                        "mask_rects": [MaskRect(x=int(rect.x), y=int(rect.y), w=int(rect.w), h=int(rect.h)) for rect in mask_rects],
                        "template_levels": det.get_original_editor_levels(class_id),
                        "pose_ui": {
                            "angle_start": float(pose_ui.get("angle_start", 0.0)),
                            "angle_end": float(pose_ui.get("angle_end", 0.0)),
                            "angle_step": float(pose_ui.get("angle_step", 10.0)),
                            "scale_start": float(pose_ui.get("scale_start", 1.0)),
                            "scale_end": float(pose_ui.get("scale_end", 1.0)),
                            "scale_step": float(pose_ui.get("scale_step", 0.05)),
                        },
                        "original_mode": str(source_info.get("original_mode", "auto")),
                        "points_dirty": False,
                        "mask_dirty": False,
                    }
                self.edit_tab.class_states = class_states
                class_ids = list(class_states.keys())
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
