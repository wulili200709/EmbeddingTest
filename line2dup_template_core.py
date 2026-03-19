from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from line2dup_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from line2dup_like_matcher import (  # noqa: E402
    Feature,
    Line2DupLikeDetector,
    TemplateLevel,
    clone_template_levels,
    create_native_detector,
    decode_png_base64,
    encode_png_base64,
    ensure_native_backends_available,
)


BACKEND_ITEMS = [
    ("Original", "original"),
    ("Fusion", "fusion"),
    ("ICP (sim3)", "sim3"),
]
BACKEND_LABEL_TO_KEY = {label: key for label, key in BACKEND_ITEMS}
BACKEND_KEY_TO_LABEL = {key: label for label, key in BACKEND_ITEMS}


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


def clone_levels(levels: Sequence[TemplateLevel]) -> List[TemplateLevel]:
    return clone_template_levels(levels)


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


def label_to_angle_deg(label: int) -> float:
    label = int(label) & 7
    return float(label * (360.0 / 8.0))


def angle_deg_to_label(theta_deg: float) -> int:
    return int(theta_deg * 16.0 / 360.0 + 0.5) & 7


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
        try:
            from line2dup_like_matcher import crop_templates  # noqa: E402

            crop_templates(out)
        except Exception:
            pass
    return out


def expanded_pose_affine(
    width: int,
    height: int,
    angle_deg: float,
    scale: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    center = (float(width) * 0.5, float(height) * 0.5)
    mat = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))

    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width - 1), 0.0, 1.0],
            [float(width - 1), float(height - 1), 1.0],
            [0.0, float(height - 1), 1.0],
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

    affine3 = np.eye(3, dtype=np.float32)
    affine3[:2, :] = mat.astype(np.float32)
    return affine3, (new_w, new_h)


def transform_image_and_mask_expanded(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    affine3, (new_w, new_h) = expanded_pose_affine(image_bgr.shape[1], image_bgr.shape[0], angle_deg, scale)
    mat = affine3[:2, :]
    out_img = cv2.warpAffine(image_bgr, mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    out_mask = cv2.warpAffine(mask, mat, (new_w, new_h), flags=cv2.INTER_NEAREST)
    out_mask = (out_mask > 0).astype(np.uint8) * 255
    return out_img, out_mask


def apply_affine_to_points(transform: np.ndarray, points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    mat = np.asarray(transform, dtype=np.float32)
    for x, y in points:
        px = float(mat[0, 0] * x + mat[0, 1] * y + mat[0, 2])
        py = float(mat[1, 0] * x + mat[1, 1] * y + mat[1, 2])
        pw = 1.0
        if mat.shape[0] >= 3 and mat.shape[1] >= 3:
            pw = float(mat[2, 0] * x + mat[2, 1] * y + mat[2, 2])
        if abs(pw) > 1e-9 and abs(pw - 1.0) > 1e-9:
            px /= pw
            py /= pw
        out.append((px, py))
    return out


def make_class_source_payload(
    roi_img: np.ndarray,
    roi_mask: np.ndarray,
    roi_rect: RoiRect,
    mask_rects: Sequence[MaskRect],
    pose_infos: Sequence[Tuple[float, float]],
    pose_ui: Dict[str, float],
    original_mode: str,
    source_image_path: str = "",
) -> Dict[str, object]:
    return {
        "source": {
            "image_path": str(source_image_path or ""),
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


def load_class_source_assets(
    detector: Line2DupLikeDetector,
    class_id: str,
) -> Tuple[dict, np.ndarray, np.ndarray, RoiRect, List[MaskRect]]:
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
    pose_ui: Dict[str, float],
    levels: Sequence[int],
    num_features: int,
    weak_threshold: float,
    strong_threshold: float,
    original_mode: str,
    original_editor_levels: Optional[Sequence[TemplateLevel]] = None,
    source_image_path: str = "",
) -> Tuple[Line2DupLikeDetector, int, int]:
    ensure_native_backends_available(("original", "fusion", "sim3"))
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
            source_image_path=source_image_path,
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
        sim3_tid = int(sim3_native.add_template(src_i, class_id, mask_i, nfeat))
        if sim3_tid < 0:
            skipped += 1
            continue

        fusion_tp = Line2DupLikeDetector._template_pyramid_from_native(
            fusion_native.export_template_pyramid(class_id, fusion_tid)
        )
        sim3_tp = Line2DupLikeDetector._template_pyramid_from_native(
            sim3_native.export_template_pyramid(class_id, sim3_tid)
        )

        backend_templates["original"].append(original_tp)
        backend_templates["fusion"].append(fusion_tp)
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


__all__ = [
    "BACKEND_ITEMS",
    "BACKEND_KEY_TO_LABEL",
    "BACKEND_LABEL_TO_KEY",
    "MaskRect",
    "RoiRect",
    "apply_affine_to_points",
    "build_mask_from_rects",
    "build_multi_backend_detector",
    "clone_levels",
    "copy_detector_class",
    "expanded_pose_affine",
    "load_class_source_assets",
    "make_class_source_payload",
    "normalize_extracted_levels_to_roi",
    "parse_levels",
    "pose_infos_from_ui_values",
    "transform_image_and_mask_expanded",
    "transform_levels_for_pose",
]
