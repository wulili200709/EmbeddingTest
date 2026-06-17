from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui

from shape.core.bootstrap import ensure_repo_root_on_path
from shape.core.roi_follow import FollowResult
from shape.core.template_core import MaskRect, RoiRect, label_to_angle_deg

ensure_repo_root_on_path()

from shape.like_matcher import ShapeLikeDetector, match_quad  # noqa: E402


def _cv_to_qpixmap(image_bgr: np.ndarray) -> QtGui.QPixmap:
    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _orientation_palette() -> List[QtGui.QColor]:
    return [
        QtGui.QColor(255, 0, 0),
        QtGui.QColor(255, 128, 0),
        QtGui.QColor(255, 255, 0),
        QtGui.QColor(0, 255, 0),
        QtGui.QColor(0, 255, 255),
        QtGui.QColor(0, 128, 255),
        QtGui.QColor(0, 0, 255),
        QtGui.QColor(255, 0, 255),
    ]


def _arrow_endpoint(x: float, y: float, theta_deg: float, length: float) -> Tuple[float, float]:
    rad = math.radians(float(theta_deg))
    return float(x + length * math.cos(rad)), float(y + length * math.sin(rad))


def _apply_affine_point(transform: Optional[np.ndarray], x: float, y: float) -> Tuple[float, float]:
    if transform is None:
        return float(x), float(y)
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape == (3, 3):
        matrix = matrix[:2, :]
    px = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
    py = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
    return px, py


def _button_left() -> int:
    return int(QtCore.Qt.MouseButton.LeftButton.value)


def _button_right() -> int:
    return int(QtCore.Qt.MouseButton.RightButton.value)


def _shape_to_rect(shape: dict) -> Optional[Tuple[int, int, int, int]]:
    pts = shape.get("points", [])
    if not pts:
        return None
    if shape.get("shape_type") == "rectangle" and len(pts) >= 2:
        (x0, y0), (x1, y1) = pts[:2]
        x = int(round(min(float(x0), float(x1))))
        y = int(round(min(float(y0), float(y1))))
        w = int(round(abs(float(x1) - float(x0))))
        h = int(round(abs(float(y1) - float(y0))))
        return x, y, max(1, w), max(1, h)
    arr = np.asarray(pts, dtype=np.float32)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    return int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))


def _compact_path_text(path: str, *, max_chars: int = 72) -> str:
    text = os.path.normpath(str(path or ""))
    if len(text) <= max_chars:
        return text
    drive, tail = os.path.splitdrive(text)
    basename = os.path.basename(text)
    parent = os.path.basename(os.path.dirname(text))
    suffix = os.path.join(parent, basename) if parent else basename
    prefix = f"{drive}{os.sep}" if drive else ""
    return f"{prefix}...{os.sep}{suffix}"


def _clamp_rect_to_roi(rect: Tuple[int, int, int, int], roi: RoiRect) -> Optional[MaskRect]:
    x, y, w, h = rect
    x1 = max(int(roi.x), int(x))
    y1 = max(int(roi.y), int(y))
    x2 = min(int(roi.x + roi.w), int(x + w))
    y2 = min(int(roi.y + roi.h), int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return MaskRect(x=x1 - int(roi.x), y=y1 - int(roi.y), w=x2 - x1, h=y2 - y1)


def _overlay_follow_result(
    image_bgr: np.ndarray,
    result: FollowResult,
    label_name: str,
    *,
    elapsed_ms: Optional[float] = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    text = f"{label_name} sim={result.match.similarity:.2f} class={result.match.class_id} tid={result.match.template_id}"
    if elapsed_ms is not None and elapsed_ms >= 0.0:
        text += f" time={elapsed_ms:.1f}ms"
    cv2.putText(canvas, text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 230, 255), 4, cv2.LINE_AA)
    region_palette = [(0, 255, 255), (255, 0, 255), (0, 220, 120), (255, 180, 0), (0, 180, 255)]
    for idx, region in enumerate(result.regions):
        color = region_palette[idx % len(region_palette)]
        pts = np.round(np.asarray(region.points, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        tx = int(round(region.points[0][0]))
        ty = int(round(region.points[0][1]))
        cv2.putText(canvas, region.label_name, (tx, max(20, ty - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return canvas


def _draw_match_overlay(detector: ShapeLikeDetector, image_bgr: np.ndarray, match) -> np.ndarray:
    out = image_bgr.copy()
    color = (0, 255, 0)
    t0 = detector.get_templates(match.class_id, match.template_id, backend=match.backend)[0]
    refined_transform = match.refined_transform
    for f in t0.features:
        px = float(f.x + match.x)
        py = float(f.y + match.y)
        if refined_transform is not None:
            px, py = _apply_affine_point(refined_transform, px, py)
        pxi = int(round(px))
        pyi = int(round(py))
        theta = float(f.theta)
        if not np.isfinite(theta):
            theta = label_to_angle_deg(int(f.label))
        rad = np.deg2rad(theta)
        p2x = float(f.x + match.x + 7.0 * float(np.cos(rad)))
        p2y = float(f.y + match.y + 7.0 * float(np.sin(rad)))
        if refined_transform is not None:
            p2x, p2y = _apply_affine_point(refined_transform, p2x, p2y)
        cv2.arrowedLine(out, (pxi, pyi), (int(round(p2x)), int(round(p2y))), (0, 0, 0), 3, cv2.LINE_AA, 0, 0.35)
        cv2.arrowedLine(out, (pxi, pyi), (int(round(p2x)), int(round(p2y))), color, 1, cv2.LINE_AA, 0, 0.35)
    corners = match_quad(detector, match)
    pts_i = np.array([[int(round(x)), int(round(y))] for x, y in corners], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    return out


__all__ = [
    "_arrow_endpoint",
    "_button_left",
    "_button_right",
    "_clamp_rect_to_roi",
    "_compact_path_text",
    "_cv_to_qpixmap",
    "_draw_match_overlay",
    "_orientation_palette",
    "_overlay_follow_result",
    "_shape_to_rect",
]
