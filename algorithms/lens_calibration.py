"""Standalone pinhole-lens calibration and cached image undistortion.

Profiles describe one lens configuration and one exact sensor image size. They
do not define a production measurement plane or a universal millimetres/pixel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from algorithms import image_io
from common.safe_io import atomic_write_json


MIN_CALIBRATION_VIEWS = 10
PROFILE_SCHEMA = "lc-lens-calibration-v1"


class CalibrationError(ValueError):
    """Calibration failure with the per-input report available to the caller."""

    def __init__(self, message: str, views: list[dict] | None = None):
        super().__init__(message)
        self.views = list(views or [])


@dataclass
class CalibrationProfile:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float | None = None
    pattern: str = "manual"
    pattern_size: tuple[int, int] | None = None
    spacing_mm: float | None = None
    views: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        validate_profile(self)
        return {
            "schema": PROFILE_SCHEMA,
            "image_size": list(self.image_size),
            "camera_matrix": np.asarray(self.camera_matrix, dtype=float).tolist(),
            "dist_coeffs": np.asarray(self.dist_coeffs, dtype=float).reshape(-1).tolist(),
            "rms": self.rms,
            "pattern": self.pattern,
            "pattern_size": list(self.pattern_size) if self.pattern_size else None,
            "spacing_mm": self.spacing_mm,
            "views": self.views,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProfile":
        if not isinstance(data, dict) or data.get("schema") != PROFILE_SCHEMA:
            raise ValueError("不是本 Demo 的标定参数文件，或文件版本不受支持。")
        try:
            profile = cls(
                image_size=tuple(data["image_size"]),
                camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1),
                rms=data.get("rms"),
                pattern=str(data.get("pattern", "manual")),
                pattern_size=tuple(data["pattern_size"]) if data.get("pattern_size") else None,
                spacing_mm=data.get("spacing_mm"),
                views=list(data.get("views", [])),
                warnings=list(data.get("warnings", [])),
            )
            validate_profile(profile)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"标定参数文件无效：{exc}") from exc
        return profile


def _size_tuple(value: Sequence[int], name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name}必须包含宽和高两个整数。")
    result = []
    for item in value:
        if isinstance(item, (bool, str)) or not np.isfinite(item) or int(item) != item or item < 2:
            raise ValueError(f"{name}必须为大于等于 2 的整数。")
        result.append(int(item))
    return result[0], result[1]


def validate_profile(profile: CalibrationProfile) -> None:
    """Reject malformed coefficients before they reach OpenCV or image mapping."""
    _size_tuple(profile.image_size, "图像尺寸")
    matrix = np.asarray(profile.camera_matrix, dtype=np.float64)
    distortion = np.asarray(profile.dist_coeffs, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("相机内参必须为有限数值组成的 3×3 矩阵。")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("相机焦距参数 fx、fy 必须大于 0。")
    if not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose([matrix[0, 1], matrix[1, 0]], 0):
        raise ValueError("相机内参不是受支持的标准针孔模型。")
    if distortion.size not in (4, 5) or not np.isfinite(distortion).all():
        raise ValueError("畸变系数必须为 4 或 5 个有限数值（k1、k2、p1、p2、可选 k3）。")
    if distortion.ndim not in (1, 2) or (distortion.ndim == 2 and min(distortion.shape) != 1):
        raise ValueError("畸变系数必须是一维向量。")
    if profile.rms is not None and (not np.isfinite(profile.rms) or profile.rms < 0):
        raise ValueError("重投影误差必须为非负有限数值。")
    if profile.pattern not in ("manual", "chessboard", "circles"):
        raise ValueError("标定板类型不受支持。")
    if profile.pattern_size is not None:
        _size_tuple(profile.pattern_size, "标定板点数")
    if profile.spacing_mm is not None and (not np.isfinite(profile.spacing_mm) or profile.spacing_mm <= 0):
        raise ValueError("标定板间距必须为正数。")
    if not isinstance(profile.views, list) or any(not isinstance(v, dict) for v in profile.views):
        raise ValueError("标定图片报告格式无效。")
    if not isinstance(profile.warnings, list) or any(not isinstance(v, str) for v in profile.warnings):
        raise ValueError("标定提示格式无效。")


def save_profile(profile: CalibrationProfile, path: str | Path) -> None:
    atomic_write_json(path, profile.to_dict(), ensure_ascii=False, indent=2)


def load_profile(path: str | Path) -> CalibrationProfile:
    # Do not silently restore an older optical calibration from a backup.
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return CalibrationProfile.from_dict(json.load(handle))


def _detect_pattern(gray: np.ndarray, pattern: str, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Detect on a bounded image, then refine chessboard corners at full size."""
    scale = min(1.0, 1800.0 / max(gray.shape))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray
    sx, sy = gray.shape[1] / small.shape[1], gray.shape[0] / small.shape[0]
    if pattern == "chessboard":
        found, points = cv2.findChessboardCornersSB(small, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        points = points.astype(np.float32).reshape(-1, 1, 2)
        points[:, 0, 0] *= sx
        points[:, 0, 1] *= sy
        cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        return points

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold, params.maxThreshold = 5, 250
    params.thresholdStep = 10
    params.filterByColor, params.blobColor = True, 0
    params.filterByArea, params.minArea, params.maxArea = True, 8, float(small.size / 4)
    params.filterByCircularity, params.minCircularity = True, 0.5
    params.filterByConvexity, params.minConvexity = True, 0.8
    params.filterByInertia, params.minInertiaRatio = True, 0.1
    detector = cv2.SimpleBlobDetector_create(params)
    for candidate in (small, 255 - small):
        found, points = cv2.findCirclesGrid(
            candidate, pattern_size,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
            blobDetector=detector,
        )
        if found:
            # Circle centroids need full-resolution extraction, not cornerSubPix.
            if scale < 1.0:
                params.maxArea = float(gray.size / 4)
                full_detector = cv2.SimpleBlobDetector_create(params)
                full = gray if candidate is small else 255 - gray
                found, full_points = cv2.findCirclesGrid(
                    full, pattern_size,
                    flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                    blobDetector=full_detector,
                )
                if not found:
                    return None
                points = full_points
            return points.astype(np.float32).reshape(-1, 1, 2)
    return None


def calibrate_images(
    paths: Sequence[str | Path],
    pattern: str,
    pattern_size: tuple[int, int],
    spacing_mm: float,
    progress: Callable[[int, int, str], None] | None = None,
) -> CalibrationProfile:
    """Estimate five-coefficient distortion from at least ten distinct views."""
    if pattern not in ("chessboard", "circles"):
        raise ValueError("请选择棋盘格或对称圆点阵。")
    cols, rows = _size_tuple(pattern_size, "标定板点数")
    if cols < 3 or rows < 3:
        raise ValueError("每行和每列至少需要 3 个角点或圆点。")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("标定板间距必须大于 0。")
    paths = list(paths)
    if len(paths) < MIN_CALIBRATION_VIEWS:
        raise CalibrationError(f"请至少导入 {MIN_CALIBRATION_VIEWS} 张不同位置和倾角的原始标定板图片。")
    object_grid = np.zeros((cols * rows, 3), dtype=np.float32)
    object_grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(spacing_mm)
    image_size: tuple[int, int] | None = None
    object_points, image_points, views = [], [], []
    signatures: set[bytes] = set()

    for index, path in enumerate(paths):
        entry = {"path": str(path), "status": "failed", "message": ""}
        views.append(entry)
        gray = image_io.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            entry["message"] = "无法读取图片"
        else:
            size = (gray.shape[1], gray.shape[0])
            if image_size is None:
                image_size = size
            elif size != image_size:
                entry.update(status="size_mismatch", message=f"图片尺寸 {size} 与 {image_size} 不一致")
                raise CalibrationError(f"{Path(path).name}：分辨率不一致。请使用同一相机、同一分辨率的原始图片。", views)
            signature = hashlib.sha256(np.ascontiguousarray(gray).tobytes()).digest()
            if signature in signatures:
                entry.update(status="duplicate", message="图片像素内容重复，未计入有效张数")
            else:
                signatures.add(signature)
                try:
                    points = _detect_pattern(gray, pattern, (cols, rows))
                except cv2.error as exc:
                    points = None
                    entry["message"] = f"特征检测失败：{exc}"
                if points is None:
                    entry["message"] = entry["message"] or "未找到完整图案，请检查行列数、清晰度和遮挡"
                elif points.shape != (cols * rows, 1, 2) or not np.isfinite(points).all():
                    entry["message"] = "检测到的标定点无效"
                else:
                    object_points.append(object_grid.copy())
                    image_points.append(points.astype(np.float32))
                    entry.update(status="used", message="已检测到完整图案")
        if progress:
            progress(index + 1, len(paths), f"{Path(path).name}：{entry['message']}")

    if len(image_points) < MIN_CALIBRATION_VIEWS or image_size is None:
        raise CalibrationError(
            f"有效且不重复的图片只有 {len(image_points)} 张，需要至少 {MIN_CALIBRATION_VIEWS} 张。\n"
            "请确认棋盘格填写的是内角点数，圆点阵填写的是圆心数，并补充不同位置和倾角的照片。",
            views,
        )
    if progress:
        progress(len(paths), len(paths), "正在求解相机内参、畸变系数和重投影误差…")
    try:
        rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None,
            flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9),
        )
    except cv2.error as exc:
        raise CalibrationError(f"标定求解失败：{exc}", views) from exc
    used_views = [view for view in views if view["status"] == "used"]
    for entry, obj, measured, rot, trans in zip(used_views, object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rot, trans, matrix, distortion)
        residuals = measured.reshape(-1, 2) - projected.reshape(-1, 2)
        entry["error_px"] = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    warnings = ["重投影误差仅反映这批标定图的拟合情况；请用独立标定板图片检查边缘直线和尺寸。"]
    all_points = np.concatenate(image_points).reshape(-1, 2)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    w, h = image_size
    if low[0] > 0.15 * w or high[0] < 0.85 * w or low[1] > 0.15 * h or high[1] < 0.85 * h:
        warnings.append("标定点对视野边缘的覆盖不足，长针位两端的矫正效果需要补充边缘照片后再验证。")
    centers = np.asarray([points.reshape(-1, 2).mean(axis=0) for points in image_points])
    if np.ptp(centers[:, 0]) < 0.15 * w and np.ptp(centers[:, 1]) < 0.15 * h:
        warnings.append("标定板的位置变化较小，请增加左右、上下位置及不同倾角的照片。")
    if float(rms) > 1.0:
        warnings.append("重投影误差超过 1 px，请检查图案参数、板面平整度、图像清晰度和采集姿态。")
    profile = CalibrationProfile(image_size, matrix, distortion.reshape(-1), float(rms), pattern, (cols, rows), float(spacing_mm), views, warnings)
    validate_profile(profile)
    return profile


def manual_profile(image_size: tuple[int, int], k1: float, k2: float = 0.0) -> CalibrationProfile:
    """Make an explicitly uncalibrated profile for visual experimentation."""
    w, h = _size_tuple(image_size, "图像尺寸")
    focal = float(max(w, h))
    profile = CalibrationProfile(
        (w, h),
        np.array([[focal, 0.0, (w - 1) / 2], [0.0, focal, (h - 1) / 2], [0.0, 0.0, 1.0]], dtype=np.float64),
        np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64),
        warnings=["手动试调参数：仅供观察畸变补偿效果，尚未经标定板求解，不可据此确认尺寸精度。"],
    )
    validate_profile(profile)
    return profile


class Undistorter:
    """Cache one output-to-input map; never resize or silently accept another sensor size."""

    def __init__(self, profile: CalibrationProfile, alpha: float = 1.0):
        validate_profile(profile)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("视野保留参数必须位于 0 到 1 之间。")
        self.image_size = tuple(int(v) for v in profile.image_size)
        self.camera_matrix = np.array(profile.camera_matrix, dtype=np.float64, copy=True)
        self.dist_coeffs = np.array(profile.dist_coeffs, dtype=np.float64, copy=True).reshape(-1)
        self.alpha = float(alpha)
        started = time.perf_counter()
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, self.image_size, self.alpha, self.image_size,
        )
        if not np.isfinite(self.new_camera_matrix).all() or min(self.new_camera_matrix[0, 0], self.new_camera_matrix[1, 1]) <= 0:
            raise ValueError("参数无法生成有效矫正图像，请重新标定或减小手动畸变系数。")
        self.valid_roi = tuple(int(v) for v in roi)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, self.image_size, cv2.CV_32FC1,
        )
        if not np.isfinite(self.map_x).all() or not np.isfinite(self.map_y).all():
            raise ValueError("畸变参数生成了无效像素映射，请重新标定。")
        self.map_build_ms = (time.perf_counter() - started) * 1000.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("请输入灰度或彩色图像。")
        actual_size = (image.shape[1], image.shape[0])
        if actual_size != self.image_size:
            raise ValueError(f"图片尺寸 {actual_size[0]}×{actual_size[1]} 与标定尺寸 {self.image_size[0]}×{self.image_size[1]} 不一致；请使用同一分辨率的原始图片。")
        if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
            raise ValueError("图像数据类型不受支持。")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError("图像通道数不受支持。")
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
