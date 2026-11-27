"""Constrained single-plane radial calibration with a fixed distortion center.

This model estimates a planar homography and two radial coefficients. It does
not recover full camera intrinsics, tangential distortion or a 3-D scene model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np

from algorithms import image_io
from algorithms.lens_calibration import _detect_pattern, _size_tuple


SINGLE_SCHEMA = "lc-lens-single-v1"
SPARSE_SINGLE_SCHEMA = "lc-lens-single-v2"
SINGLE_SCHEMAS = {SINGLE_SCHEMA, SPARSE_SINGLE_SCHEMA}
MODES = {"radial", "radial_perspective"}


def _grid(pattern_size):
    columns, rows = _size_tuple(pattern_size, "棋盘内角点数")
    if min(columns, rows) < 4 or columns * rows < 20:
        raise ValueError("单张模式至少需要 20 个内角点，且每行、每列至少 4 点，例如 5×4。")
    points = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float64)
    center = np.array([(columns - 1) / 2, (rows - 1) / 2])
    scale = float(max(columns - 1, rows - 1))
    return (points - center) / scale, scale


def _indexed_grid(pattern_size, grid_indices=None):
    """Normalize observed integer grid locations; never invent missing points."""
    if grid_indices is None:
        return _grid(pattern_size)
    columns, rows = _size_tuple(pattern_size, "可见棋盘网格范围")
    indices = np.asarray(grid_indices)
    if indices.ndim != 2 or indices.shape[1] != 2 or len(indices) < 20:
        raise ValueError("稀疏棋盘至少需要 20 个实际检测点及其逐点对应的二维网格索引。")
    if indices.dtype.kind not in "iuf" or not np.isfinite(indices).all():
        raise ValueError("棋盘网格索引必须为有限整数。")
    if not np.equal(indices, np.rint(indices)).all():
        raise ValueError("棋盘网格索引必须是整数，不能使用小数格位。")
    if len(np.unique(indices, axis=0)) != len(indices):
        raise ValueError("棋盘网格索引重复，多个像素点不能对应同一个格点。")
    if np.min(indices, axis=0).tolist() != [0, 0] or np.max(indices, axis=0).tolist() != [columns - 1, rows - 1]:
        raise ValueError("稀疏网格索引必须从 0 开始，且最大索引应与可见网格行列范围一致。")
    if min(len(np.unique(indices[:, 0])), len(np.unique(indices[:, 1]))) < 4:
        raise ValueError("稀疏棋盘至少需要四个不同列和四个不同行的实际角点。")
    if np.linalg.matrix_rank(indices - indices.mean(axis=0)) < 2:
        raise ValueError("棋盘网格索引共线，无法确定二维平面。")
    center = np.array([(columns - 1) / 2, (rows - 1) / 2])
    scale = float(max(columns - 1, rows - 1))
    return (indices.astype(np.float64) - center) / scale, scale


def _validate_grid_step_confirmation(grid_indices, confirmed):
    if not isinstance(confirmed, (bool, np.bool_)):
        raise ValueError("棋盘相邻格距确认状态必须为布尔值。")
    if grid_indices is None or not confirmed:
        return
    locations = {tuple(point) for point in np.asarray(grid_indices).astype(np.int64)}
    x_edges = sum((column + 1, row) in locations for column, row in locations)
    y_edges = sum((column, row + 1) in locations for column, row in locations)
    if min(x_edges, y_edges) < 4:
        raise ValueError("缺少足够的双轴相邻格点，不能确认每一步对应所填写的物理格距。")


def _project(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _distort(points, coefficients):
    radius2 = np.sum(np.asarray(points) ** 2, axis=-1, keepdims=True)
    return points * (1 + coefficients[0] * radius2 + coefficients[1] * radius2 ** 2)


def _local_geometry(homography):
    center = homography[:2, 2] / homography[2, 2]
    derivative = (homography[:2, :2] - np.outer(center, homography[2, :2])) / homography[2, 2]
    determinant = float(np.linalg.det(derivative))
    if not np.isfinite(determinant) or abs(determinant) < 1e-9:
        raise ValueError("棋盘平面映射退化，无法确定有效的矫正尺度。")
    return center, float(np.sqrt(abs(determinant)))


def _minimum_polynomial(a, b, maximum_radius2):
    values = [1.0, 1.0 + a * maximum_radius2 + b * maximum_radius2 ** 2]
    if b > 0:
        vertex = -a / (2 * b)
        if 0 < vertex < maximum_radius2:
            values.append(1 + a * vertex + b * vertex ** 2)
    return min(values)


def _validate_radial(coefficients, radius_max):
    k1, k2 = coefficients
    radius2 = float(radius_max) ** 2
    scale_min = _minimum_polynomial(k1, k2, radius2)
    derivative_min = _minimum_polynomial(3 * k1, 5 * k2, radius2)
    if scale_min <= 0.05 or derivative_min <= 0.05:
        raise ValueError("拟合的径向模型在输出画布内发生折叠或接近折叠；请使用覆盖更大、更加清晰的原始棋盘图。")
    return scale_min, derivative_min


def _mapping_geometry(profile):
    width, height = profile.image_size
    scale = float(max(width, height))
    center = np.asarray(profile.center, np.float64)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float64)
    if profile.mode == "radial":
        ideal = (corners - center) / scale
        return center, scale, float(np.linalg.norm(ideal, axis=1).max())
    ideal_center, area_scale = _local_geometry(profile.homography)
    output_center = center + scale * ideal_center
    plane_scale = scale * area_scale
    plane_corners = (corners - output_center) / plane_scale
    denominators = np.column_stack((plane_corners, np.ones(4))) @ profile.homography[2]
    if np.min(denominators) <= 0.05:
        raise ValueError("棋盘平面在输出画布内接近无穷远或发生翻折；请减小拍摄倾角，并让棋盘覆盖更大视野。")
    ideal = _project(plane_corners, profile.homography)
    return output_center, plane_scale, float(np.linalg.norm(ideal, axis=1).max())


@dataclass
class SingleCalibrationProfile:
    image_size: tuple[int, int]
    mode: str
    center: tuple[float, float]
    radial_coeffs: np.ndarray
    homography: np.ndarray
    pattern_size: tuple[int, int]
    spacing_mm: float
    rms: float
    views: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    coverage: dict = field(default_factory=dict)
    grid_indices: np.ndarray | None = None
    grid_step_confirmed: bool = False

    @property
    def pattern(self):
        return "chessboard"

    @property
    def pixel_size_mm(self):
        """Nominal plane scale; meaningful only on this rectified board plane."""
        if self.mode != "radial_perspective" or (self.grid_indices is not None and not self.grid_step_confirmed):
            return None
        columns, rows = _size_tuple(self.pattern_size, "棋盘网格范围")
        board_scale = float(max(columns - 1, rows - 1))
        _, plane_scale, _ = _mapping_geometry(self)
        return float(self.spacing_mm * board_scale / plane_scale)

    def to_dict(self):
        validate_single_profile(self)
        payload = {
            "schema": SINGLE_SCHEMA if self.grid_indices is None else SPARSE_SINGLE_SCHEMA,
            "image_size": list(self.image_size), "mode": self.mode,
            "center": list(self.center),
            "radial_coeffs": np.asarray(self.radial_coeffs).reshape(-1).tolist(),
            "homography": np.asarray(self.homography).tolist(),
            "pattern_size": list(self.pattern_size), "spacing_mm": float(self.spacing_mm),
            "rms": float(self.rms), "views": copy.deepcopy(self.views), "warnings": list(self.warnings),
            "points": np.asarray(self.points).reshape(-1, 2).tolist(), "coverage": copy.deepcopy(self.coverage),
            "coordinate_model": "center-fixed; grid-center/max-grid-span -> H -> normalized-ideal -> radial; image-scale=max(width,height)",
            "viewport": "fixed-original-size; no automatic crop or rescale",
        }
        if self.grid_indices is not None:
            payload["grid_indices"] = np.asarray(self.grid_indices, np.int64).tolist()
            payload["grid_step_confirmed"] = bool(self.grid_step_confirmed)
        return payload

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get("schema") not in SINGLE_SCHEMAS:
            raise ValueError("不是受支持的单张棋盘标定参数文件。")
        try:
            sparse = data["schema"] == SPARSE_SINGLE_SCHEMA
            if sparse and data.get("grid_indices") is None:
                raise ValueError("稀疏参数缺少实际角点的网格索引。")
            if not sparse and data.get("grid_indices") is not None:
                raise ValueError("旧版参数不能包含稀疏网格索引，请使用 v2 格式。")
            result = cls(
                tuple(data["image_size"]), str(data["mode"]), tuple(data["center"]),
                np.asarray(data["radial_coeffs"], np.float64), np.asarray(data["homography"], np.float64),
                tuple(data["pattern_size"]), float(data["spacing_mm"]), float(data["rms"]),
                list(data.get("views", [])), list(data.get("warnings", [])),
                np.asarray(data["points"], np.float64), dict(data.get("coverage", {})),
                np.asarray(data["grid_indices"]) if sparse else None,
                data.get("grid_step_confirmed", False) if sparse else False,
            )
            validate_single_profile(result)
            if result.grid_indices is not None:
                result.grid_indices = result.grid_indices.astype(np.int64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"单张标定参数无效：{exc}") from exc
        return result


def validate_single_profile(profile):
    width, height = _size_tuple(profile.image_size, "图像尺寸")
    grid, _ = _indexed_grid(profile.pattern_size, profile.grid_indices)
    _validate_grid_step_confirmation(profile.grid_indices, profile.grid_step_confirmed)
    if profile.mode not in MODES:
        raise ValueError("单张矫正模式不受支持。")
    center = np.asarray(profile.center, np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("畸变中心必须是两个有限坐标。")
    if not np.allclose(center, [(width - 1) / 2, (height - 1) / 2], atol=1e-6, rtol=0):
        raise ValueError("此单张模型只支持固定在图像中心的畸变中心。")
    coefficients = np.asarray(profile.radial_coeffs, np.float64)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
        raise ValueError("径向系数必须包含有限的 k1、k2 两个数值。")
    homography = np.asarray(profile.homography, np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("平面变换必须是有限的 3×3 矩阵。")
    if abs(homography[2, 2] - 1) > 1e-8 or abs(np.linalg.det(homography)) < 1e-9 or np.linalg.cond(homography) > 1e8:
        raise ValueError("平面变换奇异或未正确归一化。")
    # Check all four grid-boundary corners, including holes that were not observed.
    columns, rows = profile.pattern_size
    boundary_scale = float(max(columns - 1, rows - 1))
    boundary = (np.array([[0, 0], [columns - 1, 0], [columns - 1, rows - 1], [0, rows - 1]])
                - [(columns - 1) / 2, (rows - 1) / 2]) / boundary_scale
    denominators = np.column_stack((boundary, np.ones(4))) @ homography[2]
    if np.min(denominators) <= 0.05:
        raise ValueError("棋盘范围内的平面映射存在翻折或无穷远点。")
    if not np.isfinite(profile.rms) or profile.rms < 0 or not np.isfinite(profile.spacing_mm) or profile.spacing_mm <= 0:
        raise ValueError("标定误差和棋盘间距无效。")
    points = np.asarray(profile.points, np.float64)
    if points.shape != (len(grid), 2) or not np.isfinite(points).all():
        raise ValueError("棋盘角点数量或坐标无效。")
    if len(np.unique(points, axis=0)) != len(points) or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("棋盘角点重复或退化为一条线。")
    if points[:, 0].min() < 0 or points[:, 0].max() >= width or points[:, 1].min() < 0 or points[:, 1].max() >= height:
        raise ValueError("棋盘角点超出图像范围。")
    if not isinstance(profile.views, list) or any(not isinstance(view, dict) for view in profile.views):
        raise ValueError("图片报告格式无效。")
    if not isinstance(profile.warnings, list) or any(not isinstance(item, str) for item in profile.warnings):
        raise ValueError("标定提示格式无效。")
    if not isinstance(profile.coverage, dict):
        raise ValueError("棋盘覆盖信息格式无效。")
    for key, value in profile.coverage.items():
        try:
            numeric = np.asarray(value, np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"棋盘覆盖信息 {key} 必须为有限数值。") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"棋盘覆盖信息 {key} 含有无效数值。")
        if key != "bounds" and numeric.ndim != 0:
            raise ValueError(f"棋盘覆盖信息 {key} 必须为单个数值。")
    if "fraction" in profile.coverage:
        fraction = np.asarray(profile.coverage["fraction"])
        if fraction.ndim != 0 or not 0 <= float(fraction) <= 1:
            raise ValueError("棋盘覆盖比例必须位于 0 到 1 之间。")
    if "bounds" in profile.coverage and np.asarray(profile.coverage["bounds"]).shape != (4,):
        raise ValueError("棋盘覆盖边界必须含有四个坐标。")
    _, _, radius_max = _mapping_geometry(profile)
    fitted_radius = np.linalg.norm(_project(boundary, homography), axis=1).max()
    _validate_radial(coefficients, max(radius_max, fitted_radius))


def calibrate_single_points(points, image_size, pattern_size, spacing_mm=2.0, mode="radial", progress=None,
                            grid_indices=None, grid_step_confirmed=False):
    """Fit observed grid points, optionally with explicit sparse integer indices."""
    # The preview also uses these profile types. Load the solver only when fitting.
    from scipy.optimize import least_squares

    width, height = _size_tuple(image_size, "图像尺寸")
    grid, _ = _indexed_grid(pattern_size, grid_indices)
    _validate_grid_step_confirmation(grid_indices, grid_step_confirmed)
    if mode not in MODES or not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("矫正模式或棋盘间距无效。")
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if points.shape != (len(grid), 2) or not np.isfinite(points).all():
        raise ValueError("角点必须与网格索引逐点对应（完整棋盘为列数×行数），且坐标必须为有限数值。")
    if len(np.unique(points, axis=0)) != len(points) or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("棋盘角点重复或共线，无法标定。")
    if points[:, 0].min() < 0 or points[:, 0].max() >= width or points[:, 1].min() < 0 or points[:, 1].max() >= height:
        raise ValueError("棋盘角点超出图像范围。")
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = (upper - lower) / np.array([width, height])
    hull = cv2.convexHull(points.astype(np.float32))
    fraction = float(cv2.contourArea(hull) / (width * height))
    if fraction < 0.025 or min(span) < 0.12:
        raise ValueError("棋盘覆盖范围过小，单张图无法可靠约束边缘畸变；请让棋盘覆盖更大视野后重新拍摄。")
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    scale = float(max(width, height))
    observed = (points - center) / scale
    radius_limit = float(np.linalg.norm(np.maximum(center, [width - 1, height - 1] - center) / scale))
    radial_samples = np.linspace(0, radius_limit ** 2, 24)

    def unpack(parameters):
        return np.append(parameters[:8], 1.0).reshape(3, 3), parameters[8:]

    def residual(parameters):
        homography, coefficients = unpack(parameters)
        homogeneous = np.column_stack((grid, np.ones(len(grid)))) @ homography.T
        denominators = homogeneous[:, 2:3]
        ideal = homogeneous[:, :2] / np.maximum(denominators, 0.02)
        predicted = _distort(ideal, coefficients)
        geometric = (predicted - observed).ravel() * scale
        denominator_penalty = np.minimum(denominators[:, 0] - 0.1, 0) * scale * 5
        derivative = 1 + 3 * coefficients[0] * radial_samples + 5 * coefficients[1] * radial_samples ** 2
        monotonic_penalty = np.minimum(derivative - 0.1, 0) * scale
        return np.concatenate((geometric, denominator_penalty, monotonic_penalty))

    results = []
    initial_matrix = np.array([[scale, 0, center[0]], [0, scale, center[1]], [0, 0, 1]], np.float64)
    for seed_index, seed in enumerate((-0.6, -0.2, 0.0, 0.2, 0.6), start=1):
        if progress:
            progress(1, 2, f"正在拟合径向畸变和平面姿态：初值 {seed_index}/5…")
        ideal_seed = cv2.undistortPoints(points.reshape(-1, 1, 2), initial_matrix, np.array([seed, 0., 0., 0., 0.])).reshape(-1, 2)
        homography, _ = cv2.findHomography(grid, ideal_seed, method=0)
        if homography is None or not np.isfinite(homography).all():
            continue
        homography /= homography[2, 2]
        initial = np.concatenate((homography.reshape(-1)[:8], [seed, 0.]))
        initial[:8] = np.clip(initial[:8], -19.9, 19.9)
        fit = least_squares(residual, initial, bounds=(np.array([-20.] * 8 + [-3., -10.]), np.array([20.] * 8 + [3., 10.])),
                            loss="soft_l1", f_scale=0.5, x_scale="jac", max_nfev=1200,
                            ftol=1e-11, xtol=1e-11, gtol=1e-9)
        if fit.success and np.isfinite(fit.x).all():
            results.append(fit)
    if not results:
        raise ValueError("单张参数拟合未收敛，请检查棋盘是否清晰、完整且平整。")
    fit = min(results, key=lambda result: result.cost)
    homography, coefficients = unpack(fit.x)
    predicted = center + scale * _distort(_project(grid, homography), coefficients)
    errors = np.linalg.norm(predicted - points, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    if rms > 2.0:
        raise ValueError(f"单张拟合误差过大（RMS {rms:.3f} px）；请检查行列数、板面平整度和角点检测。")
    if np.any(np.abs(coefficients) > [2.99, 9.99]):
        raise ValueError("拟合参数触及模型边界，单张图片不足以约束当前畸变；请更换覆盖更完整的棋盘照片。")
    fitted_ideal = _project(grid, homography)
    # Measure how much radial information remains after allowing the plane H
    # to absorb its eight nuisance degrees of freedom. Small singular values
    # flag weak radial constraints even when the same-image RMS looks good.
    jacobian = fit.jac[:2 * len(points)]
    nuisance = jacobian[:, :8]
    radial = jacobian[:, 8:]
    independent_radial = radial - nuisance @ np.linalg.lstsq(nuisance, radial, rcond=None)[0]
    radial_singular_values = np.linalg.svd(independent_radial, compute_uv=False)
    radial_sigma_min = float(radial_singular_values[-1])
    radial_condition = float(radial_singular_values[0] / max(radial_sigma_min, 1e-15))
    coverage = {"fraction": fraction, "bounds": [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])],
                "span_x": float(span[0]), "span_y": float(span[1]),
                "radius_max": float(np.linalg.norm(fitted_ideal, axis=1).max()),
                "image_corner_radius": radius_limit, "grid_columns": int(pattern_size[0]), "grid_rows": int(pattern_size[1]),
                "max_residual_px": float(errors.max()),
                "radial_sigma_min": radial_sigma_min, "radial_condition": radial_condition,
                "detected_points": int(len(points)),
                "visible_grid_slots": int(pattern_size[0] * pattern_size[1])}
    warnings = ["单张受约束拟合：畸变中心固定在图像中心，只估计 k1/k2；不代表完整相机内参或切向畸变标定。",
                "RMS 是本张棋盘的拟合误差，请用另一张原图验证边缘效果。固定画布可能出现黑边或内容超出边界。"]
    if fraction < 0.25 or min(span) < 0.65 or coverage["radius_max"] < 0.8 * radius_limit:
        warnings.append("棋盘没有覆盖整个视野；棋盘外区域属于外推，尤其四角的矫正和尺寸精度尚未验证。")
    if rms > 0.5:
        warnings.append("拟合 RMS 超过 0.5 px，请放大检查角点和板面平整度。")
    if radial_sigma_min < 1.0 or radial_condition > 500:
        warnings.append("本张棋盘对两个径向系数的独立约束较弱；即使 RMS 较低，边缘外推也可能不准，请扩大棋盘覆盖范围验证。")
    if mode == "radial_perspective":
        warnings.append("平面拉正和标称 mm/px 仅适用于本次棋盘所在平面；不同高度的针脚不能直接沿用该比例。")
    if grid_indices is not None:
        warnings.append("使用实际检测到的可见格点拟合；编码区或遮挡区的缺失角点未补造。")
        if not grid_step_confirmed:
            warnings.append("尚未确认索引相邻一步对应一个真实方格；当前仅作几何预览，不输出标称 mm/px。")
    profile = SingleCalibrationProfile((width, height), mode, tuple(center), coefficients.copy(), homography.copy(),
                                       tuple(pattern_size), float(spacing_mm), rms,
                                       [{"path": "", "status": "used", "error_px": rms, "message": "单张棋盘拟合"}],
                                       warnings, points.copy(), coverage,
                                       None if grid_indices is None else np.asarray(grid_indices, np.int64).copy(),
                                       bool(grid_step_confirmed))
    validate_single_profile(profile)
    return profile


def calibrate_single_image(path, pattern_size=None, spacing_mm=2.0, mode="radial", progress=None):
    """Read one raw board; None selects indexed visible-grid detection."""
    path = Path(path)
    sidecar = path.with_suffix(".undistortion.json")
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError("发现矫正记录但无法读取，请选用相机原始图片。") from exc
        if isinstance(metadata, dict) and metadata.get("already_undistorted"):
            raise ValueError("这张图片已有矫正记录，请使用相机原始棋盘图进行标定。")
    if pattern_size is not None:
        _grid(pattern_size)
    if progress:
        progress(0, 2, "正在读取原始棋盘并检测内角点…")
    image = image_io.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("无法读取棋盘图片。")
    grid_indices, diagnostics = None, None
    if pattern_size is None:
        from algorithms.lens_grid_detection import detect_checker_grid
        points, grid_indices, pattern_size, diagnostics = detect_checker_grid(image, progress=progress)
        if not isinstance(diagnostics, dict):
            raise ValueError("自动网格检测返回了无效的诊断信息。")
    else:
        points = _detect_pattern(image, "chessboard", tuple(pattern_size))
        if points is None:
            raise ValueError("未找到完整棋盘，请检查内角点行列数、清晰度和遮挡；带编码块或被裁切的大棋盘可改用自动可见网格。")
    if progress:
        progress(1, 2, f"已检测到 {len(points)} 个角点，正在联合拟合径向畸变和平面姿态…")
    try:
        profile = calibrate_single_points(
            points, (image.shape[1], image.shape[0]), pattern_size, spacing_mm, mode, progress=progress,
            grid_indices=grid_indices,
            grid_step_confirmed=diagnostics.get("grid_step_confirmed", False) if diagnostics else False,
        )
    except ValueError as exc:
        if grid_indices is not None:
            raise ValueError(f"已识别 {len(points)} 个可见格点（网格范围 {pattern_size[0]}×{pattern_size[1]}），但标定未通过：{exc}") from exc
        raise
    profile.views[0]["path"] = str(path)
    if diagnostics is not None:
        profile.views[0]["detection"] = copy.deepcopy(diagnostics)
        profile.views[0]["message"] = "自动可见格点拟合（缺失角点未补造）"
        for warning in diagnostics.get("warnings", []):
            if isinstance(warning, str) and warning not in profile.warnings:
                profile.warnings.append(warning)
    if progress:
        progress(2, 2, f"单张拟合完成：RMS {profile.rms:.4f} px")
    return profile


class SingleUndistorter:
    """One composite inverse map on a fixed-size canvas; alpha is fixed at one."""

    def __init__(self, profile, alpha=1.0):
        validate_single_profile(profile)
        if not np.isfinite(alpha) or abs(alpha - 1.0) > 1e-12:
            raise ValueError("单张模式使用固定画布，视野参数只能为 1；不会自动裁剪或缩放。")
        self.image_size = tuple(profile.image_size)
        self.output_center, self.plane_scale, radius_max = _mapping_geometry(profile)
        self.pixel_size_mm = profile.pixel_size_mm
        started = time.perf_counter()
        width, height = self.image_size
        scale = float(max(width, height))
        yy, xx = np.indices((height, width), dtype=np.float32)
        if profile.mode == "radial":
            qx = (xx - profile.center[0]) / scale
            qy = (yy - profile.center[1]) / scale
        else:
            gx = (xx - self.output_center[0]) / self.plane_scale
            gy = (yy - self.output_center[1]) / self.plane_scale
            matrix = profile.homography
            denominator = matrix[2, 0] * gx + matrix[2, 1] * gy + 1
            qx = (matrix[0, 0] * gx + matrix[0, 1] * gy + matrix[0, 2]) / denominator
            qy = (matrix[1, 0] * gx + matrix[1, 1] * gy + matrix[1, 2]) / denominator
        radius2 = qx * qx + qy * qy
        factor = 1 + profile.radial_coeffs[0] * radius2 + profile.radial_coeffs[1] * radius2 * radius2
        self.map_x = np.asarray(profile.center[0] + scale * qx * factor, dtype=np.float32)
        self.map_y = np.asarray(profile.center[1] + scale * qy * factor, dtype=np.float32)
        if not np.isfinite(self.map_x).all() or not np.isfinite(self.map_y).all():
            raise ValueError("无法生成有限的矫正映射。")
        valid = (self.map_x >= 0) & (self.map_x <= width - 1) & (self.map_y >= 0) & (self.map_y <= height - 1)
        self.valid_mask = valid
        # A conservative fully valid rectangle; it is not guaranteed maximal.
        self.valid_roi = _conservative_valid_rectangle(valid)
        self.map_build_ms = (time.perf_counter() - started) * 1000

    def apply(self, image):
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("请输入灰度或彩色原图。")
        if (image.shape[1], image.shape[0]) != self.image_size:
            raise ValueError(f"图片尺寸与单张标定尺寸 {self.image_size[0]}×{self.image_size[1]} 不一致。")
        if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
            raise ValueError("图像数据类型不受支持。")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError("图像通道数不受支持。")
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _conservative_valid_rectangle(mask):
    """Trim invalid borders to obtain a valid rectangle without cropping output."""
    height, width = mask.shape
    if not np.any(mask):
        return (0, 0, 0, 0)
    # Repeatedly trim the side containing the most invalid boundary pixels.
    left, top, right, bottom = 0, 0, width, height
    while left < right and top < bottom:
        counts = [int(np.count_nonzero(~mask[top, left:right])),
                  int(np.count_nonzero(~mask[bottom - 1, left:right])),
                  int(np.count_nonzero(~mask[top:bottom, left])),
                  int(np.count_nonzero(~mask[top:bottom, right - 1]))]
        if max(counts) == 0:
            # A valid border encloses a valid interior for the checked nonfolding mapping.
            if mask[top:bottom, left:right].all():
                return (left, top, right - left, bottom - top)
            return (0, 0, 0, 0)
        side = int(np.argmax(counts))
        if side == 0:
            top += 1
        elif side == 1:
            bottom -= 1
        elif side == 2:
            left += 1
        else:
            right -= 1
    return (0, 0, 0, 0)
