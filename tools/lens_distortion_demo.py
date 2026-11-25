"""Standalone lens calibration / undistortion preview. Run with Python 3.12."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.image_io import imread, imwrite
from algorithms.lens_calibration import (
    CalibrationProfile, Undistorter, calibrate_images, load_profile,
    manual_profile, save_profile,
)
from algorithms.lens_single_calibration import (
    SingleCalibrationProfile, SingleUndistorter, calibrate_single_image,
)
from common.safe_io import atomic_write_json

IMAGE_FILTER = "图片 (*.png *.bmp *.jpg *.jpeg *.tif *.tiff);;所有文件 (*)"


def load_demo_profile(path):
    """Dispatch by explicit schema; preserve compatibility with multi-view files."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and data.get("schema") in {"lc-lens-single-v1", "lc-lens-single-v2"}:
        return SingleCalibrationProfile.from_dict(data)
    return load_profile(path)


class ImageView(QtWidgets.QGraphicsView):
    """Two views share zoom and viewport center, but keep separate scenes."""
    viewChanged = QtCore.Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.corner_item = QtWidgets.QGraphicsPathItem()
        self.corner_item.setPen(QtGui.QPen(QtGui.QColor("#00c8aa"), 0))
        self.corner_item.setZValue(1)
        self.scene().addItem(self.corner_item)
        self.setBackgroundBrush(QtGui.QColor("#172331"))
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.grid = False
        self._syncing = False
        self.horizontalScrollBar().valueChanged.connect(self._broadcast)
        self.verticalScrollBar().valueChanged.connect(self._broadcast)
        self.setMinimumSize(200, 220)

    def set_image(self, image):
        self.set_corners(None)
        if image is None:
            self.item.setPixmap(QtGui.QPixmap())
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimage = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        self.item.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.setSceneRect(0, 0, w, h)

    def set_corners(self, points):
        path = QtGui.QPainterPath()
        if points is not None:
            points = np.asarray(points, dtype=float).reshape(-1, 2)
            radius = max(2.0, self.sceneRect().width() / 450.0)
            for x, y in points:
                path.addEllipse(QtCore.QPointF(float(x), float(y)), radius, radius)
            if len(points) >= 3:
                hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
                path.moveTo(float(hull[0, 0]), float(hull[0, 1]))
                for x, y in hull[1:]:
                    path.lineTo(float(x), float(y))
                path.closeSubpath()
        self.corner_item.setPath(path)

    def _broadcast(self, *_):
        if not self._syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.viewChanged.emit(self.transform().m11(), center.x(), center.y())

    def sync_view(self, scale, x, y):
        self._syncing = True
        self.setTransform(QtGui.QTransform.fromScale(scale, scale))
        self.centerOn(x, y)
        self._syncing = False

    def fit_image(self):
        if self.item.pixmap().isNull():
            return
        self._syncing = True
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._syncing = False
        self._broadcast()

    def wheelEvent(self, event):
        if self.item.pixmap().isNull():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        new_scale = self.transform().m11() * factor
        if not 0.02 <= new_scale <= 12:
            return
        self._syncing = True
        before = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        after = self.mapToScene(event.position().toPoint())
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(center + before - after)
        self._syncing = False
        self._broadcast()
        event.accept()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.grid or self.item.pixmap().isNull():
            return
        bounds = self.sceneRect()
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 230, 200, 150), 0))
        for fraction in range(1, 10):
            x, y = bounds.width() * fraction / 10, bounds.height() * fraction / 10
            painter.drawLine(QtCore.QLineF(x, 0, x, bounds.height()))
            painter.drawLine(QtCore.QLineF(0, y, bounds.width(), y))


class CalibrationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    resultReady = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, paths, pattern, pattern_size, spacing, parent=None, mode="multi"):
        super().__init__(parent)
        self.args = (paths, pattern, pattern_size, spacing)
        self.mode = mode

    def run(self):
        def progress(done, total, message):
            if self.isInterruptionRequested():
                raise RuntimeError("已取消标定。")
            self.progress.emit(done, total, message)
        try:
            if self.mode == "multi":
                result = calibrate_images(*self.args, progress=progress)
            else:
                paths, _pattern, pattern_size, spacing = self.args
                result = calibrate_single_image(paths[0], pattern_size, spacing_mm=spacing,
                                                mode=self.mode, progress=progress)
            if not self.isInterruptionRequested():
                self.resultReady.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


def synthetic_example():
    """Known synthetic barrel distortion; never represents a real camera calibration."""
    w, h = 1280, 800
    ideal = np.full((h, w, 3), 244, np.uint8)
    for x in range(40, w, 80):
        cv2.line(ideal, (x, 0), (x, h - 1), (187, 199, 207), 1)
    for y in range(40, h, 80):
        cv2.line(ideal, (0, y), (w - 1, y), (187, 199, 207), 1)
    cv2.rectangle(ideal, (80, 260), (1200, 540), (55, 71, 87), -1)
    for x in range(100, 1190, 40):
        cv2.rectangle(ideal, (x - 5, 100), (x + 5, 280), (60, 140, 206), -1)
        cv2.rectangle(ideal, (x - 5, 520), (x + 5, 700), (60, 140, 206), -1)
    cv2.line(ideal, (80, 100), (1200, 100), (74, 74, 215), 2)
    cv2.line(ideal, (80, 700), (1200, 700), (74, 74, 215), 2)
    cv2.putText(ideal, "SYNTHETIC LENS DEMO", (290, 415), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (240, 240, 240), 2, cv2.LINE_AA)
    profile = manual_profile((w, h), -0.32, 0.06)
    yy, xx = np.indices((h, w), dtype=np.float32)
    points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, profile.camera_matrix, profile.dist_coeffs,
                                    P=profile.camera_matrix).reshape(h, w, 2)
    raw = cv2.remap(ideal, undistorted[:, :, 0], undistorted[:, :, 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(244, 244, 244))
    return raw, profile


def synthetic_single_board():
    """A rasterized perspective checkerboard with known simulated barrel distortion."""
    w, h, cols, rows, tile = 1280, 800, 13, 7, 70
    texture = np.empty(((rows + 1) * tile, (cols + 1) * tile), np.uint8)
    for row in range(rows + 1):
        for col in range(cols + 1):
            texture[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = 245 if (row + col) % 2 else 15
    corners = np.float32([[0, 0], [texture.shape[1] - 1, 0],
                          [texture.shape[1] - 1, texture.shape[0] - 1], [0, texture.shape[0] - 1]])
    projected = np.float32([[70, 45], [1210, 95], [1240, 755], [40, 715]])
    homography = cv2.getPerspectiveTransform(corners, projected)
    ideal = cv2.warpPerspective(texture, homography, (w, h), borderValue=210)
    camera = np.array([[w, 0, (w - 1) / 2], [0, w, (h - 1) / 2], [0, 0, 1.0]])
    yy, xx = np.indices((h, w), dtype=np.float32)
    points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    sample = cv2.undistortPoints(points, camera, np.array([-0.28, 0.06, 0, 0, 0.0]),
                                P=camera).reshape(h, w, 2)
    raw = cv2.remap(ideal, sample[:, :, 0], sample[:, :, 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=210)
    return cv2.cvtColor(cv2.GaussianBlur(raw, (3, 3), 0.4), cv2.COLOR_GRAY2BGR), (cols, rows)


def note(text):
    widget = QtWidgets.QLabel(text)
    widget.setWordWrap(True)
    widget.setObjectName("note")
    return widget


class LensDemoWindow(QtWidgets.QMainWindow):
    def __init__(self, workdir=None):
        super().__init__()
        self.workdir = Path(workdir or ROOT).resolve()
        self.profile_directory = self.workdir / "experiments/lens_calibration/profiles"
        self.export_directory = self.workdir / "artifacts/reports/lens_distortion_demo"
        self.setWindowTitle("镜头畸变矫正 Demo · 自动单图版 · LC System")
        self.resize(1440, 900)
        self.image = None
        self.corrected = None
        self.source_path = ""
        self.profile = None
        self.profile_source = "未加载参数"
        self.rectifier = None
        self.worker = None
        self.calibration_paths = []
        self.last_times = []
        self._example_directory = None
        self._single_example_path = None
        self._build_ui()

    def _build_ui(self):
        base = QtWidgets.QWidget()
        self.setCentralWidget(base)
        layout = QtWidgets.QVBoxLayout(base)
        layout.setContentsMargins(22, 16, 22, 18)
        heading = QtWidgets.QLabel("镜头畸变矫正实验台")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(note("用标准板标定，用真实产品验证。支持桶形 / 枕形畸变，预览与生产程序独立。"))
        top = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("打开待矫正原图")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.open_image)
        self.demo_button = QtWidgets.QPushButton("长针位模拟示例")
        self.demo_button.clicked.connect(self.load_example)
        self.single_demo_button = QtWidgets.QPushButton("单图标定示例")
        self.single_demo_button.clicked.connect(self.load_single_example)
        self.image_label = QtWidgets.QLabel("先打开一张相机原图，或体验模拟示例")
        self.image_label.setWordWrap(True)
        top.addWidget(self.open_button)
        top.addWidget(self.demo_button)
        top.addWidget(self.single_demo_button)
        top.addWidget(self.image_label, 1)
        layout.addLayout(top)
        body = QtWidgets.QHBoxLayout()
        layout.addLayout(body, 1)
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setMinimumWidth(330)
        self.sidebar.setMaximumWidth(370)
        left = QtWidgets.QVBoxLayout(self.sidebar)
        left.setContentsMargins(0, 0, 12, 0)
        self.tabs = QtWidgets.QTabWidget()
        left.addWidget(self.tabs, 1)
        self._calibration_tab()
        self._manual_tab()
        controls = QtWidgets.QGroupBox("预览与导出")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("视野保留"))
        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0, 1)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1)
        self.alpha.valueChanged.connect(self._alpha_changed)
        row.addWidget(self.alpha)
        controls_layout.addLayout(row)
        self.canvas_help = note("单图模式使用固定画布，输出尺寸不变；可能出现黑边或边缘内容超出画布。")
        controls_layout.addWidget(self.canvas_help)
        row = QtWidgets.QHBoxLayout()
        fit = QtWidgets.QPushButton("适应窗口")
        fit.clicked.connect(self.fit_views)
        self.grid_check = QtWidgets.QCheckBox("参考网格")
        self.grid_check.toggled.connect(self._toggle_grid)
        row.addWidget(fit)
        row.addWidget(self.grid_check)
        controls_layout.addLayout(row)
        self.corners_check = QtWidgets.QCheckBox("显示原图标定角点与覆盖边界")
        self.corners_check.setChecked(True)
        self.corners_check.toggled.connect(self._refresh_corners)
        controls_layout.addWidget(self.corners_check)
        self.benchmark_button = QtWidgets.QPushButton("测量矫正耗时（20 次）")
        self.benchmark_button.clicked.connect(self.benchmark)
        controls_layout.addWidget(self.benchmark_button)
        self.export_button = QtWidgets.QPushButton("导出矫正图片 PNG")
        self.export_button.clicked.connect(self.export_image)
        controls_layout.addWidget(self.export_button)
        left.addWidget(controls)
        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sidebar_scroll.setMinimumWidth(355)
        sidebar_scroll.setMaximumWidth(395)
        sidebar_scroll.setWidget(self.sidebar)
        body.addWidget(sidebar_scroll)
        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, 1)
        self.profile_label = QtWidgets.QLabel("参数来源：未加载")
        self.profile_label.setObjectName("profile")
        self.profile_label.setWordWrap(True)
        right.addWidget(self.profile_label)
        compare = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.original_view, self.corrected_view = ImageView(), ImageView()
        self.original_view.viewChanged.connect(self.corrected_view.sync_view)
        self.corrected_view.viewChanged.connect(self.original_view.sync_view)
        for title, view in [("原始图像", self.original_view), ("矫正后图像", self.corrected_view)]:
            panel = QtWidgets.QWidget()
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(title)
            label.setObjectName("viewTitle")
            panel_layout.addWidget(label)
            panel_layout.addWidget(view, 1)
            compare.addWidget(panel)
        compare.setSizes([500, 500])
        right.addWidget(compare, 1)
        right.addWidget(note("滚轮同步缩放 · 按住左键拖动 · 绿色参考网格仅用于观察，不会写入导出图像"))
        self.metrics = QtWidgets.QLabel("处理耗时：—    ｜    标定误差：—")
        self.metrics.setObjectName("metrics")
        self.metrics.setWordWrap(True)
        right.addWidget(self.metrics)
        self.status = QtWidgets.QLabel("准备就绪。标定前请确认板型、行列数以及 2 mm 的实际含义。")
        self.status.setWordWrap(True)
        self.status.setObjectName("status")
        right.addWidget(self.status)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(135)
        self.log.setPlaceholderText("标定进度、每张图片的误差和参数信息将显示在这里。")
        right.addWidget(self.log)
        self._update_buttons()
        self._mode_changed()

    def _calibration_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(tab)
        self.calibration_help = note("① 导入一张清晰、平整、覆盖检测区域的棋盘格原图。默认只补偿桶形 / 枕形畸变。")
        form.addWidget(self.calibration_help)
        grid = self.grid_form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("单张 · 径向（桶形 / 枕形）", "radial")
        self.mode.addItem("单张 · 径向＋透视（固定平面）", "radial_perspective")
        self.mode.addItem("多姿态 · 相机标定（高级）", "multi")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        grid.addRow("求解模式", self.mode)
        self.auto_grid = QtWidgets.QCheckBox("自动识别可见棋盘格")
        self.auto_grid.setChecked(True)
        self.auto_grid.toggled.connect(self._auto_grid_changed)
        grid.addRow("角点识别", self.auto_grid)
        self.pattern = QtWidgets.QComboBox()
        self.pattern.addItem("棋盘格（内角点）", "chessboard")
        self.pattern.addItem("对称圆点阵（圆心）", "circles")
        self.pattern.currentIndexChanged.connect(self._pattern_changed)
        self.columns, self.rows = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin, value in [(self.columns, 9), (self.rows, 6)]:
            spin.setRange(3, 512)
            spin.setValue(value)
        self.spacing = QtWidgets.QDoubleSpinBox()
        self.spacing.setRange(0.001, 1000)
        self.spacing.setDecimals(3)
        self.spacing.setValue(2)
        self.spacing.setSuffix(" mm")
        grid.addRow("标定板类型", self.pattern)
        grid.addRow("点列数", self.columns)
        grid.addRow("点行数", self.rows)
        grid.addRow("相邻点间距", self.spacing)
        form.addLayout(grid)
        self.pattern_help = note("棋盘格填写内角点：10 × 7 个格子，对应 9 × 6 个内角点；间距填写格子边长。")
        form.addWidget(self.pattern_help)
        self.detected_grid_label = note("")
        form.addWidget(self.detected_grid_label)
        row = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("添加标定图片")
        self.add_button.clicked.connect(self.add_calibration_images)
        self.clear_button = QtWidgets.QPushButton("清空")
        self.clear_button.clicked.connect(self.clear_calibration_images)
        row.addWidget(self.add_button)
        row.addWidget(self.clear_button)
        form.addLayout(row)
        self.files = QtWidgets.QListWidget()
        self.files.setMinimumHeight(65)
        self.files.setMaximumHeight(135)
        form.addWidget(self.files, 1)
        self.count_label = note("已添加 0 张")
        form.addWidget(self.count_label)
        self.use_image_button = QtWidgets.QPushButton("使用当前打开的原图标定")
        self.use_image_button.clicked.connect(self.use_current_image)
        form.addWidget(self.use_image_button)
        row = QtWidgets.QHBoxLayout()
        self.calibrate_button = QtWidgets.QPushButton("② 计算标定并预览")
        self.calibrate_button.setObjectName("primary")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.cancel_button = QtWidgets.QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_calibration)
        row.addWidget(self.calibrate_button)
        row.addWidget(self.cancel_button)
        form.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        form.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("加载参数 JSON")
        self.load_button.clicked.connect(self.load_parameters)
        self.save_button = QtWidgets.QPushButton("保存参数 JSON")
        self.save_button.clicked.connect(self.save_parameters)
        row.addWidget(self.load_button)
        row.addWidget(self.save_button)
        form.addLayout(row)
        self.tabs.addTab(tab, "标定板求参")

    def _manual_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(tab)
        form.addWidget(note("还没有完整标定图片时，可先观察矫正方向与变化。这里采用假设的相机内参，不能作为尺寸测量标定。"))
        fields = QtWidgets.QFormLayout()
        self.k1, self.k2 = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for spin in (self.k1, self.k2):
            spin.setRange(-1.0, 1.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
        self.k1.setValue(-0.15)
        fields.addRow("径向系数 k₁", self.k1)
        fields.addRow("径向系数 k₂", self.k2)
        form.addLayout(fields)
        form.addWidget(note("一般从 k₂ = 0 开始：桶形尝试负 k₁，枕形尝试正 k₁；以标准直线验证效果。过大的系数可能使画面异常。"))
        self.manual_button = QtWidgets.QPushButton("应用试调参数")
        self.manual_button.clicked.connect(self.apply_manual)
        form.addWidget(self.manual_button)
        form.addWidget(note("矫正模型描述镜头误差。请用标定板评估效果，避免把真实弯针当成镜头畸变调整。"))
        form.addStretch()
        self.tabs.addTab(tab, "手动试调")

    def _pattern_changed(self):
        if self.mode.currentData() != "multi" and self.auto_grid.isChecked():
            self.pattern_help.setText("自动识别，无需填写行列数；可跳过部分编码区域。棋盘行列应接近水平 / 垂直，间距填写格子边长。")
        elif self.pattern.currentData() == "circles":
            self.pattern_help.setText("适用于等距对称圆点阵；填写圆心的列数、行数和相邻圆心距离。带编码点或非对称圆阵需另行适配。")
        else:
            self.pattern_help.setText("棋盘格填写内角点：10 × 7 个格子，对应 9 × 6 个内角点；间距填写格子边长。")

    def _auto_grid_changed(self):
        self._pattern_changed()
        self._update_buttons()

    def _mode_changed(self):
        mode = self.mode.currentData()
        if mode == "multi":
            self.columns.setMinimum(3)
            self.rows.setMinimum(3)
            self.calibration_help.setText("① 多姿态模式：至少 10 张不同位置、不同倾角的有效原图。这是高级模式的采集门槛。")
            self.add_button.setText("添加多张标定图片")
        else:
            self.columns.setMinimum(4)
            self.rows.setMinimum(4)
            self.pattern.setCurrentIndex(0)
            self.calibration_help.setText(
                "① 只需一张棋盘格原图，至少 20 个内角点，尽量覆盖两端和四角。校正中心固定为图像中心。"
                + (" 联合模式还会将棋盘行列对齐输出水平 / 垂直方向；标定板须处于被测平面。" if mode == "radial_perspective" else "")
            )
            self.add_button.setText("选择一张标定图片")
        self._update_count()
        self._pattern_changed()
        self._update_buttons()

    def _update_count(self):
        if self.mode.currentData() == "multi":
            self.count_label.setText(f"已添加 {len(self.calibration_paths)} 张；至少需要 10 张有效且不同的图片")
        else:
            self.count_label.setText(f"已添加 {len(self.calibration_paths)} 张；单图模式只使用列表中选中的一张")

    def use_current_image(self):
        if not self.source_path:
            self._message("请先打开一张棋盘格原图；模拟长针位图片不能用于棋盘标定。", True)
            return
        if self.source_path not in self.calibration_paths:
            self.calibration_paths.append(self.source_path)
            item = QtWidgets.QListWidgetItem(Path(self.source_path).name)
            item.setToolTip(self.source_path)
            self.files.addItem(item)
        self.files.setCurrentRow(self.calibration_paths.index(self.source_path))
        self._update_count()

    def _refresh_corners(self, *_):
        points = None
        if isinstance(self.profile, SingleCalibrationProfile) and self.corners_check.isChecked() and self.source_path:
            used_paths = [v.get("path") for v in self.profile.views if v.get("status") == "used"]
            if any(path and Path(path).resolve() == Path(self.source_path).resolve() for path in used_paths):
                points = self.profile.points
        self.original_view.set_corners(points)

    def _message(self, text, error=False):
        self.status.setText(text)
        self.status.setProperty("error", error)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.log.appendPlainText(text)

    def _update_buttons(self):
        busy = self.worker is not None and self.worker.isRunning()
        for widget in (self.mode, self.columns, self.rows, self.spacing, self.add_button,
                       self.clear_button, self.calibrate_button, self.load_button,
                       self.demo_button, self.single_demo_button, self.manual_button, self.use_image_button):
            widget.setEnabled(not busy)
        self.pattern.setEnabled(not busy and self.mode.currentData() == "multi")
        auto = self.mode.currentData() != "multi" and self.auto_grid.isChecked()
        self.auto_grid.setEnabled(not busy and self.mode.currentData() != "multi")
        for field in (self.columns, self.rows):
            self.grid_form.setRowVisible(field, not auto)
        self.detected_grid_label.setVisible(auto and bool(self.detected_grid_label.text()))
        single = isinstance(self.profile, SingleCalibrationProfile)
        self.alpha.setEnabled(not single and not busy)
        self.canvas_help.setText(
            "单图结果：输出画布尺寸固定，不额外缩放以适应全视野；可能黑边或内容超出画布。"
            if single else "0：减少黑边，可能裁掉边缘内容。\n1：保留完整视野，允许黑边；输出尺寸不变。"
        )
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(self.profile is not None and not busy)
        self.export_button.setEnabled(self.corrected is not None and not busy)
        self.benchmark_button.setEnabled(self.corrected is not None and not busy)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择相机原始图片", "", IMAGE_FILTER)
        if path:
            self.load_image_path(path)

    def load_image_path(self, path):
        marker = Path(path).with_suffix(".undistortion.json")
        if marker.exists():
            try:
                metadata = json.loads(marker.read_text(encoding="utf-8-sig"))
                if metadata.get("already_undistorted"):
                    self._message("这张图片已有本 Demo 的矫正记录，请打开相机原图，避免重复矫正。", True)
                    return
            except (OSError, ValueError, AttributeError):
                self._message("该图片旁的矫正记录无法读取，请选择可确认未经矫正的原图。", True)
                return
        image = imread(path)
        if image is None:
            self._message("无法读取图片，请检查文件格式。", True)
            return
        self.image, self.source_path = image, str(path)
        self.original_view.set_image(image)
        self.image_label.setText(f"{Path(path).name}  ·  {image.shape[1]} × {image.shape[0]} px")
        self.image_label.setToolTip(str(path))
        self.corrected = None
        self.corrected_view.set_image(None)
        self.last_times = []
        if self.profile is not None:
            self.render_correction()
        else:
            self._message("原图已打开。请标定、加载参数，或切换到手动试调。")
            self.metrics.setText("处理耗时：—    ｜    尚未矫正")
        self.fit_views()
        self._update_buttons()

    def load_example(self):
        self.image, profile = synthetic_example()
        self.source_path = ""
        self.original_view.set_image(self.image)
        self.image_label.setText("模拟长针位图像 · 1280 × 800 px（非实拍）")
        self.k1.setValue(-0.32)
        self.k2.setValue(0.06)
        self.tabs.setCurrentIndex(0)
        self.set_profile(profile, "模拟示例：已知畸变参数，非真实相机标定")
        self.fit_views()

    def load_single_example(self):
        image, pattern_size = synthetic_single_board()
        if self._example_directory is None:
            self._example_directory = tempfile.TemporaryDirectory(prefix="lc-single-lens-demo-")
        path = Path(self._example_directory.name) / "SIMULATED_checkerboard.png"
        if not imwrite(path, image):
            self._message("无法创建模拟标定图片。", True)
            return
        self._single_example_path = str(path)
        self.profile, self.rectifier = None, None
        self.profile_label.setText("参数来源：尚未求解 · 当前为模拟棋盘格原图")
        self.mode.setCurrentIndex(0)
        self.auto_grid.setChecked(True)
        self.columns.setValue(pattern_size[0])
        self.rows.setValue(pattern_size[1])
        self.spacing.setValue(2)
        self.tabs.setCurrentIndex(0)
        self.clear_calibration_images()
        self.load_image_path(str(path))
        self.use_current_image()
        self.image_label.setText("模拟棋盘格 · 1280 × 800 px · 非实拍")
        self._message("已准备 1 张模拟棋盘图（13 × 7 内角点）。点击“计算标定并预览”试用；可切换两种单图模式比较。")

    def set_profile(self, profile, source):
        self.profile, self.profile_source = profile, source
        self.rectifier = None
        self.profile_label.setText(f"参数来源：{source}  ｜  {profile.image_size[0]} × {profile.image_size[1]} px")
        if isinstance(profile, SingleCalibrationProfile):
            fraction = float(profile.coverage.get("fraction", 0))
            self.profile_label.setText(self.profile_label.text() + f"  ｜  角点覆盖：{fraction:.1%}")
            if getattr(profile, "grid_indices", None) is not None:
                columns, rows = profile.pattern_size
                self.columns.setValue(columns)
                self.rows.setValue(rows)
                self.detected_grid_label.setText(f"上次自动识别：{columns} 列 × {rows} 行范围，使用 {len(profile.points)} 个角点（允许缺点）。")
                self.log.appendPlainText(self.detected_grid_label.text())
            else:
                self.detected_grid_label.clear()
            self.log.appendPlainText(f"固定校正中心：{profile.center}\n径向系数：{profile.radial_coeffs}\n平面 H：\n{np.array2string(np.asarray(profile.homography), precision=6)}")
        else:
            self.detected_grid_label.clear()
            self.log.appendPlainText(f"K = {np.array2string(profile.camera_matrix, precision=4)}\nD = {np.array2string(profile.dist_coeffs.reshape(-1), precision=6)}")
        for warning in getattr(profile, "warnings", []):
            self.log.appendPlainText(f"提示：{warning}")
        self.render_correction()
        self._refresh_corners()
        self._update_buttons()

    def render_correction(self):
        self.corrected = None
        self.corrected_view.set_image(None)
        if self.image is None or self.profile is None:
            self._update_buttons()
            return
        try:
            if self.rectifier is None:
                self.rectifier = (SingleUndistorter(self.profile, alpha=1.0)
                                  if isinstance(self.profile, SingleCalibrationProfile)
                                  else Undistorter(self.profile, alpha=self.alpha.value()))
                self.rectifier.apply(self.image)  # Warm up; map creation is reported separately.
            started = time.perf_counter()
            self.corrected = self.rectifier.apply(self.image)
            elapsed = (time.perf_counter() - started) * 1000
            self.last_times = [elapsed]
            self.corrected_view.set_image(self.corrected)
            self.original_view._broadcast()
            rms = "无（试调参数）" if self.profile.rms is None else f"{self.profile.rms:.4f} px"
            self.metrics.setText(f"单次矫正：{elapsed:.2f} ms  ｜  建表：{self.rectifier.map_build_ms:.2f} ms  ｜  标定 RMS：{rms}")
            warnings = getattr(self.profile, "warnings", [])
            self._message("矫正完成。请放大检查长针位两端。" + (f" 有 {len(warnings)} 条标定提示，详见下方日志。" if warnings else ""))
            self._refresh_corners()
        except Exception as exc:
            self.corrected = None
            self.metrics.setText("处理耗时：—    ｜    未生成矫正结果")
            self._message(f"无法矫正：{exc}", True)
        self._update_buttons()

    def _alpha_changed(self):
        self.rectifier = None
        self.render_correction()

    def fit_views(self):
        self.original_view.fit_image()

    def _toggle_grid(self, checked):
        for view in (self.original_view, self.corrected_view):
            view.grid = checked
            view.viewport().update()

    def add_calibration_images(self):
        if self.mode.currentData() == "multi":
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择原始标定板图片", "", IMAGE_FILTER)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择一张棋盘格原图", "", IMAGE_FILTER)
            paths = [path] if path else []
        known = {str(Path(path).resolve()).casefold() for path in self.calibration_paths}
        for path in paths:
            if str(Path(path).resolve()).casefold() in known:
                continue
            self.calibration_paths.append(path)
            known.add(str(Path(path).resolve()).casefold())
            item = QtWidgets.QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            self.files.addItem(item)
        if paths:
            for index, path in enumerate(self.calibration_paths):
                if path == paths[-1]:
                    self.files.setCurrentRow(index)
                    break
        self._update_count()

    def clear_calibration_images(self):
        self.calibration_paths.clear()
        self.files.clear()
        self._update_count()

    def start_calibration(self):
        mode = self.mode.currentData()
        if mode == "multi" and len(self.calibration_paths) < 10:
            self._message("请先添加至少 10 张不同的原始标定板图片。", True)
            return
        if not self.calibration_paths:
            self._message("请先选择一张棋盘格原图。", True)
            return
        if mode != "multi":
            row = self.files.currentRow()
            paths = [self.calibration_paths[max(0, row)]]
        else:
            paths = list(self.calibration_paths)
        self.profile, self.rectifier, self.corrected = None, None, None
        self.corrected_view.set_image(None)
        self.original_view.set_corners(None)
        self.profile_label.setText("参数来源：正在重新标定")
        self.detected_grid_label.clear()
        self.metrics.setText("处理耗时：—    ｜    等待本次标定结果")
        self.log.clear()
        self._message("正在标定。图片较大时角点检测需要一些时间。")
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        if self.worker is not None:
            self.worker.deleteLater()
        pattern_size = None if mode != "multi" and self.auto_grid.isChecked() else (self.columns.value(), self.rows.value())
        self.worker = CalibrationWorker(paths, self.pattern.currentData(),
                                        pattern_size, self.spacing.value(), self, mode=mode)
        self.worker.progress.connect(self._on_progress)
        self.worker.resultReady.connect(self._on_calibrated)
        self.worker.failed.connect(self._on_calibration_failed)
        self.worker.finished.connect(self._update_buttons)
        self.worker.start()
        self._update_buttons()

    def _on_calibration_failed(self, message):
        self.profile_label.setText("参数来源：本次标定失败，未生成参数")
        self.metrics.setText("处理耗时：—    ｜    未生成矫正结果")
        self._message(message, True)

    def cancel_calibration(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self._message("已请求取消，当前图像的计算结束后停止。")

    def _on_progress(self, done, total, message):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.log.appendPlainText(message)

    def _on_calibrated(self, profile):
        used = sum(view.get("status") == "used" for view in profile.views)
        if isinstance(profile, SingleCalibrationProfile):
            mode_label = "单张径向" if profile.mode == "radial" else "单张径向＋透视"
            # Preview the calibration image first so the detected coverage is inspectable.
            path = next((v.get("path") for v in profile.views if v.get("status") == "used"), None)
            if path and str(path) == self._single_example_path:
                profile.warnings.append("此参数由模拟棋盘图求得，不能用于真实相机。")
                mode_label += " · 模拟图"
            if path:
                original = imread(path)
                if original is not None:
                    self.image, self.source_path = original, str(path)
                    self.original_view.set_image(original)
                    self.image_label.setText(f"{Path(path).name} · 标定原图 · {original.shape[1]} × {original.shape[0]} px")
            self.set_profile(profile, f"{mode_label} · {used} 张有效图")
            self.fit_views()
        else:
            self.set_profile(profile, f"多姿态标定 · {used} 张有效图")
        for view in profile.views:
            if view.get("status") == "used":
                self.log.appendPlainText(f"{Path(view['path']).name}：{view['error_px']:.4f} px")
        if self.image is None:
            self._message(f"标定完成，RMS {profile.rms:.4f} px。请打开另一张原图验证效果，并保存参数；标定提示详见下方日志。")

    def apply_manual(self):
        if self.image is None:
            self._message("请先打开待矫正原图。", True)
            return
        h, w = self.image.shape[:2]
        self.set_profile(manual_profile((w, h), self.k1.value(), self.k2.value()), "手动试调 · 假设内参，不用于尺寸标定")

    def load_parameters(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载本 Demo 保存的参数", str(self.profile_directory), "标定参数 (*.json)")
        if path:
            try:
                profile = load_demo_profile(path)
                source = ("单张径向参数" if profile.mode == "radial" else "单张径向＋透视参数") if isinstance(profile, SingleCalibrationProfile) else ("手动参数" if profile.pattern == "manual" else "多姿态标定参数")
                self.set_profile(profile, f"{source} · {Path(path).name}")
            except Exception as exc:
                self._message(f"参数加载失败：{exc}", True)

    def save_parameters(self):
        if self.profile is None:
            return
        try:
            self.profile_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._message(f"参数保存失败：{exc}", True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存参数", str(self.profile_directory / "lens_calibration.json"), "JSON (*.json)")
        if path:
            try:
                save_profile(self.profile, path)
                self._message(f"参数已保存：{path}")
            except Exception as exc:
                self._message(f"参数保存失败：{exc}", True)

    def benchmark(self):
        if self.corrected is None or self.rectifier is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            times = []
            for _ in range(3):
                self.rectifier.apply(self.image)
            for _ in range(20):
                started = time.perf_counter()
                self.rectifier.apply(self.image)
                times.append((time.perf_counter() - started) * 1000)
            self.last_times = times
            self.metrics.setText(f"20 次矫正中位数：{np.median(times):.2f} ms  ｜  P95：{np.percentile(times, 95):.2f} ms  ｜  建表：{self.rectifier.map_build_ms:.2f} ms")
            self._message("计时完成：仅含缓存映射后的 remap 和结果分配，不含读取图片、界面显示或其他检测算法。")
        except Exception as exc:
            self._message(f"计时失败：{exc}", True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_image(self):
        if self.corrected is None:
            return
        name = Path(self.source_path).stem if self.source_path else "synthetic"
        try:
            self.export_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._message(f"导出失败：{exc}", True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出矫正图片", str(self.export_directory / f"{name}_undistorted.png"), "PNG 图片 (*.png)")
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() != ".png":
            self._message("请使用 .png 后缀保存，避免有损压缩影响边缘。", True)
            return
        if self.source_path and output.resolve() == Path(self.source_path).resolve():
            self._message("请使用新的文件名保存矫正结果，保留原始图片供重新标定和对比。", True)
            return
        try:
            if not imwrite(output, self.corrected):
                raise OSError("图片写入失败")
            atomic_write_json(output.with_suffix(".undistortion.json"), {
                "already_undistorted": True, "source_image": self.source_path or "synthetic_demo",
                "profile_source": self.profile_source, "alpha": 1.0 if isinstance(self.profile, SingleCalibrationProfile) else self.alpha.value(),
                "canvas_policy": "fixed" if isinstance(self.profile, SingleCalibrationProfile) else "optimal_camera_matrix",
                "profile": self.profile.to_dict(), "remap_ms": self.last_times,
            })
            self._message(f"已导出：{output.name}，以及矫正参数记录；请勿对该图片重复矫正。")
        except Exception as exc:
            self._message(f"导出失败：{exc}", True)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.cancel_calibration()
            self._message("正在停止标定，请等待当前计算结束后再关闭窗口。")
            event.ignore()
        else:
            if self._example_directory is not None:
                self._example_directory.cleanup()
                self._example_directory = None
            event.accept()


STYLE = """
QWidget { font-family: 'Microsoft YaHei UI'; font-size: 13px; color: #203347; }
QMainWindow, QTabWidget::pane, QTabBar::tab { background: #f3f6fa; }
QLabel#heading { font-size: 25px; font-weight: 700; color: #10283f; }
QLabel#note { color: #607286; font-size: 12px; }
QLabel#profile { background: #e3edf6; border-radius: 6px; padding: 10px; color: #214e74; }
QLabel#viewTitle { font-size: 14px; font-weight: 700; padding: 7px; }
QLabel#metrics { background: #e5f2eb; color: #216346; padding: 12px; border-radius: 6px; }
QLabel#status { padding: 8px; color: #395267; }
QLabel#status[error=true] { color: #ac3333; background: #fdeaea; }
QPushButton { background: white; border: 1px solid #cbd6e1; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { border-color: #4c8dbc; background: #edf5fc; }
QPushButton#primary { background: #216c9c; color: white; border-color: #216c9c; }
QPushButton:disabled { background: #e9edf2; color: #9aa8b5; border-color: #dde4eb; }
QGroupBox { border: 1px solid #d3dde7; border-radius: 6px; margin-top: 12px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QSpinBox, QDoubleSpinBox, QComboBox { background: white; border: 1px solid #cbd6e1; border-radius: 4px; padding: 5px; }
QListWidget, QPlainTextEdit { background: white; border: 1px solid #d3dde7; border-radius: 4px; }
QPlainTextEdit { font-family: Consolas, 'Microsoft YaHei UI'; font-size: 12px; }
QTabBar::tab { padding: 9px 16px; border-bottom: 2px solid #dbe3eb; }
QTabBar::tab:selected { color: #216c9c; border-bottom-color: #216c9c; }
QProgressBar { border: 1px solid #d3dde7; border-radius: 3px; text-align: center; max-height: 16px; }
QProgressBar::chunk { background: #62a6c4; }
"""


def _run_window(args):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = LensDemoWindow(workdir=args.workdir)
    window.show()
    if args.example:
        window.load_example()
    if args.single_example:
        window.load_single_example()
    if args.profile:
        try:
            window.set_profile(load_demo_profile(args.profile), f"文件：{Path(args.profile).name}")
        except Exception as exc:
            window._message(str(exc), True)
    if args.image:
        window.load_image_path(args.image)
    QtCore.QTimer.singleShot(0, window.fit_views)
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT, help="Workspace root for profiles, exports and logs")
    parser.add_argument("--image", help="Open a raw image")
    parser.add_argument("--profile", help="Load a saved calibration JSON")
    parser.add_argument("--example", action="store_true", help="Show a synthetic barrel-distortion example")
    parser.add_argument("--single-example", action="store_true", help="Prepare one synthetic checkerboard for calibration")
    args = parser.parse_args(argv)
    args.workdir = args.workdir.resolve()
    try:
        return _run_window(args)
    except Exception:
        log = args.workdir / "artifacts/reports/lens_distortion_demo/startup_error.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
