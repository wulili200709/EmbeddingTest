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
    MIN_CALIBRATION_VIEWS, CalibrationProfile, Undistorter, calibrate_images, load_profile,
    manual_profile, save_profile,
)
from algorithms.lens_single_calibration import (
    SingleCalibrationProfile, SingleUndistorter, calibrate_single_image,
)
from common.safe_io import atomic_write_json
from ui.lens_calibration_theme import IMAGE_BACKGROUND, apply_lens_calibration_theme
from ui.lens_calibration_text import diagnostic_text, lens_label

def image_filter():
    formats = "(*.png *.bmp *.jpg *.jpeg *.tif *.tiff)"
    return lens_label(f"图片 {formats};;所有文件 (*)", f"Images {formats};;All files (*)")


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
        self.setBackgroundBrush(QtGui.QColor(IMAGE_BACKGROUND))
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
                raise RuntimeError(lens_label("已取消标定。", "Calibration canceled."))
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
        apply_lens_calibration_theme(self)
        self.workdir = Path(workdir or ROOT).resolve()
        self.profile_directory = self.workdir / "experiments/lens_calibration/profiles"
        self.export_directory = self.workdir / "artifacts/reports/lens_distortion_demo"
        self.setWindowTitle(lens_label("镜头标定与预览", "Lens calibration and preview"))
        screen = self.screen().availableGeometry()
        self.resize(min(1180, screen.width() - 32), min(750, screen.height() - 64))
        self.image = None
        self.corrected = None
        self.source_path = ""
        self.profile = None
        self.profile_source = ""
        self.rectifier = None
        self.worker = None
        self.calibration_paths = []
        self.last_times = []
        self._example_directory = None
        self._single_example_path = None
        self._camera_target = None
        self._apply_to_camera_callback = None
        self._build_ui()

    def _build_ui(self):
        base = QtWidgets.QWidget()
        self.setCentralWidget(base)
        layout = QtWidgets.QVBoxLayout(base)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        header = QtWidgets.QHBoxLayout()
        self.camera_context_label = note("")
        self.camera_context_label.hide()
        header.addWidget(self.camera_context_label, 1)
        self.camera_apply_hint = QtWidgets.QLabel()
        self.camera_apply_hint.setObjectName("note")
        self.camera_apply_hint.hide()
        header.addWidget(self.camera_apply_hint)
        self.apply_camera_button = QtWidgets.QPushButton()
        self.apply_camera_button.setObjectName("primary")
        self.apply_camera_button.clicked.connect(self.apply_to_camera)
        self.apply_camera_button.hide()
        header.addWidget(self.apply_camera_button)
        layout.addLayout(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        layout.addLayout(body, 1)
        self.sidebar = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(self.sidebar)
        left.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        left.addWidget(self.tabs)
        self._calibration_tab()
        self._manual_tab()

        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0, 1)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1)
        self.alpha.valueChanged.connect(self._alpha_changed)
        self.preview_form = QtWidgets.QFormLayout()
        self.preview_form.addRow(lens_label("视野保留", "Retained field of view"), self.alpha)
        left.addLayout(self.preview_form)
        left.addStretch()
        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(330)
        sidebar_scroll.setWidget(self.sidebar)
        body.addWidget(sidebar_scroll)

        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, 1)
        toolbar = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton(lens_label("打开原图…", "Open image…"))
        self.open_button.clicked.connect(self.open_image)
        toolbar.addWidget(self.open_button)
        self.image_label = note("")
        toolbar.addWidget(self.image_label, 1)
        fit = QtWidgets.QPushButton(lens_label("适应窗口", "Fit view"))
        fit.clicked.connect(self.fit_views)
        toolbar.addWidget(fit)
        more = QtWidgets.QPushButton(lens_label("更多", "More"))
        menu = QtWidgets.QMenu(more)
        more.setMenu(menu)
        self.grid_check = menu.addAction(lens_label("参考网格", "Reference grid"))
        self.grid_check.setCheckable(True)
        self.grid_check.toggled.connect(self._toggle_grid)
        self.corners_check = menu.addAction(lens_label("显示标定角点", "Show corners"))
        self.corners_check.setCheckable(True)
        self.corners_check.setChecked(True)
        self.corners_check.toggled.connect(self._refresh_corners)
        menu.addSeparator()
        self.load_button = menu.addAction(lens_label("导入参数 JSON…", "Import profile JSON…"), self.load_parameters)
        self.save_button = menu.addAction(lens_label("导出参数 JSON…", "Export profile JSON…"), self.save_parameters)
        self.export_button = menu.addAction(lens_label("导出矫正图…", "Export corrected image…"), self.export_image)
        self.benchmark_button = menu.addAction(lens_label("测量矫正耗时", "Measure correction time"), self.benchmark)
        self.examples_menu = menu.addMenu(lens_label("模拟示例", "Examples"))
        self.demo_button = self.examples_menu.addAction(lens_label("长针位", "Pin array"), self.load_example)
        self.single_demo_button = self.examples_menu.addAction(lens_label("棋盘格", "Checkerboard"), self.load_single_example)
        toolbar.addWidget(more)
        right.addLayout(toolbar)
        self.profile_label = note("")
        right.addWidget(self.profile_label)
        compare = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.original_view, self.corrected_view = ImageView(), ImageView()
        self.original_view.viewChanged.connect(self.corrected_view.sync_view)
        self.corrected_view.viewChanged.connect(self.original_view.sync_view)
        for title, view in [(lens_label("原图", "Original"), self.original_view), (lens_label("矫正后", "Corrected"), self.corrected_view)]:
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
        self.metrics = note("")
        right.addWidget(self.metrics)
        footer = QtWidgets.QHBoxLayout()
        self.status = note("")
        self.status.setObjectName("status")
        footer.addWidget(self.status, 1)
        details = QtWidgets.QPushButton(lens_label("日志", "Log"))
        details.setCheckable(True)
        footer.addWidget(details)
        right.addLayout(footer)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.hide()
        details.toggled.connect(self.log.setVisible)
        right.addWidget(self.log)
        self._mode_changed()

    def _calibration_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(tab)
        form.setSpacing(8)
        grid = self.grid_form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem(lens_label("单张 · 径向", "Single image · radial"), "radial")
        self.mode.addItem(lens_label("单张 · 径向＋透视", "Radial + perspective"), "radial_perspective")
        self.mode.addItem(lens_label("多姿态", "Multi-view"), "multi")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        grid.addRow(lens_label("模式", "Mode"), self.mode)
        self.auto_grid = QtWidgets.QCheckBox(lens_label("自动识别", "Detect automatically"))
        self.auto_grid.setChecked(True)
        self.auto_grid.toggled.connect(self._update_buttons)
        grid.addRow(lens_label("棋盘角点", "Grid detection"), self.auto_grid)
        self.pattern = QtWidgets.QComboBox()
        self.pattern.addItem(lens_label("棋盘格", "Checkerboard"), "chessboard")
        self.pattern.addItem(lens_label("对称圆点阵", "Symmetric circles"), "circles")
        self.columns, self.rows = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin, value in [(self.columns, 9), (self.rows, 6)]:
            spin.setRange(3, 512)
            spin.setValue(value)
        self.spacing = QtWidgets.QDoubleSpinBox()
        self.spacing.setRange(0.001, 1000)
        self.spacing.setDecimals(3)
        self.spacing.setValue(2)
        self.spacing.setSuffix(" mm")
        grid.addRow(lens_label("标定板", "Board type"), self.pattern)
        grid.addRow(lens_label("点列数", "Columns"), self.columns)
        grid.addRow(lens_label("点行数", "Rows"), self.rows)
        grid.addRow(lens_label("点间距", "Point spacing"), self.spacing)
        form.addLayout(grid)
        self.detected_grid_label = note("")
        form.addWidget(self.detected_grid_label)
        row = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton()
        self.add_button.clicked.connect(self.add_calibration_images)
        self.clear_button = QtWidgets.QPushButton(lens_label("清空", "Clear"))
        self.clear_button.clicked.connect(self.clear_calibration_images)
        row.addWidget(self.add_button)
        row.addWidget(self.clear_button)
        form.addLayout(row)
        self.files = QtWidgets.QListWidget()
        self.files.setMinimumHeight(58)
        self.files.setMaximumHeight(94)
        form.addWidget(self.files, 1)
        self.count_label = note("")
        form.addWidget(self.count_label)
        self.use_image_button = QtWidgets.QPushButton(lens_label("使用当前原图", "Use current image"))
        self.use_image_button.clicked.connect(self.use_current_image)
        form.addWidget(self.use_image_button)
        row = QtWidgets.QHBoxLayout()
        self.calibrate_button = QtWidgets.QPushButton(lens_label("计算标定", "Calibrate"))
        self.calibrate_button.setObjectName("primary")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.cancel_button = QtWidgets.QPushButton(lens_label("取消", "Cancel"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_calibration)
        row.addWidget(self.calibrate_button)
        row.addWidget(self.cancel_button)
        form.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        form.addWidget(self.progress)
        form.addStretch()
        self.tabs.addTab(tab, lens_label("标定", "Calibration"))

    def _manual_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(tab)
        fields = QtWidgets.QFormLayout()
        self.k1, self.k2 = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for spin in (self.k1, self.k2):
            spin.setRange(-1.0, 1.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
        self.k1.setValue(-0.15)
        fields.addRow(lens_label("径向系数 k₁", "Radial k₁"), self.k1)
        fields.addRow(lens_label("径向系数 k₂", "Radial k₂"), self.k2)
        form.addLayout(fields)
        self.manual_button = QtWidgets.QPushButton(lens_label("预览参数", "Preview parameters"))
        self.manual_button.clicked.connect(self.apply_manual)
        form.addWidget(self.manual_button)
        form.addStretch()
        self.tabs.addTab(tab, lens_label("手动预览", "Manual preview"))

    def _mode_changed(self):
        multi = self.mode.currentData() == "multi"
        self.columns.setMinimum(3 if multi else 4)
        self.rows.setMinimum(3 if multi else 4)
        if not multi:
            self.pattern.setCurrentIndex(0)
        self.add_button.setText(lens_label("添加标定图…", "Add board images…") if multi else lens_label("选择标定图…", "Select board image…"))
        self._update_buttons()

    def use_current_image(self):
        if not self.source_path:
            self._message(lens_label("请先打开一张棋盘格原图。", "Open an original checkerboard image first."), True)
            return
        if self.source_path not in self.calibration_paths:
            self.calibration_paths.append(self.source_path)
            item = QtWidgets.QListWidgetItem(Path(self.source_path).name)
            item.setToolTip(self.source_path)
            self.files.addItem(item)
        self.files.setCurrentRow(self.calibration_paths.index(self.source_path))
        self._update_buttons()

    def _refresh_corners(self, *_):
        points = None
        if isinstance(self.profile, SingleCalibrationProfile) and self.corners_check.isChecked() and self.source_path:
            used_paths = [v.get("path") for v in self.profile.views if v.get("status") == "used"]
            if any(path and Path(path).resolve() == Path(self.source_path).resolve() for path in used_paths):
                points = self.profile.points
        self.original_view.set_corners(points)

    def _message(self, text, error=False):
        text = diagnostic_text(text)
        self.status.setText(text)
        self.status.setProperty("error", error)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        if text:
            self.log.appendPlainText(text)

    def _update_buttons(self):
        busy = self.worker is not None and self.worker.isRunning()
        for widget in (self.mode, self.columns, self.rows, self.spacing, self.add_button,
                       self.clear_button, self.load_button,
                       self.demo_button, self.single_demo_button, self.manual_button, self.use_image_button):
            widget.setEnabled(not busy)
        multi = self.mode.currentData() == "multi"
        self.grid_form.setRowVisible(self.pattern, multi)
        self.pattern.setEnabled(not busy)
        self.grid_form.setRowVisible(self.auto_grid, not multi)
        auto = self.mode.currentData() != "multi" and self.auto_grid.isChecked()
        self.auto_grid.setEnabled(not busy and self.mode.currentData() != "multi")
        for field in (self.columns, self.rows):
            self.grid_form.setRowVisible(field, not auto)
        self.detected_grid_label.setVisible(auto and bool(self.detected_grid_label.text()))
        single = isinstance(self.profile, SingleCalibrationProfile)
        self.alpha.setEnabled(not single and not busy)
        self.preview_form.setRowVisible(self.alpha, self.profile is not None and not single)
        self.profile_label.setVisible(bool(self.profile_label.text()))
        self.metrics.setVisible(bool(self.metrics.text()))
        self.use_image_button.setVisible(bool(self.source_path) and self.source_path not in self.calibration_paths)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(self.profile is not None and not busy)
        self.export_button.setEnabled(self.corrected is not None and not busy)
        self.benchmark_button.setEnabled(self.corrected is not None and not busy)
        self.apply_camera_button.setEnabled(
            self._camera_target is not None and not busy and self._can_apply_camera_profile()
        )
        count = len(self.calibration_paths)
        minimum = MIN_CALIBRATION_VIEWS if multi else 1
        self.calibrate_button.setEnabled(not busy and count >= minimum)
        self.count_label.setText(lens_label(f"{count} 张 / 至少 {minimum} 张", f"{count} images / {minimum} required"))
        self.count_label.setVisible(multi)
        self.files.setVisible(count > 0)
        self.clear_button.setVisible(count > 0)

    def set_camera_context(self, *, product_dir, camera_role, camera_serial="", apply_callback):
        """Bind this editor to the selected physical camera, never the next selected row."""
        self._camera_target = (str(Path(product_dir).resolve()), str(camera_role), str(camera_serial))
        self._apply_to_camera_callback = apply_callback
        product = Path(product_dir).name
        serial = camera_serial or lens_label("未绑定", "Unbound")
        self.setWindowTitle(lens_label(f"镜头标定 · {camera_role}", f"Lens calibration · {camera_role}"))
        self.camera_context_label.setText(lens_label(
            f"产品  {product}     /     相机  {camera_role}     /     序列号  {serial}",
            f"Product  {product}     /     Camera  {camera_role}     /     Serial  {serial}",
        ))
        self.camera_context_label.setToolTip(self._camera_target[0])
        self.camera_context_label.show()
        self.examples_menu.menuAction().setVisible(False)
        self.apply_camera_button.setText(lens_label(f"应用到 {camera_role}", f"Apply to {camera_role}"))
        self.apply_camera_button.show()
        self._update_buttons()

    def _can_apply_camera_profile(self):
        if self.profile is None or self.profile.rms is None or getattr(self.profile, "pattern", "") == "manual":
            return False
        if any("模拟" in str(value) or "synthetic" in str(value).lower()
               for value in getattr(self.profile, "warnings", [])):
            return False
        return not any(
            str(view.get("path", "")) == str(self._single_example_path) or
            Path(str(view.get("path", ""))).name.upper().startswith("SIMULATED_")
            for view in getattr(self.profile, "views", [])
        )

    def _mark_camera_preview_pending(self):
        if self._camera_target is not None:
            role = self._camera_target[1]
            self.camera_apply_hint.setText(lens_label(f"尚未应用到 {role}", f"Not applied to {role}"))
            self.camera_apply_hint.setVisible(self.profile is not None)

    def apply_to_camera(self):
        if self._camera_target is None or not callable(self._apply_to_camera_callback):
            return
        if self.worker is not None and self.worker.isRunning():
            return
        if not self._can_apply_camera_profile():
            self._message(lens_label("请先使用实际相机原图完成标定。", "Calibrate with real camera images first."), True)
            return
        role = self._camera_target[1]
        alpha = 1.0 if isinstance(self.profile, SingleCalibrationProfile) else self.alpha.value()
        try:
            applied = self._apply_to_camera_callback(self.profile, alpha)
        except Exception as exc:
            self._message(str(exc), True)
            return
        if applied:
            self.camera_apply_hint.setText(lens_label(f"已保存并启用到 {role}", f"Applied to {role}"))
            self._message("")

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, lens_label("打开原图", "Open original image"), "", image_filter())
        if path:
            self.load_image_path(path)

    def load_image_path(self, path):
        marker = Path(path).with_suffix(".undistortion.json")
        if marker.exists():
            try:
                metadata = json.loads(marker.read_text(encoding="utf-8-sig"))
                if metadata.get("already_undistorted"):
                    self._message(lens_label("图片已矫正，请选择原图以免重复矫正。", "This image is already corrected. Select the original."), True)
                    return
            except (OSError, ValueError, AttributeError):
                self._message(lens_label("矫正记录无法读取，请选择原图。", "Unreadable correction record. Select the original image."), True)
                return
        image = imread(path)
        if image is None:
            self._message(lens_label("无法读取图片，请检查文件格式。", "Cannot read the image. Check the file format."), True)
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
            self._message("")
            self.metrics.clear()
        self.fit_views()
        self._update_buttons()

    def load_example(self):
        self.image, profile = synthetic_example()
        self.source_path = ""
        self.original_view.set_image(self.image)
        self.image_label.setText(lens_label("模拟长针位 · 1280 × 800 px", "Synthetic pin array · 1280 × 800 px"))
        self.k1.setValue(-0.32)
        self.k2.setValue(0.06)
        self.tabs.setCurrentIndex(0)
        self.set_profile(profile, lens_label("模拟长针位", "Synthetic pin array"))
        self.fit_views()

    def load_single_example(self):
        self._mark_camera_preview_pending()
        image, pattern_size = synthetic_single_board()
        if self._example_directory is None:
            self._example_directory = tempfile.TemporaryDirectory(prefix="lc-single-lens-demo-")
        path = Path(self._example_directory.name) / "SIMULATED_checkerboard.png"
        if not imwrite(path, image):
            self._message(lens_label("无法创建模拟标定图片。", "Cannot create the example image."), True)
            return
        self._single_example_path = str(path)
        self.profile, self.rectifier = None, None
        self.profile_label.setText(lens_label("模拟棋盘格", "Synthetic checkerboard"))
        self.mode.setCurrentIndex(0)
        self.auto_grid.setChecked(True)
        self.columns.setValue(pattern_size[0])
        self.rows.setValue(pattern_size[1])
        self.spacing.setValue(2)
        self.tabs.setCurrentIndex(0)
        self.clear_calibration_images()
        self.load_image_path(str(path))
        self.use_current_image()
        self.image_label.setText(lens_label("模拟棋盘格 · 1280 × 800 px", "Synthetic checkerboard · 1280 × 800 px"))
        self._message(lens_label("模拟棋盘已加载 · 13 × 7 内角点", "Example loaded · 13 × 7 inner corners"))

    def set_profile(self, profile, source):
        self.profile, self.profile_source = profile, source
        self._mark_camera_preview_pending()
        self.rectifier = None
        self.profile_label.setText(f"{source}  ·  {profile.image_size[0]} × {profile.image_size[1]} px")
        if isinstance(profile, SingleCalibrationProfile):
            fraction = float(profile.coverage.get("fraction", 0))
            self.profile_label.setText(self.profile_label.text() + lens_label(f"  ·  覆盖 {fraction:.1%}", f"  ·  Coverage {fraction:.1%}"))
            if getattr(profile, "grid_indices", None) is not None:
                columns, rows = profile.pattern_size
                self.columns.setValue(columns)
                self.rows.setValue(rows)
                self.detected_grid_label.setText(lens_label(f"自动识别 {columns} × {rows} · {len(profile.points)} 个角点",
                                                   f"Detected {columns} × {rows} · {len(profile.points)} corners"))
                self.log.appendPlainText(self.detected_grid_label.text())
            else:
                self.detected_grid_label.clear()
            self.log.appendPlainText(f"c = {profile.center}\nk = {profile.radial_coeffs}\nH =\n{np.array2string(np.asarray(profile.homography), precision=6)}")
        else:
            self.detected_grid_label.clear()
            self.log.appendPlainText(f"K = {np.array2string(profile.camera_matrix, precision=4)}\nD = {np.array2string(profile.dist_coeffs.reshape(-1), precision=6)}")
        for warning in getattr(profile, "warnings", []):
            self.log.appendPlainText(diagnostic_text(warning))
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
            rms = "—" if self.profile.rms is None else f"{self.profile.rms:.4f} px"
            self.metrics.setText(lens_label(f"矫正 {elapsed:.2f} ms", f"Correction {elapsed:.2f} ms") + f"   ·   RMS {rms}")
            self._message("")
            self._refresh_corners()
        except Exception as exc:
            self.corrected = None
            self.metrics.clear()
            self._message(lens_label("无法矫正", "Correction failed") + " · " + diagnostic_text(exc), True)
        self._update_buttons()

    def _alpha_changed(self):
        self._mark_camera_preview_pending()
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
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, lens_label("选择标定图", "Select board images"), "", image_filter())
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, lens_label("选择一张棋盘格原图", "Select checkerboard image"), "", image_filter())
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
        self._update_buttons()

    def clear_calibration_images(self):
        self.calibration_paths.clear()
        self.files.clear()
        self._update_buttons()

    def start_calibration(self):
        mode = self.mode.currentData()
        if mode == "multi" and len(self.calibration_paths) < MIN_CALIBRATION_VIEWS:
            self._message(lens_label(f"请先添加至少 {MIN_CALIBRATION_VIEWS} 张不同的原始标定板图片。",
                            f"Add at least {MIN_CALIBRATION_VIEWS} distinct board images."), True)
            return
        if not self.calibration_paths:
            self._message(lens_label("请先选择一张棋盘格原图。", "Select a checkerboard image first."), True)
            return
        if mode != "multi":
            row = self.files.currentRow()
            paths = [self.calibration_paths[max(0, row)]]
        else:
            paths = list(self.calibration_paths)
        self.profile, self.rectifier, self.corrected = None, None, None
        self._mark_camera_preview_pending()
        self.corrected_view.set_image(None)
        self.original_view.set_corners(None)
        self.profile_label.setText(lens_label("正在标定…", "Calibrating…"))
        self.detected_grid_label.clear()
        self.metrics.clear()
        self.log.clear()
        self._message(lens_label("正在检测标定板…", "Detecting board…"))
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
        self.profile_label.setText(lens_label("标定失败", "Calibration failed"))
        self.metrics.clear()
        self._message(message, True)

    def cancel_calibration(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self._message(lens_label("正在取消…", "Canceling after the current image…"))

    def _on_progress(self, done, total, message):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.log.appendPlainText(diagnostic_text(message))

    def _on_calibrated(self, profile):
        used = sum(view.get("status") == "used" for view in profile.views)
        if isinstance(profile, SingleCalibrationProfile):
            mode_label = lens_label("单张径向", "Single-image radial") if profile.mode == "radial" else lens_label("单张径向＋透视", "Radial + perspective")
            # Preview the calibration image first so the detected coverage is inspectable.
            path = next((v.get("path") for v in profile.views if v.get("status") == "used"), None)
            if path and str(path) == self._single_example_path:
                profile.warnings.append("此参数由模拟棋盘图求得，不能用于真实相机。")
                mode_label += lens_label(" · 模拟图", " · synthetic")
            if path:
                original = imread(path)
                if original is not None:
                    self.image, self.source_path = original, str(path)
                    self.original_view.set_image(original)
                    self.image_label.setText(f"{Path(path).name} · {original.shape[1]} × {original.shape[0]} px")
            self.set_profile(profile, mode_label)
            self.fit_views()
        else:
            self.set_profile(profile, lens_label(f"多姿态 · {used} 张", f"Multi-view · {used} images"))
        for view in profile.views:
            if view.get("status") == "used":
                self.log.appendPlainText(f"{Path(view['path']).name} · {view['error_px']:.4f} px")
        if self.image is None:
            self._message(lens_label(f"标定完成 · RMS {profile.rms:.4f} px", f"Calibrated · RMS {profile.rms:.4f} px"))

    def apply_manual(self):
        if self.image is None:
            self._message(lens_label("请先打开待矫正原图。", "Open an original image first."), True)
            return
        h, w = self.image.shape[:2]
        self.set_profile(manual_profile((w, h), self.k1.value(), self.k2.value()), lens_label("手动预览", "Manual preview"))

    def load_parameters(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, lens_label("导入标定参数", "Import calibration profile"), str(self.profile_directory), lens_label("标定参数 (*.json)", "Calibration profiles (*.json)"))
        if path:
            try:
                profile = load_demo_profile(path)
                self.set_profile(profile, Path(path).name)
            except Exception as exc:
                self._message(lens_label("参数加载失败", "Cannot load profile") + " · " + diagnostic_text(exc), True)

    def save_parameters(self):
        if self.profile is None:
            return
        try:
            self.profile_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._message(lens_label("参数保存失败", "Cannot save profile") + " · " + diagnostic_text(exc), True)
            return
        filename = f"{self._camera_target[1]}_lens_calibration.json" if self._camera_target else "lens_calibration.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, lens_label("保存参数", "Save profile"), str(self.profile_directory / filename), "JSON (*.json)")
        if path:
            try:
                save_profile(self.profile, path)
                self._message(lens_label(f"已保存 {path}", f"Saved {path}"))
            except Exception as exc:
                self._message(lens_label("参数保存失败", "Cannot save profile") + " · " + diagnostic_text(exc), True)

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
            self.metrics.setText(f"P50 {np.median(times):.2f} ms   ·   P95 {np.percentile(times, 95):.2f} ms")
            self._message(lens_label("计时完成。", "Timing complete."))
        except Exception as exc:
            self._message(lens_label("计时失败", "Timing failed") + " · " + diagnostic_text(exc), True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_image(self):
        if self.corrected is None:
            return
        name = Path(self.source_path).stem if self.source_path else "synthetic"
        try:
            self.export_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._message(lens_label("导出失败", "Export failed") + " · " + diagnostic_text(exc), True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, lens_label("导出矫正图", "Export corrected image"), str(self.export_directory / f"{name}_undistorted.png"), lens_label("PNG 图片 (*.png)", "PNG (*.png)"))
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() != ".png":
            self._message(lens_label("请使用 .png 后缀。", "Use a .png filename."), True)
            return
        if self.source_path and output.resolve() == Path(self.source_path).resolve():
            self._message(lens_label("请使用新文件名保留原图。", "Choose a new filename to preserve the original image."), True)
            return
        try:
            if not imwrite(output, self.corrected):
                raise OSError(lens_label("图片写入失败", "Cannot write image"))
            atomic_write_json(output.with_suffix(".undistortion.json"), {
                "already_undistorted": True, "source_image": self.source_path or "synthetic_demo",
                "profile_source": self.profile_source, "alpha": 1.0 if isinstance(self.profile, SingleCalibrationProfile) else self.alpha.value(),
                "canvas_policy": "fixed" if isinstance(self.profile, SingleCalibrationProfile) else "optimal_camera_matrix",
                "profile": self.profile.to_dict(), "remap_ms": self.last_times,
            })
            self._message(lens_label(f"已导出 {output.name}", f"Exported {output.name}"))
        except Exception as exc:
            self._message(lens_label("导出失败", "Export failed") + " · " + diagnostic_text(exc), True)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.cancel_calibration()
            self._message(lens_label("标定正在停止，完成后可关闭窗口。", "Stopping calibration. Close the window when it finishes."))
            event.ignore()
        else:
            if self._example_directory is not None:
                self._example_directory.cleanup()
                self._example_directory = None
            event.accept()


def _run_window(args):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = LensDemoWindow(workdir=args.workdir)
    window.show()
    if args.example:
        window.load_example()
    if args.single_example:
        window.load_single_example()
    if args.profile:
        try:
            window.set_profile(load_demo_profile(args.profile), args.profile.name)
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
