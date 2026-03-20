"""
qr_gui_pyside6.py

一个轻量桌面工具：
- 选择 OK/NG/TEST 图片（默认使用当前目录下 OK/NG/TEST 文件夹）
- 在图片上画 ROI（矩形），保存为同名 labelme json（label=roi）
- 选择 backbone / score_mode(proto|topk) / margin / topk
- 训练注册（用 OK/NG 的 ROI embedding）
- 对 TEST 批量预测，表格显示结果

依赖：PySide6, torch, torchvision, pillow, numpy
"""

from __future__ import annotations

import os
import json
import time
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

from PySide6 import QtCore, QtGui, QtWidgets

from embedding_analysis_dialog import EmbeddingAnalysisDialog
import line2dup_locator
from line2dup_recipe import Line2DupRecipe
from line2dup_template_page_pyside6 import Line2DupTemplateDialog
from product_params import ProductRuntimeParams, load_product_params, save_product_params
import qr_core
from roi_canvas_pyside6 import OverlayShape, RoiCanvas
from traditional_algorithms import (
    TRADITIONAL_ALGORITHMS,
    TraditionalThresholdModel,
    compute_roi_metrics,
    is_traditional_algorithm,
    metric_value,
    train_threshold_model,
)


SUPPORTED_EMBEDDING_ALGORITHMS = ["efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large"]
SUPPORTED_ALGORITHMS = SUPPORTED_EMBEDDING_ALGORITHMS + TRADITIONAL_ALGORITHMS
SUPPORTED_SCORE_MODES = ["proto", "topk"]
SUPPORTED_LOC_MODES = ["line2dup"]
SUPPORTED_SHAPES = ["rect", "polygon"]
ROI_OVERLAY_PALETTE = [
    QtGui.QColor(255, 215, 0),
    QtGui.QColor(255, 64, 128),
    QtGui.QColor(0, 0, 255),
    QtGui.QColor(0, 255, 128),
    QtGui.QColor(255, 128, 0),
    QtGui.QColor(128, 255, 0),
]


def _find_default_dir(name: str) -> str:
    # 兼容 TEST/test
    if name.lower() == "test":
        if os.path.isdir("TEST"):
            return "TEST"
        if os.path.isdir("test"):
            return "test"
        if os.path.isdir("Test"):
            return "Test"
        return "TEST"
    return name


def _load_folder_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return qr_core.load_images(folder)


def _pixmap_from_path(path: str) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(path)
    return pm


@dataclass
class ShapeState:
    # rect: xywh; polygon: points
    shape_type: str = "rect"  # "rect" | "polygon"
    xywh: Optional[Tuple[int, int, int, int]] = None
    points: Optional[List[Tuple[float, float]]] = None


class ImageCanvas(QtWidgets.QLabel):
    """
    用 QLabel 显示图片，并支持鼠标拖拽画矩形 ROI。
    ROI 统一保存为原图坐标（不是缩放坐标）。
    """

    shapesChanged = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setScaledContents(False)
        self.setMouseTracking(True)

        self._img_path: Optional[str] = None
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._scaled_pm: Optional[QtGui.QPixmap] = None
        self._scale: float = 1.0
        self._offset = QtCore.QPoint(0, 0)  # top-left of scaled image inside label

        self._dragging = False
        self._p0 = QtCore.QPoint()
        self._p1 = QtCore.QPoint()

        self.roi = ShapeState()
        self.draw_shape = "rect"  # "rect" or "polygon"

        # polygon drawing
        self._poly_pts: List[Tuple[float, float]] = []
        self._mouse_pos: Optional[QtCore.QPoint] = None  # 当前鼠标位置（用于显示实时线段）
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def has_image(self) -> bool:
        return self._pixmap is not None and self._img_path is not None

    def image_path(self) -> Optional[str]:
        return self._img_path

    def load_image(self, path: str):
        self._img_path = path
        self._pixmap = _pixmap_from_path(path)
        self.roi = ShapeState()
        self._dragging = False
        self._p0 = QtCore.QPoint()
        self._p1 = QtCore.QPoint()
        self._poly_pts = []
        # load existing json if present
        j = qr_core.labelme_json_of_image(path)
        if os.path.exists(j):
            # 先尝试读取polygon points
            poly_pts = qr_core.try_read_polygon_points_from_labelme(j, "roi")
            if poly_pts and len(poly_pts) >= 3:
                self.roi.shape_type = "polygon"
                self.roi.points = poly_pts
                self.roi.xywh = None
            else:
                # 如果没有polygon，读取rect
                xywh = qr_core.try_read_xywh_from_labelme(j, "roi")
                if xywh:
                    self.roi.shape_type = "rect"
                    self.roi.xywh = xywh
                    self.roi.points = None
        self._update_scaled_pixmap()
        self.update()
        self.shapesChanged.emit()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap is None:
            self.setPixmap(QtGui.QPixmap())
            self._scaled_pm = None
            self._scale = 1.0
            self._offset = QtCore.QPoint(0, 0)
            return

        label_w = max(1, self.width())
        label_h = max(1, self.height())
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0:
            return

        scale = min(label_w / pm_w, label_h / pm_h)
        new_w = max(1, int(pm_w * scale))
        new_h = max(1, int(pm_h * scale))
        scaled = self._pixmap.scaled(new_w, new_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._scaled_pm = scaled
        self._scale = float(scale)
        off_x = int((label_w - new_w) / 2)
        off_y = int((label_h - new_h) / 2)
        self._offset = QtCore.QPoint(off_x, off_y)
        self.setPixmap(scaled)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def _pos_to_image_xy(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        """
        把 label 上的坐标映射回原图像素坐标。
        """
        if self._pixmap is None or self._scaled_pm is None:
            return None
        x = pos.x() - self._offset.x()
        y = pos.y() - self._offset.y()
        if x < 0 or y < 0 or x >= self._scaled_pm.width() or y >= self._scaled_pm.height():
            return None
        ix = int(round(x / self._scale))
        iy = int(round(y / self._scale))
        ix = max(0, min(ix, self._pixmap.width() - 1))
        iy = max(0, min(iy, self._pixmap.height() - 1))
        return ix, iy

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self.has_image():
            return

        # polygon: left click add point, right click finish
        if self.draw_shape == "polygon":
            if e.button() == QtCore.Qt.LeftButton:
                p = self._pos_to_image_xy(e.position().toPoint())
                if p is None:
                    return
                self._poly_pts.append((float(p[0]), float(p[1])))
                self.update()
                self.shapesChanged.emit()
                return
            if e.button() == QtCore.Qt.RightButton:
                # finish polygon (need >=3 points)
                if len(self._poly_pts) >= 3:
                    self.roi.shape_type = "polygon"
                    self.roi.points = list(self._poly_pts)
                    self.roi.xywh = None
                self._poly_pts = []
                self.update()
                self.shapesChanged.emit()
                return

        # rect
        if e.button() == QtCore.Qt.LeftButton:
            p = self._pos_to_image_xy(e.position().toPoint())
            if p is not None:
                self._dragging = True
                self._p0 = QtCore.QPoint(*p)
                self._p1 = QtCore.QPoint(*p)
                self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._dragging and self.has_image():
            p = self._pos_to_image_xy(e.position().toPoint())
            if p is not None:
                self._p1 = QtCore.QPoint(*p)
                self.update()
        # polygon模式下跟踪鼠标位置，用于显示实时线段
        if self.draw_shape == "polygon" and self.has_image() and self._poly_pts:
            self._mouse_pos = self._pos_to_image_xy(e.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton and self._dragging and self.has_image():
            self._dragging = False
            p = self._pos_to_image_xy(e.position().toPoint())
            if p is not None:
                self._p1 = QtCore.QPoint(*p)
            x0, y0 = self._p0.x(), self._p0.y()
            x1, y1 = self._p1.x(), self._p1.y()
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if w >= 1 and h >= 1:
                self.roi.shape_type = "rect"
                self.roi.xywh = (int(x), int(y), int(w), int(h))
                self.roi.points = None
            else:
                self.roi.xywh = None
                self.roi.points = None
            self.update()
            self.shapesChanged.emit()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        # polygon editing
        if self.draw_shape == "polygon":
            if e.key() == QtCore.Qt.Key_Escape:
                self._poly_pts = []
                self.update()
                self.shapesChanged.emit()
                return
            if e.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                if self._poly_pts:
                    self._poly_pts.pop()
                    self.update()
                    self.shapesChanged.emit()
                return
        super().keyPressEvent(e)

    def paintEvent(self, e: QtGui.QPaintEvent):
        super().paintEvent(e)
        if not self.has_image():
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        def draw_rect(xywh, color):
            x, y, w, h = xywh
            sx = int(round(x * self._scale)) + self._offset.x()
            sy = int(round(y * self._scale)) + self._offset.y()
            sw = int(round(w * self._scale))
            sh = int(round(h * self._scale))
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(QtCore.QRect(sx, sy, sw, sh))

        def draw_poly(points, color):
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            qpts = []
            for x, y in points:
                sx = int(round(x * self._scale)) + self._offset.x()
                sy = int(round(y * self._scale)) + self._offset.y()
                qpts.append(QtCore.QPoint(sx, sy))
            if len(qpts) >= 2:
                painter.drawPolyline(QtGui.QPolygon(qpts))

        # ROI (green)
        if self._scaled_pm is not None:
            if self.roi.shape_type == "rect" and self.roi.xywh is not None:
                draw_rect(self.roi.xywh, QtGui.QColor(0, 255, 0))
            if self.roi.shape_type == "polygon" and self.roi.points is not None:
                draw_poly(self.roi.points + [self.roi.points[0]], QtGui.QColor(0, 255, 0))

        # current polygon points (yellow)
        if self._scaled_pm is not None and self._poly_pts:
            draw_poly(self._poly_pts, QtGui.QColor(255, 255, 0))
            # 画点
            for x, y in self._poly_pts:
                sx = int(round(x * self._scale)) + self._offset.x()
                sy = int(round(y * self._scale)) + self._offset.y()
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 1))
                painter.drawEllipse(QtCore.QPoint(sx, sy), 3, 3)
            # 显示从最后一个点到鼠标的实时线段
            if self._mouse_pos is not None:
                last_x, last_y = self._poly_pts[-1]
                last_sx = int(round(last_x * self._scale)) + self._offset.x()
                last_sy = int(round(last_y * self._scale)) + self._offset.y()
                mouse_sx = int(round(self._mouse_pos[0] * self._scale)) + self._offset.x()
                mouse_sy = int(round(self._mouse_pos[1] * self._scale)) + self._offset.y()
                pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 128), 1, QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(last_sx, last_sy, mouse_sx, mouse_sy)

        # draw current dragging box
        if self._dragging and self._scaled_pm is not None:
            x0, y0 = self._p0.x(), self._p0.y()
            x1, y1 = self._p1.x(), self._p1.y()
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            sx = int(round(x * self._scale)) + self._offset.x()
            sy = int(round(y * self._scale)) + self._offset.y()
            sw = int(round(w * self._scale))
            sh = int(round(h * self._scale))
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QtCore.QRect(sx, sy, sw, sh))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quick Register (OK/NG) - ROI 标注/训练/测试")

        self._session_dir = os.path.join(os.path.dirname(__file__), ".qr_session")
        self._products_json = os.path.join(self._session_dir, "products.json")
        
        # 加载产品配置
        self.products = self._load_products()
        self.current_product = self.products.get("current_product", "Default")
        
        # 设置当前产品的路径
        self._update_product_paths()

        self.ok_dir = _find_default_dir("OK")
        self.ng_dir = _find_default_dir("NG")
        self.test_dir = _find_default_dir("TEST")

        # 启动默认空列表：用户手动添加即可
        self.ok_files: List[str] = []
        self.ng_files: List[str] = []
        self.test_files: List[str] = []

        self.model: Optional[qr_core.RegisterModel] = None
        self.ref_image: Optional[str] = None
        self.loc_method: str = "line2dup"
        self.line2dup_recipe: Optional[Line2DupRecipe] = None
        self.product_params = ProductRuntimeParams()
        self._line2dup_match_ms_by_image: Dict[str, float] = {}
        self._line2dup_autogen_ms_by_image: Dict[str, float] = {}
        self._current_result_rows: List[Dict[str, object]] = []
        self._updating_runtime_params = False

        self._build_ui()
        self._refresh_lists()
        self._try_load_session_on_start()

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QHBoxLayout(cw)

        # left: lists + controls
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        # 产品选择区域
        product_box = QtWidgets.QGroupBox("产品")
        product_layout = QtWidgets.QHBoxLayout(product_box)
        product_layout.addWidget(QtWidgets.QLabel("当前产品："))
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.addItems(self.products["products"])
        self.cmb_product.setCurrentText(self.current_product)
        self.cmb_product.currentTextChanged.connect(self._on_product_changed)
        product_layout.addWidget(self.cmb_product, 1)
        self.btn_new_product = QtWidgets.QPushButton("新建")
        self.btn_new_product.clicked.connect(self._new_product)
        product_layout.addWidget(self.btn_new_product)
        left.addWidget(product_box)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        left.addWidget(self.tabs, 1)

        # OK tab
        self.ok_list = QtWidgets.QListWidget()
        self.ok_list.itemSelectionChanged.connect(self._on_select_ok)
        ok_tab = QtWidgets.QWidget()
        ok_l = QtWidgets.QVBoxLayout(ok_tab)
        ok_l.addWidget(QtWidgets.QLabel("OK 图片（选择一张可标 ROI）"))
        ok_l.addWidget(self.ok_list, 1)
        btns = QtWidgets.QHBoxLayout()
        self.btn_add_ok = QtWidgets.QPushButton("添加到 OK…")
        self.btn_add_ok.clicked.connect(lambda: self._add_images_to("OK"))
        self.btn_del_ok = QtWidgets.QPushButton("从 OK 移除")
        self.btn_del_ok.clicked.connect(lambda: self._remove_selected_from("OK"))
        btns.addWidget(self.btn_add_ok)
        btns.addWidget(self.btn_del_ok)
        ok_l.addLayout(btns)
        self.tabs.addTab(ok_tab, "OK")

        # NG tab
        self.ng_list = QtWidgets.QListWidget()
        self.ng_list.itemSelectionChanged.connect(self._on_select_ng)
        ng_tab = QtWidgets.QWidget()
        ng_l = QtWidgets.QVBoxLayout(ng_tab)
        ng_l.addWidget(QtWidgets.QLabel("NG 图片（选择一张可标 ROI）"))
        ng_l.addWidget(self.ng_list, 1)
        btns2 = QtWidgets.QHBoxLayout()
        self.btn_add_ng = QtWidgets.QPushButton("添加到 NG…")
        self.btn_add_ng.clicked.connect(lambda: self._add_images_to("NG"))
        self.btn_del_ng = QtWidgets.QPushButton("从 NG 移除")
        self.btn_del_ng.clicked.connect(lambda: self._remove_selected_from("NG"))
        btns2.addWidget(self.btn_add_ng)
        btns2.addWidget(self.btn_del_ng)
        ng_l.addLayout(btns2)
        self.tabs.addTab(ng_tab, "NG")

        # TEST tab
        self.test_list = QtWidgets.QListWidget()
        self.test_list.itemSelectionChanged.connect(self._on_select_test)
        test_tab = QtWidgets.QWidget()
        t_l = QtWidgets.QVBoxLayout(test_tab)
        t_l.addWidget(QtWidgets.QLabel("TEST 图片（可加载测试/也可标 ROI）"))
        t_l.addWidget(self.test_list, 1)
        btns3 = QtWidgets.QHBoxLayout()
        self.btn_add_test = QtWidgets.QPushButton("加载 TEST…")
        self.btn_add_test.clicked.connect(lambda: self._add_images_to("TEST"))
        self.btn_del_test = QtWidgets.QPushButton("从 TEST 移除")
        self.btn_del_test.clicked.connect(lambda: self._remove_selected_from("TEST"))
        btns3.addWidget(self.btn_add_test)
        btns3.addWidget(self.btn_del_test)
        t_l.addLayout(btns3)
        self.tabs.addTab(test_tab, "TEST")

        # controls
        box = QtWidgets.QGroupBox("参数")
        form = QtWidgets.QFormLayout(box)
        self.cmb_algorithm = QtWidgets.QComboBox()
        self.cmb_algorithm.addItems(SUPPORTED_ALGORITHMS)
        self.cmb_algorithm.currentTextChanged.connect(self._on_algorithm_changed)
        self.cmb_backbone = self.cmb_algorithm
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(SUPPORTED_SCORE_MODES)
        self.cmb_mode.currentTextChanged.connect(self._on_runtime_params_changed)
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setDecimals(4)
        self.spin_margin.setSingleStep(0.005)
        self.spin_margin.setRange(-1.0, 1.0)
        self.spin_margin.setValue(0.02)
        self.spin_margin.valueChanged.connect(self._on_runtime_params_changed)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setRange(1, 50)
        self.spin_topk.setValue(3)
        self.spin_topk.valueChanged.connect(self._on_runtime_params_changed)
        form.addRow("算法", self.cmb_algorithm)
        form.addRow("判定方式", self.cmb_mode)
        form.addRow("Margin", self.spin_margin)
        form.addRow("TopK(仅topk)", self.spin_topk)
        left.addWidget(box, 0)

        act = QtWidgets.QHBoxLayout()
        self.btn_train = QtWidgets.QPushButton("训练/注册 (OK+NG)")
        self.btn_train.clicked.connect(self._train)
        self.btn_test = QtWidgets.QPushButton("测试 TEST")
        self.btn_test.clicked.connect(self._run_test)
        self.btn_export_test = QtWidgets.QPushButton("保存测试结果")
        self.btn_export_test.clicked.connect(self._export_current_results_csv)
        self.btn_validate_margin = QtWidgets.QPushButton("验证/建议Margin")
        self.btn_validate_margin.clicked.connect(self._run_margin_validation)
        self.btn_embedding_analysis = QtWidgets.QPushButton("特征分析")
        self.btn_embedding_analysis.clicked.connect(self._open_embedding_analysis_dialog)
        self.btn_baseline_debug = QtWidgets.QPushButton("传统基线调试")
        self.btn_baseline_debug.clicked.connect(self._run_traditional_baseline_debug)
        act.addWidget(self.btn_train)
        act.addWidget(self.btn_test)
        act.addWidget(self.btn_export_test)
        act.addWidget(self.btn_validate_margin)
        act.addWidget(self.btn_embedding_analysis)
        act.addWidget(self.btn_baseline_debug)
        left.addLayout(act)

        self.lbl_status = QtWidgets.QLabel("状态：未训练")
        left.addWidget(self.lbl_status)

        # right: image + results
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        self.canvas = RoiCanvas()
        self.canvas.setMinimumSize(640, 480)
        self.canvas.shapesChanged.connect(self._on_shapes_changed)
        right.addWidget(self.canvas, 2)

        roi_bar = QtWidgets.QHBoxLayout()
        self._manual_roi_bar = roi_bar
        roi_bar.addWidget(QtWidgets.QLabel("形状："))
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(SUPPORTED_SHAPES)
        self.cmb_shape.setCurrentText("rect")
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        roi_bar.addWidget(self.cmb_shape)

        roi_bar.addWidget(QtWidgets.QLabel("标注："))
        self.cmb_label = QtWidgets.QComboBox()
        self.cmb_label.addItems(["roi", "anchor", "anchor_mask"])
        self.cmb_label.setCurrentText("roi")
        self.cmb_label.currentTextChanged.connect(self._on_label_changed)
        roi_bar.addWidget(self.cmb_label)

        self.btn_save = QtWidgets.QPushButton("保存标注 -> labelme json")
        self.btn_save.clicked.connect(self._save_current_rect)
        self.btn_clear = QtWidgets.QPushButton("清空当前标注")
        self.btn_clear.clicked.connect(self._clear_current_rect)
        roi_bar.addWidget(self.btn_save)
        roi_bar.addWidget(self.btn_clear)
        roi_bar.addStretch()
        self.btn_clear_session = QtWidgets.QPushButton("清空会话")
        self.btn_clear_session.clicked.connect(self._clear_session)
        roi_bar.addWidget(self.btn_clear_session)
        right.addLayout(roi_bar)

        auto_box = QtWidgets.QGroupBox("自动 ROI")
        auto_l = QtWidgets.QGridLayout(auto_box)
        self._auto_roi_layout = auto_l
        self.lbl_ref = QtWidgets.QLabel("参考图：未设置")
        self.btn_set_ref = QtWidgets.QPushButton("设为参考图(当前)")
        self.btn_set_ref.clicked.connect(self._set_ref_from_current)
        self.btn_pick_ref = QtWidgets.QPushButton("选择参考图…")
        self.btn_pick_ref.clicked.connect(self._pick_ref_image)
        self.btn_build_shape = QtWidgets.QPushButton("生成模板")
        self.btn_build_shape.clicked.connect(self._build_shape_model)
        self.btn_edit_line2dup = QtWidgets.QPushButton("line2dup 模板页")
        self.btn_edit_line2dup.clicked.connect(self._open_line2dup_template_page)
        auto_l.addWidget(self.lbl_ref, 0, 0, 1, 3)
        auto_l.addWidget(self.btn_set_ref, 1, 0)
        auto_l.addWidget(self.btn_pick_ref, 1, 1)
        auto_l.addWidget(self.btn_build_shape, 1, 2)
        auto_l.addWidget(self.btn_edit_line2dup, 2, 2)

        auto_l.addWidget(QtWidgets.QLabel("定位方式："), 2, 0)
        self.cmb_loc = QtWidgets.QComboBox()
        self.cmb_loc.addItems(SUPPORTED_LOC_MODES)
        self.cmb_loc.setCurrentText(self.loc_method)
        self.cmb_loc.currentTextChanged.connect(self._on_loc_method_changed)
        auto_l.addWidget(self.cmb_loc, 2, 1)

        self.chk_only_missing = QtWidgets.QCheckBox("仅缺失ROI")
        self.chk_only_missing.setChecked(True)
        auto_l.addWidget(self.chk_only_missing, 3, 2)

        self.btn_autogen = QtWidgets.QPushButton("批量生成ROI(当前列表)")
        self.btn_autogen.clicked.connect(self._autogen_roi_current_tab)
        self.btn_autogen_all = QtWidgets.QPushButton("批量生成ROI(全部列表)")
        self.btn_autogen_all.clicked.connect(self._autogen_roi_all)
        self.btn_clear_roi_batch = QtWidgets.QPushButton("清空ROI(当前列表)")
        self.btn_clear_roi_batch.clicked.connect(self._clear_roi_current_tab)
        auto_l.addWidget(self.btn_autogen, 4, 0, 1, 2)
        auto_l.addWidget(self.btn_autogen_all, 4, 2)
        auto_l.addWidget(self.btn_clear_roi_batch, 5, 0, 1, 2)
        right.addWidget(auto_box)
        self._update_loc_ui()
        for index in [0, 1, 2, 3, 4, 5]:
            item = self._manual_roi_bar.itemAt(index)
            if item is not None and item.widget() is not None:
                item.widget().setVisible(False)
        for pos in [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]:
            item = self._auto_roi_layout.itemAtPosition(*pos)
            if item is not None and item.widget() is not None:
                item.widget().setVisible(False)



        self.table = QtWidgets.QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            ["文件", "GT", "Pred", "diff", "sim_ok", "sim_ng", "value", "threshold", "match_ms", "total_ms", "json"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_table_click)
        right.addWidget(self.table, 1)

    def _refresh_lists(self):
        def fill(listw: QtWidgets.QListWidget, files: List[str]):
            listw.clear()
            for p in files:
                it = QtWidgets.QListWidgetItem(os.path.basename(p))
                it.setToolTip(p)
                listw.addItem(it)

        fill(self.ok_list, self.ok_files)
        fill(self.ng_list, self.ng_files)
        fill(self.test_list, self.test_files)

    def _current_label(self) -> str:
        return self.cmb_label.currentText()

    def _update_save_label_text(self) -> None:
        label = self._current_label()
        self.btn_save.setText(f"保存标注({label}) -> labelme json")

    def _set_overlay_shapes(self, img_path: str, current_label: str) -> None:
        j = qr_core.labelme_json_of_image(img_path)
        overlays: List[OverlayShape] = []

        recipe = self.line2dup_recipe
        if recipe is None and os.path.exists(self._line2dup_recipe_path):
            try:
                recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
                self.line2dup_recipe = recipe
            except Exception:
                recipe = None
        if self.loc_method == "line2dup" and recipe is not None and recipe.search_points:
            points = [
                (float(pt[0]), float(pt[1]))
                for pt in (recipe.search_points or [])
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if len(points) >= 2:
                if str(recipe.search_shape_type or "rectangle") == "rectangle" and len(points) == 2:
                    (x0, y0), (x1, y1) = points[:2]
                    x = int(round(min(x0, x1)))
                    y = int(round(min(y0, y1)))
                    w = max(1, int(round(abs(x1 - x0))))
                    h = max(1, int(round(abs(y1 - y0))))
                    overlays.append(
                        OverlayShape(
                            shape_type="rect",
                            xywh=(x, y, w, h),
                            color=QtGui.QColor(0, 0, 255),
                            width=0.5,
                            dash=False,
                        )
                    )
                elif len(points) >= 3:
                    overlays.append(
                        OverlayShape(
                            shape_type="polygon",
                            points=points,
                            color=QtGui.QColor(0, 0, 255),
                            width=0.5,
                            dash=False,
                        )
                    )

        if not os.path.exists(j):
            self.canvas.set_overlays(overlays)
            return

        def add_shape(label: str, color: QtGui.QColor, *, width: int = 2, dash: bool = False) -> None:
            poly_pts = qr_core.try_read_polygon_points_from_labelme(j, label)
            if poly_pts and len(poly_pts) >= 3:
                overlays.append(OverlayShape(shape_type="polygon", points=poly_pts, color=color, width=width, dash=dash))
                return
            xywh = qr_core.try_read_xywh_from_labelme(j, label)
            if xywh:
                overlays.append(OverlayShape(shape_type="rect", xywh=xywh, color=color, width=width, dash=dash))

        seen_labels = set()
        for idx, label in enumerate(qr_core.sorted_label_names_from_labelme(j, label_prefix="roi")):
            if label == current_label:
                continue
            seen_labels.add(label)
            add_shape(label, ROI_OVERLAY_PALETTE[idx % len(ROI_OVERLAY_PALETTE)], width=2, dash=False)

        for label, color, dash in [
            ("anchor", QtGui.QColor(0, 255, 255), True),
            ("roi", QtGui.QColor(255, 165, 0), False),
            ("anchor_mask", QtGui.QColor(255, 0, 0), True),
        ]:
            if label == current_label or label in seen_labels:
                continue
            add_shape(label, color, width=2, dash=dash)

        self.canvas.set_overlays(overlays)

    def _load_shape_for_label(self, img_path: str, label_name: str) -> None:
        self.canvas.clear_roi()
        j = qr_core.labelme_json_of_image(img_path)
        loaded = False
        if os.path.exists(j):
            poly_pts = qr_core.try_read_polygon_points_from_labelme(j, label_name)
            if poly_pts and len(poly_pts) >= 3:
                self.canvas.set_roi_polygon(poly_pts)
                self.cmb_shape.setCurrentText("polygon")
                loaded = True
            xywh = qr_core.try_read_xywh_from_labelme(j, label_name)
            if xywh:
                self.canvas.set_roi_rect(xywh)
                self.cmb_shape.setCurrentText("rect")
                loaded = True
        self._set_overlay_shapes(img_path, label_name)
        if not loaded:
            self._on_shapes_changed()

    def _load_canvas_image(self, path: str) -> None:
        self.canvas.set_image(path, pixmap=_pixmap_from_path(path))
        self._load_shape_for_label(path, self._current_label())

    def _set_status_for_current_image(self, path: str) -> None:
        match_ms = self._line2dup_match_ms_by_image.get(path)
        total_ms = self._line2dup_autogen_ms_by_image.get(path)
        if match_ms is None and total_ms is None:
            return
        parts = [f"当前图：{os.path.basename(path)}"]
        if match_ms is not None:
            parts.append(f"模板匹配={match_ms:.1f}ms")
        if total_ms is not None:
            parts.append(f"生成ROI={total_ms:.1f}ms")
        self.lbl_status.setText("状态：" + "  ".join(parts))

    def _current_selected_path(self) -> Optional[str]:
        tab = self.tabs.currentIndex()
        if tab == 0:
            items = self.ok_list.selectedItems()
            if not items:
                return None
            row = self.ok_list.row(items[0])
            if row >= len(self.ok_files):  # 防止列表为空时越界
                return None
            return self.ok_files[row]
        if tab == 1:
            items = self.ng_list.selectedItems()
            if not items:
                return None
            row = self.ng_list.row(items[0])
            if row >= len(self.ng_files):  # 防止列表为空时越界
                return None
            return self.ng_files[row]
        items = self.test_list.selectedItems()
        if not items:
            return None
        row = self.test_list.row(items[0])
        if row >= len(self.test_files):  # 防止列表为空时越界
            return None
        return self.test_files[row]

    def _on_select_ok(self):
        p = self._current_selected_path()
        if p:
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    def _on_select_ng(self):
        p = self._current_selected_path()
        if p:
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    def _on_select_test(self):
        p = self._current_selected_path()
        if p:
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    def _add_images_to(self, kind: str):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            f"选择要加入 {kind} 的图片",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return

        if kind == "OK":
            self.ok_files.extend(files)
            self.ok_files = sorted(list(dict.fromkeys(self.ok_files)))
        elif kind == "NG":
            self.ng_files.extend(files)
            self.ng_files = sorted(list(dict.fromkeys(self.ng_files)))
        else:
            self.test_files.extend(files)
            self.test_files = sorted(list(dict.fromkeys(self.test_files)))
        self._refresh_lists()
        self._save_session()

    def _remove_selected_from(self, kind: str):
        if kind == "OK":
            items = self.ok_list.selectedItems()
            if not items:
                return
            idx = self.ok_list.row(items[0])
            self.ok_files.pop(idx)
        elif kind == "NG":
            items = self.ng_list.selectedItems()
            if not items:
                return
            idx = self.ng_list.row(items[0])
            self.ng_files.pop(idx)
        else:
            items = self.test_list.selectedItems()
            if not items:
                return
            idx = self.test_list.row(items[0])
            self.test_files.pop(idx)
        self._refresh_lists()
        self._save_session()

    def _on_tab_changed(self, index: int):
        """Tab切换时自动选择第一项并显示"""
        if index == 0 and self.ok_files:
            self.ok_list.setCurrentRow(0)
            # 强制加载图片，即使已经选中第0行
            self._load_canvas_image(self.ok_files[0])
        elif index == 1 and self.ng_files:
            self.ng_list.setCurrentRow(0)
            # 强制加载图片，即使已经选中第0行
            self._load_canvas_image(self.ng_files[0])
        elif index == 2 and self.test_files:
            self.test_list.setCurrentRow(0)
            # 强制加载图片，即使已经选中第0行
            self._load_canvas_image(self.test_files[0])

    def _on_shape_changed(self):
        if self._current_label() == "anchor_mask" and self.cmb_shape.currentText() != "polygon":
            # anchor_mask 只允许 polygon
            self.cmb_shape.setCurrentText("polygon")
            return
        self.canvas.draw_shape = self.cmb_shape.currentText()
        # 清空当前正在画的 polygon 点，避免混乱
        self.canvas._poly_pts = []
        self.canvas.update()
        self._on_shapes_changed()

    def _on_label_changed(self):
        label = self._current_label()
        if label == "anchor_mask":
            self.cmb_shape.setCurrentText("polygon")
            self.cmb_shape.setEnabled(False)
        else:
            self.cmb_shape.setEnabled(True)
        self._update_save_label_text()
        p = self.canvas.image_path()
        if p:
            self._load_shape_for_label(p, label)

    def _clear_current_rect(self):
        # 清空正在画的 polygon 点
        self.canvas.clear_roi()

        # 如果已经保存过 json，则同步从 json 删除该标注（否则切图会被读回来，看起来像“清不掉”）
        p = self.canvas.image_path()
        if p is not None:
            try:
                deleted = qr_core.delete_labelme_shape(p, label_name=self._current_label())
            except Exception:
                deleted = False
            if deleted:
                # 重新加载，确保 UI 与 json 一致
                self._load_canvas_image(p)
                return

        self.canvas.update()
        self._on_shapes_changed()

    def _on_shapes_changed(self):
        p = self.canvas.image_path()
        if p is None:
            self.btn_save.setEnabled(False)
            return
        st = self.canvas.roi
        ok = (st.shape_type == "rect" and st.xywh is not None) or (st.shape_type == "polygon" and st.points is not None)
        self.btn_save.setEnabled(ok)

    def _roi_xywh_from_canvas(self) -> Optional[Tuple[int, int, int, int]]:
        """Extract ROI bounding box from canvas (needed for testing)"""
        roi = self.canvas.roi_xywh()
        if roi is not None:
            return roi
        # fallback: try load from json (roi label)
        p = self.canvas.image_path()
        if p:
            j = qr_core.labelme_json_of_image(p)
            if os.path.exists(j):
                xywh = qr_core.try_read_xywh_from_labelme(j, "roi")
                if xywh:
                    return xywh
        return None


    def _save_current_rect(self):
        p = self.canvas.image_path()
        if p is None:
            return
        st = self.canvas.roi
        label_name = self._current_label()

        if label_name == "anchor_mask" and st.shape_type != "polygon":
            QtWidgets.QMessageBox.warning(self, "提示", "anchor_mask 只能用多边形标注（polygon）")
            return

        if st.shape_type == "rect":
            if st.xywh is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先拖拽画出矩形标注")
                return
            jpath = qr_core.upsert_labelme_rect(p, st.xywh, label_name=label_name)
        else:
            if not st.points or len(st.points) < 3:
                QtWidgets.QMessageBox.warning(self, "提示", "多边形至少需要 3 个点（左键点选加点，右键结束）")
                return
            jpath = qr_core.upsert_labelme_polygon(p, st.points, label_name=label_name)

        QtWidgets.QMessageBox.information(
            self,
            "已保存",
            f"已更新 labelme json：\n{jpath}\n(label={label_name}, type={st.shape_type})",
        )
        # reload to ensure consistent
        self._load_canvas_image(p)

    def _set_reference(self, path: str) -> None:
        self.ref_image = path
        if self.lbl_ref is not None:
            self.lbl_ref.setText(f"参考图：{os.path.basename(path)}")
            self.lbl_ref.setToolTip(path)
        try:
            recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
            recipe.reference_image = path
            recipe.model_path = self._line2dup_model_path
            line2dup_locator.save_recipe_for_product(self._product_dir, recipe)
            self.line2dup_recipe = recipe
        except Exception:
            pass
        self._save_session()

    def _set_ref_from_current(self):
        p = self.canvas.image_path()
        if not p:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在右侧打开一张图片")
            return
        self._set_reference(p)

    def _pick_ref_image(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择参考图",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not p:
            return
        self._set_reference(p)

    def _open_line2dup_template_page(self):
        initial = self.ref_image or self.canvas.image_path() or ""
        dlg = Line2DupTemplateDialog(
            product_name=self.current_product,
            product_dir=self._product_dir,
            initial_image_path=initial,
            parent=self,
        )
        dlg.modelSaved.connect(self._on_line2dup_model_saved)
        dlg.exec()

    def _on_line2dup_model_saved(self, model_path: str, recipe_path: str) -> None:
        try:
            self.line2dup_recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
        except Exception:
            self.line2dup_recipe = None
        msg = f"状态：line2dup 模型已保存 {os.path.basename(model_path)}"
        self.lbl_status.setText(msg)

    def _update_loc_ui(self) -> None:
        method = self.loc_method
        if hasattr(self, "btn_build_shape"):
            self.btn_build_shape.setVisible(method == "shape_model")
        if hasattr(self, "btn_edit_line2dup"):
            self.btn_edit_line2dup.setVisible(method == "line2dup")

    def _build_shape_model(self):
        if not self.ref_image or not os.path.exists(self.ref_image):
            QtWidgets.QMessageBox.warning(self, "提示", "请先设置参考图")
            return
        ref_json = qr_core.labelme_json_of_image(self.ref_image)
        if not os.path.exists(ref_json):
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少标注 json（需要 anchor + roi）")
            return
        if qr_core.try_read_xywh_from_labelme(ref_json, "anchor") is None:
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 anchor 标注")
            return
        if qr_core.try_read_xywh_from_labelme(ref_json, "roi") is None:
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 roi 标注")
            return
        try:
            model_path = qr_core.create_shape_model_from_reference(
                ref_img_path=self.ref_image,
                model_path=self._shape_model_path,
                anchor_label="anchor",
                anchor_mask_label="anchor_mask",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "生成模板失败", str(e))
            return
        QtWidgets.QMessageBox.information(self, "生成模板完成", f"已保存模板：\n{model_path}")

    def _on_loc_method_changed(self, method: str):
        self.loc_method = method
        self._update_loc_ui()
        self._save_session()

    def _line2dup_output_labels(self) -> List[str]:
        recipe = self.line2dup_recipe
        if recipe is None and os.path.exists(self._line2dup_recipe_path):
            try:
                recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
                self.line2dup_recipe = recipe
            except Exception:
                recipe = None
        if recipe is not None and recipe.reference_regions:
            labels = [
                str(region.get("output_label") or region.get("reference_label") or "")
                for region in (recipe.reference_regions or [])
                if isinstance(region, dict)
            ]
            labels = [label for label in labels if label]
            if labels:
                return labels
        return ["roi"]

    def _missing_roi_files(self, paths: List[str]) -> List[str]:
        missing: List[str] = []
        labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        for p in paths:
            j = qr_core.labelme_json_of_image(p)
            if not os.path.exists(j):
                missing.append(p)
                continue
            if any(qr_core.read_shape_from_labelme(j, label) is None for label in labels):
                missing.append(p)
        return missing

    def _resolve_autogen_targets(
        self,
        paths: List[str],
        *,
        only_missing: bool,
        silent: bool,
    ) -> List[str]:
        self._skip_empty_autogen_message = False
        if not paths:
            return []
        missing = self._missing_roi_files(paths)
        if not missing:
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "这些图片已经存在 ROI。")
                self._skip_empty_autogen_message = True
            return []

        missing_set = set(missing)
        existing = [p for p in paths if p not in missing_set]
        if not existing or silent:
            return list(missing) if only_missing else list(paths)

        default_button = (
            QtWidgets.QMessageBox.StandardButton.No
            if only_missing
            else QtWidgets.QMessageBox.StandardButton.Yes
        )
        reply = QtWidgets.QMessageBox.question(
            self,
            "覆盖已存在ROI？",
            (
                f"当前列表中已有 ROI 的图片有 {len(existing)} 张。\n"
                "是否覆盖并重新创建这些 ROI？\n\n"
                "选择“是”将重建整个列表；选择“否”只创建缺失 ROI；选择“取消”终止。"
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            default_button,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
            self._skip_empty_autogen_message = True
            return []
        if reply == QtWidgets.QMessageBox.StandardButton.No:
            return list(missing)
        return list(paths)

    def _autogen_roi_for_images(self, paths: List[str], only_missing: bool, silent: bool = False) -> None:
        if not paths:
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "没有可处理的图片")
            return
        ref_image = self.ref_image
        method = self.loc_method
        if method == "line2dup":
            try:
                recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
                self.line2dup_recipe = recipe
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "提示", f"无法加载 line2dup recipe：{exc}")
                return
            if recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
                if self.ref_image != ref_image:
                    self._set_reference(ref_image)
            if not os.path.exists(self._line2dup_model_path):
                QtWidgets.QMessageBox.warning(self, "提示", "当前产品还没有 line2dup 模型，请先创建模板。")
                return
            labels = self._line2dup_output_labels()
            recipe_region_labels = {
                str(region.get("output_label") or region.get("reference_label") or "").strip()
                for region in (recipe.reference_regions or [])
                if isinstance(region, dict)
            }
            recipe_region_labels.discard("")
            if (not ref_image or not os.path.exists(ref_image)) and not recipe_region_labels:
                QtWidgets.QMessageBox.warning(self, "提示", "line2dup 需要参考图或已保存的参考 ROI。")
                return
            missing_labels: List[str] = []
            if labels:
                missing_labels = [label for label in labels if label not in recipe_region_labels]
                if missing_labels:
                    ref_json = qr_core.labelme_json_of_image(ref_image) if ref_image else ""
                    if not ref_json or not os.path.exists(ref_json):
                        QtWidgets.QMessageBox.warning(
                            self,
                            "提示",
                            f"参考图缺少 labelme json，且 recipe 中也没有这些参考ROI：{', '.join(missing_labels)}",
                        )
                        return
                    missing_labels = [
                        label
                        for label in missing_labels
                        if qr_core.read_shape_from_labelme(ref_json, label) is None
                    ]
                    if missing_labels:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "提示",
                            f"参考图缺少参考ROI标注：{', '.join(missing_labels)}",
                        )
                        return
        else:
            if not ref_image or not os.path.exists(ref_image):
                QtWidgets.QMessageBox.warning(self, "提示", "请先设置参考图")
                return

            ref_json = qr_core.labelme_json_of_image(ref_image)
            if not os.path.exists(ref_json):
                QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少标注 json（需要 anchor + roi）")
                return
            if qr_core.try_read_xywh_from_labelme(ref_json, "anchor") is None:
                QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 anchor 标注")
                return
            if qr_core.try_read_xywh_from_labelme(ref_json, "roi") is None:
                QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 roi 标注")
                return

        todo = self._resolve_autogen_targets(paths, only_missing=only_missing, silent=silent)
        if not todo:
            if getattr(self, "_skip_empty_autogen_message", False):
                self._skip_empty_autogen_message = False
                return
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "这些图片已存在 ROI")
            return

        ok = 0
        errs: List[str] = []
        for p in todo:
            try:
                if method == "shape_model":
                    qr_core.autogen_roi_json_from_shape_model(
                        tgt_img_path=p,
                        ref_img_path=ref_image,
                        model_path=self._shape_model_path,
                        anchor_label="anchor",
                        roi_label="roi",
                        anchor_mask_label="anchor_mask",
                    )
                elif method == "line2dup":
                    run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                        tgt_img_path=p,
                        ref_img_path=ref_image,
                        product_dir=self._product_dir,
                    )
                    self._line2dup_match_ms_by_image[p] = float(run.locate_ms)
                    self._line2dup_autogen_ms_by_image[p] = float(run.total_ms)
                else:
                    qr_core.autogen_roi_json_from_reference(
                        tgt_img_path=p,
                        ref_img_path=ref_image,
                        method=method,
                        anchor_label="anchor",
                        roi_label="roi",
                    )
                ok += 1
            except Exception as e:
                errs.append(f"{os.path.basename(p)}: {e}")

        if not silent:
            msg = f"自动 ROI 完成：成功 {ok} / 失败 {len(errs)}"
            if errs:
                msg += "\n\n失败示例（前10）：\n" + "\n".join(errs[:10])
            QtWidgets.QMessageBox.information(self, "完成", msg)
            if ok:
                self.lbl_status.setText(f"状态：当前列表已生成ROI，成功 {ok} 张，失败 {len(errs)} 张")

        cur = self.canvas.image_path()
        if cur and cur in todo:
            self._load_canvas_image(cur)
            self._set_status_for_current_image(cur)

    def _autogen_roi_current_tab(self):
        tab = self.tabs.currentIndex()
        if tab == 0:
            paths = list(self.ok_files)
        elif tab == 1:
            paths = list(self.ng_files)
        else:
            paths = list(self.test_files)
        self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

    def _autogen_roi_all(self):
        paths = list(self.ok_files) + list(self.ng_files) + list(self.test_files)
        self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

    def _clear_roi_for_images(self, paths: List[str], silent: bool = False) -> None:
        if not paths:
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "没有可处理的图片")
            return
        labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        labels = [label for label in labels if label]
        if not labels:
            labels = ["roi"]

        removed = 0
        touched = 0
        for path in paths:
            any_removed = False
            for label in labels:
                try:
                    if qr_core.delete_labelme_shape(path, label):
                        removed += 1
                        any_removed = True
                except Exception:
                    pass
            if any_removed:
                touched += 1
                self._line2dup_match_ms_by_image.pop(path, None)
                self._line2dup_autogen_ms_by_image.pop(path, None)

        cur = self.canvas.image_path()
        if cur and cur in paths:
            self._load_canvas_image(cur)
            self._set_status_for_current_image(cur)

        if not silent:
            QtWidgets.QMessageBox.information(
                self,
                "完成",
                f"已清空 ROI：{touched} 张图片，删除 {removed} 个标签。\n标签: {', '.join(labels)}",
            )
            self.lbl_status.setText(f"状态：已清空 ROI，图片 {touched} 张，标签 {removed} 个")

    def _clear_roi_current_tab(self):
        tab = self.tabs.currentIndex()
        if tab == 0:
            paths = list(self.ok_files)
            tab_name = "OK"
        elif tab == 1:
            paths = list(self.ng_files)
            tab_name = "NG"
        else:
            paths = list(self.test_files)
            tab_name = "TEST"

        if not paths:
            QtWidgets.QMessageBox.information(self, "提示", "当前列表没有图片")
            return

        labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        reply = QtWidgets.QMessageBox.question(
            self,
            "清空ROI",
            f"确定清空当前 {tab_name} 列表中的 ROI 吗？\n将删除标签: {', '.join(labels)}",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._clear_roi_for_images(paths, silent=False)

    def _load_products(self) -> dict:
        """加载产品配置，如果不存在则创建默认"""
        if os.path.exists(self._products_json):
            try:
                with open(self._products_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        # 默认配置
        return {
            "products": ["Default"],
            "current_product": "Default"
        }

    def _update_product_paths(self):
        """根据当前产品更新路径"""
        product_dir = os.path.join(self._session_dir, self.current_product)
        os.makedirs(product_dir, exist_ok=True)
        self._product_dir = product_dir
        self._session_json = os.path.join(product_dir, "session.json")
        self._product_params_path = os.path.join(product_dir, "product_params.json")
        self._shape_model_path = os.path.join(product_dir, "shape_model.npz")
        paths = line2dup_locator.product_paths(product_dir)
        self._line2dup_model_path = paths.model_path
        self._line2dup_recipe_path = paths.recipe_path

    def _save_products(self):
        """保存产品配置"""
        os.makedirs(self._session_dir, exist_ok=True)
        with open(self._products_json, "w", encoding="utf-8") as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)

    def _on_product_changed(self, product_name: str):
        """切换产品"""
        if not product_name or product_name == self.current_product:
            return
        
        # 保存当前产品的会话
        self._save_session()
        
        # 切换产品
        self.current_product = product_name
        self._update_product_paths()
        
        # 更新products.json
        self.products["current_product"] = product_name
        self._save_products()
        
        # 清空当前状态
        self.model = None
        self.line2dup_recipe = None
        self.ref_image = None
        self._line2dup_match_ms_by_image = {}
        self._line2dup_autogen_ms_by_image = {}
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self.table.setRowCount(0)
        self._current_result_rows = []
        self.canvas.clear_image()
        self.lbl_ref.setText("参考图：未设置")
        self.lbl_ref.setToolTip("")
        self.lbl_status.setText("状态：已切换产品")
        
        # 加载新产品的会话
        self._try_load_session_on_start()
        self._refresh_lists()

    def _new_product(self):
        """创建新产品"""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "新建产品", "请输入产品名称:"
        )
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # 检查重名
        if name in self.products["products"]:
            QtWidgets.QMessageBox.warning(self, "错误", "产品名称已存在")
            return
        
        # 验证名称（只允许字母、数字、下划线和中文）
        import re
        if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$', name):
            QtWidgets.QMessageBox.warning(
                self, "错误", 
                "产品名称只能包含字母、数字、下划线和中文字符"
            )
            return
        
        # 添加产品
        self.products["products"].append(name)
        self._save_products()
        self.cmb_product.addItem(name)
        self.cmb_product.setCurrentText(name)
        # 切换逻辑会自动触发

    def _current_algorithm(self) -> str:
        return str(self.cmb_algorithm.currentText() or "").strip()

    def _is_embedding_algorithm(self, algorithm: Optional[str] = None) -> bool:
        return not is_traditional_algorithm(algorithm or self._current_algorithm())

    def _embedding_model_path(self, algorithm: str) -> str:
        return os.path.join(self._product_dir, f"register_model_{algorithm}.npz")

    def _load_runtime_params(self) -> None:
        self.product_params = load_product_params(self._product_params_path)
        algorithm = str(self.product_params.algorithm or "")
        if algorithm not in SUPPORTED_ALGORITHMS:
            self.product_params.algorithm = SUPPORTED_EMBEDDING_ALGORITHMS[0]
        if str(self.product_params.score_mode or "") not in SUPPORTED_SCORE_MODES:
            self.product_params.score_mode = SUPPORTED_SCORE_MODES[0]
        self.product_params.topk = max(1, int(self.product_params.topk))
        self.product_params.margin = float(self.product_params.margin)

    def _save_runtime_params(self) -> None:
        save_product_params(self.product_params, self._product_params_path)

    def _apply_runtime_params_to_ui(self) -> None:
        self._updating_runtime_params = True
        try:
            algorithm = self.product_params.algorithm if self.product_params.algorithm in SUPPORTED_ALGORITHMS else SUPPORTED_ALGORITHMS[0]
            score_mode = self.product_params.score_mode if self.product_params.score_mode in SUPPORTED_SCORE_MODES else SUPPORTED_SCORE_MODES[0]
            self.cmb_algorithm.setCurrentText(algorithm)
            self.cmb_mode.setCurrentText(score_mode)
            self.spin_margin.setValue(float(self.product_params.margin))
            self.spin_topk.setValue(max(1, int(self.product_params.topk)))
        finally:
            self._updating_runtime_params = False
        self._update_runtime_widgets()

    def _update_runtime_widgets(self) -> None:
        embedding = self._is_embedding_algorithm()
        self.cmb_mode.setEnabled(embedding)
        self.spin_topk.setEnabled(embedding and self.cmb_mode.currentText() == "topk")
        self.btn_validate_margin.setEnabled(embedding)
        self.btn_embedding_analysis.setEnabled(embedding)

    def _on_runtime_params_changed(self, *args) -> None:
        if self._updating_runtime_params:
            return
        self.product_params.algorithm = self._current_algorithm()
        self.product_params.score_mode = self.cmb_mode.currentText()
        self.product_params.margin = float(self.spin_margin.value())
        self.product_params.topk = int(self.spin_topk.value())
        if self.model is not None and self._is_embedding_algorithm():
            self.model.score_mode = self.product_params.score_mode
            self.model.margin = self.product_params.margin
            self.model.topk = self.product_params.topk
        self._save_runtime_params()
        self._update_runtime_widgets()

    def _load_embedding_model_for_algorithm(self, algorithm: str) -> None:
        if not self._is_embedding_algorithm(algorithm):
            self.model = None
            self.lbl_status.setText(f"状态：当前算法={algorithm}，使用传统阈值方法")
            return

        model_file = self._embedding_model_path(algorithm)
        if not os.path.exists(model_file):
            self.model = None
            self.lbl_status.setText(f"状态：{algorithm} 模型未训练")
            return

        self.model = qr_core.load_register_model_npz(model_file)
        self.model.score_mode = self.product_params.score_mode
        self.model.margin = float(self.product_params.margin)
        self.model.topk = int(self.product_params.topk)
        self.lbl_status.setText(
            f"状态：已加载模型  algorithm={algorithm}  mode={self.model.score_mode}  "
            f"margin={self.model.margin:.4f}  topk={self.model.topk}"
        )

    def _on_algorithm_changed(self, algorithm: str):
        if self._updating_runtime_params:
            return
        if not algorithm:
            return
        self.product_params.algorithm = algorithm
        self._save_runtime_params()
        self._update_runtime_widgets()
        try:
            self._load_embedding_model_for_algorithm(algorithm)
        except Exception as exc:
            self.model = None
            self.lbl_status.setText(f"状态：加载算法 {algorithm} 失败 - {exc}")

    def _train(self):
        self.model = None
        self.table.setRowCount(0)
        self._current_result_rows = []
        label_names = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        candidate_paths = list(self.ok_files) + list(self.ng_files)
        missing_paths = self._missing_roi_files(candidate_paths)
        if missing_paths and self.loc_method == "line2dup":
            try:
                self._autogen_roi_for_images(missing_paths, only_missing=False, silent=True)
                missing_paths = self._missing_roi_files(candidate_paths)
            except Exception:
                pass
        missing = [os.path.basename(p) for p in missing_paths]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "缺少ROI标注",
                f"需要每张 OK/NG 图都具备这些 ROI：{', '.join(label_names)}。\n请逐张打开图片 -> 画 ROI -> 保存。\n缺少：\n"
                + "\n".join(missing[:50]),
            )
            return

        algorithm = self._current_algorithm()
        mode = self.cmb_mode.currentText()
        margin = float(self.spin_margin.value())
        topk = int(self.spin_topk.value())
        self.product_params.algorithm = algorithm
        self.product_params.score_mode = mode
        self.product_params.margin = margin
        self.product_params.topk = topk

        try:
            if self._is_embedding_algorithm(algorithm):
                self.model = qr_core.train_register_model(
                    self.ok_files,
                    self.ng_files,
                    backbone=algorithm,
                    score_mode=mode,
                    margin=margin,
                    topk=topk,
                    label_name=label_names[0],
                    label_names=label_names,
                )
                qr_core.save_register_model_npz(self.model, self._embedding_model_path(algorithm))
                self.lbl_status.setText(
                    f"状态：已训练  algorithm={algorithm}  mode={mode}  margin={margin:.4f}  topk={topk}"
                )
                message = "OK/NG 注册完成，可以开始测试。"
            else:
                threshold_model, train_rows = train_threshold_model(
                    self.ok_files,
                    self.ng_files,
                    algorithm,
                    preferred_label=label_names[0],
                )
                self.product_params.traditional_models[algorithm] = threshold_model.to_dict()
                rows: List[Dict[str, object]] = []
                for sample in train_rows:
                    pred, diff = threshold_model.predict(float(sample["value"]))
                    rows.append(
                        {
                            "file_path": str(sample.get("file_path", "")),
                            "file_name": str(sample.get("file_name", "")),
                            "gt": str(sample.get("gt", "")),
                            "pred": pred,
                            "diff": float(diff),
                            "sim_ok": None,
                            "sim_ng": None,
                            "value": float(sample["value"]),
                            "threshold": float(threshold_model.threshold),
                            "match_ms": None,
                            "total_ms": None,
                            "json_name": os.path.basename(qr_core.labelme_json_of_image(str(sample.get("file_path", "")))),
                        }
                    )
                self._populate_results_table(rows)
                self.lbl_status.setText(
                    f"状态：已训练传统算法  algorithm={algorithm}  threshold={threshold_model.threshold:.4f}  "
                    f"ok_when={threshold_model.ok_when}  acc={threshold_model.accuracy:.4f}"
                )
                message = "传统算法阈值模型训练完成，可以开始测试。"
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "训练失败", str(e))
            return

        self._save_runtime_params()
        self._save_session()
        self._update_runtime_widgets()
        QtWidgets.QMessageBox.information(self, "训练完成", message)

    def _predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        prefer_canvas_roi: bool = False,
    ) -> Dict[str, object]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        algorithm = self._current_algorithm()
        total_t0 = time.perf_counter()
        match_ms: Optional[float] = None
        if self.loc_method == "line2dup":
            recipe = self.line2dup_recipe
            if recipe is None and os.path.exists(self._line2dup_recipe_path):
                recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
                self.line2dup_recipe = recipe
            ref_image = self.ref_image
            if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
            if ref_image and os.path.exists(ref_image):
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self._product_dir,
                )
                match_ms = float(run.locate_ms)
                self._line2dup_match_ms_by_image[path] = match_ms
                self._line2dup_autogen_ms_by_image[path] = float(run.total_ms)
        elif self.ref_image and os.path.exists(self.ref_image):
            self._autogen_roi_for_images([path], only_missing=True, silent=True)

        labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        roi = None
        if prefer_canvas_roi and len(labels) == 1 and self.canvas.image_path() == path:
            roi = self._roi_xywh_from_canvas()

        if self._is_embedding_algorithm(algorithm):
            if self.model is None or self.model.backbone != algorithm:
                self._load_embedding_model_for_algorithm(algorithm)
            if self.model is None:
                raise RuntimeError(f"algorithm model not loaded: {algorithm}")
            self.model.score_mode = self.product_params.score_mode
            self.model.margin = float(self.product_params.margin)
            self.model.topk = int(self.product_params.topk)

            if len(labels) == 1 and roi is None:
                j = qr_core.labelme_json_of_image(path)
                if not os.path.exists(j):
                    raise FileNotFoundError(f"缺少 labelme json: {j}")

            if feat_net is None:
                feat_net, _ = qr_core.load_backbone(self.model.backbone, device=self.model.device)
            if len(labels) > 1:
                e = qr_core.embed_many(path, feat_net, labels, device=self.model.device)
            else:
                e = qr_core.embed_one(
                    path,
                    feat_net,
                    label_name=labels[0],
                    device=self.model.device,
                    roi_xywh=roi,
                )
            pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(e, self.model)
            value: Optional[float] = None
            threshold: Optional[float] = None
        else:
            model_dict = self.product_params.traditional_models.get(algorithm)
            if not isinstance(model_dict, dict):
                raise RuntimeError(f"传统算法 {algorithm} 尚未训练")
            threshold_model = TraditionalThresholdModel.from_dict(model_dict)
            metrics = compute_roi_metrics(path, preferred_label=threshold_model.roi_label or labels[0])
            value = metric_value(metrics, algorithm)
            pred, diff = threshold_model.predict(value)
            sim_ok = None
            sim_ng = None
            threshold = float(threshold_model.threshold)

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        return {
            "file_path": path,
            "file_name": os.path.basename(path),
            "gt": "",
            "pred": pred,
            "diff": float(diff),
            "sim_ok": float(sim_ok) if sim_ok is not None else None,
            "sim_ng": float(sim_ng) if sim_ng is not None else None,
            "value": float(value) if value is not None else None,
            "threshold": float(threshold) if threshold is not None else None,
            "match_ms": match_ms,
            "total_ms": float(total_ms),
            "json_name": os.path.basename(qr_core.labelme_json_of_image(path)),
        }

    def _populate_results_table(self, rows: List[Dict[str, object]]) -> None:
        self._current_result_rows = list(rows)
        self.table.setRowCount(0)
        for row_idx, row in enumerate(rows):
            self.table.insertRow(row_idx)
            values = [
                str(row.get("file_name", "")),
                str(row.get("gt", "")),
                str(row.get("pred", "")),
                f"{float(row.get('diff', 0.0)):.4f}" if row.get("diff") is not None else "",
                f"{float(row.get('sim_ok', 0.0)):.4f}" if row.get("sim_ok") is not None else "",
                f"{float(row.get('sim_ng', 0.0)):.4f}" if row.get("sim_ng") is not None else "",
                f"{float(row.get('value', 0.0)):.4f}" if row.get("value") is not None else "",
                f"{float(row.get('threshold', 0.0)):.4f}" if row.get("threshold") is not None else "",
                f"{float(row.get('match_ms', 0.0)):.1f}" if row.get("match_ms") is not None else "",
                f"{float(row.get('total_ms', 0.0)):.1f}" if row.get("total_ms") is not None else "",
                str(row.get("json_name", "")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.UserRole, str(row.get("file_path", "")))
                gt = str(row.get("gt", ""))
                pred = str(row.get("pred", ""))
                if gt and pred and gt != pred:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(192, 32, 32)))
                self.table.setItem(row_idx, col_idx, item)

    def _suggest_margin_from_rows(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        labeled = [row for row in rows if str(row.get("gt", "")) in {"OK", "NG"} and row.get("diff") is not None]
        if not labeled:
            raise RuntimeError("no labeled rows for margin suggestion")

        current_margin = float(self.spin_margin.value())
        diffs = sorted({float(row["diff"]) for row in labeled})
        candidates: List[float] = []
        if diffs:
            candidates.append(diffs[0] - 1e-6)
            candidates.extend(diffs)
            candidates.extend((a + b) * 0.5 for a, b in zip(diffs, diffs[1:]))
            candidates.append(diffs[-1] + 1e-6)

        def _accuracy(threshold: float) -> Tuple[float, int, int, int, int]:
            tp = tn = fp = fn = 0
            for row in labeled:
                gt = str(row["gt"])
                pred = "OK" if float(row["diff"]) >= threshold else "NG"
                if gt == "OK" and pred == "OK":
                    tp += 1
                elif gt == "OK" and pred == "NG":
                    fn += 1
                elif gt == "NG" and pred == "NG":
                    tn += 1
                else:
                    fp += 1
            acc = float(tp + tn) / float(len(labeled))
            return acc, tp, tn, fp, fn

        best_margin = current_margin
        best_acc = -1.0
        best_conf = (0, 0, 0, 0)
        for candidate in candidates:
            acc, tp, tn, fp, fn = _accuracy(candidate)
            if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(candidate - current_margin) < abs(best_margin - current_margin)):
                best_acc = acc
                best_margin = float(candidate)
                best_conf = (tp, tn, fp, fn)

        current_acc, current_tp, current_tn, current_fp, current_fn = _accuracy(current_margin)
        ok_diffs = [float(row["diff"]) for row in labeled if str(row["gt"]) == "OK"]
        ng_diffs = [float(row["diff"]) for row in labeled if str(row["gt"]) == "NG"]
        safe_range = None
        if ok_diffs and ng_diffs:
            lower = max(ng_diffs)
            upper = min(ok_diffs)
            if lower < upper:
                safe_range = (float(lower), float(upper))
                best_margin = float((lower + upper) * 0.5)
                best_acc, *conf = _accuracy(best_margin)
                best_conf = tuple(conf)

        return {
            "current_margin": current_margin,
            "current_accuracy": float(current_acc),
            "current_confusion": {
                "tp_ok": current_tp,
                "tn_ng": current_tn,
                "fp_ok_as_ng": current_fn,
                "fp_ng_as_ok": current_fp,
            },
            "suggested_margin": float(best_margin),
            "suggested_accuracy": float(best_acc),
            "suggested_confusion": {
                "tp_ok": best_conf[0],
                "tn_ng": best_conf[1],
                "fp_ng_as_ok": best_conf[2],
                "fp_ok_as_ng": best_conf[3],
            },
            "ok_diff_min": float(min(ok_diffs)) if ok_diffs else None,
            "ok_diff_max": float(max(ok_diffs)) if ok_diffs else None,
            "ng_diff_min": float(min(ng_diffs)) if ng_diffs else None,
            "ng_diff_max": float(max(ng_diffs)) if ng_diffs else None,
            "safe_range": safe_range,
        }

    def _save_margin_report(
        self,
        rows: List[Dict[str, object]],
        summary: Dict[str, object],
    ) -> Tuple[str, str]:
        report_dir = os.path.join(self._product_dir, "margin_reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"margin_report_{self._current_algorithm()}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {
            "product": self.current_product,
            "algorithm": self._current_algorithm(),
            "score_mode": self.cmb_mode.currentText(),
            "topk": int(self.spin_topk.value()),
            "margin": float(self.spin_margin.value()),
            "loc_method": self.loc_method,
            "summary": summary,
            "rows": rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["file", "gt", "pred", "diff", "sim_ok", "sim_ng", "value", "threshold", "match_ms", "total_ms", "json"]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.get("file_name", ""),
                        row.get("gt", ""),
                        row.get("pred", ""),
                        row.get("diff", ""),
                        row.get("sim_ok", ""),
                        row.get("sim_ng", ""),
                        row.get("value", ""),
                        row.get("threshold", ""),
                        row.get("match_ms", ""),
                        row.get("total_ms", ""),
                        row.get("json_name", ""),
                    ]
                )
        return json_path, csv_path

    def _current_tab_paths_and_name(self) -> Tuple[List[str], str]:
        tab = self.tabs.currentIndex()
        if tab == 0:
            return list(self.ok_files), "OK"
        if tab == 1:
            return list(self.ng_files), "NG"
        return list(self.test_files), "TEST"

    def _load_roi_mask_crop(self, img_path: str, preferred_label: str = "roi1") -> Dict[str, object]:
        jpath = qr_core.labelme_json_of_image(img_path)
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"缺少 labelme json: {jpath}")

        label_name = preferred_label
        shape = qr_core.read_shape_from_labelme(jpath, preferred_label)
        if shape is None:
            label_name = "roi"
            shape = qr_core.read_shape_from_labelme(jpath, label_name)
        if shape is None:
            raise RuntimeError(f"{os.path.basename(img_path)} 缺少 {preferred_label}/roi 标注")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        h_img, w_img = img_bgr.shape[:2]

        pts = np.asarray(shape.get("points", []), dtype=np.float32)
        if pts.size == 0:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI points empty")
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        x = max(0, int(np.floor(float(x_min))))
        y = max(0, int(np.floor(float(y_min))))
        x2 = min(w_img, int(np.ceil(float(x_max))))
        y2 = min(h_img, int(np.ceil(float(y_max))))
        if x2 <= x or y2 <= y:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI bbox invalid")

        crop_bgr = img_bgr[y:y2, x:x2].copy()
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
        rel_pts = pts - np.array([[x, y]], dtype=np.float32)
        if str(shape.get("shape_type", "rectangle")) == "polygon" and len(rel_pts) >= 3:
            cv2.fillPoly(mask, [np.round(rel_pts).astype(np.int32)], 255)
        else:
            p0 = rel_pts.min(axis=0)
            p1 = rel_pts.max(axis=0)
            rx = max(0, int(np.floor(float(p0[0]))))
            ry = max(0, int(np.floor(float(p0[1]))))
            rx2 = min(mask.shape[1], int(np.ceil(float(p1[0]))))
            ry2 = min(mask.shape[0], int(np.ceil(float(p1[1]))))
            mask[ry:ry2, rx:rx2] = 255
        return {
            "label_name": label_name,
            "crop_bgr": crop_bgr,
            "crop_gray": crop_gray,
            "mask": mask,
            "bbox_xywh": (x, y, x2 - x, y2 - y),
        }

    def _compute_traditional_baseline_metrics(self, img_path: str, preferred_label: str = "roi1") -> Dict[str, object]:
        roi = self._load_roi_mask_crop(img_path, preferred_label=preferred_label)
        crop_gray = np.asarray(roi["crop_gray"], dtype=np.float32)
        crop_bgr = np.asarray(roi["crop_bgr"], dtype=np.uint8)
        mask = np.asarray(roi["mask"], dtype=np.uint8)
        valid_gray = crop_gray[mask > 0]
        if valid_gray.size == 0:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI valid pixels empty")
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        valid_hsv = hsv[mask > 0]
        if valid_hsv.size == 0:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI HSV pixels empty")
        valid_hsv = np.asarray(valid_hsv, dtype=np.float32)
        h_vals = valid_hsv[:, 0]
        s_vals = valid_hsv[:, 1]
        v_vals = valid_hsv[:, 2]

        return {
            "file_path": img_path,
            "file_name": os.path.basename(img_path),
            "roi_label": str(roi["label_name"]),
            "bbox_xywh": list(roi["bbox_xywh"]),
            "mean_intensity": float(np.mean(valid_gray)),
            "hsv_h_mean": float(np.mean(h_vals)),
            "hsv_h_std": float(np.std(h_vals)),
            "hsv_s_mean": float(np.mean(s_vals)),
            "hsv_s_std": float(np.std(s_vals)),
            "hsv_v_mean": float(np.mean(v_vals)),
            "hsv_v_std": float(np.std(v_vals)),
            "roi_area": int(valid_gray.size),
        }

    def _save_traditional_baseline_report(self, rows: List[Dict[str, object]], tab_name: str) -> Tuple[str, str]:
        report_dir = os.path.join(self._product_dir, "traditional_baseline_reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"baseline_roi1_hsv_{tab_name.lower()}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {
            "product": self.current_product,
            "tab": tab_name,
            "rows": rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file",
                    "roi_label",
                    "bbox_xywh",
                    "mean_intensity",
                    "hsv_h_mean",
                    "hsv_h_std",
                    "hsv_s_mean",
                    "hsv_s_std",
                    "hsv_v_mean",
                    "hsv_v_std",
                    "roi_area",
                    "error",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.get("file_name", ""),
                        row.get("roi_label", ""),
                        row.get("bbox_xywh", ""),
                        row.get("mean_intensity", ""),
                        row.get("hsv_h_mean", ""),
                        row.get("hsv_h_std", ""),
                        row.get("hsv_s_mean", ""),
                        row.get("hsv_s_std", ""),
                        row.get("hsv_v_mean", ""),
                        row.get("hsv_v_std", ""),
                        row.get("roi_area", ""),
                        row.get("error", ""),
                    ]
                )
        return json_path, csv_path

    def _run_traditional_baseline_debug(self):
        paths, tab_name = self._current_tab_paths_and_name()
        if not paths:
            QtWidgets.QMessageBox.information(self, "提示", "当前列表没有图片")
            return

        rows: List[Dict[str, object]] = []
        ok = 0
        for path in paths:
            try:
                row = self._compute_traditional_baseline_metrics(path, preferred_label="roi1")
                ok += 1
            except Exception as exc:
                row = {
                    "file_path": path,
                    "file_name": os.path.basename(path),
                    "roi_label": "",
                    "bbox_xywh": "",
                    "mean_intensity": "",
                    "hsv_h_mean": "",
                    "hsv_h_std": "",
                    "hsv_s_mean": "",
                    "hsv_s_std": "",
                    "hsv_v_mean": "",
                    "hsv_v_std": "",
                    "roi_area": "",
                    "error": str(exc),
                }
            rows.append(row)

        json_path, csv_path = self._save_traditional_baseline_report(rows, tab_name=tab_name)
        self.lbl_status.setText(f"状态：传统基线调试已完成，成功 {ok}/{len(paths)}，结果已保存")
        QtWidgets.QMessageBox.information(
            self,
            "传统基线调试",
            f"已完成当前 {tab_name} 列表的 ROI1/ROI 指标计算。\n"
            f"成功: {ok}/{len(paths)}\n\n"
            f"JSON:\n{json_path}\n\nCSV:\n{csv_path}",
        )

    def _daily_test_log_path(self) -> str:
        log_dir = os.path.join(self._product_dir, "test_logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, datetime.now().strftime("%Y%m%d") + ".csv")

    def _append_test_log(self, row: Dict[str, object]) -> str:
        csv_path = self._daily_test_log_path()
        fields = [
            "timestamp",
            "product",
            "algorithm",
            "score_mode",
            "margin",
            "topk",
            "file_name",
            "gt",
            "pred",
            "diff",
            "sim_ok",
            "sim_ng",
            "value",
            "threshold",
            "match_ms",
            "total_ms",
            "json_name",
        ]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "product": self.current_product,
                    "algorithm": self._current_algorithm(),
                    "score_mode": self.cmb_mode.currentText(),
                    "margin": float(self.spin_margin.value()),
                    "topk": int(self.spin_topk.value()),
                    "file_name": row.get("file_name", ""),
                    "gt": row.get("gt", ""),
                    "pred": row.get("pred", ""),
                    "diff": row.get("diff", ""),
                    "sim_ok": row.get("sim_ok", ""),
                    "sim_ng": row.get("sim_ng", ""),
                    "value": row.get("value", ""),
                    "threshold": row.get("threshold", ""),
                    "match_ms": row.get("match_ms", ""),
                    "total_ms": row.get("total_ms", ""),
                    "json_name": row.get("json_name", ""),
                }
            )
        return csv_path

    def _export_current_results_csv(self) -> None:
        if not self._current_result_rows:
            QtWidgets.QMessageBox.information(self, "提示", "当前没有可导出的测试结果")
            return
        export_dir = os.path.join(self._product_dir, "test_exports")
        os.makedirs(export_dir, exist_ok=True)
        csv_path = os.path.join(export_dir, f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "gt", "pred", "diff", "sim_ok", "sim_ng", "value", "threshold", "match_ms", "total_ms", "json"])
            for row in self._current_result_rows:
                writer.writerow(
                    [
                        row.get("file_name", ""),
                        row.get("gt", ""),
                        row.get("pred", ""),
                        row.get("diff", ""),
                        row.get("sim_ok", ""),
                        row.get("sim_ng", ""),
                        row.get("value", ""),
                        row.get("threshold", ""),
                        row.get("match_ms", ""),
                        row.get("total_ms", ""),
                        row.get("json_name", ""),
                    ]
                )
        QtWidgets.QMessageBox.information(self, "导出完成", f"测试结果已导出到：\n{csv_path}")

    def _run_margin_validation(self):
        if not self._is_embedding_algorithm():
            QtWidgets.QMessageBox.information(self, "提示", "传统算法不支持 Margin 建议，请切回嵌入式算法")
            return
        if self.model is None or self.model.backbone != self._current_algorithm():
            try:
                self._load_embedding_model_for_algorithm(self._current_algorithm())
            except Exception:
                pass
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先训练/注册（OK+NG）")
            return
        if not self.ok_files or not self.ng_files:
            QtWidgets.QMessageBox.warning(self, "提示", "需要至少一批 OK 和 NG 图片才能建议 margin。")
            return

        feat_net, _ = qr_core.load_backbone(self.model.backbone, device=self.model.device)
        rows: List[Dict[str, object]] = []
        try:
            for path in self.ok_files:
                row = self._predict_image(path, feat_net=feat_net, prefer_canvas_roi=False)
                row["gt"] = "OK"
                rows.append(row)
            for path in self.ng_files:
                row = self._predict_image(path, feat_net=feat_net, prefer_canvas_roi=False)
                row["gt"] = "NG"
                rows.append(row)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "验证失败", str(ex))
            return

        self._populate_results_table(rows)
        summary = self._suggest_margin_from_rows(rows)
        json_path, csv_path = self._save_margin_report(rows, summary)

        safe_range = summary.get("safe_range")
        safe_text = ""
        if isinstance(safe_range, tuple):
            safe_text = f"\n安全区间: {safe_range[0]:.4f} ~ {safe_range[1]:.4f}"

        self.lbl_status.setText(
            "状态："
            + f"当前margin={summary['current_margin']:.4f} acc={summary['current_accuracy']:.4f}  "
            + f"建议margin={summary['suggested_margin']:.4f} acc={summary['suggested_accuracy']:.4f}"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Margin 建议",
            f"当前 margin: {summary['current_margin']:.4f}\n"
            f"当前准确率: {summary['current_accuracy']:.4f}\n"
            f"建议 margin: {summary['suggested_margin']:.4f}\n"
            f"建议准确率: {summary['suggested_accuracy']:.4f}"
            + safe_text
            + f"\n\n报告已保存:\n{json_path}\n{csv_path}",
        )

    def _run_test(self):
        p = self.canvas.image_path()
        if p is None or not os.path.exists(p):
            QtWidgets.QMessageBox.warning(self, "提示", "请先打开一张测试图片")
            return

        algorithm = self._current_algorithm()
        if self._is_embedding_algorithm(algorithm):
            if self.model is None or self.model.backbone != algorithm:
                try:
                    self._load_embedding_model_for_algorithm(algorithm)
                except Exception:
                    pass
            if self.model is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先训练/注册（OK+NG）")
                return

        try:
            feat_net = None
            if self._is_embedding_algorithm(algorithm) and self.model is not None:
                feat_net, _ = qr_core.load_backbone(self.model.backbone, device=self.model.device)
            row = self._predict_image(p, feat_net=feat_net, prefer_canvas_roi=True)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "测试失败", str(ex))
            return

        self._populate_results_table([row])
        log_path = self._append_test_log(row)
        metric_text = ""
        if row.get("value") is not None:
            metric_text = f"  value={float(row['value']):.4f}  threshold={float(row.get('threshold', 0.0)):.4f}"
        self.lbl_status.setText(
            "状态："
            + f"TEST={os.path.basename(p)}  pred={row['pred']}  diff={float(row['diff']):.4f}"
            + metric_text
            + (f"  模板匹配={float(row['match_ms']):.1f}ms" if row.get("match_ms") is not None else "")
            + f"  总耗时={float(row['total_ms']):.1f}ms  日志={os.path.basename(log_path)}"
        )
        self._load_canvas_image(p)
        self._save_session()

    def _open_embedding_analysis_dialog(self):
        if not self._is_embedding_algorithm():
            QtWidgets.QMessageBox.information(self, "提示", "传统算法没有 embedding 可视化，请切回嵌入式算法")
            return
        dialog = EmbeddingAnalysisDialog(
            session_root=self._session_dir,
            initial_product=self.current_product,
            initial_backbone=self._current_algorithm(),
            parent=self,
        )
        dialog.exec()

    def _on_table_click(self, row: int, _col: int):
        """Click on test result row to load the corresponding image"""
        it = self.table.item(row, 0)
        if it is None:
            return
        p = it.data(QtCore.Qt.UserRole)
        if isinstance(p, str) and os.path.exists(p):
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)


    def _save_session(self):
        os.makedirs(self._session_dir, exist_ok=True)
        data = {
            "ok_files": self.ok_files,
            "ng_files": self.ng_files,
            "test_files": self.test_files,
            "ref_image": self.ref_image,
            "loc_method": "line2dup",
        }
        with open(self._session_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _try_load_session_on_start(self):
        self._load_runtime_params()
        self._apply_runtime_params_to_ui()
        data: Dict[str, object] = {}
        if os.path.exists(self._session_json):
            try:
                with open(self._session_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        def _filter_exists(xs):
            return [p for p in xs if isinstance(p, str) and os.path.exists(p)]

        self.ok_files = _filter_exists(data.get("ok_files", []))
        self.ng_files = _filter_exists(data.get("ng_files", []))
        self.test_files = _filter_exists(data.get("test_files", []))
        ref_image = data.get("ref_image", "")
        self.ref_image = ref_image if isinstance(ref_image, str) and os.path.exists(ref_image) else None
        if self.ref_image:
            self.lbl_ref.setText(f"参考图：{os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        self.loc_method = "line2dup"
        self.cmb_loc.setCurrentText(self.loc_method)
        self._refresh_lists()
        if os.path.exists(self._line2dup_recipe_path):
            try:
                self.line2dup_recipe = line2dup_locator.load_recipe_for_product(self._product_dir)
                if not self.ref_image and self.line2dup_recipe.reference_image and os.path.exists(self.line2dup_recipe.reference_image):
                    self.ref_image = self.line2dup_recipe.reference_image
                    self.lbl_ref.setText(f"参考图：{os.path.basename(self.ref_image)}")
                    self.lbl_ref.setToolTip(self.ref_image)
            except Exception:
                self.line2dup_recipe = None
        try:
            self._load_embedding_model_for_algorithm(self._current_algorithm())
        except Exception:
            self.model = None

    def _clear_session(self):
        ret = QtWidgets.QMessageBox.question(self, "清空会话", "确定清空会话（列表/参考图/模型缓存）吗？")
        if ret != QtWidgets.QMessageBox.Yes:
            return
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self.model = None
        self.line2dup_recipe = None
        self.ref_image = None
        self._line2dup_match_ms_by_image = {}
        self._line2dup_autogen_ms_by_image = {}
        self.lbl_ref.setText("参考图：未设置")
        self.lbl_status.setText("状态：未训练")
        self.table.setRowCount(0)
        self._current_result_rows = []
        self._refresh_lists()
        try:
            if os.path.exists(self._session_json):
                os.remove(self._session_json)
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()

