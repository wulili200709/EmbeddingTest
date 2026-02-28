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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from PySide6 import QtCore, QtGui, QtWidgets

import qr_core


SUPPORTED_BACKBONES = ["efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large"]
SUPPORTED_SCORE_MODES = ["proto", "topk"]
SUPPORTED_LOC_MODES = ["none", "template", "orb"]
SUPPORTED_SHAPES = ["rect", "polygon"]


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

        self._session_dir = os.path.join(os.getcwd(), ".qr_session")
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
        self.cmb_backbone = QtWidgets.QComboBox()
        self.cmb_backbone.addItems(SUPPORTED_BACKBONES)
        self.cmb_backbone.currentTextChanged.connect(self._on_backbone_changed)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(SUPPORTED_SCORE_MODES)
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setDecimals(4)
        self.spin_margin.setSingleStep(0.005)
        self.spin_margin.setRange(-1.0, 1.0)
        self.spin_margin.setValue(0.02)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setRange(1, 50)
        self.spin_topk.setValue(3)
        form.addRow("BACKBONE", self.cmb_backbone)
        form.addRow("算法", self.cmb_mode)
        form.addRow("Margin", self.spin_margin)
        form.addRow("TopK(仅topk)", self.spin_topk)
        left.addWidget(box, 0)

        act = QtWidgets.QHBoxLayout()
        self.btn_train = QtWidgets.QPushButton("训练/注册 (OK+NG)")
        self.btn_train.clicked.connect(self._train)
        self.btn_test = QtWidgets.QPushButton("测试 TEST")
        self.btn_test.clicked.connect(self._run_test)
        act.addWidget(self.btn_train)
        act.addWidget(self.btn_test)
        left.addLayout(act)

        self.lbl_status = QtWidgets.QLabel("状态：未训练")
        left.addWidget(self.lbl_status)

        # right: image + results
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(640, 480)
        self.canvas.shapesChanged.connect(self._on_shapes_changed)
        right.addWidget(self.canvas, 2)

        roi_bar = QtWidgets.QHBoxLayout()
        roi_bar.addWidget(QtWidgets.QLabel("形状："))
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(SUPPORTED_SHAPES)
        self.cmb_shape.setCurrentText("rect")
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        roi_bar.addWidget(self.cmb_shape)

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



        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["文件", "Pred", "diff", "sim_ok", "sim_ng", "json"])
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
            self.canvas.load_image(p)

    def _on_select_ng(self):
        p = self._current_selected_path()
        if p:
            self.canvas.load_image(p)

    def _on_select_test(self):
        p = self._current_selected_path()
        if p:
            self.canvas.load_image(p)

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
            self.canvas.load_image(self.ok_files[0])
        elif index == 1 and self.ng_files:
            self.ng_list.setCurrentRow(0)
            # 强制加载图片，即使已经选中第0行
            self.canvas.load_image(self.ng_files[0])
        elif index == 2 and self.test_files:
            self.test_list.setCurrentRow(0)
            # 强制加载图片，即使已经选中第0行
            self.canvas.load_image(self.test_files[0])

    def _on_shape_changed(self):
        self.canvas.draw_shape = self.cmb_shape.currentText()
        # 清空当前正在画的 polygon 点，避免混乱
        self.canvas._poly_pts = []
        self.canvas.update()
        self._on_shapes_changed()

    def _clear_current_rect(self):
        # 清空正在画的 polygon 点
        self.canvas._poly_pts = []
        self.canvas.roi.xywh = None
        self.canvas.roi.points = None

        # 如果已经保存过 json，则同步从 json 删除该标注（否则切图会被读回来，看起来像“清不掉”）
        p = self.canvas.image_path()
        if p is not None:
            try:
                deleted = qr_core.delete_labelme_shape(p, label_name="roi")
            except Exception:
                deleted = False
            if deleted:
                # 重新加载，确保 UI 与 json 一致
                self.canvas.load_image(p)
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
        st = self.canvas.roi
        if st.shape_type == "rect" and st.xywh is not None:
            return st.xywh
        if st.shape_type == "polygon" and st.points:
            xs = [p[0] for p in st.points]
            ys = [p[1] for p in st.points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            return int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))
        return None


    def _save_current_rect(self):
        p = self.canvas.image_path()
        if p is None:
            return
        st = self.canvas.roi

        if st.shape_type == "rect":
            if st.xywh is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先拖拽画出矩形标注")
                return
            jpath = qr_core.upsert_labelme_rect(p, st.xywh, label_name="roi")
        else:
            if not st.points or len(st.points) < 3:
                QtWidgets.QMessageBox.warning(self, "提示", "多边形至少需要 3 个点（左键点选加点，右键结束）")
                return
            jpath = qr_core.upsert_labelme_polygon(p, st.points, label_name="roi")

        QtWidgets.QMessageBox.information(self, "已保存", f"已更新 labelme json：\n{jpath}\n(label=roi, type={st.shape_type})")
        # reload to ensure consistent
        self.canvas.load_image(p)

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
        self._session_json = os.path.join(product_dir, "session.json")

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
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self.table.setRowCount(0)
        self.canvas.clear()
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

    def _on_backbone_changed(self, backbone: str):
        """Backbone切换时自动加载对应的模型，并恢复该模型的训练参数"""
        if not backbone:
            return
        # 使用当前产品目录
        product_dir = os.path.join(self._session_dir, self.current_product)
        model_file = os.path.join(product_dir, f"register_model_{backbone}.npz")
        if os.path.exists(model_file):
            try:
                self.model = qr_core.load_register_model_npz(model_file)
                # 从模型中恢复训练参数到UI
                self.cmb_mode.setCurrentText(self.model.score_mode)
                self.spin_margin.setValue(self.model.margin)
                self.spin_topk.setValue(self.model.topk)
                self.lbl_status.setText(
                    f"状态：已加载模型  backbone={self.model.backbone}  mode={self.model.score_mode}  "
                    f"margin={self.model.margin:.4f}  topk={self.model.topk}"
                )
            except Exception as e:
                self.model = None
                self.lbl_status.setText(f"状态：加载{backbone}模型失败 - {e}")
        else:
            self.model = None
            self.lbl_status.setText(f"状态：{backbone}模型未训练")

    def _train(self):
        # 每次点击都视为“新注册”
        self.model = None
        self.table.setRowCount(0)
        # 检查 json 是否齐全
        missing = []
        for p in list(self.ok_files) + list(self.ng_files):
            j = qr_core.labelme_json_of_image(p)
            if not os.path.exists(j):
                missing.append(os.path.basename(p))
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "缺少ROI标注",
                "需要每张 OK/NG 图都有同名 json(roi)。\n请逐张打开图片 -> 画 ROI -> 保存。\n缺少：\n"
                + "\n".join(missing[:50]),
            )
            return

        backbone = self.cmb_backbone.currentText()
        mode = self.cmb_mode.currentText()
        margin = float(self.spin_margin.value())
        topk = int(self.spin_topk.value())

        try:
            self.model = qr_core.train_register_model(
                self.ok_files,
                self.ng_files,
                backbone=backbone,
                score_mode=mode,
                margin=margin,
                topk=topk,
                label_name="roi",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "训练失败", str(e))
            return

        self.lbl_status.setText(
            f"状态：已训练  backbone={backbone}  mode={mode}  margin={margin:.4f}  topk={topk}"
        )
        # 自动保存：重启后可直接测试（根据backbone保存不同文件）
        try:
            product_dir = os.path.join(self._session_dir, self.current_product)
            os.makedirs(product_dir, exist_ok=True)
            assert self.model is not None
            model_file = os.path.join(product_dir, f"register_model_{backbone}.npz")
            qr_core.save_register_model_npz(self.model, model_file)
            self._save_session()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "保存会话失败", str(e))
        QtWidgets.QMessageBox.information(self, "训练完成", "OK/NG 注册完成，可以开始测试。")

    def _run_test(self):
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先训练/注册（OK+NG）")
            return
        
        # 测试当前画布上的图片和ROI
        p = self.canvas.image_path()
        roi = self._roi_xywh_from_canvas()
        if p is None or not os.path.exists(p) or roi is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先打开一张测试图片，并在图上画 ROI")
            return
        
        feat_net, _ = qr_core.load_backbone(self.model.backbone, device=self.model.device)
        try:
            e = qr_core.embed_one(p, feat_net, label_name=self.model.label_name, device=self.model.device, roi_xywh=roi)
            pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(e, self.model)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "测试失败", str(ex))
            return
        
        self.table.setRowCount(0)
        self.table.insertRow(0)
        vals = [os.path.basename(p), pred, f"{diff:.4f}", f"{sim_ok:.4f}", f"{sim_ng:.4f}", os.path.basename(qr_core.labelme_json_of_image(p))]
        for c, v in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(v)
            if c == 0:
                item.setData(QtCore.Qt.UserRole, p)
            self.table.setItem(0, c, item)
        self._save_session()

    def _on_table_click(self, row: int, _col: int):
        """Click on test result row to load the corresponding image"""
        it = self.table.item(row, 0)
        if it is None:
            return
        p = it.data(QtCore.Qt.UserRole)
        if isinstance(p, str) and os.path.exists(p):
            self.canvas.load_image(p)


    def _save_session(self):
        """只保存文件列表，训练参数随模型保存在.npz中"""
        os.makedirs(self._session_dir, exist_ok=True)
        data = {
            "ok_files": self.ok_files,
            "ng_files": self.ng_files,
            "test_files": self.test_files,
        }
        with open(self._session_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _try_load_session_on_start(self):
        if not os.path.exists(self._session_json):
            return
        try:
            with open(self._session_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        def _filter_exists(xs):
            return [p for p in xs if isinstance(p, str) and os.path.exists(p)]

        self.ok_files = _filter_exists(data.get("ok_files", []))
        self.ng_files = _filter_exists(data.get("ng_files", []))
        self.test_files = _filter_exists(data.get("test_files", []))
        self._refresh_lists()

        # 不再从session.json恢复训练参数，参数将在加载模型时恢复

        # 自动加载模型（重启可继续测试）- 根据当前backbone加载对应模型
        current_backbone = self.cmb_backbone.currentText()
        product_dir = os.path.join(self._session_dir, self.current_product)
        model_file = os.path.join(product_dir, f"register_model_{current_backbone}.npz")
        if os.path.exists(model_file):
            try:
                self.model = qr_core.load_register_model_npz(model_file)
                self.lbl_status.setText(
                    f"状态：已加载模型  backbone={self.model.backbone}  mode={self.model.score_mode}  "
                    f"margin={self.model.margin:.4f}  topk={self.model.topk}"
                )
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
        self.ref_image = None
        self.lbl_ref.setText("参考图：未设置")
        self.lbl_status.setText("状态：未训练")
        self.table.setRowCount(0)
        self._refresh_lists()
        try:
            if os.path.exists(self._session_json):
                os.remove(self._session_json)
            if os.path.exists(self._model_npz):
                os.remove(self._model_npz)
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

