"""
roi_canvas_pyside6.py

可复用的 PySide6 ROI 画布组件：
- 用 QLabel 显示图片
- 支持 rect / polygon 两种 ROI 绘制
- ROI 坐标统一保存在原图坐标系（非缩放坐标）

说明：
- 这个模块只负责“画/编辑 ROI + 坐标换算 + 可视化叠加”
- ROI 的保存/读取（例如 labelme json）放在调用方（如 qr_core）处理
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


def pixmap_from_path(path: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(path)


@dataclass
class ShapeState:
    # rect: xywh; polygon: points
    shape_type: str = "rect"  # "rect" | "polygon"
    xywh: Optional[Tuple[int, int, int, int]] = None
    points: Optional[List[Tuple[float, float]]] = None


class RoiCanvas(QtWidgets.QLabel):
    """
    用 QLabel 显示图片，并支持鼠标拖拽画矩形 ROI / 点击画 polygon ROI。
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
        self._mouse_pos: Optional[Tuple[int, int]] = None  # 原图坐标（用于显示实时线段）
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def has_image(self) -> bool:
        return self._pixmap is not None and self._img_path is not None

    def image_path(self) -> Optional[str]:
        return self._img_path

    def set_image(self, path: str, pixmap: Optional[QtGui.QPixmap] = None) -> None:
        self._img_path = path
        self._pixmap = pixmap if pixmap is not None else pixmap_from_path(path)
        self.roi = ShapeState()
        self._dragging = False
        self._p0 = QtCore.QPoint()
        self._p1 = QtCore.QPoint()
        self._poly_pts = []
        self._mouse_pos = None
        self._update_scaled_pixmap()
        self.update()
        self.shapesChanged.emit()

    def clear_roi(self) -> None:
        self.roi.xywh = None
        self.roi.points = None
        self._poly_pts = []
        self._mouse_pos = None
        self.update()
        self.shapesChanged.emit()

    def set_roi_rect(self, xywh: Optional[Tuple[int, int, int, int]]) -> None:
        if xywh is None:
            self.roi.xywh = None
            self.roi.points = None
            return
        self.roi.shape_type = "rect"
        self.roi.xywh = xywh
        self.roi.points = None
        self._poly_pts = []
        self._mouse_pos = None
        self.update()
        self.shapesChanged.emit()

    def set_roi_polygon(self, points: Optional[List[Tuple[float, float]]]) -> None:
        if not points or len(points) < 3:
            self.roi.points = None
            self.roi.xywh = None
            return
        self.roi.shape_type = "polygon"
        self.roi.points = list(points)
        self.roi.xywh = None
        self._poly_pts = []
        self._mouse_pos = None
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
                self._mouse_pos = None
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
                self._mouse_pos = None
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

