from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ui.roi_overlay_colors import ROI_STROKE_WIDTH


def pixmap_from_path(path: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(path)


def _enum_to_int(value: object) -> int:
    return int(getattr(value, "value", value))


@dataclass
class ShapeState:
    shape_type: str = "rect"
    xywh: Optional[Tuple[int, int, int, int]] = None
    points: Optional[List[Tuple[float, float]]] = None


@dataclass
class OverlayShape:
    shape_type: str
    xywh: Optional[Tuple[int, int, int, int]] = None
    points: Optional[List[Tuple[float, float]]] = None
    segments: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    text: str = ""
    text_pos: Optional[Tuple[float, float]] = None
    text_offset: Optional[Tuple[float, float]] = None
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(0, 128, 255))
    width: float = 1.0
    dash: bool = True


class RoiCanvas(QtWidgets.QLabel):
    shapesChanged = QtCore.Signal()
    imagePressed = QtCore.Signal(int, int, int)
    imageMoved = QtCore.Signal(int, int, int)
    imageReleased = QtCore.Signal(int, int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setScaledContents(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._img_path: Optional[str] = None
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._scaled_pm: Optional[QtGui.QPixmap] = None
        self._scale: float = 1.0
        self._offset = QtCore.QPoint(0, 0)
        self._zoom: float = 1.0
        self._zoom_min: float = 0.2
        self._zoom_max: float = 8.0
        self._pan_offset = QtCore.QPointF()
        self._panning = False
        self._pan_last_pos = QtCore.QPointF()
        self._space_pressed = False

        self._dragging = False
        self._p0 = QtCore.QPoint()
        self._p1 = QtCore.QPoint()

        self.roi = ShapeState()
        self.draw_shape = "rect"
        self._overlays: List[OverlayShape] = []
        self._poly_pts: List[Tuple[float, float]] = []
        self._mouse_pos: Optional[Tuple[int, int]] = None
        self._interaction_enabled = True
        self._outside_image_events_enabled = False

        self._roi_color = QtGui.QColor(0, 255, 0)
        self._roi_dash = False
        self._roi_width = ROI_STROKE_WIDTH
        self._preview_color = QtGui.QColor(255, 0, 0)
        self._preview_dash = True
        self._preview_width = ROI_STROKE_WIDTH

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
        self._overlays = []
        self._zoom = 1.0
        self._pan_offset = QtCore.QPointF()
        self._panning = False
        self._space_pressed = False
        self._update_scaled_pixmap()
        self._refresh_pan_cursor()
        self.update()
        self.shapesChanged.emit()

    def clear_image(self) -> None:
        self._img_path = None
        self._pixmap = None
        self._scaled_pm = None
        self._scale = 1.0
        self._offset = QtCore.QPoint(0, 0)
        self._zoom = 1.0
        self._pan_offset = QtCore.QPointF()
        self._panning = False
        self._space_pressed = False
        self.roi = ShapeState()
        self._dragging = False
        self._p0 = QtCore.QPoint()
        self._p1 = QtCore.QPoint()
        self._poly_pts = []
        self._mouse_pos = None
        self._overlays = []
        self.setPixmap(QtGui.QPixmap())
        self._refresh_pan_cursor()
        self.update()
        self.shapesChanged.emit()

    def zoom_factor(self) -> float:
        return float(self._zoom)

    def set_zoom(self, zoom: float, *, anchor: Optional[QtCore.QPointF] = None) -> None:
        old_zoom = float(self._zoom)
        new_zoom = max(self._zoom_min, min(self._zoom_max, float(zoom)))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        if anchor is None:
            anchor = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        anchor_from_center = QtCore.QPointF(anchor) - center
        ratio = new_zoom / max(old_zoom, 1e-9)
        self._pan_offset = anchor_from_center - (
            anchor_from_center - self._pan_offset
        ) * ratio
        self._zoom = new_zoom
        if self._zoom <= 1.0:
            self._pan_offset = QtCore.QPointF()
        self._update_scaled_pixmap()
        self._refresh_pan_cursor()
        self.update()

    def reset_zoom(self) -> None:
        self.reset_view()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_offset = QtCore.QPointF()
        self._panning = False
        self._update_scaled_pixmap()
        self._refresh_pan_cursor()
        self.update()

    def pan_offset(self) -> QtCore.QPointF:
        return QtCore.QPointF(self._pan_offset)

    def clear_roi(self, *, emit_signal: bool = True) -> None:
        self.roi.xywh = None
        self.roi.points = None
        self._poly_pts = []
        self._mouse_pos = None
        self.update()
        if emit_signal:
            self.shapesChanged.emit()

    def set_roi_rect(self, xywh: Optional[Tuple[int, int, int, int]], *, emit_signal: bool = True) -> None:
        if xywh is None:
            self.roi.shape_type = "rect"
            self.roi.xywh = None
            self.roi.points = None
        else:
            self.roi.shape_type = "rect"
            self.roi.xywh = tuple(int(v) for v in xywh)
            self.roi.points = None
        self._poly_pts = []
        self._mouse_pos = None
        self.update()
        if emit_signal:
            self.shapesChanged.emit()

    def set_roi_polygon(self, points: Optional[List[Tuple[float, float]]], *, emit_signal: bool = True) -> None:
        if not points or len(points) < 3:
            self.roi.shape_type = "polygon"
            self.roi.points = None
            self.roi.xywh = None
        else:
            self.roi.shape_type = "polygon"
            self.roi.points = [(float(x), float(y)) for x, y in points]
            self.roi.xywh = None
        self._poly_pts = []
        self._mouse_pos = None
        self.update()
        if emit_signal:
            self.shapesChanged.emit()

    def set_overlays(self, overlays: Optional[List[OverlayShape]]) -> None:
        self._overlays = list(overlays or [])
        self.update()

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)

    def set_outside_image_events_enabled(self, enabled: bool) -> None:
        self._outside_image_events_enabled = bool(enabled)

    def set_roi_style(
        self,
        *,
        roi_color: Optional[QtGui.QColor] = None,
        roi_dash: Optional[bool] = None,
        roi_width: Optional[float] = None,
        preview_color: Optional[QtGui.QColor] = None,
        preview_dash: Optional[bool] = None,
        preview_width: Optional[float] = None,
    ) -> None:
        if roi_color is not None:
            self._roi_color = QtGui.QColor(roi_color)
        if roi_dash is not None:
            self._roi_dash = bool(roi_dash)
        if roi_width is not None:
            self._roi_width = float(roi_width)
        if preview_color is not None:
            self._preview_color = QtGui.QColor(preview_color)
        if preview_dash is not None:
            self._preview_dash = bool(preview_dash)
        if preview_width is not None:
            self._preview_width = float(preview_width)
        self.update()

    def roi_xywh(self) -> Optional[Tuple[int, int, int, int]]:
        if self.roi.shape_type == "rect" and self.roi.xywh is not None:
            return self.roi.xywh
        if self.roi.shape_type == "polygon" and self.roi.points:
            xs = [float(x) for x, _y in self.roi.points]
            ys = [float(y) for _x, y in self.roi.points]
            x0 = int(round(min(xs)))
            y0 = int(round(min(ys)))
            x1 = int(round(max(xs)))
            y1 = int(round(max(ys)))
            return x0, y0, max(1, x1 - x0), max(1, y1 - y0)
        return None

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            self.setPixmap(QtGui.QPixmap())
            self._scaled_pm = None
            self._scale = 1.0
            self._offset = QtCore.QPoint(0, 0)
            self._pan_offset = QtCore.QPointF()
            return

        label_w = max(1, self.width())
        label_h = max(1, self.height())
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0:
            return

        scale = min(label_w / pm_w, label_h / pm_h) * float(self._zoom)
        new_w = max(1, int(pm_w * scale))
        new_h = max(1, int(pm_h * scale))
        scaled = self._pixmap.scaled(new_w, new_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._scaled_pm = scaled
        self._scale = float(scale)
        self._clamp_pan_offset(new_w, new_h)
        self._offset = QtCore.QPoint(
            int(round((label_w - new_w) / 2.0 + self._pan_offset.x())),
            int(round((label_h - new_h) / 2.0 + self._pan_offset.y())),
        )
        self.setPixmap(scaled)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def _clamp_pan_offset(self, scaled_width: int | None = None, scaled_height: int | None = None) -> None:
        if self._pixmap is None or self._scaled_pm is None and (
            scaled_width is None or scaled_height is None
        ):
            self._pan_offset = QtCore.QPointF()
            return
        width = float(
            scaled_width if scaled_width is not None else self._scaled_pm.width()
        )
        height = float(
            scaled_height if scaled_height is not None else self._scaled_pm.height()
        )
        limit_x = max(0.0, (width - max(1, self.width())) / 2.0)
        limit_y = max(0.0, (height - max(1, self.height())) / 2.0)
        self._pan_offset = QtCore.QPointF(
            max(-limit_x, min(limit_x, self._pan_offset.x())),
            max(-limit_y, min(limit_y, self._pan_offset.y())),
        )

    def _can_pan(self) -> bool:
        return bool(
            self._scaled_pm is not None
            and (
                self._scaled_pm.width() > self.width() + 1
                or self._scaled_pm.height() > self.height() + 1
            )
        )

    def _refresh_pan_cursor(self) -> None:
        if self._panning:
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif self._space_pressed and self.has_image():
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def _begin_pan(self, position: QtCore.QPointF) -> bool:
        if not self.has_image() or not self._can_pan():
            return False
        self._panning = True
        self._pan_last_pos = QtCore.QPointF(position)
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        self._refresh_pan_cursor()
        return True

    def _pos_to_image_xy(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
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

    def _pos_to_unclamped_image_xy(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        if self._pixmap is None or self._scaled_pm is None:
            return None
        x = pos.x() - self._offset.x()
        y = pos.y() - self._offset.y()
        return int(round(x / self._scale)), int(round(y / self._scale))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.has_image():
            return
        wants_pan = event.button() == QtCore.Qt.MouseButton.MiddleButton or (
            event.button() == QtCore.Qt.MouseButton.LeftButton and self._space_pressed
        )
        if wants_pan:
            if self._begin_pan(event.position()):
                event.accept()
            else:
                event.ignore()
            return
        image_xy = self._pos_to_image_xy(event.position().toPoint())
        signal_xy = image_xy
        if signal_xy is None and self._outside_image_events_enabled:
            signal_xy = self._pos_to_unclamped_image_xy(event.position().toPoint())
        if signal_xy is not None:
            self.imagePressed.emit(_enum_to_int(event.button()), int(signal_xy[0]), int(signal_xy[1]))
        if not self._interaction_enabled:
            return

        if self.draw_shape == "polygon":
            if event.button() == QtCore.Qt.LeftButton and image_xy is not None:
                self._poly_pts.append((float(image_xy[0]), float(image_xy[1])))
                self.update()
                self.shapesChanged.emit()
                return
            if event.button() == QtCore.Qt.RightButton:
                if len(self._poly_pts) >= 3:
                    self.roi.shape_type = "polygon"
                    self.roi.points = list(self._poly_pts)
                    self.roi.xywh = None
                self._poly_pts = []
                self._mouse_pos = None
                self.update()
                self.shapesChanged.emit()
                return

        if event.button() == QtCore.Qt.LeftButton and image_xy is not None:
            self._dragging = True
            self._p0 = QtCore.QPoint(*image_xy)
            self._p1 = QtCore.QPoint(*image_xy)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            current_pos = event.position()
            self._pan_offset += current_pos - self._pan_last_pos
            self._pan_last_pos = current_pos
            self._update_scaled_pixmap()
            self.update()
            event.accept()
            return
        image_xy = self._pos_to_image_xy(event.position().toPoint()) if self.has_image() else None
        signal_xy = image_xy
        if signal_xy is None and self.has_image() and self._outside_image_events_enabled:
            signal_xy = self._pos_to_unclamped_image_xy(event.position().toPoint())
        if signal_xy is not None:
            self.imageMoved.emit(_enum_to_int(event.buttons()), int(signal_xy[0]), int(signal_xy[1]))
        if not self._interaction_enabled:
            return

        if self._dragging and image_xy is not None:
            self._p1 = QtCore.QPoint(*image_xy)
            self.update()
        if self.draw_shape == "polygon" and self.has_image() and self._poly_pts:
            self._mouse_pos = image_xy
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning and event.button() in {
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.MouseButton.LeftButton,
        }:
            self._panning = False
            self._refresh_pan_cursor()
            event.accept()
            return
        image_xy = self._pos_to_image_xy(event.position().toPoint()) if self.has_image() else None
        signal_xy = image_xy
        if signal_xy is None and self.has_image() and self._outside_image_events_enabled:
            signal_xy = self._pos_to_unclamped_image_xy(event.position().toPoint())
        if signal_xy is not None:
            self.imageReleased.emit(_enum_to_int(event.button()), int(signal_xy[0]), int(signal_xy[1]))
        if not self._interaction_enabled:
            return

        if event.button() == QtCore.Qt.LeftButton and self._dragging and self.has_image():
            self._dragging = False
            if image_xy is not None:
                self._p1 = QtCore.QPoint(*image_xy)
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

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self._refresh_pan_cursor()
            event.accept()
            return
        if self.draw_shape == "polygon":
            if event.key() == QtCore.Qt.Key_Escape:
                self._poly_pts = []
                self._mouse_pos = None
                self.update()
                self.shapesChanged.emit()
                return
            if event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete) and self._poly_pts:
                self._poly_pts.pop()
                self.update()
                self.shapesChanged.emit()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if self._panning:
                self._panning = False
            self._refresh_pan_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton and self.has_image():
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self.has_image():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.set_zoom(self._zoom * factor, anchor=event.position())
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self.has_image():
            super().paintEvent(event)
            return

        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOption()
        option.initFrom(self)
        self.style().drawPrimitive(
            QtWidgets.QStyle.PrimitiveElement.PE_Widget,
            option,
            painter,
            self,
        )
        if self._scaled_pm is not None:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.drawPixmap(self._offset, self._scaled_pm)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        def draw_rect(xywh: Tuple[int, int, int, int], color: QtGui.QColor, width: float = 2.0, dash: bool = False) -> None:
            x, y, w, h = xywh
            sx = int(round(x * self._scale)) + self._offset.x()
            sy = int(round(y * self._scale)) + self._offset.y()
            sw = int(round(w * self._scale))
            sh = int(round(h * self._scale))
            pen = QtGui.QPen(color)
            pen.setWidthF(float(width))
            pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QtCore.QRect(sx, sy, sw, sh))

        def draw_poly(points: List[Tuple[float, float]], color: QtGui.QColor, width: float = 2.0, dash: bool = False) -> None:
            pen = QtGui.QPen(color)
            pen.setWidthF(float(width))
            pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            qpts = [
                QtCore.QPoint(
                    int(round(x * self._scale)) + self._offset.x(),
                    int(round(y * self._scale)) + self._offset.y(),
                )
                for x, y in points
            ]
            if len(qpts) >= 2:
                painter.drawPolyline(QtGui.QPolygon(qpts))

        def draw_points(points: List[Tuple[float, float]], color: QtGui.QColor, size: float = 2.0) -> None:
            pen = QtGui.QPen(color)
            pen.setWidthF(max(1.0, float(size)))
            painter.setPen(pen)
            qpts = [
                QtCore.QPoint(
                    int(round(x * self._scale)) + self._offset.x(),
                    int(round(y * self._scale)) + self._offset.y(),
                )
                for x, y in points
            ]
            if qpts:
                painter.drawPoints(QtGui.QPolygon(qpts))

        def draw_crosshairs(points: List[Tuple[float, float]], color: QtGui.QColor, size: float = 7.0) -> None:
            pen = QtGui.QPen(color)
            pen.setWidthF(1.25)
            pen.setCapStyle(QtCore.Qt.FlatCap)
            painter.setPen(pen)
            arm = max(5.0, float(size))
            for x, y in points:
                sx = float(x) * self._scale + self._offset.x()
                sy = float(y) * self._scale + self._offset.y()
                painter.drawLine(QtCore.QPointF(sx - arm, sy), QtCore.QPointF(sx + arm, sy))
                painter.drawLine(QtCore.QPointF(sx, sy - arm), QtCore.QPointF(sx, sy + arm))

        def draw_segments(
            segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
            color: QtGui.QColor,
            width: float = 1.0,
            dash: bool = False,
        ) -> None:
            pen = QtGui.QPen(color)
            pen.setWidthF(float(width))
            pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            for p0, p1 in segments:
                sx0 = int(round(p0[0] * self._scale)) + self._offset.x()
                sy0 = int(round(p0[1] * self._scale)) + self._offset.y()
                sx1 = int(round(p1[0] * self._scale)) + self._offset.x()
                sy1 = int(round(p1[1] * self._scale)) + self._offset.y()
                painter.drawLine(sx0, sy0, sx1, sy1)

        def draw_dimension(
            segment: Tuple[Tuple[float, float], Tuple[float, float]],
            color: QtGui.QColor,
            text: str = "",
            width: float = 2.0,
            text_pos: Optional[Tuple[float, float]] = None,
        ) -> None:
            p0, p1 = segment
            sx0 = float(p0[0] * self._scale) + self._offset.x()
            sy0 = float(p0[1] * self._scale) + self._offset.y()
            sx1 = float(p1[0] * self._scale) + self._offset.x()
            sy1 = float(p1[1] * self._scale) + self._offset.y()
            pen = QtGui.QPen(color)
            pen.setWidthF(float(width))
            pen.setStyle(QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(sx0, sy0), QtCore.QPointF(sx1, sy1))

            dx = sx1 - sx0
            dy = sy1 - sy0
            length = max(1.0, math.hypot(dx, dy))
            ux = dx / length
            uy = dy / length
            nx = -uy
            ny = ux
            head = max(7.0, min(14.0, length * 0.2))
            for sx, sy, sign in ((sx0, sy0, 1.0), (sx1, sy1, -1.0)):
                tip = QtCore.QPointF(sx, sy)
                back_x = sx + sign * ux * head
                back_y = sy + sign * uy * head
                poly = QtGui.QPolygonF(
                    [
                        tip,
                        QtCore.QPointF(back_x + nx * head * 0.45, back_y + ny * head * 0.45),
                        QtCore.QPointF(back_x - nx * head * 0.45, back_y - ny * head * 0.45),
                    ]
                )
                painter.setBrush(QtGui.QBrush(color))
                painter.drawPolygon(poly)

            if text:
                if text_pos is None:
                    tx = (sx0 + sx1) * 0.5 + nx * 10.0
                    ty = (sy0 + sy1) * 0.5 + ny * 10.0
                else:
                    tx = float(text_pos[0] * self._scale) + self._offset.x()
                    ty = float(text_pos[1] * self._scale) + self._offset.y()
                font = painter.font()
                font.setPointSize(max(8, int(round(10 * max(1.0, min(self._scale, 1.4))))))
                painter.setFont(font)
                metrics = QtGui.QFontMetrics(font)
                rect = metrics.boundingRect(text)
                box = QtCore.QRectF(
                    tx - rect.width() / 2 - 5,
                    ty - rect.height() / 2 - 3,
                    rect.width() + 10,
                    rect.height() + 6,
                )
                bg = QtGui.QColor(0, 0, 0, 170)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(bg))
                painter.drawRoundedRect(box, 3, 3)
                painter.setPen(QtGui.QPen(color))
                painter.drawText(box, QtCore.Qt.AlignCenter, text)

        def draw_text_overlay(
            text: str,
            color: QtGui.QColor,
            font_size: float = 16.0,
            text_pos: Optional[Tuple[float, float]] = None,
        ) -> None:
            lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
            if not lines or self._scaled_pm is None:
                return

            painter.save()
            image_rect = QtCore.QRect(
                self._offset.x(),
                self._offset.y(),
                self._scaled_pm.width(),
                self._scaled_pm.height(),
            )
            painter.setClipRect(image_rect)

            font = painter.font()
            font.setPixelSize(max(10, int(round(float(font_size)))))
            font.setBold(True)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            padding = 7
            margin_x = 12 if text_pos is None else max(0, int(round(float(text_pos[0]))))
            margin_y = 12 if text_pos is None else max(0, int(round(float(text_pos[1]))))
            available_width = max(80, image_rect.width() - margin_x - padding * 2 - 4)
            visible_lines = [
                metrics.elidedText(line, QtCore.Qt.TextElideMode.ElideRight, available_width)
                for line in lines
            ]
            text_width = max(metrics.horizontalAdvance(line) for line in visible_lines)
            line_height = metrics.height()
            box = QtCore.QRectF(
                image_rect.left() + margin_x,
                image_rect.top() + margin_y,
                text_width + padding * 2,
                line_height * len(visible_lines) + padding * 2,
            )
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 178)))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QtGui.QPen(color))
            for index, line in enumerate(visible_lines):
                baseline = box.top() + padding + metrics.ascent() + index * line_height
                painter.drawText(QtCore.QPointF(box.left() + padding, baseline), line)
            painter.restore()

        def draw_point_text_overlay(
            text: str,
            color: QtGui.QColor,
            font_size: float = 10.0,
            text_pos: Optional[Tuple[float, float]] = None,
            text_offset: Optional[Tuple[float, float]] = None,
        ) -> None:
            if not str(text or "").strip() or text_pos is None or self._scaled_pm is None:
                return
            painter.save()
            image_rect = QtCore.QRectF(
                float(self._offset.x()),
                float(self._offset.y()),
                float(self._scaled_pm.width()),
                float(self._scaled_pm.height()),
            )
            painter.setClipRect(image_rect)
            offset_x, offset_y = text_offset or (0.0, 13.0)
            anchor_x = float(text_pos[0]) * self._scale + self._offset.x() + float(offset_x)
            anchor_y = float(text_pos[1]) * self._scale + self._offset.y() + float(offset_y)
            font = painter.font()
            font.setPixelSize(max(9, int(round(float(font_size)))))
            font.setBold(False)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            rect = metrics.boundingRect(str(text))
            box = QtCore.QRectF(
                anchor_x - rect.width() / 2.0 - 4.0,
                anchor_y,
                rect.width() + 8.0,
                rect.height() + 4.0,
            )
            if box.left() < image_rect.left() + 2.0:
                box.moveLeft(image_rect.left() + 2.0)
            if box.right() > image_rect.right() - 2.0:
                box.moveRight(image_rect.right() - 2.0)
            if box.bottom() > image_rect.bottom() - 2.0:
                box.moveBottom(anchor_y - 5.0)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 180)))
            painter.drawRoundedRect(box, 3.0, 3.0)
            painter.setPen(QtGui.QPen(color))
            painter.drawText(box, QtCore.Qt.AlignCenter, str(text))
            painter.restore()

        if self._scaled_pm is not None and self._overlays:
            for overlay in self._overlays:
                if overlay.shape_type == "rect" and overlay.xywh is not None:
                    draw_rect(overlay.xywh, overlay.color, width=overlay.width, dash=overlay.dash)
                elif overlay.shape_type == "polygon" and overlay.points is not None:
                    draw_poly(overlay.points + [overlay.points[0]], overlay.color, width=overlay.width, dash=overlay.dash)
                elif overlay.shape_type == "points" and overlay.points is not None:
                    draw_points(overlay.points, overlay.color, size=overlay.width)
                elif overlay.shape_type == "crosshair" and overlay.points is not None:
                    draw_crosshairs(overlay.points, overlay.color, size=overlay.width)
                elif overlay.shape_type == "segments" and overlay.segments is not None:
                    draw_segments(overlay.segments, overlay.color, width=overlay.width, dash=overlay.dash)
                elif overlay.shape_type == "dimension" and overlay.segments:
                    draw_dimension(
                        overlay.segments[0],
                        overlay.color,
                        text=overlay.text,
                        width=overlay.width,
                        text_pos=overlay.text_pos,
                    )
                elif overlay.shape_type == "text" and overlay.text:
                    draw_text_overlay(
                        overlay.text,
                        overlay.color,
                        font_size=overlay.width,
                        text_pos=overlay.text_pos,
                    )
                elif overlay.shape_type == "point_text" and overlay.text:
                    draw_point_text_overlay(
                        overlay.text,
                        overlay.color,
                        font_size=overlay.width,
                        text_pos=overlay.text_pos,
                        text_offset=overlay.text_offset,
                    )

        if self._scaled_pm is not None:
            if self.roi.shape_type == "rect" and self.roi.xywh is not None:
                draw_rect(self.roi.xywh, self._roi_color, width=self._roi_width, dash=self._roi_dash)
            elif self.roi.shape_type == "polygon" and self.roi.points is not None:
                draw_poly(self.roi.points + [self.roi.points[0]], self._roi_color, width=self._roi_width, dash=self._roi_dash)

        if self._scaled_pm is not None and self._poly_pts:
            draw_poly(self._poly_pts, self._preview_color, width=self._preview_width, dash=self._preview_dash)
            painter.setBrush(QtGui.QBrush(self._preview_color))
            painter.setPen(QtGui.QPen(self._preview_color, 1))
            for x, y in self._poly_pts:
                sx = int(round(x * self._scale)) + self._offset.x()
                sy = int(round(y * self._scale)) + self._offset.y()
                painter.drawEllipse(QtCore.QPoint(sx, sy), 3, 3)
            if self._mouse_pos is not None:
                last_x, last_y = self._poly_pts[-1]
                last_sx = int(round(last_x * self._scale)) + self._offset.x()
                last_sy = int(round(last_y * self._scale)) + self._offset.y()
                mouse_sx = int(round(self._mouse_pos[0] * self._scale)) + self._offset.x()
                mouse_sy = int(round(self._mouse_pos[1] * self._scale)) + self._offset.y()
                ghost = QtGui.QColor(self._preview_color)
                ghost.setAlpha(128)
                pen = QtGui.QPen(ghost, 1, QtCore.Qt.DashLine if self._preview_dash else QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(last_sx, last_sy, mouse_sx, mouse_sy)

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
            pen = QtGui.QPen(
                self._preview_color,
                int(self._preview_width),
                QtCore.Qt.DashLine if self._preview_dash else QtCore.Qt.SolidLine,
            )
            painter.setPen(pen)
            painter.drawRect(QtCore.QRect(sx, sy, sw, sh))
