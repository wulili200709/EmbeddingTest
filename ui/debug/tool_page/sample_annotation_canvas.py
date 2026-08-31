from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ui.debug import OverlayShape
from ui.i18n import tr


def _pixmap_from_path(path: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(path)

class _SampleAnnotationCanvas(QtWidgets.QWidget):
    imagePressed = QtCore.Signal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._pixmap = QtGui.QPixmap()
        self._scaled_pixmap = QtGui.QPixmap()
        self._overlays: list[OverlayShape] = []
        self._scale = 1.0
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 12.0
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._image_offset = QtCore.QPointF(0.0, 0.0)
        self._pressed_button = 0
        self._press_pos = QtCore.QPointF()
        self._pan_anchor = QtCore.QPointF()
        self._is_panning = False

    def clear_image(self) -> None:
        self._pixmap = QtGui.QPixmap()
        self._scaled_pixmap = QtGui.QPixmap()
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._zoom = 1.0
        self._pressed_button = 0
        self._is_panning = False
        self.unsetCursor()
        self.update()

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = QtGui.QPixmap(pixmap)
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._zoom = 1.0
        self._update_scaled_pixmap()
        self._refresh_cursor()
        self.update()

    def set_overlays(self, overlays: Optional[List[OverlayShape]]) -> None:
        self._overlays = list(overlays or [])
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pixmap.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        old_scale = self._scale
        anchor_widget = event.position()
        anchor_image = self._widget_to_image(anchor_widget)
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._zoom = max(self._zoom_min, min(self._zoom_max, self._zoom * factor))
        self._update_scaled_pixmap()
        if anchor_image is not None and old_scale > 0.0:
            new_scale = self._scale
            new_offset_x = float(anchor_widget.x()) - (anchor_image[0] * new_scale)
            new_offset_y = float(anchor_widget.y()) - (anchor_image[1] * new_scale)
            self._pan_offset = QtCore.QPointF(
                new_offset_x - self._base_image_offset().x(),
                new_offset_y - self._base_image_offset().y(),
            )
            self._pan_offset = self._clamp_pan_offset(self._pan_offset)
            self._image_offset = QtCore.QPointF(
                self._base_image_offset().x() + self._pan_offset.x(),
                self._base_image_offset().y() + self._pan_offset.y(),
            )
        self.update()
        self._refresh_cursor()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull():
            super().mousePressEvent(event)
            return
        pressed_button = int(getattr(event.button(), "value", event.button()))
        left_button = int(getattr(QtCore.Qt.MouseButton.LeftButton, "value", QtCore.Qt.MouseButton.LeftButton))
        middle_button = int(getattr(QtCore.Qt.MouseButton.MiddleButton, "value", QtCore.Qt.MouseButton.MiddleButton))
        right_button = int(getattr(QtCore.Qt.MouseButton.RightButton, "value", QtCore.Qt.MouseButton.RightButton))
        if pressed_button not in {left_button, middle_button, right_button}:
            super().mousePressEvent(event)
            return
        self._pressed_button = pressed_button
        self._press_pos = event.position()
        self._pan_anchor = QtCore.QPointF(self._pan_offset)
        self._is_panning = False
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull() or self._pressed_button == 0:
            super().mouseMoveEvent(event)
            return
        left_button = int(getattr(QtCore.Qt.MouseButton.LeftButton, "value", QtCore.Qt.MouseButton.LeftButton))
        middle_button = int(getattr(QtCore.Qt.MouseButton.MiddleButton, "value", QtCore.Qt.MouseButton.MiddleButton))
        if self._pressed_button not in {left_button, middle_button}:
            return
        if not self._can_pan():
            return
        if not self._is_panning:
            if (event.position() - self._press_pos).manhattanLength() < 6:
                return
            self._is_panning = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        delta = event.position() - self._press_pos
        self._pan_offset = self._clamp_pan_offset(
            QtCore.QPointF(
                self._pan_anchor.x() + delta.x(),
                self._pan_anchor.y() + delta.y(),
            )
        )
        self._image_offset = QtCore.QPointF(
            self._base_image_offset().x() + self._pan_offset.x(),
            self._base_image_offset().y() + self._pan_offset.y(),
        )
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull():
            super().mouseReleaseEvent(event)
            return
        released_button = int(getattr(event.button(), "value", event.button()))
        image_xy = self._widget_to_image(event.position())
        was_panning = bool(self._is_panning)
        if released_button == self._pressed_button and not was_panning and image_xy is not None:
            self.imagePressed.emit(released_button, int(round(image_xy[0])), int(round(image_xy[1])))
        self._pressed_button = 0
        self._is_panning = False
        self._refresh_cursor()
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#111111"))
        if self._pixmap.isNull() or self._scaled_pixmap.isNull():
            painter.setPen(QtGui.QColor("#8a8a8a"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, tr("sample.no_selected_image"))
            return
        top_left = self._image_offset
        painter.drawPixmap(int(round(top_left.x())), int(round(top_left.y())), self._scaled_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for overlay in self._overlays:
            pen = QtGui.QPen(QtGui.QColor(overlay.color))
            pen.setWidthF(float(overlay.width))
            pen.setStyle(QtCore.Qt.DashLine if overlay.dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            if overlay.shape_type == "rect" and overlay.xywh is not None:
                x, y, w, h = overlay.xywh
                painter.drawRect(
                    QtCore.QRectF(
                        top_left.x() + float(x) * self._scale,
                        top_left.y() + float(y) * self._scale,
                        float(w) * self._scale,
                        float(h) * self._scale,
                    )
                )
            elif overlay.shape_type == "polygon" and overlay.points:
                polygon = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(
                            top_left.x() + float(x) * self._scale,
                            top_left.y() + float(y) * self._scale,
                        )
                        for x, y in overlay.points
                    ]
                )
                painter.drawPolygon(polygon)

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap.isNull():
            self._scaled_pixmap = QtGui.QPixmap()
            self._scale = 1.0
            self._image_offset = QtCore.QPointF(0.0, 0.0)
            return
        base_scale = min(
            max(1, self.width()) / max(1, self._pixmap.width()),
            max(1, self.height()) / max(1, self._pixmap.height()),
        )
        self._scale = max(0.01, float(base_scale) * float(self._zoom))
        target_size = QtCore.QSize(
            max(1, int(round(self._pixmap.width() * self._scale))),
            max(1, int(round(self._pixmap.height() * self._scale))),
        )
        self._scaled_pixmap = self._pixmap.scaled(
            target_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        base_offset = self._base_image_offset()
        self._pan_offset = self._clamp_pan_offset(self._pan_offset)
        self._image_offset = QtCore.QPointF(base_offset.x() + self._pan_offset.x(), base_offset.y() + self._pan_offset.y())
        self._refresh_cursor()

    def _base_image_offset(self) -> QtCore.QPointF:
        # Keep the scaled image centered even when it is larger than the
        # viewport.  The old max(0, ...) anchoring put oversized images at
        # (0, 0), while the pan clamp assumed a centered origin.  That mismatch
        # made one drag direction appear locked after zooming.
        return QtCore.QPointF(
            float((self.width() - self._scaled_pixmap.width()) / 2.0),
            float((self.height() - self._scaled_pixmap.height()) / 2.0),
        )

    def _refresh_cursor(self) -> None:
        if self._is_panning:
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif self._can_pan():
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def _widget_to_image(self, widget_pos: QtCore.QPointF) -> tuple[float, float] | None:
        if self._pixmap.isNull() or self._scale <= 0.0:
            return None
        image_x = (float(widget_pos.x()) - self._image_offset.x()) / self._scale
        image_y = (float(widget_pos.y()) - self._image_offset.y()) / self._scale
        if image_x < 0.0 or image_y < 0.0 or image_x > self._pixmap.width() or image_y > self._pixmap.height():
            return None
        return (image_x, image_y)

    def _can_pan(self) -> bool:
        return (
            not self._scaled_pixmap.isNull()
            and (self._scaled_pixmap.width() > self.width() or self._scaled_pixmap.height() > self.height())
        )

    def _clamp_pan_offset(self, pan_offset: QtCore.QPointF) -> QtCore.QPointF:
        if self._scaled_pixmap.isNull():
            return QtCore.QPointF(0.0, 0.0)
        max_abs_x = max(0.0, (self._scaled_pixmap.width() - self.width()) / 2.0)
        max_abs_y = max(0.0, (self._scaled_pixmap.height() - self.height()) / 2.0)
        return QtCore.QPointF(
            max(-max_abs_x, min(max_abs_x, float(pan_offset.x()))),
            max(-max_abs_y, min(max_abs_y, float(pan_offset.y()))),
        )

