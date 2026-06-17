from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui

from ui.i18n import tr


try:
    from devices import IoController
except Exception:
    IoController = None  # type: ignore[assignment]

try:
    from services import (
        FrameGrabService,
        HikCameraManager,
        HikCameraSettings,
        HikFrame,
        frame_to_bgr_image,
        frame_to_rgb_image,
    )
except Exception:
    FrameGrabService = None  # type: ignore[assignment]
    HikCameraManager = None  # type: ignore[assignment]
    HikCameraSettings = None  # type: ignore[assignment]
    HikFrame = None  # type: ignore[assignment]
    frame_to_bgr_image = None  # type: ignore[assignment]
    frame_to_rgb_image = None  # type: ignore[assignment]


def qimage_from_hik_frame(frame: "HikFrame") -> QtGui.QImage:
    if frame_to_rgb_image is None:
        raise RuntimeError("camera frame conversion service is unavailable")
    image = frame_to_rgb_image(frame)
    if image.ndim == 2:
        height, width = image.shape
        return QtGui.QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()

    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = np.ascontiguousarray(image[:, :, :3])
        height, width, _ = rgb.shape
        return QtGui.QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        ).copy()

    raise ValueError("unsupported frame shape for debug preview")


class DebugCameraPreviewThread(QtCore.QThread):
    frameReady = QtCore.Signal(QtGui.QImage)
    errorOccurred = QtCore.Signal(str)

    def __init__(self, frame_grab_service, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._frame_grab_service = frame_grab_service
        self._running = False
        self._last_error = ""

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                frame = self._frame_grab_service.capture_once("debug", timeout_ms=300)
            except Exception as exc:
                if self._running:
                    message = str(exc)
                    if message != self._last_error:
                        self.errorOccurred.emit(message)
                        self._last_error = message
                    self.msleep(120)
                continue

            self._last_error = ""
            try:
                image = qimage_from_hik_frame(frame)
            except Exception as exc:
                if self._running:
                    self.errorOccurred.emit(tr("debug.preview_convert_failed", error=exc))
                    self.msleep(120)
                continue

            self.frameReady.emit(image)
            self.msleep(30)
