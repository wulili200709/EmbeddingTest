from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.lazy_api import preload as preload_qr_core
from ui.window_common import embedding_test_root


APP_NAME = "LC System"
WINDOWS_APP_ID = "LCSystem.App.TaskbarV2"


def resource_path(filename: str) -> Path:
    return embedding_test_root(__file__) / "res" / filename


def app_icon() -> QtGui.QIcon:
    for name in ("logo.ico", "logo.png"):
        path = resource_path(name)
        if path.exists():
            icon = QtGui.QIcon(str(path))
            if not icon.isNull():
                return icon
    return QtGui.QIcon()


def load_app_version() -> str:
    setup_path = embedding_test_root(__file__) / "setup.py"
    try:
        text = setup_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "3.0.1"
    match = re.search(r'version\s*=\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip() or "dev"
    return "3.0.1"


APP_VERSION = load_app_version()


def set_windows_app_id(app_id: str) -> None:
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def shell_icon(sp: QtWidgets.QStyle.StandardPixmap) -> QtGui.QIcon:
    return QtWidgets.QApplication.style().standardIcon(sp)


class AlgorithmEngineWarmupThread(QtCore.QThread):
    warmupFinished = QtCore.Signal(bool, str)

    def run(self) -> None:
        try:
            preload_qr_core()
        except Exception as exc:
            self.warmupFinished.emit(False, str(exc))
            return
        self.warmupFinished.emit(True, "")


class BrandBannerWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._source = QtGui.QPixmap()
        self.setMinimumHeight(36)

    def set_source_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._source = QtGui.QPixmap(pixmap)
        self.setVisible(not self._source.isNull())
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#313131"))

        if self._source.isNull():
            painter.end()
            return

        if self.width() > self._source.width():
            foreground = self._source.scaled(
                max(1, self.width()),
                max(1, self.height()),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            x = 0
            y = 0
        else:
            foreground = QtGui.QPixmap(self._source)
            x = 0
            y = max(0, (self.height() - foreground.height()) // 2)

        painter.drawPixmap(x, y, foreground)
        painter.fillRect(0, 0, self.width(), 1, QtGui.QColor("#505050"))
        painter.end()
