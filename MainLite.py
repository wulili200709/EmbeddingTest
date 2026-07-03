from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from common.app_paths import packaged_embedding_test_root

os.environ["LC_SYSTEM_LITE"] = "1"

from ui.debug_main_window import DebugMainWindow


APP_NAME = "LC System Lite"
WINDOWS_APP_ID = "LCSystem.Lite"


def _normalize_application_font(app: QtWidgets.QApplication) -> None:
    font = QtGui.QFont(app.font())
    if font.pointSizeF() > 0:
        return
    if font.pixelSize() > 0:
        font.setPointSize(max(1, int(round(font.pixelSize() * 0.75))))
    else:
        font.setPointSize(10)
    app.setFont(font)


def _resource_path(filename: str) -> Path:
    return packaged_embedding_test_root(__file__) / "res" / filename


def _app_icon() -> QtGui.QIcon:
    for name in ("logo.ico", "logo.png"):
        path = _resource_path(name)
        if path.exists():
            icon = QtGui.QIcon(str(path))
            if not icon.isNull():
                return icon
    return QtGui.QIcon()


def _set_windows_app_id(app_id: str) -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def main() -> None:
    _set_windows_app_id(WINDOWS_APP_ID)
    app = QtWidgets.QApplication(sys.argv)
    _normalize_application_font(app)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    icon = _app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = DebugMainWindow(lite_mode=True)
    if not icon.isNull():
        window.setWindowIcon(icon)

    screen = app.primaryScreen()
    available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1366, 768)
    small_screen = available.width() <= 1366 or available.height() <= 800
    if small_screen:
        window.showMaximized()
    else:
        window.resize(
            min(1400, max(1200, available.width() - 80)),
            min(900, max(800, available.height() - 80)),
        )
        window.show()
    app.exec()


if __name__ == "__main__":
    main()
