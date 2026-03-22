"""
独立运行主窗口。

当前第一版只显示 RuntimeModePage；
内部持有一个隐藏 ToolPage，作为运行配置、检测项和预测入口来源。
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from application import (
    DEFAULT_RELEASE_PASSWORD,
    ProductRuntimeContext,
    RuntimeController,
)
from .runtime_mode_pyside6 import RuntimeModePage
from ui.window_common import (
    build_default_session_and_algo,
    connect_runtime_dialogs,
    connect_runtime_page,
    detect_runtime_import_error,
    update_runtime_preview,
)


_RUNTIME_IMPORT_ERROR = detect_runtime_import_error()


class RunMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quick Register Runtime")

        self.session, self.algo = build_default_session_and_algo(__file__)
        self.runtime_context = ProductRuntimeContext(self.session, self.algo)

        self.runtime_page = RuntimeModePage()
        self.runtime_page.edit_release_password.setText(DEFAULT_RELEASE_PASSWORD)
        self.setCentralWidget(self.runtime_page)

        self.runtime_ctrl = RuntimeController(
            session=self.session,
            algo=self.algo,
            runtime_context=self.runtime_context,
            import_error=_RUNTIME_IMPORT_ERROR,
            release_password=DEFAULT_RELEASE_PASSWORD,
            parent=self,
        )

        self._connect_signals()
        self.runtime_ctrl.refresh_all_status()

    def _connect_signals(self) -> None:
        connect_runtime_page(self.runtime_page, self.runtime_ctrl)
        self.runtime_ctrl.previewUpdated.connect(self._on_runtime_preview_updated)

        connect_runtime_dialogs(self, self.runtime_ctrl)

    def _on_runtime_preview_updated(self, role: str, path: str) -> None:
        update_runtime_preview(self.runtime_page, role, path)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.runtime_ctrl.disconnect(silent=True)
        super().closeEvent(event)


__all__ = ["RunMainWindow"]
