"""
独立调试主窗口。

当前第一版只承载 ToolPage，运行链路入口由 RunMainWindow 负责。
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from ui.debug import ToolPage
from .window_common import build_default_session_and_algo


class DebugMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quick Register Debug")

        self.session, self.algo = build_default_session_and_algo(__file__)

        self.tool_page = ToolPage(self.session, self.algo, parent=self)
        self.setCentralWidget(self.tool_page)

        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.load_session()

    def _on_product_change_request(self, new_name: str) -> None:
        self.tool_page.apply_product_switch(new_name)

    def _on_session_clear_request(self) -> None:
        self.tool_page.reset_for_clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        super().closeEvent(event)


__all__ = ["DebugMainWindow"]
