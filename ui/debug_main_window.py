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
        self.tool_page.productDeleteRequested.connect(self._on_product_delete_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.load_session()

    def _on_product_change_request(self, new_name: str) -> None:
        self.tool_page.apply_product_switch(new_name)

    def _on_product_delete_request(self, product_name: str) -> None:
        name = str(product_name or "").strip()
        if not name:
            return
        ret = QtWidgets.QMessageBox.question(
            self,
            "\u5220\u9664\u4ea7\u54c1",
            f"\u786e\u8ba4\u5220\u9664\u4ea7\u54c1 {name}?\n"
            "\u4ea7\u54c1\u76ee\u5f55\u4f1a\u79fb\u52a8\u5230 _deleted\uff0c\u53ef\u624b\u52a8\u6062\u590d\u3002",
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return
        error = self.session.delete_product(name)
        if error:
            QtWidgets.QMessageBox.critical(self, "\u5220\u9664\u4ea7\u54c1", error)
            return
        self.tool_page.refresh_product_selector()
        self.tool_page.apply_product_switch(self.session.current_product)

    def _on_session_clear_request(self) -> None:
        self.tool_page.reset_for_clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        super().closeEvent(event)


__all__ = ["DebugMainWindow"]
