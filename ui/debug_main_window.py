
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ui.debug import ToolPage
from ui.i18n import LANG_EN, LANG_ZH, language_code, set_language, tr
from .window_common import build_default_session_and_algo


class DebugMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, lite_mode: bool = False) -> None:
        super().__init__()
        self.lite_mode = bool(lite_mode)
        self.setWindowTitle("LC System Lite" if self.lite_mode else "Quick Register Debug")

        self.session, self.algo = build_default_session_and_algo(__file__)

        self.tool_page = ToolPage(self.session, self.algo, parent=self, lite_mode=self.lite_mode)
        self.setCentralWidget(self.tool_page)

        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.productDeleteRequested.connect(self._on_product_delete_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.load_session()
        self._build_menu_bar()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu(tr("menu.file"))
        file_menu.addAction(tr("action.reload_debug"), self._reload_debug_session)
        file_menu.addSeparator()
        file_menu.addAction(tr("action.exit"), self.close)

        tools_menu = self.menuBar().addMenu(tr("menu.tools"))
        if not self.lite_mode:
            tools_menu.addAction(tr("action.camera_tool"), self.tool_page.open_camera_debug_dialog)
            tools_menu.addAction(tr("action.io_tool"), self.tool_page.open_io_debug_dialog)
            tools_menu.addSeparator()
        tools_menu.addAction(tr("action.template_editor"), self.tool_page.open_template_editor_dialog)
        tools_menu.addAction("NCC位置修正工具", self.tool_page.open_ncc_match_dialog)
        tools_menu.addAction(tr("action.auto_region"), self.tool_page.open_template_match_dialog)
        tools_menu.addSeparator()
        tools_menu.addAction(tr("action.margin_validation"), self.tool_page.open_margin_validation_tool)
        if not self.lite_mode:
            tools_menu.addAction(tr("action.embedding_analysis"), self.tool_page.open_embedding_analysis_tool)
        tools_menu.addAction(tr("action.baseline_debug"), self.tool_page.open_baseline_debug_tool)

        path_menu = self.menuBar().addMenu(tr("menu.path"))
        path_menu.addAction(tr("action.open_product_dir"), self._open_current_product_dir)
        path_menu.addAction(tr("action.open_session_dir"), self._open_session_dir)

        language_menu = self.menuBar().addMenu(tr("menu.language"))
        self.act_language_zh = language_menu.addAction(tr("language.zh"))
        self.act_language_en = language_menu.addAction(tr("language.en"))
        self.act_language_zh.setCheckable(True)
        self.act_language_en.setCheckable(True)
        language_group = QtGui.QActionGroup(self)
        language_group.setExclusive(True)
        language_group.addAction(self.act_language_zh)
        language_group.addAction(self.act_language_en)
        self.act_language_zh.setChecked(language_code() == LANG_ZH)
        self.act_language_en.setChecked(language_code() == LANG_EN)
        self.act_language_zh.triggered.connect(lambda _checked=False: self._change_language(LANG_ZH))
        self.act_language_en.triggered.connect(lambda _checked=False: self._change_language(LANG_EN))

    def _reload_debug_session(self) -> None:
        self.tool_page.load_session()

    def _open_in_explorer(self, path: str) -> None:
        target = Path(path)
        if not target.exists():
            QtWidgets.QMessageBox.information(
                self,
                tr("dialog.path"),
                tr("dialog.path_missing", path=target),
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))

    def _open_current_product_dir(self) -> None:
        self._open_in_explorer(self.session.product_dir)

    def _open_session_dir(self) -> None:
        self._open_in_explorer(self.session.session_dir)

    def _change_language(self, language: str) -> None:
        set_language(language)
        if hasattr(self.tool_page, "retranslate_ui"):
            self.tool_page.retranslate_ui()
        self.menuBar().clear()
        self._build_menu_bar()

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
