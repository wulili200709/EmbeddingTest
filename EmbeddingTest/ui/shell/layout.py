from __future__ import annotations

from PySide6 import QtWidgets

from ui.debug import ToolPage
from ui.runtime import RuntimeModePage

from .support import BrandBannerWidget, shell_icon


SP = QtWidgets.QStyle.StandardPixmap


def build_main_window_ui(window) -> None:
    central_widget = QtWidgets.QWidget()
    window.setCentralWidget(central_widget)
    root_layout = QtWidgets.QVBoxLayout(central_widget)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(0)

    body = QtWidgets.QWidget()
    outer = QtWidgets.QHBoxLayout(body)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    nav_frame = QtWidgets.QFrame()
    nav_frame.setObjectName("workspaceNav")
    nav_frame.setFixedWidth(96)
    nav_frame.setStyleSheet(
        "#workspaceNav{background:#333333;border-right:1px solid #4a4a4a;}"
        "QPushButton{color:#d0d0d0;background:transparent;border:none;padding:12px 8px;text-align:center;}"
        "QPushButton:checked{background:#3d3d40;color:#ffffff;border-left:3px solid #3794ff;}"
        "QPushButton:hover{background:#3a3a3a;}"
    )
    nav_layout = QtWidgets.QVBoxLayout(nav_frame)
    nav_layout.setContentsMargins(0, 8, 0, 8)
    nav_layout.setSpacing(4)

    window.btn_workspace_debug = QtWidgets.QPushButton(
        shell_icon(SP.SP_FileDialogDetailedView), "调试界面"
    )
    window.btn_workspace_debug.setCheckable(True)
    window.btn_workspace_debug.clicked.connect(lambda: window._switch_workspace("debug"))
    nav_layout.addWidget(window.btn_workspace_debug)

    window.btn_workspace_runtime = QtWidgets.QPushButton(
        shell_icon(SP.SP_MediaPlay), "运行界面"
    )
    window.btn_workspace_runtime.setCheckable(True)
    window.btn_workspace_runtime.clicked.connect(lambda: window._switch_workspace("runtime"))
    nav_layout.addWidget(window.btn_workspace_runtime)
    nav_layout.addStretch(1)
    outer.addWidget(nav_frame, 0)

    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(0)

    window.lbl_workspace_title = QtWidgets.QLabel()
    window.lbl_workspace_title.hide()
    window.lbl_workspace_hint = QtWidgets.QLabel()
    window.lbl_workspace_hint.hide()

    window.main_pages = QtWidgets.QStackedWidget()
    content_layout.addWidget(window.main_pages, 1)
    outer.addWidget(content, 1)
    root_layout.addWidget(body, 1)

    window.tool_page = ToolPage(window.session, window.algo, parent=window)
    window.main_pages.addWidget(window.tool_page)

    window.runtime_page = RuntimeModePage()
    window.runtime_page.edit_release_password.setText(window._release_password)
    window.main_pages.addWidget(window.runtime_page)

    window._bottom_status_bar = QtWidgets.QStatusBar()
    window._bottom_status_bar.setSizeGripEnabled(False)
    root_layout.addWidget(window._bottom_status_bar, 0)

    window.lbl_brand_banner = BrandBannerWidget()
    window.lbl_brand_banner.setObjectName("brandBanner")
    window.lbl_brand_banner.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    window.lbl_brand_banner.setFixedHeight(36)
    window.lbl_brand_banner.setStyleSheet(
        "#brandBanner{background:#313131;padding:0px;margin:0px;}"
    )
    root_layout.addWidget(window.lbl_brand_banner, 0)
    window._update_brand_banner_pixmap()
