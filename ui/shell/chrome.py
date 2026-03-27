from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.proxy import is_ready as is_qr_core_ready

from .support import APP_NAME, APP_VERSION, shell_icon


SP = QtWidgets.QStyle.StandardPixmap


def build_menu_bar(window) -> None:
    window.menuBar().hide()

    menu_style = (
        "QMenu{background:#3a3a3a;color:#e0e0e0;border:1px solid #505050;}"
        "QMenu::item{padding:6px 24px;}"
        "QMenu::item:selected{background:#3794ff;}"
        "QMenu::separator{background:#505050;height:1px;margin:4px 8px;}"
    )

    button_style = (
        "QToolButton{color:#e0e0e0;background:transparent;border:none;padding:5px 10px;border-radius:4px;}"
        "QToolButton:hover{background:#4a4a4a;}"
        "QToolButton:pressed{background:#555555;}"
        "QToolButton::menu-indicator{image:none;}"
    )

    def _make_popup_button(
        text: str,
        icon: QtGui.QIcon,
        menu: QtWidgets.QMenu,
        *,
        icon_only: bool = False,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(window)
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setIcon(icon)
        button.setMenu(menu)
        button.setStyleSheet(button_style)
        if icon_only:
            button.setIconSize(QtCore.QSize(18, 18))
            button.setFixedSize(30, 30)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        else:
            button.setText(text)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        return button

    def _make_action_button(
        text: str,
        icon: QtGui.QIcon,
        slot,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(window)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setIcon(icon)
        button.setText(text)
        button.setStyleSheet(button_style)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        button.clicked.connect(slot)
        return button

    top_bar = QtWidgets.QFrame(window)
    top_bar.setFixedHeight(34)
    top_bar.setStyleSheet("QFrame{background:#3a3a3a;border-bottom:1px solid #505050;}")
    top_layout = QtWidgets.QHBoxLayout(top_bar)
    top_layout.setContentsMargins(6, 2, 6, 2)
    top_layout.setSpacing(2)

    file_menu = QtWidgets.QMenu("文件", window)
    file_menu.setStyleSheet(menu_style)
    act_exit = file_menu.addAction(shell_icon(SP.SP_DialogCloseButton), "退出")
    act_exit.triggered.connect(window.close)

    view_menu = QtWidgets.QMenu("视图", window)
    view_menu.setStyleSheet(menu_style)
    window.act_show_debug = view_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), "切换到调试界面"
    )
    window.act_show_debug.triggered.connect(lambda: window._switch_workspace("debug"))
    window.act_show_runtime = view_menu.addAction(
        shell_icon(SP.SP_MediaPlay), "切换到运行界面"
    )
    window.act_show_runtime.triggered.connect(lambda: window._switch_workspace("runtime"))

    tools_menu = QtWidgets.QMenu("工具", window)
    tools_menu.setStyleSheet(menu_style)
    window.act_reload_debug = tools_menu.addAction(
        shell_icon(SP.SP_BrowserReload), "重新加载调试会话"
    )
    window.act_reload_debug.triggered.connect(window._reload_debug_session)
    tools_menu.addSeparator()

    hardware_menu = tools_menu.addMenu(shell_icon(SP.SP_ComputerIcon), "工程调试工具")
    hardware_menu.addAction(
        shell_icon(SP.SP_DesktopIcon), "相机取图工具"
    ).triggered.connect(window.tool_page.open_camera_debug_dialog)
    hardware_menu.addAction(
        shell_icon(SP.SP_DriveNetIcon), "DI / DO 调试工具"
    ).triggered.connect(window.tool_page.open_io_debug_dialog)
    hardware_menu.addAction(
        shell_icon(SP.SP_FileDialogContentsView), "位置修正工具"
    ).triggered.connect(window.tool_page.open_template_editor_dialog)
    hardware_menu.addAction(
        shell_icon(SP.SP_FileDialogListView), "自动区域工具"
    ).triggered.connect(window.tool_page.open_template_match_dialog)

    tools_menu.addSeparator()
    algo_menu = tools_menu.addMenu(shell_icon(SP.SP_FileDialogInfoView), "算法工具")
    algo_menu.addAction(
        shell_icon(SP.SP_DialogApplyButton), "执行 Margin 验证"
    ).triggered.connect(window.tool_page.open_margin_validation_tool)
    algo_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), "打开特征分析"
    ).triggered.connect(window.tool_page.open_embedding_analysis_tool)
    algo_menu.addAction(
        shell_icon(SP.SP_FileDialogListView), "打开传统基线调试"
    ).triggered.connect(window.tool_page.open_baseline_debug_tool)

    runtime_menu = QtWidgets.QMenu("控制", window)
    runtime_menu.setStyleSheet(menu_style)
    runtime_menu.addAction(
        shell_icon(SP.SP_BrowserReload), "刷新相机列表"
    ).triggered.connect(window.runtime_page.refreshCamerasRequested.emit)
    runtime_menu.addAction(
        shell_icon(SP.SP_DriveNetIcon), "连接相机..."
    ).triggered.connect(window._show_connect_dialog)
    runtime_menu.addAction(
        shell_icon(SP.SP_DialogDiscardButton), "断开相机"
    ).triggered.connect(window.runtime_ctrl.disconnect)
    capture_menu = runtime_menu.addMenu(shell_icon(SP.SP_DialogSaveButton), "运行图像保存")
    window.runtime_capture_policy_group = QtGui.QActionGroup(window)
    window.runtime_capture_policy_group.setExclusive(True)
    window.act_runtime_capture_keep_all = capture_menu.addAction("全部保留")
    window.act_runtime_capture_keep_all.setCheckable(True)
    window.runtime_capture_policy_group.addAction(window.act_runtime_capture_keep_all)
    window.act_runtime_capture_keep_all.triggered.connect(
        lambda checked=False: window._apply_runtime_capture_policy(
            "all",
            persist=True,
            show_message=True,
        )
    )
    window.act_runtime_capture_keep_ng_only = capture_menu.addAction("仅保留NG")
    window.act_runtime_capture_keep_ng_only.setCheckable(True)
    window.runtime_capture_policy_group.addAction(window.act_runtime_capture_keep_ng_only)
    window.act_runtime_capture_keep_ng_only.triggered.connect(
        lambda checked=False: window._apply_runtime_capture_policy(
            "ng_only",
            persist=True,
            show_message=True,
        )
    )
    window._sync_runtime_capture_policy_actions()
    runtime_menu.addSeparator()
    runtime_menu.addAction(
        shell_icon(SP.SP_MediaPlay), "脚踏触发"
    ).triggered.connect(window.runtime_page.triggerRequested.emit)
    runtime_menu.addAction(
        shell_icon(SP.SP_DialogYesButton), "密码放行..."
    ).triggered.connect(window._show_release_dialog)
    runtime_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), "修改放行密码..."
    ).triggered.connect(window._show_change_release_password_dialog)
    runtime_menu.addSeparator()
    runtime_menu.addAction(
        shell_icon(SP.SP_BrowserReload), "刷新运行状态"
    ).triggered.connect(
        lambda: window.runtime_ctrl.refresh_all_status("手动刷新运行状态")
    )

    path_menu = QtWidgets.QMenu("路径", window)
    if runtime_menu.actions():
        runtime_menu.removeAction(runtime_menu.actions()[-1])

    path_menu.setStyleSheet(menu_style)
    path_menu.addAction(
        shell_icon(SP.SP_DirHomeIcon), "打开系统根目录"
    ).triggered.connect(window._open_workspace_root)
    path_menu.addAction(
        shell_icon(SP.SP_DirOpenIcon), "打开当前产品目录"
    ).triggered.connect(window._open_current_product_dir)
    path_menu.addAction(
        shell_icon(SP.SP_DirIcon), "打开会话目录"
    ).triggered.connect(window._open_session_dir)
    path_menu.addAction(
        shell_icon(SP.SP_DirLinkIcon), "打开运行记录目录"
    ).triggered.connect(window._open_runtime_records_dir)

    top_layout.addWidget(
        _make_popup_button("文件", shell_icon(SP.SP_DirOpenIcon), file_menu),
        0,
    )
    top_layout.addWidget(
        _make_popup_button("视图", shell_icon(SP.SP_FileDialogDetailedView), view_menu),
        0,
    )
    top_layout.addWidget(
        _make_popup_button("工具", shell_icon(SP.SP_ComputerIcon), tools_menu),
        0,
    )
    top_layout.addWidget(
        _make_popup_button("控制", shell_icon(SP.SP_MediaPlay), runtime_menu),
        0,
    )
    top_layout.addWidget(
        _make_popup_button("路径", shell_icon(SP.SP_DirIcon), path_menu),
        0,
    )
    top_layout.addWidget(
        _make_action_button("帮助", shell_icon(SP.SP_MessageBoxInformation), window._show_about_dialog),
        0,
    )
    top_layout.addStretch(1)

    window.setMenuWidget(top_bar)


def build_status_bar(window) -> None:
    label_style = "color:#888;font-size:11px;"
    io_dot_style = "font-size:14px;font-weight:bold;"
    window.lbl_status_workspace = QtWidgets.QLabel("工作区：调试界面")
    window.lbl_status_workspace.setStyleSheet(label_style)
    window.lbl_status_product = QtWidgets.QLabel(f"产品：{window.session.current_product}")
    window.lbl_status_product.setStyleSheet(label_style)
    window.lbl_status_engine = QtWidgets.QLabel()
    window.lbl_status_engine.setStyleSheet(label_style)
    window.lbl_status_io_dot = QtWidgets.QLabel("●")
    window.lbl_status_io_dot.setStyleSheet(f"color:#c74e39;{io_dot_style}")
    window.lbl_status_io_text = QtWidgets.QLabel("IO: 未初始化")
    window.lbl_status_io_text.setStyleSheet(label_style)
    window.lbl_status_path = QtWidgets.QLabel(f"产品目录：{window.session.product_dir}")
    window.lbl_status_path.setStyleSheet(label_style)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_workspace)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_product)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_engine)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_io_dot)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_io_text)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_path, 1)
    sync_shell_status(window)
    window._set_algorithm_engine_status(
        "算法引擎：已就绪" if is_qr_core_ready() else "算法引擎：加载中..."
    )


def switch_workspace(window, workspace: str) -> None:
    is_runtime = workspace == "runtime"
    window.btn_workspace_debug.setChecked(not is_runtime)
    window.btn_workspace_runtime.setChecked(is_runtime)
    window.main_pages.setCurrentWidget(window.runtime_page if is_runtime else window.tool_page)
    if is_runtime:
        window.lbl_workspace_title.setText("运行界面")
        window.lbl_workspace_hint.setText("实时检测画面与检测项结果")
        window.lbl_status_workspace.setText("工作区：运行界面")
        runtime_message = window._activate_runtime_workspace()
        window.runtime_ctrl.refresh_all_status("已切换到运行界面")
        if runtime_message:
            window._bottom_status_bar.showMessage(runtime_message, 3000)
        return

    window.lbl_workspace_title.setText("调试界面")
    window.lbl_workspace_hint.setText("模板配置、ROI标注、测试与训练")
    window.lbl_status_workspace.setText("工作区：调试界面")


def sync_shell_status(window) -> None:
    window.lbl_status_product.setText(f"产品：{window.session.current_product}")
    window.lbl_status_path.setText(f"产品目录：{window.session.product_dir}")


def update_brand_banner_pixmap(window) -> None:
    if not hasattr(window, "lbl_brand_banner"):
        return
    if window._brand_banner_source.isNull():
        window.lbl_brand_banner.hide()
        return
    window.lbl_brand_banner.show()
    window.lbl_brand_banner.set_source_pixmap(window._brand_banner_source)


def show_about_dialog(window) -> None:
    QtWidgets.QMessageBox.information(
        window,
        f"About {APP_NAME}",
        f"{APP_NAME}\n版本：{APP_VERSION}",
    )
