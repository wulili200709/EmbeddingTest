from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.lazy_api import is_ready as is_qr_core_ready
from ui.i18n import LANG_EN, LANG_ZH, language_code, set_language, tr

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

    file_menu = QtWidgets.QMenu(tr("menu.file"), window)
    file_menu.setStyleSheet(menu_style)
    act_exit = file_menu.addAction(shell_icon(SP.SP_DialogCloseButton), tr("action.exit"))
    act_exit.triggered.connect(window.close)

    view_menu = QtWidgets.QMenu(tr("menu.view"), window)
    view_menu.setStyleSheet(menu_style)
    window.act_show_debug = view_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), tr("action.switch_debug")
    )
    window.act_show_debug.triggered.connect(lambda: window._switch_workspace("debug"))
    window.act_show_runtime = view_menu.addAction(
        shell_icon(SP.SP_MediaPlay), tr("action.switch_runtime")
    )
    window.act_show_runtime.triggered.connect(lambda: window._switch_workspace("runtime"))

    tools_menu = QtWidgets.QMenu(tr("menu.tools"), window)
    tools_menu.setStyleSheet(menu_style)
    window.act_reload_debug = tools_menu.addAction(
        shell_icon(SP.SP_BrowserReload), tr("action.reload_debug")
    )
    window.act_reload_debug.triggered.connect(window._reload_debug_session)
    tools_menu.addSeparator()

    hardware_menu = tools_menu.addMenu(shell_icon(SP.SP_ComputerIcon), tr("menu.engineering_tools"))
    act_camera_tool = hardware_menu.addAction(
        shell_icon(SP.SP_DesktopIcon), tr("action.camera_tool")
    )
    act_camera_tool.triggered.connect(window.tool_page.open_camera_debug_dialog)
    act_io_tool = hardware_menu.addAction(
        shell_icon(SP.SP_DriveNetIcon), tr("action.io_tool")
    )
    act_io_tool.triggered.connect(window.tool_page.open_io_debug_dialog)
    act_template_editor = hardware_menu.addAction(
        shell_icon(SP.SP_FileDialogContentsView), tr("action.template_editor")
    )
    act_template_editor.triggered.connect(window.tool_page.open_template_editor_dialog)
    act_ncc_tool = hardware_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), tr("action.ncc_tool")
    )
    act_ncc_tool.triggered.connect(window.tool_page.open_ncc_match_dialog)
    act_auto_region = hardware_menu.addAction(
        shell_icon(SP.SP_FileDialogListView), tr("action.auto_region")
    )
    act_auto_region.triggered.connect(window.tool_page.open_template_match_dialog)

    tools_menu.addSeparator()
    algo_menu = tools_menu.addMenu(shell_icon(SP.SP_FileDialogInfoView), tr("menu.algorithm_tools"))
    act_margin_validation = algo_menu.addAction(
        shell_icon(SP.SP_DialogApplyButton), tr("action.margin_validation")
    ).triggered.connect(window.tool_page.open_margin_validation_tool)
    act_embedding_analysis = algo_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), tr("action.embedding_analysis")
    ).triggered.connect(window.tool_page.open_embedding_analysis_tool)
    act_baseline_debug = algo_menu.addAction(
        shell_icon(SP.SP_FileDialogListView), tr("action.baseline_debug")
    ).triggered.connect(window.tool_page.open_baseline_debug_tool)

    act_current_product_capture_plan = tools_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView),
        tr("action.current_product_capture_plan"),
    )
    act_current_product_capture_plan.triggered.connect(
        window.tool_page.open_current_product_capture_plan_dialog
    )

    runtime_menu = QtWidgets.QMenu(tr("menu.control"), window)
    runtime_menu.setStyleSheet(menu_style)
    act_refresh_cameras = runtime_menu.addAction(
        shell_icon(SP.SP_BrowserReload), tr("action.refresh_cameras")
    ).triggered.connect(window.runtime_page.refreshCamerasRequested.emit)
    act_open_mvs = runtime_menu.addAction(
        shell_icon(SP.SP_ComputerIcon), tr("action.open_mvs")
    ).triggered.connect(window._open_mvs)
    act_connect_camera = runtime_menu.addAction(
        shell_icon(SP.SP_DriveNetIcon), tr("action.connect_camera")
    ).triggered.connect(window._show_connect_dialog)
    act_disconnect_camera = runtime_menu.addAction(
        shell_icon(SP.SP_DialogDiscardButton), tr("action.disconnect_camera")
    ).triggered.connect(window._disconnect_runtime_cameras)
    capture_menu = runtime_menu.addMenu(shell_icon(SP.SP_DialogSaveButton), tr("menu.runtime_capture"))
    window.runtime_capture_policy_group = QtGui.QActionGroup(window)
    window.runtime_capture_policy_group.setExclusive(True)
    window.act_runtime_capture_keep_all = capture_menu.addAction(tr("action.keep_all"))
    window.act_runtime_capture_keep_all.setCheckable(True)
    window.runtime_capture_policy_group.addAction(window.act_runtime_capture_keep_all)
    window.act_runtime_capture_keep_all.triggered.connect(
        lambda checked=False: window._apply_runtime_capture_policy(
            "all",
            persist=True,
            show_message=True,
        )
    )
    window.act_runtime_capture_keep_ng_only = capture_menu.addAction(tr("action.keep_ng_only"))
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
    act_tower_light = runtime_menu.addAction(
        shell_icon(SP.SP_FileDialogInfoView), tr("action.tower_light")
    ).triggered.connect(window._show_tower_light_settings_dialog)
    runtime_menu.addSeparator()
    act_foot_trigger = runtime_menu.addAction(
        shell_icon(SP.SP_MediaPlay), tr("action.foot_trigger")
    ).triggered.connect(window.runtime_page.triggerRequested.emit)
    act_password_release = runtime_menu.addAction(
        shell_icon(SP.SP_DialogYesButton), tr("action.password_release")
    ).triggered.connect(window._show_release_dialog)
    act_change_release_password = runtime_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), tr("action.change_release_password")
    ).triggered.connect(window._show_change_release_password_dialog)
    runtime_menu.addSeparator()
    act_refresh_runtime = runtime_menu.addAction(
        shell_icon(SP.SP_BrowserReload), tr("action.refresh_runtime")
    ).triggered.connect(
        lambda: window.runtime_ctrl.refresh_all_status(tr("action.refresh_runtime"))
    )

    path_menu = QtWidgets.QMenu(tr("menu.path"), window)
    if runtime_menu.actions():
        runtime_menu.removeAction(runtime_menu.actions()[-1])

    path_menu.setStyleSheet(menu_style)
    act_open_system_root = path_menu.addAction(
        shell_icon(SP.SP_DirHomeIcon), tr("action.open_system_root")
    ).triggered.connect(window._open_workspace_root)
    act_open_product_dir = path_menu.addAction(
        shell_icon(SP.SP_DirOpenIcon), tr("action.open_product_dir")
    ).triggered.connect(window._open_current_product_dir)
    act_open_session_dir = path_menu.addAction(
        shell_icon(SP.SP_DirIcon), tr("action.open_session_dir")
    ).triggered.connect(window._open_session_dir)
    act_open_runtime_images = path_menu.addAction(
        shell_icon(SP.SP_DirLinkIcon), tr("action.open_runtime_images")
    ).triggered.connect(window._open_runtime_capture_dir)
    act_save_image_path = path_menu.addAction(
        shell_icon(SP.SP_DialogSaveButton), tr("action.save_image_path")
    ).triggered.connect(window._show_runtime_capture_directory_dialog)
    path_menu.addSeparator()
    act_open_runtime_records = path_menu.addAction(
        shell_icon(SP.SP_DirLinkIcon), tr("action.open_runtime_records")
    ).triggered.connect(window._open_runtime_records_dir)
    act_save_runtime_records = path_menu.addAction(
        shell_icon(SP.SP_DialogSaveButton), tr("action.save_runtime_records")
    ).triggered.connect(window._show_runtime_records_directory_dialog)

    system_menu = QtWidgets.QMenu(tr("menu.system"), window)
    system_menu.setStyleSheet(menu_style)
    window.act_auth_login = system_menu.addAction(shell_icon(SP.SP_DialogApplyButton), tr("action.auth_login"))
    window.act_auth_login.triggered.connect(window._show_login_dialog)
    window.act_auth_logout = system_menu.addAction(shell_icon(SP.SP_DialogCloseButton), tr("action.auth_logout"))
    window.act_auth_logout.triggered.connect(window._logout_current_user)
    window.act_change_current_password = system_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), tr("action.change_current_password")
    )
    window.act_change_current_password.triggered.connect(
        lambda checked=False: window._show_change_current_user_password()
    )
    system_menu.addSeparator()
    window.act_user_permissions = system_menu.addAction(
        shell_icon(SP.SP_FileDialogDetailedView), tr("action.user_permissions")
    )
    window.act_user_permissions.triggered.connect(window._show_user_permission_dialog)
    window.act_audit_log = system_menu.addAction(shell_icon(SP.SP_FileDialogListView), tr("action.audit_log"))
    window.act_audit_log.triggered.connect(window._show_audit_log_dialog)
    window.act_runtime_records_query = system_menu.addAction(
        shell_icon(SP.SP_FileDialogListView), tr("action.runtime_records_query")
    )
    window.act_runtime_records_query.triggered.connect(window._show_runtime_records_dialog)
    window.act_software_versions = system_menu.addAction(
        shell_icon(SP.SP_FileDialogInfoView), tr("action.software_versions")
    )
    window.act_software_versions.triggered.connect(window._show_software_version_dialog)

    language_menu = QtWidgets.QMenu(tr("menu.language"), window)
    language_menu.setStyleSheet(menu_style)
    language_group = QtGui.QActionGroup(window)
    language_group.setExclusive(True)
    window.act_language_zh = language_menu.addAction(tr("language.zh"))
    window.act_language_zh.setCheckable(True)
    window.act_language_en = language_menu.addAction(tr("language.en"))
    window.act_language_en.setCheckable(True)
    language_group.addAction(window.act_language_zh)
    language_group.addAction(window.act_language_en)
    window.act_language_zh.setChecked(language_code() == LANG_ZH)
    window.act_language_en.setChecked(language_code() == LANG_EN)
    window.act_language_zh.triggered.connect(lambda checked=False: window._change_language(LANG_ZH))
    window.act_language_en.triggered.connect(lambda checked=False: window._change_language(LANG_EN))

    help_menu = QtWidgets.QMenu(tr("menu.help"), window)
    help_menu.setStyleSheet(menu_style)
    act_about_version = help_menu.addAction(
        shell_icon(SP.SP_MessageBoxInformation), tr("software.version")
    )
    act_about_version.triggered.connect(window._show_about_dialog)
    window.act_detailed_camera_diagnostics = help_menu.addAction(
        shell_icon(SP.SP_FileDialogInfoView),
        tr("action.detailed_camera_diagnostics"),
    )
    window.act_detailed_camera_diagnostics.setCheckable(True)
    window.act_detailed_camera_diagnostics.triggered.connect(
        window._toggle_detailed_camera_diagnostics
    )

    top_layout.addWidget(
        _make_popup_button(tr("menu.file"), shell_icon(SP.SP_DirOpenIcon), file_menu),
        0,
    )
    window.btn_menu_file = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.view"), shell_icon(SP.SP_FileDialogDetailedView), view_menu),
        0,
    )
    window.btn_menu_view = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.tools"), shell_icon(SP.SP_ComputerIcon), tools_menu),
        0,
    )
    window.btn_menu_tools = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.control"), shell_icon(SP.SP_MediaPlay), runtime_menu),
        0,
    )
    window.btn_menu_control = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.path"), shell_icon(SP.SP_DirIcon), path_menu),
        0,
    )
    window.btn_menu_path = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.system"), shell_icon(SP.SP_FileDialogInfoView), system_menu),
        0,
    )
    window.btn_menu_system = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.language"), shell_icon(SP.SP_FileDialogInfoView), language_menu),
        0,
    )
    window.btn_menu_language = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addWidget(
        _make_popup_button(tr("menu.help"), shell_icon(SP.SP_MessageBoxInformation), help_menu),
        0,
    )
    window.btn_menu_help = top_layout.itemAt(top_layout.count() - 1).widget()
    top_layout.addStretch(1)

    window._shell_i18n_refs = {
        "menus": {
            "file": file_menu,
            "view": view_menu,
            "tools": tools_menu,
            "hardware": hardware_menu,
            "algo": algo_menu,
            "runtime": runtime_menu,
            "capture": capture_menu,
            "path": path_menu,
            "system": system_menu,
            "language": language_menu,
            "help": help_menu,
        },
        "actions": {
            "exit": act_exit,
            "camera_tool": act_camera_tool,
            "io_tool": act_io_tool,
            "template_editor": act_template_editor,
            "ncc_tool": act_ncc_tool,
            "auto_region": act_auto_region,
            "margin_validation": algo_menu.actions()[0],
            "embedding_analysis": algo_menu.actions()[1],
            "baseline_debug": algo_menu.actions()[2],
            "current_product_capture_plan": act_current_product_capture_plan,
            "refresh_cameras": runtime_menu.actions()[0],
            "open_mvs": runtime_menu.actions()[1],
            "connect_camera": runtime_menu.actions()[2],
            "disconnect_camera": runtime_menu.actions()[3],
            "tower_light": runtime_menu.actions()[5],
            "foot_trigger": runtime_menu.actions()[7],
            "password_release": runtime_menu.actions()[8],
            "change_release_password": runtime_menu.actions()[9],
            "open_system_root": path_menu.actions()[0],
            "open_product_dir": path_menu.actions()[1],
            "open_session_dir": path_menu.actions()[2],
            "open_runtime_images": path_menu.actions()[3],
            "save_image_path": path_menu.actions()[4],
            "open_runtime_records": path_menu.actions()[6],
            "save_runtime_records": path_menu.actions()[7],
            "auth_login": window.act_auth_login,
            "auth_logout": window.act_auth_logout,
            "change_current_password": window.act_change_current_password,
            "user_permissions": window.act_user_permissions,
            "audit_log": window.act_audit_log,
            "runtime_records_query": window.act_runtime_records_query,
            "about_version": act_about_version,
            "detailed_camera_diagnostics": window.act_detailed_camera_diagnostics,
            "software_versions": window.act_software_versions,
        },
        "buttons": {
            "file": window.btn_menu_file,
            "view": window.btn_menu_view,
            "tools": window.btn_menu_tools,
            "control": window.btn_menu_control,
            "path": window.btn_menu_path,
            "system": window.btn_menu_system,
            "language": window.btn_menu_language,
            "help": window.btn_menu_help,
        },
    }
    window.setMenuWidget(top_bar)


def build_status_bar(window) -> None:
    label_style = "color:#888;font-size:11px;padding:0 6px;"
    io_dot_style = "font-size:14px;font-weight:bold;"
    def _make_separator() -> QtWidgets.QLabel:
        separator = QtWidgets.QLabel("|")
        separator.setStyleSheet("color:#555;font-size:11px;padding:0 2px;")
        return separator

    window.lbl_status_workspace = QtWidgets.QLabel(tr("status.workspace", workspace=tr("workspace.debug")))
    window.lbl_status_workspace.setStyleSheet(label_style)
    window.lbl_status_product = QtWidgets.QLabel(tr("status.product", product=window.session.current_product))
    window.lbl_status_product.setStyleSheet(label_style)
    window.lbl_status_engine = QtWidgets.QLabel()
    window.lbl_status_engine.setStyleSheet(label_style)
    window.lbl_status_io_dot = QtWidgets.QLabel("●")
    window.lbl_status_io_dot.setStyleSheet(f"color:#c74e39;{io_dot_style}")
    window.lbl_status_io_text = QtWidgets.QLabel(tr("status.io_uninitialized"))
    window.lbl_status_io_text.setStyleSheet(label_style)
    window.lbl_status_path = QtWidgets.QLabel(tr("status.product_dir", path=window.session.product_dir))
    window.lbl_status_path.setStyleSheet(label_style)
    window.lbl_status_path.setMinimumWidth(260)
    window.lbl_status_user = QtWidgets.QLabel("")
    window.lbl_status_user.setStyleSheet(label_style)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_workspace)
    window._bottom_status_bar.addPermanentWidget(_make_separator())
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_product)
    window._bottom_status_bar.addPermanentWidget(_make_separator())
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_engine)
    window._bottom_status_bar.addPermanentWidget(_make_separator())
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_io_dot)
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_io_text)
    window._bottom_status_bar.addPermanentWidget(_make_separator())
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_user)
    window._bottom_status_bar.addPermanentWidget(_make_separator())
    window._bottom_status_bar.addPermanentWidget(window.lbl_status_path, 1)
    sync_shell_status(window)
    window._set_algorithm_engine_status_key(
        "status.engine_ready" if is_qr_core_ready() else "status.engine_loading"
    )


def switch_workspace(window, workspace: str) -> None:
    is_runtime = workspace == "runtime"
    window.btn_workspace_debug.setChecked(not is_runtime)
    window.btn_workspace_runtime.setChecked(is_runtime)
    window.main_pages.setCurrentWidget(window.runtime_page if is_runtime else window.tool_page)
    if is_runtime:
        window.lbl_workspace_title.setText(tr("workspace.runtime"))
        window.lbl_workspace_hint.setText(tr("workspace.runtime_hint"))
        window.lbl_status_workspace.setText(tr("status.workspace", workspace=tr("workspace.runtime")))
        runtime_message = window._activate_runtime_workspace()
        window.runtime_ctrl.refresh_all_status(tr("status.runtime_switched"))
        if runtime_message:
            window._bottom_status_bar.showMessage(runtime_message, 3000)
        return

    window.lbl_workspace_title.setText(tr("workspace.debug"))
    window.lbl_workspace_hint.setText(tr("workspace.debug_hint"))
    window.lbl_status_workspace.setText(tr("status.workspace", workspace=tr("workspace.debug")))


def sync_shell_status(window) -> None:
    window.lbl_status_product.setText(tr("status.product", product=window.session.current_product))
    product_dir = str(window.session.product_dir or "").strip()
    if product_dir:
        path_obj = Path(product_dir)
        parts = list(path_obj.parts)
        if len(parts) > 4:
            compact = "\\".join(parts[-4:])
            display_text = tr("status.product_dir", path=f"...\\{compact}")
        else:
            display_text = tr("status.product_dir", path=product_dir)
    else:
        display_text = tr("status.product_dir_empty")
    window.lbl_status_path.setText(display_text)
    window.lbl_status_path.setToolTip(product_dir)


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
        tr("dialog.about_title"),
        tr("dialog.about_body"),
    )


def retranslate_shell_chrome(window) -> None:
    refs = getattr(window, "_shell_i18n_refs", None) or {}
    menus = refs.get("menus", {})
    actions = refs.get("actions", {})
    buttons = refs.get("buttons", {})

    menu_keys = {
        "file": "menu.file",
        "view": "menu.view",
        "tools": "menu.tools",
        "hardware": "menu.engineering_tools",
        "algo": "menu.algorithm_tools",
        "runtime": "menu.control",
        "capture": "menu.runtime_capture",
        "path": "menu.path",
        "system": "menu.system",
        "language": "menu.language",
        "help": "menu.help",
    }
    for name, key in menu_keys.items():
        menu = menus.get(name)
        if menu is not None:
            menu.setTitle(tr(key) if "." in key else key)

    button_keys = {
        "file": "menu.file",
        "view": "menu.view",
        "tools": "menu.tools",
        "control": "menu.control",
        "path": "menu.path",
        "system": "menu.system",
        "language": "menu.language",
        "help": "menu.help",
    }
    for name, key in button_keys.items():
        button = buttons.get(name)
        if button is not None:
            button.setText(tr(key) if "." in key else key)

    action_keys = {
        "exit": "action.exit",
        "camera_tool": "action.camera_tool",
        "io_tool": "action.io_tool",
        "template_editor": "action.template_editor",
        "ncc_tool": "action.ncc_tool",
        "auto_region": "action.auto_region",
        "margin_validation": "action.margin_validation",
        "embedding_analysis": "action.embedding_analysis",
        "baseline_debug": "action.baseline_debug",
        "current_product_capture_plan": "action.current_product_capture_plan",
        "refresh_cameras": "action.refresh_cameras",
        "open_mvs": "action.open_mvs",
        "connect_camera": "action.connect_camera",
        "disconnect_camera": "action.disconnect_camera",
        "tower_light": "action.tower_light",
        "foot_trigger": "action.foot_trigger",
        "password_release": "action.password_release",
        "change_release_password": "action.change_release_password",
        "open_system_root": "action.open_system_root",
        "open_product_dir": "action.open_product_dir",
        "open_session_dir": "action.open_session_dir",
        "open_runtime_images": "action.open_runtime_images",
        "save_image_path": "action.save_image_path",
        "open_runtime_records": "action.open_runtime_records",
        "save_runtime_records": "action.save_runtime_records",
        "auth_login": "action.auth_login",
        "auth_logout": "action.auth_logout",
        "change_current_password": "action.change_current_password",
        "user_permissions": "action.user_permissions",
        "audit_log": "action.audit_log",
        "runtime_records_query": "action.runtime_records_query",
        "about_version": "software.version",
        "detailed_camera_diagnostics": "action.detailed_camera_diagnostics",
        "software_versions": "action.software_versions",
    }
    for name, key in action_keys.items():
        action = actions.get(name)
        if action is not None:
            action.setText(tr(key) if "." in key else key)

    if hasattr(window, "act_show_debug"):
        window.act_show_debug.setText(tr("action.switch_debug"))
    if hasattr(window, "act_show_runtime"):
        window.act_show_runtime.setText(tr("action.switch_runtime"))
    if hasattr(window, "act_reload_debug"):
        window.act_reload_debug.setText(tr("action.reload_debug"))
    if hasattr(window, "act_runtime_capture_keep_all"):
        window.act_runtime_capture_keep_all.setText(tr("action.keep_all"))
    if hasattr(window, "act_runtime_capture_keep_ng_only"):
        window.act_runtime_capture_keep_ng_only.setText(tr("action.keep_ng_only"))
    if hasattr(window, "act_language_zh"):
        window.act_language_zh.setText(tr("language.zh"))
        window.act_language_zh.setChecked(language_code() == LANG_ZH)
    if hasattr(window, "act_language_en"):
        window.act_language_en.setText(tr("language.en"))
        window.act_language_en.setChecked(language_code() == LANG_EN)

    if hasattr(window, "btn_workspace_debug"):
        window.btn_workspace_debug.setText(tr("workspace.debug"))
    if hasattr(window, "btn_workspace_runtime"):
        window.btn_workspace_runtime.setText(tr("workspace.runtime"))
    if hasattr(window, "lbl_sidebar_runtime_result_title"):
        window.lbl_sidebar_runtime_result_title.setText(tr("sidebar.final_result"))

    is_runtime = hasattr(window, "runtime_page") and window.main_pages.currentWidget() == window.runtime_page
    window.btn_workspace_debug.setChecked(not is_runtime)
    window.btn_workspace_runtime.setChecked(is_runtime)
    if is_runtime:
        window.lbl_workspace_title.setText(tr("workspace.runtime"))
        window.lbl_workspace_hint.setText(tr("workspace.runtime_hint"))
        window.lbl_status_workspace.setText(tr("status.workspace", workspace=tr("workspace.runtime")))
    else:
        window.lbl_workspace_title.setText(tr("workspace.debug"))
        window.lbl_workspace_hint.setText(tr("workspace.debug_hint"))
        window.lbl_status_workspace.setText(tr("status.workspace", workspace=tr("workspace.debug")))
    sync_shell_status(window)
    status_key = getattr(window, "_algorithm_engine_status_key", "") or (
        "status.engine_ready" if is_qr_core_ready() else "status.engine_loading"
    )
    window._set_algorithm_engine_status_key(status_key)
