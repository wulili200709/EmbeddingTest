"""
qr_gui_pyside6.py

MainWindow — Step 4 重构后的极简薄壳。

职责：
  ① 创建 ProductSession / AlgorithmController / ToolPage / RuntimeController
  ② 连接所有跨边界 Signal（RuntimeModePage ↔ RuntimeController ↔ ToolPage）
  ③ 弹出对话框（RuntimeController 只发 Signal，由此处响应）
  ④ 处理产品切换 / 会话清空的跨组件协调
  ⑤ 窗口生命周期（closeEvent）

业务逻辑完全委托给：
  - ProductSession      — 产品 / 路径 / session.json
  - AlgorithmController — 算法参数 / 模型 / 训练 / 推理
  - ToolPage            — ROI 标注 / 自动定位 / 预测 / 分析
  - RuntimeController   — 相机连接 / 触发检测 / 放行
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from qr_core_proxy import is_ready as is_qr_core_ready, preload as preload_qr_core

from application import (
    DEFAULT_RELEASE_PASSWORD,
    RuntimeController,
    ToolPageRuntimeContext,
)
from ui.debug import ToolPage
from ui.runtime import RuntimeModePage
from ui.window_common import (
    build_default_session_and_algo,
    connect_runtime_dialogs,
    connect_runtime_page,
    connect_runtime_refresh_sources,
    detect_runtime_import_error,
    embedding_test_root,
    update_runtime_preview,
)


_RUNTIME_IMPORT_ERROR = detect_runtime_import_error()
_DEFAULT_ADMIN_PASSWORD = "admin123"
_SYSTEM_PASSWORDS_FILENAME = "system_passwords.json"
_APP_NAME = "LC System"
_WINDOWS_APP_ID = "LCSystem.App"


def _resource_path(filename: str) -> Path:
    return embedding_test_root(__file__) / "res" / filename


def _load_app_version() -> str:
    setup_path = embedding_test_root(__file__).parent / "setup.py"
    try:
        text = setup_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "dev"
    match = re.search(r'version\s*=\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip() or "dev"
    return "dev"


_APP_VERSION = _load_app_version()


def _set_windows_app_id(app_id: str) -> None:
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def _icon(sp: QtWidgets.QStyle.StandardPixmap) -> QtGui.QIcon:
    return QtWidgets.QApplication.style().standardIcon(sp)


SP = QtWidgets.QStyle.StandardPixmap


class _AlgorithmEngineWarmupThread(QtCore.QThread):
    warmupFinished = QtCore.Signal(bool, str)

    def run(self) -> None:
        try:
            preload_qr_core()
        except Exception as exc:
            self.warmupFinished.emit(False, str(exc))
            return
        self.warmupFinished.emit(True, "")


class _BrandBannerWidget(QtWidgets.QWidget):
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


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(_APP_NAME)
        logo_path = _resource_path("logo.png")
        if logo_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(logo_path)))
        self.setStyleSheet(
            "QMainWindow{background:#2d2d2d;}"
            "QMenuBar{background:#3a3a3a;color:#e0e0e0;border-bottom:1px solid #505050;}"
            "QMenuBar::item:selected{background:#4a4a4a;}"
            "QMenu{background:#3a3a3a;color:#e0e0e0;border:1px solid #505050;}"
            "QMenu::item:selected{background:#3794ff;}"
            "QStatusBar{background:#3a3a3a;color:#aaa;border-top:1px solid #505050;}"
        )

        self.session, self.algo = build_default_session_and_algo(__file__)
        self._password_settings = self._load_password_settings()
        self._release_password = self._password_settings["run_password"]
        self._admin_password = self._password_settings["engineer_password"]
        self._engine_warmup_thread: Optional[_AlgorithmEngineWarmupThread] = None
        self._brand_banner_source = QtGui.QPixmap(str(_resource_path("logo2.png")))
        self._startup_runtime_auto_connect_done = False

        # ── UI 组装 ────────────────────────────────────────────────────
        self._build_ui()

        # ── 运行控制器（需要 tool_page 已创建） ─────────────────────────
        self.runtime_ctrl = RuntimeController(
            session=self.session,
            algo=self.algo,
            runtime_context=ToolPageRuntimeContext(self.tool_page),
            import_error=_RUNTIME_IMPORT_ERROR,
            release_password=self._release_password,
            parent=self,
        )

        # ── 信号连接 ───────────────────────────────────────────────────
        self._connect_signals()
        self._build_menu_bar()
        self._build_status_bar()

        # ── 初始化加载 ─────────────────────────────────────────────────
        self.tool_page.load_session()          # 发射 sessionLoaded → refresh_all_status
        self.runtime_ctrl.refresh_all_status() # 初始状态推送
        self._switch_workspace("debug")
        self._switch_workspace("debug")
        QtCore.QTimer.singleShot(0, self.runtime_ctrl.reset_all_camera_triggers_off)
        QtCore.QTimer.singleShot(0, self._start_algorithm_engine_warmup)
        QtCore.QTimer.singleShot(150, self._startup_auto_connect_runtime_cameras)

    def _on_camera_settings_applied(self, serial: str, settings_payload) -> None:
        self.runtime_ctrl.apply_camera_settings_for_serial(serial, settings_payload)

    def _prepare_runtime_for_debug_camera(self, serial: str) -> None:
        serial_text = str(serial).strip()
        if not serial_text:
            return
        if not self.runtime_ctrl.connected_roles():
            return
        if serial_text not in set(self.runtime_page.camera_bindings().values()):
            return
        self.runtime_ctrl.disconnect(silent=True)
        self._bottom_status_bar.showMessage("已释放运行相机，切换到调试连接", 3000)

    def _restore_runtime_camera_bindings_from_session(self) -> None:
        session_data = self.session.load_session()
        self.runtime_page.edit_cam1_serial.setText(session_data.runtime_cam1_serial or "")
        self.runtime_page.edit_cam2_serial.setText(session_data.runtime_cam2_serial or "")

    def _persist_runtime_camera_bindings(self, bindings: Optional[dict[str, str]] = None) -> None:
        session_data = self.session.load_session()
        current_bindings = dict(bindings or self.runtime_page.camera_bindings())
        session_data.runtime_cam1_serial = str(current_bindings.get("cam1", "")).strip()
        session_data.runtime_cam2_serial = str(current_bindings.get("cam2", "")).strip()
        self.session.save_session(session_data)

    def _startup_auto_connect_runtime_cameras(self) -> None:
        if self._startup_runtime_auto_connect_done:
            return
        self._startup_runtime_auto_connect_done = True
        if _RUNTIME_IMPORT_ERROR is not None:
            return
        if self.runtime_ctrl.connected_roles():
            return

        bindings = self.runtime_page.camera_bindings()
        if not bindings:
            return

        if self.runtime_ctrl.try_connect_cameras(bindings):
            self._bottom_status_bar.showMessage("已自动连接运行相机", 3000)
            return

        QtCore.QTimer.singleShot(0, self._show_connect_dialog)

    @QtCore.Slot(list)
    def _on_runtime_active_roles_changed(self, roles: list[str]) -> None:
        if not roles:
            return
        self._persist_runtime_camera_bindings()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root_layout = QtWidgets.QVBoxLayout(cw)
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

        self.btn_workspace_debug = QtWidgets.QPushButton(
            _icon(SP.SP_FileDialogDetailedView), "调试界面"
        )
        self.btn_workspace_debug.setCheckable(True)
        self.btn_workspace_debug.clicked.connect(lambda: self._switch_workspace("debug"))
        nav_layout.addWidget(self.btn_workspace_debug)

        self.btn_workspace_runtime = QtWidgets.QPushButton(
            _icon(SP.SP_MediaPlay), "运行界面"
        )
        self.btn_workspace_runtime.setCheckable(True)
        self.btn_workspace_runtime.clicked.connect(lambda: self._switch_workspace("runtime"))
        nav_layout.addWidget(self.btn_workspace_runtime)
        nav_layout.addStretch(1)
        outer.addWidget(nav_frame, 0)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.lbl_workspace_title = QtWidgets.QLabel()
        self.lbl_workspace_title.hide()
        self.lbl_workspace_hint = QtWidgets.QLabel()
        self.lbl_workspace_hint.hide()

        self.main_pages = QtWidgets.QStackedWidget()
        content_layout.addWidget(self.main_pages, 1)
        outer.addWidget(content, 1)
        root_layout.addWidget(body, 1)

        self.tool_page = ToolPage(self.session, self.algo, parent=self)
        self.main_pages.addWidget(self.tool_page)

        self.runtime_page = RuntimeModePage()
        self.runtime_page.edit_release_password.setText(self._release_password)
        self.main_pages.addWidget(self.runtime_page)

        self._bottom_status_bar = QtWidgets.QStatusBar()
        self._bottom_status_bar.setSizeGripEnabled(False)
        root_layout.addWidget(self._bottom_status_bar, 0)

        self.lbl_brand_banner = _BrandBannerWidget()
        self.lbl_brand_banner.setObjectName("brandBanner")
        self.lbl_brand_banner.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.lbl_brand_banner.setFixedHeight(36)
        self.lbl_brand_banner.setStyleSheet(
            "#brandBanner{background:#313131;padding:0px;margin:0px;}"
        )
        root_layout.addWidget(self.lbl_brand_banner, 0)
        self._update_brand_banner_pixmap()

    def _password_settings_path(self) -> Path:
        config_dir = embedding_test_root(__file__) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / _SYSTEM_PASSWORDS_FILENAME

    def _default_password_settings(self) -> dict[str, str]:
        return {
            "run_password": DEFAULT_RELEASE_PASSWORD,
            "engineer_password": _DEFAULT_ADMIN_PASSWORD,
        }

    def _load_password_settings(self) -> dict[str, str]:
        settings = self._default_password_settings()
        path = self._password_settings_path()
        raw: dict = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                raw = {}

        if isinstance(raw, dict):
            run_password = str(raw.get("run_password", "")).strip()
            engineer_password = str(raw.get("engineer_password", "")).strip()
            if run_password:
                settings["run_password"] = run_password
            if engineer_password:
                settings["engineer_password"] = engineer_password

        try:
            self._save_password_settings(settings)
        except Exception:
            pass
        return settings

    def _save_password_settings(self, settings: dict[str, str]) -> None:
        path = self._password_settings_path()
        payload = {
            "run_password": str(settings.get("run_password", DEFAULT_RELEASE_PASSWORD)).strip()
            or DEFAULT_RELEASE_PASSWORD,
            "engineer_password": str(settings.get("engineer_password", _DEFAULT_ADMIN_PASSWORD)).strip()
            or _DEFAULT_ADMIN_PASSWORD,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _apply_release_password(self, password: str) -> bool:
        password_text = str(password).strip()
        if len(password_text) < 4:
            QtWidgets.QMessageBox.warning(self, "修改放行密码", "新密码至少需要 4 位。")
            return False

        settings = dict(self._password_settings)
        settings["run_password"] = password_text
        try:
            self._save_password_settings(settings)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "修改放行密码",
                f"保存密码失败：\n{exc}",
            )
            return False

        self._password_settings = settings
        self._release_password = password_text
        self.runtime_page.edit_release_password.setText(password_text)
        self.runtime_ctrl.update_release_password(password_text)
        self._bottom_status_bar.showMessage("放行密码已更新", 3000)
        return True

    def _confirm_admin_password(self) -> bool:
        admin_password, ok = QtWidgets.QInputDialog.getText(
            self,
            "管理员验证",
            "输入管理员密码：",
            QtWidgets.QLineEdit.Password,
        )
        if not ok:
            return False
        if str(admin_password) != self._admin_password:
            QtWidgets.QMessageBox.warning(self, "管理员验证", "管理员密码错误。")
            return False
        return True

    def _show_change_release_password_dialog(self) -> None:
        if not self._confirm_admin_password():
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("修改放行密码")
        dialog.setMinimumWidth(360)
        layout = QtWidgets.QFormLayout(dialog)

        label_tip = QtWidgets.QLabel("仅管理员可修改 NG 放行密码。")
        label_tip.setStyleSheet("color:#666;")
        layout.addRow(label_tip)

        edit_new_password = QtWidgets.QLineEdit()
        edit_new_password.setEchoMode(QtWidgets.QLineEdit.Password)
        edit_new_password.setPlaceholderText("输入新的放行密码")
        edit_confirm_password = QtWidgets.QLineEdit()
        edit_confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        edit_confirm_password.setPlaceholderText("再次输入新密码")
        layout.addRow("新密码", edit_new_password)
        layout.addRow("确认密码", edit_confirm_password)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        while True:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            new_password = edit_new_password.text().strip()
            confirm_password = edit_confirm_password.text().strip()
            if not new_password:
                QtWidgets.QMessageBox.warning(self, "修改放行密码", "新密码不能为空。")
                continue
            if new_password != confirm_password:
                QtWidgets.QMessageBox.warning(self, "修改放行密码", "两次输入的新密码不一致。")
                continue
            if not self._apply_release_password(new_password):
                continue
            QtWidgets.QMessageBox.information(self, "修改放行密码", "放行密码已更新。")
            return

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # ToolPage → MainWindow（跨组件协调）
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(self._sync_shell_status)
        self.tool_page.sessionLoaded.connect(self._restore_runtime_camera_bindings_from_session)
        connect_runtime_refresh_sources(
            self.tool_page,
            self.runtime_ctrl,
            session_loaded_message="工具页会话已加载",
        )
        connect_runtime_page(self.runtime_page, self.runtime_ctrl)
        self.tool_page.debugCameraConnectRequested.connect(self._prepare_runtime_for_debug_camera)
        self.tool_page.debugCameraConnected.connect(self._on_debug_camera_connected)
        self.tool_page.cameraSettingsApplied.connect(self._on_camera_settings_applied)
        self.runtime_ctrl.previewUpdated.connect(self._on_runtime_preview_updated)
        self.runtime_ctrl.productNameChanged.connect(lambda *_: self._sync_shell_status())
        self.runtime_ctrl.activeCameraRolesChanged.connect(self._on_runtime_active_roles_changed)
        self.runtime_ctrl.triggerResultReady.connect(self._on_runtime_trigger_result)

        connect_runtime_dialogs(self, self.runtime_ctrl)

    def _build_menu_bar(self) -> None:
        self.menuBar().hide()

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
            button = QtWidgets.QToolButton(self)
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
            button = QtWidgets.QToolButton(self)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setIcon(icon)
            button.setText(text)
            button.setStyleSheet(button_style)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            button.clicked.connect(slot)
            return button

        top_bar = QtWidgets.QFrame(self)
        top_bar.setFixedHeight(34)
        top_bar.setStyleSheet("QFrame{background:#3a3a3a;border-bottom:1px solid #505050;}")
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 2, 6, 2)
        top_layout.setSpacing(2)

        # ── 文件 ──
        file_menu = QtWidgets.QMenu("文件", self)
        file_menu.setStyleSheet(menu_style)
        act_exit = file_menu.addAction(_icon(SP.SP_DialogCloseButton), "退出")
        act_exit.triggered.connect(self.close)

        # ── 视图 ──
        view_menu = QtWidgets.QMenu("视图", self)
        view_menu.setStyleSheet(menu_style)
        self.act_show_debug = view_menu.addAction(
            _icon(SP.SP_FileDialogDetailedView), "切换到调试界面"
        )
        self.act_show_debug.triggered.connect(lambda: self._switch_workspace("debug"))
        self.act_show_runtime = view_menu.addAction(
            _icon(SP.SP_MediaPlay), "切换到运行界面"
        )
        self.act_show_runtime.triggered.connect(lambda: self._switch_workspace("runtime"))

        # ── 工具 ──
        tools_menu = QtWidgets.QMenu("工具", self)
        tools_menu.setStyleSheet(menu_style)
        self.act_reload_debug = tools_menu.addAction(
            _icon(SP.SP_BrowserReload), "重新加载调试会话"
        )
        self.act_reload_debug.triggered.connect(self._reload_debug_session)
        tools_menu.addSeparator()

        hardware_menu = tools_menu.addMenu(_icon(SP.SP_ComputerIcon), "工程调试工具")
        hardware_menu.addAction(
            _icon(SP.SP_DesktopIcon), "相机取图 / 参数工具"
        ).triggered.connect(self.tool_page.open_camera_debug_dialog)
        hardware_menu.addAction(
            _icon(SP.SP_DriveNetIcon), "DI / DO 调试工具"
        ).triggered.connect(self.tool_page.open_io_debug_dialog)
        hardware_menu.addAction(
            _icon(SP.SP_FileDialogContentsView), "模板工具"
        ).triggered.connect(self.tool_page.open_template_editor_dialog)
        hardware_menu.addAction(
            _icon(SP.SP_FileDialogListView), "自动生成ROI工具"
        ).triggered.connect(self.tool_page.open_template_match_dialog)

        tools_menu.addSeparator()
        algo_menu = tools_menu.addMenu(_icon(SP.SP_FileDialogInfoView), "算法工具")
        algo_menu.addAction(
            _icon(SP.SP_DialogApplyButton), "执行 Margin 验证"
        ).triggered.connect(self.tool_page.open_margin_validation_tool)
        algo_menu.addAction(
            _icon(SP.SP_FileDialogDetailedView), "打开特征分析"
        ).triggered.connect(self.tool_page.open_embedding_analysis_tool)
        algo_menu.addAction(
            _icon(SP.SP_FileDialogListView), "打开传统基线调试"
        ).triggered.connect(self.tool_page.open_baseline_debug_tool)

        # ── 控制 ──
        runtime_menu = QtWidgets.QMenu("控制", self)
        runtime_menu.setStyleSheet(menu_style)
        runtime_menu.addAction(
            _icon(SP.SP_BrowserReload), "刷新相机列表"
        ).triggered.connect(self.runtime_page.refreshCamerasRequested.emit)
        runtime_menu.addAction(
            _icon(SP.SP_DriveNetIcon), "连接相机..."
        ).triggered.connect(self._show_connect_dialog)
        runtime_menu.addAction(
            _icon(SP.SP_DialogDiscardButton), "断开相机"
        ).triggered.connect(self.runtime_ctrl.disconnect)
        runtime_menu.addSeparator()
        runtime_menu.addAction(
            _icon(SP.SP_MediaPlay), "脚踏触发"
        ).triggered.connect(self.runtime_page.triggerRequested.emit)
        runtime_menu.addAction(
            _icon(SP.SP_DialogYesButton), "密码放行..."
        ).triggered.connect(self._show_release_dialog)
        runtime_menu.addAction(
            _icon(SP.SP_FileDialogDetailedView), "修改放行密码..."
        ).triggered.connect(self._show_change_release_password_dialog)
        runtime_menu.addSeparator()
        runtime_menu.addAction(
            _icon(SP.SP_BrowserReload), "刷新运行状态"
        ).triggered.connect(
            lambda: self.runtime_ctrl.refresh_all_status("手动刷新运行状态")
        )

        # ── 路径 ──
        path_menu = QtWidgets.QMenu("路径", self)
        path_menu.setStyleSheet(menu_style)
        path_menu.addAction(
            _icon(SP.SP_DirHomeIcon), "打开 EmbeddingTest 根目录"
        ).triggered.connect(self._open_workspace_root)
        path_menu.addAction(
            _icon(SP.SP_DirOpenIcon), "打开当前产品目录"
        ).triggered.connect(self._open_current_product_dir)
        path_menu.addAction(
            _icon(SP.SP_DirIcon), "打开会话目录"
        ).triggered.connect(self._open_session_dir)
        path_menu.addAction(
            _icon(SP.SP_DirLinkIcon), "打开运行记录目录"
        ).triggered.connect(self._open_runtime_records_dir)

        top_layout.addWidget(
            _make_popup_button("文件", _icon(SP.SP_DirOpenIcon), file_menu),
            0,
        )
        top_layout.addWidget(
            _make_popup_button("视图", _icon(SP.SP_FileDialogDetailedView), view_menu),
            0,
        )
        top_layout.addWidget(
            _make_popup_button("工具", _icon(SP.SP_ComputerIcon), tools_menu),
            0,
        )
        top_layout.addWidget(
            _make_popup_button("控制", _icon(SP.SP_MediaPlay), runtime_menu),
            0,
        )
        top_layout.addWidget(
            _make_popup_button("路径", _icon(SP.SP_DirIcon), path_menu),
            0,
        )
        top_layout.addWidget(
            _make_action_button("Help", _icon(SP.SP_MessageBoxInformation), self._show_about_dialog),
            0,
        )
        top_layout.addStretch(1)

        self.setMenuWidget(top_bar)

    def _build_status_bar(self) -> None:
        _sb_style = "color:#888;font-size:11px;"
        self.lbl_status_workspace = QtWidgets.QLabel("工作区：调试界面")
        self.lbl_status_workspace.setStyleSheet(_sb_style)
        self.lbl_status_product = QtWidgets.QLabel(f"产品：{self.session.current_product}")
        self.lbl_status_product.setStyleSheet(_sb_style)
        self.lbl_status_engine = QtWidgets.QLabel()
        self.lbl_status_engine.setStyleSheet(_sb_style)
        self.lbl_status_path = QtWidgets.QLabel(f"产品目录：{self.session.product_dir}")
        self.lbl_status_path.setStyleSheet(_sb_style)
        self._bottom_status_bar.addPermanentWidget(self.lbl_status_workspace)
        self._bottom_status_bar.addPermanentWidget(self.lbl_status_product)
        self._bottom_status_bar.addPermanentWidget(self.lbl_status_engine)
        self._bottom_status_bar.addPermanentWidget(self.lbl_status_path, 1)
        self._sync_shell_status()
        self._set_algorithm_engine_status(
            "算法引擎：已就绪" if is_qr_core_ready() else "算法引擎：加载中..."
        )

    def _switch_workspace(self, workspace: str) -> None:
        is_runtime = workspace == "runtime"
        self.btn_workspace_debug.setChecked(not is_runtime)
        self.btn_workspace_runtime.setChecked(is_runtime)
        self.main_pages.setCurrentWidget(self.runtime_page if is_runtime else self.tool_page)
        if is_runtime:
            self.lbl_workspace_title.setText("运行界面")
            self.lbl_workspace_hint.setText("实时检测画面与检测项结果")
            self.lbl_status_workspace.setText("工作区：运行界面")
            runtime_message = self._activate_runtime_workspace()
            self.runtime_ctrl.refresh_all_status("已切换到运行界面")
            if runtime_message:
                self._bottom_status_bar.showMessage(runtime_message, 3000)
        else:
            self.lbl_workspace_title.setText("调试界面")
            self.lbl_workspace_hint.setText("模板配置、ROI标注、测试与训练")
            self.lbl_status_workspace.setText("工作区：调试界面")

    def _activate_runtime_workspace_legacy(self) -> str:
        if self.runtime_ctrl.connected_roles():
            return "运行链路已连接"

        bindings = self.runtime_page.camera_bindings()
        debug_serial = self.tool_page.connected_debug_camera_serial()

        if debug_serial and not str(bindings.get("cam1", "")).strip():
            self.runtime_page.edit_cam1_serial.setText(debug_serial)
            bindings = self.runtime_page.camera_bindings()

        if not bindings:
            return ""

        if debug_serial and debug_serial in {str(value).strip() for value in bindings.values()}:
            self.tool_page.release_debug_camera_for_runtime()

        self.runtime_ctrl.connect_cameras(bindings)
        connected_roles = self.runtime_ctrl.connected_roles()
        if connected_roles:
            return "已自动切换到运行检测状态"
        return ""

    def _activate_runtime_workspace(self) -> str:
        return self._ensure_runtime_camera_connection()

    def _on_debug_camera_connected(self, serial: str) -> None:
        if self.main_pages.currentWidget() is not self.runtime_page:
            return
        runtime_message = self._ensure_runtime_camera_connection(debug_serial=serial)
        if runtime_message:
            self._bottom_status_bar.showMessage(runtime_message, 3000)

    def _ensure_runtime_camera_connection(self, *, debug_serial: str = "") -> str:
        if self.runtime_ctrl.connected_roles():
            return ""

        bindings = self.runtime_page.camera_bindings()
        debug_serial = str(debug_serial or self.tool_page.connected_debug_camera_serial()).strip()

        if debug_serial and not str(bindings.get("cam1", "")).strip():
            self.runtime_page.edit_cam1_serial.setText(debug_serial)
            bindings = self.runtime_page.camera_bindings()

        if not bindings:
            return ""

        if debug_serial and debug_serial in {str(value).strip() for value in bindings.values()}:
            self.tool_page.release_debug_camera_for_runtime()

        self.runtime_ctrl.connect_cameras(bindings)
        if self.runtime_ctrl.connected_roles():
            return "已自动切换到运行检测状态"
        return ""

    def _sync_shell_status(self) -> None:
        self.lbl_status_product.setText(f"产品：{self.session.current_product}")
        self.lbl_status_path.setText(f"产品目录：{self.session.product_dir}")

    def _update_brand_banner_pixmap(self) -> None:
        if not hasattr(self, "lbl_brand_banner"):
            return
        if self._brand_banner_source.isNull():
            self.lbl_brand_banner.hide()
            return
        self.lbl_brand_banner.show()
        self.lbl_brand_banner.set_source_pixmap(self._brand_banner_source)

    def _show_about_dialog(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            f"About {_APP_NAME}",
            f"{_APP_NAME}\n版本：{_APP_VERSION}",
        )

    def _set_algorithm_engine_status(self, text: str, *, tooltip: str = "") -> None:
        self.lbl_status_engine.setText(text)
        self.lbl_status_engine.setToolTip(tooltip or text)

    def _start_algorithm_engine_warmup(self) -> None:
        if is_qr_core_ready():
            self._set_algorithm_engine_status("算法引擎：已就绪")
            self._preload_current_embedding_model()
            return
        if self._engine_warmup_thread is not None and self._engine_warmup_thread.isRunning():
            return
        self._set_algorithm_engine_status("算法引擎：加载中...")
        worker = _AlgorithmEngineWarmupThread(self)
        worker.warmupFinished.connect(self._on_algorithm_engine_warmup_finished)
        worker.finished.connect(self._on_algorithm_engine_warmup_thread_finished)
        self._engine_warmup_thread = worker
        worker.start()

    @QtCore.Slot(bool, str)
    def _on_algorithm_engine_warmup_finished(self, success: bool, message: str) -> None:
        if success:
            self._set_algorithm_engine_status("算法引擎：已就绪")
            self._preload_current_embedding_model()
            return
        self._set_algorithm_engine_status("算法引擎：加载失败", tooltip=message)

    @QtCore.Slot()
    def _on_algorithm_engine_warmup_thread_finished(self) -> None:
        self._engine_warmup_thread = None

    def _preload_current_embedding_model(self) -> None:
        algorithm = self.tool_page.current_algorithm()
        if not self.algo.is_embedding_algorithm(algorithm):
            return
        try:
            self.tool_page.load_embedding_model(algorithm)
        except Exception:
            self.algo.model = None

    def _open_in_explorer(self, path: str) -> None:
        target = Path(path)
        if not target.exists():
            QtWidgets.QMessageBox.information(self, "路径", f"路径不存在：\n{target}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))

    def _open_current_product_dir(self) -> None:
        self._open_in_explorer(self.session.product_dir)

    def _open_session_dir(self) -> None:
        self._open_in_explorer(self.session.session_dir)

    def _open_workspace_root(self) -> None:
        self._open_in_explorer(str(embedding_test_root(__file__)))

    def _open_runtime_records_dir(self) -> None:
        self._open_in_explorer(str(Path(self.session.product_dir) / "runtime_records"))

    def _reload_debug_session(self) -> None:
        self.tool_page.load_session()
        if is_qr_core_ready():
            self._preload_current_embedding_model()
        self.runtime_ctrl.refresh_all_status("已重新加载调试会话")
        self._sync_shell_status()

    def _show_connect_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("连接相机")
        dlg.setMinimumWidth(360)
        layout = QtWidgets.QFormLayout(dlg)

        edit_cam1 = QtWidgets.QLineEdit(self.runtime_page.edit_cam1_serial.text())
        edit_cam1.setPlaceholderText("Cam1 串号")
        edit_cam2 = QtWidgets.QLineEdit(self.runtime_page.edit_cam2_serial.text())
        edit_cam2.setPlaceholderText("Cam2 串号（可选）")
        layout.addRow("Cam1 串号", edit_cam1)
        layout.addRow("Cam2 串号", edit_cam2)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addRow(btn_box)

        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.runtime_page.edit_cam1_serial.setText(edit_cam1.text().strip())
            self.runtime_page.edit_cam2_serial.setText(edit_cam2.text().strip())
            self.runtime_page.connectCamerasRequested.emit(
                self.runtime_page.camera_bindings()
            )

    def _show_release_dialog(self) -> None:
        pwd, ok = QtWidgets.QInputDialog.getText(
            self, "密码放行", "输入放行密码：",
            QtWidgets.QLineEdit.Password,
        )
        if ok and pwd:
            self.runtime_page.releaseRequested.emit(pwd)

    @QtCore.Slot(str, str)
    def _on_runtime_trigger_result(self, result: str, _detail: str) -> None:
        if str(result).strip().upper() != "NG":
            return
        QtCore.QTimer.singleShot(80, self._show_release_dialog)

    # ------------------------------------------------------------------
    # 跨组件协调（ToolPage 请求 → 先处理运行链路 → 再委托 ToolPage）
    # ------------------------------------------------------------------

    def _on_product_change_request(self, new_name: str) -> None:
        self.runtime_ctrl.disconnect(silent=True)
        self.tool_page.apply_product_switch(new_name)
        if is_qr_core_ready():
            self._preload_current_embedding_model()
        self._sync_shell_status()
        self.runtime_ctrl.refresh_all_status("产品已切换，请重新连接运行链路")

    def _on_session_clear_request(self) -> None:
        self.runtime_ctrl.disconnect(silent=True)
        self.tool_page.reset_for_clear()
        self._sync_shell_status()
        self.runtime_ctrl.refresh_all_status("会话已清空，请重新准备运行链路")

    # ------------------------------------------------------------------
    # 预览图更新（RuntimeController Signal → RuntimeModePage）
    # ------------------------------------------------------------------

    def _on_runtime_preview_updated(self, role: str, path: str) -> None:
        update_runtime_preview(self.runtime_page, role, path)

    # ------------------------------------------------------------------
    # 窗口生命周期
    # ------------------------------------------------------------------

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "lbl_brand_banner"):
            self.lbl_brand_banner.update()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.tool_page._cleanup_debug_hardware()
        except Exception:
            pass
        self.runtime_ctrl.disconnect(silent=True)
        super().closeEvent(event)


def main() -> None:
    _set_windows_app_id(_WINDOWS_APP_ID)
    app = QtWidgets.QApplication([])
    app.setApplicationName(_APP_NAME)
    app.setApplicationDisplayName(_APP_NAME)
    logo_path = _resource_path("logo.png")
    if logo_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(logo_path)))
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
