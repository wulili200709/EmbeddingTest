"""
ui.shell.main_window

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

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from application import (
    DEFAULT_RELEASE_PASSWORD,
    ProductRuntimeContext,
    RuntimeController,
)
from ui.shell.chrome import (
    build_menu_bar as _build_shell_menu_bar,
    build_status_bar as _build_shell_status_bar,
    show_about_dialog as _show_shell_about_dialog,
    switch_workspace as _switch_shell_workspace,
    sync_shell_status as _sync_shell_status_impl,
    update_brand_banner_pixmap as _update_brand_banner_pixmap_impl,
)
from ui.shell.dialogs import (
    DEFAULT_ADMIN_PASSWORD,
    PasswordSettingsStore,
    RuntimeRecordSettingsStore,
    TowerLightSettingsStore,
    confirm_admin_password,
    prompt_change_release_password,
    prompt_connect_camera_bindings,
    prompt_password_dialog,
    prompt_tower_light_settings,
)
from ui.shell.engine import (
    on_algorithm_engine_warmup_finished as _on_algorithm_engine_warmup_finished_impl,
    on_algorithm_engine_warmup_thread_finished as _on_algorithm_engine_warmup_thread_finished_impl,
    preload_current_embedding_model as _preload_current_embedding_model_impl,
    reload_debug_session as _reload_debug_session_impl,
    start_algorithm_engine_warmup as _start_algorithm_engine_warmup_impl,
)
from ui.shell.layout import build_main_window_ui as _build_main_window_ui
from ui.shell.runtime_bridge import (
    activate_runtime_workspace as _activate_runtime_workspace_impl,
    activate_runtime_workspace_legacy as _activate_runtime_workspace_legacy_impl,
    apply_runtime_capture_policy as _apply_runtime_capture_policy_impl,
    ensure_runtime_camera_connection as _ensure_runtime_camera_connection_impl,
    load_runtime_capture_policy_from_session as _load_runtime_capture_policy_from_session_impl,
    normalize_runtime_capture_policy as _normalize_runtime_capture_policy_impl,
    on_debug_camera_connected as _on_debug_camera_connected_impl,
    on_product_change_request as _on_product_change_request_impl,
    on_runtime_active_roles_changed as _on_runtime_active_roles_changed_impl,
    on_runtime_preview_updated as _on_runtime_preview_updated_impl,
    on_session_clear_request as _on_session_clear_request_impl,
    persist_runtime_camera_bindings as _persist_runtime_camera_bindings_impl,
    persist_runtime_capture_policy as _persist_runtime_capture_policy_impl,
    prepare_runtime_for_debug_camera as _prepare_runtime_for_debug_camera_impl,
    restore_runtime_camera_bindings_from_session as _restore_runtime_camera_bindings_from_session_impl,
    restore_runtime_capture_policy_from_session as _restore_runtime_capture_policy_from_session_impl,
    runtime_capture_policy_text as _runtime_capture_policy_text_impl,
    startup_auto_connect_runtime_cameras as _startup_auto_connect_runtime_cameras_impl,
    sync_runtime_capture_policy_actions as _sync_runtime_capture_policy_actions_impl,
)
from ui.shell.support import (
    APP_NAME as _APP_NAME,
    WINDOWS_APP_ID as _WINDOWS_APP_ID,
    app_icon as _app_icon,
    resource_path as _resource_path,
    set_windows_app_id as _set_windows_app_id,
)
from ui.window_common import (
    build_default_session_and_algo,
    connect_runtime_dialogs,
    connect_runtime_page,
    connect_runtime_refresh_sources,
    detect_runtime_import_error,
    embedding_test_root,
)


_RUNTIME_IMPORT_ERROR = detect_runtime_import_error()

# Test-stage switch:
# False = 测试模式：NG时不自动弹出放行密码框，且不进入NG锁定，可直接继续下一次测试
# True  = 产线模式：NG时自动弹出放行密码框，并进入NG锁定
AUTO_SHOW_RELEASE_DIALOG_ON_NG = True


def _normalize_application_font(app: QtWidgets.QApplication) -> None:
    font = QtGui.QFont(app.font())
    if font.pointSizeF() > 0:
        return
    if font.pixelSize() > 0:
        font.setPointSize(max(1, int(round(font.pixelSize() * 0.75))))
    else:
        font.setPointSize(10)
    app.setFont(font)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(_APP_NAME)
        app_icon = _app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.setStyleSheet(
            "QMainWindow{background:#2d2d2d;}"
            "QMenuBar{background:#3a3a3a;color:#e0e0e0;border-bottom:1px solid #505050;}"
            "QMenuBar::item:selected{background:#4a4a4a;}"
            "QMenu{background:#3a3a3a;color:#e0e0e0;border:1px solid #505050;}"
            "QMenu::item:selected{background:#3794ff;}"
            "QStatusBar{background:#3a3a3a;color:#aaa;border-top:1px solid #505050;}"
        )

        self.session, self.algo = build_default_session_and_algo(__file__)
        self._password_store = PasswordSettingsStore(
            default_release_password=DEFAULT_RELEASE_PASSWORD,
            default_admin_password=DEFAULT_ADMIN_PASSWORD,
        )
        self._runtime_record_store = RuntimeRecordSettingsStore()
        self._tower_light_store = TowerLightSettingsStore()
        self._password_settings = self._password_store.load()
        self._runtime_record_settings = self._runtime_record_store.load()
        self._tower_light_settings = self._tower_light_store.load()
        self._release_password = self._password_settings["run_password"]
        self._admin_password = self._password_settings["engineer_password"]
        self._engine_warmup_thread: Optional[QtCore.QThread] = None
        self._brand_banner_source = QtGui.QPixmap(str(_resource_path("logo2.png")))
        self._startup_runtime_auto_connect_done = False
        self._runtime_capture_policy = self._load_runtime_capture_policy_from_session()

        # ── UI 组装 ────────────────────────────────────────────────────
        self._build_ui()

        # ── 运行控制器（需要 tool_page 已创建） ─────────────────────────
        self.runtime_ctrl = RuntimeController(
            session=self.session,
            algo=self.algo,
            runtime_context=ProductRuntimeContext(self.session, self.algo),
            import_error=_RUNTIME_IMPORT_ERROR,
            release_password=self._release_password,
            lock_on_ng=AUTO_SHOW_RELEASE_DIALOG_ON_NG,
            parent=self,
        )
        self.runtime_ctrl.set_capture_retention_policy(self._runtime_capture_policy)
        self._apply_runtime_records_directory_setting()
        self.runtime_ctrl.update_tower_light_settings(self._tower_light_settings)

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
        QtCore.QTimer.singleShot(0, self.runtime_ctrl.initialize_startup_io)
        QtCore.QTimer.singleShot(0, self._start_algorithm_engine_warmup)
        QtCore.QTimer.singleShot(150, self._startup_auto_connect_runtime_cameras)

    def _on_camera_settings_applied(self, serial: str, settings_payload) -> None:
        self.runtime_ctrl.apply_camera_settings_for_serial(serial, settings_payload)

    def _prepare_runtime_for_debug_camera(self, serial: str) -> None:
        _prepare_runtime_for_debug_camera_impl(self, serial)

    def _restore_runtime_camera_bindings_from_session(self) -> None:
        _restore_runtime_camera_bindings_from_session_impl(self)

    def _persist_runtime_camera_bindings(self, bindings: Optional[dict[str, str]] = None) -> None:
        _persist_runtime_camera_bindings_impl(self, bindings)

    def _normalize_runtime_capture_policy(self, policy: str) -> str:
        return _normalize_runtime_capture_policy_impl(policy)

    def _runtime_capture_policy_text(self, policy: str) -> str:
        return _runtime_capture_policy_text_impl(policy)

    def _load_runtime_capture_policy_from_session(self) -> str:
        return _load_runtime_capture_policy_from_session_impl(self)

    def _persist_runtime_capture_policy(self, policy: str) -> None:
        _persist_runtime_capture_policy_impl(self, policy)

    def _sync_runtime_capture_policy_actions(self) -> None:
        _sync_runtime_capture_policy_actions_impl(self)

    def _apply_runtime_capture_policy(
        self,
        policy: str,
        *,
        persist: bool,
        show_message: bool,
    ) -> None:
        _apply_runtime_capture_policy_impl(
            self,
            policy,
            persist=persist,
            show_message=show_message,
        )

    def _restore_runtime_capture_policy_from_session(self) -> None:
        _restore_runtime_capture_policy_from_session_impl(self)

    def _startup_auto_connect_runtime_cameras(self) -> None:
        _startup_auto_connect_runtime_cameras_impl(self, import_error=_RUNTIME_IMPORT_ERROR)

    @QtCore.Slot(list)
    def _on_runtime_active_roles_changed(self, roles: list[str]) -> None:
        _on_runtime_active_roles_changed_impl(self, roles)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        _build_main_window_ui(self)

    def _apply_release_password(self, password: str) -> bool:
        password_text = str(password).strip()
        if len(password_text) < 4:
            QtWidgets.QMessageBox.warning(self, "修改放行密码", "新密码至少需要 4 位。")
            return False

        settings = dict(self._password_settings)
        settings["run_password"] = password_text
        try:
            self._password_store.save(settings)
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

    def _show_change_release_password_dialog(self) -> None:
        if not confirm_admin_password(self, admin_password=self._admin_password):
            return

        while True:
            new_password = prompt_change_release_password(self)
            if new_password is None:
                return
            if not self._apply_release_password(new_password):
                continue
            QtWidgets.QMessageBox.information(self, "修改放行密码", "放行密码已更新。")
            return

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _show_tower_light_settings_dialog(self) -> None:
        if not confirm_admin_password(self, admin_password=self._admin_password):
            return

        new_settings = prompt_tower_light_settings(
            self,
            current_settings=self._tower_light_settings,
        )
        if new_settings is None:
            return

        try:
            self._tower_light_store.save(new_settings)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "\u5854\u706f\u65f6\u5e8f\u8bbe\u7f6e",
                f"\u4fdd\u5b58\u5854\u706f\u53c2\u6570\u5931\u8d25\uff1a\n{exc}",
            )
            return

        self._tower_light_settings = dict(new_settings)
        self.runtime_ctrl.update_tower_light_settings(self._tower_light_settings)
        self._bottom_status_bar.showMessage("\u5854\u706f\u65f6\u5e8f\u53c2\u6570\u5df2\u66f4\u65b0", 3000)

    def _connect_signals(self) -> None:
        # ToolPage → MainWindow（跨组件协调）
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(self._sync_shell_status)
        self.tool_page.sessionLoaded.connect(self._restore_runtime_camera_bindings_from_session)
        self.tool_page.sessionLoaded.connect(self._restore_runtime_capture_policy_from_session)
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
        self.runtime_ctrl.ioStatusChanged.connect(self._on_runtime_io_status_changed)

        connect_runtime_dialogs(self, self.runtime_ctrl)

    def _build_menu_bar(self) -> None:
        _build_shell_menu_bar(self)

    def _build_status_bar(self) -> None:
        _build_shell_status_bar(self)

    def _switch_workspace(self, workspace: str) -> None:
        _switch_shell_workspace(self, workspace)

    def _activate_runtime_workspace_legacy(self) -> str:
        return _activate_runtime_workspace_legacy_impl(self)

    def _activate_runtime_workspace(self) -> str:
        return _activate_runtime_workspace_impl(self)

    def _on_debug_camera_connected(self, role: str, serial: str) -> None:
        _on_debug_camera_connected_impl(self, role, serial)

    def _ensure_runtime_camera_connection(self, *, debug_role: str = "", debug_serial: str = "") -> str:
        return _ensure_runtime_camera_connection_impl(
            self,
            debug_role=debug_role,
            debug_serial=debug_serial,
        )

    def _sync_shell_status(self) -> None:
        _sync_shell_status_impl(self)
        if hasattr(self, "runtime_ctrl"):
            self._apply_runtime_records_directory_setting()

    def _update_brand_banner_pixmap(self) -> None:
        _update_brand_banner_pixmap_impl(self)

    def _show_about_dialog(self) -> None:
        _show_shell_about_dialog(self)

    def _set_algorithm_engine_status(self, text: str, *, tooltip: str = "") -> None:
        self.lbl_status_engine.setText(text)
        self.lbl_status_engine.setToolTip(tooltip or text)

    @QtCore.Slot(bool, str, object)
    def _on_runtime_io_status_changed(self, ready: bool, detail: str, controller: object) -> None:
        self.lbl_status_io_dot.setStyleSheet(
            "color:#2ea043;font-size:14px;font-weight:bold;" if ready else "color:#c74e39;font-size:14px;font-weight:bold;"
        )
        self.lbl_status_io_text.setText("IO: 已就绪" if ready else "IO: 初始化失败")
        self.lbl_status_io_text.setToolTip(detail or self.lbl_status_io_text.text())
        self.tool_page.set_runtime_io_state(ready, detail, controller)

    def _start_algorithm_engine_warmup(self) -> None:
        _start_algorithm_engine_warmup_impl(self)

    @QtCore.Slot(bool, str)
    def _on_algorithm_engine_warmup_finished(self, success: bool, message: str) -> None:
        _on_algorithm_engine_warmup_finished_impl(self, success, message)

    @QtCore.Slot()
    def _on_algorithm_engine_warmup_thread_finished(self) -> None:
        _on_algorithm_engine_warmup_thread_finished_impl(self)

    def _preload_current_embedding_model(self) -> None:
        _preload_current_embedding_model_impl(self)

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
        self._open_in_explorer(self.runtime_ctrl.runtime_records_directory())

    def _apply_runtime_records_directory_setting(self) -> None:
        configured_dir = str(self._runtime_record_settings.get("runtime_records_dir", "")).strip()
        self.runtime_ctrl.update_runtime_records_directory(
            configured_dir or (Path(self.session.product_dir) / "runtime_records")
        )

    def _show_runtime_records_directory_dialog(self) -> None:
        current_dir = str(self._runtime_record_settings.get("runtime_records_dir", "")).strip()
        if not current_dir:
            current_dir = str(Path(self.session.product_dir) / "runtime_records")
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "保存运行记录",
            current_dir,
        )
        if not selected_dir:
            return

        settings = {
            "runtime_records_dir": str(selected_dir).strip(),
        }
        try:
            self._runtime_record_store.save(settings)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "保存运行记录",
                f"保存运行记录目录失败：\n{exc}",
            )
            return

        self._runtime_record_settings = settings
        self._apply_runtime_records_directory_setting()
        self._bottom_status_bar.showMessage("运行记录保存目录已更新", 3000)

    def _reload_debug_session(self) -> None:
        _reload_debug_session_impl(self)

    def _show_connect_dialog(self) -> None:
        bindings = prompt_connect_camera_bindings(
            self,
            cam1_serial=self.runtime_page.edit_cam1_serial.text(),
            cam2_serial=self.runtime_page.edit_cam2_serial.text(),
        )
        if bindings is None:
            return
        cam1_serial, cam2_serial = bindings
        self.runtime_page.edit_cam1_serial.setText(cam1_serial)
        self.runtime_page.edit_cam2_serial.setText(cam2_serial)
        self.runtime_page.connectCamerasRequested.emit(
            self.runtime_page.camera_bindings()
        )

    def _show_release_dialog(self) -> None:
        pwd, ok = prompt_password_dialog(
            self,
            title="\u5bc6\u7801\u653e\u884c",
            label="\u8f93\u5165\u653e\u884c\u5bc6\u7801\uff1a",
        )
        if ok and pwd:
            self.runtime_page.releaseRequested.emit(pwd)

    @QtCore.Slot(str, str)
    def _on_runtime_trigger_result(self, result: str, _detail: str) -> None:
        if str(result).strip().upper() != "NG":
            return
        if not AUTO_SHOW_RELEASE_DIALOG_ON_NG:
            return
        QtCore.QTimer.singleShot(80, self._show_release_dialog)

    # ------------------------------------------------------------------
    # 跨组件协调（ToolPage 请求 → 先处理运行链路 → 再委托 ToolPage）
    # ------------------------------------------------------------------

    def _on_product_change_request(self, new_name: str) -> None:
        _on_product_change_request_impl(self, new_name)

    def _on_session_clear_request(self) -> None:
        _on_session_clear_request_impl(self)

    # ------------------------------------------------------------------
    # 预览图更新（RuntimeController Signal → RuntimeModePage）
    # ------------------------------------------------------------------

    def _on_runtime_preview_updated(self, role: str, path: str) -> None:
        _on_runtime_preview_updated_impl(self, role, path)

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
        worker = self._engine_warmup_thread
        if worker is not None and worker.isRunning():
            worker.wait()
            self._engine_warmup_thread = None
        self.runtime_ctrl.disconnect(silent=True)
        super().closeEvent(event)


def main() -> None:
    _set_windows_app_id(_WINDOWS_APP_ID)
    app = QtWidgets.QApplication([])
    _normalize_application_font(app)
    app.setApplicationName(_APP_NAME)
    app.setApplicationDisplayName(_APP_NAME)
    app_icon = _app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    w = MainWindow()
    screen = app.primaryScreen()
    available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1366, 768)
    small_screen = available.width() <= 1366 or available.height() <= 800
    if small_screen:
        w.showMaximized()
    else:
        target_width = min(1400, max(1200, available.width() - 80))
        target_height = min(900, max(800, available.height() - 80))
        w.resize(target_width, target_height)
        w.show()
    app.exec()


if __name__ == "__main__":
    main()
