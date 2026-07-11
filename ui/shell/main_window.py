
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from common.camera_roles import (
    CAMERA_ROLES,
    DEFAULT_CAMERA_ROLE,
    configured_camera_roles,
    normalize_camera_role,
)
from algorithms.lazy_api import is_ready as is_qr_core_ready
from application import (
    DEFAULT_RELEASE_PASSWORD,
    ProductRuntimeContext,
    RuntimeController,
)
from infrastructure.audit_store import AuditStore, PermissionService
from infrastructure.camera_settings_store import (
    CAPTURE_MODE_SINGLE_MULTI_LIGHT,
    CameraSettingsStore,
    normalize_capture_mode,
)
from ui.shell.audit_dialogs import (
    AuditLogDialog,
    ChangePasswordDialog,
    LoginDialog,
    SoftwareVersionDialog,
    UserPermissionDialog,
)
from ui.shell.chrome import (
    build_menu_bar as _build_shell_menu_bar,
    build_status_bar as _build_shell_status_bar,
    retranslate_shell_chrome as _retranslate_shell_chrome,
    show_about_dialog as _show_shell_about_dialog,
    switch_workspace as _switch_shell_workspace,
    sync_shell_status as _sync_shell_status_impl,
    update_brand_banner_pixmap as _update_brand_banner_pixmap_impl,
)
from ui.shell.dialogs import (
    DEFAULT_ADMIN_PASSWORD,
    PasswordSettingsStore,
    RuntimeModeSettingsStore,
    RuntimeRecordSettingsStore,
    TowerLightSettingsStore,
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
    APP_VERSION as _APP_VERSION,
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
from ui.i18n import set_language, tr
# Test-stage switch:
# False = 测试模式：NG时不自动弹出放行密码框，且不进入NG锁定，可直接继续下一次测试
# True  = 产线模式：NG时自动弹出放行密码框，并进入NG锁定
AUTO_SHOW_RELEASE_DIALOG_ON_NG = True  # default; config/runtime_mode_settings.json overrides this at startup


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
        self._runtime_mode_store = RuntimeModeSettingsStore()
        self._password_settings = self._password_store.load()
        self._runtime_record_settings = self._runtime_record_store.load()
        self._tower_light_settings = self._tower_light_store.load()
        self._runtime_mode_settings = self._runtime_mode_store.load()
        self._release_password = self._password_settings["run_password"]
        self._admin_password = self._password_settings["engineer_password"]
        self._audit_store = AuditStore()
        self._permission_service = PermissionService(self._audit_store)
        self._auto_show_release_dialog_on_ng = bool(
            self._runtime_mode_settings.get(
                "auto_show_release_dialog_on_ng",
                AUTO_SHOW_RELEASE_DIALOG_ON_NG,
            )
        )
        self._runtime_import_error = detect_runtime_import_error()
        self._engine_warmup_thread: Optional[QtCore.QThread] = None
        self._brand_banner_source = QtGui.QPixmap(str(_resource_path("logo2.png")))
        self._startup_runtime_auto_connect_done = False
        self._startup_runtime_auto_connect_attempts = 0
        self._startup_after_show_scheduled = False
        self._startup_after_show_completed = False
        self._allow_initial_tool_session_load = False
        self._initial_ui_ready_timer_started = False
        self._runtime_capture_policy = self._load_runtime_capture_policy_from_session()

        # ── UI 组装 ────────────────────────────────────────────────────
        self._build_ui()
        self._set_sidebar_runtime_result("-", "")
        self._sync_configured_camera_roles()

        # ── 运行控制器（需要 tool_page 已创建） ─────────────────────────
        self.runtime_ctrl = RuntimeController(
            session=self.session,
            algo=self.algo,
            runtime_context=ProductRuntimeContext(self.session, self.algo),
            import_error=self._runtime_import_error,
            release_password=self._release_password,
            lock_on_ng=self._auto_show_release_dialog_on_ng,
            parent=self,
        )
        self.runtime_ctrl.set_capture_retention_policy(self._runtime_capture_policy)
        self._apply_runtime_records_directory_setting()
        self._apply_runtime_capture_directory_setting()
        self.runtime_ctrl.update_tower_light_settings(self._tower_light_settings)

        # ── 信号连接 ───────────────────────────────────────────────────
        self._connect_signals()
        self._build_menu_bar()
        self._build_status_bar()
        self._sync_permission_ui()

        # ── 初始化加载 ─────────────────────────────────────────────────
        self.tool_page.load_session()          # 发射 sessionLoaded → refresh_all_status
        self.runtime_ctrl.refresh_all_status() # 初始状态推送
        self._switch_workspace("debug")
        self._switch_workspace("debug")
        QtCore.QTimer.singleShot(80, self.runtime_ctrl.reset_all_camera_triggers_off)
        QtCore.QTimer.singleShot(80, self.runtime_ctrl.initialize_startup_io)
        QtCore.QTimer.singleShot(80, self._start_algorithm_engine_warmup)
        QtCore.QTimer.singleShot(2000, self._startup_auto_connect_runtime_cameras)

    def _has_permission(self, permission_key: str) -> bool:
        return self._permission_service.has(permission_key)

    def _require_permission(self, permission_key: str, action_name: str = "") -> bool:
        if self._has_permission(permission_key):
            return True
        message = tr("auth.permission_denied_message", action=action_name or permission_key)
        QtWidgets.QMessageBox.warning(self, tr("auth.permission_denied_title"), message)
        return False

    def _audit_event(
        self,
        *,
        module: str,
        action: str,
        target: str = "",
        before_value: str = "",
        after_value: str = "",
        result: str = "成功",
        remark: str = "",
        product_name: str | None = None,
    ) -> None:
        ctx = self._permission_service.audit_context()
        try:
            self._audit_store.log_event(
                user_name=ctx["user_name"],
                role_name=ctx["role_name"],
                product_name=self.session.current_product if product_name is None else product_name,
                module=module,
                action=action,
                target=target,
                before_value=before_value,
                after_value=after_value,
                result=result,
                remark=remark,
                software_version=_APP_VERSION,
            )
        except Exception:
            pass

    def _sync_permission_ui(self) -> None:
        role_key = getattr(self._permission_service.current_user, "role_key", "")
        role_name = tr(f"role.{role_key}") if role_key else self._permission_service.role_name
        user_text = tr(
            "status.user_role",
            user=self._permission_service.user_name,
            role=role_name,
        )
        if hasattr(self, "lbl_status_user"):
            self.lbl_status_user.setText(user_text)
        if hasattr(self, "act_auth_login"):
            self.act_auth_login.setEnabled(self._permission_service.user_name == "operator")
        if hasattr(self, "act_auth_logout"):
            self.act_auth_logout.setEnabled(self._permission_service.user_name != "operator")
        if hasattr(self, "act_change_current_password"):
            self.act_change_current_password.setEnabled(self._permission_service.user_name != "operator")
        if hasattr(self, "act_user_permissions"):
            self.act_user_permissions.setEnabled(self._has_permission("user.manage"))
        if hasattr(self, "act_audit_log"):
            self.act_audit_log.setEnabled(self._has_permission("audit.view"))
        if hasattr(self, "act_software_versions"):
            self.act_software_versions.setEnabled(
                self._has_permission("software.version_log") or self._has_permission("audit.view")
            )

        refs = getattr(self, "_shell_i18n_refs", {}) or {}
        actions = refs.get("actions", {})
        permission_by_action = {
            "camera_tool": "camera.edit_params",
            "io_tool": "io.debug",
            "template_editor": "template.edit_roi",
            "auto_region": "template.edit_roi",
            "margin_validation": "template.edit_params",
            "embedding_analysis": "template.edit_params",
            "baseline_debug": "template.edit_params",
            "connect_camera": "runtime.connect_camera",
            "disconnect_camera": "runtime.connect_camera",
            "foot_trigger": "runtime.run",
            "password_release": "runtime.release_ng",
            "tower_light": "settings.tower_light",
            "change_release_password": "settings.passwords",
            "save_image_path": "settings.record_path",
            "save_runtime_records": "settings.record_path",
        }
        for action_name, permission in permission_by_action.items():
            action = actions.get(action_name)
            if action is not None:
                action.setEnabled(self._has_permission(permission))

        if hasattr(self, "tool_page"):
            if hasattr(self.tool_page, "cmb_product"):
                self.tool_page.cmb_product.setEnabled(self._has_permission("product.select"))
            if hasattr(self.tool_page, "btn_new_product"):
                self.tool_page.btn_new_product.setEnabled(self._has_permission("product.create"))
            if hasattr(self.tool_page, "btn_delete_product"):
                self.tool_page.btn_delete_product.setEnabled(self._has_permission("product.delete"))
            if hasattr(self.tool_page, "inspection_items_table"):
                self.tool_page.inspection_items_table.setEnabled(self._has_permission("inspection.edit_items"))
            for attr in (
                "btn_import_train",
                "btn_train_to_test",
                "btn_sample_annotation",
                "btn_del_ok",
                "btn_test_to_train",
                "btn_add_test",
                "btn_del_test",
                "btn_sample_annotation_test",
            ):
                button = getattr(self.tool_page, attr, None)
                if button is not None:
                    button.setEnabled(self._has_permission("sample.manage"))
            for attr in ("btn_algorithm_picker", "cmb_mode", "spin_margin", "spin_topk"):
                widget = getattr(self.tool_page, attr, None)
                if widget is not None:
                    widget.setEnabled(self._has_permission("template.edit_params"))
            for attr in (
                "btn_save",
                "btn_clear",
                "btn_set_ref",
                "btn_pick_ref",
                "btn_autogen",
                "btn_autogen_all",
                "btn_clear_roi_batch",
                "cmb_shape",
                "cmb_label",
            ):
                widget = getattr(self.tool_page, attr, None)
                if widget is not None:
                    widget.setEnabled(self._has_permission("template.edit_roi"))
            widget = getattr(self.tool_page, "cmb_loc", None)
            if widget is not None:
                widget.setEnabled(self._has_permission("template.edit_params"))
            for attr in ("btn_add_line_distance_tool", "btn_delete_line_distance_tool"):
                button = getattr(self.tool_page, attr, None)
                if button is not None:
                    button.setEnabled(self._has_permission("inspection.edit_items"))
            for attr in (
                "btn_debug_read_camera_settings",
                "btn_debug_apply_camera_settings",
                "spin_debug_exposure",
                "spin_debug_gain",
                "spin_debug_digital_shift",
                "chk_debug_digital_shift_enable",
                "cmb_debug_trigger_mode",
                "cmb_debug_light_source_mode",
                "capture_mode_frame",
            ):
                widget = getattr(self.tool_page, attr, None)
                if widget is not None:
                    widget.setEnabled(self._has_permission("camera.edit_params"))
            table = getattr(self.tool_page, "capture_channel_table", None)
            if table is not None:
                table.setEnabled(self._has_permission("camera.edit_params"))
            for attr in ("btn_debug_connect_camera", "btn_debug_disconnect_camera"):
                button = getattr(self.tool_page, attr, None)
                if button is not None:
                    button.setEnabled(self._has_permission("runtime.connect_camera"))
            apply_io_state = getattr(self.tool_page, "_apply_runtime_io_debug_state", None)
            if callable(apply_io_state):
                apply_io_state()
            for attr in ("btn_train", "btn_train_current"):
                button = getattr(self.tool_page, attr, None)
                if button is not None:
                    button.setEnabled(self._has_permission("model.train"))
            update_runtime_widgets = getattr(self.tool_page, "_update_runtime_widgets", None)
            if callable(update_runtime_widgets):
                update_runtime_widgets()
        if hasattr(self, "runtime_page"):
            runtime_allowed = self._has_permission("runtime.run")
            camera_allowed = self._has_permission("runtime.connect_camera")
            release_allowed = self._has_permission("runtime.release_ng")
            for attr in ("btn_trigger", "btn_simulate_foot", "btn_trigger_cam1", "btn_trigger_cam2", "btn_trigger_cam3"):
                button = getattr(self.runtime_page, attr, None)
                if button is not None:
                    button.setEnabled(runtime_allowed)
            for attr in ("btn_connect_cameras", "btn_disconnect_cameras"):
                button = getattr(self.runtime_page, attr, None)
                if button is not None:
                    button.setEnabled(camera_allowed)
            button = getattr(self.runtime_page, "btn_release", None)
            if button is not None:
                button.setEnabled(release_allowed)

    def _show_login_dialog(self) -> None:
        dialog = LoginDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        user_name, password = dialog.credentials()
        user = self._permission_service.login(user_name, password)
        if user is None:
            QtWidgets.QMessageBox.warning(self, tr("auth.login.title"), tr("auth.login_failed"))
            return
        self._audit_event(module="权限", action="登录", product_name="")
        self._sync_permission_ui()
        self._bottom_status_bar.showMessage(tr("auth.logged_in", user=user.user_name), 3000)

    def _logout_current_user(self) -> None:
        if self._permission_service.user_name != "operator":
            self._audit_event(module="权限", action="退出登录", product_name="")
        self._permission_service.logout()
        self._sync_permission_ui()
        self._bottom_status_bar.showMessage(tr("auth.logged_out_operator"), 3000)

    def _show_change_current_user_password(self, *, required: bool = False) -> bool:
        dialog = ChangePasswordDialog(self, title=tr("auth.change_current_password.title"))
        while True:
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return not required
            try:
                self._audit_store.set_user_password_by_name(
                    self._permission_service.user_name,
                    dialog.password(),
                    must_change_password=False,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, tr("auth.change_password.title"), str(exc))
                if not required:
                    return False
                continue
            self._permission_service.current_user.must_change_password = False
            self._audit_event(module="权限", action="修改密码", product_name="")
            QtWidgets.QMessageBox.information(
                self,
                tr("auth.change_password.title"),
                tr("auth.password_updated"),
            )
            return True

    def _show_user_permission_dialog(self) -> None:
        if not self._require_permission("user.manage", tr("action.user_permissions")):
            return
        dialog = UserPermissionDialog(self, self._audit_store)
        dialog.exec()
        self._audit_event(module="权限", action="维护用户与权限", product_name="")
        self._sync_permission_ui()

    def _show_audit_log_dialog(self) -> None:
        if not self._require_permission("audit.view", tr("action.audit_log")):
            return
        dialog = AuditLogDialog(
            self,
            self._audit_store,
            can_export=self._has_permission("audit.export"),
        )
        dialog.exec()

    def _show_software_version_dialog(self) -> None:
        if not (
            self._has_permission("software.version_log") or self._has_permission("audit.view")
        ):
            self._require_permission("audit.view", tr("action.software_versions"))
            return
        dialog = SoftwareVersionDialog(
            self,
            self._audit_store,
            can_edit=self._has_permission("software.version_log"),
            current_user=self._permission_service.user_name,
            software_version=_APP_VERSION,
        )
        dialog.exec()

    def _disconnect_runtime_cameras(self) -> None:
        if not self._require_permission("runtime.connect_camera", "断开相机"):
            return
        before = ", ".join(self.runtime_ctrl.connected_roles())
        self.runtime_ctrl.disconnect()
        self._audit_event(module="运行", action="断开相机", before_value=before)

    def _connect_runtime_cameras(self, bindings) -> None:
        if not self._require_permission("runtime.connect_camera", "连接相机"):
            return
        if self.runtime_ctrl.connected_roles():
            QtWidgets.QMessageBox.information(self, "连接相机", "相机已经连接。")
            self.runtime_ctrl.refresh_all_status("相机已经连接")
            return
        self.runtime_ctrl.connect_cameras(self._runtime_physical_camera_bindings(bindings))

    def _trigger_runtime(self) -> None:
        if not self._require_permission("runtime.run", "运行检测"):
            return
        self.runtime_ctrl.trigger()

    def _trigger_runtime_camera(self, camera_index: int) -> None:
        if not self._require_permission("runtime.run", "运行检测"):
            return
        self.runtime_ctrl.trigger_camera(camera_index)

    def _release_runtime(self, password: str) -> None:
        if not self._require_permission("runtime.release_ng", "NG放行"):
            return
        self.runtime_ctrl.release(password)
        self._audit_event(module="运行", action="NG放行")

    def _on_camera_settings_applied(self, serial: str, settings_payload) -> None:
        if not self._require_permission("camera.edit_params", "修改相机参数"):
            return
        self._audit_event(
            module="相机参数",
            action="修改相机参数",
            target=str(serial or ""),
            after_value=str(dict(settings_payload or {})),
        )
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
        if persist and not self._require_permission("settings.record_path", "运行图片保留策略"):
            return
        before = str(self._runtime_capture_policy)
        _apply_runtime_capture_policy_impl(
            self,
            policy,
            persist=persist,
            show_message=show_message,
        )
        if persist and before != str(self._runtime_capture_policy):
            self._audit_event(
                module="设置",
                action="修改图片保留策略",
                before_value=before,
                after_value=str(self._runtime_capture_policy),
                product_name="",
            )

    def _restore_runtime_capture_policy_from_session(self) -> None:
        _restore_runtime_capture_policy_from_session_impl(self)

    def _startup_auto_connect_runtime_cameras(self) -> None:
        _startup_auto_connect_runtime_cameras_impl(self, import_error=self._runtime_import_error)

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
        if not self._require_permission("settings.passwords", "修改放行密码"):
            return

        while True:
            new_password = prompt_change_release_password(self)
            if new_password is None:
                return
            if not self._apply_release_password(new_password):
                continue
            self._audit_event(module="设置", action="修改放行密码", product_name="")
            QtWidgets.QMessageBox.information(self, "修改放行密码", "放行密码已更新。")
            return

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _show_tower_light_settings_dialog(self) -> None:
        if not self._require_permission("settings.tower_light", "三色灯时序设置"):
            return

        before = dict(self._tower_light_settings)
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
        self._audit_event(
            module="三色灯设置",
            action="修改时序参数",
            before_value=str(before),
            after_value=str(new_settings),
            product_name="",
        )
        self._bottom_status_bar.showMessage("\u5854\u706f\u65f6\u5e8f\u53c2\u6570\u5df2\u66f4\u65b0", 3000)

    def _on_runtime_camera_layout_settings_changed(self, settings: dict) -> None:
        merged = dict(self._runtime_mode_settings)
        merged.update(dict(settings or {}))
        try:
            self._runtime_mode_store.save(merged)
        except Exception as exc:
            self._bottom_status_bar.showMessage(
                f"\u8fd0\u884c\u753b\u9762\u5e03\u5c40\u4fdd\u5b58\u5931\u8d25: {exc}",
                5000,
            )
            return
        self._runtime_mode_settings = merged

    def _connect_signals(self) -> None:
        # ToolPage → MainWindow（跨组件协调）
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.productDeleteRequested.connect(self._on_product_delete_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(self._sync_shell_status)
        self.tool_page.sessionLoaded.connect(self._sync_permission_ui)
        self.tool_page.sessionLoaded.connect(self._restore_runtime_camera_bindings_from_session)
        self.tool_page.sessionLoaded.connect(self._restore_runtime_capture_policy_from_session)
        self.tool_page.inspectionItemsChanged.connect(self._sync_configured_camera_roles)
        connect_runtime_refresh_sources(
            self.tool_page,
            self.runtime_ctrl,
            session_loaded_message="工具页会话已加载",
        )
        connect_runtime_page(
            self.runtime_page,
            self.runtime_ctrl,
            connect_handler=self._connect_runtime_cameras,
            disconnect_handler=self._disconnect_runtime_cameras,
            trigger_handler=self._trigger_runtime,
            trigger_camera_handler=self._trigger_runtime_camera,
            release_handler=self._release_runtime,
        )
        self.tool_page.debugCameraConnectRequested.connect(self._prepare_runtime_for_debug_camera)
        self.tool_page.debugCameraConnected.connect(self._on_debug_camera_connected)
        self.tool_page.cameraSettingsApplied.connect(self._on_camera_settings_applied)
        self.runtime_ctrl.previewUpdated.connect(self._on_runtime_preview_updated)
        self.runtime_ctrl.productNameChanged.connect(lambda *_: self._sync_shell_status())
        self.runtime_ctrl.activeCameraRolesChanged.connect(self._on_runtime_active_roles_changed)
        self.runtime_ctrl.triggerResultReady.connect(self._update_sidebar_runtime_result)
        self.runtime_ctrl.triggerResultReady.connect(self._on_runtime_trigger_result)
        self.runtime_ctrl.ioStatusChanged.connect(self._on_runtime_io_status_changed)
        self.runtime_page.cameraLayoutSettingsChanged.connect(
            self._on_runtime_camera_layout_settings_changed
        )

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
            self._apply_runtime_capture_directory_setting()

    def _change_language(self, language: str) -> None:
        set_language(language)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        _retranslate_shell_chrome(self)
        if hasattr(self, "runtime_page"):
            self.runtime_page.retranslate_ui()
        if hasattr(self, "tool_page") and hasattr(self.tool_page, "retranslate_ui"):
            self.tool_page.retranslate_ui()
        self._sync_permission_ui()

    def _update_brand_banner_pixmap(self) -> None:
        _update_brand_banner_pixmap_impl(self)

    def _show_about_dialog(self) -> None:
        _show_shell_about_dialog(self)

    def _configured_runtime_camera_roles(self) -> list[str]:
        roles: list[str] = []
        capture_mode_roles: list[str] = []
        try:
            session_data = self.session.load_session()
            roles.extend(getattr(session_data, "runtime_camera_roles", []) or [])
        except Exception:
            pass

        try:
            capture_config = CameraSettingsStore(self.session.camera_settings_path).load_capture_config()
            if normalize_capture_mode(capture_config.get("capture_mode")) == CAPTURE_MODE_SINGLE_MULTI_LIGHT:
                for channel in list(capture_config.get("capture_channels", []) or []):
                    if not bool(channel.get("enabled", True)):
                        continue
                    role = normalize_camera_role(channel.get("role", ""))
                    if role and role not in capture_mode_roles:
                        capture_mode_roles.append(role)
        except Exception:
            pass
        if capture_mode_roles:
            return configured_camera_roles(capture_mode_roles)

        for item in list(getattr(self.tool_page, "inspection_items", []) or []):
            if not bool(getattr(item, "enabled", True)):
                continue
            role = normalize_camera_role(getattr(item, "camera_id", ""))
            if role:
                roles.append(role)

        if not roles:
            roles.extend(
                role
                for role in CAMERA_ROLES
                if self.runtime_page.camera_serial(role)
            )
        return configured_camera_roles(roles or [DEFAULT_CAMERA_ROLE])

    def _runtime_physical_camera_roles(self) -> list[str]:
        try:
            capture_config = CameraSettingsStore(self.session.camera_settings_path).load_capture_config()
            if normalize_capture_mode(capture_config.get("capture_mode")) == CAPTURE_MODE_SINGLE_MULTI_LIGHT:
                physical_roles: list[str] = []
                for channel in list(capture_config.get("capture_channels", []) or []):
                    if not bool(channel.get("enabled", True)):
                        continue
                    role = normalize_camera_role(channel.get("physical_role", ""))
                    if role and role not in physical_roles:
                        physical_roles.append(role)
                return configured_camera_roles(physical_roles or [DEFAULT_CAMERA_ROLE])
        except Exception:
            pass
        return list(CAMERA_ROLES)

    def _runtime_physical_camera_bindings(self, bindings: Optional[dict[str, str]] = None) -> dict[str, str]:
        raw_bindings = dict(bindings or {})
        physical_roles = set(self._runtime_physical_camera_roles())
        result: dict[str, str] = {}
        for role in CAMERA_ROLES:
            if role not in physical_roles:
                continue
            serial = str(raw_bindings.get(role, "") or self.runtime_page.camera_serial(role)).strip()
            if serial:
                result[role] = serial
        return result

    def _persist_product_runtime_camera_roles(self, roles: list[str]) -> None:
        session_data = self.session.load_session()
        session_data.runtime_camera_roles = configured_camera_roles(roles or [DEFAULT_CAMERA_ROLE])
        self.session.save_session(session_data)

    def _sync_configured_camera_roles(self) -> None:
        roles = self._configured_runtime_camera_roles()
        if hasattr(self, "runtime_page"):
            self.runtime_page.set_configured_camera_roles(roles)
        if hasattr(self, "tool_page"):
            self.tool_page.set_configured_camera_roles(roles)

    def _set_algorithm_engine_status(self, text: str, *, tooltip: str = "") -> None:
        legacy_keys = {
            "算法引擎：已就绪": "status.engine_ready",
            "算法引擎：加载中...": "status.engine_loading",
            "算法引擎：加载失败": "status.engine_failed",
        }
        key = legacy_keys.get(str(text or "").strip())
        if key:
            self._set_algorithm_engine_status_key(key, tooltip=tooltip)
            return
        self._algorithm_engine_status_key = ""
        self.lbl_status_engine.setText(text)
        self.lbl_status_engine.setToolTip(tooltip or text)

    def _set_algorithm_engine_status_key(self, key: str, *, tooltip: str = "") -> None:
        self._algorithm_engine_status_key = str(key or "").strip()
        text = tr(self._algorithm_engine_status_key) if self._algorithm_engine_status_key else ""
        self.lbl_status_engine.setText(text)
        self.lbl_status_engine.setToolTip(tooltip or text)

    def _set_sidebar_runtime_result(self, result: str, detail: str = "") -> None:
        normalized = str(result or "").strip().upper()
        display = str(result or "").strip() or "-"
        background = "#555555"
        border = "#666666"
        font_size = 34

        if normalized == "OK":
            display = "OK"
            background = "#379b37"
            border = "#46b346"
        elif normalized == "NG":
            display = "NG"
            background = "#dc1e1e"
            border = "#ef4444"
        elif normalized in {"ERROR", "BLOCKED"}:
            display = normalized
            background = "#dc1e1e"
            border = "#ef4444"

        if len(display) > 6:
            font_size = 16
        elif len(display) > 2:
            font_size = 20

        self.sidebar_runtime_result_frame.setStyleSheet(
            f"#sidebarRuntimeResultFrame{{background:{background};border:1px solid {border};border-radius:6px;}}"
        )
        self.lbl_sidebar_runtime_result.setText(display)
        self.lbl_sidebar_runtime_result.setToolTip(detail or display)
        self.lbl_sidebar_runtime_result.setStyleSheet(
            f"color:white;font-size:{font_size}px;font-weight:bold;"
        )

    @QtCore.Slot(str, str)
    def _update_sidebar_runtime_result(self, result: str, detail: str) -> None:
        self._set_sidebar_runtime_result(result, detail)

    @QtCore.Slot(bool, str, object)
    def _on_runtime_io_status_changed(self, ready: bool, detail: str, controller: object) -> None:
        self.lbl_status_io_dot.setStyleSheet(
            "color:#2ea043;font-size:14px;font-weight:bold;" if ready else "color:#c74e39;font-size:14px;font-weight:bold;"
        )
        self.lbl_status_io_text.setText(tr("status.io_ready") if ready else tr("status.io_failed"))
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

    def _open_workspace_root(self) -> None:
        self._open_in_explorer(str(embedding_test_root(__file__)))

    def _open_runtime_records_dir(self) -> None:
        self._open_in_explorer(self.runtime_ctrl.runtime_records_directory())

    def _open_runtime_capture_dir(self) -> None:
        self._open_in_explorer(self.runtime_ctrl.runtime_capture_directory())

    def _apply_runtime_records_directory_setting(self) -> None:
        configured_dir = str(self._runtime_record_settings.get("runtime_records_dir", "")).strip()
        self.runtime_ctrl.update_runtime_records_directory(
            configured_dir or (Path(self.session.product_dir) / "runtime_records")
        )

    def _apply_runtime_capture_directory_setting(self) -> None:
        configured_dir = str(self._runtime_record_settings.get("runtime_images_dir", "")).strip()
        self.runtime_ctrl.update_runtime_capture_directory(
            configured_dir or (Path(self.session.product_dir) / "runtime_capture")
        )

    def _show_runtime_records_directory_dialog(self) -> None:
        if not self._require_permission("settings.record_path", "运行记录保存目录"):
            return
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

        settings = dict(self._runtime_record_settings)
        settings["runtime_records_dir"] = str(selected_dir).strip()
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
        self._audit_event(
            module="设置",
            action="修改运行记录目录",
            before_value=current_dir,
            after_value=str(selected_dir),
            product_name="",
        )
        self._bottom_status_bar.showMessage("运行记录保存目录已更新", 3000)

    def _show_runtime_capture_directory_dialog(self) -> None:
        if not self._require_permission("settings.record_path", "运行图片保存目录"):
            return
        current_dir = str(self._runtime_record_settings.get("runtime_images_dir", "")).strip()
        if not current_dir:
            current_dir = str(Path(self.session.product_dir) / "runtime_capture")
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "保存图片路径",
            current_dir,
        )
        if not selected_dir:
            return

        settings = dict(self._runtime_record_settings)
        settings["runtime_images_dir"] = str(selected_dir).strip()
        try:
            self._runtime_record_store.save(settings)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "保存图片路径",
                f"保存运行图片目录失败：\n{exc}",
            )
            return

        self._runtime_record_settings = settings
        self._apply_runtime_capture_directory_setting()
        self._audit_event(
            module="设置",
            action="修改运行图片目录",
            before_value=current_dir,
            after_value=str(selected_dir),
            product_name="",
        )
        self._bottom_status_bar.showMessage("运行图片保存目录已更新", 3000)

    def _reload_debug_session(self) -> None:
        _reload_debug_session_impl(self)

    def _show_connect_dialog(self) -> None:
        if not self._require_permission("runtime.connect_camera", "连接相机"):
            return
        if self.runtime_ctrl.connected_roles():
            QtWidgets.QMessageBox.information(self, "连接相机", "相机已经连接。")
            self.runtime_ctrl.refresh_all_status("相机已经连接")
            return
        physical_roles = self._runtime_physical_camera_roles()
        configured_roles = self._configured_runtime_camera_roles()
        single_multi_light = set(physical_roles) != set(CAMERA_ROLES)
        result = prompt_connect_camera_bindings(
            self,
            cam1_serial=self.runtime_page.camera_serial("cam1"),
            cam2_serial=self.runtime_page.camera_serial("cam2"),
            cam3_serial=self.runtime_page.camera_serial("cam3"),
            enabled_roles=physical_roles if single_multi_light else configured_roles,
            visible_roles=physical_roles if single_multi_light else None,
        )
        if result is None:
            return
        serials, enabled_roles = result
        connect_roles = physical_roles if single_multi_light else enabled_roles
        missing_roles = [
            role
            for role in connect_roles
            if not str(serials.get(role, "")).strip()
        ]
        if missing_roles:
            QtWidgets.QMessageBox.warning(
                self,
                "连接相机",
                "已启用的相机需要填写序列号：" + ", ".join(missing_roles),
            )
            return
        for role in CAMERA_ROLES:
            serial = str(serials.get(role, "")).strip()
            self.runtime_page.set_camera_serial(role, serial)
        self._persist_runtime_camera_bindings(serials)
        self._persist_product_runtime_camera_roles(configured_roles if single_multi_light else enabled_roles)
        self._sync_configured_camera_roles()
        self._audit_event(
            module="运行",
            action="连接相机",
            after_value=", ".join(
                f"{role}={serials.get(role, '')}" for role in connect_roles
            ),
        )
        self.runtime_page.connectCamerasRequested.emit(
            self._runtime_physical_camera_bindings(serials)
        )

    def _show_release_dialog(self) -> None:
        if not self._require_permission("runtime.release_ng", "NG放行"):
            return
        pwd, ok = prompt_password_dialog(
            self,
            title=tr("dialog.release_title"),
            label=tr("dialog.release_label"),
        )
        if ok and pwd:
            self.runtime_page.releaseRequested.emit(pwd)

    @QtCore.Slot(str, str)
    def _on_runtime_trigger_result(self, result: str, _detail: str) -> None:
        if str(result).strip().upper() != "NG":
            return
        if not self._auto_show_release_dialog_on_ng:
            return
        QtCore.QTimer.singleShot(80, self._show_release_dialog)

    # ------------------------------------------------------------------
    # 跨组件协调（ToolPage 请求 → 先处理运行链路 → 再委托 ToolPage）
    # ------------------------------------------------------------------

    def _on_product_change_request(self, new_name: str) -> None:
        if not self._require_permission("product.select", "选择产品"):
            self.tool_page.refresh_product_selector()
            return
        before = str(self.session.current_product or "")
        _on_product_change_request_impl(self, new_name)
        after = str(self.session.current_product or "")
        if before != after:
            self._audit_event(
                module="产品",
                action="选择产品",
                before_value=before,
                after_value=after,
            )
        self._sync_permission_ui()

    def _on_product_delete_request(self, product_name: str) -> None:
        name = str(product_name or "").strip()
        if not name:
            return
        if name == "Default":
            QtWidgets.QMessageBox.warning(
                self,
                "\u5220\u9664\u4ea7\u54c1",
                "Default \u4ea7\u54c1\u4e0d\u80fd\u5220\u9664",
            )
            return
        if not self._require_permission("product.delete", "删除产品"):
            return
        ret = QtWidgets.QMessageBox.question(
            self,
            "\u5220\u9664\u4ea7\u54c1",
            f"\u786e\u8ba4\u5220\u9664\u4ea7\u54c1 {name}?\n"
            "\u4ea7\u54c1\u76ee\u5f55\u4f1a\u79fb\u52a8\u5230 _deleted\uff0c\u53ef\u624b\u52a8\u6062\u590d\u3002",
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        self.runtime_ctrl.disconnect(silent=True)
        error = self.session.delete_product(name)
        if error:
            QtWidgets.QMessageBox.critical(self, "\u5220\u9664\u4ea7\u54c1", error)
            return

        self.tool_page.refresh_product_selector()
        self.tool_page.apply_product_switch(self.session.current_product)
        if is_qr_core_ready():
            self._preload_current_embedding_model()
        self._sync_shell_status()
        self.runtime_ctrl.refresh_all_status("\u4ea7\u54c1\u5df2\u5220\u9664\uff0c\u8bf7\u91cd\u65b0\u8fde\u63a5\u8fd0\u884c\u94fe\u8def")
        self._audit_event(
            module="产品",
            action="删除产品",
            before_value=name,
            product_name=name,
        )
        self._bottom_status_bar.showMessage(f"\u4ea7\u54c1\u5df2\u5220\u9664: {name}", 3000)

    def _on_session_clear_request(self) -> None:
        _on_session_clear_request_impl(self)

    # ------------------------------------------------------------------
    # 预览图更新（RuntimeController Signal → RuntimeModePage）
    # ------------------------------------------------------------------

    def _on_runtime_preview_updated(self, role: str, source: object) -> None:
        _on_runtime_preview_updated_impl(self, role, source)

    # ------------------------------------------------------------------
    # 窗口生命周期
    # ------------------------------------------------------------------

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "lbl_brand_banner"):
            self.lbl_brand_banner.update()

    def _mark_initial_ui_ready(self) -> None:
        self._allow_initial_tool_session_load = True
        self._initial_ui_ready_timer_started = False

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._allow_initial_tool_session_load or self._initial_ui_ready_timer_started:
            return
        self._initial_ui_ready_timer_started = True
        QtCore.QTimer.singleShot(120, self._mark_initial_ui_ready)

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
        self.runtime_ctrl.shutdown_persistence(wait=True)
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
