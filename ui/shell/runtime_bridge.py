from __future__ import annotations

from typing import Optional

from PySide6 import QtCore

from algorithms.proxy import is_ready as is_qr_core_ready
from infrastructure.camera_settings_store import CameraSettingsStore
from ui.window_common import update_runtime_preview


_STARTUP_AUTO_CONNECT_RETRY_DELAYS_MS = (5000, 10000, 15000, 30000)


def _stored_camera_serial_for_role(window, role: str) -> str:
    try:
        return CameraSettingsStore(window.session.camera_settings_path).serial_for_role(role)
    except Exception:
        return ""


def prepare_runtime_for_debug_camera(window, serial: str) -> None:
    serial_text = str(serial).strip()
    if not serial_text:
        return
    if not window.runtime_ctrl.connected_roles():
        return
    if serial_text not in set(window.runtime_page.camera_bindings().values()):
        return
    window.runtime_ctrl.disconnect(silent=True)
    window._bottom_status_bar.showMessage("已释放运行相机，切换到调试连接", 3000)


def restore_runtime_camera_bindings_from_session(window) -> None:
    session_data = window.session.load_session()
    cam1_serial = str(session_data.runtime_cam1_serial or "").strip() or _stored_camera_serial_for_role(window, "cam1")
    cam2_serial = str(session_data.runtime_cam2_serial or "").strip() or _stored_camera_serial_for_role(window, "cam2")
    window.runtime_page.edit_cam1_serial.setText(cam1_serial)
    window.runtime_page.edit_cam2_serial.setText(cam2_serial)
    if (
        cam1_serial != str(session_data.runtime_cam1_serial or "").strip()
        or cam2_serial != str(session_data.runtime_cam2_serial or "").strip()
    ):
        session_data.runtime_cam1_serial = cam1_serial
        session_data.runtime_cam2_serial = cam2_serial
        window.session.save_session(session_data)
    sync_configured_roles = getattr(window, "_sync_configured_camera_roles", None)
    if callable(sync_configured_roles):
        sync_configured_roles()


def persist_runtime_camera_bindings(
    window,
    bindings: Optional[dict[str, str]] = None,
) -> None:
    session_data = window.session.load_session()
    current_bindings = dict(bindings or window.runtime_page.camera_bindings())
    session_data.runtime_cam1_serial = (
        str(current_bindings.get("cam1", "")).strip()
        or str(session_data.runtime_cam1_serial or "").strip()
        or _stored_camera_serial_for_role(window, "cam1")
    )
    session_data.runtime_cam2_serial = (
        str(current_bindings.get("cam2", "")).strip()
        or str(session_data.runtime_cam2_serial or "").strip()
        or _stored_camera_serial_for_role(window, "cam2")
    )
    window.session.save_session(session_data)


def normalize_runtime_capture_policy(policy: str) -> str:
    return "all" if str(policy or "").strip().lower() == "all" else "ng_only"


def runtime_capture_policy_text(policy: str) -> str:
    return "全部保留" if normalize_runtime_capture_policy(policy) == "all" else "仅保留NG"


def load_runtime_capture_policy_from_session(window) -> str:
    session_data = window.session.load_session()
    return normalize_runtime_capture_policy(session_data.runtime_capture_policy)


def persist_runtime_capture_policy(window, policy: str) -> None:
    normalized = normalize_runtime_capture_policy(policy)
    session_data = window.session.load_session()
    session_data.runtime_capture_policy = normalized
    window.session.save_session(session_data)


def sync_runtime_capture_policy_actions(window) -> None:
    policy = normalize_runtime_capture_policy(window._runtime_capture_policy)
    if hasattr(window, "act_runtime_capture_keep_all"):
        window.act_runtime_capture_keep_all.setChecked(policy == "all")
    if hasattr(window, "act_runtime_capture_keep_ng_only"):
        window.act_runtime_capture_keep_ng_only.setChecked(policy == "ng_only")


def apply_runtime_capture_policy(
    window,
    policy: str,
    *,
    persist: bool,
    show_message: bool,
) -> None:
    normalized = normalize_runtime_capture_policy(policy)
    window._runtime_capture_policy = normalized
    if hasattr(window, "runtime_ctrl") and window.runtime_ctrl is not None:
        window.runtime_ctrl.set_capture_retention_policy(normalized)
    sync_runtime_capture_policy_actions(window)
    if persist:
        persist_runtime_capture_policy(window, normalized)
    if show_message:
        window._bottom_status_bar.showMessage(
            f"运行图像保存：{runtime_capture_policy_text(normalized)}",
            3000,
        )


def restore_runtime_capture_policy_from_session(window) -> None:
    apply_runtime_capture_policy(
        window,
        load_runtime_capture_policy_from_session(window),
        persist=False,
        show_message=False,
    )


def startup_auto_connect_runtime_cameras(window, *, import_error) -> None:
    if window._startup_runtime_auto_connect_done:
        return
    if import_error is not None:
        window._startup_runtime_auto_connect_done = True
        return
    if window.runtime_ctrl.connected_roles():
        window._startup_runtime_auto_connect_done = True
        return

    bindings = window.runtime_page.camera_bindings()
    if not bindings:
        restore_runtime_camera_bindings_from_session(window)
        bindings = window.runtime_page.camera_bindings()
    if not bindings:
        window._startup_runtime_auto_connect_done = True
        return

    attempt = int(getattr(window, "_startup_runtime_auto_connect_attempts", 0) or 0) + 1
    window._startup_runtime_auto_connect_attempts = attempt
    if window.runtime_ctrl.try_connect_cameras(bindings):
        window._startup_runtime_auto_connect_done = True
        window._startup_runtime_auto_connect_attempts = 0
        window._bottom_status_bar.showMessage("\u5df2\u81ea\u52a8\u8fde\u63a5\u8fd0\u884c\u76f8\u673a", 3000)
        return

    max_attempts = len(_STARTUP_AUTO_CONNECT_RETRY_DELAYS_MS) + 1
    if attempt < max_attempts:
        delay_ms = _STARTUP_AUTO_CONNECT_RETRY_DELAYS_MS[
            min(attempt - 1, len(_STARTUP_AUTO_CONNECT_RETRY_DELAYS_MS) - 1)
        ]
        window._bottom_status_bar.showMessage(
            f"\u76f8\u673a\u81ea\u52a8\u8fde\u63a5\u5931\u8d25\uff0c{delay_ms // 1000}\u79d2\u540e\u91cd\u8bd5",
            3000,
        )
        QtCore.QTimer.singleShot(delay_ms, window._startup_auto_connect_runtime_cameras)
        return

    window._startup_runtime_auto_connect_done = True
    window._startup_runtime_auto_connect_attempts = 0
    QtCore.QTimer.singleShot(0, window._show_connect_dialog)


def on_runtime_active_roles_changed(window, roles: list[str]) -> None:
    if not roles:
        return
    persist_runtime_camera_bindings(window)


def activate_runtime_workspace_legacy(window) -> str:
    if window.runtime_ctrl.connected_roles():
        return "运行链路已连接"

    bindings = window.runtime_page.camera_bindings()
    debug_serial = window.tool_page.connected_debug_camera_serial()

    if debug_serial and not str(bindings.get("cam1", "")).strip():
        window.runtime_page.edit_cam1_serial.setText(debug_serial)
        sync_configured_roles = getattr(window, "_sync_configured_camera_roles", None)
        if callable(sync_configured_roles):
            sync_configured_roles()
        bindings = window.runtime_page.camera_bindings()

    if not bindings:
        return ""

    if debug_serial and debug_serial in {str(value).strip() for value in bindings.values()}:
        window.tool_page.release_debug_camera_for_runtime()

    window.runtime_ctrl.connect_cameras(bindings)
    if window.runtime_ctrl.connected_roles():
        return "已自动切换到运行检测状态"
    return ""


def activate_runtime_workspace(window) -> str:
    return ensure_runtime_camera_connection(window)


def on_debug_camera_connected(window, role: str, serial: str) -> None:
    if window.main_pages.currentWidget() is not window.runtime_page:
        return
    runtime_message = ensure_runtime_camera_connection(window, debug_role=role, debug_serial=serial)
    if runtime_message:
        window._bottom_status_bar.showMessage(runtime_message, 3000)


def ensure_runtime_camera_connection(window, *, debug_role: str = "", debug_serial: str = "") -> str:
    if window.runtime_ctrl.connected_roles():
        return ""

    bindings = window.runtime_page.camera_bindings()
    debug_serial = str(debug_serial or window.tool_page.connected_debug_camera_serial()).strip()
    debug_role = str(debug_role or getattr(window.tool_page, "_selected_debug_camera_role", lambda: "cam1")()).strip() or "cam1"

    if debug_serial and not str(bindings.get(debug_role, "")).strip():
        if debug_role == "cam2":
            window.runtime_page.edit_cam2_serial.setText(debug_serial)
        else:
            window.runtime_page.edit_cam1_serial.setText(debug_serial)
        sync_configured_roles = getattr(window, "_sync_configured_camera_roles", None)
        if callable(sync_configured_roles):
            sync_configured_roles()
        bindings = window.runtime_page.camera_bindings()

    if not bindings:
        return ""

    if debug_serial and debug_serial in {str(value).strip() for value in bindings.values()}:
        window.tool_page.release_debug_camera_for_runtime()

    window.runtime_ctrl.connect_cameras(bindings)
    if window.runtime_ctrl.connected_roles():
        return "已自动切换到运行检测状态"
    return ""


def on_product_change_request(window, new_name: str) -> None:
    window.runtime_ctrl.disconnect(silent=True)
    window.tool_page.apply_product_switch(new_name)
    if is_qr_core_ready():
        window._preload_current_embedding_model()
    window._sync_shell_status()
    window.runtime_ctrl.refresh_all_status("产品已切换，请重新连接运行链路")


def on_session_clear_request(window) -> None:
    window.runtime_ctrl.disconnect(silent=True)
    window.tool_page.reset_for_clear()
    window._sync_shell_status()
    window.runtime_ctrl.refresh_all_status("会话已清空，请重新准备运行链路")


def on_runtime_preview_updated(window, role: str, source: object) -> None:
    update_runtime_preview(window.runtime_page, role, source)
