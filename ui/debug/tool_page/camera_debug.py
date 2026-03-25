"""Debug-camera helper methods for ToolPage."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui


def _embedding_test_root(tool_page) -> Path:
    return Path(__file__).resolve().parents[3]


def _selected_debug_camera_serial(tool_page) -> str:
    return str(tool_page.cmb_debug_camera.currentData() or "").strip()


def _selected_debug_camera_role(tool_page) -> str:
    combo = getattr(tool_page, "cmb_debug_camera_role", None)
    if combo is None:
        getter = getattr(tool_page, "current_camera_role", None)
        if callable(getter):
            return str(getter() or "cam1").strip() or "cam1"
        return "cam1"
    return str(combo.currentData() or combo.currentText() or "cam1").strip() or "cam1"


def _load_debug_role_binding(tool_page, role: str) -> str:
    role_text = str(role or "").strip() or "cam1"
    preferred_serial = str(tool_page._camera_settings_store.serial_for_role(role_text) or "").strip()
    if preferred_serial:
        return preferred_serial
    if role_text == "cam2":
        return str(tool_page.session.load_session().runtime_cam2_serial or "").strip()
    return str(tool_page.session.load_session().runtime_cam1_serial or "").strip()


def _save_debug_role_binding(tool_page, role: str, serial: str) -> None:
    role_text = str(role or "").strip() or "cam1"
    serial_text = str(serial or "").strip()
    current_settings = tool_page._camera_settings_store.load_for_role(role_text, serial=serial_text) or {}
    tool_page._camera_settings_store.save_for_role(role_text, serial_text, current_settings)
    session_data = tool_page.session.load_session()
    if role_text == "cam2":
        session_data.runtime_cam2_serial = serial_text
    else:
        session_data.runtime_cam1_serial = serial_text
    tool_page.session.save_session(session_data)


def _apply_debug_role_binding_to_camera_combo(tool_page) -> None:
    combo = getattr(tool_page, "cmb_debug_camera", None)
    if combo is None:
        return
    role = tool_page._selected_debug_camera_role()
    preferred_serial = tool_page._load_debug_role_binding(role)
    if not preferred_serial:
        preferred_serial = tool_page._camera_settings_store.serial_for_role(role)
    if not preferred_serial:
        return
    index = combo.findData(preferred_serial)
    if index >= 0 and combo.currentIndex() != index:
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)


def _refresh_debug_role_status(tool_page) -> None:
    role = tool_page._selected_debug_camera_role()
    current = getattr(tool_page, "lbl_debug_current_role", None)
    if current is not None:
        current.setText(f"当前调试角色：{role}")


def _debug_camera_settings_payload_from_ui(tool_page) -> dict[str, object]:
    return {
        "trigger_mode": str(tool_page.cmb_debug_trigger_mode.currentText() or "continuous"),
        "exposure_time_us": float(tool_page.spin_debug_exposure.value()),
        "gain": float(tool_page.spin_debug_gain.value()),
    }


def _load_saved_debug_camera_settings_to_ui(tool_page, serial: str) -> bool:
    role = tool_page._selected_debug_camera_role()
    tool_page._debug_camera_block_spin_apply = True
    try:
        payload = tool_page._camera_settings_store.load_for_role(role, serial=serial)
        if not payload:
            return False
        if payload.get("exposure_time_us") is not None:
            tool_page.spin_debug_exposure.setValue(float(payload["exposure_time_us"]))
        if payload.get("gain") is not None:
            tool_page.spin_debug_gain.setValue(float(payload["gain"]))
        trigger_mode = str(payload.get("trigger_mode") or "").strip()
        if trigger_mode:
            tool_page.cmb_debug_trigger_mode.setCurrentText(trigger_mode)
        return True
    finally:
        tool_page._debug_camera_block_spin_apply = False


def _save_debug_camera_settings(tool_page, serial: str, settings: dict[str, object]) -> None:
    serial_text = str(serial).strip()
    if not serial_text:
        return
    role = tool_page._selected_debug_camera_role()
    tool_page._camera_settings_store.save_for_role(role, serial_text, settings)


def _set_debug_preview_placeholder(tool_page, text: str) -> None:
    tool_page.view_debug_camera.set_runtime_pixmap(None, placeholder=text)


def _show_debug_preview_image(tool_page, image: QtGui.QImage) -> None:
    tool_page.view_debug_camera.set_runtime_pixmap(QtGui.QPixmap.fromImage(image))


def _set_debug_preview_running(tool_page, running: bool) -> None:
    if not hasattr(tool_page, "btn_debug_live_preview"):
        return
    tool_page.btn_debug_live_preview.blockSignals(True)
    tool_page.btn_debug_live_preview.setChecked(running)
    tool_page.btn_debug_live_preview.setText("停止预览" if running else "实时预览")
    tool_page.btn_debug_live_preview.blockSignals(False)


def _selected_debug_camera_info(tool_page):
    serial = str(tool_page.cmb_debug_camera.currentData() or "").strip()
    for info in tool_page._debug_camera_infos:
        if str(getattr(info, "serial_number", "")) == serial:
            return info
    return None


def _debug_camera_device(tool_page):
    if tool_page._debug_frame_grab_service is None:
        return None
    try:
        return tool_page._debug_frame_grab_service.get_device("debug")
    except Exception:
        return None


def _default_io_mapping_path(tool_page) -> str:
    return str(tool_page._embedding_test_root() / "config" / "defaults" / "io_mapping.json")


def _find_debug_nkio_config_path(tool_page):
    root = tool_page._embedding_test_root().parent
    candidates = [
        root / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _cleanup_debug_hardware(tool_page) -> None:
    try:
        tool_page._stop_debug_camera_preview()
    except Exception:
        pass
    try:
        tool_page._close_debug_io(silent=True)
    except Exception:
        pass
    try:
        tool_page._disconnect_debug_camera()
    except Exception:
        pass
