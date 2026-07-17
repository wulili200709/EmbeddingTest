"""Debug-camera helper methods for ToolPage."""

from __future__ import annotations

import configparser
import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from common.app_paths import packaged_embedding_test_root, packaged_repo_root
from common.camera_roles import CAMERA_ROLES, DEFAULT_CAMERA_ROLE, normalize_camera_role
from infrastructure.camera_settings_store import (
    CAPTURE_DEFAULT_EXPOSURE_US,
    CAPTURE_LIGHT_OUTPUTS,
    CAPTURE_MODE_FLEXIBLE,
    CAPTURE_MODE_INDEPENDENT,
    CameraSettingsStore,
    LIGHT_SOURCE_MODE_BOARD_IO,
    light_source_mode_from_mapping,
    normalize_capture_light_output,
    normalize_capture_mode,
    uses_channel_capture_mapping,
)
from ui.i18n import tr


def _embedding_test_root(tool_page) -> Path:
    return packaged_embedding_test_root(__file__)


def _selected_debug_camera_serial(tool_page) -> str:
    return str(tool_page.cmb_debug_camera.currentData() or "").strip()


def _selected_debug_camera_role(tool_page) -> str:
    combo = getattr(tool_page, "cmb_debug_camera_role", None)
    if combo is None:
        getter = getattr(tool_page, "current_camera_role", None)
        if callable(getter):
            return normalize_camera_role(getter(), default=DEFAULT_CAMERA_ROLE)
        return DEFAULT_CAMERA_ROLE
    return normalize_camera_role(combo.currentData() or combo.currentText(), default=DEFAULT_CAMERA_ROLE)


def _debug_capture_channel_for_role(tool_page, role: object = None) -> dict[str, object]:
    logical_role = normalize_camera_role(
        role if role is not None else _selected_debug_camera_role(tool_page),
        default=DEFAULT_CAMERA_ROLE,
    )
    try:
        config = tool_page._camera_settings_store.load_capture_config()
    except Exception:
        return {}
    if not uses_channel_capture_mapping(config.get("capture_mode")):
        return {}
    for raw in list(config.get("capture_channels", []) or []):
        if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
            continue
        if normalize_camera_role(raw.get("role"), default="") != logical_role:
            continue
        channel = dict(raw)
        channel["role"] = logical_role
        channel["physical_role"] = normalize_camera_role(
            raw.get("physical_role"), default=logical_role
        ) or logical_role
        channel["light_output"] = normalize_capture_light_output(raw.get("light_output"))
        return channel
    return {}


def _debug_physical_camera_role(tool_page, role: object = None) -> str:
    logical_role = normalize_camera_role(
        role if role is not None else _selected_debug_camera_role(tool_page),
        default=DEFAULT_CAMERA_ROLE,
    )
    channel = _debug_capture_channel_for_role(tool_page, logical_role)
    return normalize_camera_role(channel.get("physical_role"), default=logical_role) or logical_role


def _debug_capture_light_index(tool_page, role: object = None) -> int:
    channel = _debug_capture_channel_for_role(tool_page, role)
    output = normalize_capture_light_output(channel.get("light_output"))
    for index, camera_role in enumerate(CAMERA_ROLES, start=1):
        if output == f"DO_LIGHT_CAM{index}" or output.endswith(camera_role.upper()):
            return index
    return 1


def _load_debug_role_binding(tool_page, role: str) -> str:
    role_text = _debug_physical_camera_role(tool_page, role)
    preferred_serial = str(CameraSettingsStore().serial_for_role(role_text) or "").strip()
    if preferred_serial:
        return preferred_serial
    preferred_serial = str(tool_page._camera_settings_store.serial_for_role(role_text) or "").strip()
    if preferred_serial:
        CameraSettingsStore().save_serial_for_role(role_text, preferred_serial)
        return preferred_serial
    session_data = tool_page.session.load_session()
    legacy_serial = str(getattr(session_data, "runtime_camera_serials", {}).get(role_text, "") or "").strip()
    if legacy_serial:
        CameraSettingsStore().save_serial_for_role(role_text, legacy_serial)
    return legacy_serial


def _save_debug_role_binding(tool_page, role: str, serial: str) -> None:
    role_text = _debug_physical_camera_role(tool_page, role)
    serial_text = str(serial or "").strip()
    CameraSettingsStore().save_serial_for_role(role_text, serial_text)


def _apply_debug_role_binding_to_camera_combo(tool_page) -> None:
    combo = getattr(tool_page, "cmb_debug_camera", None)
    if combo is None:
        return
    role = tool_page._selected_debug_camera_role()
    preferred_serial = tool_page._load_debug_role_binding(role)
    if not preferred_serial:
        preferred_serial = CameraSettingsStore().serial_for_role(role)
    if not preferred_serial:
        return
    index = combo.findData(preferred_serial)
    if index >= 0 and combo.currentIndex() != index:
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)


def _refresh_debug_role_status(tool_page) -> None:
    role = tool_page._selected_debug_camera_role()
    physical_role = _debug_physical_camera_role(tool_page, role)
    channel = _debug_capture_channel_for_role(tool_page, role)
    current = getattr(tool_page, "lbl_debug_current_role", None)
    if current is not None:
        if channel:
            current.setText(
                tr(
                    "debug.current_debug_mapping",
                    role=role,
                    physical_role=physical_role,
                    light=str(channel.get("light_output", "") or "-"),
                    exposure=float(channel.get("exposure_time_us", CAPTURE_DEFAULT_EXPOSURE_US) or CAPTURE_DEFAULT_EXPOSURE_US),
                )
            )
        else:
            current.setText(tr("debug.current_debug_role", role=role))


def _debug_camera_settings_payload_from_ui(tool_page) -> dict[str, object]:
    return {
        "trigger_mode": str(tool_page.cmb_debug_trigger_mode.currentText() or "continuous"),
        "exposure_time_us": float(tool_page.spin_debug_exposure.value()),
        "gain": float(tool_page.spin_debug_gain.value()),
        "digital_shift_enable": bool(tool_page.chk_debug_digital_shift_enable.isChecked()),
        "digital_shift": float(tool_page.spin_debug_digital_shift.value()),
        "light_source_mode": str(
            tool_page.cmb_debug_light_source_mode.currentData() or LIGHT_SOURCE_MODE_BOARD_IO
        ),
    }


def _load_saved_debug_camera_settings_to_ui(tool_page, serial: str) -> bool:
    role = tool_page._selected_debug_camera_role()
    physical_role = _debug_physical_camera_role(tool_page, role)
    channel = _debug_capture_channel_for_role(tool_page, role)
    tool_page._debug_camera_block_spin_apply = True
    try:
        payload = tool_page._camera_settings_store.load_for_role(physical_role, serial=serial) or {}
        if payload.get("exposure_time_us") is not None:
            tool_page.spin_debug_exposure.setValue(float(payload["exposure_time_us"]))
        if payload.get("gain") is not None:
            tool_page.spin_debug_gain.setValue(float(payload["gain"]))
        if payload.get("digital_shift_enable") is not None:
            tool_page.chk_debug_digital_shift_enable.setChecked(bool(payload["digital_shift_enable"]))
        if payload.get("digital_shift") is not None:
            tool_page.spin_debug_digital_shift.setValue(float(payload["digital_shift"]))
        trigger_mode = str(payload.get("trigger_mode") or "").strip()
        if trigger_mode:
            tool_page.cmb_debug_trigger_mode.setCurrentText(trigger_mode)
        light_source_mode = light_source_mode_from_mapping(payload)
        index = tool_page.cmb_debug_light_source_mode.findData(light_source_mode)
        if index >= 0:
            tool_page.cmb_debug_light_source_mode.setCurrentIndex(index)
        if channel:
            tool_page.spin_debug_exposure.setValue(
                float(channel.get("exposure_time_us", CAPTURE_DEFAULT_EXPOSURE_US) or CAPTURE_DEFAULT_EXPOSURE_US)
            )
            tool_page.spin_debug_gain.setValue(float(channel.get("gain", 0.0) or 0.0))
            board_index = tool_page.cmb_debug_light_source_mode.findData(LIGHT_SOURCE_MODE_BOARD_IO)
            if board_index >= 0:
                tool_page.cmb_debug_light_source_mode.setCurrentIndex(board_index)
        return bool(payload or channel)
    finally:
        tool_page._debug_camera_block_spin_apply = False


def _save_debug_camera_settings(tool_page, serial: str, settings: dict[str, object]) -> None:
    serial_text = str(serial).strip()
    if not serial_text:
        return
    role = tool_page._selected_debug_camera_role()
    physical_role = _debug_physical_camera_role(tool_page, role)
    tool_page._camera_settings_store.save_for_role(physical_role, serial_text, settings)
    channel = _debug_capture_channel_for_role(tool_page, role)
    if not channel:
        return
    try:
        config = tool_page._camera_settings_store.load_capture_config()
        channels = list(config.get("capture_channels", []) or [])
        for raw in channels:
            if not isinstance(raw, dict):
                continue
            if normalize_camera_role(raw.get("role"), default="") != role:
                continue
            raw["exposure_time_us"] = float(settings.get("exposure_time_us", CAPTURE_DEFAULT_EXPOSURE_US))
            raw["gain"] = float(settings.get("gain", 0.0))
            break
        tool_page._camera_settings_store.save_capture_config(config.get("capture_mode"), channels)
    except Exception:
        return


def _capture_light_output_label(output: object) -> str:
    output_text = normalize_capture_light_output(output)
    label_keys = {
        "DO_LIGHT_CAM1": "debug.io_name.light_cam1",
        "DO_LIGHT_CAM2": "debug.io_name.light_cam2",
        "DO_LIGHT_CAM3": "debug.io_name.light_cam3",
    }
    return str(label_keys.get(output_text, output_text))


def _populate_capture_light_combo(combo: QtWidgets.QComboBox, selected: object = "") -> None:
    current = normalize_capture_light_output(selected)
    blocker = QtCore.QSignalBlocker(combo)
    try:
        combo.clear()
        for output in CAPTURE_LIGHT_OUTPUTS:
            combo.addItem(tr(_capture_light_output_label(output)), output)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
    finally:
        del blocker


def _capture_mode_from_ui(tool_page) -> str:
    combo = getattr(tool_page, "cmb_capture_mode", None)
    if combo is None:
        return CAPTURE_MODE_INDEPENDENT
    return normalize_capture_mode(combo.currentData() or combo.currentText())


def _capture_physical_role(role: str, mode: str) -> str:
    return role if mode == CAPTURE_MODE_INDEPENDENT else DEFAULT_CAMERA_ROLE


def _update_capture_channel_visibility(tool_page) -> None:
    visible = _capture_mode_from_ui(tool_page) != CAPTURE_MODE_INDEPENDENT
    for attr in ("lbl_capture_channel_title", "capture_channel_table"):
        widget = getattr(tool_page, attr, None)
        if widget is not None:
            widget.setVisible(visible)


def _sync_capture_camera_roles(tool_page) -> None:
    sync_roles = getattr(tool_page.window(), "_sync_configured_camera_roles", None)
    if callable(sync_roles):
        sync_roles()


def _capture_channels_from_ui(tool_page) -> list[dict[str, object]]:
    table = getattr(tool_page, "capture_channel_table", None)
    if table is None:
        return []
    mode = _capture_mode_from_ui(tool_page)
    channels: list[dict[str, object]] = []
    for row in range(table.rowCount()):
        role_item = table.item(row, 1)
        role = normalize_camera_role(role_item.text() if role_item is not None else "", default="")
        if not role:
            continue
        enabled_item = table.item(row, 0)
        enabled = (
            enabled_item.checkState() == QtCore.Qt.CheckState.Checked
            if enabled_item is not None
            else True
        )
        physical_widget = table.cellWidget(row, 2)
        light_widget = table.cellWidget(row, 3)
        exposure_widget = table.cellWidget(row, 4)
        gain_widget = table.cellWidget(row, 5)
        physical_role = (
            physical_widget.currentData()
            if isinstance(physical_widget, QtWidgets.QComboBox)
            else _capture_physical_role(role, mode)
        )
        light_output = (
            light_widget.currentData()
            if isinstance(light_widget, QtWidgets.QComboBox)
            else f"DO_LIGHT_CAM{row + 1}"
        )
        exposure = (
            float(exposure_widget.value())
            if isinstance(exposure_widget, QtWidgets.QDoubleSpinBox)
            else CAPTURE_DEFAULT_EXPOSURE_US
        )
        gain = float(gain_widget.value()) if isinstance(gain_widget, QtWidgets.QDoubleSpinBox) else 0.0
        channels.append(
            {
                "enabled": enabled,
                "role": role,
                "physical_role": normalize_camera_role(
                    physical_role,
                    default=_capture_physical_role(role, mode),
                ),
                "light_output": normalize_capture_light_output(light_output, default=f"DO_LIGHT_CAM{row + 1}"),
                "exposure_time_us": exposure,
                "gain": gain,
                "stable_delay_ms": 50,
            }
        )
    return channels


def _set_capture_channel_row(tool_page, row: int, channel: dict[str, object]) -> None:
    table = getattr(tool_page, "capture_channel_table", None)
    if table is None:
        return
    role = normalize_camera_role(channel.get("role"), default=CAMERA_ROLES[row])
    enabled_item = table.item(row, 0)
    if enabled_item is not None:
        enabled_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if bool(channel.get("enabled", True))
            else QtCore.Qt.CheckState.Unchecked
        )
    role_item = table.item(row, 1)
    if role_item is not None:
        role_item.setText(role)

    physical_combo = table.cellWidget(row, 2)
    if isinstance(physical_combo, QtWidgets.QComboBox):
        physical_role = normalize_camera_role(
            channel.get("physical_role"),
            default=role,
        )
        index = physical_combo.findData(physical_role)
        physical_combo.setCurrentIndex(index if index >= 0 else row)

    light_combo = table.cellWidget(row, 3)
    if isinstance(light_combo, QtWidgets.QComboBox):
        _populate_capture_light_combo(light_combo, channel.get("light_output"))

    exposure_spin = table.cellWidget(row, 4)
    if isinstance(exposure_spin, QtWidgets.QDoubleSpinBox):
        exposure_spin.setValue(
            float(channel.get("exposure_time_us", CAPTURE_DEFAULT_EXPOSURE_US) or CAPTURE_DEFAULT_EXPOSURE_US)
        )

    gain_spin = table.cellWidget(row, 5)
    if isinstance(gain_spin, QtWidgets.QDoubleSpinBox):
        gain_spin.setValue(float(channel.get("gain", 0.0) or 0.0))


def _load_capture_config_to_ui(tool_page) -> None:
    if not hasattr(tool_page, "capture_channel_table"):
        return
    tool_page._capture_config_loading = True
    try:
        config = tool_page._camera_settings_store.load_capture_config()
        mode = normalize_capture_mode(config.get("capture_mode"))
        combo = getattr(tool_page, "cmb_capture_mode", None)
        if combo is not None:
            index = combo.findData(mode)
            combo.setCurrentIndex(index if index >= 0 else 0)
        channels = list(config.get("capture_channels", []) or [])
        for row, channel in enumerate(channels[: len(CAMERA_ROLES)]):
            _set_capture_channel_row(tool_page, row, dict(channel))
    finally:
        tool_page._capture_config_loading = False
        _update_capture_channel_visibility(tool_page)


def _save_capture_config_from_ui(tool_page) -> None:
    if getattr(tool_page, "_capture_config_loading", False):
        return
    if not hasattr(tool_page, "capture_channel_table"):
        return
    mode = _capture_mode_from_ui(tool_page)
    tool_page._camera_settings_store.save_capture_config(mode, _capture_channels_from_ui(tool_page))


def _on_capture_mode_changed(tool_page, _index: int = 0) -> None:
    _update_capture_channel_visibility(tool_page)
    table = getattr(tool_page, "capture_channel_table", None)
    if table is not None and not getattr(tool_page, "_capture_config_loading", False):
        mode = _capture_mode_from_ui(tool_page)
        for row in range(table.rowCount()):
            role_item = table.item(row, 1)
            role = normalize_camera_role(
                role_item.text() if role_item is not None else "",
                default=CAMERA_ROLES[row],
            )
            physical_combo = table.cellWidget(row, 2)
            if not isinstance(physical_combo, QtWidgets.QComboBox):
                continue
            if mode == CAPTURE_MODE_FLEXIBLE:
                # Flexible mapping keeps the user-selected physical camera for
                # each logical channel, including shared-camera exposures.
                continue
            target_role = _capture_physical_role(role, mode)
            index = physical_combo.findData(target_role)
            if index >= 0:
                physical_combo.setCurrentIndex(index)
    _save_capture_config_from_ui(tool_page)
    _sync_capture_camera_roles(tool_page)


def _on_capture_channel_item_changed(tool_page, _item=None) -> None:
    _save_capture_config_from_ui(tool_page)
    _sync_capture_camera_roles(tool_page)


def _on_capture_channel_editor_changed(tool_page, *_args) -> None:
    _save_capture_config_from_ui(tool_page)
    _apply_debug_role_binding_to_camera_combo(tool_page)
    _refresh_debug_role_status(tool_page)


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


def _load_nkio_runtime_options(mapping_path: str | Path) -> dict[str, str]:
    try:
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("nkio_config_path", "nkio_dll_path"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result[key] = text
    return result


def _selected_nkio_config_from_sdk_bin(root: Path):
    select_ini = root / "NKDIOLC_SDK" / "Bin" / "select.ini"
    if not select_ini.exists():
        return None

    parser = configparser.ConfigParser()
    parser.optionxform = str
    try:
        parser.read(select_ini, encoding="utf-8")
    except Exception:
        return None

    if not parser.has_section("SELECTED"):
        return None

    config_path = str(parser.get("SELECTED", "ConfigPath", fallback="") or "").strip()
    if not config_path:
        return None

    relative_path = config_path.lstrip("/\\").replace("/", "\\")
    candidate = root / "NKDIOLC_SDK" / "Bin" / Path(relative_path)
    if candidate.exists():
        return str(candidate)
    return None


def _find_debug_nkio_config_path(tool_page):
    mapping_path = tool_page._default_io_mapping_path()
    runtime_options = _load_nkio_runtime_options(mapping_path)
    configured_path = runtime_options.get("nkio_config_path")
    if configured_path:
        configured = Path(configured_path)
        if configured.exists():
            return str(configured)

    root = packaged_repo_root(__file__)
    selected_path = _selected_nkio_config_from_sdk_bin(root)
    if selected_path:
        return selected_path
    candidates = [
        root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Debug" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "Bin" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
        root / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _find_debug_nkio_dll_path(tool_page):
    mapping_path = tool_page._default_io_mapping_path()
    runtime_options = _load_nkio_runtime_options(mapping_path)
    configured_path = runtime_options.get("nkio_dll_path")
    if not configured_path:
        return None
    configured = Path(configured_path)
    if configured.exists():
        return str(configured)
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
