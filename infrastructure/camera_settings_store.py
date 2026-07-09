from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from common.app_paths import writable_embedding_test_root
from common.safe_io import atomic_write_json, load_json_with_backup


_CAMERA_SETTINGS_FILENAME = "camera_settings.json"
LIGHT_SOURCE_MODE_BOARD_IO = "board_io"
LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE = "camera_line1_strobe"

_CAMERA_APPLY_SETTING_KEYS = (
    "exposure_time_us",
    "gain",
    "trigger_mode",
    "acquisition_frame_rate_enable",
    "acquisition_frame_rate",
    "digital_shift_enable",
    "digital_shift",
)


def default_camera_settings_path() -> Path:
    return writable_embedding_test_root(__file__) / "config" / _CAMERA_SETTINGS_FILENAME


def hik_settings_kwargs_from_mapping(
    settings: Mapping[str, Any] | None,
    *,
    default_trigger_mode: str = "software",
    force_trigger_mode: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_settings_payload(settings)
    trigger_mode = force_trigger_mode or str(
        normalized.get("trigger_mode") or default_trigger_mode
    )
    payload: dict[str, Any] = {"trigger_mode": trigger_mode}
    for key in _CAMERA_APPLY_SETTING_KEYS:
        if key == "trigger_mode":
            continue
        if key in normalized:
            payload[key] = normalized[key]
    return payload


def normalize_light_source_mode(
    value: object,
    *,
    default: str = LIGHT_SOURCE_MODE_BOARD_IO,
) -> str:
    text = str(value or "").strip().lower()
    if text in {
        LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE,
        "camera_gpio_strobe",
        "camera_strobe",
        "camera_line1",
        "line1",
    }:
        return LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE
    if text in {LIGHT_SOURCE_MODE_BOARD_IO, "board_do", "io", "do"}:
        return LIGHT_SOURCE_MODE_BOARD_IO
    return normalize_light_source_mode(default, default=LIGHT_SOURCE_MODE_BOARD_IO) if text else default


def light_source_mode_from_mapping(
    settings: Mapping[str, Any] | None,
    *,
    default: str = LIGHT_SOURCE_MODE_BOARD_IO,
) -> str:
    if not isinstance(settings, Mapping):
        return normalize_light_source_mode(default)
    return normalize_light_source_mode(settings.get("light_source_mode"), default=default)


class CameraSettingsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_camera_settings_path()

    @property
    def path(self) -> Path:
        return self._path

    def set_path(self, path: str | Path) -> None:
        self._path = Path(path)

    def load_for_serial(self, serial: str) -> dict[str, Any] | None:
        serial_text = str(serial).strip()
        if not serial_text:
            return None
        payload = self._load_all()
        serial_block = payload.get("by_serial")
        raw = None
        if isinstance(serial_block, dict):
            raw = serial_block.get(serial_text)
        if raw is None:
            raw = payload.get(serial_text)
        if not isinstance(raw, dict):
            return None
        normalized = _normalize_settings_payload(raw)
        return normalized or None

    def save_for_serial(self, serial: str, settings: Mapping[str, Any]) -> None:
        serial_text = str(serial).strip()
        if not serial_text:
            return
        payload = self._load_all()
        by_serial = payload.get("by_serial")
        if not isinstance(by_serial, dict):
            by_serial = {}
            payload["by_serial"] = by_serial
        by_serial[serial_text] = _normalize_settings_payload(settings)
        atomic_write_json(self._path, payload, ensure_ascii=False, indent=2)

    def load_for_role(self, role: str, *, serial: str = "") -> dict[str, Any] | None:
        role_text = str(role).strip()
        payload = self._load_all()
        if role_text:
            by_role = payload.get("by_role")
            if isinstance(by_role, dict):
                raw = by_role.get(role_text)
                if isinstance(raw, dict):
                    settings = raw.get("settings")
                    if isinstance(settings, dict):
                        normalized = _normalize_settings_payload(settings)
                        if normalized:
                            return normalized
        return self.load_for_serial(serial)

    def save_for_role(self, role: str, serial: str, settings: Mapping[str, Any]) -> None:
        role_text = str(role).strip()
        serial_text = str(serial).strip()
        normalized = _normalize_settings_payload(settings)
        payload = self._load_all()
        if role_text:
            by_role = payload.get("by_role")
            if not isinstance(by_role, dict):
                by_role = {}
                payload["by_role"] = by_role
            by_role[role_text] = {
                "serial": serial_text,
                "settings": normalized,
            }
        if serial_text:
            by_serial = payload.get("by_serial")
            if not isinstance(by_serial, dict):
                by_serial = {}
                payload["by_serial"] = by_serial
            by_serial[serial_text] = normalized
        atomic_write_json(self._path, payload, ensure_ascii=False, indent=2)

    def save_serial_for_role(self, role: str, serial: str) -> None:
        role_text = str(role).strip()
        if not role_text:
            return
        payload = self._load_all()
        by_role = payload.get("by_role")
        if not isinstance(by_role, dict):
            by_role = {}
            payload["by_role"] = by_role
        by_role[role_text] = {"serial": str(serial or "").strip()}
        atomic_write_json(self._path, payload, ensure_ascii=False, indent=2)

    def serial_for_role(self, role: str) -> str:
        role_text = str(role).strip()
        if not role_text:
            return ""
        payload = self._load_all()
        by_role = payload.get("by_role")
        if not isinstance(by_role, dict):
            return ""
        raw = by_role.get(role_text)
        if not isinstance(raw, dict):
            return ""
        return str(raw.get("serial", "")).strip()

    def _load_all(self) -> dict[str, Any]:
        raw = load_json_with_backup(self._path, default={})
        return raw if isinstance(raw, dict) else {}


def _normalize_settings_payload(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        return {}

    payload: dict[str, Any] = {}
    if settings.get("exposure_time_us") is not None:
        payload["exposure_time_us"] = float(settings["exposure_time_us"])
    if settings.get("gain") is not None:
        payload["gain"] = float(settings["gain"])
    if settings.get("digital_shift_enable") is not None:
        payload["digital_shift_enable"] = bool(settings["digital_shift_enable"])
    if settings.get("digital_shift") is not None:
        payload["digital_shift"] = float(settings["digital_shift"])

    trigger_mode = str(settings.get("trigger_mode") or "").strip()
    if trigger_mode:
        payload["trigger_mode"] = trigger_mode

    light_source_mode = normalize_light_source_mode(
        settings.get("light_source_mode"),
        default="",
    )
    if light_source_mode:
        payload["light_source_mode"] = light_source_mode

    if settings.get("acquisition_frame_rate_enable") is not None:
        payload["acquisition_frame_rate_enable"] = bool(
            settings["acquisition_frame_rate_enable"]
        )
    if settings.get("acquisition_frame_rate") is not None:
        payload["acquisition_frame_rate"] = float(settings["acquisition_frame_rate"])

    return payload
