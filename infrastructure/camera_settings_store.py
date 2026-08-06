from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from common.app_paths import writable_embedding_test_root
from common.camera_roles import CAMERA_ROLES, DEFAULT_CAMERA_ROLE, normalize_camera_role
from common.safe_io import atomic_write_json, load_json_with_backup


_CAMERA_SETTINGS_FILENAME = "camera_settings.json"
CAMERA_SETTINGS_SCHEMA_VERSION = 2
LIGHT_SOURCE_MODE_BOARD_IO = "board_io"
LIGHT_SOURCE_MODE_CAMERA_LINE1_STROBE = "camera_line1_strobe"
CAPTURE_MODE_INDEPENDENT = "independent"
CAPTURE_MODE_SINGLE_MULTI_LIGHT = "single_multi_light"
CAPTURE_MODE_FLEXIBLE = "flexible"
CAPTURE_LIGHT_OUTPUTS = ("DO_LIGHT_CAM1", "DO_LIGHT_CAM2", "DO_LIGHT_CAM3")
CAPTURE_DEFAULT_EXPOSURE_US = 5000.0

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


def normalize_capture_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"single_multi_light", "single_camera_multi_light", "multi_light", "single"}:
        return CAPTURE_MODE_SINGLE_MULTI_LIGHT
    if text in {"flexible", "flexible_mapping", "channel_mapping", "mixed"}:
        return CAPTURE_MODE_FLEXIBLE
    return CAPTURE_MODE_INDEPENDENT


def uses_channel_capture_mapping(value: object) -> bool:
    return normalize_capture_mode(value) in {
        CAPTURE_MODE_SINGLE_MULTI_LIGHT,
        CAPTURE_MODE_FLEXIBLE,
    }


def normalize_capture_light_output(value: object, *, default: str = "DO_LIGHT_CAM1") -> str:
    text = str(value or "").strip().upper()
    if text in CAPTURE_LIGHT_OUTPUTS:
        return text
    default_text = str(default or "").strip().upper()
    return default_text if default_text in CAPTURE_LIGHT_OUTPUTS else CAPTURE_LIGHT_OUTPUTS[0]


def default_capture_channels(mode: object = CAPTURE_MODE_INDEPENDENT) -> list[dict[str, Any]]:
    normalized_mode = normalize_capture_mode(mode)
    channels: list[dict[str, Any]] = []
    for index, role in enumerate(CAMERA_ROLES, start=1):
        channels.append(
            {
                "enabled": True,
                "role": role,
                "physical_role": role if normalized_mode != CAPTURE_MODE_SINGLE_MULTI_LIGHT else DEFAULT_CAMERA_ROLE,
                "light_output": f"DO_LIGHT_CAM{index}",
                "exposure_time_us": CAPTURE_DEFAULT_EXPOSURE_US,
                "gain": 0.0,
                "stable_delay_ms": 50,
            }
        )
    return channels


def _normalize_capture_channel(
    channel: Mapping[str, Any] | None,
    *,
    role: str,
    mode: str,
    index: int,
) -> dict[str, Any]:
    raw = channel if isinstance(channel, Mapping) else {}
    role_text = normalize_camera_role(raw.get("role"), default=role) or role
    default_light = f"DO_LIGHT_CAM{index}"
    default_physical = role_text if mode != CAPTURE_MODE_SINGLE_MULTI_LIGHT else DEFAULT_CAMERA_ROLE

    def _float_value(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _int_value(key: str, default: int) -> int:
        try:
            return max(0, int(float(raw.get(key, default))))
        except (TypeError, ValueError):
            return int(default)

    return {
        "enabled": bool(raw.get("enabled", True)),
        "role": role_text,
        "physical_role": normalize_camera_role(raw.get("physical_role"), default=default_physical)
        or default_physical,
        "light_output": normalize_capture_light_output(raw.get("light_output"), default=default_light),
        "exposure_time_us": _float_value("exposure_time_us", CAPTURE_DEFAULT_EXPOSURE_US),
        "gain": _float_value("gain", 0.0),
        "stable_delay_ms": _int_value("stable_delay_ms", 50),
    }


def normalize_capture_channels(
    channels: object,
    *,
    mode: object = CAPTURE_MODE_INDEPENDENT,
) -> list[dict[str, Any]]:
    normalized_mode = normalize_capture_mode(mode)
    raw_by_role: dict[str, Mapping[str, Any]] = {}
    has_explicit_list = isinstance(channels, list)
    if isinstance(channels, list):
        for raw in channels:
            if not isinstance(raw, Mapping):
                continue
            role = normalize_camera_role(raw.get("role"), default="")
            if role:
                raw_by_role[role] = raw
    normalized_channels: list[dict[str, Any]] = []
    for index, role in enumerate(CAMERA_ROLES, start=1):
        raw = raw_by_role.get(role)
        channel = _normalize_capture_channel(
            raw,
            role=role,
            mode=normalized_mode,
            index=index,
        )
        if has_explicit_list and raw is None:
            channel["enabled"] = False
        normalized_channels.append(channel)
    return normalized_channels


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

    def load_capture_config(self) -> dict[str, Any]:
        payload = self._load_all()
        mode = normalize_capture_mode(payload.get("capture_mode"))
        channels = normalize_capture_channels(payload.get("capture_channels"), mode=mode)
        try:
            schema_version = int(payload.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            schema_version = 0
        if payload and schema_version < CAMERA_SETTINGS_SCHEMA_VERSION:
            # Migrate legacy/copied products in place while preserving camera
            # settings and serial bindings outside the capture section.
            payload["schema_version"] = CAMERA_SETTINGS_SCHEMA_VERSION
            payload["capture_mode"] = mode
            payload["capture_channels"] = channels
            atomic_write_json(self._path, payload, ensure_ascii=False, indent=2)
            schema_version = CAMERA_SETTINGS_SCHEMA_VERSION
        return {
            "schema_version": schema_version or CAMERA_SETTINGS_SCHEMA_VERSION,
            "capture_mode": mode,
            "capture_channels": channels,
        }

    def save_capture_config(self, mode: object, channels: object) -> None:
        payload = self._load_all()
        normalized_mode = normalize_capture_mode(mode)
        payload["schema_version"] = CAMERA_SETTINGS_SCHEMA_VERSION
        payload["capture_mode"] = normalized_mode
        payload["capture_channels"] = normalize_capture_channels(channels, mode=normalized_mode)
        atomic_write_json(self._path, payload, ensure_ascii=False, indent=2)

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
