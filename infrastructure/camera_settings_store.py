from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app_paths import writable_embedding_test_root


_CAMERA_SETTINGS_FILENAME = "camera_settings.json"
_KNOWN_SETTING_KEYS = (
    "exposure_time_us",
    "gain",
    "trigger_mode",
    "acquisition_frame_rate_enable",
    "acquisition_frame_rate",
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
    for key in _KNOWN_SETTING_KEYS:
        if key == "trigger_mode":
            continue
        if key in normalized:
            payload[key] = normalized[key]
    return payload


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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}


def _normalize_settings_payload(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        return {}

    payload: dict[str, Any] = {}
    if settings.get("exposure_time_us") is not None:
        payload["exposure_time_us"] = float(settings["exposure_time_us"])
    if settings.get("gain") is not None:
        payload["gain"] = float(settings["gain"])

    trigger_mode = str(settings.get("trigger_mode") or "").strip()
    if trigger_mode:
        payload["trigger_mode"] = trigger_mode

    if settings.get("acquisition_frame_rate_enable") is not None:
        payload["acquisition_frame_rate_enable"] = bool(
            settings["acquisition_frame_rate_enable"]
        )
    if settings.get("acquisition_frame_rate") is not None:
        payload["acquisition_frame_rate"] = float(settings["acquisition_frame_rate"])

    return payload
