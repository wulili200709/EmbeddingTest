from __future__ import annotations

from typing import Any

from common.camera_roles import CAMERA_ROLES, normalize_camera_role
from infrastructure.camera_settings_store import (
    CAPTURE_MODE_SINGLE_MULTI_LIGHT,
    normalize_capture_light_output,
    normalize_capture_mode,
)


def physical_connected_roles(runtime) -> list[str]:
    service = getattr(runtime, "_frame_grab_service", None)
    if service is None:
        return []
    try:
        raw_roles = service.roles()
    except Exception:
        return []
    roles: list[str] = []
    for role in raw_roles:
        role_text = normalize_camera_role(role)
        if role_text and role_text not in roles:
            roles.append(role_text)
    return roles


def capture_config(runtime) -> dict[str, Any]:
    store = getattr(runtime, "_camera_settings_store", None)
    if store is None:
        return {"capture_mode": "independent", "capture_channels": []}
    try:
        return dict(store.load_capture_config())
    except Exception:
        return {"capture_mode": "independent", "capture_channels": []}


def is_single_multi_light_mode(runtime) -> bool:
    config = capture_config(runtime)
    return normalize_capture_mode(config.get("capture_mode")) == CAPTURE_MODE_SINGLE_MULTI_LIGHT


def enabled_single_multi_light_channels(runtime) -> list[dict[str, Any]]:
    config = capture_config(runtime)
    if normalize_capture_mode(config.get("capture_mode")) != CAPTURE_MODE_SINGLE_MULTI_LIGHT:
        return []

    channels: list[dict[str, Any]] = []
    for index, raw in enumerate(list(config.get("capture_channels", []) or []), start=1):
        if not isinstance(raw, dict):
            continue
        if not bool(raw.get("enabled", True)):
            continue
        role = normalize_camera_role(raw.get("role", ""))
        if not role:
            continue
        physical_role = normalize_camera_role(raw.get("physical_role", ""), default=role) or role
        channel = dict(raw)
        channel["role"] = role
        channel["physical_role"] = physical_role
        channel["light_output"] = normalize_capture_light_output(
            channel.get("light_output"),
            default=f"DO_LIGHT_CAM{index}",
        )
        channels.append(channel)
    return channels


def active_runtime_roles(runtime) -> list[str]:
    channels = enabled_single_multi_light_channels(runtime)
    if not channels:
        return physical_connected_roles(runtime)

    connected = set(physical_connected_roles(runtime))
    roles: list[str] = []
    for channel in channels:
        if channel.get("physical_role") not in connected:
            continue
        role = normalize_camera_role(channel.get("role", ""))
        if role and role not in roles:
            roles.append(role)
    return roles


def channels_for_roles(runtime, requested_roles=None) -> list[dict[str, Any]]:
    channels = enabled_single_multi_light_channels(runtime)
    if requested_roles is None:
        return channels
    requested = {
        normalize_camera_role(role)
        for role in requested_roles
        if normalize_camera_role(role)
    }
    return [channel for channel in channels if channel.get("role") in requested]


def light_output_index(channel: dict[str, Any]) -> int:
    text = normalize_capture_light_output(channel.get("light_output"))
    for index, role in enumerate(CAMERA_ROLES, start=1):
        if text == f"DO_LIGHT_CAM{index}" or text.endswith(role.upper()):
            return index
    return 1
