from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from common.camera_roles import CAMERA_ROLES, configured_camera_roles, normalize_camera_role
from infrastructure.camera_settings_store import (
    normalize_capture_light_output,
)
from .capture_plan import CapturePlan, build_capture_plan


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


def physical_connected_bindings(runtime) -> dict[str, str]:
    """Return connected physical roles and their device serial numbers."""
    service = getattr(runtime, "_frame_grab_service", None)
    if service is None:
        return {}
    bindings: dict[str, str] = {}
    for role in physical_connected_roles(runtime):
        try:
            device = service.get_device(role)
        except Exception:
            bindings[role] = ""
            continue
        bindings[role] = str(getattr(device, "serial_number", "") or "").strip()
    return bindings


def capture_config(runtime) -> dict[str, Any]:
    store = getattr(runtime, "_camera_settings_store", None)
    if store is None:
        return {"capture_mode": "independent", "capture_channels": []}
    lock = getattr(runtime, "_capture_config_lock", None)
    context = lock if lock is not None else nullcontext()
    try:
        path = Path(getattr(store, "path"))
        stat = path.stat()
        signature = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        signature = None

    with context:
        cached_signature = getattr(runtime, "_capture_config_cache_signature", None)
        cached = getattr(runtime, "_capture_config_cache", None)
        if signature is not None and cached_signature == signature and isinstance(cached, dict):
            return _copy_capture_config(cached)
        try:
            loaded = dict(store.load_capture_config())
        except Exception:
            return {"capture_mode": "independent", "capture_channels": []}
        if signature is not None:
            runtime._capture_config_cache_signature = signature
            runtime._capture_config_cache = _copy_capture_config(loaded)
        return _copy_capture_config(loaded)


def _copy_capture_config(config: dict[str, Any]) -> dict[str, Any]:
    copied = dict(config or {})
    copied["capture_channels"] = [
        dict(channel)
        for channel in list(copied.get("capture_channels", []) or [])
        if isinstance(channel, dict)
    ]
    return copied


def runtime_capture_plan(
    runtime,
    *,
    configured_roles=None,
) -> CapturePlan:
    return build_capture_plan(
        capture_config(runtime),
        configured_roles=configured_roles,
    )


def is_single_multi_light_mode(runtime) -> bool:
    return runtime_capture_plan(runtime).uses_channel_mapping


def enabled_single_multi_light_channels(runtime) -> list[dict[str, Any]]:
    plan = runtime_capture_plan(runtime)
    if not plan.uses_channel_mapping:
        return []
    return [channel.to_mapping() for channel in plan.channels_for_roles()]


def active_runtime_roles(runtime) -> list[str]:
    plan = runtime_capture_plan(runtime)
    if not plan.uses_channel_mapping:
        return physical_connected_roles(runtime)
    return list(plan.active_logical_roles(physical_connected_roles(runtime)))


def required_runtime_roles(runtime) -> list[str]:
    """Return the logical camera roles required by the current product."""
    plan = runtime_capture_plan(runtime)
    if plan.uses_channel_mapping:
        return list(plan.logical_roles)

    session = getattr(runtime, "_session", None)
    loader = getattr(session, "load_session", None)
    if callable(loader):
        try:
            session_roles = configured_camera_roles(
                getattr(loader(), "runtime_camera_roles", []) or []
            )
        except Exception:
            session_roles = []
        if session_roles:
            return session_roles

    item_roles = configured_camera_roles(
        getattr(item, "camera_id", "")
        for item in list(getattr(getattr(runtime, "_runtime_context", None), "inspection_items", []) or [])
        if bool(getattr(item, "enabled", True))
    )
    if item_roles:
        return item_roles

    return active_runtime_roles(runtime)


def channels_for_roles(runtime, requested_roles=None) -> list[dict[str, Any]]:
    plan = runtime_capture_plan(runtime)
    if not plan.uses_channel_mapping:
        return []
    return [channel.to_mapping() for channel in plan.channels_for_roles(requested_roles)]


def light_output_index(channel: dict[str, Any]) -> int:
    text = normalize_capture_light_output(channel.get("light_output"))
    for index, role in enumerate(CAMERA_ROLES, start=1):
        if text == f"DO_LIGHT_CAM{index}" or text.endswith(role.upper()):
            return index
    return 1
