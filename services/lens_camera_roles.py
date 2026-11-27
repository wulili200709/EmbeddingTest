"""Read-only mapping from inspection channels to physical camera calibration."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from common.camera_roles import normalize_camera_role
from infrastructure.camera_settings_store import (
    default_camera_settings_path,
    normalize_capture_channels,
    normalize_capture_mode,
    uses_channel_capture_mapping,
)


def _read_capture_config(product_dir=""):
    path = (Path(product_dir) / "camera_settings.json") if str(product_dir or "").strip() else Path(default_camera_settings_path())
    found = False
    for candidate in (path, Path(str(path) + ".bak")):
        if not candidate.exists():
            continue
        found = True
        try:
            with candidate.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
            if isinstance(payload, Mapping):
                return payload
        except (OSError, ValueError):
            continue
    if found:
        raise ValueError("Cannot read camera capture routing configuration")
    return {}


def resolve_correction_role(product_dir, logical_role, *, capture_config=None) -> str:
    """Use the same channel defaults as acquisition, without migrating settings.

    ProductSession and RuntimeController use the product's camera_settings.json.
    No fallback to another product or to machine settings is allowed here.
    """
    role = normalize_camera_role(logical_role, default="")
    if not role:
        raise ValueError(f"Invalid camera role: {logical_role!r}")
    if capture_config is None:
        capture_config = _read_capture_config(product_dir)
    if not isinstance(capture_config, Mapping):
        raise ValueError("Camera capture configuration must be an object")
    mode = normalize_capture_mode(capture_config.get("capture_mode"))
    if not uses_channel_capture_mapping(mode):
        return role
    channels = normalize_capture_channels(capture_config.get("capture_channels"), mode=mode)
    channel = next((value for value in channels if value["role"] == role), None)
    return normalize_camera_role(channel["physical_role"], default=role) if channel is not None else role
