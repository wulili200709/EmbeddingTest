from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from common.camera_roles import CAMERA_ROLES, configured_camera_roles, normalize_camera_role
from infrastructure.camera_settings_store import (
    CAPTURE_MODE_INDEPENDENT,
    normalize_capture_channels,
    normalize_capture_light_output,
    normalize_capture_mode,
    uses_channel_capture_mapping,
)


@dataclass(frozen=True)
class CaptureChannel:
    """One logical inspection channel and its physical acquisition route."""

    role: str
    physical_role: str
    light_output: str
    exposure_time_us: float
    gain: float
    stable_delay_ms: int
    enabled: bool = True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "role": self.role,
            "physical_role": self.physical_role,
            "light_output": self.light_output,
            "exposure_time_us": float(self.exposure_time_us),
            "gain": float(self.gain),
            "stable_delay_ms": int(self.stable_delay_ms),
        }


@dataclass(frozen=True)
class CapturePlan:
    """Normalized, immutable source of truth for runtime acquisition routing."""

    mode: str
    channels: tuple[CaptureChannel, ...]

    @property
    def uses_channel_mapping(self) -> bool:
        return uses_channel_capture_mapping(self.mode)

    @property
    def logical_roles(self) -> tuple[str, ...]:
        return tuple(channel.role for channel in self.channels if channel.enabled)

    @property
    def physical_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        for channel in self.channels:
            if not channel.enabled or channel.physical_role in roles:
                continue
            roles.append(channel.physical_role)
        return tuple(roles)

    def channels_for_roles(self, roles: Iterable[object] | None = None) -> tuple[CaptureChannel, ...]:
        if roles is None:
            return tuple(channel for channel in self.channels if channel.enabled)
        requested = set(configured_camera_roles(roles))
        return tuple(
            channel
            for channel in self.channels
            if channel.enabled and channel.role in requested
        )

    def active_logical_roles(self, connected_physical_roles: Iterable[object]) -> tuple[str, ...]:
        connected = set(configured_camera_roles(connected_physical_roles))
        return tuple(
            channel.role
            for channel in self.channels
            if channel.enabled and channel.physical_role in connected
        )

    def physical_bindings(self, serials_by_role: Mapping[str, object]) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for role in self.physical_roles:
            serial = str(serials_by_role.get(role, "") or "").strip()
            if serial:
                bindings[role] = serial
        return bindings

    def validate_bindings(self, serials_by_role: Mapping[str, object]) -> tuple[str, ...]:
        issues: list[str] = []
        bindings = self.physical_bindings(serials_by_role)
        missing = [role for role in self.physical_roles if role not in bindings]
        if missing:
            issues.append("缺少物理相机序列号: " + ", ".join(missing))
        serial_to_roles: dict[str, list[str]] = {}
        for role, serial in bindings.items():
            serial_to_roles.setdefault(serial, []).append(role)
        duplicates = [
            f"{serial}=>{','.join(roles)}"
            for serial, roles in serial_to_roles.items()
            if len(roles) > 1
        ]
        if duplicates:
            issues.append("同一序列号被绑定到多个物理相机角色: " + "; ".join(duplicates))
        return tuple(issues)

    def channel_for_role(self, role: object) -> CaptureChannel | None:
        role_text = normalize_camera_role(role)
        return next(
            (channel for channel in self.channels if channel.enabled and channel.role == role_text),
            None,
        )


def build_capture_plan(
    config: Mapping[str, Any] | None,
    *,
    configured_roles: Iterable[object] | None = None,
) -> CapturePlan:
    raw = dict(config or {})
    mode = normalize_capture_mode(raw.get("capture_mode"))
    normalized = normalize_capture_channels(raw.get("capture_channels"), mode=mode)
    requested_roles = configured_camera_roles(configured_roles or [])
    requested_set = set(requested_roles)

    channels: list[CaptureChannel] = []
    for index, item in enumerate(normalized, start=1):
        role = normalize_camera_role(item.get("role"))
        if not role:
            continue
        enabled = bool(item.get("enabled", True))
        if mode == CAPTURE_MODE_INDEPENDENT:
            enabled = enabled and (not requested_set or role in requested_set)
            physical_role = role
        else:
            physical_role = normalize_camera_role(item.get("physical_role"), default=role) or role
        channels.append(
            CaptureChannel(
                role=role,
                physical_role=physical_role,
                light_output=normalize_capture_light_output(
                    item.get("light_output"),
                    default=f"DO_LIGHT_CAM{index}",
                ),
                exposure_time_us=float(item.get("exposure_time_us", 5000.0) or 5000.0),
                gain=float(item.get("gain", 0.0) or 0.0),
                stable_delay_ms=max(0, int(float(item.get("stable_delay_ms", 50) or 0))),
                enabled=enabled,
            )
        )

    # Keep the established role order even if a malformed file supplied rows
    # out of order.
    role_order = {role: index for index, role in enumerate(CAMERA_ROLES)}
    channels.sort(key=lambda channel: role_order.get(channel.role, len(role_order)))
    return CapturePlan(mode=mode, channels=tuple(channels))


__all__ = ["CaptureChannel", "CapturePlan", "build_capture_plan"]
