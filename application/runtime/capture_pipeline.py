from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping

from common.camera_roles import camera_index_for_role
from .capture_channels import light_output_index


@dataclass(frozen=True)
class RuntimeCapturedFrame:
    role: str
    physical_role: str
    light_index: int
    frame: object
    capture_ms: float

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "physical_role": self.physical_role,
            "light_index": self.light_index,
            "frame": self.frame,
            "capture_ms": self.capture_ms,
        }


def _non_negative_int(value: object, default: int) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return max(0, int(default))


def capture_runtime_channel(
    runtime,
    channel: Mapping[str, object],
    *,
    apply_camera_settings: Callable[[dict[str, object]], None] | None = None,
    before_capture: Callable[[str, int], None] | None = None,
    force_board_io_light: bool = False,
) -> RuntimeCapturedFrame:
    role = str(channel.get("role", "") or "").strip()
    physical_role = str(channel.get("physical_role", role) or role).strip()
    light_index = _non_negative_int(channel.get("light_index", 0), 0)
    if light_index <= 0:
        light_index = light_output_index(dict(channel)) or camera_index_for_role(role)

    normalized_channel = dict(channel)
    started_at = time.perf_counter()
    light_prepared = False
    if apply_camera_settings is not None:
        apply_camera_settings(normalized_channel)
    if force_board_io_light and hasattr(runtime._light_controller, "set_camera_light_mode"):
        runtime._light_controller.set_camera_light_mode(light_index, "board_io")
    try:
        runtime._light_controller.prepare_capture(light_index)
        light_prepared = True
        requires_stable_delay = True
        delay_getter = getattr(runtime._light_controller, "requires_stable_delay", None)
        if callable(delay_getter):
            requires_stable_delay = bool(delay_getter(light_index))
        stable_delay_ms = _non_negative_int(channel.get("stable_delay_ms", 50), 50)
        if stable_delay_ms > 0 and requires_stable_delay:
            time.sleep(stable_delay_ms / 1000.0)
        if before_capture is not None:
            before_capture(role, light_index)
        frame = runtime._frame_grab_service.capture_once(physical_role, timeout_ms=1000)
    finally:
        if light_prepared:
            runtime._light_controller.finish_capture(light_index)

    return RuntimeCapturedFrame(
        role=role,
        physical_role=physical_role,
        light_index=light_index,
        frame=frame,
        capture_ms=(time.perf_counter() - started_at) * 1000.0,
    )


__all__ = ["RuntimeCapturedFrame", "capture_runtime_channel"]
