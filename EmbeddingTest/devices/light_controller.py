from __future__ import annotations

import threading
from typing import Iterable

from .io_controller import IoController


class LightController:
    """High-level controller for camera light outputs."""

    def __init__(
        self,
        io: IoController,
        *,
        supported_cameras: Iterable[int] = (1, 2),
    ) -> None:
        self.io = io
        self.supported_cameras = tuple(int(camera_index) for camera_index in supported_cameras)
        self._camera_light_modes = {
            camera_index: "board_io" for camera_index in self.supported_cameras
        }
        self._lock = threading.Lock()

    def set_camera_light_mode(self, camera_index: int, mode: str) -> None:
        camera_index = self._validate_camera_index(camera_index)
        normalized = self._normalize_camera_light_mode(mode)
        with self._lock:
            self._camera_light_modes[camera_index] = normalized
            if normalized != "board_io":
                self._set_camera_light_if_mapped(camera_index, False)

    def requires_stable_delay(self, camera_index: int) -> bool:
        camera_index = self._validate_camera_index(camera_index)
        return self._camera_light_modes.get(camera_index, "board_io") == "board_io"

    def turn_on(self, camera_index: int) -> None:
        camera_index = self._validate_camera_index(camera_index)
        if not self.requires_stable_delay(camera_index):
            return
        with self._lock:
            self._set_camera_light_if_mapped(camera_index, True)

    def turn_off(self, camera_index: int) -> None:
        camera_index = self._validate_camera_index(camera_index)
        if not self.requires_stable_delay(camera_index):
            return
        with self._lock:
            self._set_camera_light_if_mapped(camera_index, False)

    def turn_off_all(self) -> None:
        with self._lock:
            for camera_index in self.supported_cameras:
                self._set_camera_light_if_mapped(camera_index, False)

    def prepare_capture(self, camera_index: int) -> None:
        """Enable the target camera light and make sure the others are off."""
        camera_index = self._validate_camera_index(camera_index)
        with self._lock:
            for other_index in self.supported_cameras:
                mode = self._camera_light_modes.get(other_index, "board_io")
                should_enable = (
                    other_index == camera_index
                    and mode == "board_io"
                    and self._camera_light_modes.get(camera_index, "board_io") == "board_io"
                )
                self._set_camera_light_if_mapped(other_index, should_enable)

    def finish_capture(self, camera_index: int) -> None:
        camera_index = self._validate_camera_index(camera_index)
        if not self.requires_stable_delay(camera_index):
            return
        with self._lock:
            self._set_camera_light_if_mapped(camera_index, False)

    def switch_to(self, camera_index: int) -> None:
        self.prepare_capture(camera_index)

    def snapshot(self) -> dict[int, bool]:
        return {
            camera_index: self.io.read_output(f"light_cam{camera_index}")
            for camera_index in self.supported_cameras
            if self._camera_light_is_mapped(camera_index)
        }

    def _validate_camera_index(self, camera_index: int) -> int:
        camera_index = int(camera_index)
        if camera_index not in self.supported_cameras:
            raise ValueError(
                f"unsupported camera index: {camera_index}; "
                f"supported={self.supported_cameras}"
            )
        return camera_index

    @staticmethod
    def _normalize_camera_light_mode(mode: object) -> str:
        text = str(mode or "").strip().lower()
        if text in {"camera_line1_strobe", "camera_gpio_strobe", "camera_strobe", "line1"}:
            return "camera_line1_strobe"
        return "board_io"

    def _camera_light_is_mapped(self, camera_index: int) -> bool:
        mapping = getattr(self.io, "mapping", None)
        if mapping is None or not hasattr(mapping, "do_names"):
            return True
        try:
            return f"light_cam{int(camera_index)}" in set(mapping.do_names())
        except Exception:
            return True

    def _set_camera_light_if_mapped(self, camera_index: int, on: bool) -> None:
        if not self._camera_light_is_mapped(camera_index):
            return
        self.io.set_camera_light(camera_index, on)
