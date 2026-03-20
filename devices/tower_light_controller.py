from __future__ import annotations

import threading

from .io_controller import IoController


class TowerLightController:
    """State-oriented controller for the red/green/blue tower light."""

    def __init__(
        self,
        io: IoController,
        *,
        flash_ms: int = 200,
        idle_blue_delay_s: float = 30.0,
    ) -> None:
        self.io = io
        self.flash_s = max(0.01, float(flash_ms) / 1000.0)
        self.idle_blue_delay_s = max(0.0, float(idle_blue_delay_s))
        self._lock = threading.Lock()
        self._flash_timer: threading.Timer | None = None
        self._idle_timer: threading.Timer | None = None
        self._state = "off"

    @property
    def state(self) -> str:
        return self._state

    def close(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()

    def all_off(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self.io.tower_all_off()
            self._state = "off"

    def enter_waiting(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self.io.set_tower_light(red=False, green=False, blue=True)
            self._state = "waiting"

    def enter_inspecting(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self.io.tower_all_off()
            self._state = "inspecting"

    def show_ok(self) -> None:
        self._flash_result(color="green", state_name="ok")

    def show_ng(self) -> None:
        self._flash_result(color="red", state_name="ng")

    def schedule_idle_waiting(self) -> None:
        with self._lock:
            self._cancel_idle_timer_locked()
            if self.idle_blue_delay_s <= 0:
                self.io.set_tower_light(red=False, green=False, blue=True)
                self._state = "waiting"
                return
            timer = threading.Timer(self.idle_blue_delay_s, self.enter_waiting)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _flash_result(self, *, color: str, state_name: str) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self.io.tower_all_off()
            if color == "green":
                self.io.set_tower_light(green=True)
            elif color == "red":
                self.io.set_tower_light(red=True)
            else:
                raise ValueError(f"unsupported tower light color: {color}")
            self._state = state_name

            timer = threading.Timer(self.flash_s, self._finish_flash_and_schedule_idle)
            timer.daemon = True
            self._flash_timer = timer
            timer.start()

    def _finish_flash_and_schedule_idle(self) -> None:
        with self._lock:
            self.io.tower_all_off()
            self._state = "post_result"
            self._flash_timer = None

            if self.idle_blue_delay_s <= 0:
                self.io.set_tower_light(red=False, green=False, blue=True)
                self._state = "waiting"
                return

            timer = threading.Timer(self.idle_blue_delay_s, self.enter_waiting)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _cancel_flash_timer_locked(self) -> None:
        if self._flash_timer is not None:
            self._flash_timer.cancel()
            self._flash_timer = None

    def _cancel_idle_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
