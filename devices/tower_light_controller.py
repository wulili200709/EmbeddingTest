from __future__ import annotations

import threading

from .io_controller import IoController


class TowerLightController:
    """State-oriented controller for the red/green/blue tower light."""

    def __init__(
        self,
        io: IoController,
        *,
        ok_flash_ms: int = 200,
        ng_flash_ms: int = 200,
        ng_buzzer_ms: int = 500,
        idle_blue_delay_s: float = 30.0,
    ) -> None:
        self.io = io
        self.ok_flash_s = max(0.01, float(ok_flash_ms) / 1000.0)
        self.ng_flash_s = max(0.01, float(ng_flash_ms) / 1000.0)
        self.ng_buzzer_s = max(0.0, float(ng_buzzer_ms) / 1000.0)
        self.idle_blue_delay_s = max(0.0, float(idle_blue_delay_s))
        self._lock = threading.Lock()
        self._flash_timer: threading.Timer | None = None
        self._idle_timer: threading.Timer | None = None
        self._buzzer_timer: threading.Timer | None = None
        self._buzzer_token: object | None = None
        self._state = "off"

    @property
    def state(self) -> str:
        return self._state

    def close(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self._cancel_buzzer_timer_locked()
            self._set_buzzer_safely(False)

    def all_off(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self._cancel_buzzer_timer_locked()
            self.io.tower_all_off()
            self._set_buzzer_safely(False)
            self._state = "off"

    def enter_waiting(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self._cancel_buzzer_timer_locked()
            self.io.set_tower_light(red=False, green=False, blue=True)
            self._set_buzzer_safely(False)
            self._state = "waiting"

    def enter_inspecting(self) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            self._cancel_buzzer_timer_locked()
            self.io.tower_all_off()
            self._set_buzzer_safely(False)
            self._state = "inspecting"

    def show_ok(self) -> None:
        self._flash_result(color="green", state_name="ok", flash_s=self.ok_flash_s)

    def show_ng(self) -> None:
        self._flash_result(color="red", state_name="ng", flash_s=self.ng_flash_s)
        if self.ng_buzzer_s > 0.0:
            self._pulse_buzzer(self.ng_buzzer_s)

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

    def _flash_result(self, *, color: str, state_name: str, flash_s: float) -> None:
        with self._lock:
            self._cancel_flash_timer_locked()
            self._cancel_idle_timer_locked()
            if self._cancel_buzzer_timer_locked():
                self._set_buzzer_safely(False)
            self.io.tower_all_off()
            if color == "green":
                self.io.set_tower_light(green=True)
            elif color == "red":
                self.io.set_tower_light(red=True)
            else:
                raise ValueError(f"unsupported tower light color: {color}")
            self._state = state_name

            timer = threading.Timer(max(0.01, float(flash_s)), self._finish_flash_and_schedule_idle)
            timer.daemon = True
            self._flash_timer = timer
            timer.start()

    def _pulse_buzzer(self, duration_s: float) -> None:
        duration = max(0.0, float(duration_s))
        with self._lock:
            self._cancel_buzzer_timer_locked()
            if duration <= 0.0:
                self._set_buzzer_safely(False)
                return
            self._set_buzzer_safely(True)
            token = object()
            timer = threading.Timer(duration, self._finish_buzzer_pulse, args=(token,))
            timer.daemon = True
            self._buzzer_token = token
            self._buzzer_timer = timer
            timer.start()

    def _finish_buzzer_pulse(self, token: object) -> None:
        with self._lock:
            if token is not self._buzzer_token:
                return
            self._set_buzzer_safely(False)
            self._buzzer_timer = None
            self._buzzer_token = None

    def _set_buzzer_safely(self, on: bool) -> None:
        try:
            self.io.set_buzzer(bool(on))
        except Exception:
            pass

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

    def _cancel_buzzer_timer_locked(self) -> bool:
        had_timer = self._buzzer_timer is not None
        if self._buzzer_timer is not None:
            self._buzzer_timer.cancel()
            self._buzzer_timer = None
        self._buzzer_token = None
        return had_timer
