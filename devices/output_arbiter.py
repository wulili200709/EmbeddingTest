from __future__ import annotations

import threading
from typing import Mapping

from .io_controller import IoController


class OutputArbiter:
    """Single runtime owner for conveyor outputs and shared buzzer requests.

    Conveyor faults and short result pulses share DO8. Their requests are
    combined so a result timer can never silence a latched conveyor fault.
    Motion output batches are forwarded as one board-word update.
    """

    def __init__(self, io: IoController) -> None:
        self.io = io
        self._lock = threading.RLock()
        self._line_buzzer = False
        self._result_buzzer = False
        self._effective_buzzer: bool | None = None

    def set_line_output(self, name: str, on: bool) -> None:
        output_name = str(name)
        value = bool(on)
        with self._lock:
            if output_name == "buzzer":
                self._line_buzzer = value
                self._write_effective_buzzer_locked()
                return
            self.io.set_output(output_name, value)

    def set_line_outputs(self, updates: Mapping[str, bool]) -> None:
        normalized = {str(name): bool(on) for name, on in updates.items()}
        with self._lock:
            buzzer = normalized.pop("buzzer", None)
            if normalized:
                self.io.set_outputs(normalized)
            if buzzer is not None:
                self._line_buzzer = buzzer
                self._write_effective_buzzer_locked()

    def set_result_buzzer(self, on: bool) -> None:
        with self._lock:
            self._result_buzzer = bool(on)
            self._write_effective_buzzer_locked()

    def reset(self) -> None:
        with self._lock:
            self._line_buzzer = False
            self._result_buzzer = False
            self._write_effective_buzzer_locked(force=True)

    def _write_effective_buzzer_locked(self, *, force: bool = False) -> None:
        effective = self._line_buzzer or self._result_buzzer
        if not force and self._effective_buzzer is effective:
            return
        self.io.set_buzzer(effective)
        self._effective_buzzer = effective


__all__ = ["OutputArbiter"]
