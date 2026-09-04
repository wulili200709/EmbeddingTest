from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from .io_controller import IoController
from .nkio_errors import NkioBusyError, NkioError


@dataclass(frozen=True)
class DiEvent:
    name: str
    state: bool
    previous_state: bool
    edge: str  # "rising" | "falling" | "change"
    timestamp: float
    monotonic_timestamp: float = 0.0
    raw_word: int | None = None


DiCallback = Callable[[DiEvent], None]
DiErrorCallback = Callable[[str, Exception], None]


class DiPoller:
    """Poll DI inputs, apply debounce, and emit stable edge events."""

    def __init__(
        self,
        io: IoController,
        *,
        input_names: Iterable[str] | None = None,
        poll_interval_ms: int = 20,
        debounce_ms: int = 50,
        busy_retry_count: int = 3,
        busy_retry_delay_ms: int = 5,
    ) -> None:
        self.io = io
        self.input_names = list(input_names) if input_names is not None else list(io.mapping.di_names())
        self.poll_interval_s = max(0.001, float(poll_interval_ms) / 1000.0)
        self.debounce_s = max(0.0, float(debounce_ms) / 1000.0)
        self.busy_retry_count = max(0, int(busy_retry_count))
        self.busy_retry_delay_s = max(0.0, float(busy_retry_delay_ms) / 1000.0)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._on_change: list[DiCallback] = []
        self._on_rising: list[DiCallback] = []
        self._on_falling: list[DiCallback] = []
        self._on_error: list[DiErrorCallback] = []

        self._stable_state: dict[str, bool] = {}
        self._candidate_state: dict[str, bool] = {}
        self._candidate_since: dict[str, float] = {}

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        if not self.io.is_open:
            raise RuntimeError("IoController must be open before starting DiPoller")
        self._stop_event.clear()
        self._initialize_states()
        self._thread = threading.Thread(target=self._run_loop, name="DiPoller", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 1.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def set_callback(self, callback: DiCallback) -> None:
        self._on_change = [callback]

    def add_change_callback(self, callback: DiCallback) -> None:
        self._on_change.append(callback)

    def add_rising_callback(self, callback: DiCallback) -> None:
        self._on_rising.append(callback)

    def add_falling_callback(self, callback: DiCallback) -> None:
        self._on_falling.append(callback)

    def add_error_callback(self, callback: DiErrorCallback) -> None:
        self._on_error.append(callback)

    def snapshot(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._stable_state)

    def _initialize_states(self) -> None:
        now = time.perf_counter()
        snapshot, _raw_word = self._read_inputs_with_busy_retry()
        with self._lock:
            self._stable_state.clear()
            self._candidate_state.clear()
            self._candidate_since.clear()
            for name in self.input_names:
                state = bool(snapshot[name])
                self._stable_state[name] = state
                self._candidate_state[name] = state
                self._candidate_since[name] = now

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.perf_counter()
            try:
                self._scan_once(now)
            except NkioError as exc:
                # A transient board error must not terminate the DI thread.
                # The next polling cycle retries naturally; stable/candidate
                # state is intentionally left untouched for debounce safety.
                for callback in list(self._on_error):
                    callback("DI_WORD", exc)
            self._stop_event.wait(self.poll_interval_s)

    def _scan_once(self, now: float) -> None:
        snapshot, raw_word = self._read_inputs_with_busy_retry()
        for name in self.input_names:
            self._process_input(name, now, bool(snapshot[name]), raw_word=raw_word)

    def _process_input(
        self,
        name: str,
        now: float,
        raw_state: bool,
        *,
        raw_word: int | None = None,
    ) -> None:
        event: DiEvent | None = None

        with self._lock:
            stable = self._stable_state[name]
            candidate = self._candidate_state[name]

            if raw_state != candidate:
                self._candidate_state[name] = raw_state
                self._candidate_since[name] = now
                return

            if raw_state == stable:
                return

            stable_for = now - self._candidate_since[name]
            if stable_for < self.debounce_s:
                return

            self._stable_state[name] = raw_state
            edge = "rising" if raw_state and not stable else "falling"
            event = DiEvent(
                name=name,
                state=raw_state,
                previous_state=stable,
                edge=edge,
                timestamp=time.time(),
                monotonic_timestamp=float(now),
                raw_word=None if raw_word is None else int(raw_word),
            )

        if event is not None:
            self._emit_event(event)

    def _read_inputs_with_busy_retry(self) -> tuple[dict[str, bool], int | None]:
        for retry_index in range(self.busy_retry_count + 1):
            try:
                detailed_reader = getattr(self.io, "snapshot_inputs_with_raw_word", None)
                if callable(detailed_reader):
                    snapshot, raw_word = detailed_reader(self.input_names)
                    return dict(snapshot), int(raw_word)
                return self.io.snapshot_inputs(self.input_names), None
            except NkioBusyError:
                if retry_index >= self.busy_retry_count:
                    raise
                if self._stop_event.wait(self.busy_retry_delay_s):
                    raise
        raise RuntimeError("unreachable DI busy-retry state")

    def _emit_event(self, event: DiEvent) -> None:
        for callback in list(self._on_change):
            callback(event)
        if event.edge == "rising":
            for callback in list(self._on_rising):
                callback(event)
        elif event.edge == "falling":
            for callback in list(self._on_falling):
                callback(event)
