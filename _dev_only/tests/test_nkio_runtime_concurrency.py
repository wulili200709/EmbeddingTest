from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from devices.di_poller import DiPoller
from devices.nkio_board import NkioBoard
from devices.nkio_errors import NkioBusyError


class _OverlapDetectingRaw:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._active_calls = 0
        self.overlap_detected = False
        self.do_word = 0

    def _enter(self) -> None:
        with self._state_lock:
            self._active_calls += 1
            if self._active_calls > 1:
                self.overlap_detected = True

    def _leave(self) -> None:
        with self._state_lock:
            self._active_calls -= 1

    def read_di_word(self, _index: int) -> tuple[int, int]:
        self._enter()
        try:
            time.sleep(0.001)
            return 0, 0
        finally:
            self._leave()

    def write_do_word(self, _index: int, value: int) -> int:
        self._enter()
        try:
            time.sleep(0.001)
            self.do_word = int(value)
            return 0
        finally:
            self._leave()


class _BusyOnceIo:
    is_open = True

    def __init__(self) -> None:
        self.call_count = 0

    def read_input(self, _name: str) -> bool:
        self.call_count += 1
        if self.call_count == 2:
            raise NkioBusyError(
                code=1,
                operation="NKDIO_PollingReadDiWord",
                detail="index=0",
            )
        return self.call_count >= 3


class _PersistentlyBusyIo:
    is_open = True

    def __init__(self) -> None:
        self.call_count = 0

    def read_input(self, _name: str) -> bool:
        self.call_count += 1
        if self.call_count == 1:
            return False
        raise NkioBusyError(
            code=1,
            operation="NKDIO_PollingReadDiWord",
            detail="index=0",
        )


class NkioRuntimeConcurrencyTest(unittest.TestCase):
    def test_board_serializes_di_reads_and_do_writes(self) -> None:
        raw = _OverlapDetectingRaw()
        with mock.patch("devices.nkio_board.NkioRawLib", return_value=raw):
            board = NkioBoard(Path("unused.ini"))
        board._opened = True

        start = threading.Barrier(3)

        def poll_di() -> None:
            start.wait()
            for _ in range(30):
                board.read_di_word()

        def switch_light() -> None:
            start.wait()
            for index in range(30):
                board.write_do_word(index & 1)

        threads = [
            threading.Thread(target=poll_di),
            threading.Thread(target=switch_light),
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=3.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(raw.overlap_detected)

    def test_transient_busy_is_retried_and_rising_edge_is_delivered(self) -> None:
        io = _BusyOnceIo()
        rising = threading.Event()
        poller = DiPoller(
            io,
            input_names=["foot_switch"],
            poll_interval_ms=2,
            debounce_ms=0,
            busy_retry_count=2,
            busy_retry_delay_ms=1,
        )
        poller.add_rising_callback(lambda _event: rising.set())

        try:
            poller.start()
            self.assertTrue(rising.wait(timeout=1.0))
            self.assertTrue(poller.is_running)
        finally:
            poller.stop()

    def test_repeated_busy_does_not_terminate_di_thread(self) -> None:
        io = _PersistentlyBusyIo()
        poller = DiPoller(
            io,
            input_names=["foot_switch"],
            poll_interval_ms=2,
            debounce_ms=0,
            busy_retry_count=1,
            busy_retry_delay_ms=1,
        )

        try:
            poller.start()
            time.sleep(0.05)
            self.assertGreater(io.call_count, 2)
            self.assertTrue(poller.is_running)
        finally:
            poller.stop()


if __name__ == "__main__":
    unittest.main()
