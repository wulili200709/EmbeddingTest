from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6 import QtCore


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.hardware import (
    _fire_delayed_trigger_from_di,
    _on_foot_switch_rising,
    _schedule_trigger_from_di,
)


class _FakeSignal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(str(message))


class _FakeEvent:
    def __init__(self, name: str = "foot_switch") -> None:
        self.name = name


class _FakeRuntime:
    def __init__(self, delay_ms: int = 0) -> None:
        self._foot_trigger_delay_ms = delay_ms
        self._pending_di_trigger_delay_ms = 0
        self._di_trigger_delay_pending = False
        self.logAppended = _FakeSignal()
        self.trigger_calls = 0

    def trigger(self) -> None:
        self.trigger_calls += 1

    def _fire_delayed_trigger_from_di(self) -> None:
        _fire_delayed_trigger_from_di(self)


class RuntimeFootTriggerDelayTest(unittest.TestCase):
    def test_zero_delay_queues_immediate_di_trigger(self) -> None:
        runtime = _FakeRuntime(delay_ms=0)

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_foot_switch_rising(runtime, _FakeEvent())

        invoke_method.assert_called_once_with(
            runtime,
            "_trigger_from_di",
            QtCore.Qt.QueuedConnection,
        )
        self.assertFalse(runtime._di_trigger_delay_pending)
        self.assertEqual(runtime._pending_di_trigger_delay_ms, 0)

    def test_positive_delay_queues_scheduler_once(self) -> None:
        runtime = _FakeRuntime(delay_ms=180)

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_foot_switch_rising(runtime, _FakeEvent())

        invoke_method.assert_called_once_with(
            runtime,
            "_schedule_trigger_from_di",
            QtCore.Qt.QueuedConnection,
        )
        self.assertTrue(runtime._di_trigger_delay_pending)
        self.assertEqual(runtime._pending_di_trigger_delay_ms, 180)

    def test_duplicate_edge_is_ignored_while_delay_pending(self) -> None:
        runtime = _FakeRuntime(delay_ms=180)
        runtime._di_trigger_delay_pending = True
        runtime._pending_di_trigger_delay_ms = 180

        with patch("application.runtime.hardware.QtCore.QMetaObject.invokeMethod") as invoke_method:
            _on_foot_switch_rising(runtime, _FakeEvent())

        invoke_method.assert_not_called()
        self.assertIn(
            "[foot-switch] delayed trigger already pending; ignoring new edge",
            runtime.logAppended.messages,
        )

    def test_schedule_trigger_uses_qtimer_and_fires_after_delay(self) -> None:
        runtime = _FakeRuntime(delay_ms=180)
        runtime._di_trigger_delay_pending = True
        runtime._pending_di_trigger_delay_ms = 180
        captured: dict[str, object] = {}

        def _capture_single_shot(delay_ms: int, callback) -> None:
            captured["delay_ms"] = delay_ms
            captured["callback"] = callback

        with patch("application.runtime.hardware.QtCore.QTimer.singleShot", side_effect=_capture_single_shot) as single_shot:
            _schedule_trigger_from_di(runtime)

        single_shot.assert_called_once()
        self.assertEqual(captured["delay_ms"], 180)
        self.assertEqual(runtime.trigger_calls, 0)
        self.assertTrue(runtime._di_trigger_delay_pending)
        self.assertIn("[foot-switch] waiting 180 ms before trigger", runtime.logAppended.messages)

        callback = captured.get("callback")
        self.assertTrue(callable(callback))
        assert callable(callback)
        callback()

        self.assertEqual(runtime.trigger_calls, 1)
        self.assertFalse(runtime._di_trigger_delay_pending)
        self.assertEqual(runtime._pending_di_trigger_delay_ms, 0)
        self.assertIn("[foot-switch] delayed trigger fired", runtime.logAppended.messages)

    def test_schedule_trigger_without_delay_fires_immediately(self) -> None:
        runtime = _FakeRuntime(delay_ms=0)
        runtime._di_trigger_delay_pending = True
        runtime._pending_di_trigger_delay_ms = 0

        _schedule_trigger_from_di(runtime)

        self.assertEqual(runtime.trigger_calls, 1)
        self.assertFalse(runtime._di_trigger_delay_pending)
        self.assertEqual(runtime._pending_di_trigger_delay_ms, 0)


if __name__ == "__main__":
    unittest.main()
