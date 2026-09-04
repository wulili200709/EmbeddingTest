from __future__ import annotations

import unittest

from devices.di_poller import DiPoller
from devices.io_controller import IoController
from devices.io_mapping import IoMapping


class _FakeBoard:
    def __init__(self, word: int) -> None:
        self.word = int(word)
        self.read_count = 0
        self.is_open = True

    def read_di_word(self) -> int:
        self.read_count += 1
        return self.word


class DiSnapshotTests(unittest.TestCase):
    def test_all_business_inputs_are_decoded_from_one_board_read(self) -> None:
        board = _FakeBoard(0b0101)
        mapping = IoMapping.from_dict(
            {
                "di": {
                    "active_high_0": {"channel": 0, "active_high": True},
                    "active_low_1": {"channel": 1, "active_high": False},
                    "active_high_2": {"channel": 2, "active_high": True},
                },
                "do": {},
            }
        )
        controller = IoController(board=board, mapping=mapping)  # type: ignore[arg-type]

        snapshot = controller.snapshot_inputs()

        self.assertEqual(
            snapshot,
            {"active_high_0": True, "active_low_1": True, "active_high_2": True},
        )
        self.assertEqual(board.read_count, 1)

        detailed, raw_word = controller.snapshot_inputs_with_raw_word()
        self.assertEqual(detailed, snapshot)
        self.assertEqual(raw_word, 0b0101)
        self.assertEqual(board.read_count, 2)

    def test_one_poller_scan_uses_one_snapshot_and_emits_debounced_changes(self) -> None:
        class FakeIo:
            is_open = True

            def __init__(self) -> None:
                self.read_count = 0
                self.states = {"a": False, "b": False}

            def snapshot_inputs(self, names=None):
                self.read_count += 1
                selected = list(names or self.states)
                return {name: self.states[name] for name in selected}

        io = FakeIo()
        poller = DiPoller(io, input_names=["a", "b"], debounce_ms=0)  # type: ignore[arg-type]
        events = []
        poller.add_change_callback(events.append)
        poller._initialize_states()
        io.states = {"a": True, "b": True}

        poller._scan_once(1.0)
        poller._scan_once(1.0)

        self.assertEqual(io.read_count, 3)
        self.assertEqual([(event.name, event.state) for event in events], [("a", True), ("b", True)])
        self.assertEqual([event.monotonic_timestamp for event in events], [1.0, 1.0])
        self.assertEqual([event.raw_word for event in events], [None, None])

    def test_poller_event_keeps_source_di_word(self) -> None:
        board = _FakeBoard(0)
        mapping = IoMapping.from_dict(
            {
                "di": {"good_outlet_sensor": {"channel": 6, "active_high": True}},
                "do": {},
            }
        )
        controller = IoController(board=board, mapping=mapping)  # type: ignore[arg-type]
        poller = DiPoller(
            controller,
            input_names=["good_outlet_sensor"],
            debounce_ms=0,
        )
        events = []
        poller.add_change_callback(events.append)
        poller._initialize_states()

        board.word = 1 << 6
        poller._scan_once(12.5)
        poller._scan_once(12.5)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "good_outlet_sensor")
        self.assertTrue(events[0].state)
        self.assertEqual(events[0].raw_word, 0x0040)
        self.assertEqual(events[0].monotonic_timestamp, 12.5)


if __name__ == "__main__":
    unittest.main()
