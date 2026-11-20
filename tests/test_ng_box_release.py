import unittest
from unittest.mock import patch

from devices.di_poller import DiPoller
from services.inspection_scheduler import InspectionScheduler
from services.permission_manager import PermissionManager
from services.run_state import RunState, RunStateMachine


class NgBoxReleaseTests(unittest.TestCase):
    def scheduler(self, enabled=True):
        return InspectionScheduler(RunStateMachine(), PermissionManager("test"), enabled)

    def complete_ng(self, scheduler, timestamp):
        scheduler.on_inspecting_started()
        scheduler.on_aggregating_started()
        with patch("services.inspection_scheduler.time.time", return_value=timestamp):
            scheduler.on_completed(final_ok=False)

    def test_only_edge_after_current_ng_releases_and_next_ng_locks_again(self):
        scheduler = self.scheduler()
        self.assertFalse(scheduler.release_from_ng_box(9))
        self.complete_ng(scheduler, 10)
        self.assertFalse(scheduler.can_accept_trigger().allowed)
        self.assertFalse(scheduler.release_from_ng_box(9))
        self.assertFalse(scheduler.release_from_ng_box(10))
        self.assertTrue(scheduler.release_from_ng_box(11))
        self.assertTrue(scheduler.permission_manager.has_pending_release)
        self.assertTrue(scheduler.can_accept_trigger().allowed)
        self.assertFalse(scheduler.release_from_ng_box(12))
        self.complete_ng(scheduler, 20)
        self.assertFalse(scheduler.release_from_ng_box(11))
        self.assertFalse(scheduler.can_accept_trigger().allowed)
        self.assertTrue(scheduler.release_from_ng_box(21))

    def test_disabled_ng_does_not_require_signal(self):
        scheduler = self.scheduler(False)
        self.complete_ng(scheduler, 10)
        self.assertEqual(scheduler.state, RunState.WaitingTrigger)
        self.assertTrue(scheduler.can_accept_trigger().allowed)
        self.assertFalse(scheduler.release_from_ng_box(11))

    def test_error_lock_can_be_released_by_fresh_edge(self):
        scheduler = self.scheduler()
        with patch("services.inspection_scheduler.time.time", return_value=10):
            scheduler.on_error()
        self.assertFalse(scheduler.release_from_ng_box(9))
        self.assertTrue(scheduler.release_from_ng_box(11))

    def test_initial_high_does_not_emit_edge_and_bounce_is_filtered(self):
        class Io:
            state = True
            def read_input(self, name):
                return self.state
        io = Io()
        poller = DiPoller(io, input_names=["ng_box"], debounce_ms=50)
        events = []
        poller.add_change_callback(events.append)
        poller._initialize_states()
        poller._process_input("ng_box", 1)
        self.assertFalse(events)
        io.state = False
        poller._process_input("ng_box", 2)
        poller._process_input("ng_box", 2.1)
        io.state = True
        poller._process_input("ng_box", 3)
        io.state = False
        poller._process_input("ng_box", 3.01)
        poller._process_input("ng_box", 3.1)
        self.assertEqual([event.edge for event in events], ["falling"])
        io.state = True
        poller._process_input("ng_box", 4)
        poller._process_input("ng_box", 4.1)
        self.assertEqual([event.edge for event in events], ["falling", "rising"])


if __name__ == "__main__":
    unittest.main()
