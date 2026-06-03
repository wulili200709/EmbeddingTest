from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from services.inspection_scheduler import InspectionScheduler
from services.permission_manager import PermissionManager
from services.run_state import RunState, RunStateMachine


class InspectionSchedulerNoLockTest(unittest.TestCase):
    def test_ng_completion_returns_to_waiting_when_lock_disabled(self) -> None:
        scheduler = InspectionScheduler(
            state_machine=RunStateMachine(),
            permission_manager=PermissionManager("1234"),
            lock_on_ng=False,
        )

        scheduler.state_machine.transition_to(RunState.Inspecting)
        scheduler.state_machine.transition_to(RunState.Aggregating)
        scheduler.on_completed(final_ok=False)

        self.assertEqual(scheduler.state, RunState.WaitingTrigger)
        self.assertFalse(scheduler.permission_manager.is_locked)
        self.assertTrue(scheduler.can_accept_trigger().allowed)

    def test_error_returns_to_waiting_when_lock_disabled(self) -> None:
        scheduler = InspectionScheduler(
            state_machine=RunStateMachine(),
            permission_manager=PermissionManager("1234"),
            lock_on_ng=False,
        )

        scheduler.state_machine.transition_to(RunState.Inspecting)
        scheduler.on_error(lock_as_ng=True)

        self.assertEqual(scheduler.state, RunState.WaitingTrigger)
        self.assertFalse(scheduler.permission_manager.is_locked)
        self.assertTrue(scheduler.can_accept_trigger().allowed)


if __name__ == "__main__":
    unittest.main()
