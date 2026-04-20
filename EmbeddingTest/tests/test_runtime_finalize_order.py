from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.execution import _finalize_trigger_outcome


class _Signal:
    def __init__(self) -> None:
        self.values: list[tuple] = []

    def emit(self, *args) -> None:
        self.values.append(tuple(args))


class _TowerLight:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def show_ng(self) -> None:
        self._events.append("tower_ng")

    def show_ok(self) -> None:
        self._events.append("tower_ok")


class _Runtime:
    def __init__(self) -> None:
        self.events: list[str] = []
        self._last_record_path = ""
        self._record_service = None
        self._last_preview_frames = {}
        self._frame_lock = threading.RLock()
        self._capture_retention_policy = "ng_only"
        self._session = SimpleNamespace(current_product="demo", product_dir="")
        self._runtime_context = SimpleNamespace(inspection_items=[])
        self._last_item_results_by_camera = {}
        self._lock_on_ng = False
        self._last_capture_paths = {}
        self._tower_light_controller = _TowerLight(self.events)
        self.recordPathChanged = _Signal()
        self.previewUpdated = _Signal()
        self.triggerResultReady = _Signal()
        self.logAppended = _Signal()

    def _connected_roles(self) -> list[str]:
        return ["cam1"]

    def _set_conveyor_run(self, running: bool, *, reason: str = "") -> bool:
        self.events.append(f"conveyor_{'run' if running else 'stop'}")
        return True

    def _write_runtime_record(self, runtime_result) -> None:
        self.events.append("write_record")

    def _write_release_log(self, **_kwargs) -> None:
        self.events.append("write_release_log")

    def _emit_runtime_context(self) -> None:
        self.events.append("emit_context")

    def _update_status(self, _message=None) -> None:
        self.events.append("update_status")


class RuntimeFinalizeOrderTest(unittest.TestCase):
    def test_ng_stops_conveyor_and_shows_tower_light_before_record(self) -> None:
        runtime = _Runtime()
        outcome = SimpleNamespace(
            final_result="NG",
            camera_outcomes={
                "cam1": SimpleNamespace(
                    result="NG",
                    message="cam1 pred=NG",
                    capture_ms=1.0,
                    match_ms=2.0,
                    infer_ms=3.0,
                )
            },
            duration_ms=10,
            error_message="",
        )

        _finalize_trigger_outcome(runtime, outcome, release_status_before=None)

        self.assertEqual(runtime.events.count("conveyor_stop"), 1)
        self.assertLess(runtime.events.index("conveyor_stop"), runtime.events.index("tower_ng"))
        self.assertLess(runtime.events.index("tower_ng"), runtime.events.index("write_record"))


if __name__ == "__main__":
    unittest.main()
