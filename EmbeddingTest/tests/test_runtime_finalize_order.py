from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.execution import _finalize_trigger_outcome
from application.runtime.hardware import _apply_io_logic_event
from application.runtime.preview_frame import build_runtime_preview_frame


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

    def _set_buzzer(self, on: bool, *, reason: str = "") -> bool:
        self.events.append(f"buzzer_{'on' if on else 'off'}")
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
    def test_ng_stops_conveyor_turns_on_buzzer_and_shows_tower_light_before_record(self) -> None:
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
        self.assertEqual(runtime.events.count("buzzer_on"), 1)
        self.assertLess(runtime.events.index("conveyor_stop"), runtime.events.index("tower_ng"))
        self.assertLess(runtime.events.index("buzzer_on"), runtime.events.index("tower_ng"))
        self.assertLess(runtime.events.index("tower_ng"), runtime.events.index("write_record"))

    def test_empty_ng_override_disables_default_io_fallback(self) -> None:
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

        with TemporaryDirectory() as tmp:
            runtime._session.product_dir = tmp
            (Path(tmp) / "runtime_io_logic.json").write_text(
                json.dumps({"ng": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime._apply_io_logic_event = lambda event_name, **kwargs: _apply_io_logic_event(
                runtime,
                event_name,
                **kwargs,
            )

            _finalize_trigger_outcome(runtime, outcome, release_status_before=None)

        self.assertNotIn("conveyor_stop", runtime.events)
        self.assertNotIn("buzzer_on", runtime.events)
        self.assertIn("tower_ng", runtime.events)
        self.assertIn("write_record", runtime.events)

    def test_error_ng_without_camera_outcomes_exports_latest_preview_frame(self) -> None:
        runtime = _Runtime()
        outcome = SimpleNamespace(
            final_result="NG",
            camera_outcomes={},
            duration_ms=7,
            error_message="match failure",
        )

        with TemporaryDirectory() as tmp:
            runtime._session.product_dir = tmp
            runtime._capture_retention_policy = "all"
            runtime._last_preview_frames = {
                "cam1": build_runtime_preview_frame(
                    role="cam1",
                    image_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
                    product_dir=tmp,
                    camera_role="cam1",
                )
            }

            _finalize_trigger_outcome(runtime, outcome, release_status_before=None)

            exported_path = Path(runtime._last_capture_paths.get("cam1", ""))
            self.assertTrue(exported_path.exists())
            self.assertEqual(exported_path.suffix.lower(), ".png")
            self.assertIn("match failure", runtime.logAppended.values[-2][0])
            self.assertEqual(
                runtime._last_runtime_result.camera_results["cam1"].image_path,
                str(exported_path),
            )


if __name__ == "__main__":
    unittest.main()
