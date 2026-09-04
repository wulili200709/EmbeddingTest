from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from application.runtime.execution import _finalize_trigger_outcome
from ui.runtime.runtime_mode_pyside6 import RuntimeModePage


class _Signal:
    def __init__(self, callback=None) -> None:
        self._callback = callback

    def emit(self, *args) -> None:
        if self._callback is not None:
            self._callback(*args)


class RuntimeResultCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_first_ng_is_not_reset_before_async_record_write(self) -> None:
        page = RuntimeModePage()
        with tempfile.TemporaryDirectory() as temporary_dir:
            record_path = Path(temporary_dir) / "daily.csv"
            record_path.write_text("", encoding="utf-8")
            runtime = SimpleNamespace(
                _session=SimpleNamespace(
                    current_product="product-1",
                    product_dir=temporary_dir,
                    shape_recipe_path="",
                ),
                _runtime_context=SimpleNamespace(inspection_items=[]),
                _last_runtime_result=SimpleNamespace(task_id="pending"),
                _last_preview_frames={},
                _last_item_results_by_camera={},
                _record_service=SimpleNamespace(
                    writer=SimpleNamespace(file_path_for_date=lambda: record_path)
                ),
                _last_record_path=None,
                _capture_retention_policy="ng",
                _lock_on_ng=False,
            )
            runtime.recordPathChanged = _Signal(page.set_record_path)
            runtime.triggerResultReady = _Signal(page.set_final_result)
            runtime.logAppended = _Signal()
            runtime.previewUpdated = _Signal()
            runtime._update_status = lambda _message: page.set_current_product("product-1")
            runtime._submit_persistence_task = lambda *_args, **_kwargs: None
            runtime._write_runtime_record = lambda _result: None
            runtime._write_release_log = lambda **_kwargs: None

            outcome = SimpleNamespace(
                final_result="NG",
                camera_outcomes={},
                duration_ms=1,
                error_message="match failure",
            )
            _finalize_trigger_outcome(
                runtime,
                outcome,
                None,
                active_roles=["cam1"],
            )

            self.assertEqual(page._ng_count_total, 1)
            self.assertEqual(page.lbl_ng_count.text(), "NG: 1")
        page.close()


if __name__ == "__main__":
    unittest.main()
