from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime import execution
from application.runtime.preview_frame import build_runtime_preview_frame
from application.runtime_controller import RuntimeController
from domain.inspection_items import InspectionItem
from services.inspection_runtime import FinalInspectionOutcome


class _FakeRuntimeContext:
    def __init__(self, items) -> None:
        self.inspection_items = list(items)

    def reload(self) -> None:
        return None


class _FakeFrameGrabService:
    def roles(self):
        return ["cam1"]


class _BlockingRecordService:
    def __init__(self, record_path: Path) -> None:
        self._record_path = Path(record_path)
        self.writer = SimpleNamespace(file_path_for_date=lambda dt=None: self._record_path)
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls: list[dict] = []

    def write_product_result(self, **kwargs):
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("record write release was not signaled")
        self.calls.append(dict(kwargs))
        self.finished.set()
        return self._record_path


class RuntimeAsyncPersistenceTest(unittest.TestCase):
    def _build_controller(self, product_dir: str) -> RuntimeController:
        items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
            )
        ]
        controller = RuntimeController(
            session=SimpleNamespace(
                current_product="Demo",
                line2dup_recipe_path="",
                product_dir=product_dir,
            ),
            algo=SimpleNamespace(),
            runtime_context=_FakeRuntimeContext(items),
        )
        controller._frame_grab_service = _FakeFrameGrabService()
        controller._scheduler = SimpleNamespace(state=SimpleNamespace(value="WaitingTrigger"))
        return controller

    def test_finalize_emits_result_before_async_record_write_finishes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            controller = self._build_controller(tmpdir)
            controller.set_capture_retention_policy("ng_only")
            record_service = _BlockingRecordService(Path(tmpdir) / "runtime_records" / "2026-04-08.csv")
            controller._record_service = record_service

            result_event = threading.Event()
            result_payload: list[tuple[str, str]] = []
            controller.triggerResultReady.connect(
                lambda result, detail: (result_payload.append((result, detail)), result_event.set())
            )

            outcome = FinalInspectionOutcome(
                final_result="OK",
                camera_outcomes={
                    "cam1": SimpleNamespace(
                        result="OK",
                        message="cam1 ok",
                        capture_ms=12.0,
                        match_ms=23.0,
                        infer_ms=34.0,
                    )
                },
                duration_ms=78,
            )

            try:
                controller._finalize_trigger_outcome(outcome, release_status_before=None)
                self.assertTrue(result_event.wait(0.3))
                self.assertTrue(record_service.started.wait(0.3))
                self.assertFalse(record_service.finished.is_set())
                self.assertEqual(result_payload[0][0], "OK")
                self.assertEqual(controller._last_runtime_result.final_result, "OK")
            finally:
                record_service.release.set()
                controller.shutdown_persistence(wait=True)

            self.assertTrue(record_service.finished.is_set())
            self.assertEqual(len(record_service.calls), 1)

    def test_finalize_emits_result_before_async_capture_export_finishes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            controller = self._build_controller(tmpdir)
            controller.set_capture_retention_policy("all")
            controller._last_preview_frames["cam1"] = build_runtime_preview_frame(
                role="cam1",
                image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
                product_dir=tmpdir,
                camera_role="cam1",
            )

            result_event = threading.Event()
            export_started = threading.Event()
            release_export = threading.Event()
            controller.triggerResultReady.connect(lambda *_args: result_event.set())

            def _blocking_export(frame, capture_dir, *, stamp=None):
                self.assertTrue(result_event.is_set())
                export_started.set()
                if not release_export.wait(3):
                    raise TimeoutError("capture export release was not signaled")
                target_dir = Path(capture_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / "async_cam1.png"
                target_path.write_bytes(b"png")
                target_path.with_suffix(".json").write_text("{}", encoding="utf-8")
                return build_runtime_preview_frame(
                    role=frame.role,
                    image_bgr=frame.image_bgr,
                    source_path=str(target_path),
                    product_dir=frame.product_dir,
                    camera_role=frame.camera_role,
                    roi_shapes=frame.roi_shapes,
                )

            outcome = FinalInspectionOutcome(
                final_result="OK",
                camera_outcomes={
                    "cam1": SimpleNamespace(
                        result="OK",
                        message="cam1 ok",
                        capture_ms=10.0,
                        match_ms=20.0,
                        infer_ms=30.0,
                    )
                },
                duration_ms=60,
            )

            with patch.object(execution, "export_runtime_preview_frame", side_effect=_blocking_export):
                try:
                    controller._finalize_trigger_outcome(outcome, release_status_before=None)
                    self.assertTrue(result_event.wait(0.3))
                    self.assertTrue(export_started.wait(0.3))
                    self.assertEqual(controller._last_capture_paths, {})
                finally:
                    release_export.set()
                    controller.shutdown_persistence(wait=True)

            self.assertIn("cam1", controller._last_capture_paths)
            self.assertTrue(Path(controller._last_capture_paths["cam1"]).exists())


if __name__ == "__main__":
    unittest.main()
