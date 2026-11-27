from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.product_session import ProductSession
from application.runtime.controller import RuntimeController
from domain.inspection_items import InspectionItem
from domain.inspection_models import InspectionItemResult
from services.inspection_runtime import FinalInspectionOutcome


class _FakeRuntimeContext:
    def __init__(self, items) -> None:
        self.inspection_items = list(items)

    def reload(self) -> None:
        return None


class _FakeFrameGrabService:
    def roles(self):
        return ["cam1"]


class _FakePermissionManager:
    def status(self):
        return SimpleNamespace(
            is_locked=False,
            has_pending_release=False,
            release_consumed=False,
        )


class _FakeRunner:
    def on_roles_trigger(self, roles):
        return FinalInspectionOutcome(
            final_result="NG",
            camera_outcomes={},
            duration_ms=7,
            error_message="match failure",
        )


class RuntimeFailureItemResetTest(unittest.TestCase):
    def test_trigger_clears_stale_item_results_before_failed_cycle(self) -> None:
        items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam1",
                roi_label="roi2",
            ),
        ]
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        with patch("infrastructure.audit_store.default_audit_db_path", return_value=Path(folder.name) / "audit.db"):
            controller = RuntimeController(
                session=ProductSession(folder.name),
                algo=SimpleNamespace(),
                runtime_context=_FakeRuntimeContext(items),
            )
        self.addCleanup(controller.shutdown_persistence)
        self.addCleanup(controller.shutdown_trigger_worker)
        controller._frame_grab_service = _FakeFrameGrabService()
        controller._permission_manager = _FakePermissionManager()
        controller._scheduler = SimpleNamespace(
            state=SimpleNamespace(value="WaitingTrigger"),
            can_accept_trigger=lambda: SimpleNamespace(reason=""),
        )
        controller._runner = _FakeRunner()
        controller._last_item_results_by_camera = {
            "cam1": [
                InspectionItemResult(
                    item_id="roi1",
                    display_name="ROI1",
                    camera_id="cam1",
                    roi_label="roi1",
                    result="OK",
                ),
                InspectionItemResult(
                    item_id="roi2",
                    display_name="ROI2",
                    camera_id="cam1",
                    roi_label="roi2",
                    result="OK",
                ),
            ]
        }

        controller._run_trigger()

        results = {item.item_id: item.result for item in controller._last_runtime_result.item_results}
        self.assertEqual(results, {"roi1": "NG", "roi2": "NG"})
        self.assertEqual(controller._last_runtime_result.final_result, "NG")
