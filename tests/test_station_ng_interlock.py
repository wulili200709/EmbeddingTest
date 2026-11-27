import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets

from application.runtime.controller import RuntimeController
from infrastructure.audit_store import RuntimeRecordStore
from services.inspection_scheduler import InspectionScheduler
from services.permission_manager import PermissionManager
from services.run_state import RunState, RunStateMachine
from ui.runtime.runtime_mode_pyside6 import RuntimeModePage


class StationNgInterlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = Path(folder.name) / "audit.db"
        self.store = RuntimeRecordStore(self.db)

    def scheduler(self):
        manager = PermissionManager("test", interlock_store=RuntimeRecordStore(self.db))
        return InspectionScheduler(RunStateMachine(), manager)

    def test_abrupt_exit_product_change_reconnect_and_fresh_di2_edge(self):
        # Exit without running shutdown handlers after a production NG.
        subprocess.run([sys.executable, "-c", """
import os, sys
from infrastructure.audit_store import RuntimeRecordStore
from services.permission_manager import PermissionManager
from services.inspection_scheduler import InspectionScheduler
from services.run_state import RunStateMachine
s = InspectionScheduler(RunStateMachine(), PermissionManager("test", interlock_store=RuntimeRecordStore(sys.argv[1])))
s.on_inspecting_started()
s.on_aggregating_started()
s.on_completed(final_ok=False)
os._exit(0)
""", str(self.db)], check=True, cwd=Path(__file__).resolve().parents[1], timeout=60)
        with patch("infrastructure.audit_store.default_audit_db_path", return_value=self.db):
            product_dir = self.db.parent / "other_product"
            product_dir.mkdir()
            ctrl = RuntimeController(
                SimpleNamespace(current_product="other_product", product_dir=product_dir,
                                camera_settings_path=product_dir / "camera_settings.json"),
                Mock(), Mock(), release_password="test", lock_on_ng=False,
            )
        self.addCleanup(ctrl.shutdown_persistence)
        self.addCleanup(ctrl.shutdown_trigger_worker)
        ctrl._emit_runtime_context = Mock()
        ctrl._connected_roles = Mock(return_value=[])
        ctrl._build_pending_runtime_result = Mock()
        states, results, previews = [], [], []
        ctrl.runtimeStateChanged.connect(states.append)
        ctrl.triggerResultReady.connect(lambda code, detail: results.append(code))
        ctrl.previewCycleStarted.connect(lambda *args: previews.append(args))
        ctrl._update_status()
        self.assertEqual(states[-1], "LockedByNg")
        with patch.object(ctrl, "_try_create_io_controller", return_value=None), patch(
            "application.runtime.controller.HikCameraManager"
        ):
            self.assertTrue(ctrl._rebuild_runner())
            ctrl.disconnect(silent=True)
            self.assertEqual(states[-1], "LockedByNg")
            self.assertTrue(ctrl._rebuild_runner())
        self.assertFalse(ctrl._scheduler.begin_precheck().allowed)
        ctrl._run_trigger()
        ctrl._run_trigger_camera(1)
        self.assertEqual(results, ["BLOCKED", "BLOCKED"])
        self.assertFalse(previews)
        ctrl.release("test")
        self.assertEqual(self.store.load_ng_interlock(), "locked")
        ctrl._on_ng_box_event(SimpleNamespace(edge="rising", state=True, timestamp=0))
        self.assertEqual(self.store.load_ng_interlock(), "locked")
        ctrl._on_ng_box_event(SimpleNamespace(edge="rising", state=True, timestamp=time.time() + 1))
        self.assertEqual(self.store.load_ng_interlock(), "released")
        self.assertTrue(ctrl._scheduler.begin_precheck().allowed)
        restored = self.scheduler()
        self.assertEqual(restored.state, RunState.ReleasedPendingConsume)
        restored.on_capture_started(1)
        self.assertEqual(self.store.load_ng_interlock(), "waiting")
        restored.on_error()
        self.assertFalse(self.scheduler().can_accept_trigger().allowed)

    def test_failed_release_commit_keeps_station_locked_until_retry(self):
        scheduler = self.scheduler()
        scheduler.on_error()
        with patch.object(RuntimeRecordStore, "save_ng_interlock", side_effect=sqlite3.OperationalError("disk full")):
            self.assertFalse(scheduler.release_from_ng_box(time.time() + 1))
        self.assertFalse(scheduler.can_accept_trigger().allowed)
        self.assertEqual(self.store.load_ng_interlock(), "locked")
        self.assertEqual(scheduler.permission_manager.storage_error, "disk full")
        self.assertTrue(scheduler.release_from_ng_box(time.time() + 2))

    def test_old_database_without_an_interlock_starts_ready(self):
        old_db = self.db.parent / "old.db"
        with closing(sqlite3.connect(old_db)) as conn, conn:
            conn.execute("CREATE TABLE historical_data (value TEXT)")
            conn.execute("INSERT INTO historical_data VALUES ('NG')")
        store = RuntimeRecordStore(old_db)
        self.assertEqual(store.load_ng_interlock(), "waiting")
        with closing(sqlite3.connect(old_db)) as conn:
            self.assertEqual(conn.execute("SELECT value FROM historical_data").fetchone(), ("NG",))

    def test_runtime_buttons_follow_interlock_and_operator_permission(self):
        page = RuntimeModePage()
        self.addCleanup(page.close)
        page.set_configured_camera_roles(["cam1"])
        page.set_active_camera_roles(["cam1"])
        page.set_inspection_items([{"camera_id": "cam1", "item_id": "roi1"}])
        page.set_runtime_state("LockedByNg")
        page.set_trigger_permission(True)
        self.assertFalse(page.btn_simulate_foot.isEnabled())
        self.assertFalse(page.btn_trigger_cam1.isEnabled())
        page.set_trigger_permission(False)
        page.set_runtime_state("ReleasedPendingConsume")
        self.assertFalse(page.btn_trigger_cam1.isEnabled())
        page.set_trigger_permission(True)
        self.assertTrue(page.btn_simulate_foot.isEnabled())
        self.assertTrue(page.btn_trigger_cam1.isEnabled())
