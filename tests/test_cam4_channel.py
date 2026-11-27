import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtWidgets

from application.runtime import execution
from application.runtime.capture_plan import build_capture_plan
from common.camera_roles import CAMERA_ROLES
from domain.inspection_models import CameraRuntimeResult, InspectionItemResult, RuntimeInspectionResult
from infrastructure.camera_settings_store import CameraSettingsStore
from services.inspection_runtime import CameraInspectionOutcome
from services.inspection_scheduler import InspectionScheduler
from services.permission_manager import PermissionManager
from services.record_writer import CsvRecordWriter, TestRecordService
from services.run_state import RunState, RunStateMachine
from shape.core.locator import product_paths
from ui.debug.tool_page import camera_debug, camera_debug_view
from ui.runtime.runtime_mode_pyside6 import RuntimeModePage


class FourthChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_configure_fourth_row_on_existing_product_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CameraSettingsStore(Path(folder) / "camera_settings.json")
            roles = ("cam1", "cam2", "cam3")
            store.save_capture_config("flexible", [{"role": role, "physical_role": "cam1"} for role in roles])
            owner = Mock()
            styles = dict.fromkeys(("dark_bg", "panel_bg", "header_bg", "text_light", "text_dim"), "#333333")
            styles.update(input_style="", compact_btn="")
            camera_debug_view.build_camera_debug_page(
                owner, styles=styles, standard_icon=self.app.style().standardIcon,
                standard_pixmap=QtWidgets.QStyle.StandardPixmap,
            )
            owner.cmb_capture_mode.setCurrentIndex(owner.cmb_capture_mode.findData("flexible"))
            table = owner.capture_channel_table
            self.assertEqual(table.rowCount(), 4)
            for i, channel in enumerate(store.load_capture_config()["capture_channels"]):
                camera_debug._set_capture_channel_row(owner, i, channel)
            self.assertEqual(table.item(3, 0).checkState(), QtCore.Qt.CheckState.Unchecked)
            self.assertEqual(table.item(3, 1).text(), "cam4")
            physical = table.cellWidget(3, 2)
            physical.setCurrentIndex(physical.findData("cam1"))
            table.item(3, 0).setCheckState(QtCore.Qt.CheckState.Checked)
            store.save_capture_config("flexible", camera_debug._capture_channels_from_ui(owner))
            plan = build_capture_plan(CameraSettingsStore(store.path).load_capture_config())
            self.assertEqual(plan.logical_roles, (*roles, "cam4"))
            self.assertEqual(plan.physical_roles, ("cam1",))
            self.assertEqual(plan.channel_for_role("cam4").light_output, "DO_LIGHT_CAM1")
            self.assertNotEqual(product_paths(folder, "cam1").recipe_path, product_paths(folder, "cam4").recipe_path)
            owner.camera_debug_page.close()
            owner.capture_plan_page.close()

    def test_fourth_exposure_uses_same_camera_and_can_fail_the_workpiece(self):
        roles = ("cam1", "cam2", "cam3", "cam4")
        with tempfile.TemporaryDirectory() as folder:
            store = CameraSettingsStore(Path(folder) / "camera_settings.json")
            store.save_capture_config("flexible", [
                {"role": role, "physical_role": "cam1", "light_output": "DO_LIGHT_CAM1", "stable_delay_ms": 0}
                for role in roles
            ])
            scheduler = InspectionScheduler(RunStateMachine(), PermissionManager("test"), lock_on_ng=False)
            frames = Mock()
            frames.capture_once.side_effect = range(1, 5)
            def inspect_frame(role, frame, *, physical_role):
                return CameraInspectionOutcome(role, "NG" if role == "cam4" else "OK")

            inspect = Mock(side_effect=inspect_frame)
            runtime = SimpleNamespace(
                _camera_settings_store=store, _scheduler=scheduler, _frame_grab_service=frames,
                _inspect_frame=inspect, logAppended=Mock(), _tower_light_controller=Mock(), _light_controller=Mock(),
            )
            with (
                patch.object(execution, "_precheck_for_capture_channels", return_value=(True, "")),
                patch.object(execution, "_apply_capture_channel_camera_settings"),
            ):
                result = execution._run_single_multi_light_trigger(runtime)
            self.assertEqual(frames.capture_once.call_args_list, [call("cam1", timeout_ms=1000)] * 4)
            self.assertEqual(inspect.call_args_list, [call(role, i, physical_role="cam1") for i, role in enumerate(roles, 1)])
            self.assertEqual(tuple(result.camera_outcomes), roles)
            self.assertEqual(result.final_result, "NG")
            self.assertEqual(scheduler.state, RunState.WaitingTrigger)
            runtime._tower_light_controller.show_ng.assert_called_once()

    def test_same_roi_name_in_four_views_is_saved_in_one_product_record(self):
        roles = ("cam1", "cam2", "cam3", "cam4")
        items = [
            InspectionItemResult(role, "feature", role, "feature", result="NG" if role == "cam4" else "OK")
            for role in roles
        ]
        result = RuntimeInspectionResult(
            "part-1", "example", final_result="NG", item_results=items,
            camera_results={item.camera_id: CameraRuntimeResult(item.camera_id, item.result) for item in items},
        )
        with tempfile.TemporaryDirectory() as folder:
            service = TestRecordService(CsvRecordWriter(folder))
            execution._write_runtime_record_sync(SimpleNamespace(logAppended=Mock()), service, result, lock_on_ng=False)
            with next(Path(folder).glob("*.csv")).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual([rows[0][f"{role}.feature"] for role in roles], ["OK", "OK", "OK", "NG"])

    def test_four_previews_and_sparse_cam4_layout(self):
        page = RuntimeModePage()
        for roles in (["cam1", "cam2", "cam3", "cam4"], ["cam1", "cam3", "cam4"], ["cam4"]):
            page.set_configured_camera_roles(roles)
            self.assertEqual(page._camera_grid.count(), len(roles))
            canvases = [page._camera_grid.itemAt(i).widget()._view for i in range(len(roles))]
            self.assertCountEqual(canvases, [getattr(page, f"view_{role}") for role in roles])
        triggered = []
        page.triggerCameraRequested.connect(triggered.append)
        page.btn_trigger_cam4.setEnabled(True)
        page.btn_trigger_cam4.click()
        self.assertEqual(triggered, [4])
        page.close()

    def test_trigger_buttons_follow_enabled_channels_on_one_physical_camera(self):
        page = RuntimeModePage()
        self.addCleanup(page.close)
        buttons = {role: getattr(page, f"btn_trigger_{role}") for role in CAMERA_ROLES}
        page.set_physical_camera_roles(["cam1"])
        page.set_inspection_items([{"camera_id": role, "item_id": role} for role in CAMERA_ROLES])
        triggered = []
        page.triggerCameraRequested.connect(triggered.append)

        for roles in [("cam1",), ("cam1", "cam3"), ("cam4",), (), CAMERA_ROLES]:
            with self.subTest(roles=roles):
                plan = build_capture_plan({
                    "capture_mode": "flexible",
                    "capture_channels": [
                        {"role": role, "physical_role": "cam1", "enabled": role in roles}
                        for role in CAMERA_ROLES
                    ],
                })
                page.set_configured_camera_roles(list(plan.logical_roles))
                page.set_active_camera_roles([])
                self.assertEqual([role for role, btn in buttons.items() if not btn.isHidden()], list(roles))
                self.assertFalse(any(btn.isEnabled() for btn in buttons.values()))
                page.set_active_camera_roles(list(plan.active_logical_roles(["cam1"])))
                triggered.clear()
                for button in buttons.values():
                    button.click()
                self.assertEqual(triggered, [CAMERA_ROLES.index(role) + 1 for role in roles])
