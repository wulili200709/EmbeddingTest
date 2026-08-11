from __future__ import annotations

import tempfile
import threading
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PySide6 import QtWidgets

from application.runtime.capture_plan import build_capture_plan
from application.runtime import controller as controller_module
from application.runtime import execution
from application.runtime.preview_frame import RuntimePreviewFrame
from common.runtime_camera_logging import RuntimeCameraLogService
from infrastructure.camera_settings_store import (
    CAMERA_SETTINGS_SCHEMA_VERSION,
    CameraSettingsStore,
)
from ui.debug.tool_page import camera_debug
from ui.runtime.runtime_mode_pyside6 import RuntimeModePage
from ui.window_common import _runtime_source_pixmap, update_runtime_preview


def _flexible_config() -> dict:
    return {
        "capture_mode": "flexible",
        "capture_channels": [
            {
                "enabled": True,
                "role": "cam1",
                "physical_role": "cam1",
                "light_output": "DO_LIGHT_CAM1",
            },
            {
                "enabled": True,
                "role": "cam2",
                "physical_role": "cam1",
                "light_output": "DO_LIGHT_CAM2",
            },
            {
                "enabled": True,
                "role": "cam3",
                "physical_role": "cam2",
                "light_output": "DO_LIGHT_CAM3",
            },
        ],
    }


class CapturePlanTests(unittest.TestCase):
    def test_unsaved_physical_camera_resets_stale_digital_shift_ui(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        role_combo = QtWidgets.QComboBox()
        role_combo.addItem("cam2", "cam2")
        camera_store = mock.Mock()
        camera_store.load_capture_config.return_value = {
            "capture_mode": "independent",
            "capture_channels": [],
        }
        camera_store.load_for_role.return_value = {}

        digital_shift_enable = QtWidgets.QCheckBox()
        digital_shift_enable.setChecked(True)
        digital_shift = QtWidgets.QDoubleSpinBox()
        digital_shift.setRange(0.0, 16.0)
        digital_shift.setValue(5.5)
        digital_shift.setEnabled(True)
        light_mode = QtWidgets.QComboBox()
        light_mode.addItem("Board IO", "board_io")
        tool_page = SimpleNamespace(
            cmb_debug_camera_role=role_combo,
            _camera_settings_store=camera_store,
            _debug_camera_block_spin_apply=False,
            _selected_debug_camera_role=lambda: "cam2",
            _debug_physical_camera_role=lambda _role: "cam2",
            _debug_capture_channel_for_role=lambda _role: {},
            spin_debug_exposure=QtWidgets.QDoubleSpinBox(),
            spin_debug_gain=QtWidgets.QDoubleSpinBox(),
            chk_debug_digital_shift_enable=digital_shift_enable,
            spin_debug_digital_shift=digital_shift,
            cmb_debug_trigger_mode=QtWidgets.QComboBox(),
            cmb_debug_light_source_mode=light_mode,
        )

        loaded = camera_debug._load_saved_debug_camera_settings_to_ui(tool_page, "SERIAL-2")

        self.assertFalse(loaded)
        self.assertFalse(digital_shift_enable.isChecked())
        self.assertFalse(digital_shift.isEnabled())
        self.assertEqual(digital_shift.value(), 0.0)
        camera_store.load_for_role.assert_called_once_with("cam2", serial="SERIAL-2")
        app.processEvents()

    def test_connect_releases_stale_physical_binding_before_rebuild(self) -> None:
        class Signal:
            def emit(self, *_args) -> None:
                pass

        runtime = SimpleNamespace(
            _sync_camera_settings_store_path=mock.Mock(),
            logAppended=Signal(),
            disconnect=mock.Mock(),
            _rebuild_runner=mock.Mock(return_value=False),
        )
        with mock.patch.object(
            controller_module,
            "physical_connected_bindings",
            return_value={"cam1": "S1"},
        ):
            controller_module.RuntimeController.connect_cameras(runtime, {"cam3": "S3"})

        runtime.disconnect.assert_called_once_with(silent=True, close_io=False)
        runtime._rebuild_runner.assert_called_once_with()

    def test_connect_keeps_identical_physical_binding_open(self) -> None:
        class Signal:
            def emit(self, *_args) -> None:
                pass

        runtime = SimpleNamespace(
            _sync_camera_settings_store_path=mock.Mock(),
            warningOccurred=Signal(),
            logAppended=Signal(),
            _update_status=mock.Mock(),
            disconnect=mock.Mock(),
            _rebuild_runner=mock.Mock(return_value=False),
        )
        with mock.patch.object(
            controller_module,
            "physical_connected_bindings",
            return_value={"cam3": "S3"},
        ):
            controller_module.RuntimeController.connect_cameras(runtime, {"cam3": "S3"})

        runtime.disconnect.assert_not_called()
        runtime._rebuild_runner.assert_not_called()

    def test_preview_conversion_handles_single_channel_and_uint16_frames(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        images = (
            np.full((6, 8, 1), 127, dtype=np.uint8),
            np.arange(48, dtype=np.uint16).reshape(6, 8) * 100,
        )
        for image in images:
            with self.subTest(shape=image.shape, dtype=str(image.dtype)):
                frame = RuntimePreviewFrame(role="cam3", image_bgr=image)
                pixmap = _runtime_source_pixmap(frame)
                self.assertFalse(pixmap.isNull())
                self.assertEqual((pixmap.width(), pixmap.height()), (8, 6))
        app.processEvents()

    def test_bgr_preview_uses_qt_bgr_format_without_swapping_colors(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, :] = (0, 0, 255)  # red expressed as BGR
        pixmap = _runtime_source_pixmap(RuntimePreviewFrame(role="cam3", image_bgr=image))
        color = pixmap.toImage().pixelColor(0, 0)
        self.assertGreater(color.red(), 240)
        self.assertLess(color.green(), 15)
        self.assertLess(color.blue(), 15)
        app.processEvents()

    def test_runtime_camera_log_filters_and_writes_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime_camera.log"
            service = RuntimeCameraLogService(
                path,
                max_bytes=4096,
                backup_count=2,
            )
            service.record("[capture] hidden while detail is disabled")
            service.record("[trigger-summary] trigger=T1 result=OK")
            service.record("[camera] capture failed: timeout")
            service.set_detailed(True, duration_seconds=60)
            service.record("[capture] visible while detail is enabled")
            service.shutdown()

            content = path.read_text(encoding="utf-8")
            self.assertNotIn("hidden while detail is disabled", content)
            self.assertIn("trigger=T1 result=OK", content)
            self.assertIn("capture failed: timeout", content)
            self.assertIn("visible while detail is enabled", content)

    def test_trigger_summary_contains_frame_route_without_image_data(self) -> None:
        frame = RuntimePreviewFrame(
            role="cam3",
            image_bgr=np.zeros((2, 2), dtype=np.uint8),
            trigger_id="T7",
            physical_role="cam2",
            camera_serial="S2",
            frame_number=88,
        )
        outcome = SimpleNamespace(
            final_result="OK",
            duration_ms=123,
            error_message="",
            camera_outcomes={
                "cam3": controller_module.CameraInspectionOutcome(role="cam3", result="OK")
            },
        )
        message = execution._trigger_summary_message(
            trigger_id="T7",
            outcome=outcome,
            roles=["cam3"],
            preview_frames={"cam3": frame},
        )
        self.assertIn("trigger=T7", message)
        self.assertIn("cam3:OK/physical=cam2/serial=S2/frame=88", message)
        self.assertNotIn("image_bgr", message)

    def test_independent_mode_preserves_every_selected_role_combination(self) -> None:
        all_roles = ("cam1", "cam2", "cam3")
        for size in range(1, len(all_roles) + 1):
            for roles in combinations(all_roles, size):
                with self.subTest(roles=roles):
                    plan = build_capture_plan({}, configured_roles=roles)
                    self.assertEqual(plan.logical_roles, roles)
                    self.assertEqual(plan.physical_roles, roles)

    def test_independent_sparse_roles_only_bind_selected_cameras(self) -> None:
        plan = build_capture_plan({}, configured_roles=["cam1", "cam3"])
        self.assertEqual(plan.logical_roles, ("cam1", "cam3"))
        self.assertEqual(plan.physical_roles, ("cam1", "cam3"))
        self.assertEqual(
            plan.physical_bindings({"cam1": "S1", "cam2": "S2", "cam3": "S3"}),
            {"cam1": "S1", "cam3": "S3"},
        )

    def test_independent_selection_overrides_stale_multi_light_channel_flags(self) -> None:
        stale_config = {
            "capture_mode": "independent",
            "capture_channels": [
                {"enabled": True, "role": "cam1", "physical_role": "cam1"},
                {"enabled": True, "role": "cam2", "physical_role": "cam1"},
                {"enabled": False, "role": "cam3", "physical_role": "cam1"},
            ],
        }

        plan = build_capture_plan(stale_config, configured_roles=["cam1", "cam3"])

        self.assertEqual(plan.logical_roles, ("cam1", "cam3"))
        self.assertEqual(plan.physical_roles, ("cam1", "cam3"))
        self.assertEqual(
            plan.physical_bindings({"cam1": "S1", "cam2": "S2", "cam3": "S3"}),
            {"cam1": "S1", "cam3": "S3"},
        )

    def test_mapped_mode_still_obeys_channel_enabled_flags(self) -> None:
        mapped_config = _flexible_config()
        mapped_config["capture_channels"][2]["enabled"] = False

        plan = build_capture_plan(mapped_config, configured_roles=["cam1", "cam3"])

        self.assertEqual(plan.logical_roles, ("cam1", "cam2"))
        self.assertEqual(plan.physical_roles, ("cam1",))

    def test_two_physical_cameras_can_feed_three_logical_channels(self) -> None:
        plan = build_capture_plan(_flexible_config(), configured_roles=["cam1"])
        self.assertTrue(plan.uses_channel_mapping)
        self.assertEqual(plan.logical_roles, ("cam1", "cam2", "cam3"))
        self.assertEqual(plan.physical_roles, ("cam1", "cam2"))
        self.assertEqual(
            [channel.physical_role for channel in plan.channels_for_roles()],
            ["cam1", "cam1", "cam2"],
        )
        self.assertEqual(
            plan.physical_bindings({"cam1": "S1", "cam2": "S2", "cam3": "STALE"}),
            {"cam1": "S1", "cam2": "S2"},
        )

    def test_mapped_trigger_captures_three_channels_from_two_physical_cameras(self) -> None:
        class Signal:
            def emit(self, *_args) -> None:
                pass

        class Store:
            @staticmethod
            def load_capture_config() -> dict:
                return _flexible_config()

        class Scheduler:
            @staticmethod
            def begin_precheck():
                return SimpleNamespace(allowed=True)

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        class Light:
            @staticmethod
            def requires_stable_delay(_index: int) -> bool:
                return False

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        class Frames:
            def __init__(self) -> None:
                self.calls: list[str] = []

            @staticmethod
            def roles() -> list[str]:
                return ["cam1", "cam2"]

            def capture_once(self, role: str, *, timeout_ms: int):
                self.calls.append(role)
                return SimpleNamespace(role=role, timeout_ms=timeout_ms)

        frames = Frames()
        inspected: list[tuple[str, str]] = []
        runtime = SimpleNamespace(
            _camera_settings_store=Store(),
            _frame_grab_service=frames,
            _scheduler=Scheduler(),
            _tower_light_controller=Light(),
            _light_controller=Light(),
            logAppended=Signal(),
        )

        def inspect(role: str, _frame, *, physical_role: str = ""):
            inspected.append((role, physical_role))
            return controller_module.CameraInspectionOutcome(role=role, result="OK")

        runtime._inspect_frame = inspect
        original_precheck = execution._precheck_for_capture_channels
        original_apply = execution._apply_capture_channel_camera_settings
        execution._precheck_for_capture_channels = lambda _runtime, _channels: (True, "")
        execution._apply_capture_channel_camera_settings = lambda _runtime, _channel: None
        try:
            outcome = execution._run_single_multi_light_trigger(runtime)
        finally:
            execution._precheck_for_capture_channels = original_precheck
            execution._apply_capture_channel_camera_settings = original_apply

        self.assertEqual(frames.calls, ["cam1", "cam1", "cam2"])
        self.assertEqual(
            inspected,
            [("cam1", "cam1"), ("cam2", "cam1"), ("cam3", "cam2")],
        )
        self.assertEqual(outcome.final_result, "OK")

    def test_duplicate_physical_serial_is_rejected(self) -> None:
        plan = build_capture_plan(_flexible_config())
        issues = plan.validate_bindings({"cam1": "SAME", "cam2": "SAME"})
        self.assertTrue(any("同一序列号" in issue for issue in issues))

    def test_stale_preview_trigger_is_not_accepted(self) -> None:
        page = SimpleNamespace(_camera_preview_trigger_ids={"cam3": "new-trigger"})
        old_frame = RuntimePreviewFrame(
            role="cam3",
            image_bgr=np.zeros((2, 2), dtype=np.uint8),
            trigger_id="old-trigger",
        )
        new_frame = RuntimePreviewFrame(
            role="cam3",
            image_bgr=np.zeros((2, 2), dtype=np.uint8),
            trigger_id="new-trigger",
        )
        accepts = RuntimeModePage.accepts_camera_preview
        self.assertFalse(accepts(page, "cam3", old_frame))
        self.assertTrue(accepts(page, "cam3", new_frame))

    def test_legacy_capture_config_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "camera_settings.json"
            path.write_text(
                '{"capture_mode":"single_multi_light","capture_channels":[]}',
                encoding="utf-8",
            )
            config = CameraSettingsStore(path).load_capture_config()
            self.assertEqual(config["schema_version"], CAMERA_SETTINGS_SCHEMA_VERSION)
            self.assertIn(f'"schema_version": {CAMERA_SETTINGS_SCHEMA_VERSION}', path.read_text(encoding="utf-8"))

    def test_preview_frame_records_logical_and_physical_identity(self) -> None:
        class Signal:
            def __init__(self) -> None:
                self.events: list[tuple] = []

            def emit(self, *args) -> None:
                self.events.append(args)

        class Executor:
            @staticmethod
            def execute(_request):
                return SimpleNamespace(
                    item_results=[],
                    roi_shapes=(),
                    measurements=(),
                    result="OK",
                    detail="",
                    match_ms=1.0,
                    infer_ms=2.0,
                )

        preview_signal = Signal()
        runtime = SimpleNamespace(
            _session=SimpleNamespace(product_dir=""),
            _last_runtime_result=SimpleNamespace(task_id="trigger-7"),
            _frame_lock=threading.RLock(),
            _inspect_lock=threading.RLock(),
            _last_preview_frames={},
            _last_item_results_by_camera={},
            _inspection_executor=Executor(),
            _runtime_context=SimpleNamespace(inspection_items=[]),
            previewUpdated=preview_signal,
            logAppended=Signal(),
        )
        frame = SimpleNamespace(
            camera_serial="SERIAL-2",
            frame_num=88,
            host_timestamp=123456,
        )
        original_converter = controller_module.frame_to_bgr_image
        controller_module.frame_to_bgr_image = lambda _frame: np.zeros((4, 6, 3), dtype=np.uint8)
        try:
            outcome = execution._inspect_frame(
                runtime,
                "cam3",
                frame,
                physical_role="cam2",
            )
        finally:
            controller_module.frame_to_bgr_image = original_converter
        self.assertEqual(outcome.result, "OK")
        preview = preview_signal.events[0][1]
        self.assertEqual(preview.trigger_id, "trigger-7")
        self.assertEqual(preview.role, "cam3")
        self.assertEqual(preview.physical_role, "cam2")
        self.assertEqual(preview.camera_serial, "SERIAL-2")
        self.assertEqual(preview.frame_number, 88)

    def test_runtime_page_separates_physical_roles_and_rejects_old_frame(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        page = RuntimeModePage()
        page.set_configured_camera_roles(["cam1", "cam2", "cam3"])
        page.set_active_camera_roles(["cam1", "cam2", "cam3"])
        page.set_physical_camera_roles(["cam1", "cam2"])
        page.set_physical_camera_bindings({"cam1": "S1", "cam2": "S2"})
        self.assertIn("物理相机: cam1(S1), cam2(S2)", page.lbl_footer_connection.text())
        self.assertIn("检测通道: cam1, cam2, cam3", page.lbl_footer_connection.text())

        page.begin_camera_preview_cycle("cam3", "new-trigger")
        old_frame = RuntimePreviewFrame(
            role="cam3",
            image_bgr=np.full((6, 8, 3), 50, dtype=np.uint8),
            trigger_id="old-trigger",
        )
        update_runtime_preview(page, "cam3", old_frame)
        self.assertIsNone(page.view_cam3._pixmap)

        new_frame = RuntimePreviewFrame(
            role="cam3",
            image_bgr=np.full((6, 8, 3), 100, dtype=np.uint8),
            trigger_id="new-trigger",
        )
        update_runtime_preview(page, "cam1", new_frame)
        self.assertIsNotNone(page.view_cam3._pixmap)
        self.assertIsNone(page.view_cam1._pixmap)
        page.deleteLater()
        app.processEvents()

    def test_sparse_cam1_cam3_layout_rebinds_cam3_view_to_canvas_two(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        page = RuntimeModePage()
        page.resize(1124, 700)
        page.set_configured_camera_roles(["cam1", "cam3"])
        page.set_active_camera_roles(["cam1", "cam3"])
        page.show()
        app.processEvents()

        self.assertEqual(page._camera_slot_roles[:2], ["cam1", "cam3"])
        self.assertIs(page._camera_slots[1]._view, page.view_cam3)
        self.assertTrue(page.view_cam3.isVisible())
        self.assertGreater(page.view_cam3.width(), 0)
        self.assertGreater(page.view_cam3.height(), 0)

        frame = RuntimePreviewFrame(
            role="cam3",
            image_bgr=np.full((60, 80, 3), 180, dtype=np.uint8),
        )
        update_runtime_preview(page, "cam3", frame)
        app.processEvents()
        pixmap = page.view_cam3.pixmap()
        self.assertIsNotNone(pixmap)
        self.assertFalse(pixmap.isNull())
        self.assertEqual(page.view_cam3.text(), "")
        page.close()
        page.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
