"""Offscreen coverage of explicit calibration-to-physical-camera application.

All products, profile inputs and JSON exports are temporary. No camera SDK,
real calibration file or production configuration is accessed by these tests.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6 import QtCore, QtWidgets

from algorithms.lens_calibration import CalibrationProfile, manual_profile, save_profile
from services.lens_correction import get_config, prepare_array
from tools.lens_distortion_demo import LensDemoWindow
from ui.i18n import LANG_ZH, language_code, set_language
from ui.lens_correction_dialog import LensCorrectionSettingsDialog


def fixture_profile(k1=-0.17, *, warnings=None, views=None, rms=0.2):
    """Known, tiny numeric fixture; it does not represent any physical camera."""
    return CalibrationProfile.from_dict({
        "schema": "lc-lens-calibration-v1", "image_size": [96, 72],
        "camera_matrix": [[100., 0., 47.5], [0., 99., 35.5], [0., 0., 1.]],
        "dist_coeffs": [k1, 0.02, 0.001, -0.001, 0.0],
        "pattern": "chessboard", "pattern_size": [5, 4], "spacing_mm": 2.0,
        "rms": rms, "warnings": list(warnings or []),
        "views": list(views if views is not None else [{"path": "fixture-board.png", "status": "used"}]),
    })


class LensCameraWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lens-camera-flow-")
        self.root = Path(self.temporary.name)
        self.product = self.root / "产品_临时测试"
        self.product.mkdir()
        self.settings_path = self.product / "lens_correction_settings.json"
        self.stack = ExitStack()
        self.windows = []
        self.old_language = language_code()
        set_language(LANG_ZH, persist=False)
        self.stack.enter_context(patch("services.lens_camera_roles.resolve_correction_role", side_effect=lambda _product, role: role))
        self.warning = self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "warning", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "critical", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "information", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Discard))
        # Exercise the real constructor, but redirect its optional export folders.
        self.demo_factory = self.stack.enter_context(patch(
            "tools.lens_distortion_demo.LensDemoWindow", side_effect=lambda: LensDemoWindow(workdir=self.root),
        ))
        self.frame = np.random.default_rng(61).integers(0, 256, (72, 96, 3), np.uint8)

    def tearDown(self):
        for window in reversed(self.windows):
            window.close()
            window.deleteLater()
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.stack.close()
        set_language(self.old_language, persist=False)
        self.temporary.cleanup()

    def settings(self):
        owner = QtWidgets.QWidget()
        runtime = SimpleNamespace(_trigger_lock=threading.RLock(), _accept_trigger_jobs=True,
                                  _busy=False, _di_poller=None, _trigger_future=None)
        observed_gates = []
        refresh = Mock(side_effect=lambda: observed_gates.append(runtime._accept_trigger_jobs))
        owner.runtime_ctrl = runtime
        owner.tool_page = SimpleNamespace(_reload_lens_correction_context=refresh, current_camera_role=lambda: "cam1")
        self.windows.append(owner)
        dialog = LensCorrectionSettingsDialog(owner, str(self.product), ["cam1", "cam2"],
                                              {"cam1": "SN-FIRST", "cam2": "SN-SECOND"})
        self.windows.append(dialog)
        return dialog, runtime, refresh, observed_gates

    def open_camera(self, dialog, role="cam2"):
        dialog.table.selectRow(dialog._roles.index(role))
        dialog._open_demo()
        return dialog._demo_windows[-1]

    def preview(self, window, profile=None, *, alpha=1.0):
        window.image = self.frame.copy()
        window.original_view.set_image(window.image)
        window.alpha.setValue(alpha)
        window.set_profile(profile or fixture_profile(), "临时数值测试参数")
        self.assertIsNotNone(window.corrected)

    def assert_pending(self, window):
        self.assertIn("尚未应用", window.camera_apply_hint.text())

    def test_default_window_does_not_select_an_existing_profile(self):
        existing = self.root / "experiments/lens_calibration/profiles/older.json"
        save_profile(fixture_profile(), existing)
        window = LensDemoWindow(workdir=self.root)
        self.windows.append(window)
        self.assertIsNone(window.profile)
        self.assertIsNone(window.image)
        self.assertEqual(window.source_path, "")
        self.assertFalse(window.apply_camera_button.isEnabled())
        dialog, _, _, _ = self.settings()
        camera_window = self.open_camera(dialog)
        self.assertIsNone(camera_window.profile)
        self.assertFalse(camera_window.apply_camera_button.isEnabled())
        self.assertFalse(self.settings_path.exists())

    def test_camera_target_is_visible_and_frozen_when_settings_selection_changes(self):
        dialog, _, refresh, _ = self.settings()
        window = self.open_camera(dialog, "cam2")
        for expected in (self.product.name, "cam2", "SN-SECOND"):
            self.assertIn(expected, window.camera_context_label.text())
        self.assertIn("cam2", window.apply_camera_button.text())
        self.assertIn("cam2", window.windowTitle())
        self.assertEqual(window._camera_target, (str(self.product.resolve()), "cam2", "SN-SECOND"))
        self.preview(window)
        dialog.table.selectRow(dialog._roles.index("cam1"))
        # A pending edit in a different row must not get committed by cam2 Apply.
        first_row = dialog._roles.index("cam1")
        pending_file = self.root / "cam1-pending-calibration.json"
        save_profile(fixture_profile(k1=0.12), pending_file)
        self.assertTrue(dialog.import_profile(first_row, str(pending_file)))
        dialog.table.item(first_row, 1).setCheckState(QtCore.Qt.CheckState.Checked)
        dialog.table.item(first_row, 4).setText("SN-FIRST-UNSAVED")
        window.apply_camera_button.click()
        second = get_config(self.product, "cam2", physical=True)
        first = get_config(self.product, "cam1", physical=True)
        self.assertTrue(second.enabled)
        self.assertEqual(second.camera_serial, "SN-SECOND")
        self.assertFalse(first.enabled)
        self.assertFalse(first.profile_id)
        self.assertEqual(first.camera_serial, "")
        self.assertEqual(dialog._pending_profiles["cam1"], str(pending_file.resolve()))
        self.assertEqual(dialog.table.item(first_row, 1).checkState(), QtCore.Qt.CheckState.Checked)
        self.assertEqual(dialog.table.item(first_row, 4).text(), "SN-FIRST-UNSAVED")
        self.assertEqual(set(json.loads(self.settings_path.read_text(encoding="utf-8"))["cameras"]), {"cam2"})
        refresh.assert_called_once()

    def test_json_export_does_not_apply_or_change_camera_configuration(self):
        dialog, _, refresh, _ = self.settings()
        window = self.open_camera(dialog)
        self.preview(window)
        exported = self.root / "explicit-export.json"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(exported), "JSON (*.json)")):
            window.save_parameters()
        self.assertTrue(exported.is_file())
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8"))["schema"], "lc-lens-calibration-v1")
        self.assertFalse(self.settings_path.exists())
        refresh.assert_not_called()
        self.assert_pending(window)

    def test_direct_apply_saves_preview_alpha_and_refreshes_while_trigger_gate_is_closed(self):
        dialog, runtime, refresh, observed = self.settings()
        window = self.open_camera(dialog)
        self.preview(window, alpha=0.35)
        expected_image = window.corrected.copy()
        window.apply_to_camera()
        saved = get_config(self.product, "cam2", physical=True)
        self.assertTrue(saved.enabled)
        self.assertAlmostEqual(saved.alpha, window.alpha.value())
        self.assertAlmostEqual(saved.alpha, 0.35)
        self.assertEqual(observed, [False])
        self.assertTrue(runtime._accept_trigger_jobs)
        refresh.assert_called_once()
        self.assertIn("已保存并启用到 cam2", window.camera_apply_hint.text())
        self.assertEqual(dialog._configs["cam2"], saved)
        self.assertEqual(dialog.table.item(dialog._roles.index("cam2"), 1).checkState(), QtCore.Qt.CheckState.Checked)
        np.testing.assert_array_equal(window.corrected, expected_image)
        production = prepare_array(self.frame, self.product, "cam2", camera_serial="SN-SECOND")
        np.testing.assert_array_equal(production.image, expected_image)

    def test_active_runtime_prevents_apply_without_changing_camera_config(self):
        dialog, runtime, refresh, _ = self.settings()
        window = self.open_camera(dialog)
        self.preview(window)
        runtime._busy = True
        window.apply_to_camera()
        self.assertFalse(self.settings_path.exists())
        self.assertTrue(runtime._accept_trigger_jobs)
        refresh.assert_not_called()
        self.assertTrue(self.warning.called)
        self.assert_pending(window)

    def test_changed_serial_rejects_old_camera_window(self):
        dialog, _, refresh, _ = self.settings()
        window = self.open_camera(dialog)
        self.preview(window)
        dialog.table.item(dialog._roles.index("cam2"), 4).setText("SN-REPLACED")
        dialog.table.selectRow(dialog._roles.index("cam1"))
        window.apply_to_camera()
        self.assertFalse(self.settings_path.exists())
        refresh.assert_not_called()
        self.assertTrue(self.warning.called)
        self.assertIn("绑定信息已改变", self.warning.call_args.args[2])
        self.assert_pending(window)

    def test_simulated_and_manual_parameters_cannot_apply(self):
        dialog, _, refresh, _ = self.settings()
        window = self.open_camera(dialog)
        rejected = [
            manual_profile((96, 72), -0.2),
            fixture_profile(warnings=["synthetic calibration fixture"]),
            fixture_profile(warnings=["模拟棋盘，不适用于真实相机"]),
            fixture_profile(views=[{"path": "SIMULATED_board.png", "status": "used"}]),
            fixture_profile(rms=None),
        ]
        for profile in rejected:
            with self.subTest(pattern=profile.pattern, warnings=profile.warnings, rms=profile.rms):
                self.preview(window, profile)
                self.assertFalse(window.apply_camera_button.isEnabled())
                window.apply_to_camera()  # Calling the slot cannot bypass its guard.
                self.assertFalse(self.settings_path.exists())
        refresh.assert_not_called()

    def test_new_profile_or_alpha_requires_another_explicit_apply(self):
        dialog, _, _, _ = self.settings()
        window = self.open_camera(dialog)
        self.preview(window)
        window.apply_to_camera()
        first = get_config(self.product, "cam2", physical=True)
        window.set_profile(fixture_profile(k1=-0.11), "另一组临时测试参数")
        self.assert_pending(window)
        self.assertEqual(get_config(self.product, "cam2", physical=True).profile_id, first.profile_id)
        window.apply_to_camera()
        second = get_config(self.product, "cam2", physical=True)
        self.assertNotEqual(second.profile_id, first.profile_id)
        window.alpha.setValue(0.4)
        self.assert_pending(window)
        self.assertEqual(get_config(self.product, "cam2", physical=True), second)

    def test_reopening_same_target_reuses_window_and_preview(self):
        dialog, _, _, _ = self.settings()
        first = self.open_camera(dialog)
        profile = fixture_profile()
        self.preview(first, profile)
        dialog._open_demo()
        self.assertEqual(len(dialog._demo_windows), 1)
        self.assertIs(dialog._demo_windows[0], first)
        self.assertIs(dialog._demo_windows[0].profile, profile)
        self.assertEqual(self.demo_factory.call_count, 1)


if __name__ == "__main__":
    unittest.main()
