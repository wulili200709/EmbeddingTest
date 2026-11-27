"""Corrected editor images, explicit configuration and stale-window protection."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6 import QtCore, QtWidgets

from algorithms.image_io import imread, imwrite
from algorithms.lens_calibration import manual_profile, save_profile
from services.lens_correction import active_signature, get_config, prepare_path, save_config
from ui.lens_correction_dialog import LensCorrectionSettingsDialog, prepare_editor_image
from ui.shape_template.template_page_pyside6 import ShapeTemplateDialog


class LensCorrectionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lens-ui-")
        self.root = Path(self.temp.name)
        self.product = self.root / "product"
        self.product.mkdir()
        self.raw_path = self.root / "cam1_raw.png"
        self.raw = np.random.default_rng(4).integers(0, 255, (96, 128, 3), dtype=np.uint8)
        self.assertTrue(imwrite(self.raw_path, self.raw))
        self.profile_path = self.root / "calibration.json"
        save_profile(manual_profile((128, 96), -.2), self.profile_path)
        self.stack = ExitStack()
        self.stack.enter_context(patch("services.lens_camera_roles.resolve_correction_role", side_effect=lambda product, role: role))
        self.warning = self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "warning", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "critical", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "information", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Discard))
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.stack.close()
        self.temp.cleanup()

    def settings(self):
        window = LensCorrectionSettingsDialog(None, str(self.product), ["cam1", "cam2"])
        self.windows.append(window)
        return window

    def enable(self):
        return save_config(self.product, "cam1", enabled=True, profile_path=self.profile_path)

    def shape(self):
        window = ShapeTemplateDialog(product_name="product", product_dir=str(self.product), initial_image_path=str(self.raw_path))
        self.windows.append(window)
        return window

    def test_settings_are_off_and_do_not_load_existing_external_profile(self):
        window = self.settings()
        self.assertFalse(get_config(self.product, "cam1").enabled)
        self.assertEqual(window.table.item(0, 1).checkState(), QtCore.Qt.CheckState.Unchecked)
        self.assertFalse(window._configs["cam1"].profile_id)
        self.assertFalse((self.product / "lens_correction_settings.json").exists())
        window.reject()
        self.assertFalse((self.product / "lens_correction_settings.json").exists())

    def test_import_is_staged_until_explicit_save_and_only_selected_camera_enables(self):
        window = self.settings()
        self.assertTrue(window.import_profile(0, str(self.profile_path)))
        self.assertFalse((self.product / "lens_correction_settings.json").exists())
        self.assertEqual(window.table.item(0, 1).checkState(), QtCore.Qt.CheckState.Unchecked)
        saved = []
        window.saved.connect(lambda product, roles: saved.append((product, roles)))
        window.table.item(0, 1).setCheckState(QtCore.Qt.CheckState.Checked)
        window._save()
        self.assertTrue(window.changed)
        self.assertTrue(get_config(self.product, "cam1").enabled)
        self.assertFalse(get_config(self.product, "cam2").enabled)
        self.assertEqual(saved, [(str(self.product), ["cam1"])])

    def test_enable_without_profile_cannot_save(self):
        window = self.settings()
        window.table.item(1, 1).setCheckState(QtCore.Qt.CheckState.Checked)
        window._save()
        self.assertFalse(window.changed)
        self.assertFalse((self.product / "lens_correction_settings.json").exists())
        self.assertTrue(self.warning.called)

    def test_invalid_import_preserves_current_selection(self):
        window = self.settings()
        bad = self.root / "invalid.json"
        bad.write_text("{}", encoding="utf-8")
        self.assertFalse(window.import_profile(0, str(bad)))
        self.assertFalse(window._pending_profiles)
        self.assertFalse((self.product / "lens_correction_settings.json").exists())

    def _settings_with_runtime(self, reload_callback):
        owner = QtWidgets.QWidget()
        runtime = SimpleNamespace(_trigger_lock=threading.RLock(), _accept_trigger_jobs=True,
                                  _busy=False, _di_poller=None, _trigger_future=None)
        owner.runtime_ctrl = runtime
        owner.tool_page = SimpleNamespace(_reload_lens_correction_context=lambda: reload_callback(runtime))
        self.windows.append(owner)
        dialog = LensCorrectionSettingsDialog(owner, str(self.product), ["cam1"])
        self.windows.insert(0, dialog)
        dialog.import_profile(0, str(self.profile_path))
        dialog.table.item(0, 1).setCheckState(QtCore.Qt.CheckState.Checked)
        return dialog, runtime

    def test_save_keeps_trigger_gate_closed_until_context_reload_finishes(self):
        observed = []
        dialog, runtime = self._settings_with_runtime(lambda rt: observed.append(rt._accept_trigger_jobs))
        dialog._save()
        self.assertEqual(observed, [False])
        self.assertTrue(runtime._accept_trigger_jobs)
        self.assertTrue(dialog.changed)

    def test_failed_context_reload_keeps_runtime_triggering_disabled(self):
        def fail(runtime):
            raise RuntimeError("fixture context failure")
        dialog, runtime = self._settings_with_runtime(fail)
        dialog._save()
        self.assertTrue(get_config(self.product, "cam1").enabled)
        self.assertFalse(runtime._accept_trigger_jobs)
        self.assertTrue(self.warning.called)

    def test_trigger_started_after_initial_guard_prevents_settings_commit(self):
        dialog, runtime = self._settings_with_runtime(lambda rt: None)
        def first_guard(owner):
            runtime._busy = True
            return True
        with patch("ui.lens_correction_dialog._can_change_settings", side_effect=first_guard):
            dialog._save()
        self.assertFalse((self.product / "lens_correction_settings.json").exists())
        self.assertTrue(runtime._accept_trigger_jobs)

    def test_lighting_channels_share_physical_rows_without_double_mapping(self):
        mapping = {"cam1": "cam2", "cam2": "cam1", "cam3": "cam2"}
        with patch("services.lens_camera_roles.resolve_correction_role", side_effect=lambda product, role: mapping[role]):
            window = LensCorrectionSettingsDialog(None, str(self.product), ["cam1", "cam2", "cam3"])
            self.windows.append(window)
            self.assertEqual(window._roles, ["cam2", "cam1"])
            self.assertTrue(window.import_profile(0, str(self.profile_path)))
            window.table.item(0, 1).setCheckState(QtCore.Qt.CheckState.Checked)
            window._save()
            self.assertTrue(get_config(self.product, "cam2", physical=True).enabled)
            self.assertFalse(get_config(self.product, "cam1", physical=True).enabled)

    def test_corrected_shape_rejects_original_space_model(self):
        self.enable()
        window = self.shape()
        self.assertFalse(window._model_matches_lens_context(str(self.product / "shape/cam1/model.json")))
        self.assertTrue(window._model_matches_lens_context(window.paths.model_path))

    def test_shape_disabled_uses_original_pixels_and_path(self):
        window = self.shape()
        self.assertEqual(window.image_path, str(self.raw_path))
        np.testing.assert_array_equal(window.image_bgr, self.raw)

    def test_managed_image_browsed_with_correction_off_returns_to_raw(self):
        self.enable()
        corrected = prepare_path(self.raw_path, self.product, "cam1")
        save_config(self.product, "cam1", enabled=False)
        self.assertEqual(prepare_editor_image(corrected, str(self.product), "cam1"), str(self.raw_path))
        window = self.shape()
        window._set_image(corrected, reset_state=True)
        self.assertEqual(window.image_path, str(self.raw_path))
        np.testing.assert_array_equal(window.image_bgr, self.raw)

    def test_managed_image_without_raw_is_rejected_when_correction_off(self):
        self.enable()
        corrected = prepare_path(self.raw_path, self.product, "cam1")
        save_config(self.product, "cam1", enabled=False)
        self.raw_path.unlink()
        with self.assertRaises(ValueError):
            prepare_editor_image(corrected, str(self.product), "cam1")

    def test_lite_halcon_crop_allowed_when_off_and_rejected_when_on(self):
        from MainLite import LiteDebugMainWindow
        owner = SimpleNamespace(session=SimpleNamespace(product_dir=str(self.product)),
                                _lite_crop_context=lambda path: {"camera_role": "cam1", "roi_label": "roi1"})
        LiteDebugMainWindow._validate_lens_crop_input(owner, str(self.raw_path), "cam1")
        self.enable()
        with self.assertRaisesRegex(RuntimeError, "HALCON"):
            LiteDebugMainWindow._validate_lens_crop_input(owner, str(self.raw_path), "cam1")

    def test_shape_corrected_array_and_all_authoring_canvases_agree(self):
        self.enable()
        window = self.shape()
        expected_path = prepare_path(self.raw_path, self.product, "cam1")
        self.assertEqual(window.image_path, expected_path)
        np.testing.assert_array_equal(window.image_bgr, imread(expected_path))
        self.assertFalse(np.array_equal(window.image_bgr, self.raw))
        for canvas in (window.create_canvas, window.ref_canvas, window.group_canvas):
            self.assertEqual(canvas.image_path(), expected_path)

    def test_shape_stale_window_cannot_save_or_build(self):
        window = self.shape()
        self.enable()
        with patch("ui.shape_template.template_page_pyside6.save_recipe") as save, patch.object(window, "_start_build_worker") as build:
            window._save_recipe()
            window._build_and_save()
        save.assert_not_called()
        build.assert_not_called()
        self.assertTrue(self.warning.called)

    def test_shape_find_receives_corrected_pixels(self):
        self.enable()
        window = self.shape()
        Path(window.paths.model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(window.paths.model_path).write_text("{}", encoding="utf-8")
        item = QtWidgets.QListWidgetItem(self.raw_path.name)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, str(self.raw_path))
        window.list_find_images.addItem(item)
        with patch.object(window, "_get_detector_for_model", return_value=object()), patch("ui.shape_template.template_page_pyside6.locate_and_follow", side_effect=RuntimeError("fixture-stop")) as locate:
            window._run_find_for_item(item)
        self.assertEqual(locate.call_count, 1)
        expected = imread(prepare_path(self.raw_path, self.product, "cam1"))
        np.testing.assert_array_equal(locate.call_args.args[0], expected)

    def test_ncc_worker_receives_corrected_pixels(self):
        from ncc.model import create_default_model
        from ncc.ui.workbench_dialog import _NccFindWorker
        self.enable()
        model = create_default_model()
        worker = _NccFindWorker(model_path="fixture.json", model=model, scene_paths=[str(self.raw_path)],
                                options=model.options, search_roi=None, product_dir=str(self.product),
                                camera_role="cam1", lens_signature=active_signature(self.product, "cam1"))
        compiled = MagicMock()
        with patch("ncc.ui.workbench_dialog.NccCompiledModel", return_value=compiled):
            worker.run()
        self.assertEqual(compiled.match.call_count, 1)
        np.testing.assert_array_equal(compiled.match.call_args.args[0], imread(prepare_path(self.raw_path, self.product, "cam1")))

    def test_ncc_stale_window_cannot_write_model(self):
        from ncc.ui.workbench_dialog import NccMatchWorkbenchDialog
        window = NccMatchWorkbenchDialog(product_name="product", product_dir=str(self.product), initial_image_path=str(self.raw_path))
        self.windows.append(window)
        self.enable()
        with patch("ncc.ui.workbench_dialog.save_model") as save:
            window._save_model_metadata()
        save.assert_not_called()
        self.assertTrue(self.warning.called)

    def test_ncc_saved_source_keeps_correction_identity(self):
        from ncc.authoring import source_image_path
        from ncc.ui.workbench_dialog import NccMatchWorkbenchDialog
        self.enable()
        window = NccMatchWorkbenchDialog(product_name="product", product_dir=str(self.product), initial_image_path=str(self.raw_path))
        self.windows.append(window)
        window.source_canvas.set_roi_rect((20, 20, 30, 25), emit_signal=False)
        window._save_template(show_progress=False)
        copied = source_image_path(window._model_path, window._model)
        self.assertTrue(Path(copied).with_suffix(".undistortion.json").exists())
        self.assertEqual(prepare_path(copied, self.product, "cam1"), copied)
        np.testing.assert_array_equal(imread(copied), imread(prepare_path(self.raw_path, self.product, "cam1")))


if __name__ == "__main__":
    unittest.main()
