"""End-user state transitions for the standalone preview, without a camera."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets, QtTest
from algorithms.image_io import imread, imwrite
from algorithms.lens_calibration import load_profile, save_profile
from algorithms.lens_single_calibration import SingleCalibrationProfile
from tools.lens_distortion_demo import LensDemoWindow, load_demo_profile
from ui.i18n import LANG_EN, LANG_ZH, language_code, set_language


class LensDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.addCleanup(set_language, language_code(), persist=False)
        set_language(LANG_ZH, persist=False)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.window = LensDemoWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def test_english_camera_window_translates_controls_progress_and_failures(self):
        set_language(LANG_EN, persist=False)
        window = LensDemoWindow(workdir=self.root)
        self.addCleanup(window.close)
        window.set_camera_context(product_dir=self.root, camera_role="cam2", apply_callback=lambda *_: False)
        window.show()
        self.assertFalse(window.log.isVisible())
        self.assertFalse(window.examples_menu.menuAction().isVisible())
        self.assertEqual(window.apply_camera_button.text(), "Apply to cam2")
        self.assertEqual(window.tabs.tabText(0), "Calibration")
        for mode in range(window.mode.count()):
            window.mode.setCurrentIndex(mode)
            for widget in window.findChildren(QtWidgets.QLabel) + window.findChildren(QtWidgets.QAbstractButton):
                self.assertNotRegex(widget.text(), r"[\u4e00-\u9fff]")
            self.assertNotRegex(window.mode.currentText(), r"[\u4e00-\u9fff]")
        window.start_calibration()
        self.assertIn("10 distinct board images", window.status.text())
        window._on_progress(1, 2, "board.bmp：已检测到完整图案")
        self.assertIn("board.bmp: Complete pattern detected", window.log.toPlainText())
        window._on_calibration_failed("已识别 24 个可见格点（网格范围 6×4），但标定未通过：棋盘角点超出图像范围。")
        self.assertIn("6 × 4", window.status.text())
        self.assertIn("outside the image", window.status.text())

    def test_example_export_round_trip_and_double_correction_guard(self):
        w = self.window
        w.load_example()
        self.assertIsNotNone(w.corrected)
        self.assertFalse(np.array_equal(w.image, w.corrected))
        path = self.root / "矫正结果.png"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(path), "")):
            w.export_image()
        np.testing.assert_array_equal(imread(path), w.corrected)
        marker = json.loads(path.with_suffix(".undistortion.json").read_text(encoding="utf-8"))
        self.assertTrue(marker["already_undistorted"])
        self.assertEqual(marker["profile"]["image_size"], [1280, 800])
        before = w.image
        w.load_image_path(path)
        self.assertIs(w.image, before)
        self.assertIn("重复矫正", w.status.text())

    def test_mismatch_clears_preview_then_manual_can_recover(self):
        w = self.window
        w.load_example()
        path = self.root / "different.png"
        imwrite(path, np.zeros((240, 320, 3), np.uint8))
        w.load_image_path(path)
        self.assertIsNone(w.corrected)
        self.assertFalse(w.export_button.isEnabled())
        self.assertIn("不一致", w.status.text())
        w.apply_manual()
        self.assertEqual(w.corrected.shape, (240, 320, 3))
        self.assertIsNone(w.profile.rms)
        self.assertTrue(w.export_button.isEnabled())

    def test_profile_reload_and_alpha_invalidate_maps(self):
        w = self.window
        w.load_example()
        path = self.root / "parameters.json"
        save_profile(w.profile, path)
        expected = w.corrected.copy()
        w.set_profile(load_profile(path), "loaded")
        np.testing.assert_array_equal(w.corrected, expected)
        old = w.rectifier
        w.alpha.setValue(0)
        self.assertIsNot(w.rectifier, old)
        self.assertFalse(np.array_equal(w.corrected, expected))

    def test_benchmark_and_missing_calibration_images(self):
        w = self.window
        self.assertEqual(w.mode.currentData(), "radial")
        w.start_calibration()
        self.assertIsNone(w.worker)
        self.assertIn("一张", w.status.text())
        w.mode.setCurrentIndex(2)
        w.start_calibration()
        self.assertIn("10", w.status.text())
        w.load_example()
        w.benchmark()
        self.assertEqual(len(w.last_times), 20)
        self.assertTrue(all(ms > 0 for ms in w.last_times))
        self.assertIn("P95", w.metrics.text())

    def test_one_selected_image_calibrates_in_background_and_round_trips(self):
        w = self.window
        w.load_single_example()
        self.assertEqual(len(w.calibration_paths), 1)
        self.assertEqual(w.columns.value(), 13)
        self.assertEqual(w.rows.value(), 7)
        # The explicit complete-board route remains available and keeps v1 files.
        w.auto_grid.setChecked(False)
        # The selected file, not a stale first file, is used in single-image mode.
        w.calibration_paths.insert(0, str(self.root / "does_not_exist.png"))
        w.files.insertItem(0, "does_not_exist.png")
        w.files.setCurrentRow(1)
        w.start_calibration()
        deadline = time.monotonic() + 30
        while w.worker.isRunning() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertFalse(w.worker.isRunning(), "calibration worker timed out")
        self.app.processEvents()
        self.assertIsInstance(w.profile, SingleCalibrationProfile, w.status.text())
        self.assertEqual(len(w.profile.views), 1)
        self.assertLess(w.profile.rms, 0.5)
        self.assertFalse(w.alpha.isEnabled())
        self.assertTrue(w.save_button.isEnabled())
        self.assertIsNotNone(w.corrected)
        self.assertFalse(w.original_view.corner_item.path().isEmpty())
        parameters = self.root / "single.json"
        save_profile(w.profile, parameters)
        expected = w.corrected.copy()
        w.set_profile(load_demo_profile(parameters), "reloaded single")
        np.testing.assert_array_equal(w.corrected, expected)
        path = self.root / "单张矫正.png"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(path), "")):
            w.export_image()
        metadata = json.loads(path.with_suffix(".undistortion.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["profile"]["schema"], "lc-lens-single-v1")
        self.assertEqual(metadata["canvas_policy"], "fixed")
        self.assertEqual(metadata["alpha"], 1.0)
        # A different product image must not receive the board's corner overlay.
        product = self.root / "product.png"
        imwrite(product, np.full_like(w.image, 120))
        w.load_image_path(product)
        self.assertTrue(w.original_view.corner_item.path().isEmpty())

    def test_auto_grid_needs_no_dimensions_and_reloads_sparse_parameters(self):
        w = self.window
        w.load_single_example()
        self.assertTrue(w.auto_grid.isChecked())
        self.assertFalse(w.grid_form.isRowVisible(w.columns))
        w.columns.setValue(4)
        w.rows.setValue(5)
        w.start_calibration()
        deadline = time.monotonic() + 40
        while w.worker.isRunning() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertFalse(w.worker.isRunning(), "automatic calibration timed out")
        self.app.processEvents()
        self.assertIsInstance(w.profile, SingleCalibrationProfile, w.status.text())
        self.assertIsNotNone(w.profile.grid_indices)
        self.assertGreater(len(w.profile.points), 40)
        self.assertIn("自动识别", w.detected_grid_label.text())
        self.assertIsNotNone(w.corrected)
        path = self.root / "automatic.json"
        save_profile(w.profile, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema"], "lc-lens-single-v2")
        expected = w.corrected.copy()
        restored = load_demo_profile(path)
        np.testing.assert_array_equal(restored.homography, w.profile.homography)
        np.testing.assert_array_equal(restored.radial_coeffs, w.profile.radial_coeffs)
        np.testing.assert_array_equal(restored.grid_indices, w.profile.grid_indices)
        w.set_profile(restored, "automatic reload")
        # Interpolation at a quantization boundary may round by one intensity level.
        np.testing.assert_allclose(w.corrected, expected, rtol=0, atol=1)
        w.mode.setCurrentIndex(2)
        self.assertTrue(w.grid_form.isRowVisible(w.columns))
        self.assertFalse(w.auto_grid.isEnabled())

    def test_failed_new_calibration_cannot_reuse_an_old_result(self):
        w = self.window
        w.load_example()
        self.assertIsNotNone(w.profile)
        raw = self.root / "blank_board.png"
        imwrite(raw, np.full_like(w.image, 127))
        w.load_image_path(raw)
        w.use_current_image()
        w.start_calibration()
        self.assertIsNone(w.profile)
        self.assertIsNone(w.corrected)
        deadline = time.monotonic() + 15
        while w.worker.isRunning() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertFalse(w.worker.isRunning())
        self.app.processEvents()
        self.assertIn("失败", w.profile_label.text())
        self.assertFalse(w.save_button.isEnabled())
        self.assertFalse(w.export_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
