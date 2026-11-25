"""Demo workspace paths preserve saved products and use independent output roots."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from algorithms.image_io import imread, imwrite
from shape.core.recipe import ShapeRecipe, load_recipe, save_recipe
from tools import lens_distortion_demo as lens
from tools import roi_group_demo as groups


class DemoWorkspacePathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="demo-workspace-")
        self.root = Path(self.temporary.name)
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def product(self, path, *, name="black-user-group"):
        role = path / "shape/cam1"
        role.mkdir(parents=True)
        samples = path / "samples/train"
        samples.mkdir(parents=True)
        reference = samples / "reference.png"
        self.assertTrue(imwrite(reference, np.full((24, 36, 3), 73, np.uint8)))
        model = role / "model.json"
        model.write_text(json.dumps({"classes": {"fixture": {"source": {"image_path": str(reference)}}}}), encoding="utf-8")
        save_recipe(ShapeRecipe(
            model_path=str(model), reference_image=str(reference),
            localization_groups=[{"id": "keep-me", "name": name, "points": [[2, 3], [13, 15]]}],
        ), str(role / "recipe.json"))
        return role / "recipe.json"

    def run_prepare(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return groups.main(["--workdir", str(self.root), "--prepare-only", *extra])

    def test_existing_new_copy_opens_without_a_source_and_never_overwrites_user_edits(self):
        product = self.root / "experiments/localization_groups/product_1125"
        recipe = self.product(product)
        before = recipe.read_bytes()
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", side_effect=AssertionError("must not ask for source")):
            self.assertEqual(self.run_prepare(), 0)
            self.assertEqual(self.run_prepare("--source-product", str(self.root / "missing-old-dist")), 0)
        self.assertEqual(recipe.read_bytes(), before)
        self.assertEqual(load_recipe(str(recipe)).localization_groups[0]["name"], "black-user-group")

    def test_legacy_copy_is_reused_until_new_copy_exists_and_explicit_path_wins(self):
        legacy = self.root / "reports/localization_groups_demo/product_1125"
        recipe = self.product(legacy)
        original = recipe.read_bytes()
        self.assertEqual(groups.default_demo_product(self.root), legacy)
        self.assertEqual(self.run_prepare(), 0)
        self.assertFalse((self.root / "experiments").exists())
        self.assertEqual(recipe.read_bytes(), original)
        new = self.root / "experiments/localization_groups/product_1125"
        self.product(new, name="new-copy")
        self.assertEqual(groups.default_demo_product(self.root), new)
        self.assertEqual(self.run_prepare("--product-dir", str(legacy)), 0)
        self.assertEqual(recipe.read_bytes(), original)

    def test_first_copy_requires_an_explicit_source_and_then_survives_source_removal(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.run_prepare()
        self.assertEqual(error.exception.code, 2)
        source = self.root / "source-product"
        source_recipe = self.product(source)
        before = source_recipe.read_bytes()
        self.assertEqual(self.run_prepare("--source-product", str(source)), 0)
        destination = groups.default_demo_product(self.root)
        recipe = load_recipe(str(destination / "shape/cam1/recipe.json"))
        self.assertEqual(Path(recipe.reference_image), destination / "samples/train/reference.png")
        self.assertEqual(Path(recipe.model_path), destination / "shape/cam1/model.json")
        self.assertEqual(source_recipe.read_bytes(), before)
        source.rename(self.root / "source-no-longer-available")
        self.assertEqual(self.run_prepare(), 0)
        self.assertEqual(recipe.localization_groups[0]["name"], "black-user-group")

    def test_cancelled_source_picker_creates_no_copy_and_failures_use_report_logs(self):
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value="") as picker:
            self.assertEqual(groups.main(["--workdir", str(self.root)]), 0)
        self.assertEqual(picker.call_count, 1)
        self.assertFalse((self.root / "experiments").exists())
        with self.assertRaises(RuntimeError):
            self.run_prepare("--source-product", str(self.root / "absent"))
        log = self.root / "artifacts/reports/localization_groups_demo/startup_error.txt"
        self.assertTrue(log.is_file())
        self.assertIn("absent", log.read_text(encoding="utf-8"))

    def test_existing_incomplete_destination_is_not_replaced_or_bypassed(self):
        destination = self.root / "experiments/localization_groups/product_1125"
        destination.mkdir(parents=True)
        sentinel = destination / "do-not-overwrite.txt"
        sentinel.write_text("user data", encoding="utf-8")
        self.product(self.root / "reports/localization_groups_demo/product_1125")
        self.assertEqual(groups.default_demo_product(self.root), destination)
        with self.assertRaises(RuntimeError):
            self.run_prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")

    def test_lens_default_dialog_paths_export_pixels_and_legacy_profile_loading(self):
        window = lens.LensDemoWindow(workdir=self.root)
        self.windows.append(window)
        window.load_example()
        original_pixels = window.corrected.copy()
        profile_path = self.root / "experiments/lens_calibration/profiles/lens_calibration.json"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(profile_path), "")) as save:
            window.save_parameters()
        self.assertEqual(Path(save.call_args.args[2]), profile_path)
        self.assertTrue(profile_path.is_file())
        export = self.root / "artifacts/reports/lens_distortion_demo/synthetic_undistorted.png"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(export), "")) as save:
            window.export_image()
        self.assertEqual(Path(save.call_args.args[2]), export)
        np.testing.assert_array_equal(imread(export), original_pixels)
        self.assertTrue(json.loads(export.with_suffix(".undistortion.json").read_text(encoding="utf-8"))["already_undistorted"])
        legacy = self.root / "old_saved_calibration.json"
        legacy.write_bytes(profile_path.read_bytes())
        with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(legacy), "")) as load:
            window.load_parameters()
        self.assertEqual(Path(load.call_args.args[2]), profile_path.parent)
        np.testing.assert_array_equal(window.corrected, original_pixels)

    def test_lens_cli_retains_options_and_startup_failure_uses_selected_workspace(self):
        with patch.object(lens, "_run_window", return_value=0) as run:
            self.assertEqual(lens.main(["--workdir", str(self.root), "--image", "raw.bmp", "--profile", "old.json",
                                       "--example", "--single-example"]), 0)
        args = run.call_args.args[0]
        self.assertEqual((args.image, args.profile, args.example, args.single_example), ("raw.bmp", "old.json", True, True))
        self.assertEqual(args.workdir, self.root)
        with patch.object(lens, "_run_window", side_effect=RuntimeError("fixture startup failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture startup failure"):
                lens.main(["--workdir", str(self.root)])
        log = self.root / "artifacts/reports/lens_distortion_demo/startup_error.txt"
        self.assertIn("fixture startup failure", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
