from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PySide6 import QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from line2dup.core.locator import product_paths
from line2dup.core.recipe import Line2DupRecipe, load_recipe, save_recipe
from line2dup.like_matcher import Feature, Line2DupLikeDetector, TemplateLevel, encode_png_base64
from line2dup.ui import template_page_pyside6 as template_page_module


class Line2DupTemplatePageStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_recipe_round_trip_preserves_template_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "line2dup" / "cam1" / "recipe.json"
            recipe = Line2DupRecipe(
                class_id="demo",
                array_count=6,
                array_pitch_x=42.5,
                array_pitch_y=-3.0,
                template_levels="3,5,7",
                template_num_features=256,
                template_weak_threshold=22.0,
                template_strong_threshold=44.0,
                template_angle_start=5.0,
                template_angle_end=25.0,
                template_angle_step=5.0,
                template_scale_start=1.1,
                template_scale_end=1.3,
                template_scale_step=0.1,
            )

            save_recipe(recipe, str(recipe_path))
            loaded = load_recipe(str(recipe_path))

            self.assertEqual(loaded.array_count, 6)
            self.assertAlmostEqual(loaded.array_pitch_x, 42.5)
            self.assertAlmostEqual(loaded.array_pitch_y, -3.0)
            self.assertEqual(loaded.template_levels, "3,5,7")
            self.assertEqual(loaded.template_num_features, 256)
            self.assertAlmostEqual(loaded.template_weak_threshold, 22.0)
            self.assertAlmostEqual(loaded.template_strong_threshold, 44.0)
            self.assertAlmostEqual(loaded.template_angle_start, 5.0)
            self.assertAlmostEqual(loaded.template_angle_end, 25.0)
            self.assertAlmostEqual(loaded.template_angle_step, 5.0)
            self.assertAlmostEqual(loaded.template_scale_start, 1.1)
            self.assertAlmostEqual(loaded.template_scale_end, 1.3)
            self.assertAlmostEqual(loaded.template_scale_step, 0.1)

    def test_loading_existing_model_restores_template_params_and_extract_button(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            ref_image = self._write_reference_image(product_dir)
            self._write_model(product_dir, ref_image)

            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                self.assertEqual(dialog.edit_levels.text(), "3,5,7")
                self.assertEqual(dialog.spin_num_features.value(), 256)
                self.assertAlmostEqual(dialog.spin_weak.value(), 22.0)
                self.assertAlmostEqual(dialog.spin_strong.value(), 44.0)
                self.assertAlmostEqual(dialog.spin_angle_start.value(), 5.0)
                self.assertAlmostEqual(dialog.spin_angle_end.value(), 25.0)
                self.assertAlmostEqual(dialog.spin_angle_step.value(), 5.0)
                self.assertAlmostEqual(dialog.spin_scale_start.value(), 1.1)
                self.assertAlmostEqual(dialog.spin_scale_end.value(), 1.3)
                self.assertAlmostEqual(dialog.spin_scale_step.value(), 0.1)
                self.assertIsNotNone(dialog.template_roi)
                self.assertTrue(dialog.btn_extract_points.isEnabled())
            finally:
                dialog.close()

    def test_loading_recipe_restores_array_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            ref_image = self._write_reference_image(product_dir)
            paths = product_paths(str(product_dir), "cam1")
            Path(paths.role_dir).mkdir(parents=True, exist_ok=True)
            save_recipe(
                Line2DupRecipe(
                    reference_image=str(ref_image),
                    class_id="demo",
                    array_count=8,
                    array_pitch_x=36.0,
                    array_pitch_y=4.5,
                    reference_regions=[
                        {
                            "reference_label": "roi1",
                            "output_label": "roi1",
                            "display_name": "roi1",
                            "shape_type": "rectangle",
                            "points": [[1, 2], [11, 12]],
                        }
                    ],
                ),
                paths.recipe_path,
            )

            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                self.assertEqual(dialog.spin_array_count.value(), 8)
                self.assertAlmostEqual(dialog.spin_array_pitch_x.value(), 36.0)
                self.assertAlmostEqual(dialog.spin_array_pitch_y.value(), 4.5)
            finally:
                dialog.close()

    def test_array_preview_validation_requires_pitch_for_array_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            ref_image = self._write_reference_image(product_dir)
            self._write_model(product_dir, ref_image)

            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                dialog.spin_array_count.setValue(7)
                dialog._reference_regions = [
                    {
                        "reference_label": "roi1",
                        "output_label": "roi1",
                        "display_name": "roi1",
                        "shape_type": "rectangle",
                        "points": [[1.0, 2.0], [11.0, 12.0]],
                    }
                ]
                recipe = dialog._recipe_from_controls(use_find_values=True)
                recipe.reference_image = str(ref_image)
                self.assertIn("pitch_x / pitch_y", dialog._array_preview_validation_error(recipe))

                dialog.spin_array_pitch_x.setValue(36.0)
                recipe_ok = dialog._recipe_from_controls(use_find_values=True)
                recipe_ok.reference_image = str(ref_image)
                self.assertEqual(dialog._array_preview_validation_error(recipe_ok), "")
            finally:
                dialog.close()

    def test_right_click_delete_still_allows_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            ref_image = self._write_reference_image(product_dir)
            self._write_model(product_dir, ref_image)

            dialog = template_page_module.Line2DupTemplateDialog(
                product_name="demo",
                product_dir=str(product_dir),
                camera_role="cam1",
            )
            try:
                dialog.chk_edit_points.setChecked(True)
                points = dialog._editor_feature_abs_points()
                self.assertGreaterEqual(len(points), 1)
                x, y = points[0]
                dialog._hover_feature_index = 0
                dialog._on_create_canvas_pressed(template_page_module._button_right(), int(round(x)), int(round(y)))
                self.assertTrue(dialog.points_dirty)

                stub_detector = Line2DupLikeDetector(
                    num_features=int(dialog.spin_num_features.value()),
                    T_levels=[int(v.strip()) for v in dialog.edit_levels.text().split(",") if v.strip()],
                    weak_threshold=float(dialog.spin_weak.value()),
                    strong_threshold=float(dialog.spin_strong.value()),
                )
                stub_detector.set_original_editor_levels(dialog.edit_class_id.text().strip() or "demo", dialog.editor_levels)

                with (
                    mock.patch.object(
                        template_page_module,
                        "build_multi_backend_detector",
                        return_value=(stub_detector, 1, 0),
                    ) as build_detector,
                    mock.patch.object(template_page_module, "save_detector_model") as save_model,
                    mock.patch("PySide6.QtWidgets.QMessageBox.information"),
                ):
                    dialog._build_and_save()

                self.assertEqual(build_detector.call_args.kwargs["original_mode"], "manual_points")
                save_model.assert_called_once()
                self.assertFalse(dialog.points_dirty)
            finally:
                dialog.close()

    def _write_reference_image(self, product_dir: Path) -> Path:
        ref_image = product_dir / "debug_capture" / "cam1_ref.png"
        ref_image.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[4:20, 8:24] = (255, 255, 255)
        cv2.imwrite(str(ref_image), image)
        return ref_image

    def _write_model(self, product_dir: Path, ref_image: Path) -> None:
        paths = product_paths(str(product_dir), "cam1")
        Path(paths.role_dir).mkdir(parents=True, exist_ok=True)
        roi = np.zeros((8, 10, 3), dtype=np.uint8)
        roi[2:6, 3:7] = (255, 255, 255)
        mask = np.ones((8, 10), dtype=np.uint8) * 255
        level0 = self._level_to_dict(
            TemplateLevel(
                width=9,
                height=7,
                tl_x=0,
                tl_y=0,
                pyramid_level=0,
                features=[
                    Feature(x=2, y=2, label=0, theta=0.0),
                    Feature(x=6, y=4, label=1, theta=45.0),
                ],
            )
        )
        model_dict = {
            "format": "line2dup_like_model_v2",
            "params": {
                "num_features": 256,
                "T_levels": [3, 5, 7],
                "weak_threshold": 22.0,
                "strong_threshold": 44.0,
            },
            "classes": {
                "demo": {
                    "source": {
                        "image_path": str(ref_image),
                        "roi_png": encode_png_base64(roi),
                        "mask_png": encode_png_base64(mask),
                        "roi_x": 3,
                        "roi_y": 4,
                        "roi_w": 10,
                        "roi_h": 8,
                        "mask_rects": [],
                    },
                    "pose_infos": {
                        "items": [{"angle": 0.0, "scale": 1.0}],
                        "ui": {
                            "angle_start": 5.0,
                            "angle_end": 25.0,
                            "angle_step": 5.0,
                            "scale_start": 1.1,
                            "scale_end": 1.3,
                            "scale_step": 0.1,
                        },
                    },
                    "original_mode": "manual_points",
                    "meta": [{"angle": 0.0, "scale": 1.0}],
                    "backends": {
                        "original": [{"template_id": 0, "levels": [level0]}],
                        "fusion": [],
                        "fusionv2": [],
                        "sim3": [],
                    },
                    "original_editor_levels": [level0],
                }
            },
        }
        Path(paths.model_path).write_text(json.dumps(model_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _level_to_dict(level: TemplateLevel) -> dict:
        return {
            "width": int(level.width),
            "height": int(level.height),
            "tl_x": int(level.tl_x),
            "tl_y": int(level.tl_y),
            "pyramid_level": int(level.pyramid_level),
            "features": [
                {
                    "x": int(feature.x),
                    "y": int(feature.y),
                    "label": int(feature.label),
                    "theta": float(feature.theta),
                }
                for feature in level.features
            ],
        }


if __name__ == "__main__":
    unittest.main()
