"""User flows for positioning groups using real Qt widgets and temporary files."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets

from algorithms.image_io import imwrite
from shape.core.localization_groups import LocalizationGroupError
from shape.core.recipe import has_bound_localization_groups, load_recipe
from shape.core.roi_follow import FollowRegion, FollowResult
from shape.like_matcher import Match
from ui.i18n import LANG_ZH, language_code, set_language
from ui.shape_template import template_page_pyside6 as page


class LocalizationGroupsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.previous_language = language_code()
        set_language(LANG_ZH, persist=False)
        self.temp = tempfile.TemporaryDirectory(prefix="local-groups-ui-")
        self.root = Path(self.temp.name)
        self.product = self.root / "产品1125"
        self.reference = self.root / "参考图.png"
        self.second_reference = self.root / "新参考图.png"
        self.scene = self.root / "测试原图.png"
        self.other_scene = self.root / "另一张测试图.png"
        raster = np.random.default_rng(23).integers(20, 235, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(raster, (30, 35), (95, 95), (240, 240, 240), 3)
        for index, path in enumerate((self.reference, self.second_reference, self.scene, self.other_scene)):
            self.assertTrue(imwrite(path, np.roll(raster, index * 3, axis=1)))
        self.stack = ExitStack()
        self.warnings = self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "warning", return_value=QtWidgets.QMessageBox.Ok))
        self.criticals = self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "critical", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "information", return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Discard))
        self.dialogs = []
        self.dialog = self._open_dialog(initial_image=str(self.reference))

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
            dialog.deleteLater()
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.stack.close()
        self.temp.cleanup()
        set_language(self.previous_language, persist=False)

    def _open_dialog(self, initial_image=""):
        dialog = page.ShapeTemplateDialog(
            product_name="产品1125", product_dir=str(self.product), initial_image_path=initial_image,
        )
        self.dialogs.append(dialog)
        dialog.show()
        self.app.processEvents()
        return dialog

    def _save(self, dialog=None):
        dialog = dialog or self.dialog
        dialog.btn_save_groups.click()
        self.app.processEvents()
        self.assertFalse(self.warnings.called, self.warnings.call_args)
        self.assertFalse(self.criticals.called, self.criticals.call_args)
        return load_recipe(dialog.paths.recipe_path)

    def _create_group_and_bind_roi(self, name="左侧壳体定位"):
        dialog = self.dialog
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.btn_add_group.click()
        self.app.processEvents()
        self.assertEqual(dialog.list_localization_groups.count(), 1)
        group_id = dialog.list_localization_groups.item(0).data(QtCore.Qt.UserRole)
        dialog.group_canvas.set_roi_rect((30, 35, 66, 61))
        dialog.edit_group_name.setText(name)
        QtTest.QTest.keyClick(dialog.edit_group_name, QtCore.Qt.Key_Return)
        dialog.spin_group_radius.setValue(32)
        self.app.processEvents()
        self.assertIn(name, dialog.list_localization_groups.item(0).text())

        dialog.tabs.setCurrentWidget(dialog.tab_reference)
        dialog.btn_add_reference_roi.click()
        dialog.ref_canvas.set_roi_rect((140, 90, 40, 35))
        self.app.processEvents()
        self.assertEqual(dialog.table_reference_regions.rowCount(), 1)
        dialog.table_reference_regions.setCurrentCell(0, 0)
        self.app.processEvents()
        index = dialog.cmb_region_group.findData(group_id)
        self.assertGreater(index, 0)
        self.assertTrue(dialog.cmb_region_group.isEnabled())
        dialog.cmb_region_group.setCurrentIndex(index)
        self.app.processEvents()
        self.assertEqual(dialog.cmb_region_group.currentData(), group_id)
        return group_id

    def _prepare_find(self):
        dialog = self.dialog
        Path(dialog.paths.model_path).write_text("{}", encoding="utf-8")
        # Only the heavyweight/native model and global matcher are substituted;
        # image IO, widgets, recipe creation, persistence and result UI stay real.
        self.stack.enter_context(patch.object(dialog, "_get_detector_for_model", return_value=object()))
        self.stack.enter_context(patch.object(page, "_draw_match_overlay", side_effect=lambda _det, image, _match: image.copy()))
        dialog.tabs.setCurrentWidget(dialog.tab_find)
        with patch.object(QtWidgets.QFileDialog, "getOpenFileNames", return_value=([str(self.scene), str(self.other_scene)], "Images")):
            dialog.btn_add_find_images.click()
        dialog.list_find_images.setCurrentRow(0)
        self.app.processEvents()
        return dialog.list_find_images.item(0)

    def _result(self, group_id, *, valid=True, grouped=True):
        points = [(142., 91.), (182., 91.), (182., 126.), (142., 126.)]
        group_results = []
        if grouped:
            group_results = [{
                "id": group_id, "name": "左侧壳体定位", "used": True, "valid": valid,
                "score": 0.92 if valid else 0.56, "peak_margin": 0.15 if valid else 0.01,
                "delta_ref": [2., 1.], "delta_scene": [2., 1.],
                "reason": "" if valid else "候选位置存在歧义",
            }]
        region = FollowRegion("roi1", points, (142, 91, 40, 35), "polygon", group_id if grouped else "", valid)
        return FollowResult(
            Match(30, 35, 88., "fixture", 0), [region], points, region.bbox, "polygon", group_results,
        )

    def test_draw_rename_bind_save_and_reopen_preserves_the_user_configuration(self):
        group_id = self._create_group_and_bind_roi()
        dialog = self.dialog
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.list_localization_groups.setCurrentRow(0)
        dialog.edit_group_name.setText("上排稳定壳体")
        QtTest.QTest.keyClick(dialog.edit_group_name, QtCore.Qt.Key_Return)
        self.app.processEvents()
        dialog.tabs.setCurrentWidget(dialog.tab_reference)
        self.assertEqual(dialog.cmb_region_group.currentData(), group_id)
        self.assertEqual(dialog.cmb_region_group.currentText(), "上排稳定壳体")
        saved = self._save()
        self.assertEqual(saved.localization_groups[0]["points"], [[30., 35.], [96., 96.]])
        self.assertEqual(saved.localization_groups[0]["name"], "上排稳定壳体")
        self.assertEqual(saved.localization_groups[0]["search_radius"], 32)
        self.assertEqual(saved.reference_regions[0]["group_id"], group_id)
        self.assertEqual(saved.reference_regions[0]["points"], [[140., 90.], [180., 125.]])

        dialog.close()
        reopened = self._open_dialog()
        reopened.tabs.setCurrentWidget(reopened.tab_reference)
        reopened.table_reference_regions.setCurrentCell(0, 0)
        self.app.processEvents()
        self.assertEqual(reopened.cmb_region_group.currentData(), group_id)
        self.assertEqual(reopened.cmb_region_group.currentText(), "上排稳定壳体")
        self.assertIn("上排稳定壳体", reopened.list_localization_groups.item(0).text())

    def test_cancel_bound_group_deletion_preserves_bindings(self):
        group_id = self._create_group_and_bind_roi()
        dialog = self.dialog
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.list_localization_groups.setCurrentRow(0)
        with patch.object(dialog, "_confirm_remove_bound_group", return_value=False) as confirm:
            dialog.btn_delete_group.click()
        confirm.assert_called_once_with("左侧壳体定位", ["roi1"])
        self.app.processEvents()
        self.assertEqual(dialog.list_localization_groups.count(), 1)
        self.assertIn("roi1", dialog.lbl_group_status.text())
        self.assertEqual(self._save().reference_regions[0]["group_id"], group_id)

        dialog.tabs.setCurrentWidget(dialog.tab_reference)
        dialog.table_reference_regions.setCurrentCell(0, 0)
        dialog.cmb_region_group.setCurrentIndex(dialog.cmb_region_group.findData(""))
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.list_localization_groups.setCurrentRow(0)
        dialog.btn_delete_group.click()
        self.app.processEvents()
        self.assertEqual(dialog.list_localization_groups.count(), 0)
        saved = self._save()
        self.assertEqual(saved.localization_groups, [])
        self.assertEqual(saved.reference_regions[0]["group_id"], "")

    def test_confirmed_bound_group_deletion_keeps_rois_and_unrelated_groups_after_reopen(self):
        group_id = self._create_group_and_bind_roi()
        dialog = self.dialog
        original_points = copy.deepcopy(dialog._reference_regions[0]["points"])
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.btn_add_group.click()
        other_id = dialog._localization_groups[1]["id"]
        dialog.tabs.setCurrentWidget(dialog.tab_reference)
        dialog.btn_add_reference_roi.click()
        dialog.ref_canvas.set_roi_rect((200, 120, 20, 30))
        dialog.table_reference_regions.setCurrentCell(1, 0)
        dialog.cmb_region_group.setCurrentIndex(dialog.cmb_region_group.findData(other_id))
        self._save()
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.list_localization_groups.setCurrentRow(0)

        def confirm_actual_dialog():
            box = self.app.activeModalWidget()
            self.assertIsInstance(box, QtWidgets.QMessageBox)
            self.assertIn("roi1", box.text())
            self.assertIn("检测框本身保留", box.informativeText())
            self.assertEqual(box.defaultButton().text(), "取消")
            button = next(button for button in box.buttons() if "ROI 改为全局定位" in button.text())
            button.click()

        QtCore.QTimer.singleShot(0, confirm_actual_dialog)
        dialog.btn_delete_group.click()
        self.app.processEvents()
        self.assertEqual(dialog.table_reference_regions.rowCount(), 2)
        self.assertNotIn(group_id, [g["id"] for g in dialog._localization_groups])
        self.assertEqual(dialog._reference_regions[0]["points"], original_points)
        self.assertEqual(dialog._reference_regions[0]["group_id"], "")
        self.assertEqual(dialog._reference_regions[1]["group_id"], other_id)
        self.assertIn("全局定位", dialog.table_reference_regions.item(0, 4).text())
        self.assertIn("检测 ROI 保留", dialog.lbl_group_status.text())
        self.assertTrue(dialog._reference_dirty)
        self._save()
        dialog.close()
        reopened = self._open_dialog()
        self.assertEqual([g["id"] for g in reopened._localization_groups], [other_id])
        self.assertEqual(reopened._reference_regions[0]["points"], original_points)
        self.assertEqual(reopened._reference_regions[0]["group_id"], "")
        self.assertEqual(reopened._reference_regions[1]["group_id"], other_id)

    def test_changing_reference_invalidates_bound_anchors_until_redrawn(self):
        group_id = self._create_group_and_bind_roi()
        self._save()
        dialog = self.dialog
        dialog.tabs.setCurrentWidget(dialog.tab_create)
        with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(self.second_reference), "Images")):
            dialog.btn_open_image.click()
        self.app.processEvents()
        saved = self._save()
        self.assertEqual(Path(saved.reference_image), self.second_reference)
        self.assertTrue(saved.localization_groups[0]["needs_rebuild"])
        self.assertEqual(saved.reference_regions[0]["group_id"], group_id)

        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.list_localization_groups.setCurrentRow(0)
        dialog.btn_redraw_group.click()
        dialog.group_canvas.set_roi_rect((45, 38, 70, 55))
        self.app.processEvents()
        rebuilt = self._save()
        self.assertFalse(rebuilt.localization_groups[0]["needs_rebuild"])
        self.assertEqual(rebuilt.localization_groups[0]["points"], [[45., 38.], [115., 93.]])
        self.assertEqual(rebuilt.reference_regions[0]["group_id"], group_id)

    def test_find_comparison_switch_changes_only_the_run_and_clears_displayed_results(self):
        group_id = self._create_group_and_bind_roi()
        self._save()
        dialog = self.dialog
        item = self._prepare_find()
        persisted_before = Path(dialog.paths.recipe_path).read_bytes()
        observed_recipes = []

        def locate(_image, _reference, recipe, **_kwargs):
            observed_recipes.append(copy.deepcopy(recipe))
            return self._result(group_id, grouped=has_bound_localization_groups(recipe))

        with patch.object(page, "locate_and_follow", side_effect=locate):
            dialog.btn_run_find_selected.click()
            self.assertEqual(dialog.table_group_results.rowCount(), 1)
            self.assertIn("groups=1", item.text())
            dialog.chk_enable_groups.setChecked(False)
            self.app.processEvents()
            self.assertEqual(dialog.table_group_results.rowCount(), 0)
            self.assertNotIn("groups=1", item.text())
            dialog.btn_run_find_selected.click()
            self.assertIn("仅全局定位", item.text())
            self.assertFalse(has_bound_localization_groups(observed_recipes[-1]))
            self.assertEqual(dialog.cmb_region_group.currentData(), group_id)
            self.assertEqual(Path(dialog.paths.recipe_path).read_bytes(), persisted_before)
            dialog.chk_enable_groups.setChecked(True)
            dialog.btn_run_find_selected.click()
            self.assertTrue(has_bound_localization_groups(observed_recipes[-1]))
            self.assertEqual(dialog.table_group_results.item(0, 1).text(), "成功")
        self.assertEqual(load_recipe(dialog.paths.recipe_path).reference_regions[0]["group_id"], group_id)

    def test_group_failure_replaces_previous_success_and_does_not_retry_as_global(self):
        group_id = self._create_group_and_bind_roi()
        self._save()
        dialog = self.dialog
        item = self._prepare_find()
        with patch.object(page, "locate_and_follow", return_value=self._result(group_id)):
            dialog.btn_run_find_selected.click()
        self.assertEqual(dialog.table_group_results.item(0, 1).text(), "成功")
        dialog.chk_auto_threshold_sweep.setChecked(True)
        failed = self._result(group_id, valid=False)
        error = LocalizationGroupError(failed.group_results, preview_result=failed)
        with patch.object(page, "locate_and_follow", side_effect=error) as locate:
            dialog.btn_run_find_selected.click()
        self.assertEqual(locate.call_count, 1, "a failed positioning group must not retry or fall back to global ROIs")
        self.assertIn("ERROR", item.text())
        self.assertIn("候选位置存在歧义", dialog.lbl_find_status.text())
        self.assertEqual(dialog.table_group_results.item(0, 1).text(), "失败")
        # Selecting a different image and returning must restore the failure,
        # rather than resurrecting the previous successful result from cache.
        dialog.list_find_images.setCurrentRow(1)
        self.app.processEvents()
        self.assertEqual(dialog.table_group_results.rowCount(), 0)
        dialog.list_find_images.setCurrentRow(0)
        self.app.processEvents()
        self.assertIn("候选位置存在歧义", dialog.lbl_find_status.text())
        self.assertEqual(dialog.table_group_results.item(0, 1).text(), "失败")
        self.assertEqual(load_recipe(dialog.paths.recipe_path).reference_regions[0]["group_id"], group_id)

    def test_dangling_binding_stays_visible_after_reopening_and_adding_another_group(self):
        group_id = self._create_group_and_bind_roi()
        self._save()
        destination = Path(self.dialog.paths.recipe_path)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        payload["localization_groups"] = []
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.dialog.close()
        reopened = self._open_dialog()
        reopened.tabs.setCurrentWidget(reopened.tab_reference)
        reopened.table_reference_regions.setCurrentCell(0, 0)
        self.app.processEvents()
        self.assertEqual(reopened.cmb_region_group.currentData(), group_id)
        self.assertIn("失效组", reopened.cmb_region_group.currentText())
        reopened.tabs.setCurrentWidget(reopened.tab_groups)
        reopened.btn_add_group.click()
        self.app.processEvents()
        self.assertEqual(reopened.cmb_region_group.currentData(), group_id)
        self.assertEqual(self._save(reopened).reference_regions[0]["group_id"], group_id)


if __name__ == "__main__":
    unittest.main()
