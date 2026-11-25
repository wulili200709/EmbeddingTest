"""Language changes re-render group UI and cached results without rerunning matching."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from PySide6 import QtCore, QtTest, QtWidgets

import test_localization_groups_ui as flows
from shape.core.localization_groups import LocalizationGroupError
from ui.i18n import LANG_EN, LANG_ZH, set_language, tr
from ui.shape_template import template_page_pyside6 as page


class LocalizationGroupsLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flows.LocalizationGroupsUiTests.setUpClass()

    def setUp(self):
        self.flow = flows.LocalizationGroupsUiTests("runTest")
        self.flow.setUp()
        self.dialog = self.flow.dialog

    def tearDown(self):
        self.flow.tearDown()

    def switch(self, language):
        set_language(language, persist=False)
        self.dialog.retranslate_ui()
        self.flow.app.processEvents()

    def test_language_switch_preserves_pending_name_bindings_and_saved_configuration(self):
        gid = self.flow._create_group_and_bind_roi("用户上排壳体")
        self.flow._save()
        dialog = self.dialog
        original_recipe = Path(dialog.paths.recipe_path).read_bytes()
        original_groups = copy.deepcopy(dialog._localization_groups)
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.list_localization_groups.setCurrentRow(0)
        dialog.edit_group_name.setFocus()
        dialog.edit_group_name.setText("用户尚未提交的新名字")
        dialog.edit_group_name.setModified(True)
        dialog.edit_group_name.setSelection(2, 3)
        dialog._set_group_status("template.groups.saved", groups=1, rois=1)

        self.switch(LANG_EN)
        self.assertEqual(dialog.btn_add_group.text(), "Add group")
        self.assertEqual(dialog.edit_group_name.text(), "用户尚未提交的新名字")
        self.assertTrue(dialog.edit_group_name.isModified())
        self.assertEqual(dialog.edit_group_name.selectedText(), "尚未提")
        self.assertEqual(dialog._localization_groups, original_groups)
        self.assertFalse(dialog._reference_dirty)
        self.assertEqual(dialog.cmb_region_group.currentData(), gid)
        self.assertEqual(dialog.lbl_group_status.text(), tr("template.groups.saved", groups=1, rois=1))
        self.assertIn("Search", dialog._group_form_labels[1][0].text())
        self.assertNotIn("定位", dialog.spin_group_radius.toolTip())
        self.assertEqual(Path(dialog.paths.recipe_path).read_bytes(), original_recipe)

        self.switch(LANG_ZH)
        self.assertEqual(dialog.btn_add_group.text(), "新建定位组")
        self.assertEqual(dialog.edit_group_name.text(), "用户尚未提交的新名字")
        self.assertEqual(dialog._localization_groups, original_groups)
        self.assertEqual(Path(dialog.paths.recipe_path).read_bytes(), original_recipe)

    def test_cached_failure_and_global_comparison_translate_without_matching_again(self):
        gid = self.flow._create_group_and_bind_roi("固定壳体")
        self.flow._save()
        dialog = self.dialog
        item = self.flow._prepare_find()
        failed = self.flow._result(gid, valid=False)
        failed.group_results[0].update(
            reason_code="ambiguous_peak", reason_params={"margin": .01, "minimum": .03},
        )
        error = LocalizationGroupError(failed.group_results, preview_result=failed)
        with patch.object(page, "locate_and_follow", side_effect=error) as locate:
            dialog.btn_run_find_selected.click()
            self.assertEqual(locate.call_count, 1)
            self.assertEqual(dialog.table_group_results.item(0, 1).text(), "失败")
            previous_reason = dialog.table_group_results.item(0, 5).text()
            self.switch(LANG_EN)
            self.assertEqual(locate.call_count, 1)
            self.assertEqual(dialog.table_group_results.item(0, 1).text(), "Failed")
            current_reason = dialog.table_group_results.item(0, 5).text()
            self.assertNotEqual(previous_reason, current_reason)
            self.assertNotIn("峰值", current_reason)
            self.assertIn("0.010", current_reason)
            self.assertIn(current_reason, dialog.lbl_find_status.text())
            self.assertIn(current_reason, item.text())
            self.assertEqual(dialog.table_group_results.item(0, 5).toolTip(), current_reason)
            dialog.list_find_images.setCurrentRow(1)
            dialog.list_find_images.setCurrentRow(0)
            self.flow.app.processEvents()
            self.assertIn(current_reason, dialog.lbl_find_status.text())
            self.assertEqual(locate.call_count, 1)
            self.switch(LANG_ZH)
            self.assertEqual(dialog.table_group_results.item(0, 5).text(), previous_reason)

        dialog.chk_enable_groups.setChecked(False)
        with patch.object(page, "locate_and_follow", return_value=self.flow._result(gid, grouped=False)) as locate:
            dialog.btn_run_find_selected.click()
            self.assertIn("仅全局定位", item.text())
            self.switch(LANG_EN)
            self.assertEqual(locate.call_count, 1)
            self.assertIn(tr("template.groups.global_only"), item.text())
            self.assertNotIn("仅全局定位", item.text())
            self.assertEqual(dialog._reference_regions[0]["group_id"], gid)

    def test_new_group_uses_current_language_and_delete_dialog_uses_english_with_safe_default(self):
        gid = self.flow._create_group_and_bind_roi("自定义名称")
        dialog = self.dialog
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        self.switch(LANG_EN)
        dialog.btn_add_group.click()
        self.assertEqual(dialog._localization_groups[1]["name"], "Group 1")
        dialog.list_localization_groups.setCurrentRow(0)
        dialog.edit_group_name.setFocus()
        observed = {}

        def cancel_actual_dialog():
            box = self.flow.app.activeModalWidget()
            if not isinstance(box, QtWidgets.QMessageBox):
                observed["error"] = "No confirmation dialog"
                return
            observed.update(title=box.windowTitle(), text=box.text(), info=box.informativeText(),
                            default=box.defaultButton().text(), buttons=[b.text() for b in box.buttons()])
            box.defaultButton().click()

        QtCore.QTimer.singleShot(0, cancel_actual_dialog)
        QtTest.QTest.mouseClick(dialog.btn_delete_group, QtCore.Qt.LeftButton)
        self.flow.app.processEvents()
        self.assertNotIn("error", observed)
        self.assertEqual(observed["title"], "Delete group")
        self.assertEqual(observed["default"], "Cancel")
        self.assertIn("roi1", observed["text"])
        self.assertIn("自定义名称", observed["text"])
        self.assertIn(tr("template.groups.delete_confirm"), observed["buttons"])
        self.assertNotIn("检测", observed["info"])
        self.assertEqual(dialog._reference_regions[0]["group_id"], gid)
        self.assertIn("cancelled", dialog.lbl_group_status.text().lower())
        self.switch(LANG_ZH)
        self.assertIn("已取消", dialog.lbl_group_status.text())
        self.assertEqual(dialog._localization_groups[0]["name"], "自定义名称")
        self.assertEqual(dialog._localization_groups[1]["name"], "Group 1")


if __name__ == "__main__":
    unittest.main()
