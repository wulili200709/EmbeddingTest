"""Formal ToolPage/editor/reload wiring, with no camera or production-data writes."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets

from algorithms.image_io import imwrite
from application import AlgorithmController, ProductSession
from application.runtime_context import ProductRuntimeContext
from common.model_library import ModelLibrary
from shape.core import locator
from shape.core.localization_groups import LocalizationGroupError
from shape.core.recipe import ShapeRecipe, load_recipe, save_recipe
from ui.debug import ShapeTemplateDialog, ToolPage
from ui.i18n import LANG_EN, LANG_ZH, language_code, set_language
from ui.shell import main_window as shell
from ui.window_common import connect_runtime_refresh_sources


class LocalizationGroupsMainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="formal-group-integration-")
        self.root = Path(self.temporary.name)
        self.previous_language = language_code()
        set_language(LANG_ZH, persist=False)
        self.stack = ExitStack()
        for method in ("warning", "critical", "information"):
            self.stack.enter_context(patch.object(QtWidgets.QMessageBox, method, return_value=QtWidgets.QMessageBox.Ok))
        self.stack.enter_context(patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Discard))
        self.session = ProductSession(str(self.root / "formal_orders"))
        self.session.load()
        self.session.switch_product("Default")
        self.reference = self.root / "reference.png"
        self.image = np.random.default_rng(240).integers(20, 235, (240, 320, 3), dtype=np.uint8)
        self.assertTrue(imwrite(self.reference, self.image))
        self.paths = locator.product_paths(self.session.product_dir, "cam1")
        self.legacy_recipe = ShapeRecipe(
            model_path=self.paths.model_path, reference_image=str(self.reference),
            reference_regions=[{"output_label": "roi1", "shape_type": "rectangle", "points": [[140., 90.], [180., 125.]]}],
        )
        locator.save_recipe_for_product(self.session.product_dir, self.legacy_recipe)
        self.algo = AlgorithmController()
        self.algo.model_library = ModelLibrary(self.root / "model_library")
        self.algo.product_params.algorithm = "meanintensity"
        self.algo.save_params(self.session.product_params_path)
        self.owner = QtWidgets.QWidget()
        self.page = ToolPage(self.session, self.algo, parent=self.owner, lite_mode=True)
        self.page._defer_initial_session_load = False
        self.page.load_session()
        self.context = ProductRuntimeContext(self.session, self.algo)
        self.reload_messages = []

        def reload_runtime(message):
            self.reload_messages.append(message)
            self.context.reload()

        self.runtime = SimpleNamespace(refresh_all_status=reload_runtime)
        connect_runtime_refresh_sources(self.page, self.runtime, session_loaded_message="integration reload")
        self.app.processEvents()

    def tearDown(self):
        dialog = getattr(self.page, "_template_editor_dialog", None)
        if dialog is not None:
            dialog.close()
        self.page.close()
        self.page.deleteLater()
        self.owner.close()
        self.owner.deleteLater()
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.stack.close()
        set_language(self.previous_language, persist=False)
        self.temporary.cleanup()

    def open_formal_editor(self):
        self.page.open_template_editor_dialog()
        self.app.processEvents()
        dialog = self.page._template_editor_dialog
        self.assertIsInstance(dialog, ShapeTemplateDialog)
        self.assertEqual(Path(dialog.product_dir), Path(self.session.product_dir))
        return dialog

    def add_and_save_group(self, dialog):
        dialog.tabs.setCurrentWidget(dialog.tab_groups)
        dialog.btn_add_group.click()
        dialog.group_canvas.set_roi_rect((30, 35, 66, 61))
        dialog.edit_group_name.setText("用户定义定位组")
        QtTest.QTest.keyClick(dialog.edit_group_name, QtCore.Qt.Key_Return)
        group_id = dialog._localization_groups[0]["id"]
        dialog.tabs.setCurrentWidget(dialog.tab_reference)
        dialog.table_reference_regions.setCurrentCell(0, 0)
        self.app.processEvents()
        dialog.cmb_region_group.setCurrentIndex(dialog.cmb_region_group.findData(group_id))
        dialog.btn_save_groups.click()
        self.app.processEvents()
        saved = load_recipe(self.paths.recipe_path)
        self.assertEqual(saved.reference_regions[0]["group_id"], group_id)
        return group_id

    def configure_second_camera(self):
        camera2 = copy.deepcopy(self.legacy_recipe)
        camera2.model_path = locator.product_paths(self.session.product_dir, "cam2").model_path
        locator.save_recipe_for_product(self.session.product_dir, camera2, "cam2")
        self.page._configured_camera_roles = ["cam1", "cam2"]
        self.page._apply_configured_camera_roles_to_ui()
        self.page.shape_recipe_for_role("cam2", force_reload=True)
        return Path(locator.product_paths(self.session.product_dir, "cam2").recipe_path)

    def shell_host(self):
        return SimpleNamespace(
            session=self.session, tool_page=self.page,
            runtime_ctrl=SimpleNamespace(disconnect=Mock(), refresh_all_status=Mock()),
            _require_permission=lambda *_args: True, _preload_current_embedding_model=Mock(),
            _sync_shell_status=Mock(), _sync_permission_ui=Mock(), _audit_event=Mock(),
            _bottom_status_bar=SimpleNamespace(showMessage=Mock()),
        )

    def test_formal_entry_keeps_legacy_product_and_does_not_import_demo_configuration(self):
        # A separate demo has configured groups; the formal editor must use only
        # the current ProductSession, without searching/copying demo settings.
        demo_recipe_path = self.root / "independent_demo" / "recipe.json"
        demo = copy.deepcopy(self.legacy_recipe)
        demo.localization_groups = [{"id": "demo-only", "name": "Do not import", "points": [[1., 1.], [30., 30.]]}]
        demo.reference_regions[0]["group_id"] = "demo-only"
        save_recipe(demo, str(demo_recipe_path))
        demo_before = demo_recipe_path.read_bytes()
        formal_before = Path(self.paths.recipe_path).read_bytes()
        dialog = self.open_formal_editor()
        self.assertEqual(dialog._localization_groups, [])
        self.assertFalse(dialog._reference_regions[0].get("group_id"))
        self.assertEqual(dialog.table_reference_regions.rowCount(), 1)
        self.assertEqual(Path(self.paths.recipe_path).read_bytes(), formal_before)
        self.assertEqual(demo_recipe_path.read_bytes(), demo_before)
        self.assertFalse(Path(self.paths.model_path).exists(), "opening the editor must not build or replace the model")

    def test_group_save_reloads_live_context_and_failure_cannot_reach_inference(self):
        self.assertEqual(self.context._ensure_recipe_loaded("cam1").localization_groups, [])
        dialog = self.open_formal_editor()
        group_id = self.add_and_save_group(dialog)
        self.assertGreaterEqual(len(self.reload_messages), 1)
        recipe = self.context._ensure_recipe_loaded("cam1")
        self.assertEqual(recipe.localization_groups[0]["id"], group_id)
        self.assertEqual(recipe.reference_regions[0]["group_id"], group_id)
        self.assertEqual([item.roi_label for item in self.context.inspection_items], ["roi1"])
        failure = LocalizationGroupError([{"id": group_id, "used": True, "valid": False, "reason": "ambiguous"}])
        with patch("application.runtime_context.shape_locator.autogen_runtime_roi_shapes_timed", side_effect=failure), patch(
            "application.runtime_context._predict_learning_items_batch_rows_from_frame"
        ) as predict:
            with self.assertRaises(LocalizationGroupError):
                self.context.predict_items_batch_from_frame(self.image, camera_role="cam1", items=self.context.inspection_items)
        predict.assert_not_called()

    def test_formal_model_saved_signal_invalidates_model_cache_and_refreshes_groups(self):
        dialog = self.open_formal_editor()
        group_id = self.add_and_save_group(dialog)
        before = len(self.reload_messages)
        # Substitute the native model-build boundary; the actual Qt signal,
        # ToolPage receiver, cache invalidation and runtime reload remain real.
        with patch.object(locator, "invalidate_shape_model_cache", wraps=locator.invalidate_shape_model_cache) as invalidate:
            dialog.modelSaved.emit(self.paths.model_path, self.paths.recipe_path)
            self.app.processEvents()
        invalidate.assert_called_once_with(self.paths.model_path)
        self.assertGreater(len(self.reload_messages), before)
        self.assertEqual(self.context._ensure_recipe_loaded("cam1").localization_groups[0]["id"], group_id)

    def test_formal_reference_replacement_marks_saved_groups_and_reloads_live_context(self):
        dialog = self.open_formal_editor()
        self.add_and_save_group(dialog)
        replacement = self.root / "replacement_reference.png"
        self.assertTrue(imwrite(replacement, self.image))
        before = len(self.reload_messages)
        self.page._set_reference(str(replacement))
        self.app.processEvents()
        self.assertGreater(len(self.reload_messages), before)
        recipe = self.context._ensure_recipe_loaded("cam1")
        self.assertEqual(Path(recipe.reference_image), replacement)
        self.assertTrue(recipe.localization_groups[0]["needs_rebuild"])

    def test_main_language_dispatch_retranslates_live_group_controls_without_rebinding(self):
        dialog = self.open_formal_editor()
        group_id = self.add_and_save_group(dialog)
        dialog._localization_groups[0]["needs_rebuild"] = True
        dialog._refresh_group_list(select_id=group_id)
        groups_before = copy.deepcopy(dialog._localization_groups)
        regions_before = copy.deepcopy(dialog._reference_regions)
        file_before = Path(self.paths.recipe_path).read_bytes()
        chinese_button = dialog.btn_add_group.text()
        host = SimpleNamespace(tool_page=self.page, runtime_page=SimpleNamespace(retranslate_ui=Mock()),
                               _sync_permission_ui=Mock())
        host.retranslate_ui = lambda: shell.MainWindow.retranslate_ui(host)
        with patch.object(shell, "_retranslate_shell_chrome"), patch.object(
            shell, "set_language", side_effect=lambda language: set_language(language, persist=False)
        ), patch.object(dialog, "retranslate_ui", wraps=dialog.retranslate_ui) as retranslate:
            shell.MainWindow._change_language(host, LANG_EN)
            self.app.processEvents()
            self.assertGreaterEqual(retranslate.call_count, 1)
            self.assertNotEqual(dialog.btn_add_group.text(), chinese_button)
            for text in (dialog.btn_add_group.text(), dialog.tabs.tabText(dialog.tabs.indexOf(dialog.tab_groups)),
                         dialog.table_reference_regions.horizontalHeaderItem(4).text(), dialog.cmb_region_group.itemText(0)):
                self.assertTrue(text)
                self.assertNotRegex(text, r"[\u4e00-\u9fff]", text)
            item_text = dialog.list_localization_groups.item(0).text()
            self.assertIn("用户定义定位组", item_text)
            self.assertNotRegex(item_text.replace("用户定义定位组", ""), r"[\u4e00-\u9fff]")
            self.assertEqual(dialog.cmb_region_group.currentData(), group_id)
            shell.MainWindow._change_language(host, LANG_ZH)
            self.app.processEvents()
        self.assertEqual(dialog.btn_add_group.text(), chinese_button)
        self.assertEqual(dialog._localization_groups, groups_before)
        self.assertEqual(dialog._reference_regions, regions_before)
        self.assertEqual(Path(self.paths.recipe_path).read_bytes(), file_before)

    def test_nonmodal_camera1_save_refreshes_camera1_without_changing_selected_camera2(self):
        camera2_path = self.configure_second_camera()
        camera2_before = camera2_path.read_bytes()
        dialog = self.open_formal_editor()
        self.page._set_current_camera_role("cam2")
        self.assertEqual(self.page.current_camera_role(), "cam2")
        group_id = self.add_and_save_group(dialog)
        self.assertEqual(self.page.current_camera_role(), "cam2")
        self.assertFalse(self.page.shape_recipe.reference_regions[0].get("group_id"))
        self.assertEqual(self.page.shape_recipe_for_role("cam1").reference_regions[0]["group_id"], group_id)
        self.assertEqual(self.context._ensure_recipe_loaded("cam1").reference_regions[0]["group_id"], group_id)
        self.page._set_loc_method_for_role("cam2", "ncc")
        dialog.modelSaved.emit(self.paths.model_path, self.paths.recipe_path)
        self.app.processEvents()
        self.assertEqual(self.page.current_camera_role(), "cam2")
        self.assertEqual(self.page.loc_method_for_role("cam2"), "ncc")
        self.assertEqual(camera2_path.read_bytes(), camera2_before)

    def test_replacing_editor_for_another_camera_survives_old_deferred_destroy_signal(self):
        self.configure_second_camera()
        old_dialog = self.open_formal_editor()
        self.page._set_current_camera_role("cam2")
        self.page.open_template_editor_dialog()
        new_dialog = self.page._template_editor_dialog
        self.assertIsNot(new_dialog, old_dialog)
        self.assertEqual(new_dialog.camera_role, "cam2")
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.assertIs(self.page._template_editor_dialog, new_dialog)

    def test_cancel_unsaved_editor_close_keeps_product_selector_and_runtime_unchanged(self):
        self.assertEqual(self.session.create_product("Other"), "")
        self.page.refresh_product_selector()
        dialog = self.open_formal_editor()
        dialog.btn_add_group.click()
        self.assertTrue(dialog._reference_dirty)
        host = self.shell_host()
        self.page.productChangeRequested.connect(lambda name: shell.MainWindow._on_product_change_request(host, name))
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.page.cmb_product.setCurrentText("Other")
            self.app.processEvents()
        self.assertEqual(self.session.current_product, "Default")
        self.assertEqual(self.page.cmb_product.currentText(), "Default")
        self.assertIs(self.page._template_editor_dialog, dialog)
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog._reference_dirty)
        host.runtime_ctrl.disconnect.assert_not_called()
        host.runtime_ctrl.refresh_all_status.assert_not_called()

    def test_product_switch_closes_editor_and_ignores_previous_product_callbacks(self):
        dialog = self.open_formal_editor()
        self.add_and_save_group(dialog)
        previous_dir = self.session.product_dir
        previous_recipe_bytes = Path(self.paths.recipe_path).read_bytes()
        self.assertEqual(self.session.create_product("Other"), "")
        self.page.apply_product_switch("Other")
        self.app.processEvents()
        self.assertEqual(self.session.current_product, "Other")
        self.assertIsNone(self.page._template_editor_dialog)
        before = len(self.reload_messages)
        with patch.object(self.page, "_save_session", wraps=self.page._save_session) as save_session:
            self.page._on_shape_reference_regions_changed(camera_role="cam1", product_dir=previous_dir)
            self.page._on_shape_model_saved(self.paths.model_path, self.paths.recipe_path,
                                            camera_role="cam1", product_dir=previous_dir)
        save_session.assert_not_called()
        self.assertEqual(len(self.reload_messages), before)
        self.assertEqual(Path(self.paths.recipe_path).read_bytes(), previous_recipe_bytes)
        self.assertIsNone(self.context._ensure_recipe_loaded("cam1"))

    def test_deleting_another_product_keeps_unrelated_unsaved_editor_open(self):
        self.assertEqual(self.session.create_product("Other"), "")
        self.page.refresh_product_selector()
        dialog = self.open_formal_editor()
        dialog.btn_add_group.click()
        self.assertTrue(dialog._reference_dirty)
        host = self.shell_host()
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes) as question:
            shell.MainWindow._on_product_delete_request(host, "Other")
            self.app.processEvents()
        self.assertEqual(question.call_count, 1, "an unrelated editor must not receive a close/discard prompt")
        self.assertNotIn("Other", self.session.product_names)
        self.assertEqual(self.session.current_product, "Default")
        self.assertEqual(self.page.cmb_product.currentText(), "Default")
        self.assertIs(self.page._template_editor_dialog, dialog)
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog._reference_dirty)
        host.runtime_ctrl.disconnect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
