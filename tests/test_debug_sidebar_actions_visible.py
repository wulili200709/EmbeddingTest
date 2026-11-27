from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from application import AlgorithmController, ProductSession
from common.model_library import ModelLibrary
from domain import load_inspection_items
from ui.debug.tool_page import ToolPage
from ui.i18n import LANG_EN, LANG_ZH, set_language, tr


class DebugSidebarActionsVisibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        set_language(LANG_ZH, persist=False)
        session = ProductSession(str(root / "orders"))
        session.load()
        session.switch_product("Default")
        Path(session.inspection_items_path).write_text(json.dumps([
            {"item_id": f"roi{i}", "display_name": f"ROI {i}", "camera_id": "cam1",
             "roi_label": f"roi{i}", "algorithm_code": "meanintensity"}
            for i in range(1, 8)
        ]), encoding="utf-8")
        algo = AlgorithmController()
        algo.model_library = ModelLibrary(root / "library")
        self.page = ToolPage(session, algo, lite_mode=True)
        self.page.inspection_items = load_inspection_items(session.inspection_items_path)
        self.page._refresh_inspection_items_table()
        self.page.inspection_items_table.selectRow(1)
        self.page.resize(1280, 720)
        self.page.show()
        self.flush()

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.flush()
        self.temporary.cleanup()

    def flush(self):
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def assert_actions_visible(self, requested_height=720):
        self.flush()
        self.assertLessEqual(self.page.height(), requested_height,
                             "Expanding settings must not force the window below the screen")
        for name in ("btn_train", "btn_train_current", "btn_test", "btn_export_test", "btn_clear_session"):
            button = getattr(self.page, name)
            with self.subTest(button=name):
                self.assertTrue(button.isVisible())
                rect = QtCore.QRect(button.mapTo(self.page, QtCore.QPoint()), button.size())
                self.assertTrue(self.page.rect().contains(rect), f"{name} is outside the page")
                self.assertFalse(button.visibleRegion().isEmpty(), f"{name} is clipped")

    def test_all_expanded_settings_keep_test_actions_inside_720px_window(self):
        panel = self.page.model_source_panel
        panel.mode.setCurrentIndex(panel.mode.findData("group"))
        panel.group.setCurrentText("PIN")
        self.assert_actions_visible()

    def test_toggle_and_roi_refresh_do_not_move_actions_off_screen(self):
        for number in range(3):
            self.page.btn_toggle_algo.setChecked(number % 2 == 0)
            self.page.btn_toggle_tools.setChecked(number % 2 == 0)
            self.page.btn_toggle_model_source.setChecked(number % 2 == 0)
            self.page.inspection_items_table.selectRow(number)
            self.page._update_runtime_widgets()
            self.assert_actions_visible()

    def test_scrolling_configuration_keeps_actions_pinned(self):
        scroll = self.page.sidebar_config_scroll
        start = self.page.btn_test.mapTo(self.page, QtCore.QPoint())
        image_list_position = self.page.tabs.mapTo(self.page, QtCore.QPoint())
        self.assertFalse(scroll.widget().isAncestorOf(self.page.btn_test))
        self.assertFalse(scroll.widget().isAncestorOf(self.page.tabs))
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        for position in (scroll.verticalScrollBar().maximum(), 0):
            scroll.verticalScrollBar().setValue(position)
            self.assert_actions_visible()
            self.assertEqual(self.page.btn_test.mapTo(self.page, QtCore.QPoint()), start)
            self.assertEqual(self.page.tabs.mapTo(self.page, QtCore.QPoint()), image_list_position)

    def test_training_progress_and_confirmation_keep_test_actions_visible(self):
        self.page.training_controller.set_running(True)
        self.assert_actions_visible()
        self.page.training_controller.set_running(False)
        self.page._training_roi_pending_actions["cam1"] = "current"
        self.page._sync_training_action_buttons()
        self.assertTrue(self.page.btn_train_current_cancel.isVisible())
        self.assert_actions_visible()

    def test_resize_and_language_changes_need_no_maximize_workaround(self):
        for height, language in ((900, LANG_EN), (720, LANG_ZH), (768, LANG_EN), (720, LANG_ZH)):
            self.page.resize(1280, height)
            set_language(language, persist=False)
            self.page.retranslate_ui()
            self.assert_actions_visible(height)
            button = self.page.btn_copy_product
            self.assertEqual(button.text(), tr("debug.copy_product"))
            self.assertGreaterEqual(button.width(), button.sizeHint().width())

    def test_main_window_menu_and_status_bar_do_not_hide_test_actions(self):
        host = QtWidgets.QMainWindow()
        host.menuBar().addMenu("菜单")
        host.statusBar().showMessage("调试界面")
        host.setCentralWidget(self.page)
        host.resize(1280, 720)
        host.show()
        try:
            self.page.model_source_panel.mode.setCurrentIndex(self.page.model_source_panel.mode.findData("group"))
            self.flush()
            self.assertLessEqual(host.height(), 720)
            for name in ("btn_train", "btn_train_current", "btn_test", "btn_clear_session"):
                button = getattr(self.page, name)
                rect = QtCore.QRect(button.mapTo(host, QtCore.QPoint()), button.size())
                self.assertTrue(host.rect().contains(rect), name)
                self.assertFalse(button.visibleRegion().isEmpty(), name)
        finally:
            host.takeCentralWidget()
            host.close()
            host.deleteLater()


if __name__ == "__main__":
    unittest.main()
