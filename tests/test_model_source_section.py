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
from domain import InspectionItem, load_inspection_items
from ui.debug.tool_page import ToolPage
from ui.i18n import language_code, LANG_EN, LANG_ZH, set_language


class ModelSourceSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.addCleanup(set_language, language_code(), persist=False)
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        set_language(LANG_EN, persist=False)
        session = ProductSession(str(root / "legacy_order"))
        session.load()
        session.switch_product("Default")
        Path(session.inspection_items_path).write_text(json.dumps([
            {"item_id": f"roi{number}", "display_name": f"Legacy part {number}", "camera_id": "cam1", "roi_label": f"roi{number}", "algorithm_code": "meanintensity"}
            for number in range(1, 6)
        ]), encoding="utf-8")
        algo = AlgorithmController()
        algo.model_library = ModelLibrary(root / "model_library")
        self.page = ToolPage(session, algo, lite_mode=True)
        self.page.inspection_items = load_inspection_items(session.inspection_items_path)
        self.page._refresh_inspection_items_table()
        self.page.inspection_items_table.selectRow(0)
        self.page.resize(1100, 760)
        self.page.show()
        self.app.processEvents()

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_section_defaults_open_outside_tools_scroll(self):
        toggle = self.page.btn_toggle_model_source
        frame = self.page.model_source_frame
        self.page.sidebar_config_scroll.ensureWidgetVisible(toggle, 0, 4)
        self.app.processEvents()
        self.assertTrue(toggle.isChecked())
        self.assertEqual(toggle.arrowType(), QtCore.Qt.ArrowType.DownArrow)
        self.assertTrue(frame.isVisible())
        self.assertIs(frame, self.page.model_source_panel.frame)
        self.assertFalse(self.page.tool_config_scroll.widget().isAncestorOf(frame))
        self.assertTrue(self.page.inspection_items_table.isVisible())
        self.assertGreaterEqual(self.page.inspection_items_table.minimumHeight(), 170)
        self.assertEqual(toggle.width(), self.page.btn_toggle_model_source.parentWidget().width())
        self.assertLessEqual(toggle.mapTo(self.page, toggle.rect().bottomRight()).y(), 760)

    def test_collapsed_state_survives_refresh_selection_and_language_changes(self):
        toggle = self.page.btn_toggle_model_source
        toggle.setChecked(False)
        self.assertEqual(toggle.arrowType(), QtCore.Qt.ArrowType.RightArrow)
        self.assertTrue(self.page.model_source_frame.isHidden())
        self.page.model_source_panel.refresh()
        self.page.model_source_panel.reload_library()
        self.page._update_runtime_widgets()
        self.page.inspection_items_table.selectRow(1)
        set_language(LANG_ZH, persist=False)
        self.page.retranslate_ui()
        self.assertFalse(toggle.isChecked())
        self.assertTrue(self.page.model_source_frame.isHidden())
        self.assertEqual(toggle.text(), "模型来源")
        self.assertEqual(toggle.arrowType(), QtCore.Qt.ArrowType.RightArrow)
        toggle.setChecked(True)
        self.assertTrue(self.page.model_source_frame.isVisible())

    def test_legacy_order_binding_remains_after_collapsing_and_reopening(self):
        item = self.page.inspection_items[0]
        self.assertEqual(item.model_source, "independent")
        panel = self.page.model_source_panel
        panel.mode.setCurrentIndex(panel.mode.findData("group"))
        panel.group.setCurrentText("Same parts")
        panel.apply()
        self.page.btn_toggle_model_source.setChecked(False)
        panel.refresh()
        self.page.btn_toggle_model_source.setChecked(True)
        self.assertEqual(panel.mode.currentData(), "group")
        self.assertEqual(panel.group.currentText(), "Same parts")
        saved = load_inspection_items(self.page.session.inspection_items_path)[0]
        self.assertEqual((saved.model_source, saved.task_group), ("group", "Same parts"))

    def test_model_section_is_independent_of_tools_collapse(self):
        self.page.btn_toggle_tools.setChecked(False)
        self.assertTrue(self.page.tool_config_scroll.isHidden())
        self.assertTrue(self.page.btn_toggle_model_source.isVisible())
        self.assertTrue(self.page.model_source_frame.isVisible())
        self.page.btn_toggle_tools.setChecked(True)
        self.assertTrue(self.page.inspection_items_table.isVisible())

    def test_measurement_selection_preserves_user_collapse_state(self):
        self.page.inspection_items.append(InspectionItem("line", "Reference line", "cam1", "roi6", "find_line"))
        self.page._refresh_inspection_items_table()
        self.page.btn_toggle_model_source.setChecked(False)
        self.page.inspection_items_table.selectRow(5)
        self.assertFalse(self.page.btn_toggle_model_source.isEnabled())
        self.assertTrue(self.page.model_source_frame.isHidden())
        self.page.inspection_items_table.selectRow(0)
        self.assertTrue(self.page.btn_toggle_model_source.isEnabled())
        self.assertFalse(self.page.btn_toggle_model_source.isChecked())
        self.assertTrue(self.page.model_source_frame.isHidden())


if __name__ == "__main__":
    unittest.main()
