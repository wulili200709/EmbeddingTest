from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

from PySide6 import QtWidgets

from application import AlgorithmController, ProductSession
from common.model_library import ModelLibrary
from domain import InspectionItem, load_inspection_items
from ui.debug.tool_page.page import ToolPage
from ui.i18n import language_code, LANG_EN, set_language


class AlgorithmModelSourceSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.addCleanup(set_language, language_code(), persist=False)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.session = ProductSession(str(self.root / "orders"))
        self.session.load()
        self.session.switch_product("Default")
        self.algo = AlgorithmController()
        self.algo.model_library = ModelLibrary(self.root / "library")
        self.algo.set_learning_backbone("efficientnet_b0", "cam1")
        set_language(LANG_EN, persist=False)
        self.page = ToolPage(self.session, self.algo, lite_mode=True)

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def select_items(self, items):
        self.page.inspection_items = items
        self.page._refresh_inspection_items_table()
        self.page.inspection_items_table.selectRow(0)
        self.app.processEvents()

    def publication(self):
        return self.algo.model_library.publish(
            "Shared presence", "meanintensity",
            traditional_model={"algorithm": "meanintensity", "threshold": 80.,
                               "ok_when": "greater_equal", "roi_label": "training_roi"},
        )

    def library_item(self, label, publication):
        return InspectionItem(label, label, "cam1", label, "meanintensity",
                              model_source="library", library_model_id=publication["model_id"],
                              library_version=publication["version"])

    def choose_tool(self, algorithm):
        action = self.page._algorithm_actions[algorithm]
        self.assertTrue(action.isEnabled())
        action.trigger()
        self.app.processEvents()

    def test_independent_menu_changes_learning_subtype_and_tool_without_reverting(self):
        item = InspectionItem("roi1", "ROI 1", "cam1", "roi1")
        self.select_items([item])
        self.assertTrue(self.page.btn_algorithm_picker.isEnabled())
        for algorithm in ("mobilenet_v3_small", "mobilenet_v3_large", "meanintensity", "find_line"):
            self.choose_tool(algorithm)
            self.assertEqual(self.page.current_algorithm(), algorithm)
            self.assertEqual(self.algo.resolve_item_algorithm(item), algorithm)
            self.assertEqual(item.model_source, "independent")

    def test_public_reference_can_choose_another_tool_without_changing_published_model(self):
        publication = self.publication()
        item = self.library_item("roi1", publication)
        peer = self.library_item("roi2", publication)
        snapshot = self.algo.model_library.resolve(publication["model_id"], publication["version"])
        self.select_items([item, peer])
        self.assertTrue(self.page.btn_algorithm_picker.isEnabled())
        self.assertIn("independent training", self.page.btn_algorithm_picker.toolTip())
        self.choose_tool("meanstd")
        self.assertEqual((item.model_source, item.algorithm_code), ("independent", "meanstd"))
        self.assertEqual((item.library_model_id, item.library_version), ("", ""))
        self.assertEqual(peer.model_source, "library")
        self.assertEqual(self.algo.model_library.resolve(publication["model_id"], publication["version"]), snapshot)
        saved = load_inspection_items(self.session.inspection_items_path)
        self.assertEqual([entry.model_source for entry in saved], ["independent", "library"])
        self.assertEqual(self.page.current_algorithm(), "meanstd")

    def test_same_public_tool_choice_and_refresh_keep_reference(self):
        item = self.library_item("roi1", self.publication())
        self.select_items([item])
        key = item.effective_model_key
        self.choose_tool("meanintensity")
        self.page._refresh_inspection_items_table()
        self.assertEqual(item.model_source, "library")
        self.assertEqual(item.effective_model_key, key)

    def test_changing_group_tool_detaches_only_current_roi(self):
        items = [InspectionItem(label, label, "cam1", label, "meanintensity",
                                model_source="group", task_group="Presence")
                 for label in ("roi1", "roi2")]
        self.select_items(items)
        self.choose_tool("meanstd")
        self.assertEqual((items[0].model_source, items[0].task_group), ("independent", ""))
        self.assertEqual((items[1].model_source, items[1].task_group), ("group", "Presence"))

    def test_table_tool_selector_can_change_public_reference_too(self):
        item = self.library_item("roi1", self.publication())
        self.select_items([item])
        combo = self.page.inspection_items_table.cellWidget(0, 3)
        self.assertTrue(combo.isEnabled())
        combo.setCurrentIndex(combo.findData("meanstd"))
        self.app.processEvents()
        self.assertEqual((item.model_source, item.algorithm_code), ("independent", "meanstd"))
        self.assertEqual(self.page.current_algorithm(), "meanstd")


if __name__ == "__main__":
    unittest.main()
