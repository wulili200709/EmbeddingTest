from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from application import AlgorithmController, ProductSession
from application.runtime_context import ProductRuntimeContext
from common.model_library import ModelLibrary
from domain import InspectionItem, load_inspection_items, save_inspection_items
from infrastructure.product_params import load_product_params
from ui.debug.tool_page import ToolPage
from ui.i18n import LANG_EN, set_language
from ui.window_common import connect_runtime_refresh_sources


class AlgorithmRuntimeReloadTests(unittest.TestCase):
    """Keep the real synchronous UI -> runtime reload connection in these tests."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        set_language(LANG_EN, persist=False)
        self.session = ProductSession(str(self.root / "orders"))
        self.session.load()
        self.session.switch_product("Default")
        self.algo = AlgorithmController()
        self.algo.model_library = ModelLibrary(self.root / "library")
        self.algo.set_learning_backbone("efficientnet_b0", "cam1")
        self.algo.product_params.algorithm = "efficientnet_b0"
        self.algo.save_params(self.session.product_params_path)
        self.page = None
        self.reload_count = 0

    def tearDown(self):
        if self.page is not None:
            self.page.close()
            self.page.deleteLater()
            self.app.processEvents()
        self.temporary.cleanup()

    def open_page(self, item):
        save_inspection_items([item], self.session.inspection_items_path)
        self.page = ToolPage(self.session, self.algo, lite_mode=True)
        self.page._defer_initial_session_load = False
        self.page.inspection_items = [item]
        self.page._refresh_inspection_items_table()
        self.page.inspection_items_table.selectRow(0)
        self.context = ProductRuntimeContext(self.session, self.algo)

        def refresh_runtime(_message):
            self.reload_count += 1
            self.context.reload()

        self.runtime = SimpleNamespace(refresh_all_status=refresh_runtime)
        connect_runtime_refresh_sources(self.page, self.runtime, session_loaded_message="test")
        self.app.processEvents()

    def assert_algorithm(self, algorithm):
        self.assertEqual(self.page.current_algorithm(), algorithm)
        self.assertEqual(self.algo.product_params.algorithm, algorithm)
        self.assertEqual(load_product_params(self.session.product_params_path).algorithm, algorithm)
        self.page._refresh_inspection_items_table()
        self.app.processEvents()
        self.assertEqual(self.page.current_algorithm(), algorithm)

    def test_learning_subtypes_survive_synchronous_runtime_reload(self):
        self.open_page(InspectionItem("roi", "Part", "cam1", "roi"))
        for choice in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"):
            with self.subTest(algorithm=choice):
                self.page._algorithm_actions[choice].trigger()
                self.app.processEvents()
                self.assert_algorithm(choice)
                self.assertEqual(self.algo.current_learning_backbone("cam1"), choice)
                self.assertEqual(self.algo.resolve_item_algorithm(self.context.inspection_items[0]), choice)
        self.assertGreaterEqual(self.reload_count, 3)

    def test_menu_switching_to_traditional_and_measurement_survives_reload(self):
        self.open_page(InspectionItem("roi", "Part", "cam1", "roi"))
        for choice in ("meanintensity", "find_line", "multi_pin_tip_height"):
            with self.subTest(algorithm=choice):
                self.page._algorithm_actions[choice].trigger()
                self.app.processEvents()
                self.assert_algorithm(choice)
                self.assertEqual(self.context.inspection_items[0].algorithm_code, choice)

    def test_public_reference_tool_change_detaches_and_keeps_public_version_unchanged(self):
        publication = self.algo.model_library.publish(
            "Shared part", "meanintensity",
            traditional_model={"algorithm": "meanintensity", "threshold": 10.,
                               "ok_when": "greater_equal", "ok_mean": 20.,
                               "ng_mean": 0., "accuracy": 1., "roi_label": "source_roi"},
        )
        manifest_path = Path(publication["manifest_path"])
        asset_path = manifest_path.parent / "threshold_model.json"
        manifest_before, asset_before = manifest_path.read_bytes(), asset_path.read_bytes()
        item = InspectionItem(
            "roi", "Part", "cam1", "roi", algorithm_code="meanintensity",
            model_source="library", library_model_id=publication["model_id"],
            library_version=publication["version"],
        )
        self.open_page(item)
        self.page._algorithm_actions["mobilenet_v3_small"].trigger()
        self.app.processEvents()
        self.assert_algorithm("mobilenet_v3_small")
        saved = load_inspection_items(self.session.inspection_items_path)[0]
        self.assertEqual(saved.model_source, "independent")
        self.assertEqual((saved.task_group, saved.library_model_id, saved.library_version), ("", "", ""))
        self.assertEqual(self.context.inspection_items[0].model_source, "independent")
        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertEqual(asset_path.read_bytes(), asset_before)
        self.assertEqual(json.loads(manifest_before)["version"], publication["version"])

    def test_table_tool_choice_is_saved_before_runtime_reload(self):
        self.open_page(InspectionItem("roi", "Part", "cam1", "roi", algorithm_code="meanintensity"))
        for choice in ("meanhsv_h", "meanstd", "find_line"):
            with self.subTest(algorithm=choice):
                combo = self.page.inspection_items_table.cellWidget(0, 3)
                index = combo.findData(choice)
                self.assertGreaterEqual(index, 0)
                combo.setCurrentIndex(index)
                self.app.processEvents()
                self.assert_algorithm(choice)
                self.assertEqual(self.context.inspection_items[0].algorithm_code, choice)
        self.assertGreaterEqual(self.reload_count, 3)


if __name__ == "__main__":
    unittest.main()
