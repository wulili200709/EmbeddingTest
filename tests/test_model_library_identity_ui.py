"""Disambiguate homonymous public models and publish to the intended identity."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6 import QtWidgets

from application import AlgorithmController
from common.model_library import ModelLibrary
from domain import InspectionItem
from test_model_source_ui import _Owner
from ui.debug.tool_page.model_source_panel import ModelSourcePanel, binding_status
from ui.i18n import LANG_EN, set_language


class ModelLibraryIdentityUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        set_language(LANG_EN, persist=False)
        item = InspectionItem("roi1", "Current pin", "cam1", "roi1", "meanintensity")
        self.owner = _Owner(self.root, [item])
        self.owner.algo = AlgorithmController()
        self.owner.algo.model_library = ModelLibrary(self.root / "library")
        self.owner.algo.product_params.algorithm = "meanintensity"
        self.owner.algo.product_params.traditional_models = {
            self.owner.algo.traditional_model_storage_key("meanintensity", model_key=item.model_key):
                self.threshold(90.),
        }
        self.panel = ModelSourcePanel(self.owner, QtWidgets.QVBoxLayout(self.owner))
        self.owner.model_source_panel = self.panel
        self.first_id = "12345678" + "a" * 24
        self.second_id = "12345678" + "b" * 24

    def tearDown(self):
        self.owner.close()
        self.owner.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    @staticmethod
    def threshold(value):
        return {"algorithm": "meanintensity", "threshold": value, "ok_when": "greater_equal"}

    def publish(self, identifier, *, threshold=80., order="order-A", roi="roi1", provenance=True):
        return self.owner.algo.model_library.publish(
            "PINTEST", "meanintensity", model_id=identifier,
            traditional_model=self.threshold(threshold),
            source={"order": order, "camera_id": "cam1", "roi": roi,
                    "roi_display_name": f"Pin {roi}", "group": "PIN"} if provenance else None,
        )

    def index_for(self, publication):
        identity = (publication["model_id"], publication["version"])
        return next((index for index in range(self.panel.library.count())
                     if self.panel.library.itemData(index) == identity), -1)

    def select(self, publication):
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("library"))
        index = self.index_for(publication)
        self.assertGreater(index, 0)
        self.panel.library.setCurrentIndex(index)
        return index

    def test_same_name_and_version_show_distinct_ids_and_provenance_before_apply(self):
        first = self.publish(self.first_id)
        second = self.publish(self.second_id, threshold=128., order="order-B", roi="roi2")
        self.panel.reload_library()
        first_index = self.select(first)
        first_text = self.panel.library.itemText(first_index)
        second_index = self.select(second)
        second_text = self.panel.library.itemText(second_index)
        self.assertNotEqual(first_text, second_text)
        self.assertIn("12345678a", first_text)
        self.assertIn("12345678b", second_text)
        self.assertIn("Grayscale Tool", second_text)
        details = self.panel.library_details.text()
        self.assertIn("order-B", details)
        self.assertIn("Pin roi2", details)
        self.assertIn(second["model_id"], details)
        self.assertIn("UTC", details)
        self.assertFalse(self.panel.library_details.isHidden())
        self.assertEqual(self.owner.selected.model_source, "independent")
        self.assertTrue(self.panel.apply())
        bound = self.owner.selected
        self.assertEqual((bound.library_model_id, bound.library_version), (self.second_id, "v000001"))
        self.assertEqual(self.owner.algo.get_traditional_model_dict(
            "meanintensity", model_key=bound.effective_model_key)["threshold"], 128.)
        status, tooltip, _color = binding_status(self.owner, bound)
        self.assertIn("12345678b", status)
        self.assertIn(self.second_id, tooltip)
        self.assertIn("order-B", tooltip)

    def test_new_version_keeps_identity_and_does_not_upgrade_existing_roi(self):
        first = self.publish(self.first_id)
        self.panel.reload_library()
        self.select(first)
        self.assertTrue(self.panel.apply())
        old_manifest = Path(first["manifest_path"]).read_bytes()
        second = self.publish(self.first_id, threshold=150., order="order-B")
        self.panel.reload_library()
        self.assertEqual(self.panel.library.currentData(), (self.first_id, first["version"]))
        self.assertIn("order-A", self.panel.library_details.text())
        second_index = self.index_for(second)
        self.assertIn("v000002", self.panel.library.itemText(second_index))
        self.assertIn(self.first_id[:8], self.panel.library.itemText(second_index))
        self.assertEqual(Path(first["manifest_path"]).read_bytes(), old_manifest)

    def test_publish_existing_uses_latest_version_and_updates_only_selected_identity(self):
        self.publish(self.first_id)
        latest = self.publish(self.first_id, threshold=100.)
        other = self.publish(self.second_id, order="order-B")
        protected = {path: path.read_bytes() for path in self.owner.algo.model_library.root.rglob("*.json")}
        self.panel.reload_library()

        def choose_existing(_parent, _title, _prompt, options, *_args):
            target = next(option for option in options if "12345678a" in option)
            self.assertIn(latest["version"], target)
            self.assertIn("order-A", target)
            return target, True

        with patch.object(QtWidgets.QInputDialog, "getItem", side_effect=choose_existing), \
             patch.object(QtWidgets.QInputDialog, "getText", return_value=("PINTEST", True)) as name, \
             patch.object(QtWidgets.QMessageBox, "information") as success:
            self.panel.publish()
        self.assertIn("12345678a", name.call_args.args[2])
        published = self.owner.algo.model_library.resolve(self.first_id, "v000003")
        self.assertEqual(published["traditional_model"]["threshold"], 90.)
        self.assertEqual(published["source"]["roi_display_name"], "Current pin")
        self.assertEqual(published["source"]["camera_id"], "cam1")
        self.assertIn("12345678a", success.call_args.args[2])
        self.assertEqual(self.owner.algo.model_library.resolve(other["model_id"], "v000001")["name"], "PINTEST")
        self.assertTrue(all(path.read_bytes() == content for path, content in protected.items()))

    def test_new_same_name_creates_distinct_id_with_clear_prompt_and_success_message(self):
        original = self.publish(self.first_id)
        self.panel.reload_library()

        def choose_new(_parent, _title, _prompt, options, *_args):
            return options[0], True

        with patch.object(QtWidgets.QInputDialog, "getItem", side_effect=choose_new), \
             patch.object(QtWidgets.QInputDialog, "getText", return_value=("PINTEST", True)) as name, \
             patch.object(QtWidgets.QMessageBox, "information") as success:
            self.panel.publish()
        self.assertIn("new model ID", name.call_args.args[2])
        models = self.owner.algo.model_library.list_models()
        self.assertEqual(len(models), 2)
        created = next(model for model in models if model["model_id"] != original["model_id"])
        self.assertEqual((created["name"], created["version"]), ("PINTEST", "v000001"))
        self.assertIn(created["model_id"][:8], success.call_args.args[2])

    def test_old_model_without_provenance_remains_selectable(self):
        old = self.publish(self.first_id, provenance=False)
        self.panel.reload_library()
        self.select(old)
        self.assertIn("Source: -", self.panel.library_details.text())
        self.assertIn(old["model_id"], self.panel.library_details.text())
        self.assertTrue(self.panel.apply())
        self.assertEqual(self.owner.selected.library_model_id, old["model_id"])


if __name__ == "__main__":
    unittest.main()
