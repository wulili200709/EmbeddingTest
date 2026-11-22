from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6 import QtCore, QtWidgets

from application import AlgorithmController, ProductSession
from common.model_library import ModelLibrary
from domain import InspectionItem, load_inspection_items, save_inspection_items
from ui.debug.tool_page.model_source_panel import ModelSourcePanel, binding_status, open_model_source_dialog
from ui.debug.tool_page.page import ToolPage
from ui.debug.tool_page.training_task_builder import TrainingTaskBuilder
from ui.i18n import language_code, LANG_EN, LANG_ZH, _TRANSLATIONS, set_language


class _Algo:
    def __init__(self, root: Path) -> None:
        self.model_library = ModelLibrary(root / "library")
        self.model = object()
        self.cache_clears = 0

    @staticmethod
    def is_learning_tool(code):
        return code == "shared_backbone_register"

    @staticmethod
    def is_measurement_tool(_code):
        return False

    @staticmethod
    def current_learning_backbone(_camera):
        return "b0"

    @staticmethod
    def resolve_tool_algorithm(code, _camera):
        return code

    def resolve_item_algorithm(self, item):
        if item.model_source == "library":
            return self.model_library.resolve(item.library_model_id, item.library_version)["algorithm"]
        return "b0" if self.is_learning_tool(item.algorithm_code) else item.algorithm_code

    def describe_model_binding(self, item, _product):
        if item.model_source == "library":
            manifest = self.model_library.resolve(item.library_model_id, item.library_version)
            return dict(manifest, ready=True, source="library", message="verified")
        return {"ready": True, "source": item.model_source, "message": "ready"}

    def clear_embedding_model_cache(self):
        self.cache_clears += 1


class _Owner(QtWidgets.QWidget):
    inspectionItemsChanged = QtCore.Signal()

    def __init__(self, root: Path, items: list[InspectionItem]) -> None:
        super().__init__()
        self.inspection_items = items
        self.selected = items[0]
        self.session = SimpleNamespace(product_dir=str(root), inspection_items_path=str(root / "inspection_items.json"))
        self.algo = _Algo(root)
        self.review_clears = []
        self.refreshes = 0
        self.train_files = ["good.png", "bad.png"]
        self.annotations = {("good.png", "roi1"): "OK", ("bad.png", "roi1"): "NG"}
        self.training_controller = SimpleNamespace(task_builder=TrainingTaskBuilder(self))

    def _selected_inspection_item(self):
        return self.selected

    def _persist_inspection_items(self):
        save_inspection_items(self.inspection_items, self.session.inspection_items_path)
        self.inspectionItemsChanged.emit()

    def _clear_training_roi_review_state(self, role):
        self.review_clears.append(role)

    def _refresh_inspection_items_table(self):
        self.refreshes += 1

    def _refresh_lists(self):
        pass

    def _update_runtime_widgets(self):
        pass

    def current_camera_role(self):
        return "cam1"

    def _filter_paths_for_camera(self, paths, _role):
        return paths

    def _path_has_roi_geometry(self, path, label):
        return (path, label) in self.annotations

    def _sample_roi_status_for_path(self, path, _role, label):
        return self.annotations.get((path, label), "")

    def _sample_annotation_counts_for_roi(self, label, _role, *, paths=None):
        values = [self.annotations.get((path, label), "") for path in (paths or self.train_files)]
        return values.count("OK"), values.count("NG"), values.count("")


class ModelSourceUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.addCleanup(set_language, language_code(), persist=False)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        set_language(LANG_EN, persist=False)
        self.owner = _Owner(self.root, [InspectionItem("roi1", "Pin 1", "cam1", "roi1", "meanintensity")])
        self.panel = ModelSourcePanel(self.owner, QtWidgets.QVBoxLayout(self.owner))

    def tearDown(self):
        self.owner.close()
        self.owner.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def _publish(self, *, name="Needle", model_id=None):
        return self.owner.algo.model_library.publish(
            name, "meanintensity", model_id=model_id,
            traditional_model={"algorithm": "meanintensity", "threshold": 80, "ok_when": "greater_equal"},
        )

    def test_group_selection_requires_apply_and_persists_model_binding(self):
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("group"))
        self.panel.group.setCurrentText("Needles")
        self.assertEqual(self.owner.selected.model_source, "independent")
        self.assertFalse(Path(self.owner.session.inspection_items_path).exists())

        self.panel.apply()

        item = load_inspection_items(self.owner.session.inspection_items_path)[0]
        self.assertEqual((item.model_source, item.task_group), ("group", "Needles"))
        self.assertEqual(self.owner.review_clears, ["cam1"])
        self.assertEqual(self.owner.algo.cache_clears, 1)
        self.assertIsNone(self.owner.algo.model)

    def test_library_reference_is_pinned_when_a_new_version_appears(self):
        first = self._publish()
        self.panel.reload_library()
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("library"))
        self.panel.library.setCurrentIndex(1)
        self.panel.apply()
        item = self.owner.selected
        self.assertEqual((item.library_model_id, item.library_version), (first["model_id"], first["version"]))
        self._publish(model_id=first["model_id"])
        self.panel.reload_library()
        self.assertEqual(self.panel.library.currentData(), (first["model_id"], first["version"]))
        self.assertFalse(self.panel.publish_button.isEnabled())
        self.assertIn("does not need training samples", ToolPage._current_tool_sample_stats_text(self.owner))
        self.assertIn("do not need training", ToolPage._training_validation_text(self.owner))

    def test_missing_library_version_shows_unavailable_without_sample_errors(self):
        manifest = self._publish()
        self.panel.reload_library()
        item = self.owner.selected
        item.model_source = "library"
        item.library_model_id = manifest["model_id"]
        item.library_version = "v999999"
        self.panel.refresh()
        status, _tooltip, _color = binding_status(self.owner, item)
        self.assertIn("not ready", status)
        self.assertIn("unavailable", ToolPage._training_validation_text(self.owner))
        self.assertGreaterEqual(self.panel.library.count(), 3)
        self.panel.library.setCurrentIndex(1)
        self.panel.apply()
        self.assertEqual(item.library_version, manifest["version"])
        self.assertIn("do not need training", ToolPage._training_validation_text(self.owner))

    def test_group_validation_uses_other_roi_samples(self):
        first = self.owner.selected
        first.model_source = "group"
        first.task_group = "Needles"
        second = InspectionItem("roi2", "Pin 2", "cam1", "roi2", "meanintensity", model_source="group", task_group="Needles")
        self.owner.inspection_items.append(second)
        self.owner.selected = second
        self.panel.refresh()
        self.assertIn("Needles is ready", ToolPage._training_validation_text(self.owner))
        summary = ToolPage._current_tool_sample_stats_text(self.owner)
        self.assertIn("OK 1", summary)
        self.assertIn("NG 1", summary)

    def test_group_stays_selectable_after_all_members_switch_sources_and_page_reloads(self):
        first = self.owner.selected
        first.model_source, first.task_group = "group", "PIN"
        second = InspectionItem("roi2", "Pin 2", "cam1", "roi2", "meanintensity",
                                model_source="group", task_group="PIN")
        self.owner.inspection_items.append(second)
        save_inspection_items(self.owner.inspection_items, self.owner.session.inspection_items_path)
        key = first.effective_model_key
        for item in (first, second):
            self.owner.selected = item
            self.panel.refresh()
            self.panel.mode.setCurrentIndex(self.panel.mode.findData("independent"))
            self.assertTrue(self.panel.apply())
        self._publish()
        self.panel.reload_library()
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("library"))
        self.panel.library.setCurrentIndex(1)
        self.assertTrue(self.panel.apply())
        # Reload bindings and create a new panel, as reopening an order would.
        self.owner.inspection_items = load_inspection_items(self.owner.session.inspection_items_path)
        self.owner.selected = self.owner.inspection_items[0]
        panel = ModelSourcePanel(self.owner, self.owner.layout())
        try:
            self.assertEqual(panel.group.count(), 1)
            self.assertEqual(panel.group.itemText(0), "PIN")
            panel.mode.setCurrentIndex(panel.mode.findData("group"))
            self.assertEqual(panel.group.currentText(), "PIN")
            self.assertEqual(self.owner.selected.model_source, "independent")
            self.assertTrue(panel.apply())
            self.assertEqual(self.owner.selected.effective_model_key, key)
            self.assertEqual(self.owner.inspection_items[1].model_source, "library")
        finally:
            panel.frame.deleteLater()

    def test_old_group_file_can_be_selected_loaded_and_used_without_retraining(self):
        import numpy as np
        from algorithms.embedding import RegisterModel, save_register_model_npz
        from algorithms.embedding_core import compute_prototypes
        from application.runtime_context import _predict_learning_items_batch_rows

        self.owner.algo = AlgorithmController()
        self.owner.algo.model_library = ModelLibrary(self.root / "library")
        self.owner.algo.set_learning_backbone("efficientnet_b0", "cam1")
        self.owner.algo.product_params.score_mode = "proto"
        self.owner.algo.product_params.margin = .02
        self.owner.algo.product_params.topk = 1
        self.owner.inspection_items = [
            InspectionItem(f"roi{i}", f"Pin {i}", "cam1", f"roi{i}") for i in (1, 2)
        ]
        self.owner.selected = self.owner.inspection_items[0]
        save_inspection_items(self.owner.inspection_items, self.owner.session.inspection_items_path)
        old = InspectionItem("old_roi", "Old pin", "cam1", "old_roi", model_source="group", task_group="PIN")
        path = Path(self.owner.algo.embedding_model_path("efficientnet_b0", str(self.root), model_key=old.effective_model_key))
        ok, ng = np.eye(4, dtype=np.float32)[:2]
        ok_bank, ng_bank = np.stack([ok]), np.stack([ng])
        ok_proto, ng_proto = compute_prototypes(ok_bank, ng_bank)
        save_register_model_npz(RegisterModel(
            backbone="efficientnet_b0", score_mode="proto", margin=.02, topk=1,
            label_name="old_roi", ok_proto=ok_proto, ng_proto=ng_proto,
            ok_bank=ok_bank, ng_bank=ng_bank,
        ), str(path))
        before = path.read_bytes()
        self.assertFalse((self.root / "model_groups.json").exists())
        with patch.object(self.owner.algo, "train") as train:
            for item in self.owner.inspection_items:
                self.owner.selected = item
                self.panel.refresh()
                self.panel.mode.setCurrentIndex(self.panel.mode.findData("group"))
                self.assertEqual(self.panel.group.currentText(), "PIN")
                self.assertTrue(self.panel.apply())
                self.assertEqual(item.effective_model_key, old.effective_model_key)
                self.assertTrue(self.owner.algo.describe_model_binding(item, str(self.root))["ready"])
            loader = lambda algorithm, model_key: self.owner.algo.load_model_for_algorithm(
                algorithm, str(self.root), model_key=model_key)
            with patch("application.runtime_context.qr_core.embed_batch", return_value=np.stack([ok, ng])) as embed:
                rows = _predict_learning_items_batch_rows(
                    path="current-image.png", items=self.owner.inspection_items, match_ms=0.,
                    algo=self.owner.algo, load_embedding_model=loader, feat_net=object())
            train.assert_not_called()
        self.assertEqual([rows[item.model_key]["pred"] for item in self.owner.inspection_items], ["OK", "NG"])
        self.assertEqual(embed.call_args.args[2], ["roi1", "roi2"])
        self.assertEqual(path.read_bytes(), before)

    def test_library_failure_does_not_modify_original_binding(self):
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("library"))
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.panel.apply()
        warning.assert_called_once()
        self.assertEqual(self.owner.selected.model_source, "independent")

    def test_real_tool_page_can_select_and_rebind_missing_public_model(self):
        manifest = self._publish()
        session = ProductSession(str(self.root / "preview_session"))
        session.load()
        session.switch_product("Default")
        algo = AlgorithmController()
        algo.model_library = self.owner.algo.model_library
        page = ToolPage(session, algo, lite_mode=True)
        try:
            item = InspectionItem(
                "missing", "Missing reference", "cam1", "roi3", "meanintensity",
                model_source="library", library_model_id=manifest["model_id"], library_version="v999999",
            )
            page.inspection_items = [item]
            page._refresh_inspection_items_table()
            page.inspection_items_table.selectRow(0)
            page.model_source_panel.reload_library()
            self.assertFalse(page.btn_train_current.isEnabled())
            self.assertIn("not ready", page.model_source_panel.status.text())
            page.model_source_panel.library.setCurrentIndex(1)
            page.model_source_panel.apply()
            self.assertEqual(item.library_version, manifest["version"])
            self.assertIn("do not need training", page._training_validation_text())
            self.assertFalse(page.btn_train_current.isEnabled())
        finally:
            page.close()
            page.deleteLater()
            self.app.processEvents()

    def _legacy_page(self):
        session = ProductSession(str(self.root / "old_order"))
        session.load()
        session.switch_product("Default")
        legacy_items = [
            {"item_id": f"roi{index}", "display_name": f"Legacy ROI {index}", "camera_id": "cam1", "roi_label": f"roi{index}", "algorithm_code": "meanintensity", "enabled": True, "params": {}}
            for index in range(1, 6)
        ]
        Path(session.inspection_items_path).write_text(json.dumps(legacy_items), encoding="utf-8")
        algo = AlgorithmController()
        algo.model_library = self.owner.algo.model_library
        page = ToolPage(session, algo, lite_mode=True)
        page.inspection_items = load_inspection_items(session.inspection_items_path)
        page._refresh_inspection_items_table()
        page.inspection_items_table.selectRow(0)
        return page

    def test_legacy_order_header_opens_dialog_and_applies_group_without_scrolling(self):
        page = self._legacy_page()
        observations = {}
        try:
            page.resize(1100, 760)
            page.show()
            self.app.processEvents()
            scroll = page.tool_config_scroll.verticalScrollBar()
            scroll.setValue(0)
            selected_row = page.inspection_items_table.currentRow()
            def use_dialog():
                dialog = self.app.activeModalWidget()
                observations["dialog"] = isinstance(dialog, QtWidgets.QDialog)
                if dialog is None:
                    return
                try:
                    observations["identity"] = dialog.findChild(QtWidgets.QLabel, "model_source_current_item").text()
                    panel = dialog.model_source_panel
                    observations["initial"] = panel.mode.currentData()
                    panel.mode.setCurrentIndex(panel.mode.findData("group"))
                    panel.group.setCurrentText("Legacy Shared Parts")
                    observations["unapplied"] = page.inspection_items[0].model_source
                    panel.apply()
                finally:
                    dialog.reject()
            QtCore.QTimer.singleShot(0, use_dialog)
            page.btn_model_source.click()
            self.assertTrue(observations["dialog"])
            self.assertIn("Legacy ROI 1", observations["identity"])
            self.assertIn("cam1", observations["identity"])
            self.assertEqual(observations["initial"], "independent")
            self.assertEqual(observations["unapplied"], "independent")
            self.assertEqual(page.inspection_items[0].task_group, "Legacy Shared Parts")
            self.assertEqual(load_inspection_items(page.session.inspection_items_path)[0].model_source, "group")
            self.assertEqual(page.inspection_items_table.currentRow(), selected_row)
            self.assertEqual(scroll.value(), 0)
        finally:
            page.close()
            page.deleteLater()
            self.app.processEvents()

    def test_legacy_order_entry_is_visible_in_small_window_and_close_does_not_apply(self):
        page = self._legacy_page()
        observations = {}
        try:
            page.resize(1100, 760)
            page.show()
            self.app.processEvents()
            button = page.btn_model_source
            page.sidebar_config_scroll.ensureWidgetVisible(button, 0, 4)
            self.app.processEvents()
            self.assertTrue(button.isVisible())
            self.assertFalse(button.visibleRegion().isEmpty())
            self.assertTrue(button.isEnabled())
            self.assertIsNot(button.parent(), page.tool_config_scroll.widget())
            self.assertTrue(page.rect().contains(button.mapTo(page, QtCore.QPoint(0, 0))))
            self.assertTrue(page.rect().contains(button.mapTo(page, button.rect().bottomRight())))
            self.assertLessEqual(button.mapTo(page, button.rect().bottomRight()).y(), 760)
            self.assertLessEqual(button.width(), 200)
            self.assertGreaterEqual(page.inspection_items_table.minimumHeight(), 170)
            self.assertLessEqual(abs(button.mapTo(page, button.rect().center()).y() - page.btn_toggle_tools.mapTo(page, page.btn_toggle_tools.rect().center()).y()), 1)
            def edit_then_close():
                dialog = self.app.activeModalWidget()
                if dialog is None:
                    return
                try:
                    dialog.model_source_panel.mode.setCurrentIndex(dialog.model_source_panel.mode.findData("group"))
                    dialog.model_source_panel.group.setCurrentText("Do not save")
                    observations["opened"] = True
                finally:
                    dialog.reject()
            QtCore.QTimer.singleShot(0, edit_then_close)
            button.click()
            self.assertTrue(observations["opened"])
            self.assertEqual(page.inspection_items[0].model_source, "independent")
            self.assertNotIn("model_source", json.loads(Path(page.session.inspection_items_path).read_text(encoding="utf-8"))[0])
        finally:
            page.close()
            page.deleteLater()
            self.app.processEvents()

    def test_dialog_without_selection_explains_what_to_select(self):
        self.owner.selected = None
        with patch.object(QtWidgets.QMessageBox, "information") as message:
            open_model_source_dialog(self.owner)
        self.assertIn("Select an inspection item", message.call_args.args[2])

    def test_measurement_item_explains_that_no_model_is_required(self):
        self.owner.algo.is_measurement_tool = lambda _code: True
        with patch.object(QtWidgets.QMessageBox, "information") as message:
            open_model_source_dialog(self.owner)
        self.assertIn("not require training", message.call_args.args[2])

    def test_model_source_translations_have_matching_placeholders(self):
        from string import Formatter
        keys = {key for key in _TRANSLATIONS[LANG_ZH] if key.startswith("debug.model_source.")}
        self.assertEqual(keys, {key for key in _TRANSLATIONS[LANG_EN] if key.startswith("debug.model_source.")})
        for key in keys:
            fields = lambda text: {name for _literal, name, _format, _conversion in Formatter().parse(text) if name}
            self.assertEqual(fields(_TRANSLATIONS[LANG_ZH][key]), fields(_TRANSLATIONS[LANG_EN][key]), key)


if __name__ == "__main__":
    unittest.main()
