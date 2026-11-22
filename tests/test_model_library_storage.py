"""Move the default library under the session without changing public versions."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application.product_session import ProductSession
from common.app_paths import writable_embedding_test_root
from common.model_library import ModelLibrary
import common.model_library as model_library


class ModelLibraryStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.npz"
        self.source.write_bytes(b"trained model weights")
        self.legacy = ModelLibrary(self.root / "model_library")
        self.target = self.root / ".qr_session" / "model_library"

    def publish(self, **kwargs):
        return self.legacy.publish("PINTEST", "efficientnet_b0", model_file=self.source, **kwargs)

    def default_library(self):
        with patch.object(model_library, "writable_embedding_test_root", return_value=self.root):
            return ModelLibrary()

    def test_default_root_migrates_versions_preserving_ids_bytes_and_order_references(self):
        first = self.publish()
        self.source.write_bytes(b"updated model weights")
        second = self.publish(model_id=first["model_id"])
        order = self.root / ".qr_session" / "Order_A"
        order.mkdir(parents=True)
        binding = order / "inspection_items.json"
        binding.write_text(json.dumps([{"library_model_id": first["model_id"], "library_version": first["version"]}]), encoding="utf-8")
        protected = {path: path.read_bytes() for path in self.legacy.root.rglob("*") if path.is_file()}
        protected[binding] = binding.read_bytes()
        migrated = self.default_library()
        self.assertEqual(migrated.root, self.target)
        for publication in (first, second):
            loaded = migrated.resolve(publication["model_id"], publication["version"])
            self.assertEqual(Path(loaded["model_file"]).read_bytes(), Path(publication["model_file"]).read_bytes())
            self.assertEqual(Path(loaded["manifest_path"]).read_bytes(), Path(publication["manifest_path"]).read_bytes())
        self.assertTrue(all(path.read_bytes() == content for path, content in protected.items()))

    def test_merges_missing_versions_and_new_publications_only_use_new_root(self):
        first = self.publish()
        destination = self.target / first["model_id"] / first["version"]
        shutil.copytree(Path(first["directory"]), destination)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()}
        self.source.write_bytes(b"version two")
        second = self.publish(model_id=first["model_id"])
        migrated = self.default_library()
        self.assertEqual(len(migrated.list_models()), 2)
        self.default_library()
        self.assertTrue(all((path.read_bytes(), path.stat().st_mtime_ns) == state for path, state in before.items()))
        third = migrated.publish("PINTEST", "efficientnet_b0", model_id=first["model_id"], model_file=self.source)
        self.assertEqual(third["version"], "v000003")
        self.assertTrue(Path(third["directory"]).is_relative_to(self.target))
        self.assertFalse((self.legacy.root / first["model_id"] / "v000003").exists())
        self.assertEqual(migrated.resolve(second["model_id"], second["version"])["version"], "v000002")

    def test_conflicting_versions_are_rejected_without_overwriting_either_copy(self):
        old = self.publish()
        new = ModelLibrary(self.target).publish("Other content", "efficientnet_b0", model_id=old["model_id"], model_file=self.source)
        protected = {Path(model["manifest_path"]): Path(model["manifest_path"]).read_bytes() for model in (old, new)}
        with self.assertRaisesRegex(FileExistsError, "迁移冲突"):
            self.default_library()
        self.assertTrue(all(path.read_bytes() == content for path, content in protected.items()))

    def test_failed_copy_leaves_legacy_intact_and_no_partial_visible_version(self):
        published = self.publish()
        original_copy = model_library.shutil.copy2

        def fail_on_asset(source, destination):
            if Path(source).name == "model.npz":
                raise OSError("simulated interrupted copy")
            return original_copy(source, destination)

        with patch.object(model_library.shutil, "copy2", side_effect=fail_on_asset), self.assertRaises(OSError):
            self.default_library()
        self.assertEqual(Path(published["model_file"]).read_bytes(), b"trained model weights")
        self.assertFalse((self.target / published["model_id"] / published["version"]).exists())
        self.assertEqual(list(self.target.glob(".migrate-*")), [])
        self.assertTrue(self.default_library().resolve(published["model_id"], published["version"]))

    def test_legacy_publication_after_first_migration_is_imported_on_next_start(self):
        first = self.publish()
        self.default_library()
        second = self.publish(model_id=first["model_id"])
        self.assertEqual(self.default_library().resolve(second["model_id"], second["version"])["version"], "v000002")

    def test_invalid_legacy_version_is_not_activated_or_deleted(self):
        published = self.publish()
        Path(published["model_file"]).write_bytes(b"damaged model")
        migrated = self.default_library()
        self.assertEqual(migrated.list_models(), [])
        self.assertEqual(Path(published["model_file"]).read_bytes(), b"damaged model")

    def test_existing_order_at_destination_is_not_used_as_library(self):
        self.target.mkdir(parents=True)
        order_file = self.target / "inspection_items.json"
        order_file.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已有订单冲突"):
            self.default_library()
        self.assertEqual(order_file.read_text(encoding="utf-8"), "[]")

    def test_packaged_default_stays_beside_writable_orders(self):
        for has_subdirectory in (False, True):
            exe_root = self.root / ("with-app-subdirectory" if has_subdirectory else "exe-root")
            exe_root.mkdir()
            app_root = exe_root / "EmbeddingTest" if has_subdirectory else exe_root
            app_root.mkdir(exist_ok=True)
            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "executable", str(exe_root / "LCSystem.exe")), \
                 patch.object(model_library, "writable_embedding_test_root", new=writable_embedding_test_root):
                self.assertEqual(ModelLibrary().root, app_root / ".qr_session" / "model_library")


class ReservedSessionDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.session = ProductSession(str(self.root))
        self.session.load()
        self.session.switch_product("Default")
        self.library = self.root / "model_library"
        self.library.mkdir()
        self.asset = self.library / "keep-model.npz"
        self.asset.write_bytes(b"library model")

    def test_library_is_not_an_order_and_cannot_be_created_opened_copied_or_deleted(self):
        for name in ("model_library", "MODEL_LIBRARY", "_deleted",
                     "Default/../model_library", "model_library/" + "a" * 32):
            with self.subTest(name=name):
                self.assertTrue(self.session.create_product(name))
                self.assertTrue(self.session.copy_product("Default", name))
                self.assertTrue(self.session.copy_product(name, "Copy"))
                self.assertTrue(self.session.delete_product(name))
                with self.assertRaises(ValueError):
                    self.session.switch_product(name)
        self.assertEqual(self.session.current_product, "Default")
        self.assertEqual(self.asset.read_bytes(), b"library model")

    def test_reserved_entries_are_filtered_from_saved_order_list_and_current_selection(self):
        Path(self.session.products_json).write_text(json.dumps({
            "products": ["Default", "model_library", "MODEL_LIBRARY", "_deleted"],
            "current_product": "model_library",
        }), encoding="utf-8")
        self.session.load()
        self.assertEqual(self.session.product_names, ["Default"])
        self.assertEqual(self.session.current_product, "Default")

    def test_analysis_order_list_excludes_internal_directories(self):
        from ui.debug.embedding_analysis import list_product_names
        (self.root / "_deleted").mkdir()
        (self.root / "Order_A").mkdir()
        self.assertEqual(list_product_names(str(self.root)), ["Default", "Order_A"])

    def test_copying_an_order_preserves_its_reference_without_copying_the_library(self):
        self.assertEqual(self.session.create_product("Order_A"), "")
        self.session.switch_product("Order_A")
        reference = [{"library_model_id": "a" * 32, "library_version": "v000001"}]
        Path(self.session.inspection_items_path).write_text(json.dumps(reference), encoding="utf-8")
        self.assertEqual(self.session.copy_product("Order_A", "Order_B"), "")
        copied = self.root / "Order_B"
        self.assertEqual(json.loads((copied / "inspection_items.json").read_text(encoding="utf-8")), reference)
        self.assertFalse((copied / "model_library").exists())
        self.assertEqual(self.asset.read_bytes(), b"library model")


if __name__ == "__main__":
    unittest.main()
