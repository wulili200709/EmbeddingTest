"""Order groups must outlive their last ROI reference, including old orders."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from common.model_groups import OrderModelGroups, make_group_model_key
from domain import InspectionItem, load_inspection_items, save_inspection_items


class OrderModelGroupsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "order"
        self.root.mkdir()
        self.items_path = self.root / "inspection_items.json"

    def write_json(self, name, data):
        (self.root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_last_group_member_can_leave_without_losing_original_name_or_model(self):
        item = InspectionItem("roi1", "Pin", "cam1", "roi1", model_source="group", task_group="针 A/B")
        original_key = item.effective_model_key
        save_inspection_items([item], str(self.items_path))
        model_path = self.root / f"{original_key}_register_model_b2.npz"
        model_path.write_bytes(b"trained weights")
        item.model_source, item.task_group = "independent", ""
        save_inspection_items([item], str(self.items_path))
        # A subsequent save also replaces the .bak that once held the binding.
        save_inspection_items([item], str(self.items_path))
        groups = OrderModelGroups(self.root).list_groups("cam1")
        self.assertEqual([(row["task_group"], row["model_key"]) for row in groups], [("针 A/B", original_key)])
        self.assertEqual(model_path.read_bytes(), b"trained weights")
        self.assertEqual(load_inspection_items(str(self.items_path))[0].model_source, "independent")

    def test_first_save_after_upgrade_remembers_old_binding_before_overwriting_it(self):
        old = InspectionItem("roi1", "Pin", "cam1", "roi1", model_source="group", task_group="PIN / type A")
        self.write_json("inspection_items.json", [old.to_dict()])
        independent = InspectionItem("roi1", "Pin", "cam1", "roi1")
        save_inspection_items([independent], str(self.items_path))
        save_inspection_items([], str(self.items_path))
        self.assertEqual(OrderModelGroups(self.root).list_groups("cam1")[0]["task_group"], "PIN / type A")

    def test_old_embedding_files_restore_groups_without_changing_active_bindings_or_weights(self):
        active = InspectionItem("roi1", "Pin", "cam1", "roi1", model_source="library",
                                library_model_id="a" * 32, library_version="v000001")
        self.write_json("inspection_items.json", [active.to_dict()])
        snapshots = {self.items_path: self.items_path.read_bytes()}
        for camera, group, backbone in [("cam1", "PIN", "b0"), ("cam1", "PIN", "b2"),
                                        ("cam1", "Small pins", "b2"), ("cam2", "Other", "b2")]:
            path = self.root / f"{make_group_model_key(camera, group)}_register_model_{backbone}.npz"
            path.write_bytes(f"saved {group} {backbone}".encode())
            snapshots[path] = path.read_bytes()
        groups = OrderModelGroups(self.root).list_groups("cam1")
        self.assertEqual([row["task_group"] for row in groups], ["PIN", "Small pins"])
        self.assertTrue(all(path.read_bytes() == content for path, content in snapshots.items()))
        saved = OrderModelGroups(self.root).path
        timestamp = saved.stat().st_mtime_ns
        OrderModelGroups(self.root).list_groups("cam1")
        self.assertEqual(saved.stat().st_mtime_ns, timestamp)
        self.assertFalse(saved.with_suffix(".json.bak").exists())

    def test_old_traditional_threshold_remains_available_after_group_recovery(self):
        key = make_group_model_key("cam3", "PIN")
        params = {"traditional_models": {f"meanintensity::{key}": {"threshold": 80., "ok_when": "greater_equal"}}}
        self.write_json("product_params.json", params)
        before = (self.root / "product_params.json").read_bytes()
        self.assertEqual(OrderModelGroups(self.root).list_groups("cam3")[0]["model_key"], key)
        self.assertEqual(OrderModelGroups(self.root).list_groups("cam1"), [])
        self.assertEqual((self.root / "product_params.json").read_bytes(), before)

    def test_backup_and_publication_metadata_restore_names_with_punctuation(self):
        old = InspectionItem("roi1", "Pin", "cam1", "roi1", model_source="group", task_group="针 A/B")
        self.write_json("inspection_items.json", [])
        self.write_json("inspection_items.json.bak", [old.to_dict()])
        key = make_group_model_key("cam1", "Pins @ upper")
        publications = [{"source": {"order": self.root.name, "group": "Pins @ upper", "model_key": key}},
                        {"source": {"order": "another-order", "group": "Wrong", "model_key": make_group_model_key("cam1", "Wrong")}},
                        {"source": {"order": self.root.name, "group": "Wrong label", "model_key": make_group_model_key("cam1", "Wrong key")}}]
        groups = OrderModelGroups(self.root).list_groups("cam1", public_models=publications)
        self.assertEqual({row["task_group"] for row in groups}, {"针 A/B", "Pins @ upper"})
        self.assertEqual(json.loads(self.items_path.read_text(encoding="utf-8")), [])

    def test_unverifiable_filenames_are_not_bound_to_guessed_names(self):
        (self.root / "cam1__group__PIN_0000000000_register_model_b2.npz").write_bytes(b"unknown")
        (self.root / "cam1__roi1_register_model_b2.npz").write_bytes(b"independent")
        self.assertEqual(OrderModelGroups(self.root).list_groups("cam1"), [])
        self.assertFalse(OrderModelGroups(self.root).path.exists())

    @unittest.skipUnless(os.name == "nt", "Windows order paths are case insensitive")
    def test_publication_source_matches_windows_order_name_casing(self):
        name = "Pins / upper"
        key = make_group_model_key("cam1", name)
        publication = {"source": {"order": self.root.name.upper(), "group": name, "model_key": key}}
        groups = OrderModelGroups(self.root).list_groups("cam1", public_models=[publication])
        self.assertEqual([row["task_group"] for row in groups], [name])


if __name__ == "__main__":
    unittest.main()
