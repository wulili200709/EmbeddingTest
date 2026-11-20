from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.inspection_items import InspectionItem, load_inspection_items, save_inspection_items, sync_items_with_labels


class ModelBindingDataTests(unittest.TestCase):
    def test_legacy_items_keep_their_independent_model_key(self) -> None:
        item = InspectionItem.from_dict({"item_id": "roi1", "roi_label": "roi1", "task_group": "针组"})
        self.assertEqual(item.model_source, "independent")
        self.assertEqual(item.effective_model_key, "cam1__roi1")
        self.assertEqual(InspectionItem("针1", "针1", "cam1", "针1").model_key, "cam1__1")

    def test_chinese_groups_have_distinct_model_keys(self) -> None:
        first = InspectionItem("roi1", "第一根针", "cam1", "roi1", model_source="group", task_group="针 A")
        second = InspectionItem("roi2", "第二根针", "cam1", "roi2", model_source="group", task_group="针 A")
        other = InspectionItem("roi3", "另一种针", "cam1", "roi3", model_source="group", task_group="针 B")
        self.assertTrue(first.effective_model_key.startswith("cam1__group__针_A_"))
        self.assertEqual(first.effective_model_key, second.effective_model_key)
        self.assertNotEqual(first.effective_model_key, other.effective_model_key)
        self.assertNotEqual(first.model_key, second.model_key)
        colliding_slug = InspectionItem("roi4", "同形文件名", "cam1", "roi4", model_source="group", task_group="针_A")
        self.assertNotEqual(first.effective_model_key, colliding_slug.effective_model_key)

    def test_library_and_group_bindings_survive_save_load_and_roi_sync(self) -> None:
        items = [
            InspectionItem("roi1", "本地组", "cam1", "roi1", model_source="group", task_group="针组"),
            InspectionItem("roi2", "公共库", "cam1", "roi2", model_source="library", library_model_id="a" * 32, library_version="v000002"),
            InspectionItem("roi3", "独立", "cam1", "roi3", model_source="independent", task_group="旧自动分组"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "inspection_items.json")
            save_inspection_items(items, path)
            loaded = load_inspection_items(path)
            synced = sync_items_with_labels(loaded, ["roi2", "roi1", "roi3", "roi4"], display_names_by_label={"roi2": "新名称"})
        by_label = {item.roi_label: item for item in synced}
        self.assertEqual(by_label["roi1"].effective_model_key, items[0].effective_model_key)
        self.assertEqual(by_label["roi2"].effective_model_key, f"cam1__library__{'a' * 32}__v000002")
        self.assertEqual(by_label["roi2"].display_name, "新名称")
        self.assertEqual(by_label["roi3"].model_source, "independent")
        self.assertEqual(by_label["roi3"].effective_model_key, "cam1__roi3")
        self.assertEqual(by_label["roi4"].model_source, "independent")

    def test_invalid_binding_never_silently_becomes_independent(self) -> None:
        invalid = [
            {"model_source": "other"},
            {"model_source": "group", "task_group": ""},
            {"model_source": "group", "task_group": "needle__library__reserved"},
            {"model_source": "library", "library_model_id": "../other", "library_version": "v000001"},
            {"model_source": "library", "library_model_id": "a" * 32, "library_version": "latest"},
        ]
        for binding in invalid:
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                InspectionItem.from_dict({"roi_label": "roi1", **binding})

    def test_serialization_revalidates_mutated_binding(self) -> None:
        item = InspectionItem("roi1", "roi1", "cam1", "roi1")
        item.model_source = "library"
        with self.assertRaises(ValueError):
            item.to_dict()


if __name__ == "__main__":
    unittest.main()
