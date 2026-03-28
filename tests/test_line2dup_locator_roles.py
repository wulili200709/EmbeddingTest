from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from line2dup.core.locator import (
    load_recipe_for_product,
    product_paths,
    resolved_model_path_for_product,
    resolved_recipe_path_for_product,
    save_recipe_for_product,
)
from line2dup.core.recipe import Line2DupRecipe, save_recipe


class Line2DupLocatorRolesTest(unittest.TestCase):
    def test_role_paths_are_split_under_line2dup_role_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = str(Path(tmp) / "product")
            paths = product_paths(product_dir, "cam2")
            self.assertTrue(paths.model_path.endswith("line2dup\\cam2\\model.json"))
            self.assertTrue(paths.recipe_path.endswith("line2dup\\cam2\\recipe.json"))

    def test_save_and_load_recipe_use_role_specific_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            ref_path = product_dir / "debug_capture" / "cam2_ref.png"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_bytes(b"png")
            recipe = Line2DupRecipe(reference_image=str(ref_path), class_id="demo")

            save_recipe_for_product(str(product_dir), recipe, "cam2")

            recipe_json = json.loads((product_dir / "line2dup" / "cam2" / "recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(recipe_json["model_path"], "model.json")
            self.assertEqual(recipe_json["reference_image"], "../../debug_capture/cam2_ref.png")

            loaded = load_recipe_for_product(str(product_dir), "cam2")
            self.assertEqual(loaded.reference_image, str(ref_path))
            self.assertEqual(loaded.class_id, "demo")
            self.assertEqual(loaded.model_path, product_paths(str(product_dir), "cam2").model_path)

    def test_role_load_falls_back_to_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            product_dir.mkdir(parents=True, exist_ok=True)
            paths = product_paths(str(product_dir), "cam2")
            ref_path = product_dir / "debug_capture" / "legacy_ref.png"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_bytes(b"png")
            legacy_recipe = Line2DupRecipe(reference_image=str(ref_path), class_id="legacy")
            legacy_recipe.model_path = paths.legacy_model_path
            Path(paths.legacy_model_path).write_text("{}", encoding="utf-8")
            save_recipe(legacy_recipe, paths.legacy_recipe_path)

            loaded = load_recipe_for_product(str(product_dir), "cam2")

            self.assertEqual(loaded.reference_image, str(ref_path))
            self.assertEqual(resolved_recipe_path_for_product(str(product_dir), "cam2"), paths.legacy_recipe_path)
            self.assertEqual(resolved_model_path_for_product(str(product_dir), "cam2"), paths.legacy_model_path)
            self.assertEqual(loaded.model_path, paths.legacy_model_path)

    def test_load_recipe_rebases_stale_absolute_paths_after_product_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            ref_path = product_dir / "debug_capture" / "moved_ref.png"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_bytes(b"png")
            paths = product_paths(str(product_dir), "cam1")
            Path(paths.role_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.model_path).write_text("{}", encoding="utf-8")
            Path(paths.recipe_path).write_text(
                json.dumps(
                    {
                        "model_path": r"E:\old_root\product\line2dup\cam1\model.json",
                        "reference_image": r"E:\old_root\product\debug_capture\moved_ref.png",
                        "class_id": "demo",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            loaded = load_recipe_for_product(str(product_dir), "cam1")

            self.assertEqual(loaded.model_path, paths.model_path)
            self.assertEqual(loaded.reference_image, str(ref_path))


if __name__ == "__main__":
    unittest.main()
