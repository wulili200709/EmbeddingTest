"""Persistence and legacy compatibility for optional local ROI positioning groups."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from shape.core.recipe import (
    ShapeRecipe,
    has_bound_localization_groups,
    load_recipe,
    normalize_group_id,
    normalize_localization_groups,
    save_recipe,
)


def _region(label="roi2", group_id="upper"):
    return {
        "reference_label": label,
        "output_label": label,
        "display_name": "上排数字",
        "shape_type": "rectangle",
        "points": [[311.0, 924.0], [386.0, 1006.0]],
        "group_id": group_id,
    }


def _group():
    return {
        "id": "upper",
        "name": "上排定位",
        "points": [[300.0, 880.0], [420.0, 1040.0]],
        "search_radius": 64,
        "min_score": 0.65,
        "min_peak_margin": 0.03,
        "needs_rebuild": False,
    }


class LocalizationGroupRecipeTests(unittest.TestCase):
    def test_legacy_recipes_keep_global_following_without_local_groups(self):
        old = _region()
        del old["group_id"]
        recipe = ShapeRecipe.from_dict({
            "schema_version": 1,
            "backend": "fusionv2",
            "follow_mode": "match_bbox",
            "reference_regions": [old],
        })
        self.assertEqual(recipe.localization_groups, [])
        self.assertFalse(has_bound_localization_groups(recipe))
        self.assertEqual(recipe.reference_regions[0]["group_id"], "")
        self.assertEqual(recipe.reference_regions[0]["points"], old["points"])
        # Explicit reference ROIs already selected affine_roi before this change.
        self.assertEqual(recipe.follow_mode, "affine_roi")
        self.assertEqual(recipe.to_dict()["schema_version"], 1)
        bbox_only = ShapeRecipe.from_dict({"follow_mode": "match_bbox"})
        self.assertEqual(bbox_only.follow_mode, "match_bbox")
        self.assertIsNone(bbox_only.reference_regions)
        self.assertFalse(has_bound_localization_groups(bbox_only))

    def test_groups_and_bindings_survive_real_save_load_with_relative_paths(self):
        with tempfile.TemporaryDirectory(prefix="local-roi-recipe-") as directory:
            product = Path(directory) / "产品1125"
            destination = product / "shape" / "cam1" / "recipe.json"
            model = destination.with_name("model.json")
            reference = product / "samples" / "train" / "reference.png"
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"reference-path-fixture")
            recipe = ShapeRecipe(
                model_path=str(model), reference_image=str(reference),
                reference_regions=[_region(), _region("roi1", "global")],
                localization_groups=[_group()],
            )
            before = copy.deepcopy(recipe)
            save_recipe(recipe, str(destination))
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 2)
            self.assertFalse(Path(saved["model_path"]).is_absolute())
            self.assertFalse(Path(saved["reference_image"]).is_absolute())
            restored = load_recipe(str(destination))
            self.assertEqual(Path(restored.model_path).resolve(), model.resolve())
            self.assertEqual(Path(restored.reference_image).resolve(), reference.resolve())
            self.assertEqual(restored.localization_groups, [_group()])
            self.assertEqual(restored.reference_regions[0]["group_id"], "upper")
            self.assertEqual(restored.reference_regions[1]["group_id"], "")
            self.assertTrue(has_bound_localization_groups(restored))
            self.assertEqual(restored.to_dict(), ShapeRecipe.from_dict(restored.to_dict()).to_dict())
            # Serialization must not rename or mutate the editor's live objects.
            self.assertEqual(recipe.reference_regions, before.reference_regions)
            self.assertEqual(recipe.localization_groups, before.localization_groups)

    def test_dangling_and_stale_bindings_remain_explicit_editable_drafts(self):
        draft = _group()
        draft["points"] = []
        draft["needs_rebuild"] = True
        recipe = ShapeRecipe.from_dict({
            "reference_regions": [_region("roi2", "upper"), _region("roi3", "deleted_group")],
            "localization_groups": [draft],
        })
        restored = ShapeRecipe.from_dict(json.loads(json.dumps(recipe.to_dict(), allow_nan=False)))
        self.assertEqual(restored.localization_groups[0]["points"], [])
        self.assertTrue(restored.localization_groups[0]["needs_rebuild"])
        self.assertEqual(restored.reference_regions[1]["group_id"], "deleted_group")
        self.assertTrue(has_bound_localization_groups(restored))
        # Even with no definitions, consumers must enter the local-group failure
        # path instead of silently accepting a global-only or cached ROI result.
        dangling_only = ShapeRecipe(reference_regions=[_region(group_id="missing")])
        self.assertTrue(has_bound_localization_groups(dangling_only))
        self.assertEqual(dangling_only.to_dict()["schema_version"], 2)

    def test_unbound_drafts_do_not_activate_local_positioning(self):
        groups = normalize_localization_groups([{"id": "draft", "name": "待画锚点"}])
        self.assertEqual(groups[0]["points"], [])
        self.assertEqual(groups[0]["search_radius"], 64)
        self.assertEqual(groups[0]["min_score"], 0.65)
        self.assertEqual(groups[0]["min_peak_margin"], 0.03)
        self.assertFalse(groups[0]["needs_rebuild"])
        recipe = ShapeRecipe(localization_groups=groups, reference_regions=[_region(group_id="global")])
        self.assertFalse(has_bound_localization_groups(recipe))
        self.assertEqual(recipe.to_dict()["schema_version"], 2)
        self.assertEqual(normalize_group_id(" GLOBAL "), "")
        self.assertEqual(normalize_group_id(" upper "), "upper")

    def test_four_corner_rectangles_and_reversed_diagonals_preserve_geometry(self):
        four = _group()
        four["points"] = [[420, 1040], [300, 1040], [300, 880], [420, 880]]
        reverse = _group()
        reverse["points"] = [[420, 1040], [300, 880]]
        for group in (four, reverse):
            self.assertEqual(normalize_localization_groups([group])[0]["points"], _group()["points"])

    def test_unsupported_group_paths_are_not_persisted_or_used(self):
        group = _group()
        group.update(model_path="../../outside/model.json", reference_image="https://invalid.example/image.png")
        normalized = normalize_localization_groups([group])
        self.assertEqual(normalized, [_group()])
        self.assertEqual(normalize_localization_groups(None), [])

    def test_malformed_configuration_cannot_silently_become_global(self):
        invalid_changes = [
            {"id": "global"}, {"id": ""}, {"id": 3},
            {"search_radius": 0}, {"search_radius": 513}, {"search_radius": 3.5},
            {"search_radius": True}, {"min_score": float("nan")}, {"min_score": 1.1},
            {"min_peak_margin": -0.01}, {"needs_rebuild": "false"},
            {"points": [[0, 0], [float("inf"), 20]]},
            {"points": [[0, 0], [10, 1], [10, 10], [0, 10]]},
        ]
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                group = _group()
                group.update(changes)
                with self.assertRaises(ValueError):
                    ShapeRecipe.from_dict({"localization_groups": [group]})
        for groups in ({"upper": _group()}, [_group(), _group()], ["upper"]):
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    normalize_localization_groups(groups)
        for binding in (False, 1, [], {}):
            with self.subTest(binding=binding):
                with self.assertRaises(ValueError):
                    ShapeRecipe.from_dict({"reference_regions": [_region(group_id=binding)]})

    def test_future_schema_is_rejected_and_group_defaults_are_not_shared(self):
        with self.assertRaises(ValueError):
            ShapeRecipe.from_dict({"schema_version": 3})
        first, second = ShapeRecipe(), ShapeRecipe()
        first.localization_groups.append(_group())
        self.assertEqual(second.localization_groups, [])


if __name__ == "__main__":
    unittest.main()
