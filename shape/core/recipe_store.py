from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .recipe import ShapeRecipe, load_recipe, save_recipe


class ShapePathSet(Protocol):
    role_dir: str
    model_path: str
    recipe_path: str
    legacy_model_path: str
    legacy_recipe_path: str


@dataclass
class ShapeRecipeStore:
    paths: ShapePathSet

    def resolved_recipe_path(self) -> str:
        if os.path.exists(self.paths.recipe_path):
            return self.paths.recipe_path
        if os.path.exists(self.paths.legacy_recipe_path):
            return self.paths.legacy_recipe_path
        return self.paths.recipe_path

    def resolved_model_path(self) -> str:
        if os.path.exists(self.paths.model_path):
            return self.paths.model_path
        if os.path.exists(self.paths.legacy_model_path):
            return self.paths.legacy_model_path
        return self.paths.model_path

    def load(self) -> ShapeRecipe:
        recipe = load_recipe(self.resolved_recipe_path())
        recipe.model_path = self.resolved_model_path()
        return recipe

    def save(self, recipe: ShapeRecipe) -> None:
        os.makedirs(self.paths.role_dir, exist_ok=True)
        recipe.model_path = self.paths.model_path
        save_recipe(recipe, self.paths.recipe_path)

    @staticmethod
    def has_explicit_reference_regions(recipe: ShapeRecipe) -> bool:
        return recipe.reference_regions is not None
