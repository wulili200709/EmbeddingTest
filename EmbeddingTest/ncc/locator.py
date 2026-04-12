from __future__ import annotations

import os
from dataclasses import dataclass

from .model import NccMatchModel, load_model, save_model


@dataclass(frozen=True)
class ProductNccPaths:
    product_dir: str
    camera_role: str
    role_dir: str
    model_path: str
    legacy_model_path: str


def _normalize_camera_role(camera_role: str) -> str:
    role = str(camera_role or "").strip().lower()
    return role if role in {"cam1", "cam2"} else "cam1"


def product_paths(product_dir: str, camera_role: str = "cam1") -> ProductNccPaths:
    normalized_role = _normalize_camera_role(camera_role)
    role_dir = os.path.join(product_dir, "ncc", normalized_role)
    return ProductNccPaths(
        product_dir=product_dir,
        camera_role=normalized_role,
        role_dir=role_dir,
        model_path=os.path.join(role_dir, "model.json"),
        legacy_model_path=os.path.join(product_dir, "ncc_model.json"),
    )


def resolved_model_path_for_product(product_dir: str, camera_role: str = "cam1") -> str:
    paths = product_paths(product_dir, camera_role)
    if os.path.exists(paths.model_path):
        return paths.model_path
    if os.path.exists(paths.legacy_model_path):
        return paths.legacy_model_path
    return paths.model_path


def load_model_for_product(product_dir: str, camera_role: str = "cam1") -> NccMatchModel:
    return load_model(resolved_model_path_for_product(product_dir, camera_role))


def save_model_for_product(product_dir: str, model: NccMatchModel, camera_role: str = "cam1") -> str:
    paths = product_paths(product_dir, camera_role)
    os.makedirs(paths.role_dir, exist_ok=True)
    save_model(paths.model_path, model)
    return paths.model_path


__all__ = [
    "ProductNccPaths",
    "load_model_for_product",
    "product_paths",
    "resolved_model_path_for_product",
    "save_model_for_product",
]
