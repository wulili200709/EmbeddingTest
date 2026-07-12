"""Product/session persistence without UI dependencies."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from common.safe_io import atomic_write_json, backup_path_for, load_json_with_backup
from common.path_utils import (
    product_relative_path,
    resolve_existing_product_path,
    resolve_existing_product_paths,
)
from common.camera_roles import CAMERA_ROLES, DEFAULT_CAMERA_ROLE, configured_camera_roles


SUPPORTED_LOC_METHODS = {"shape", "ncc"}


def normalize_loc_method(method: object, *, default: str = "shape") -> str:
    value = str(method or "").strip().lower()
    if value == "line2dup":
        value = "shape"
    if value in SUPPORTED_LOC_METHODS:
        return value
    return default if default in SUPPORTED_LOC_METHODS else "shape"


@dataclass(frozen=True)
class ProductPaths:
    product_dir: str
    session_json: str
    product_params_path: str
    inspection_items_path: str
    camera_settings_path: str
    shape_model_path: str
    shape_recipe_path: str

    @classmethod
    def build(cls, product_dir: str) -> "ProductPaths":
        from shape.core.locator import product_paths

        paths = product_paths(product_dir, "cam1")
        return cls(
            product_dir=product_dir,
            session_json=os.path.join(product_dir, "session.json"),
            product_params_path=os.path.join(product_dir, "product_params.json"),
            inspection_items_path=os.path.join(product_dir, "inspection_items.json"),
            camera_settings_path=os.path.join(product_dir, "camera_settings.json"),
            shape_model_path=paths.model_path,
            shape_recipe_path=paths.recipe_path,
        )


@dataclass
class SessionData:
    train_files: List[str] = field(default_factory=list)
    ok_files: List[str] = field(default_factory=list)
    ng_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    ref_image: Optional[str] = None
    loc_method: str = "shape"
    loc_methods: Dict[str, str] = field(default_factory=dict)
    runtime_cam1_serial: Optional[str] = None
    runtime_cam2_serial: Optional[str] = None
    runtime_cam3_serial: Optional[str] = None
    runtime_camera_serials: Dict[str, str] = field(default_factory=dict)
    runtime_camera_roles: List[str] = field(default_factory=list)
    runtime_capture_policy: Optional[str] = None


PRODUCT_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class ProductSession:
    _DEFAULT_PRODUCTS: dict = {
        "products": ["Default"],
        "current_product": "Default",
    }

    def __init__(self, session_dir: str) -> None:
        self.session_dir: str = session_dir
        self.products_json: str = os.path.join(session_dir, "products.json")
        self._products_data: dict = dict(self._DEFAULT_PRODUCTS)
        self.current_product: str = "Default"
        self.paths: Optional[ProductPaths] = None

    @property
    def product_names(self) -> List[str]:
        return list(self._products_data.get("products", ["Default"]))

    @property
    def product_dir(self) -> str:
        return self.paths.product_dir if self.paths else ""

    @property
    def session_json(self) -> str:
        return self.paths.session_json if self.paths else ""

    @property
    def product_params_path(self) -> str:
        return self.paths.product_params_path if self.paths else ""

    @property
    def inspection_items_path(self) -> str:
        return self.paths.inspection_items_path if self.paths else ""

    @property
    def camera_settings_path(self) -> str:
        return self.paths.camera_settings_path if self.paths else ""

    @property
    def shape_model_path(self) -> str:
        return self.paths.shape_model_path if self.paths else ""

    @property
    def shape_recipe_path(self) -> str:
        return self.paths.shape_recipe_path if self.paths else ""

    def shape_paths_for_role(self, camera_role: str):
        if not self.product_dir:
            from shape.core.locator import product_paths

            return product_paths("", camera_role)
        from shape.core.locator import product_paths

        return product_paths(self.product_dir, camera_role)

    def shape_model_path_for_role(self, camera_role: str) -> str:
        return self.shape_paths_for_role(camera_role).model_path

    def shape_recipe_path_for_role(self, camera_role: str) -> str:
        return self.shape_paths_for_role(camera_role).recipe_path

    def load(self) -> None:
        data = load_json_with_backup(self.products_json, default={})
        if isinstance(data, dict) and "products" in data:
            self._products_data = data
        self._remove_missing_product_entries()
        self.current_product = str(self._products_data.get("current_product", "Default"))
        self._ensure_current_product_in_list()

    def save_products(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        atomic_write_json(self.products_json, self._products_data, ensure_ascii=False, indent=2)

    def create_product(self, name: str) -> str:
        name = name.strip()
        if not name:
            return "\u4ea7\u54c1\u7f16\u53f7\u4e0d\u80fd\u4e3a\u7a7a"
        if not PRODUCT_NAME_RE.match(name):
            return "\u4ea7\u54c1\u7f16\u53f7\u53ea\u80fd\u5305\u542b\u82f1\u6587\u3001\u6570\u5b57\u548c\u4e0b\u5212\u7ebf"
        if name in self._products_data["products"]:
            if self._product_dir_exists(name):
                return "\u4ea7\u54c1\u7f16\u53f7\u5df2\u5b58\u5728"
            self._remove_product_name(name)
        self._products_data["products"].append(name)
        self.save_products()
        return ""

    def delete_product(self, name: str) -> str:
        product_name = str(name or "").strip()
        if not product_name:
            return "\u8bf7\u5148\u9009\u62e9\u8981\u5220\u9664\u7684\u4ea7\u54c1"
        if product_name == "Default":
            return "Default \u4ea7\u54c1\u4e0d\u80fd\u5220\u9664"

        product_dir = os.path.abspath(os.path.join(self.session_dir, product_name))
        session_root = os.path.abspath(self.session_dir)
        if os.path.commonpath([session_root, product_dir]) != session_root:
            return "\u4ea7\u54c1\u8def\u5f84\u4e0d\u5728\u4f1a\u8bdd\u76ee\u5f55\u5185"

        if os.path.isdir(product_dir):
            deleted_root = os.path.join(self.session_dir, "_deleted")
            os.makedirs(deleted_root, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = os.path.join(deleted_root, f"{product_name}_{stamp}")
            suffix = 1
            while os.path.exists(target):
                suffix += 1
                target = os.path.join(deleted_root, f"{product_name}_{stamp}_{suffix}")
            try:
                shutil.move(product_dir, target)
            except Exception as exc:
                return f"\u79fb\u52a8\u4ea7\u54c1\u76ee\u5f55\u5931\u8d25: {exc}"

        self._remove_product_name(product_name)
        self.save_products()
        self._refresh_paths()
        return ""

    def switch_product(self, name: str) -> None:
        self.current_product = name
        self._products_data["current_product"] = name
        self._refresh_paths()

    def save_session(self, data: SessionData) -> None:
        if not self.session_json:
            return
        os.makedirs(self.product_dir, exist_ok=True)
        existing_payload = load_json_with_backup(self.session_json, default={})
        if not isinstance(existing_payload, dict):
            existing_payload = {}
        existing_serials = existing_payload.get("runtime_camera_serials", {})
        if not isinstance(existing_serials, dict):
            existing_serials = {}
        runtime_serials = {
            str(role).strip(): str(serial).strip()
            for role, serial in dict(existing_serials).items()
            if str(role).strip() and str(serial).strip()
        }
        runtime_serials.update(
            {
                str(role).strip(): str(serial).strip()
                for role, serial in dict(data.runtime_camera_serials or {}).items()
                if str(role).strip() and str(serial).strip()
            }
        )
        legacy_updates = {
            "cam1": data.runtime_cam1_serial,
            "cam2": data.runtime_cam2_serial,
            "cam3": data.runtime_cam3_serial,
        }
        for role, serial in legacy_updates.items():
            if serial is not None:
                serial_text = str(serial).strip()
                if serial_text:
                    runtime_serials[role] = serial_text
                else:
                    runtime_serials.pop(role, None)
        raw_roles = (
            data.runtime_camera_roles
            if data.runtime_camera_roles
            else existing_payload.get("runtime_camera_roles", [])
        )
        runtime_roles = configured_camera_roles(raw_roles) if raw_roles else []
        default_loc_method = normalize_loc_method(data.loc_method)
        existing_loc_methods = existing_payload.get("loc_methods", {})
        if not isinstance(existing_loc_methods, dict):
            existing_loc_methods = {}
        loc_methods = {
            str(role).strip(): normalize_loc_method(method, default=default_loc_method)
            for role, method in existing_loc_methods.items()
            if str(role).strip()
        }
        loc_methods.update(
            {
                str(role).strip(): normalize_loc_method(method, default=default_loc_method)
                for role, method in dict(data.loc_methods or {}).items()
                if str(role).strip()
            }
        )
        for role in CAMERA_ROLES:
            loc_methods.setdefault(role, default_loc_method)

        payload = {
            "train_files": [product_relative_path(path, base_dir=self.product_dir) for path in data.train_files],
            "ok_files": [product_relative_path(path, base_dir=self.product_dir) for path in data.ok_files],
            "ng_files": [product_relative_path(path, base_dir=self.product_dir) for path in data.ng_files],
            "test_files": [product_relative_path(path, base_dir=self.product_dir) for path in data.test_files],
            "ref_image": product_relative_path(data.ref_image or "", base_dir=self.product_dir),
            "loc_method": default_loc_method,
            "loc_methods": loc_methods,
            "runtime_cam1_serial": runtime_serials.get("cam1", ""),
            "runtime_cam2_serial": runtime_serials.get("cam2", ""),
            "runtime_cam3_serial": runtime_serials.get("cam3", ""),
            "runtime_camera_serials": runtime_serials,
            "runtime_camera_roles": runtime_roles,
            "runtime_capture_policy": (
                str(data.runtime_capture_policy).strip()
                if data.runtime_capture_policy is not None
                else str(existing_payload.get("runtime_capture_policy", "ng_only")).strip()
            ),
        }
        atomic_write_json(self.session_json, payload, ensure_ascii=False, indent=2)

    def load_session(self) -> SessionData:
        raw = load_json_with_backup(self.session_json, default={}) if self.session_json else {}
        if not isinstance(raw, dict):
            raw = {}

        def _filter(xs) -> List[str]:
            return resolve_existing_product_paths(xs, base_dir=self.product_dir, anchor_dir=self.product_dir)

        ref = raw.get("ref_image", "")
        ref_image = resolve_existing_product_path(ref, base_dir=self.product_dir, anchor_dir=self.product_dir)

        loc_method = normalize_loc_method(raw.get("loc_method", "shape"))
        raw_loc_methods = raw.get("loc_methods", {})
        if not isinstance(raw_loc_methods, dict):
            raw_loc_methods = {}
        loc_methods = {
            role: normalize_loc_method(raw_loc_methods.get(role, loc_method), default=loc_method)
            for role in CAMERA_ROLES
        }
        loc_method = loc_methods.get(DEFAULT_CAMERA_ROLE, loc_method)

        legacy_ok_files = _filter(raw.get("ok_files", []))
        legacy_ng_files = _filter(raw.get("ng_files", []))
        train_files = _filter(raw.get("train_files", []))
        if not train_files:
            train_files = list(dict.fromkeys(legacy_ok_files + legacy_ng_files))

        raw_serials = raw.get("runtime_camera_serials", {})
        if not isinstance(raw_serials, dict):
            raw_serials = {}
        runtime_serials = {
            str(role).strip(): str(serial).strip()
            for role, serial in raw_serials.items()
            if str(role).strip() and str(serial).strip()
        }
        for role, key in (
            ("cam1", "runtime_cam1_serial"),
            ("cam2", "runtime_cam2_serial"),
            ("cam3", "runtime_cam3_serial"),
        ):
            serial = str(raw.get(key, "")).strip()
            if serial:
                runtime_serials.setdefault(role, serial)

        return SessionData(
            train_files=train_files,
            ok_files=legacy_ok_files,
            ng_files=legacy_ng_files,
            test_files=_filter(raw.get("test_files", [])),
            ref_image=ref_image,
            loc_method=loc_method,
            loc_methods=loc_methods,
            runtime_cam1_serial=runtime_serials.get("cam1", ""),
            runtime_cam2_serial=runtime_serials.get("cam2", ""),
            runtime_cam3_serial=runtime_serials.get("cam3", ""),
            runtime_camera_serials=runtime_serials,
            runtime_camera_roles=(
                configured_camera_roles(raw.get("runtime_camera_roles", []) or [])
                if raw.get("runtime_camera_roles")
                else []
            ),
            runtime_capture_policy=str(raw.get("runtime_capture_policy", "ng_only")).strip() or "ng_only",
        )

    def delete_session_file(self) -> None:
        try:
            if self.session_json and os.path.exists(self.session_json):
                os.remove(self.session_json)
            backup_path = backup_path_for(self.session_json) if self.session_json else None
            if backup_path is not None and backup_path.exists():
                backup_path.unlink()
        except Exception:
            pass

    def _refresh_paths(self) -> None:
        product_dir = os.path.join(self.session_dir, self.current_product)
        os.makedirs(product_dir, exist_ok=True)
        self.paths = ProductPaths.build(product_dir)

    def _ensure_current_product_in_list(self) -> None:
        products = self._products_data.setdefault("products", ["Default"])
        if self.current_product not in products:
            products.append(self.current_product)

    def _product_dir_exists(self, name: str) -> bool:
        product_name = str(name or "").strip()
        if not product_name:
            return False
        return os.path.isdir(os.path.join(self.session_dir, product_name))

    def _remove_product_name(self, name: str) -> None:
        product_name = str(name or "").strip()
        if not product_name:
            return
        products = [
            item
            for item in self._products_data.setdefault("products", ["Default"])
            if str(item).strip() != product_name
        ]
        if not products:
            products = ["Default"]
        self._products_data["products"] = products
        if str(self._products_data.get("current_product", "")).strip() == product_name:
            self._products_data["current_product"] = products[0]
            self.current_product = products[0]

    def _remove_missing_product_entries(self) -> None:
        products = [
            str(item).strip()
            for item in self._products_data.get("products", ["Default"])
            if str(item).strip()
        ]
        kept: List[str] = []
        for name in products:
            if name == "Default" or self._product_dir_exists(name):
                kept.append(name)
        if not kept:
            kept = ["Default"]
        self._products_data["products"] = kept
        current = str(self._products_data.get("current_product", "Default")).strip() or "Default"
        if current not in kept:
            self._products_data["current_product"] = kept[0]


