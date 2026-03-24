"""
product_session.py

产品与会话管理，纯文件 I/O，零 UI 依赖。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ProductPaths:
    product_dir: str
    session_json: str
    product_params_path: str
    inspection_items_path: str
    camera_settings_path: str
    line2dup_model_path: str
    line2dup_recipe_path: str

    @classmethod
    def build(cls, product_dir: str) -> "ProductPaths":
        from line2dup.core.locator import product_paths

        paths = product_paths(product_dir)
        return cls(
            product_dir=product_dir,
            session_json=os.path.join(product_dir, "session.json"),
            product_params_path=os.path.join(product_dir, "product_params.json"),
            inspection_items_path=os.path.join(product_dir, "inspection_items.json"),
            camera_settings_path=os.path.join(product_dir, "camera_settings.json"),
            line2dup_model_path=paths.model_path,
            line2dup_recipe_path=paths.recipe_path,
        )


@dataclass
class SessionData:
    ok_files: List[str] = field(default_factory=list)
    ng_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    ref_image: Optional[str] = None
    loc_method: str = "line2dup"
    runtime_cam1_serial: Optional[str] = None
    runtime_cam2_serial: Optional[str] = None
    runtime_capture_policy: str = "ng_only"


PRODUCT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\u4e00-\u9fa5]+$")


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
    def line2dup_model_path(self) -> str:
        return self.paths.line2dup_model_path if self.paths else ""

    @property
    def line2dup_recipe_path(self) -> str:
        return self.paths.line2dup_recipe_path if self.paths else ""

    def load(self) -> None:
        if os.path.exists(self.products_json):
            try:
                with open(self.products_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "products" in data:
                    self._products_data = data
            except Exception:
                pass
        self._remove_missing_product_entries()
        self.current_product = str(self._products_data.get("current_product", "Default"))
        self._ensure_current_product_in_list()

    def save_products(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.products_json, "w", encoding="utf-8") as f:
            json.dump(self._products_data, f, ensure_ascii=False, indent=2)

    def create_product(self, name: str) -> str:
        name = name.strip()
        if not name:
            return "产品名称不能为空"
        if not PRODUCT_NAME_RE.match(name):
            return "产品名称只能包含字母、数字、下划线和中文字符"
        if name in self._products_data["products"]:
            if self._product_dir_exists(name):
                return "产品名称已存在"
            self._remove_product_name(name)
        self._products_data["products"].append(name)
        self.save_products()
        return ""

    def switch_product(self, name: str) -> None:
        self.current_product = name
        self._products_data["current_product"] = name
        self._refresh_paths()

    def save_session(self, data: SessionData) -> None:
        if not self.session_json:
            return
        os.makedirs(self.product_dir, exist_ok=True)
        existing_payload: dict = {}
        if os.path.exists(self.session_json):
            try:
                with open(self.session_json, "r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
            except Exception:
                existing_payload = {}
        payload = {
            "ok_files": data.ok_files,
            "ng_files": data.ng_files,
            "test_files": data.test_files,
            "ref_image": data.ref_image or "",
            "loc_method": data.loc_method,
            "runtime_cam1_serial": (
                str(data.runtime_cam1_serial).strip()
                if data.runtime_cam1_serial is not None
                else str(existing_payload.get("runtime_cam1_serial", "")).strip()
            ),
            "runtime_cam2_serial": (
                str(data.runtime_cam2_serial).strip()
                if data.runtime_cam2_serial is not None
                else str(existing_payload.get("runtime_cam2_serial", "")).strip()
            ),
            "runtime_capture_policy": (
                str(data.runtime_capture_policy).strip()
                if data.runtime_capture_policy is not None
                else str(existing_payload.get("runtime_capture_policy", "ng_only")).strip()
            ),
        }
        with open(self.session_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_session(self) -> SessionData:
        raw: dict = {}
        if self.session_json and os.path.exists(self.session_json):
            try:
                with open(self.session_json, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                raw = {}

        def _filter(xs) -> List[str]:
            return [p for p in xs if isinstance(p, str) and os.path.exists(p)]

        ref = raw.get("ref_image", "")
        ref_image = ref if isinstance(ref, str) and os.path.exists(ref) else None

        loc_method = str(raw.get("loc_method", "line2dup")).strip() or "line2dup"
        if loc_method != "line2dup":
            loc_method = "line2dup"

        return SessionData(
            ok_files=_filter(raw.get("ok_files", [])),
            ng_files=_filter(raw.get("ng_files", [])),
            test_files=_filter(raw.get("test_files", [])),
            ref_image=ref_image,
            loc_method=loc_method,
            runtime_cam1_serial=str(raw.get("runtime_cam1_serial", "")).strip(),
            runtime_cam2_serial=str(raw.get("runtime_cam2_serial", "")).strip(),
            runtime_capture_policy=str(raw.get("runtime_capture_policy", "ng_only")).strip() or "ng_only",
        )

    def delete_session_file(self) -> None:
        try:
            if self.session_json and os.path.exists(self.session_json):
                os.remove(self.session_json)
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
