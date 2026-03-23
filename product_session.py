"""
product_session.py

产品与会话管理，纯文件 I/O，零 UI 依赖。

职责：
  - 维护产品列表（products.json）
  - 计算并持有当前产品下所有路径
  - 读写 session.json（ok/ng/test 文件列表、参考图、定位方式）
  - 提供「新建产品 / 切换产品 / 清空会话」的数据层操作

不负责：
  - 任何 Qt Widget 操作
  - 模型加载 / 算法参数
  - 运行链路状态
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# 产品路径集合（只读快照）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductPaths:
    product_dir: str
    session_json: str
    product_params_path: str
    inspection_items_path: str
    shape_model_path: str
    line2dup_model_path: str
    line2dup_recipe_path: str

    @classmethod
    def build(cls, product_dir: str) -> "ProductPaths":
        import line2dup_locator  # 运行时延迟导入，避免循环依赖
        paths = line2dup_locator.product_paths(product_dir)
        return cls(
            product_dir=product_dir,
            session_json=os.path.join(product_dir, "session.json"),
            product_params_path=os.path.join(product_dir, "product_params.json"),
            inspection_items_path=os.path.join(product_dir, "inspection_items.json"),
            shape_model_path=os.path.join(product_dir, "shape_model.npz"),
            line2dup_model_path=paths.model_path,
            line2dup_recipe_path=paths.recipe_path,
        )


# ---------------------------------------------------------------------------
# Session 数据快照（从 session.json 读出的内容）
# ---------------------------------------------------------------------------

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


PRODUCT_NAME_RE = re.compile(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$')


# ---------------------------------------------------------------------------
# ProductSession
# ---------------------------------------------------------------------------

class ProductSession:
    """
    管理产品列表、当前产品路径、session 读写。

    使用方式：
        session = ProductSession(session_dir)
        session.load()                    # 读取 products.json
        session.switch_product("Default") # 设置当前产品并更新路径
    """

    _DEFAULT_PRODUCTS: dict = {
        "products": ["Default"],
        "current_product": "Default",
    }

    def __init__(self, session_dir: str) -> None:
        self.session_dir: str = session_dir
        self.products_json: str = os.path.join(session_dir, "products.json")

        # 产品列表元数据
        self._products_data: dict = {
            "products": ["Default"],
            "current_product": "Default",
        }

        # 当前产品名
        self.current_product: str = "Default"

        # 当前产品路径集（switch_product 后才有效）
        self.paths: Optional[ProductPaths] = None

    # ------------------------------------------------------------------
    # 属性代理（让调用方可以直接 session.product_dir 而不用 session.paths.product_dir）
    # ------------------------------------------------------------------

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
    def shape_model_path(self) -> str:
        return self.paths.shape_model_path if self.paths else ""

    @property
    def inspection_items_path(self) -> str:
        return self.paths.inspection_items_path if self.paths else ""

    @property
    def line2dup_model_path(self) -> str:
        return self.paths.line2dup_model_path if self.paths else ""

    @property
    def line2dup_recipe_path(self) -> str:
        return self.paths.line2dup_recipe_path if self.paths else ""

    # ------------------------------------------------------------------
    # 产品列表操作
    # ------------------------------------------------------------------

    def load(self) -> None:
        """从 products.json 加载产品列表；不存在则使用默认值。"""
        if os.path.exists(self.products_json):
            try:
                with open(self.products_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "products" in data:
                    self._products_data = data
            except Exception:
                pass
        self.current_product = str(self._products_data.get("current_product", "Default"))
        self._ensure_current_product_in_list()

    def save_products(self) -> None:
        """把当前产品列表元数据写回 products.json。"""
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.products_json, "w", encoding="utf-8") as f:
            json.dump(self._products_data, f, ensure_ascii=False, indent=2)

    def create_product(self, name: str) -> str:
        """
        新建产品。
        成功返回空字符串，失败返回错误描述。
        """
        name = name.strip()
        if not name:
            return "产品名称不能为空"
        if not PRODUCT_NAME_RE.match(name):
            return "产品名称只能包含字母、数字、下划线和中文字符"
        if name in self._products_data["products"]:
            return "产品名称已存在"
        self._products_data["products"].append(name)
        self.save_products()
        return ""

    def switch_product(self, name: str) -> None:
        """切换当前产品并刷新路径。"""
        self.current_product = name
        self._products_data["current_product"] = name
        self._refresh_paths()

    # ------------------------------------------------------------------
    # Session 读写
    # ------------------------------------------------------------------

    def save_session(self, data: SessionData) -> None:
        """把当前会话数据写入 session.json。"""
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
        """
        从 session.json 读取会话数据，过滤掉不存在的文件路径。
        不存在或读取失败时返回空 SessionData。
        """
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

        return SessionData(
            ok_files=_filter(raw.get("ok_files", [])),
            ng_files=_filter(raw.get("ng_files", [])),
            test_files=_filter(raw.get("test_files", [])),
            ref_image=ref_image,
            loc_method=str(raw.get("loc_method", "line2dup")),
            runtime_cam1_serial=str(raw.get("runtime_cam1_serial", "")).strip(),
            runtime_cam2_serial=str(raw.get("runtime_cam2_serial", "")).strip(),
            runtime_capture_policy=str(raw.get("runtime_capture_policy", "ng_only")).strip() or "ng_only",
        )

    def delete_session_file(self) -> None:
        """清空会话：删除 session.json 文件（不影响产品配置）。"""
        try:
            if self.session_json and os.path.exists(self.session_json):
                os.remove(self.session_json)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _refresh_paths(self) -> None:
        product_dir = os.path.join(self.session_dir, self.current_product)
        os.makedirs(product_dir, exist_ok=True)
        self.paths = ProductPaths.build(product_dir)

    def _ensure_current_product_in_list(self) -> None:
        products = self._products_data.setdefault("products", ["Default"])
        if self.current_product not in products:
            products.append(self.current_product)
