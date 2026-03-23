from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import TYPE_CHECKING, Dict, List, Protocol

from domain import (
    InspectionItem,
    inspection_item_specs_from_line2dup_recipe,
    load_inspection_items,
    output_labels_from_line2dup_recipe,
    save_inspection_items,
    sync_items_with_labels,
)
from line2dup.core import locator as line2dup_locator

if TYPE_CHECKING:
    from application import AlgorithmController, ProductSession
    from ui.debug import ToolPage


class RuntimePredictorProtocol(Protocol):
    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
    ) -> Dict[str, object]: ...


class RuntimeContextProtocol(RuntimePredictorProtocol, Protocol):
    @property
    def inspection_items(self) -> List[InspectionItem]: ...

    @property
    def loc_method(self) -> str: ...

    def current_algorithm(self) -> str: ...

    def load_embedding_model(self, algorithm: str) -> None: ...

    def reload(self) -> None: ...


@dataclass
class ToolPageRuntimeContext:
    tool_page: "ToolPage"

    @property
    def inspection_items(self) -> List[InspectionItem]:
        return self.tool_page.inspection_items

    @property
    def loc_method(self) -> str:
        return self.tool_page.loc_method

    def current_algorithm(self) -> str:
        return self.tool_page.current_algorithm()

    def load_embedding_model(self, algorithm: str) -> None:
        self.tool_page.load_embedding_model(algorithm)

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
    ) -> Dict[str, object]:
        return self.tool_page.predict_image(
            path,
            feat_net=feat_net,
            labels_override=labels_override,
        )

    def reload(self) -> None:
        # 调试窗口里的 ToolPage 本身就是当前真源，这里不额外从磁盘重载。
        return None


@dataclass
class ProductRuntimeContext:
    session: "ProductSession"
    algo: "AlgorithmController"

    def __post_init__(self) -> None:
        self._loc_method = "line2dup"
        self._inspection_items: List[InspectionItem] = []
        self._recipe = None
        self._ref_image = ""
        self.reload()

    @property
    def inspection_items(self) -> List[InspectionItem]:
        return list(self._inspection_items)

    @property
    def loc_method(self) -> str:
        return self._loc_method

    def current_algorithm(self) -> str:
        return str(self.algo.product_params.algorithm or "").strip()

    def load_embedding_model(self, algorithm: str) -> None:
        self.algo.load_model_for_algorithm(algorithm, self.session.product_dir)

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
    ) -> Dict[str, object]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        total_t0 = time.perf_counter()
        match_ms = None
        if self.loc_method == "line2dup":
            recipe = self._ensure_recipe_loaded()
            ref_image = self._reference_image(recipe)
            if ref_image and os.path.exists(ref_image):
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                )
                match_ms = float(run.locate_ms)

        labels = list(labels_override or [])
        if not labels:
            labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        result = self.algo.predict_image(
            path,
            labels=labels,
            feat_net=feat_net,
            match_ms=match_ms,
        )
        payload = result.to_dict()
        payload["infer_ms"] = (
            float(payload.get("total_ms", 0.0))
            if payload.get("total_ms") is not None
            else None
        )
        payload["total_ms"] = float((time.perf_counter() - total_t0) * 1000.0)
        return payload

    def reload(self) -> None:
        self.algo.load_params(self.session.product_params_path)
        session_data = self.session.load_session()
        self._loc_method = str(session_data.loc_method or "line2dup").strip() or "line2dup"
        self._ref_image = str(session_data.ref_image or "").strip()

        self._recipe = self._load_recipe_if_available()
        specs = inspection_item_specs_from_line2dup_recipe(self._recipe)
        labels = [str(spec.get("roi_label", "")).strip() for spec in specs if str(spec.get("roi_label", "")).strip()]
        display_names_by_label = {
            str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
            for spec in specs
            if str(spec.get("roi_label", "")).strip()
        }
        items = load_inspection_items(self.session.inspection_items_path)
        synced_items = sync_items_with_labels(
            items,
            labels,
            display_names_by_label=display_names_by_label,
        )
        if [item.to_dict() for item in synced_items] != [item.to_dict() for item in items]:
            save_inspection_items(synced_items, self.session.inspection_items_path)
        self._inspection_items = synced_items

    def _load_recipe_if_available(self):
        if not os.path.exists(self.session.line2dup_recipe_path):
            return None
        try:
            return line2dup_locator.load_recipe_for_product(self.session.product_dir)
        except Exception:
            return None

    def _ensure_recipe_loaded(self):
        if self._recipe is None:
            self._recipe = self._load_recipe_if_available()
        return self._recipe

    def _line2dup_output_labels(self) -> List[str]:
        return output_labels_from_line2dup_recipe(self._ensure_recipe_loaded())

    def _reference_image(self, recipe) -> str:
        if self._ref_image and os.path.exists(self._ref_image):
            return self._ref_image
        recipe_ref = getattr(recipe, "reference_image", "") if recipe is not None else ""
        return str(recipe_ref or "")


__all__ = [
    "ProductRuntimeContext",
    "RuntimeContextProtocol",
    "RuntimePredictorProtocol",
    "ToolPageRuntimeContext",
]
