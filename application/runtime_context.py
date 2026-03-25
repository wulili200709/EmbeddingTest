from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Protocol

import algorithms.proxy as qr_core
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


_CAMERA_ROLE_RE = re.compile(r"(?:^|[_-])(cam[12])(?=[_.-]|$)", re.IGNORECASE)


def _camera_role_from_path(path: str) -> str:
    match = _CAMERA_ROLE_RE.search(os.path.basename(str(path or "")))
    if not match:
        return "cam1"
    return str(match.group(1) or "cam1").lower()


class RuntimePredictorProtocol(Protocol):
    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
        algorithm_override: str | None = None,
        model_key_override: str | None = None,
    ) -> Dict[str, object]: ...

    def predict_items_batch(
        self,
        path: str,
        *,
        items: List[InspectionItem],
        feat_net=None,
    ) -> List[Dict[str, object]]: ...


class RuntimeContextProtocol(RuntimePredictorProtocol, Protocol):
    @property
    def inspection_items(self) -> List[InspectionItem]: ...

    @property
    def loc_method(self) -> str: ...

    def current_algorithm(self) -> str: ...

    def load_embedding_model(self, algorithm: str, model_key: str | None = None) -> None: ...

    def reload(self) -> None: ...


def _predict_learning_items_batch_rows(
    *,
    path: str,
    items: List[InspectionItem],
    match_ms: float | None,
    algo,
    load_embedding_model,
    feat_net=None,
) -> Dict[str, Dict[str, object]]:
    rows_by_key: Dict[str, Dict[str, object]] = {}
    learning_groups: Dict[str, List[InspectionItem]] = {}
    for item in items:
        algorithm = algo.resolve_tool_algorithm(item.algorithm_code)
        learning_groups.setdefault(algorithm, []).append(item)

    json_name = os.path.basename(qr_core.labelme_json_of_image(path))
    for algorithm, group in learning_groups.items():
        if not str(algorithm or "").strip():
            raise RuntimeError("please choose a learning tool subtype first")
        group_infer_t0 = time.perf_counter()
        models: List[Any] = []
        for item in group:
            load_embedding_model(algorithm, model_key=item.model_key)
            if algo.model is None:
                raise RuntimeError(f"algorithm model not loaded: {algorithm}")
            algo.apply_params_to_model()
            models.append(algo.model)
        group_feat_net = feat_net
        if group_feat_net is None or len(learning_groups) > 1:
            group_feat_net = algo.get_feat_net(
                models[0].backbone,
                getattr(models[0], "device", None),
            )
        roi_labels = [str(item.roi_label or "").strip() or "roi" for item in group]
        embeddings = qr_core.embed_batch(
            path,
            group_feat_net,
            roi_labels,
            device=getattr(models[0], "device", None),
        )
        for item, model, embedding in zip(group, models, embeddings):
            pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(embedding, model)
            rows_by_key[item.model_key] = {
                "file_path": path,
                "file_name": os.path.basename(path),
                "gt": "",
                "pred": pred,
                "diff": float(diff),
                "sim_ok": float(sim_ok),
                "sim_ng": float(sim_ng),
                "value": None,
                "threshold": None,
                "match_ms": match_ms,
                "infer_ms": 0.0,
                "total_ms": 0.0,
                "json_name": json_name,
            }
        group_infer_total_ms = float((time.perf_counter() - group_infer_t0) * 1000.0)
        per_item_infer_ms = group_infer_total_ms / float(len(group)) if group else 0.0
        for item in group:
            rows_by_key[item.model_key]["infer_ms"] = per_item_infer_ms
            rows_by_key[item.model_key]["total_ms"] = per_item_infer_ms
    return rows_by_key


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

    def load_embedding_model(self, algorithm: str, model_key: str | None = None) -> None:
        self.tool_page.load_embedding_model(algorithm, model_key=model_key)

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
        algorithm_override: str | None = None,
        model_key_override: str | None = None,
    ) -> Dict[str, object]:
        return self.tool_page.predict_image(
            path,
            feat_net=feat_net,
            labels_override=labels_override,
            algorithm_override=algorithm_override,
            model_key_override=model_key_override,
        )

    def predict_items_batch(
        self,
        path: str,
        *,
        items: List[InspectionItem],
        feat_net=None,
    ) -> List[Dict[str, object]]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        tool_page = self.tool_page
        enabled_items = [item for item in items if item.enabled]
        learning_items = [item for item in enabled_items if tool_page.algo.is_learning_tool(item.algorithm_code)]
        traditional_items = [item for item in enabled_items if not tool_page.algo.is_learning_tool(item.algorithm_code)]
        camera_role = (
            str(enabled_items[0].camera_id or "").strip()
            if enabled_items
            else str(tool_page.current_camera_role() or "cam1").strip()
        ) or "cam1"

        match_ms = None
        if tool_page.loc_method == "line2dup":
            recipe = tool_page.line2dup_recipe_for_role(camera_role)
            ref_image = tool_page.ref_image
            if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
            if ref_image and os.path.exists(ref_image):
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=tool_page.session.product_dir,
                    camera_role=camera_role,
                )
                match_ms = float(run.total_ms)
                tool_page._line2dup_match_ms_by_image[path] = match_ms
                tool_page._line2dup_autogen_ms_by_image[path] = float(run.total_ms)
        elif tool_page.ref_image and os.path.exists(tool_page.ref_image):
            tool_page._autogen_roi_for_images([path], only_missing=True, silent=True)

        rows_by_key = _predict_learning_items_batch_rows(
            path=path,
            items=learning_items,
            match_ms=match_ms,
            algo=tool_page.algo,
            load_embedding_model=tool_page.load_embedding_model,
            feat_net=feat_net,
        )
        for item in traditional_items:
            roi_label = str(item.roi_label or "").strip()
            labels_override = [roi_label] if roi_label else None
            rows_by_key[item.model_key] = self.predict_image(
                path,
                feat_net=feat_net,
                labels_override=labels_override,
                algorithm_override=item.algorithm_code,
                model_key_override=item.model_key,
            )

        return [dict(rows_by_key[item.model_key]) for item in enabled_items]

    def reload(self) -> None:
        return None


@dataclass
class ProductRuntimeContext:
    session: "ProductSession"
    algo: "AlgorithmController"

    def __post_init__(self) -> None:
        self._loc_method = "line2dup"
        self._inspection_items: List[InspectionItem] = []
        self._recipe = None
        self._recipes_by_role: Dict[str, object] = {}
        self._ref_image = ""
        self._line2dup_match_ms_by_image: Dict[str, float] = {}
        self.reload()

    @property
    def inspection_items(self) -> List[InspectionItem]:
        return list(self._inspection_items)

    @property
    def loc_method(self) -> str:
        return self._loc_method

    def current_algorithm(self) -> str:
        return str(self.algo.product_params.algorithm or "").strip()

    def load_embedding_model(self, algorithm: str, model_key: str | None = None) -> None:
        self.algo.load_model_for_algorithm(algorithm, self.session.product_dir, model_key=model_key)

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
        algorithm_override: str | None = None,
        model_key_override: str | None = None,
    ) -> Dict[str, object]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        total_t0 = time.perf_counter()
        match_ms = None
        camera_role = _camera_role_from_path(path)
        if self.loc_method == "line2dup":
            recipe = self._ensure_recipe_loaded(camera_role)
            ref_image = self._reference_image(recipe)
            if ref_image and os.path.exists(ref_image):
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=camera_role,
                )
                match_ms = float(run.total_ms)
                self._line2dup_match_ms_by_image[path] = match_ms

        labels = list(labels_override or [])
        if not labels:
            labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        effective_algorithm = ""
        if str(algorithm_override or "").strip():
            effective_algorithm = self.algo.resolve_tool_algorithm(algorithm_override)
        elif str(self.algo.product_params.algorithm or "").strip():
            effective_algorithm = self.algo.resolve_learning_algorithm(self.algo.product_params.algorithm)
        if effective_algorithm and self.algo.is_embedding_algorithm(effective_algorithm):
            if not self.algo._loaded_embedding_matches(
                effective_algorithm,
                labels=labels,
                model_key=model_key_override or "",
            ):
                self.load_embedding_model(effective_algorithm, model_key=model_key_override)
        result = self.algo.predict_image(
            path,
            labels=labels,
            feat_net=feat_net,
            match_ms=match_ms,
            algorithm_override=algorithm_override,
            model_key_override=model_key_override,
        )
        payload = result.to_dict()
        payload["infer_ms"] = (
            float(payload.get("total_ms", 0.0))
            if payload.get("total_ms") is not None
            else None
        )
        payload["total_ms"] = float((time.perf_counter() - total_t0) * 1000.0)
        return payload

    def predict_items_batch(
        self,
        path: str,
        *,
        items: List[InspectionItem],
        feat_net=None,
    ) -> List[Dict[str, object]]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        enabled_items = [item for item in items if item.enabled]
        learning_items = [item for item in enabled_items if self.algo.is_learning_tool(item.algorithm_code)]
        traditional_items = [item for item in enabled_items if not self.algo.is_learning_tool(item.algorithm_code)]

        match_ms = None
        camera_role = str(enabled_items[0].camera_id or "cam1").strip() if enabled_items else "cam1"
        if self.loc_method == "line2dup":
            recipe = self._ensure_recipe_loaded(camera_role)
            ref_image = self._reference_image(recipe)
            if ref_image and os.path.exists(ref_image):
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=camera_role,
                )
                match_ms = float(run.total_ms)
                self._line2dup_match_ms_by_image[path] = match_ms

        rows_by_key = _predict_learning_items_batch_rows(
            path=path,
            items=learning_items,
            match_ms=match_ms,
            algo=self.algo,
            load_embedding_model=self.load_embedding_model,
            feat_net=feat_net,
        )
        for item in traditional_items:
            roi_label = str(item.roi_label or "").strip()
            labels_override = [roi_label] if roi_label else None
            rows_by_key[item.model_key] = self.predict_image(
                path,
                feat_net=feat_net,
                labels_override=labels_override,
                algorithm_override=item.algorithm_code,
                model_key_override=item.model_key,
            )

        return [dict(rows_by_key[item.model_key]) for item in enabled_items]

    def reload(self) -> None:
        self.algo.load_params(self.session.product_params_path)
        self._line2dup_match_ms_by_image = {}
        session_data = self.session.load_session()
        self._loc_method = str(session_data.loc_method or "line2dup").strip() or "line2dup"
        self._ref_image = str(session_data.ref_image or "").strip()
        self._recipes_by_role = {}
        items = load_inspection_items(self.session.inspection_items_path)
        synced_items: List[InspectionItem] = []
        remaining_items: List[InspectionItem] = []
        for role in ("cam1", "cam2"):
            recipe = self._load_recipe_if_available(role)
            self._recipes_by_role[role] = recipe
            role_items = [
                item for item in items
                if str(getattr(item, "camera_id", "") or "").strip() == role
            ]
            if recipe is None and not role_items:
                continue
            specs = inspection_item_specs_from_line2dup_recipe(recipe)
            labels = [
                str(spec.get("roi_label", "")).strip()
                for spec in specs
                if str(spec.get("roi_label", "")).strip()
            ]
            display_names_by_label = {
                str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
                for spec in specs
                if str(spec.get("roi_label", "")).strip()
            }
            synced_items.extend(
                sync_items_with_labels(
                    role_items,
                    labels,
                    default_camera_id=role,
                    display_names_by_label=display_names_by_label,
                )
            )
        remaining_items = [
            item for item in items
            if str(getattr(item, "camera_id", "") or "").strip() not in {"cam1", "cam2"}
        ]
        synced_items = remaining_items + synced_items
        self._recipe = self._recipes_by_role.get("cam1")
        if [item.to_dict() for item in synced_items] != [item.to_dict() for item in items]:
            save_inspection_items(synced_items, self.session.inspection_items_path)
        self._inspection_items = synced_items

    def _load_recipe_if_available(self, camera_role: str = "cam1"):
        recipe_path = line2dup_locator.resolved_recipe_path_for_product(self.session.product_dir, camera_role)
        if not os.path.exists(recipe_path):
            return None
        try:
            return line2dup_locator.load_recipe_for_product(self.session.product_dir, camera_role)
        except Exception:
            return None

    def _ensure_recipe_loaded(self, camera_role: str = "cam1"):
        role = str(camera_role or "cam1").strip() or "cam1"
        if role not in self._recipes_by_role or self._recipes_by_role.get(role) is None:
            self._recipes_by_role[role] = self._load_recipe_if_available(role)
        if role == "cam1":
            self._recipe = self._recipes_by_role.get(role)
        return self._recipes_by_role.get(role)

    def _line2dup_output_labels(self, camera_role: str = "cam1") -> List[str]:
        return output_labels_from_line2dup_recipe(self._ensure_recipe_loaded(camera_role))

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
