from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Protocol

import algorithms.lazy_api as qr_core
import numpy as np
from common.algorithm_codes import learning_backbone_storage_code
from common.camera_roles import (
    CAMERA_ROLES,
    DEFAULT_CAMERA_ROLE,
    camera_role_from_text,
    normalize_camera_role,
)
from algorithms.measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    FIND_LINE_ALGORITHMS,
    LINE_DISTANCE_ALGORITHMS,
    PIN_CENTER_DISTANCE_ALGORITHM,
    judge_bright_block_y_distance,
    judge_edge_distance,
    judge_pin_center_distance,
    measure_bright_block_y_distance_from_array,
    measure_bright_block_center_from_array,
    measure_edge_distance_from_array,
    measure_find_line_from_array,
    measure_pin_center_distance_from_array,
)
from algorithms.traditional import TraditionalThresholdModel, compute_roi_metrics_from_array, metric_value
from common import labelme_io
from application.runtime.preview_frame import RuntimePreviewShape
from domain import (
    InspectionItem,
    load_inspection_items,
    save_inspection_items,
    sync_items_with_labels,
)
from ncc import locator as ncc_locator
from shape.core import locator as shape_locator
from shape.core.recipe_labels import inspection_item_specs_from_shape_recipe, output_labels_from_shape_recipe

if TYPE_CHECKING:
    from application import AlgorithmController, ProductSession
    from ui.debug import ToolPage


def _camera_role_from_path(path: str) -> str:
    return camera_role_from_text(os.path.basename(str(path or "")), default=DEFAULT_CAMERA_ROLE)


def _normalize_loc_method(method: object, *, default: str = "shape") -> str:
    value = str(method or "").strip().lower()
    if value == "line2dup":
        value = "shape"
    if value in {"shape", "ncc"}:
        return value
    return default if default in {"shape", "ncc"} else "shape"


class RuntimePredictorProtocol(Protocol):
    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
        algorithm_override: str | None = None,
        model_key_override: str | None = None,
        params_override: dict | None = None,
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


@dataclass(frozen=True)
class RuntimeFrameBatchPrediction:
    rows: List[Dict[str, object]]
    match_ms: float = 0.0
    roi_shapes: tuple[RuntimePreviewShape, ...] = ()


def _runtime_prediction_row(
    *,
    pred: object,
    diff: object = None,
    sim_ok: object = None,
    sim_ng: object = None,
    value: object = None,
    threshold: object = None,
    match_ms: object = None,
    infer_ms: object = 0.0,
    total_ms: object = 0.0,
    roi_label: str = "",
    detail: str = "",
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "pred": str(pred or "NG"),
        "match_ms": float(match_ms or 0.0) if match_ms is not None else 0.0,
        "infer_ms": float(infer_ms or 0.0) if infer_ms is not None else 0.0,
        "total_ms": float(total_ms or 0.0) if total_ms is not None else 0.0,
    }
    if diff is not None:
        row["diff"] = float(diff)
    if sim_ok is not None:
        row["sim_ok"] = float(sim_ok)
    if sim_ng is not None:
        row["sim_ng"] = float(sim_ng)
    if value is not None:
        row["value"] = float(value)
    if threshold is not None:
        row["threshold"] = float(threshold)
    if str(roi_label or "").strip():
        row["roi_label"] = str(roi_label).strip()
    if str(detail or "").strip():
        row["detail"] = str(detail).strip()
    return row


def _is_measurement_item(algo, item: InspectionItem) -> bool:
    checker = getattr(algo, "is_measurement_tool", None)
    return bool(callable(checker) and checker(item.algorithm_code))


def _is_line_distance_item(item: InspectionItem) -> bool:
    return str(getattr(item, "algorithm_code", "") or "").strip() in LINE_DISTANCE_ALGORITHMS


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

    json_name = os.path.basename(labelme_io.labelme_json_of_image(path))
    for algorithm, group in learning_groups.items():
        if not str(algorithm or "").strip():
            raise RuntimeError("please choose a learning tool subtype first")
        group_infer_t0 = time.perf_counter()
        models: List[Any] = []
        for item in group:
            load_embedding_model(algorithm, model_key=item.model_key)
            if algo.model is None:
                raise RuntimeError(f"algorithm model not loaded: {learning_backbone_storage_code(algorithm)}")
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


def _runtime_shape_by_label(
    roi_shapes: tuple[RuntimePreviewShape, ...],
) -> Dict[str, dict]:
    shape_by_label: Dict[str, dict] = {}
    for shape in roi_shapes:
        label = str(getattr(shape, "label", "") or "").strip()
        if not label:
            continue
        points = [
            [float(point[0]), float(point[1])]
            for point in tuple(getattr(shape, "points", ()) or ())
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not points:
            continue
        shape_by_label[label] = {
            "label": label,
            "shape_type": str(getattr(shape, "shape_type", "rectangle") or "rectangle"),
            "points": points,
        }
    return shape_by_label


def _predict_learning_items_batch_rows_from_frame(
    *,
    image_bgr: np.ndarray,
    roi_shapes: tuple[RuntimePreviewShape, ...],
    items: List[InspectionItem],
    match_ms: float | None,
    algo,
    load_embedding_model,
    feat_net=None,
) -> Dict[str, Dict[str, object]]:
    # Import the embedding helper on demand so startup does not pull torch/torchvision
    # before the main window is visible.
    from algorithms.embedding import embed_batch_from_array

    rows_by_key: Dict[str, Dict[str, object]] = {}
    learning_groups: Dict[str, List[InspectionItem]] = {}
    shape_by_label = _runtime_shape_by_label(roi_shapes)
    for item in items:
        algorithm = algo.resolve_tool_algorithm(item.algorithm_code)
        learning_groups.setdefault(algorithm, []).append(item)

    for algorithm, group in learning_groups.items():
        if not str(algorithm or "").strip():
            raise RuntimeError("please choose a learning tool subtype first")
        group_infer_t0 = time.perf_counter()
        models: List[Any] = []
        for item in group:
            load_embedding_model(algorithm, model_key=item.model_key)
            if algo.model is None:
                raise RuntimeError(f"algorithm model not loaded: {learning_backbone_storage_code(algorithm)}")
            algo.apply_params_to_model()
            models.append(algo.model)
        group_feat_net = feat_net
        if group_feat_net is None or len(learning_groups) > 1:
            group_feat_net = algo.get_feat_net(
                models[0].backbone,
                getattr(models[0], "device", None),
            )
        roi_labels = [str(item.roi_label or "").strip() or "roi" for item in group]
        embeddings = embed_batch_from_array(
            image_bgr,
            group_feat_net,
            roi_labels,
            shape_by_label=shape_by_label,
            device=getattr(models[0], "device", None),
        )
        for item, model, embedding in zip(group, models, embeddings):
            pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(embedding, model)
            rows_by_key[item.model_key] = _runtime_prediction_row(
                pred=pred,
                diff=diff,
                sim_ok=sim_ok,
                sim_ng=sim_ng,
                match_ms=match_ms,
                infer_ms=0.0,
                total_ms=0.0,
                roi_label=str(item.roi_label or "").strip(),
            )
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

    def loc_method_for_role(self, camera_role: object = None) -> str:
        getter = getattr(self.tool_page, "loc_method_for_role", None)
        if callable(getter):
            return _normalize_loc_method(getter(camera_role))
        return _normalize_loc_method(self.tool_page.loc_method)

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
        params_override: dict | None = None,
    ) -> Dict[str, object]:
        return self.tool_page.predict_image(
            path,
            feat_net=feat_net,
            labels_override=labels_override,
            algorithm_override=algorithm_override,
            model_key_override=model_key_override,
            params_override=params_override,
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
        measurement_items = [
            item
            for item in enabled_items
            if _is_measurement_item(tool_page.algo, item) and not _is_line_distance_item(item)
        ]
        traditional_items = [
            item
            for item in enabled_items
            if not tool_page.algo.is_learning_tool(item.algorithm_code)
            and not _is_measurement_item(tool_page.algo, item)
        ]
        camera_role = (
            str(enabled_items[0].camera_id or "").strip()
            if enabled_items
            else str(tool_page.current_camera_role() or "cam1").strip()
        ) or "cam1"

        match_ms = None
        method = self.loc_method_for_role(camera_role)
        if method == "shape":
            recipe = tool_page.shape_recipe_for_role(camera_role)
            ref_image = tool_page.ref_image
            if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
            if ref_image and os.path.exists(ref_image):
                run = shape_locator.autogen_roi_json_from_shape_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=tool_page.session.product_dir,
                    camera_role=camera_role,
                )
                match_ms = float(run.total_ms)
                tool_page._shape_match_ms_by_image[path] = match_ms
                tool_page._shape_autogen_ms_by_image[path] = float(run.total_ms)
        elif method == "ncc":
            run = ncc_locator.autogen_roi_json_from_ncc_timed(
                tgt_img_path=path,
                product_dir=tool_page.session.product_dir,
                camera_role=camera_role,
            )
            match_ms = float(run.total_ms)
            tool_page._shape_match_ms_by_image[path] = match_ms
            tool_page._shape_autogen_ms_by_image[path] = float(run.total_ms)
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
        for item in measurement_items:
            roi_label = str(item.roi_label or "").strip()
            rows_by_key[item.model_key] = self.predict_image(
                path,
                feat_net=feat_net,
                labels_override=[roi_label] if roi_label else None,
                algorithm_override=item.algorithm_code,
                model_key_override=item.model_key,
                params_override=dict(item.params or {}),
            )

        return [dict(rows_by_key[item.model_key]) for item in enabled_items]

    def reload(self) -> None:
        return None


@dataclass
class ProductRuntimeContext:
    session: "ProductSession"
    algo: "AlgorithmController"

    def __post_init__(self) -> None:
        self._loc_method = "shape"
        self._loc_methods_by_role: Dict[str, str] = {role: "shape" for role in CAMERA_ROLES}
        self._inspection_items: List[InspectionItem] = []
        self._recipe = None
        self._recipes_by_role: Dict[str, object] = {}
        self._ref_image = ""
        self._shape_match_ms_by_image: Dict[str, float] = {}
        self.reload()

    @property
    def inspection_items(self) -> List[InspectionItem]:
        return list(self._inspection_items)

    @property
    def loc_method(self) -> str:
        return self._loc_method

    def loc_method_for_role(self, camera_role: object = None) -> str:
        role = normalize_camera_role(camera_role, default=DEFAULT_CAMERA_ROLE)
        return _normalize_loc_method(
            self._loc_methods_by_role.get(role, self._loc_method),
            default=_normalize_loc_method(self._loc_method),
        )

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
        params_override: dict | None = None,
    ) -> Dict[str, object]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        total_t0 = time.perf_counter()
        match_ms = None
        camera_role = _camera_role_from_path(path)
        method = self.loc_method_for_role(camera_role)
        if method == "shape":
            recipe = self._ensure_recipe_loaded(camera_role)
            ref_image = self._reference_image(recipe)
            if ref_image and os.path.exists(ref_image):
                run = shape_locator.autogen_roi_json_from_shape_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=camera_role,
                )
                match_ms = float(run.total_ms)
                self._shape_match_ms_by_image[path] = match_ms
        elif method == "ncc":
            run = ncc_locator.autogen_roi_json_from_ncc_timed(
                tgt_img_path=path,
                product_dir=self.session.product_dir,
                camera_role=camera_role,
            )
            match_ms = float(run.total_ms)
            self._shape_match_ms_by_image[path] = match_ms

        labels = list(labels_override or [])
        if not labels:
            labels = self._loc_output_labels(camera_role)
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
            params_override=params_override,
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
        measurement_items = [
            item
            for item in enabled_items
            if _is_measurement_item(self.algo, item) and not _is_line_distance_item(item)
        ]
        traditional_items = [
            item
            for item in enabled_items
            if not self.algo.is_learning_tool(item.algorithm_code)
            and not _is_measurement_item(self.algo, item)
        ]

        match_ms = None
        camera_role = (
            normalize_camera_role(enabled_items[0].camera_id, default=DEFAULT_CAMERA_ROLE)
            if enabled_items
            else DEFAULT_CAMERA_ROLE
        )
        method = self.loc_method_for_role(camera_role)
        if method == "shape":
            recipe = self._ensure_recipe_loaded(camera_role)
            ref_image = self._reference_image(recipe)
            if ref_image and os.path.exists(ref_image):
                run = shape_locator.autogen_roi_json_from_shape_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=camera_role,
                )
                match_ms = float(run.total_ms)
                self._shape_match_ms_by_image[path] = match_ms
        elif method == "ncc":
            run = ncc_locator.autogen_roi_json_from_ncc_timed(
                tgt_img_path=path,
                product_dir=self.session.product_dir,
                camera_role=camera_role,
            )
            match_ms = float(run.total_ms)
            self._shape_match_ms_by_image[path] = match_ms

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
        for item in measurement_items:
            roi_label = str(item.roi_label or "").strip()
            rows_by_key[item.model_key] = self.predict_image(
                path,
                feat_net=feat_net,
                labels_override=[roi_label] if roi_label else None,
                algorithm_override=item.algorithm_code,
                model_key_override=item.model_key,
                params_override=dict(item.params or {}),
            )

        return [dict(rows_by_key[item.model_key]) for item in enabled_items]

    def predict_items_batch_from_frame(
        self,
        image_bgr,
        *,
        camera_role: str,
        items: List[InspectionItem],
        feat_net=None,
    ) -> RuntimeFrameBatchPrediction:
        image = np.asarray(image_bgr)
        if image.ndim not in {2, 3}:
            raise ValueError(f"unsupported image shape: {image.shape!r}")
        enabled_items = [item for item in items if item.enabled]
        learning_items = [item for item in enabled_items if self.algo.is_learning_tool(item.algorithm_code)]
        measurement_items = [
            item
            for item in enabled_items
            if _is_measurement_item(self.algo, item) and not _is_line_distance_item(item)
        ]
        traditional_items = [
            item
            for item in enabled_items
            if not self.algo.is_learning_tool(item.algorithm_code)
            and not _is_measurement_item(self.algo, item)
        ]

        role = normalize_camera_role(camera_role, default=DEFAULT_CAMERA_ROLE)
        match_ms = 0.0
        roi_shapes: tuple[RuntimePreviewShape, ...] = ()
        method = self.loc_method_for_role(role)
        if method == "shape":
            recipe = self._ensure_recipe_loaded(role)
            ref_image = self._reference_image(recipe)
            if ref_image and os.path.exists(ref_image):
                run = shape_locator.autogen_runtime_roi_shapes_timed(
                    scene_bgr=image,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=role,
                )
                match_ms = float(run.total_ms)
                roi_shapes = tuple(
                    RuntimePreviewShape(
                        label=str(shape.label_name or "").strip() or "roi",
                        shape_type=str(shape.shape_type or "rectangle"),
                        points=tuple((float(x), float(y)) for x, y in tuple(shape.points or ())),
                    )
                    for shape in tuple(run.roi_shapes or ())
                )
        elif method == "ncc":
            run = ncc_locator.autogen_runtime_roi_shapes_timed(
                scene_bgr=image,
                product_dir=self.session.product_dir,
                camera_role=role,
            )
            match_ms = float(run.total_ms)
            roi_shapes = tuple(
                RuntimePreviewShape(
                    label=str(shape.label_name or "").strip() or "roi",
                    shape_type=str(shape.shape_type or "polygon"),
                    points=tuple((float(x), float(y)) for x, y in tuple(shape.points or ())),
                )
                for shape in tuple(run.roi_shapes or ())
            )

        rows_by_key = _predict_learning_items_batch_rows_from_frame(
            image_bgr=image,
            roi_shapes=roi_shapes,
            items=learning_items,
            match_ms=match_ms,
            algo=self.algo,
            load_embedding_model=self.load_embedding_model,
            feat_net=feat_net,
        )
        shape_by_label = _runtime_shape_by_label(roi_shapes)
        for item in traditional_items:
            algorithm = self.algo.resolve_tool_algorithm(item.algorithm_code)
            model_dict = self.algo.get_traditional_model_dict(algorithm, model_key=item.model_key)
            if not isinstance(model_dict, dict):
                raise RuntimeError(f"traditional algorithm {algorithm} is not trained yet")
            threshold_model = TraditionalThresholdModel.from_dict(model_dict)
            metrics = compute_roi_metrics_from_array(
                image,
                shape_by_label=shape_by_label,
                preferred_label=threshold_model.roi_label or str(item.roi_label or "").strip() or "roi",
            )
            value = metric_value(metrics, algorithm)
            pred, diff = threshold_model.predict(value)
            rows_by_key[item.model_key] = _runtime_prediction_row(
                pred=pred,
                diff=diff,
                value=value,
                threshold=threshold_model.threshold,
                match_ms=match_ms,
                infer_ms=0.0,
                total_ms=0.0,
                roi_label=str(metrics.get("roi_label", "") or ""),
            )
        for item in measurement_items:
            params = dict(item.params or {})
            algorithm = self.algo.resolve_tool_algorithm(item.algorithm_code)
            measurement_payload_override = None
            if algorithm in FIND_LINE_ALGORITHMS:
                measurement = measure_find_line_from_array(
                    image,
                    shape_by_label=shape_by_label,
                    preferred_label=str(item.roi_label or "").strip() or "roi",
                    params=params,
                    algorithm=algorithm,
                )
                pred = "OK"
                judged_value = None
                lower = None
                upper = None
                residual = float(measurement.line.residual)
                detail = (
                    f"line_found pts={measurement.line.point_count}"
                    f" pos={measurement.position_px:.3f}px"
                    f" angle={measurement.angle_deg:.3f}deg"
                    f" residual={residual:.3f}"
                )
                unit = "px"
            elif algorithm == BRIGHT_BLOCK_CENTER_ALGORITHM:
                judged_value = None
                lower = None
                upper = None
                residual = 0.0
                unit = "px"
                roi_label = str(item.roi_label or "").strip() or "roi"
                try:
                    measurement = measure_bright_block_center_from_array(
                        image,
                        shape_by_label=shape_by_label,
                        preferred_label=roi_label,
                        params=params,
                    )
                    pred = "OK"
                    detail = (
                        f"bright_block_center=({measurement.center_xy[0]:.3f},"
                        f"{measurement.center_xy[1]:.3f})px"
                        f" threshold={measurement.threshold:.1f}"
                    )
                    measurement_payload_override = None
                except RuntimeError as exc:
                    pred = "NG"
                    detail = f"bright_block_center_missing: {exc}"
                    measurement = None
                    measurement_payload_override = {
                        "type": BRIGHT_BLOCK_CENTER_ALGORITHM,
                        "roi_label": roi_label,
                        "center_points": [],
                        "candidates": [],
                        "pred": pred,
                        "detail": str(exc),
                    }
            elif algorithm == PIN_CENTER_DISTANCE_ALGORITHM:
                measurement = measure_pin_center_distance_from_array(
                    image,
                    shape_by_label=shape_by_label,
                    preferred_label=str(item.roi_label or "").strip() or "roi",
                    params=params,
                )
                pred, judged_value, lower, upper, unit = judge_pin_center_distance(measurement, params)
                residual = 0.0
                detail = (
                    f"pin_center_distance={judged_value:.3f}{unit}"
                    f" raw={measurement.distance_px:.3f}px"
                    f" threshold={measurement.threshold:.1f}"
                )
            elif algorithm == BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM:
                measurement = measure_bright_block_y_distance_from_array(
                    image,
                    shape_by_label=shape_by_label,
                    preferred_label=str(item.roi_label or "").strip() or "roi",
                    params=params,
                )
                pred, judged_value, lower, upper, unit = judge_bright_block_y_distance(measurement, params)
                residual = 0.0
                detail = (
                    f"bright_block_y_distance={judged_value:.3f}{unit}"
                    f" raw={measurement.distance_px:.3f}px"
                    f" threshold={measurement.threshold:.1f}"
                )
            else:
                measurement = measure_edge_distance_from_array(
                    image,
                    shape_by_label=shape_by_label,
                    preferred_label=str(item.roi_label or "").strip() or "roi",
                    params=params,
                )
                pred, judged_value, lower, upper, unit = judge_edge_distance(measurement, params)
                residual = float(max(measurement.line_a.residual, measurement.line_b.residual))
                detail = f"distance={judged_value:.3f}{unit}"
                if measurement.distance_mm is not None:
                    detail += f" raw={measurement.distance_px:.3f}px/{measurement.distance_mm:.4f}mm"
                detail += (
                    f" pts={measurement.line_a.point_count}/{measurement.line_b.point_count}"
                    f" residual={residual:.3f}"
                )
            if lower is not None or upper is not None:
                detail += f" spec={lower if lower is not None else '-'}..{upper if upper is not None else '-'}{unit}"
            rows_by_key[item.model_key] = _runtime_prediction_row(
                pred=pred,
                diff=residual,
                value=judged_value,
                threshold=upper,
                match_ms=match_ms,
                infer_ms=0.0,
                total_ms=0.0,
                roi_label=getattr(measurement, "roi_label", str(item.roi_label or "").strip() or "roi"),
                detail=detail,
            )
            measurement_payload = (
                measurement_payload_override
                if measurement_payload_override is not None
                else measurement.to_dict()
            )
            if algorithm in {PIN_CENTER_DISTANCE_ALGORITHM, BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM}:
                measurement_payload.update(
                    {
                        "distance": float(judged_value),
                        "unit": unit,
                        "label": f"{judged_value:.3f}{unit}",
                        "pred": pred,
                    }
                )
            elif algorithm == BRIGHT_BLOCK_CENTER_ALGORITHM:
                measurement_payload.update({"pred": pred})
            rows_by_key[item.model_key]["measurement"] = measurement_payload

        return RuntimeFrameBatchPrediction(
            rows=[dict(rows_by_key[item.model_key]) for item in enabled_items],
            match_ms=float(match_ms),
            roi_shapes=roi_shapes,
        )

    def reload(self) -> None:
        self.algo.load_params(self.session.product_params_path)
        self._shape_match_ms_by_image = {}
        session_data = self.session.load_session()
        self._loc_method = _normalize_loc_method(session_data.loc_method)
        raw_loc_methods = dict(getattr(session_data, "loc_methods", {}) or {})
        self._loc_methods_by_role = {
            role: _normalize_loc_method(raw_loc_methods.get(role, self._loc_method), default=self._loc_method)
            for role in CAMERA_ROLES
        }
        self._ref_image = str(session_data.ref_image or "").strip()
        self._recipes_by_role = {}
        items = load_inspection_items(self.session.inspection_items_path)
        synced_items: List[InspectionItem] = []
        remaining_items: List[InspectionItem] = []
        for role in CAMERA_ROLES:
            method = self.loc_method_for_role(role)
            recipe = self._load_recipe_if_available(role)
            self._recipes_by_role[role] = recipe
            role_items = [
                item for item in items
                if str(getattr(item, "camera_id", "") or "").strip() == role
            ]
            if method == "ncc":
                if not ncc_locator.model_is_ready(self.session.product_dir, role):
                    if role_items:
                        synced_items.extend(role_items)
                    continue
                specs = ncc_locator.inspection_item_specs_for_product(self.session.product_dir, role)
            else:
                if recipe is None and not role_items:
                    continue
                specs = inspection_item_specs_from_shape_recipe(recipe)
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
            if normalize_camera_role(getattr(item, "camera_id", "")) not in set(CAMERA_ROLES)
        ]
        synced_items = remaining_items + synced_items
        self._recipe = self._recipes_by_role.get(DEFAULT_CAMERA_ROLE)
        if [item.to_dict() for item in synced_items] != [item.to_dict() for item in items]:
            save_inspection_items(synced_items, self.session.inspection_items_path)
        self._inspection_items = synced_items

    def _load_recipe_if_available(self, camera_role: str = "cam1"):
        recipe_path = shape_locator.resolved_recipe_path_for_product(self.session.product_dir, camera_role)
        if not os.path.exists(recipe_path):
            return None
        try:
            return shape_locator.load_recipe_for_product(self.session.product_dir, camera_role)
        except Exception:
            return None

    def _ensure_recipe_loaded(self, camera_role: str = "cam1"):
        role = normalize_camera_role(camera_role, default=DEFAULT_CAMERA_ROLE)
        if role not in self._recipes_by_role or self._recipes_by_role.get(role) is None:
            self._recipes_by_role[role] = self._load_recipe_if_available(role)
        if role == DEFAULT_CAMERA_ROLE:
            self._recipe = self._recipes_by_role.get(role)
        return self._recipes_by_role.get(role)

    def _shape_output_labels(self, camera_role: str = "cam1") -> List[str]:
        return output_labels_from_shape_recipe(self._ensure_recipe_loaded(camera_role))

    def _ncc_output_labels(self, camera_role: str = "cam1") -> List[str]:
        try:
            return ncc_locator.output_labels_for_product(self.session.product_dir, camera_role)
        except Exception:
            return ["roi"]

    def _loc_output_labels(self, camera_role: str = "cam1") -> List[str]:
        method = self.loc_method_for_role(camera_role)
        if method == "shape":
            return self._shape_output_labels(camera_role)
        if method == "ncc":
            return self._ncc_output_labels(camera_role)
        return ["roi"]

    def _reference_image(self, recipe) -> str:
        if self._ref_image and os.path.exists(self._ref_image):
            return self._ref_image
        recipe_ref = getattr(recipe, "reference_image", "") if recipe is not None else ""
        return str(recipe_ref or "")


__all__ = [
    "ProductRuntimeContext",
    "RuntimeFrameBatchPrediction",
    "RuntimeContextProtocol",
    "RuntimePredictorProtocol",
    "ToolPageRuntimeContext",
]




