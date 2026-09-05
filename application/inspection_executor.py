
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import os
from typing import List, Protocol

import numpy as np

from algorithms.measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    CENTER_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHMS,
    FittedLine,
    LINE_DISTANCE_ALGORITHMS,
    LINE_DISTANCE_REF_NORMAL_ALGORITHM,
    MULTI_PIN_TIP_HEIGHT_ALGORITHM,
    POINT_LINE_DISTANCE_ALGORITHM,
    fit_line_filtered,
)
from domain import InspectionItem, InspectionItemResult

_DECISION_POLICY_FILENAME = "decision_policy.json"
_PASS_RESULT = "PASS"


def _is_line_distance_item(item: InspectionItem) -> bool:
    return str(getattr(item, "algorithm_code", "") or "").strip() in LINE_DISTANCE_ALGORITHMS


def _is_center_distance_item(item: InspectionItem) -> bool:
    return str(getattr(item, "algorithm_code", "") or "").strip() in CENTER_DISTANCE_ALGORITHMS


def _is_point_line_distance_item(item: InspectionItem) -> bool:
    return str(getattr(item, "algorithm_code", "") or "").strip() == POINT_LINE_DISTANCE_ALGORITHM


def _is_post_distance_item(item: InspectionItem) -> bool:
    return _is_line_distance_item(item) or _is_center_distance_item(item) or _is_point_line_distance_item(item)


def _distance_helper_item_ids(items: List[InspectionItem]) -> set[str]:
    """Return item ids used only as inputs by enabled composite distance tools."""
    helper_ids: set[str] = set()
    for item in items:
        if not item.enabled or not _is_post_distance_item(item):
            continue
        params = dict(item.params or {})
        keys = (
            ("point_item_id", "line_item_id")
            if _is_point_line_distance_item(item)
            else ("center_a_item_id", "center_b_item_id")
            if _is_center_distance_item(item)
            else ("line_a_item_id", "line_b_item_id")
        )
        for key in keys:
            item_id = str(params.get(key, "") or "").strip()
            if item_id:
                helper_ids.add(item_id)
    return helper_ids


def _normalized_result(value: object) -> str:
    return str(value or "").strip().upper()


def _is_passing_result(value: object) -> bool:
    return _normalized_result(value) in {"OK", _PASS_RESULT}


def _append_detail(existing: str, extra: str) -> str:
    existing_text = str(existing or "").strip()
    extra_text = str(extra or "").strip()
    if not existing_text:
        return extra_text
    if not extra_text or extra_text in existing_text:
        return existing_text
    return f"{existing_text} {extra_text}"


class PredictorProtocol(Protocol):
    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
        algorithm_override: str | None = None,
        model_key_override: str | None = None,
        params_override: dict | None = None,
    ) -> dict: ...


@dataclass
class InspectionExecutionRequest:
    camera_id: str
    image_path: str = ""
    image_bgr: object | None = None
    items: List[InspectionItem] = field(default_factory=list)


@dataclass
class InspectionExecutionResponse:
    camera_id: str
    result: str
    detail: str = ""
    raw_row: dict | None = None
    match_ms: float = 0.0
    infer_ms: float = 0.0
    total_ms: float = 0.0
    item_results: List[InspectionItemResult] = field(default_factory=list)
    roi_shapes: tuple[object, ...] = field(default_factory=tuple)
    measurements: tuple[dict, ...] = field(default_factory=tuple)


class InspectionExecutor:
    """
    相机级检测执行器。

    说明：
      - 当前先按 camera → one predict row → 复制到 camera 下所有 item 的方式运行
      - 后续再扩展为真正 item 级独立执行
    """

    def __init__(self, predictor: PredictorProtocol, decision_policy: Mapping[str, object] | None = None) -> None:
        self._predictor = predictor
        self._decision_policy_override = dict(decision_policy) if isinstance(decision_policy, Mapping) else None

    def execute(self, request: InspectionExecutionRequest) -> InspectionExecutionResponse:
        if not request.items:
            row = self._predictor.predict_image(request.image_path)
            pred = str(row.get("pred", "NG") or "NG")
            detail = self._build_detail(row)
            match_ms, infer_ms = self._extract_timing_fields(row)
            return InspectionExecutionResponse(
                camera_id=request.camera_id,
                result=pred,
                detail=detail,
                raw_row=row,
                match_ms=match_ms,
                infer_ms=infer_ms,
                total_ms=float(row.get("total_ms") or infer_ms or 0.0),
                item_results=[],
            )

        item_results: List[InspectionItemResult] = []
        item_rows: List[dict] = []
        enabled_item_results: List[InspectionItemResult] = []
        enabled_items = [item for item in request.items if item.enabled]
        predicted_enabled_items = [item for item in enabled_items if not _is_post_distance_item(item)]
        distance_items = [item for item in enabled_items if _is_post_distance_item(item)]
        batch_rows: List[dict] | None = None
        batch_timing_breakdown: dict[str, object] = {}
        roi_shapes: tuple[object, ...] = ()
        batch_predict_from_frame = getattr(self._predictor, "predict_items_batch_from_frame", None)
        batch_predict = getattr(self._predictor, "predict_items_batch", None)
        if request.image_bgr is not None and callable(batch_predict_from_frame) and predicted_enabled_items:
            batch_prediction = batch_predict_from_frame(
                request.image_bgr,
                camera_role=request.camera_id,
                items=predicted_enabled_items,
            )
            if batch_prediction is not None:
                batch_rows = [dict(row) for row in list(getattr(batch_prediction, "rows", []) or [])]
                roi_shapes = tuple(getattr(batch_prediction, "roi_shapes", ()) or ())
                batch_timing_breakdown = {
                    str(key): value
                    for key, value in dict(
                        getattr(batch_prediction, "timing_breakdown", {}) or {}
                    ).items()
                }
        elif callable(batch_predict) and predicted_enabled_items:
            batch_rows = [dict(row) for row in batch_predict(request.image_path, items=predicted_enabled_items)]
        predicted_index = 0
        rows_by_item_id: dict[str, dict] = {}

        for item in request.items:
            if not item.enabled:
                item_results.append(
                    InspectionItemResult(
                        item_id=item.item_id,
                        display_name=item.display_name,
                        camera_id=item.camera_id,
                        roi_label=item.roi_label,
                        algorithm_code=item.algorithm_code,
                        enabled=False,
                        params=dict(item.params or {}),
                        result="DISABLED",
                    )
                )
                continue
            if _is_post_distance_item(item):
                continue

            if batch_rows is not None:
                row = dict(batch_rows[predicted_index]) if predicted_index < len(batch_rows) else {}
            else:
                roi_label = str(item.roi_label or "").strip()
                labels_override = [roi_label] if roi_label else None
                row = self._predict_image_for_item(
                    request.image_path,
                    labels_override=labels_override,
                    algorithm_override=item.algorithm_code,
                    model_key_override=item.model_key,
                    params_override=dict(item.params or {}),
                )
            predicted_index += 1
            item_rows.append(dict(row))
            item_key = str(getattr(item, "item_id", "") or "").strip()
            if item_key:
                rows_by_item_id[item_key] = dict(row)
            item_result = InspectionItemResult(
                item_id=item.item_id,
                display_name=item.display_name,
                camera_id=item.camera_id,
                roi_label=item.roi_label,
                algorithm_code=item.algorithm_code,
                enabled=True,
                params=dict(item.params or {}),
                result=str(row.get("pred", "NG") or "NG"),
                detail=self._build_detail(row),
                value=self._row_value(row),
                unit=self._row_unit(row),
            )
            item_results.append(item_result)
            enabled_item_results.append(item_result)

        post_distance_results: List[InspectionItemResult] = []
        for distance_item in distance_items:
            if _is_point_line_distance_item(distance_item):
                distance_row = self._build_point_line_distance_row(
                    distance_item=distance_item,
                    source_items=predicted_enabled_items,
                    rows_by_item_id=rows_by_item_id,
                    camera_id=request.camera_id,
                    image_path=request.image_path,
                )
            elif _is_center_distance_item(distance_item):
                distance_row = self._build_center_distance_row(
                    distance_item=distance_item,
                    center_items=predicted_enabled_items,
                    rows_by_item_id=rows_by_item_id,
                    camera_id=request.camera_id,
                    image_path=request.image_path,
                )
            else:
                distance_row = self._build_find_line_distance_row(
                    distance_item=distance_item,
                    line_items=predicted_enabled_items,
                    line_rows_by_item_id=rows_by_item_id,
                    camera_id=request.camera_id,
                    image_path=request.image_path,
                )
            if distance_row is None:
                distance_algorithm = str(getattr(distance_item, "algorithm_code", "") or "").strip() or "line_distance"
                pair_detail = (
                    "point-line distance pair missing"
                    if _is_point_line_distance_item(distance_item)
                    else "center distance pair missing"
                    if _is_center_distance_item(distance_item)
                    else "line distance pair missing"
                )
                distance_row = {
                    "file_path": request.image_path,
                    "file_name": str(distance_item.display_name or distance_item.item_id or "Distance"),
                    "pred": "NG",
                    "detail": pair_detail,
                    "algorithm": distance_algorithm,
                    "tool_name": str(distance_item.display_name or distance_item.item_id or "Distance"),
                    "camera_id": request.camera_id,
                    "roi_label": str(distance_item.roi_label or ""),
                    "params": dict(distance_item.params or {}),
                }
            item_rows.append(dict(distance_row))
            distance_result = InspectionItemResult(
                item_id=distance_item.item_id,
                display_name=distance_item.display_name,
                camera_id=distance_item.camera_id,
                roi_label=distance_item.roi_label,
                algorithm_code=distance_item.algorithm_code,
                enabled=True,
                params=dict(distance_item.params or {}),
                result=str(distance_row.get("pred", "NG") or "NG"),
                detail=self._build_detail(distance_row),
                value=self._row_value(distance_row),
                unit=self._row_unit(distance_row),
            )
            item_results.append(distance_result)
            enabled_item_results.append(distance_result)
            post_distance_results.append(distance_result)

        if not enabled_item_results:
            return InspectionExecutionResponse(
                camera_id=request.camera_id,
                result="OK",
                detail="",
                raw_row=None,
                total_ms=0.0,
                item_results=item_results,
                roi_shapes=roi_shapes,
            )

        helper_item_ids = _distance_helper_item_ids(enabled_items)
        post_distance_item_ids = {
            str(item.item_id or "").strip()
            for item in post_distance_results
        }
        decision_item_results = [
            item
            for item in enabled_item_results
            if (
                str(item.item_id or "").strip() in post_distance_item_ids
                or str(item.item_id or "").strip() not in helper_item_ids
            )
            and str(item.result or "").strip().upper() in {"OK", "NG", _PASS_RESULT}
        ]
        self._apply_decision_policy(request, decision_item_results)
        final_result = (
            "OK"
            if not decision_item_results or all(_is_passing_result(item.result) for item in decision_item_results)
            else "NG"
        )
        match_ms = max(
            (self._extract_timing_fields(row)[0] for row in item_rows),
            default=0.0,
        )
        infer_ms = sum(self._extract_timing_fields(row)[1] for row in item_rows)
        total_ms = match_ms + infer_ms if (match_ms > 0.0 or infer_ms > 0.0) else 0.0
        decision_has_ng = any(_normalized_result(item.result) == "NG" for item in decision_item_results)
        if post_distance_results and not decision_has_ng:
            distance_detail = "; ".join(
                self._strip_timing_tokens(item.detail)
                for item in post_distance_results
                if self._strip_timing_tokens(item.detail)
            )
            independent_count = sum(
                1
                for item in decision_item_results
                if str(item.item_id or "").strip() not in post_distance_item_ids
            )
            camera_detail_parts = [distance_detail] if distance_detail else ["Distance"]
            if independent_count:
                camera_detail_parts.append(f"{independent_count} other item(s) OK")
            if match_ms > 0:
                camera_detail_parts.append(f"match={match_ms:.1f}ms")
            if infer_ms > 0:
                camera_detail_parts.append(f"infer={infer_ms:.1f}ms")
            camera_detail = " ".join(camera_detail_parts)
        else:
            camera_detail = self._build_camera_detail(
                decision_item_results,
                match_ms=match_ms,
                infer_ms=infer_ms,
            )
        raw_measurements = tuple(
            dict(row.get("measurement", {}) or {})
            for row in item_rows
            if isinstance(row.get("measurement"), dict)
        )
        post_distance_measurements = tuple(
            measurement
            for measurement in raw_measurements
            if str(measurement.get("type", "") or "").strip() in LINE_DISTANCE_ALGORITHMS
            or str(measurement.get("type", "") or "").strip() in CENTER_DISTANCE_ALGORITHMS
            or str(measurement.get("type", "") or "").strip() == POINT_LINE_DISTANCE_ALGORITHM
            or str(measurement.get("type", "") or "").strip() == MULTI_PIN_TIP_HEIGHT_ALGORITHM
        )
        measurements = post_distance_measurements or raw_measurements

        return InspectionExecutionResponse(
            camera_id=request.camera_id,
            result=final_result,
            detail=camera_detail,
            raw_row={
                "pred": final_result,
                "item_rows": item_rows,
                "match_ms": match_ms,
                "infer_ms": infer_ms,
                "total_ms": total_ms,
                "timing_breakdown": batch_timing_breakdown,
            },
            match_ms=match_ms,
            infer_ms=infer_ms,
            total_ms=total_ms,
            item_results=item_results,
            roi_shapes=roi_shapes,
            measurements=measurements,
        )

    def _predict_image_for_item(self, path: str, **kwargs) -> dict:
        try:
            return self._predictor.predict_image(path, **kwargs)
        except TypeError as exc:
            if "params_override" not in str(exc):
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("params_override", None)
            return self._predictor.predict_image(path, **fallback_kwargs)

    def _apply_decision_policy(
        self,
        request: InspectionExecutionRequest,
        decision_item_results: List[InspectionItemResult],
    ) -> None:
        policy = self._load_decision_policy()
        if not bool(policy.get("enabled", False)):
            return
        groups = policy.get("groups", [])
        if not isinstance(groups, list):
            return
        for raw_group in groups:
            if not isinstance(raw_group, Mapping):
                continue
            self._apply_decision_group_policy(request, decision_item_results, raw_group)

    def _apply_decision_group_policy(
        self,
        request: InspectionExecutionRequest,
        decision_item_results: List[InspectionItemResult],
        group: Mapping[str, object],
    ) -> None:
        camera_id = str(group.get("camera_id", "") or "").strip()
        if camera_id and camera_id != str(request.camera_id or "").strip():
            return
        target_labels = self._normalized_string_list(
            group.get("roi_labels", group.get("item_ids", group.get("items", [])))
        )
        if not target_labels:
            return
        min_ok = self._positive_int(group.get("min_ok"), default=len(target_labels))
        if min_ok <= 0:
            return
        by_label = {
            str(item.roi_label or "").strip(): item
            for item in decision_item_results
            if str(item.roi_label or "").strip()
        }
        by_item_id = {
            str(item.item_id or "").strip(): item
            for item in decision_item_results
            if str(item.item_id or "").strip()
        }
        group_items: List[InspectionItemResult] = []
        for target in target_labels:
            item = by_label.get(target) or by_item_id.get(target)
            if item is None:
                return
            if _normalized_result(item.result) not in {"OK", "NG", _PASS_RESULT}:
                return
            group_items.append(item)
        if len(set(id(item) for item in group_items)) != len(target_labels):
            return

        ok_count = sum(1 for item in group_items if _is_passing_result(item.result))
        if ok_count < min_ok:
            return
        ng_items = [item for item in group_items if _normalized_result(item.result) == "NG"]
        if not ng_items:
            return
        group_name = str(group.get("name", "") or "decision_group").strip() or "decision_group"
        reason = f"{group_name} {ok_count}/{len(group_items)} OK, min_ok={min_ok}"
        for item in ng_items:
            item.result = _PASS_RESULT
            item.detail = _append_detail(item.detail, f"raw=NG {reason}")

    def _load_decision_policy(self) -> dict:
        if self._decision_policy_override is not None:
            return dict(self._decision_policy_override)
        path = self._decision_policy_path()
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _decision_policy_path(self) -> str:
        product_dir = self._product_dir_from_predictor()
        if not product_dir:
            return ""
        return os.path.join(product_dir, _DECISION_POLICY_FILENAME)

    def _product_dir_from_predictor(self) -> str:
        session = getattr(self._predictor, "session", None)
        if session is None:
            tool_page = getattr(self._predictor, "tool_page", None)
            session = getattr(tool_page, "session", None)
        product_dir = str(getattr(session, "product_dir", "") or "").strip()
        return product_dir if product_dir and os.path.isdir(product_dir) else ""

    @staticmethod
    def _normalized_string_list(value: object) -> List[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item or "").strip() for item in value if str(item or "").strip()]

    @staticmethod
    def _positive_int(value: object, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return parsed if parsed > 0 else int(default)

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)

    @staticmethod
    def _optional_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "y", "on", "enable", "enabled", "是", "启用"}

    @classmethod
    def _item_param_float(cls, item: InspectionItem, key: str) -> float | None:
        params = dict(getattr(item, "params", {}) or {})
        return cls._optional_float(params.get(key))

    @classmethod
    def _row_value(cls, row: dict) -> float | None:
        try:
            value = row.get("value")
        except Exception:
            return None
        return cls._optional_float(value)

    @staticmethod
    def _row_unit(row: dict) -> str:
        try:
            measurement = row.get("measurement")
        except Exception:
            measurement = None
        if isinstance(measurement, dict):
            unit = str(measurement.get("unit", "") or "").strip()
            if unit:
                return unit
        try:
            return str(row.get("unit", "") or "").strip()
        except Exception:
            return ""

    @classmethod
    def _compensated_value(
        cls,
        value: float,
        params: dict,
    ) -> tuple[float, float, bool, float, float]:
        raw_value = float(value)
        enabled = cls._optional_bool(params.get("compensation_enabled"))
        slope = cls._optional_float(params.get("compensation_slope"))
        intercept = cls._optional_float(params.get("compensation_intercept"))
        k = 1.0 if slope is None else float(slope)
        b = 0.0 if intercept is None else float(intercept)
        if not enabled:
            return raw_value, raw_value, False, k, b
        return float(k * raw_value + b), raw_value, True, k, b

    @staticmethod
    def _measurement_decimals(unit: object) -> int:
        return 3

    @classmethod
    def _round_measurement_value(cls, value: float, unit: object) -> float:
        decimals = cls._measurement_decimals(unit)
        return float(f"{float(value):.{decimals}f}")

    @classmethod
    def _format_measurement_value(cls, value: float, unit: object) -> str:
        unit_text = str(unit or "").strip().lower() or "px"
        decimals = cls._measurement_decimals(unit_text)
        return f"{float(value):.{decimals}f}{unit_text}"

    @classmethod
    def _format_measurement_limit(cls, value: float | None, unit: object) -> str:
        if value is None:
            return "-"
        return cls._format_measurement_value(float(value), unit)

    @staticmethod
    def _point_tuple(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    @classmethod
    def _line_segment_from_row(cls, row: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
        measurement = row.get("measurement")
        if not isinstance(measurement, dict):
            return None
        raw_segment = measurement.get("line_segment")
        if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) < 2:
            return None
        p0 = cls._point_tuple(raw_segment[0])
        p1 = cls._point_tuple(raw_segment[1])
        if p0 is None or p1 is None:
            return None
        return p0, p1

    @classmethod
    def _line_points_from_row(cls, row: dict) -> np.ndarray:
        measurement = row.get("measurement")
        if not isinstance(measurement, dict):
            return np.empty((0, 2), dtype=np.float32)
        raw_points = measurement.get("edge_points")
        points: list[tuple[float, float]] = []
        if isinstance(raw_points, (list, tuple)):
            for point in raw_points:
                parsed = cls._point_tuple(point)
                if parsed is not None:
                    points.append(parsed)
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)

    @staticmethod
    def _line_segment_length(
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> float:
        (x0, y0), (x1, y1) = segment
        return float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))

    @classmethod
    def _line_from_segment(
        cls,
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> FittedLine:
        (x0, y0), (x1, y1) = segment
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            raise RuntimeError("line distance segment invalid")
        return FittedLine(
            vx=float(dx / length),
            vy=float(dy / length),
            x0=float((float(x0) + float(x1)) * 0.5),
            y0=float((float(y0) + float(y1)) * 0.5),
            residual=0.0,
            point_count=2,
        )

    @staticmethod
    def _dominant_line_axis(
        segment_a: tuple[tuple[float, float], tuple[float, float]],
        segment_b: tuple[tuple[float, float], tuple[float, float]],
    ) -> str:
        (ax0, ay0), (ax1, ay1) = segment_a
        (bx0, by0), (bx1, by1) = segment_b
        dx = abs(float(ax1) - float(ax0)) + abs(float(bx1) - float(bx0))
        dy = abs(float(ay1) - float(ay0)) + abs(float(by1) - float(by0))
        return "y" if dy >= dx else "x"

    @staticmethod
    def _axis_range_from_points(points: np.ndarray, axis: str) -> tuple[float, float] | None:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return None
        coord = pts[:, 1] if axis == "y" else pts[:, 0]
        return float(np.min(coord)), float(np.max(coord))

    @staticmethod
    def _axis_range_from_segment(
        segment: tuple[tuple[float, float], tuple[float, float]],
        axis: str,
    ) -> tuple[float, float]:
        index = 1 if axis == "y" else 0
        v0 = float(segment[0][index])
        v1 = float(segment[1][index])
        return min(v0, v1), max(v0, v1)

    @staticmethod
    def _point_on_line_at_axis(line: FittedLine, axis: str, value: float) -> tuple[float, float] | None:
        if axis == "y":
            if abs(float(line.vy)) <= 1e-12:
                return None
            t = (float(value) - float(line.y0)) / float(line.vy)
            return float(line.x0) + t * float(line.vx), float(value)
        if abs(float(line.vx)) <= 1e-12:
            return None
        t = (float(value) - float(line.x0)) / float(line.vx)
        return float(value), float(line.y0) + t * float(line.vy)

    @classmethod
    def _segment_for_line_interval(
        cls,
        line: FittedLine,
        *,
        axis: str,
        interval: tuple[float, float] | None,
        fallback: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if interval is None:
            return fallback
        lo, hi = interval
        p0 = cls._point_on_line_at_axis(line, axis, lo)
        p1 = cls._point_on_line_at_axis(line, axis, hi)
        if p0 is None or p1 is None:
            return fallback
        if math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])) <= 1e-12:
            return fallback
        return p0, p1

    @staticmethod
    def _line_intersection(
        point_a: tuple[float, float],
        dir_a: tuple[float, float],
        point_b: tuple[float, float],
        dir_b: tuple[float, float],
    ) -> tuple[float, float] | None:
        ax, ay = point_a
        avx, avy = dir_a
        bx, by = point_b
        bvx, bvy = dir_b
        denom = float(avx) * float(bvy) - float(avy) * float(bvx)
        if abs(denom) <= 1e-12:
            return None
        t = ((float(bx) - float(ax)) * float(bvy) - (float(by) - float(ay)) * float(bvx)) / denom
        return float(ax) + t * float(avx), float(ay) + t * float(avy)

    @staticmethod
    def _angle_delta_from_lines(line_a: FittedLine, line_b: FittedLine) -> float:
        dot = abs(float(line_a.vx) * float(line_b.vx) + float(line_a.vy) * float(line_b.vy))
        dot = max(-1.0, min(1.0, dot))
        return float(math.degrees(math.acos(dot)))

    @classmethod
    def _fit_line_for_common_interval(
        cls,
        *,
        row: dict,
        segment: tuple[tuple[float, float], tuple[float, float]],
        axis: str,
        interval: tuple[float, float] | None,
        context: str,
    ) -> tuple[FittedLine, tuple[tuple[float, float], tuple[float, float]], int]:
        points = cls._line_points_from_row(row)
        fit_points = points
        if interval is not None and points.size > 0:
            coord = points[:, 1] if axis == "y" else points[:, 0]
            fit_points = points[(coord >= float(interval[0])) & (coord <= float(interval[1]))]
        if len(fit_points) >= 2:
            min_points = min(10, int(len(fit_points)))
            line, filtered_points = fit_line_filtered(fit_points, min_points=min_points, context=context)
            segment_fit = cls._segment_for_line_interval(
                line,
                axis=axis,
                interval=interval,
                fallback=segment,
            )
            return line, segment_fit, int(len(filtered_points))
        line = cls._line_from_segment(segment)
        segment_fit = cls._segment_for_line_interval(
            line,
            axis=axis,
            interval=interval,
            fallback=segment,
        )
        return line, segment_fit, 0

    @classmethod
    def _reference_normal_distance_between_rows(
        cls,
        *,
        row_a: dict,
        row_b: dict,
        segment_a: tuple[tuple[float, float], tuple[float, float]],
        segment_b: tuple[tuple[float, float], tuple[float, float]],
        item_a_id: str,
        item_b_id: str,
    ) -> dict:
        axis = cls._dominant_line_axis(segment_a, segment_b)
        points_a = cls._line_points_from_row(row_a)
        points_b = cls._line_points_from_row(row_b)
        range_a = cls._axis_range_from_points(points_a, axis) or cls._axis_range_from_segment(segment_a, axis)
        range_b = cls._axis_range_from_points(points_b, axis) or cls._axis_range_from_segment(segment_b, axis)
        lo = max(float(range_a[0]), float(range_b[0]))
        hi = min(float(range_a[1]), float(range_b[1]))
        common_interval = (lo, hi) if hi > lo + 1e-6 else None

        line_a, common_segment_a, point_count_a = cls._fit_line_for_common_interval(
            row=row_a,
            segment=segment_a,
            axis=axis,
            interval=common_interval,
            context=f"{item_a_id} common-{axis}",
        )
        line_b, common_segment_b, point_count_b = cls._fit_line_for_common_interval(
            row=row_b,
            segment=segment_b,
            axis=axis,
            interval=common_interval,
            context=f"{item_b_id} common-{axis}",
        )

        length_a = cls._line_segment_length(segment_a)
        length_b = cls._line_segment_length(segment_b)
        reference_is_a = length_a >= length_b
        ref_line = line_a if reference_is_a else line_b
        ref_segment = common_segment_a if reference_is_a else common_segment_b
        ref_item_id = item_a_id if reference_is_a else item_b_id

        ref_mid = (
            (float(ref_segment[0][0]) + float(ref_segment[1][0])) * 0.5,
            (float(ref_segment[0][1]) + float(ref_segment[1][1])) * 0.5,
        )
        normal = (-float(ref_line.vy), float(ref_line.vx))
        p_a = cls._line_intersection(
            (float(line_a.x0), float(line_a.y0)),
            (float(line_a.vx), float(line_a.vy)),
            ref_mid,
            normal,
        )
        p_b = cls._line_intersection(
            (float(line_b.x0), float(line_b.y0)),
            (float(line_b.vx), float(line_b.vy)),
            ref_mid,
            normal,
        )
        if p_a is None or p_b is None:
            raise RuntimeError("reference-normal line intersection failed")
        distance_px = float(math.hypot(float(p_b[0]) - float(p_a[0]), float(p_b[1]) - float(p_a[1])))
        return {
            "distance_px": distance_px,
            "angle_delta": cls._angle_delta_from_lines(line_a, line_b),
            "dimension_segment": (p_a, p_b),
            "distance_mode": "reference_normal_intersection",
            "reference_line_item_id": ref_item_id,
            "common_axis": axis,
            "common_interval": list(common_interval) if common_interval is not None else None,
            "line_a_segment": common_segment_a,
            "line_b_segment": common_segment_b,
            "line_a_fit": line_a.to_dict(),
            "line_b_fit": line_b.to_dict(),
            "line_a_common_points": point_count_a,
            "line_b_common_points": point_count_b,
        }

    @staticmethod
    def _line_item_ref(item: InspectionItem) -> str:
        return str(getattr(item, "item_id", "") or "").strip()

    @classmethod
    def _resolve_line_distance_pair(
        cls,
        *,
        distance_item: InspectionItem,
        line_items: List[InspectionItem],
        line_rows_by_item_id: dict[str, dict],
    ) -> tuple[InspectionItem, dict, InspectionItem, dict] | None:
        params = dict(getattr(distance_item, "params", {}) or {})
        line_a_id = str(params.get("line_a_item_id", "") or "").strip()
        line_b_id = str(params.get("line_b_item_id", "") or "").strip()
        items_by_id = {
            cls._line_item_ref(item): item
            for item in line_items
            if cls._line_item_ref(item)
        }
        if line_a_id and line_b_id:
            item_a = items_by_id.get(line_a_id)
            item_b = items_by_id.get(line_b_id)
            row_a = line_rows_by_item_id.get(line_a_id)
            row_b = line_rows_by_item_id.get(line_b_id)
            if item_a is not None and item_b is not None and row_a is not None and row_b is not None:
                return item_a, row_a, item_b, row_b

        candidates: list[tuple[InspectionItem, dict]] = []
        for item in line_items:
            key = cls._line_item_ref(item)
            row = line_rows_by_item_id.get(key)
            if row is None or cls._line_segment_from_row(row) is None:
                continue
            candidates.append((item, row))
        if len(candidates) < 2:
            return None
        return candidates[0][0], candidates[0][1], candidates[1][0], candidates[1][1]

    @staticmethod
    def _center_item_ref(item: InspectionItem) -> str:
        return str(getattr(item, "item_id", "") or "").strip()

    @staticmethod
    def _center_point_from_row(row: dict) -> tuple[float, float] | None:
        if not _is_passing_result(row.get("pred")):
            return None
        measurement = row.get("measurement")
        if not isinstance(measurement, dict):
            return None
        raw_points = measurement.get("center_points")
        if isinstance(raw_points, (list, tuple)) and raw_points:
            point = raw_points[0]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    return float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    return None
        raw_center = measurement.get("center", measurement.get("center_xy"))
        if isinstance(raw_center, (list, tuple)) and len(raw_center) >= 2:
            try:
                return float(raw_center[0]), float(raw_center[1])
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def _resolve_point_line_pair(
        cls,
        *,
        distance_item: InspectionItem,
        source_items: List[InspectionItem],
        rows_by_item_id: dict[str, dict],
    ) -> tuple[InspectionItem, dict, InspectionItem, dict] | None:
        params = dict(getattr(distance_item, "params", {}) or {})
        point_item_id = str(params.get("point_item_id", "") or "").strip()
        line_item_id = str(params.get("line_item_id", "") or "").strip()
        if not point_item_id or not line_item_id or point_item_id == line_item_id:
            return None
        items_by_id = {
            cls._center_item_ref(item): item
            for item in source_items
            if cls._center_item_ref(item)
        }
        point_item = items_by_id.get(point_item_id)
        line_item = items_by_id.get(line_item_id)
        point_row = rows_by_item_id.get(point_item_id)
        line_row = rows_by_item_id.get(line_item_id)
        if point_item is None or line_item is None or point_row is None or line_row is None:
            return None
        if cls._center_point_from_row(point_row) is None:
            return None
        if not _is_passing_result(line_row.get("pred")) or cls._line_segment_from_row(line_row) is None:
            return None
        return point_item, point_row, line_item, line_row

    @staticmethod
    def _point_to_line_distance(
        point: tuple[float, float],
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, tuple[float, float]]:
        px, py = float(point[0]), float(point[1])
        (x0, y0), (x1, y1) = segment
        vx, vy = float(x1) - float(x0), float(y1) - float(y0)
        denominator = vx * vx + vy * vy
        if denominator <= 1e-12:
            raise RuntimeError("reference line segment is too short")
        factor = ((px - float(x0)) * vx + (py - float(y0)) * vy) / denominator
        foot = (float(x0) + factor * vx, float(y0) + factor * vy)
        return float(math.hypot(px - foot[0], py - foot[1])), foot

    @classmethod
    def _resolve_center_distance_pair(
        cls,
        *,
        distance_item: InspectionItem,
        center_items: List[InspectionItem],
        rows_by_item_id: dict[str, dict],
    ) -> tuple[InspectionItem, dict, InspectionItem, dict] | None:
        params = dict(getattr(distance_item, "params", {}) or {})
        center_a_id = str(params.get("center_a_item_id", "") or "").strip()
        center_b_id = str(params.get("center_b_item_id", "") or "").strip()
        items_by_id = {
            cls._center_item_ref(item): item
            for item in center_items
            if cls._center_item_ref(item)
        }
        if center_a_id or center_b_id:
            if not center_a_id or not center_b_id or center_a_id == center_b_id:
                return None
            item_a = items_by_id.get(center_a_id)
            item_b = items_by_id.get(center_b_id)
            row_a = rows_by_item_id.get(center_a_id)
            row_b = rows_by_item_id.get(center_b_id)
            if (
                item_a is not None
                and item_b is not None
                and row_a is not None
                and row_b is not None
                and cls._center_point_from_row(row_a) is not None
                and cls._center_point_from_row(row_b) is not None
            ):
                return item_a, row_a, item_b, row_b
            return None

        candidates: list[tuple[InspectionItem, dict]] = []
        for item in center_items:
            algorithm = str(getattr(item, "algorithm_code", "") or "").strip()
            if algorithm != BRIGHT_BLOCK_CENTER_ALGORITHM:
                continue
            key = cls._center_item_ref(item)
            row = rows_by_item_id.get(key)
            if row is None or cls._center_point_from_row(row) is None:
                continue
            candidates.append((item, row))
        if len(candidates) < 2:
            return None
        return candidates[0][0], candidates[0][1], candidates[1][0], candidates[1][1]

    @staticmethod
    def _center_distance_info(
        center_a: tuple[float, float],
        center_b: tuple[float, float],
        distance_mode: object,
    ) -> tuple[float, tuple[tuple[float, float], tuple[float, float]], str]:
        mode = str(distance_mode or "vertical").strip().lower()
        if mode not in {"vertical", "horizontal", "euclidean"}:
            mode = "vertical"
        ax, ay = float(center_a[0]), float(center_a[1])
        bx, by = float(center_b[0]), float(center_b[1])
        if mode == "horizontal":
            return abs(bx - ax), ((ax, ay), (bx, ay)), mode
        if mode == "euclidean":
            return float(math.hypot(bx - ax, by - ay)), ((ax, ay), (bx, by)), mode
        return abs(by - ay), ((ax, ay), (ax, by)), mode

    @staticmethod
    def _distance_between_segments(
        segment_a: tuple[tuple[float, float], tuple[float, float]],
        segment_b: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float]:
        (ax0, ay0), (ax1, ay1) = segment_a
        (bx0, by0), (bx1, by1) = segment_b
        avx = ax1 - ax0
        avy = ay1 - ay0
        bvx = bx1 - bx0
        bvy = by1 - by0
        an = math.hypot(avx, avy)
        bn = math.hypot(bvx, bvy)
        if an <= 1e-12 or bn <= 1e-12:
            raise RuntimeError("line distance segment invalid")
        d_ab = abs(avx * (by0 - ay0) - avy * (bx0 - ax0)) / an
        d_ba = abs(bvx * (ay0 - by0) - bvy * (ax0 - bx0)) / bn
        dot = abs(avx * bvx + avy * bvy) / (an * bn)
        dot = max(-1.0, min(1.0, dot))
        angle_delta = math.degrees(math.acos(dot))
        return float((d_ab + d_ba) * 0.5), float(angle_delta)

    @staticmethod
    def _dimension_segment(
        segment_a: tuple[tuple[float, float], tuple[float, float]],
        segment_b: tuple[tuple[float, float], tuple[float, float]],
        distance_px: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        (ax0, ay0), (ax1, ay1) = segment_a
        (bx0, by0), (bx1, by1) = segment_b
        ac = ((ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5)
        bc = ((bx0 + bx1) * 0.5, (by0 + by1) * 0.5)
        avx = ax1 - ax0
        avy = ay1 - ay0
        an = math.hypot(avx, avy)
        if an <= 1e-12:
            return ac, bc
        nx = -avy / an
        ny = avx / an
        if (bc[0] - ac[0]) * nx + (bc[1] - ac[1]) * ny < 0.0:
            nx = -nx
            ny = -ny
        return ac, (ac[0] + nx * float(distance_px), ac[1] + ny * float(distance_px))

    @classmethod
    def _build_find_line_distance_row(
        cls,
        *,
        distance_item: InspectionItem,
        line_items: List[InspectionItem],
        line_rows_by_item_id: dict[str, dict],
        camera_id: str,
        image_path: str,
    ) -> dict | None:
        pair = cls._resolve_line_distance_pair(
            distance_item=distance_item,
            line_items=line_items,
            line_rows_by_item_id=line_rows_by_item_id,
        )
        if pair is None:
            return None

        item_a, row_a, item_b, row_b = pair
        segment_a = cls._line_segment_from_row(row_a)
        segment_b = cls._line_segment_from_row(row_b)
        if segment_a is None or segment_b is None:
            return None
        distance_algorithm = str(getattr(distance_item, "algorithm_code", "") or "").strip() or "line_distance"
        if distance_algorithm == LINE_DISTANCE_REF_NORMAL_ALGORITHM:
            distance_info = cls._reference_normal_distance_between_rows(
                row_a=row_a,
                row_b=row_b,
                segment_a=segment_a,
                segment_b=segment_b,
                item_a_id=cls._line_item_ref(item_a),
                item_b_id=cls._line_item_ref(item_b),
            )
            distance_px = float(distance_info["distance_px"])
            angle_delta = float(distance_info["angle_delta"])
            dimension_segment = distance_info["dimension_segment"]
        else:
            distance_px, angle_delta = cls._distance_between_segments(segment_a, segment_b)
            dimension_segment = cls._dimension_segment(segment_a, segment_b, distance_px)
            distance_info = {
                "distance_mode": "average_cross_line_distance",
                "dimension_segment": dimension_segment,
            }

        params = dict(getattr(distance_item, "params", {}) or {})
        unit = str(params.get("limit_unit", "px") or "px").strip().lower()
        if unit not in {"px", "mm"}:
            unit = "px"
        value = float(distance_px)
        pixel_size = cls._optional_float(params.get("pixel_size_mm")) or 0.0
        if unit == "mm":
            if pixel_size <= 0.0:
                pixel_size = cls._item_param_float(item_a, "pixel_size_mm") or 0.0
            if pixel_size <= 0.0:
                pixel_size = cls._item_param_float(item_b, "pixel_size_mm") or 0.0
            if pixel_size <= 0.0:
                raise RuntimeError("pixel_size_mm is required when line-distance limits use mm")
            value = float(distance_px * pixel_size)
        if unit == "mm":
            value, raw_value, compensation_enabled, compensation_slope, compensation_intercept = cls._compensated_value(
                value,
                params,
            )
        else:
            _ignored_value, raw_value, _configured_compensation_enabled, compensation_slope, compensation_intercept = cls._compensated_value(
                value,
                params,
            )
            value = raw_value
            compensation_enabled = False
        reported_value = cls._round_measurement_value(value, unit)
        lower = cls._optional_float(params.get("lower_limit", params.get(f"lower_limit_{unit}")))
        upper = cls._optional_float(params.get("upper_limit", params.get(f"upper_limit_{unit}")))
        ok = True
        if lower is not None and reported_value < lower:
            ok = False
        if upper is not None and reported_value > upper:
            ok = False

        name_a = str(getattr(item_a, "display_name", "") or getattr(item_a, "roi_label", "") or "LineA")
        name_b = str(getattr(item_b, "display_name", "") or getattr(item_b, "roi_label", "") or "LineB")
        display_name = str(getattr(distance_item, "display_name", "") or getattr(distance_item, "item_id", "") or "Line Distance")
        detail = (
            f"distance={cls._format_measurement_value(reported_value, unit)}"
            f" angle_delta={angle_delta:.3f}deg"
            f" lines={name_a}/{name_b}"
        )
        if compensation_enabled:
            detail += (
                f" compensation=k={compensation_slope:.6g},b={compensation_intercept:.6g}"
            )
        detail += f" raw={distance_px:.3f}px"
        if distance_algorithm == LINE_DISTANCE_REF_NORMAL_ALGORITHM:
            detail += f" mode=ref_normal ref={distance_info.get('reference_line_item_id', '-') or '-'}"
        if lower is not None or upper is not None:
            detail += (
                f" spec={cls._format_measurement_limit(lower, unit)}"
                f"..{cls._format_measurement_limit(upper, unit)}"
            )

        return {
            "file_path": image_path,
            "file_name": display_name,
            "gt": "",
            "pred": "OK" if ok else "NG",
            "diff": angle_delta,
            "sim_ok": None,
            "sim_ng": None,
            "value": reported_value,
            "unit": unit,
            "threshold": upper,
            "match_ms": max(
                (
                    float(row.get("match_ms") or 0.0)
                    for row in (row_a, row_b)
                    if row.get("match_ms") is not None
                ),
                default=0.0,
            ),
            "infer_ms": 0.0,
            "total_ms": 0.0,
            "json_name": str(row_a.get("json_name", row_b.get("json_name", "")) or ""),
            "detail": detail,
            "algorithm": distance_algorithm,
            "tool_name": display_name,
            "camera_id": camera_id,
            "roi_label": str(getattr(distance_item, "roi_label", "") or ""),
            "params": params,
            "measurement": {
                "type": distance_algorithm,
                "distance_px": distance_px,
                "distance": reported_value,
                "unit": unit,
                "pixel_size_mm": pixel_size,
                "compensation_enabled": compensation_enabled,
                "compensation_slope": compensation_slope,
                "compensation_intercept": compensation_intercept,
                "angle_delta_deg": angle_delta,
                "dimension_segment": [[float(x), float(y)] for x, y in dimension_segment],
                "distance_mode": str(distance_info.get("distance_mode", "") or ""),
                "reference_line_item_id": str(distance_info.get("reference_line_item_id", "") or ""),
                "common_axis": str(distance_info.get("common_axis", "") or ""),
                "common_interval": distance_info.get("common_interval"),
                "line_a_segment": (
                    [[float(x), float(y)] for x, y in distance_info["line_a_segment"]]
                    if distance_info.get("line_a_segment") is not None
                    else None
                ),
                "line_b_segment": (
                    [[float(x), float(y)] for x, y in distance_info["line_b_segment"]]
                    if distance_info.get("line_b_segment") is not None
                    else None
                ),
                "line_a_fit": distance_info.get("line_a_fit"),
                "line_b_fit": distance_info.get("line_b_fit"),
                "line_a_common_points": distance_info.get("line_a_common_points"),
                "line_b_common_points": distance_info.get("line_b_common_points"),
                "label": cls._format_measurement_value(reported_value, unit),
                "pred": "OK" if ok else "NG",
                "line_a_item_id": cls._line_item_ref(item_a),
                "line_b_item_id": cls._line_item_ref(item_b),
                "line_a": dict(row_a.get("measurement", {}) or {}),
                "line_b": dict(row_b.get("measurement", {}) or {}),
            },
        }

    @classmethod
    def _build_point_line_distance_row(
        cls,
        *,
        distance_item: InspectionItem,
        source_items: List[InspectionItem],
        rows_by_item_id: dict[str, dict],
        camera_id: str,
        image_path: str,
    ) -> dict | None:
        pair = cls._resolve_point_line_pair(
            distance_item=distance_item,
            source_items=source_items,
            rows_by_item_id=rows_by_item_id,
        )
        if pair is None:
            return None
        point_item, point_row, line_item, line_row = pair
        point = cls._center_point_from_row(point_row)
        line_segment = cls._line_segment_from_row(line_row)
        if point is None or line_segment is None:
            return None
        distance_px, foot = cls._point_to_line_distance(point, line_segment)

        params = dict(getattr(distance_item, "params", {}) or {})
        unit = str(params.get("limit_unit", "px") or "px").strip().lower()
        if unit not in {"px", "mm"}:
            unit = "px"
        value = float(distance_px)
        pixel_size = cls._optional_float(params.get("pixel_size_mm")) or 0.0
        if unit == "mm":
            if pixel_size <= 0.0:
                pixel_size = cls._item_param_float(point_item, "pixel_size_mm") or 0.0
            if pixel_size <= 0.0:
                pixel_size = cls._item_param_float(line_item, "pixel_size_mm") or 0.0
            if pixel_size <= 0.0:
                raise RuntimeError("pixel_size_mm is required when point-line limits use mm")
            value = float(distance_px * pixel_size)
            value, raw_value, compensation_enabled, compensation_slope, compensation_intercept = cls._compensated_value(
                value,
                params,
            )
        else:
            _ignored_value, raw_value, _configured_compensation_enabled, compensation_slope, compensation_intercept = cls._compensated_value(
                value,
                params,
            )
            value = raw_value
            compensation_enabled = False
        reported_value = cls._round_measurement_value(value, unit)
        lower = cls._optional_float(params.get("lower_limit", params.get(f"lower_limit_{unit}")))
        upper = cls._optional_float(params.get("upper_limit", params.get(f"upper_limit_{unit}")))
        ok = True
        if lower is not None and reported_value < lower:
            ok = False
        if upper is not None and reported_value > upper:
            ok = False

        point_name = str(getattr(point_item, "display_name", "") or getattr(point_item, "roi_label", "") or "Point")
        line_name = str(getattr(line_item, "display_name", "") or getattr(line_item, "roi_label", "") or "Line")
        display_name = str(getattr(distance_item, "display_name", "") or getattr(distance_item, "item_id", "") or "Point-Line Distance")
        detail = (
            f"point_line_distance={cls._format_measurement_value(reported_value, unit)}"
            f" point={point_name} line={line_name}"
            f" raw={distance_px:.3f}px"
        )
        if compensation_enabled:
            detail += f" compensation=k={compensation_slope:.6g},b={compensation_intercept:.6g}"
        if lower is not None or upper is not None:
            detail += (
                f" spec={cls._format_measurement_limit(lower, unit)}"
                f"..{cls._format_measurement_limit(upper, unit)}"
            )
        pred = "OK" if ok else "NG"
        dimension_segment = (point, foot)
        return {
            "file_path": image_path,
            "file_name": display_name,
            "gt": "",
            "pred": pred,
            "diff": 0.0,
            "sim_ok": None,
            "sim_ng": None,
            "value": reported_value,
            "unit": unit,
            "threshold": upper,
            "match_ms": max(
                (
                    float(row.get("match_ms") or 0.0)
                    for row in (point_row, line_row)
                    if row.get("match_ms") is not None
                ),
                default=0.0,
            ),
            "infer_ms": 0.0,
            "total_ms": 0.0,
            "json_name": str(point_row.get("json_name", line_row.get("json_name", "")) or ""),
            "detail": detail,
            "algorithm": POINT_LINE_DISTANCE_ALGORITHM,
            "tool_name": display_name,
            "camera_id": camera_id,
            "roi_label": str(getattr(distance_item, "roi_label", "") or ""),
            "params": params,
            "measurement": {
                "type": POINT_LINE_DISTANCE_ALGORITHM,
                "distance_px": float(distance_px),
                "distance": reported_value,
                "unit": unit,
                "pixel_size_mm": pixel_size,
                "compensation_enabled": compensation_enabled,
                "compensation_slope": compensation_slope,
                "compensation_intercept": compensation_intercept,
                "dimension_segment": [[float(x), float(y)] for x, y in dimension_segment],
                "line_segment": [[float(x), float(y)] for x, y in line_segment],
                "line_a_segment": [[float(x), float(y)] for x, y in line_segment],
                "center_points": [[float(point[0]), float(point[1])]],
                "projection_point": [float(foot[0]), float(foot[1])],
                "distance_mode": "point_to_line_normal",
                "label": cls._format_measurement_value(reported_value, unit),
                "pred": pred,
                "point_item_id": cls._center_item_ref(point_item),
                "line_item_id": cls._line_item_ref(line_item),
                "point_source": dict(point_row.get("measurement", {}) or {}),
                "line_source": dict(line_row.get("measurement", {}) or {}),
            },
        }

    @classmethod
    def _build_center_distance_row(
        cls,
        *,
        distance_item: InspectionItem,
        center_items: List[InspectionItem],
        rows_by_item_id: dict[str, dict],
        camera_id: str,
        image_path: str,
    ) -> dict | None:
        pair = cls._resolve_center_distance_pair(
            distance_item=distance_item,
            center_items=center_items,
            rows_by_item_id=rows_by_item_id,
        )
        if pair is None:
            return None

        item_a, row_a, item_b, row_b = pair
        center_a = cls._center_point_from_row(row_a)
        center_b = cls._center_point_from_row(row_b)
        if center_a is None or center_b is None:
            return None

        params = dict(getattr(distance_item, "params", {}) or {})
        distance_px, dimension_segment, distance_mode = cls._center_distance_info(
            center_a,
            center_b,
            params.get("distance_mode", "vertical"),
        )
        unit = str(params.get("limit_unit", "px") or "px").strip().lower()
        if unit not in {"px", "mm"}:
            unit = "px"
        value = float(distance_px)
        pixel_size = cls._optional_float(params.get("pixel_size_mm")) or 0.0
        if unit == "mm":
            if pixel_size <= 0.0:
                pixel_size = cls._item_param_float(item_a, "pixel_size_mm") or 0.0
            if pixel_size <= 0.0:
                pixel_size = cls._item_param_float(item_b, "pixel_size_mm") or 0.0
            if pixel_size <= 0.0:
                raise RuntimeError("pixel_size_mm is required when center-distance limits use mm")
            value = float(distance_px * pixel_size)
        if unit == "mm":
            value, raw_value, compensation_enabled, compensation_slope, compensation_intercept = cls._compensated_value(
                value,
                params,
            )
        else:
            _ignored_value, raw_value, _configured_compensation_enabled, compensation_slope, compensation_intercept = cls._compensated_value(
                value,
                params,
            )
            value = raw_value
            compensation_enabled = False
        reported_value = cls._round_measurement_value(value, unit)
        lower = cls._optional_float(params.get("lower_limit", params.get(f"lower_limit_{unit}")))
        upper = cls._optional_float(params.get("upper_limit", params.get(f"upper_limit_{unit}")))
        ok = True
        if lower is not None and reported_value < lower:
            ok = False
        if upper is not None and reported_value > upper:
            ok = False

        name_a = str(getattr(item_a, "display_name", "") or getattr(item_a, "roi_label", "") or "CenterA")
        name_b = str(getattr(item_b, "display_name", "") or getattr(item_b, "roi_label", "") or "CenterB")
        display_name = str(getattr(distance_item, "display_name", "") or getattr(distance_item, "item_id", "") or "Center Distance")
        detail = (
            f"center_distance={cls._format_measurement_value(reported_value, unit)}"
            f" mode={distance_mode}"
            f" centers={name_a}/{name_b}"
            f" raw={distance_px:.3f}px"
        )
        if compensation_enabled:
            detail += (
                f" compensation=k={compensation_slope:.6g},b={compensation_intercept:.6g}"
            )
        if lower is not None or upper is not None:
            detail += (
                f" spec={cls._format_measurement_limit(lower, unit)}"
                f"..{cls._format_measurement_limit(upper, unit)}"
            )

        pred = "OK" if ok else "NG"
        return {
            "file_path": image_path,
            "file_name": display_name,
            "gt": "",
            "pred": pred,
            "diff": 0.0,
            "sim_ok": None,
            "sim_ng": None,
            "value": reported_value,
            "unit": unit,
            "threshold": upper,
            "match_ms": max(
                (
                    float(row.get("match_ms") or 0.0)
                    for row in (row_a, row_b)
                    if row.get("match_ms") is not None
                ),
                default=0.0,
            ),
            "infer_ms": 0.0,
            "total_ms": 0.0,
            "json_name": str(row_a.get("json_name", row_b.get("json_name", "")) or ""),
            "detail": detail,
            "algorithm": CENTER_DISTANCE_ALGORITHM,
            "tool_name": display_name,
            "camera_id": camera_id,
            "roi_label": str(getattr(distance_item, "roi_label", "") or ""),
            "params": params,
            "measurement": {
                "type": CENTER_DISTANCE_ALGORITHM,
                "distance_px": float(distance_px),
                "distance": reported_value,
                "unit": unit,
                "pixel_size_mm": pixel_size,
                "compensation_enabled": compensation_enabled,
                "compensation_slope": compensation_slope,
                "compensation_intercept": compensation_intercept,
                "dimension_segment": [[float(x), float(y)] for x, y in dimension_segment],
                "line_segment": [[float(x), float(y)] for x, y in dimension_segment],
                "center_points": [
                    [float(center_a[0]), float(center_a[1])],
                    [float(center_b[0]), float(center_b[1])],
                ],
                "distance_mode": distance_mode,
                "label": cls._format_measurement_value(reported_value, unit),
                "pred": pred,
                "center_a_item_id": cls._center_item_ref(item_a),
                "center_b_item_id": cls._center_item_ref(item_b),
                "center_a": dict(row_a.get("measurement", {}) or {}),
                "center_b": dict(row_b.get("measurement", {}) or {}),
            },
        }

    @staticmethod
    def _build_detail(row: dict) -> str:
        detail_parts: List[str] = []
        explicit_detail = str(row.get("detail", "") or "").strip()
        if explicit_detail:
            detail_parts.append(explicit_detail)
        elif str(row.get("pred", "") or "").strip().upper() == "MEASURED" and row.get("value") is not None:
            detail_parts.append(f"value={float(row['value']):.3f}")
        if row.get("diff") is not None:
            detail_parts.append(f"diff={float(row['diff']):.4f}")
        if row.get("match_ms") is not None:
            detail_parts.append(f"match={float(row['match_ms']):.1f}ms")
        if row.get("infer_ms") is not None:
            detail_parts.append(f"infer={float(row['infer_ms']):.1f}ms")
        elif row.get("total_ms") is not None:
            detail_parts.append(f"total={float(row['total_ms']):.1f}ms")
        return " ".join(detail_parts)

    @staticmethod
    def _extract_timing_fields(row: dict) -> tuple[float, float]:
        match_ms = float(row.get("match_ms") or 0.0) if row.get("match_ms") is not None else 0.0
        infer_source = row.get("infer_ms")
        if infer_source is None:
            infer_source = row.get("total_ms")
        infer_ms = float(infer_source or 0.0) if infer_source is not None else 0.0
        return match_ms, infer_ms

    @staticmethod
    def _build_camera_detail(
        item_results: List[InspectionItemResult],
        *,
        match_ms: float,
        infer_ms: float,
    ) -> str:
        parts: List[str] = []
        ng_items = [item for item in item_results if item.result == "NG"]
        pass_items = [item for item in item_results if _normalized_result(item.result) == _PASS_RESULT]
        measured_items = [item for item in item_results if str(item.result or "").strip().upper() == "MEASURED"]
        if ng_items:
            parts.append(
                "NG: " + ", ".join(
                    item.display_name or item.item_id or item.roi_label or "item"
                    for item in ng_items
                )
            )
            if len(ng_items) == 1 and ng_items[0].detail:
                detail = InspectionExecutor._strip_timing_tokens(ng_items[0].detail)
                if detail:
                    parts.append(detail)
        elif pass_items:
            parts.append(
                "PASS: " + ", ".join(
                    item.display_name or item.item_id or item.roi_label or "item"
                    for item in pass_items
                )
            )
            detail = "; ".join(
                InspectionExecutor._strip_timing_tokens(item.detail)
                for item in pass_items
                if InspectionExecutor._strip_timing_tokens(item.detail)
            )
            if detail:
                parts.append(detail)
        elif measured_items and len(measured_items) == len(item_results):
            parts.append(
                "; ".join(
                    f"{item.display_name or item.item_id or item.roi_label or 'item'} {item.detail}".strip()
                    for item in measured_items
                )
            )
        elif len(item_results) == 1:
            if item_results[0].detail:
                detail = InspectionExecutor._strip_timing_tokens(item_results[0].detail)
                if detail:
                    parts.append(detail)
                else:
                    parts.append(f"{item_results[0].display_name or item_results[0].item_id or 'item'}=OK")
            else:
                parts.append(f"{item_results[0].display_name or item_results[0].item_id or 'item'}=OK")
        else:
            parts.append(f"{len(item_results)} items OK")

        if match_ms > 0:
            parts.append(f"match={match_ms:.1f}ms")
        if infer_ms > 0:
            parts.append(f"infer={infer_ms:.1f}ms")
        return " ".join(part for part in parts if part)

    @staticmethod
    def _strip_timing_tokens(detail: str) -> str:
        tokens = []
        for token in str(detail or "").split():
            if token.startswith(("match=", "infer=", "total=")):
                continue
            tokens.append(token)
        return " ".join(tokens)


__all__ = ["InspectionExecutionRequest", "InspectionExecutionResponse", "InspectionExecutor"]
