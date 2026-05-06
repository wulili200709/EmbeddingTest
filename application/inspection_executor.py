"""
inspection_executor.py

运行检测执行器。

当前阶段：
  - 仍复用 `ToolPage.predict_image()` 作为底层推理入口
  - 但已把“相机检测执行”从 RuntimeController 中下沉到本模块
  - item 级结果暂按“所属相机结果继承”生成
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Protocol

from domain import InspectionItem, InspectionItemResult


def _is_line_distance_item(item: InspectionItem) -> bool:
    return str(getattr(item, "algorithm_code", "") or "").strip() == "line_distance"


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

    def __init__(self, predictor: PredictorProtocol) -> None:
        self._predictor = predictor

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
        predicted_enabled_items = [item for item in enabled_items if not _is_line_distance_item(item)]
        distance_items = [item for item in enabled_items if _is_line_distance_item(item)]
        batch_rows: List[dict] | None = None
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
        elif callable(batch_predict) and predicted_enabled_items:
            batch_rows = [dict(row) for row in batch_predict(request.image_path, items=predicted_enabled_items)]
        predicted_index = 0
        line_rows_by_item_id: dict[str, dict] = {}

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
            if _is_line_distance_item(item):
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
                line_rows_by_item_id[item_key] = dict(row)
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
            )
            item_results.append(item_result)
            enabled_item_results.append(item_result)

        line_distance_results: List[InspectionItemResult] = []
        for distance_item in distance_items:
            line_distance_row = self._build_find_line_distance_row(
                distance_item=distance_item,
                line_items=predicted_enabled_items,
                line_rows_by_item_id=line_rows_by_item_id,
                camera_id=request.camera_id,
                image_path=request.image_path,
            )
            if line_distance_row is None:
                line_distance_row = {
                    "file_path": request.image_path,
                    "file_name": str(distance_item.display_name or distance_item.item_id or "Line Distance"),
                    "pred": "NG",
                    "detail": "line distance pair missing",
                    "algorithm": "line_distance",
                    "tool_name": str(distance_item.display_name or distance_item.item_id or "Line Distance"),
                    "camera_id": request.camera_id,
                    "roi_label": str(distance_item.roi_label or ""),
                    "params": dict(distance_item.params or {}),
                }
            item_rows.append(dict(line_distance_row))
            line_distance_result = InspectionItemResult(
                item_id=distance_item.item_id,
                display_name=distance_item.display_name,
                camera_id=distance_item.camera_id,
                roi_label=distance_item.roi_label,
                algorithm_code=distance_item.algorithm_code,
                enabled=True,
                params=dict(distance_item.params or {}),
                result=str(line_distance_row.get("pred", "NG") or "NG"),
                detail=self._build_detail(line_distance_row),
            )
            item_results.append(line_distance_result)
            enabled_item_results.append(line_distance_result)
            line_distance_results.append(line_distance_result)

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

        if line_distance_results:
            decision_item_results = line_distance_results
        else:
            decision_item_results = [
                item
                for item in enabled_item_results
                if str(item.result or "").strip().upper() in {"OK", "NG"}
            ]
        final_result = (
            "OK"
            if not decision_item_results or all(item.result == "OK" for item in decision_item_results)
            else "NG"
        )
        match_ms = max(
            (self._extract_timing_fields(row)[0] for row in item_rows),
            default=0.0,
        )
        infer_ms = sum(self._extract_timing_fields(row)[1] for row in item_rows)
        total_ms = match_ms + infer_ms if (match_ms > 0.0 or infer_ms > 0.0) else 0.0
        if line_distance_results:
            distance_detail = "; ".join(
                self._strip_timing_tokens(item.detail)
                for item in line_distance_results
                if self._strip_timing_tokens(item.detail)
            )
            camera_detail_parts = [distance_detail] if distance_detail else ["Line Distance"]
            if match_ms > 0:
                camera_detail_parts.append(f"match={match_ms:.1f}ms")
            if infer_ms > 0:
                camera_detail_parts.append(f"infer={infer_ms:.1f}ms")
            camera_detail = " ".join(camera_detail_parts)
        else:
            camera_detail = self._build_camera_detail(
                enabled_item_results,
                match_ms=match_ms,
                infer_ms=infer_ms,
            )
        raw_measurements = tuple(
            dict(row.get("measurement", {}) or {})
            for row in item_rows
            if isinstance(row.get("measurement"), dict)
        )
        line_distance_measurements = tuple(
            measurement
            for measurement in raw_measurements
            if str(measurement.get("type", "") or "").strip() == "line_distance"
        )
        measurements = line_distance_measurements or raw_measurements

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

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)

    @classmethod
    def _item_param_float(cls, item: InspectionItem, key: str) -> float | None:
        params = dict(getattr(item, "params", {}) or {})
        return cls._optional_float(params.get(key))

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
        distance_px, angle_delta = cls._distance_between_segments(segment_a, segment_b)
        dimension_segment = cls._dimension_segment(segment_a, segment_b, distance_px)

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
        lower = cls._optional_float(params.get("lower_limit", params.get(f"lower_limit_{unit}")))
        upper = cls._optional_float(params.get("upper_limit", params.get(f"upper_limit_{unit}")))
        ok = True
        if lower is not None and value < lower:
            ok = False
        if upper is not None and value > upper:
            ok = False

        name_a = str(getattr(item_a, "display_name", "") or getattr(item_a, "roi_label", "") or "LineA")
        name_b = str(getattr(item_b, "display_name", "") or getattr(item_b, "roi_label", "") or "LineB")
        display_name = str(getattr(distance_item, "display_name", "") or getattr(distance_item, "item_id", "") or "Line Distance")
        detail = (
            f"distance={value:.3f}{unit}"
            f" raw={distance_px:.3f}px"
            f" angle_delta={angle_delta:.3f}deg"
            f" lines={name_a}/{name_b}"
        )
        if lower is not None or upper is not None:
            detail += f" spec={lower if lower is not None else '-'}..{upper if upper is not None else '-'}{unit}"

        return {
            "file_path": image_path,
            "file_name": display_name,
            "gt": "",
            "pred": "OK" if ok else "NG",
            "diff": angle_delta,
            "sim_ok": None,
            "sim_ng": None,
            "value": value,
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
            "algorithm": "line_distance",
            "tool_name": display_name,
            "camera_id": camera_id,
            "roi_label": str(getattr(distance_item, "roi_label", "") or ""),
            "params": params,
            "measurement": {
                "type": "line_distance",
                "distance_px": distance_px,
                "distance": value,
                "unit": unit,
                "pixel_size_mm": pixel_size,
                "angle_delta_deg": angle_delta,
                "dimension_segment": [[float(x), float(y)] for x, y in dimension_segment],
                "label": f"{value:.3f}{unit}",
                "pred": "OK" if ok else "NG",
                "line_a_item_id": cls._line_item_ref(item_a),
                "line_b_item_id": cls._line_item_ref(item_b),
                "line_a": dict(row_a.get("measurement", {}) or {}),
                "line_b": dict(row_b.get("measurement", {}) or {}),
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
