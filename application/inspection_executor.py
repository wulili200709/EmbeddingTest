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
from typing import List, Protocol

from domain import InspectionItem, InspectionItemResult


class PredictorProtocol(Protocol):
    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: List[str] | None = None,
        algorithm_override: str | None = None,
        model_key_override: str | None = None,
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
        batch_rows: List[dict] | None = None
        roi_shapes: tuple[object, ...] = ()
        batch_predict_from_frame = getattr(self._predictor, "predict_items_batch_from_frame", None)
        batch_predict = getattr(self._predictor, "predict_items_batch", None)
        if request.image_bgr is not None and callable(batch_predict_from_frame) and enabled_items:
            batch_prediction = batch_predict_from_frame(
                request.image_bgr,
                camera_role=request.camera_id,
                items=enabled_items,
            )
            if batch_prediction is not None:
                batch_rows = [dict(row) for row in list(getattr(batch_prediction, "rows", []) or [])]
                roi_shapes = tuple(getattr(batch_prediction, "roi_shapes", ()) or ())
        elif callable(batch_predict) and enabled_items:
            batch_rows = [dict(row) for row in batch_predict(request.image_path, items=enabled_items)]
        enabled_index = 0

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

            if batch_rows is not None:
                row = dict(batch_rows[enabled_index]) if enabled_index < len(batch_rows) else {}
            else:
                roi_label = str(item.roi_label or "").strip()
                labels_override = [roi_label] if roi_label else None
                row = self._predictor.predict_image(
                    request.image_path,
                    labels_override=labels_override,
                    algorithm_override=item.algorithm_code,
                    model_key_override=item.model_key,
                )
            enabled_index += 1
            item_rows.append(dict(row))
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

        final_result = "OK" if all(item.result == "OK" for item in enabled_item_results) else "NG"
        match_ms = max(
            (self._extract_timing_fields(row)[0] for row in item_rows),
            default=0.0,
        )
        infer_ms = sum(self._extract_timing_fields(row)[1] for row in item_rows)
        total_ms = match_ms + infer_ms if (match_ms > 0.0 or infer_ms > 0.0) else 0.0
        camera_detail = self._build_camera_detail(
            enabled_item_results,
            match_ms=match_ms,
            infer_ms=infer_ms,
        )

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
        )

    @staticmethod
    def _build_detail(row: dict) -> str:
        detail_parts: List[str] = []
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
