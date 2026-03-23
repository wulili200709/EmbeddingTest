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
    ) -> dict: ...


@dataclass
class InspectionExecutionRequest:
    camera_id: str
    image_path: str
    items: List[InspectionItem] = field(default_factory=list)


@dataclass
class InspectionExecutionResponse:
    camera_id: str
    result: str
    detail: str = ""
    raw_row: dict | None = None
    match_ms: float = 0.0
    infer_ms: float = 0.0
    item_results: List[InspectionItemResult] = field(default_factory=list)


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
                item_results=[],
            )

        item_results: List[InspectionItemResult] = []
        enabled_items = [item for item in request.items if item.enabled]

        if not enabled_items:
            for item in request.items:
                item_results.append(
                    InspectionItemResult(
                        item_id=item.item_id,
                        display_name=item.display_name,
                        camera_id=item.camera_id,
                        roi_label=item.roi_label,
                        enabled=False,
                        result="DISABLED",
                    )
                )
            return InspectionExecutionResponse(
                camera_id=request.camera_id,
                result="OK",
                detail="",
                raw_row=None,
                item_results=item_results,
            )

        labels_override: List[str] = []
        for item in enabled_items:
            roi_label = str(item.roi_label or "").strip()
            if roi_label and roi_label not in labels_override:
                labels_override.append(roi_label)

        # Keep runtime semantics aligned with debug TEST:
        # one line2dup localization + one multi-ROI prediction per camera image.
        row = self._predictor.predict_image(
            request.image_path,
            labels_override=labels_override or None,
        )
        final_result = str(row.get("pred", "NG") or "NG")
        camera_detail = self._build_detail(row)
        match_ms, infer_ms = self._extract_timing_fields(row)

        for item in request.items:
            if not item.enabled:
                item_results.append(
                    InspectionItemResult(
                        item_id=item.item_id,
                        display_name=item.display_name,
                        camera_id=item.camera_id,
                        roi_label=item.roi_label,
                        enabled=False,
                        result="DISABLED",
                    )
                )
                continue

            item_results.append(
                InspectionItemResult(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    camera_id=item.camera_id,
                    roi_label=item.roi_label,
                    enabled=True,
                    result=final_result,
                    detail=camera_detail,
                )
            )

        return InspectionExecutionResponse(
            camera_id=request.camera_id,
            result=final_result,
            detail=camera_detail,
            raw_row=row,
            match_ms=match_ms,
            infer_ms=infer_ms,
            item_results=item_results,
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


__all__ = ["InspectionExecutionRequest", "InspectionExecutionResponse", "InspectionExecutor"]
