"""
inspection_models.py

运行检测结果的数据模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class InspectionItemResult:
    item_id: str
    display_name: str
    camera_id: str
    roi_label: str
    algorithm_code: str = ""
    enabled: bool = True
    params: Dict[str, object] = field(default_factory=dict)
    result: str = "PENDING"  # PENDING/RUNNING/OK/NG/INACTIVE/DISABLED
    detail: str = ""

    def to_runtime_row(self) -> dict:
        status_map = {
            "PENDING": ("pending", "未检测"),
            "RUNNING": ("running", "检测中"),
            "OK": ("ok", "OK"),
            "NG": ("ng", "NG"),
            "INACTIVE": ("inactive", "相机未接入"),
            "DISABLED": ("disabled", "已禁用"),
        }
        kind, text = status_map.get(self.result, ("pending", self.result or "未检测"))
        if self.detail:
            text = f"{text} ({self.detail})"
        return {
            "item_id": self.item_id,
            "display_name": self.display_name,
            "camera_id": self.camera_id,
            "roi_label": self.roi_label,
            "algorithm_code": self.algorithm_code,
            "enabled": self.enabled,
            "status_kind": kind,
            "status_text": text,
        }


@dataclass
class CameraRuntimeResult:
    camera_id: str
    result: str = ""  # OK/NG/PRECHECK_FAILED/...
    detail: str = ""
    image_path: str = ""
    capture_ms: float = 0.0
    match_ms: float = 0.0
    infer_ms: float = 0.0
    item_results: List[InspectionItemResult] = field(default_factory=list)


@dataclass
class RuntimeInspectionResult:
    task_id: str
    product_name: str
    recipe_name: str = ""
    final_result: str = ""
    duration_ms: int = 0
    capture_ms: float = 0.0
    match_ms: float = 0.0
    infer_ms: float = 0.0
    error_message: str = ""
    is_system_error: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    camera_results: Dict[str, CameraRuntimeResult] = field(default_factory=dict)
    item_results: List[InspectionItemResult] = field(default_factory=list)

    def item_rows(self) -> List[dict]:
        return [item.to_runtime_row() for item in self.item_results]

    def camera_result_map(self) -> Dict[str, str]:
        return {
            camera_id: (camera_result.result or "-")
            for camera_id, camera_result in self.camera_results.items()
        }

    def camera_detail_map(self) -> Dict[str, str]:
        return {
            camera_id: (camera_result.detail or "")
            for camera_id, camera_result in self.camera_results.items()
        }

    def timing_breakdown(self) -> Dict[str, float]:
        return {
            "capture_ms": float(self.capture_ms or 0.0),
            "match_ms": float(self.match_ms or 0.0),
            "infer_ms": float(self.infer_ms or 0.0),
            "duration_ms": float(self.duration_ms or 0.0),
        }

    def summary_text(self) -> str:
        parts: List[str] = []
        for camera_id in sorted(self.camera_results.keys()):
            result = self.camera_results[camera_id].result or "-"
            parts.append(f"{camera_id}={result}")
        if self.error_message:
            parts.append(self.error_message)
        if self.capture_ms:
            parts.append(f"capture {self.capture_ms:.1f} ms")
        if self.match_ms:
            parts.append(f"match {self.match_ms:.1f} ms")
        if self.infer_ms:
            parts.append(f"infer {self.infer_ms:.1f} ms")
        if self.duration_ms:
            parts.append(f"耗时 {self.duration_ms} ms")
        return "；".join(parts) if parts else "本次没有检测结果"

    def to_record_extra_fields(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "camera_count": len(self.camera_results),
            "item_count": len(self.item_results),
            "capture_ms": self.capture_ms,
            "match_ms": self.match_ms,
            "infer_ms": self.infer_ms,
            "camera_results_summary": "; ".join(
                f"{camera_id}:{camera_result.result or '-'}"
                for camera_id, camera_result in sorted(self.camera_results.items())
            ),
            "item_results_summary": "; ".join(
                f"{item.display_name or item.item_id}:{item.result}"
                for item in self.item_results
            ),
            "item_results_json": json.dumps(
                [
                    {
                        "item_id": item.item_id,
                        "display_name": item.display_name,
                        "camera_id": item.camera_id,
                        "roi_label": item.roi_label,
                        "algorithm_code": item.algorithm_code,
                        "enabled": item.enabled,
                        "params": dict(item.params or {}),
                        "result": item.result,
                        "detail": item.detail,
                    }
                    for item in self.item_results
                ],
                ensure_ascii=False,
            ),
        }
        for camera_id, camera_result in sorted(self.camera_results.items()):
            row[f"{camera_id}_result"] = camera_result.result or ""
            row[f"{camera_id}_detail"] = camera_result.detail or ""
            row[f"{camera_id}_image_path"] = camera_result.image_path or ""
            row[f"{camera_id}_capture_ms"] = camera_result.capture_ms
            row[f"{camera_id}_match_ms"] = camera_result.match_ms
            row[f"{camera_id}_infer_ms"] = camera_result.infer_ms
        for index, item in enumerate(self.item_results, start=1):
            prefix = f"item_{index:02d}"
            row[f"{prefix}_id"] = item.item_id
            row[f"{prefix}_name"] = item.display_name
            row[f"{prefix}_camera"] = item.camera_id
            row[f"{prefix}_roi_label"] = item.roi_label
            row[f"{prefix}_algorithm_code"] = item.algorithm_code
            row[f"{prefix}_enabled"] = item.enabled
            row[f"{prefix}_params_json"] = json.dumps(dict(item.params or {}), ensure_ascii=False)
            row[f"{prefix}_result"] = item.result
            row[f"{prefix}_detail"] = item.detail
        return row


def build_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


__all__ = [
    "InspectionItemResult",
    "CameraRuntimeResult",
    "RuntimeInspectionResult",
    "build_task_id",
]
