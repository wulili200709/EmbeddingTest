from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


_RECORD_OK_RESULTS = {"OK", "PASS"}


def _binary_record_result(value: object) -> str:
    result = str(value or "").strip().upper()
    if result in _RECORD_OK_RESULTS:
        return "OK"
    if result == "NG":
        return "NG"
    return ""


@dataclass
class InspectionItemResult:
    item_id: str
    display_name: str
    camera_id: str
    roi_label: str
    algorithm_code: str = ""
    enabled: bool = True
    params: Dict[str, object] = field(default_factory=dict)
    result: str = "PENDING"  # PENDING/RUNNING/OK/NG/PASS/MEASURED/INACTIVE/DISABLED
    detail: str = ""
    value: Optional[float] = None
    unit: str = ""

    def to_runtime_row(self) -> dict:
        status_map = {
            "PENDING": ("pending", "未检测"),
            "RUNNING": ("running", "检测中"),
            "OK": ("ok", "OK"),
            "NG": ("ng", "NG"),
            "PASS": ("pass", "PASS"),
            "MEASURED": ("measured", self.detail or "MEASURED"),
            "INACTIVE": ("inactive", "相机未接入"),
            "DISABLED": ("disabled", "已禁用"),
        }
        kind, text = status_map.get(self.result, ("pending", self.result or "未检测"))
        if self.detail and self.result != "MEASURED":
            text = f"{text} ({self.detail})"
        return {
            "item_id": self.item_id,
            "display_name": self.display_name,
            "camera_id": self.camera_id,
            "roi_label": self.roi_label,
            "algorithm_code": self.algorithm_code,
            "enabled": self.enabled,
            "params": dict(self.params or {}),
            "value": self.value,
            "unit": self.unit,
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
        payload = {
            "capture_ms": float(self.capture_ms or 0.0),
            "match_ms": float(self.match_ms or 0.0),
            "infer_ms": float(self.infer_ms or 0.0),
            "duration_ms": float(self.duration_ms or 0.0),
        }
        for camera_id, camera_result in sorted(self.camera_results.items()):
            camera_key = str(camera_id or "").strip()
            if not camera_key:
                continue
            capture_ms = float(camera_result.capture_ms or 0.0)
            match_ms = float(camera_result.match_ms or 0.0)
            infer_ms = float(camera_result.infer_ms or 0.0)
            payload[f"{camera_key}_capture_ms"] = capture_ms
            payload[f"{camera_key}_match_ms"] = match_ms
            payload[f"{camera_key}_infer_ms"] = infer_ms
            payload[f"{camera_key}_total_ms"] = capture_ms + match_ms + infer_ms
        return payload

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
        row: Dict[str, object] = {}
        indexed_items = list(enumerate(self.item_results))
        indexed_items.sort(
            key=lambda pair: (str(pair[1].camera_id or "").strip(), pair[0])
        )
        record_items: list[tuple[InspectionItemResult, str, str]] = []
        identifiers_by_base_column: dict[str, set[str]] = {}
        for _index, item in indexed_items:
            camera_id = str(item.camera_id or "").strip()
            display_name = str(item.display_name or "").strip()
            roi_label = str(item.roi_label or "").strip()
            item_id = str(item.item_id or "").strip()
            roi_name = (
                display_name
                or roi_label
                or item_id
            )
            if not camera_id or not roi_name:
                continue

            identifier = roi_label or item_id or roi_name
            base_column = f"{camera_id}.{roi_name}"
            identifiers_by_base_column.setdefault(base_column, set()).add(identifier)
            record_items.append((item, roi_name, identifier))

        for item, roi_name, identifier in record_items:
            camera_id = str(item.camera_id or "").strip()
            column_name = f"{camera_id}.{roi_name}"
            if len(identifiers_by_base_column.get(column_name, set())) > 1:
                column_name = f"{column_name} [{identifier}]"
            result = _binary_record_result(item.result)
            previous = str(row.get(column_name, "") or "").strip().upper()
            if previous == "NG" or result == "NG":
                row[column_name] = "NG"
            elif previous == "OK" or result == "OK":
                row[column_name] = "OK"
            else:
                row.setdefault(column_name, "")
        return row


def build_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


__all__ = [
    "InspectionItemResult",
    "CameraRuntimeResult",
    "RuntimeInspectionResult",
    "build_task_id",
]
