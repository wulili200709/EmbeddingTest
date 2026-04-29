"""
inspection_models.py

运行检测结果的数据模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


_TIMING_PREFIXES = ("capture", "match", "infer", "total", "耗时")


def _clean_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def _display_name_with_ng(name: object) -> str:
    text = _clean_text(name)
    if not text:
        return ""
    return text if text.upper().endswith("NG") else f"{text}NG"


def _localize_failure_text(text: object) -> str:
    raw = _clean_text(text)
    if not raw:
        return ""
    lower = raw.lower()
    if (
        "match failure" in lower
        or "did not find any match" in lower
        or "no match" in lower
        or "not find any match" in lower
    ):
        return "模板匹配失败"
    if "missing line2dup model" in lower or "line2dup template" in lower:
        return "模板未配置"
    if "missing ncc model" in lower or "ncc model" in lower:
        return "NCC模板未配置"
    return raw


def _strip_timing_parts(text: object) -> str:
    parts: List[str] = []
    for part in _clean_text(text).replace("；", " ").split():
        if part.startswith(_TIMING_PREFIXES):
            continue
        if "=" in part and part.split("=", 1)[0] in {"capture", "match", "infer", "total"}:
            continue
        parts.append(part)
    return " ".join(parts)


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
        failure_text = self.failure_summary_text()
        if failure_text:
            parts.append(failure_text)
        for camera_id in sorted(self.camera_results.keys()):
            result = self.camera_results[camera_id].result or "-"
            parts.append(f"{camera_id}={result}")
        if self.error_message and self.error_message not in parts:
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

    def failure_summary_text(self, *, max_items: int = 5) -> str:
        error_text = _localize_failure_text(self.error_message)
        if error_text:
            return error_text

        ng_items = [
            item
            for item in self.item_results
            if str(item.result or "").strip().upper() == "NG"
        ]
        names: List[str] = []
        seen: set[str] = set()
        for item in ng_items:
            name = (
                _clean_text(item.display_name)
                or _clean_text(item.item_id)
                or _clean_text(item.roi_label)
            )
            display = _display_name_with_ng(name)
            if display and display not in seen:
                names.append(display)
                seen.add(display)
        if names:
            shown = names[: max(1, int(max_items))]
            suffix = f" 等{len(names)}项NG" if len(names) > len(shown) else ""
            return "、".join(shown) + suffix

        for camera_id in sorted(self.camera_results.keys()):
            camera = self.camera_results[camera_id]
            if str(camera.result or "").strip().upper() != "NG":
                continue
            detail = _strip_timing_parts(camera.detail)
            localized = _localize_failure_text(detail)
            if localized:
                if localized.startswith("NG:"):
                    localized = localized[3:].strip()
                if localized:
                    return localized
        return ""

    def to_record_extra_fields(self) -> Dict[str, object]:
        row: Dict[str, object] = {}
        for index, item in enumerate(self.item_results, start=1):
            prefix = f"item_{index:02d}"
            row[f"{prefix}_enabled"] = item.enabled
            row[f"{prefix}_name"] = item.display_name
            row[f"{prefix}_result"] = item.result
            row[f"{prefix}_roi_label"] = item.roi_label
        return row


def build_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


__all__ = [
    "InspectionItemResult",
    "CameraRuntimeResult",
    "RuntimeInspectionResult",
    "build_task_id",
]
