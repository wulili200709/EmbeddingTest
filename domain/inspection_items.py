
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List, Mapping

from algorithms.registry import (
    SHARED_BACKBONE_ALGORITHM_CODE,
    normalize_tool_algorithm_code,
)
from common.safe_io import atomic_write_json, load_json_with_backup


SUPPORTED_CAMERA_IDS = ("cam1", "cam2")
LINE_DISTANCE_ALGORITHMS = {"line_distance", "line_distance_ref_normal"}


def _slug_token(value: object, fallback: str = "tool") -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip()).strip("._-")
    return normalized or fallback


@dataclass(init=False)
class InspectionItem:
    item_id: str
    display_name: str
    camera_id: str
    roi_label: str
    algorithm_code: str = SHARED_BACKBONE_ALGORITHM_CODE
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        item_id: str,
        display_name: str,
        camera_id: str,
        roi_label: str,
        algorithm_code: str = SHARED_BACKBONE_ALGORITHM_CODE,
        enabled: bool = True,
        params: Mapping[str, Any] | None = None,
        *,
        algorithm_type: str | None = None,
    ) -> None:
        normalized_camera_id = str(camera_id or "cam1").strip() or "cam1"
        if normalized_camera_id not in SUPPORTED_CAMERA_IDS:
            normalized_camera_id = "cam1"
        resolved_algorithm = normalize_tool_algorithm_code(
            algorithm_code if algorithm_type is None else algorithm_type
        )
        self.item_id = str(item_id or "").strip()
        self.display_name = str(display_name or "").strip()
        self.camera_id = normalized_camera_id
        self.roi_label = str(roi_label or "").strip()
        self.algorithm_code = resolved_algorithm
        self.enabled = bool(enabled)
        self.params = dict(params or {})

    @property
    def algorithm_type(self) -> str:
        return self.algorithm_code

    @algorithm_type.setter
    def algorithm_type(self, value: str) -> None:
        self.algorithm_code = normalize_tool_algorithm_code(value)

    @property
    def model_key(self) -> str:
        camera = _slug_token(self.camera_id, "cam1")
        item = _slug_token(self.item_id or self.roi_label or self.display_name, "roi")
        return f"{camera}__{item}"

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionItem":
        roi_label = str(data.get("roi_label", "")).strip()
        camera_id = str(data.get("camera_id", "cam1")).strip() or "cam1"
        display_name = str(data.get("display_name", "")).strip() or roi_label or "roi"
        item_id = str(data.get("item_id", "")).strip() or roi_label or display_name
        algorithm_code = str(data.get("algorithm_code", "")).strip()
        algorithm_type = str(data.get("algorithm_type", "")).strip()
        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}
        return cls(
            item_id=item_id,
            display_name=display_name,
            camera_id=camera_id,
            roi_label=roi_label,
            algorithm_code=algorithm_code or algorithm_type or SHARED_BACKBONE_ALGORITHM_CODE,
            enabled=bool(data.get("enabled", True)),
            params=params,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["algorithm_type"] = self.algorithm_code
        return payload


def load_inspection_items(path: str) -> List[InspectionItem]:
    if not path:
        return []
    raw = load_json_with_backup(path, default=[])
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    items: List[InspectionItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = InspectionItem.from_dict(entry)
        if item.roi_label or item.algorithm_code in LINE_DISTANCE_ALGORITHMS:
            items.append(item)
    return items


def save_inspection_items(items: Iterable[InspectionItem], path: str) -> None:
    payload = [item.to_dict() for item in items]
    atomic_write_json(path, payload, ensure_ascii=False, indent=2)


def build_default_item(
    roi_label: str,
    *,
    camera_id: str = "cam1",
    display_name: str = "",
    algorithm_code: str = SHARED_BACKBONE_ALGORITHM_CODE,
) -> InspectionItem:
    roi_label = str(roi_label).strip()
    display_name = str(display_name).strip() or roi_label or "roi"
    return InspectionItem(
        item_id=roi_label or "roi",
        display_name=display_name,
        camera_id=camera_id,
        roi_label=roi_label or "roi",
        algorithm_code=algorithm_code,
    )


def sync_items_with_labels(
    existing_items: Iterable[InspectionItem],
    labels: Iterable[str],
    *,
    default_camera_id: str = "cam1",
    display_names_by_label: Mapping[str, str] | None = None,
) -> List[InspectionItem]:
    """
    以 labels 的顺序为准同步检测项。

    规则：
      - 已存在且 roi_label 相同的项：保留 display_name / camera_id / enabled / algorithm_code / params
      - 新标签：生成默认项
      - 不再存在的标签：移除
    """
    normalized_labels = [str(label).strip() for label in labels if str(label).strip()]
    display_names_by_label = {
        str(label).strip(): str(name).strip()
        for label, name in dict(display_names_by_label or {}).items()
        if str(label).strip()
    }
    existing_by_label = {
        item.roi_label: item
        for item in existing_items
        if isinstance(item, InspectionItem) and item.roi_label
    }
    synced: List[InspectionItem] = []
    for label in normalized_labels:
        existing = existing_by_label.get(label)
        display_name = display_names_by_label.get(label, "")
        if existing is not None:
            synced.append(
                InspectionItem(
                    item_id=existing.item_id or label,
                    display_name=display_name or existing.display_name or label,
                    camera_id=existing.camera_id if existing.camera_id in SUPPORTED_CAMERA_IDS else default_camera_id,
                    roi_label=label,
                    algorithm_code=existing.algorithm_code or SHARED_BACKBONE_ALGORITHM_CODE,
                    enabled=bool(existing.enabled),
                    params=dict(existing.params or {}),
                )
            )
        else:
            synced.append(
                build_default_item(
                    label,
                    camera_id=default_camera_id,
                    display_name=display_name or label,
                )
            )
    existing_distance_items = [
        item
        for item in existing_items
        if isinstance(item, InspectionItem)
        and item.algorithm_code in LINE_DISTANCE_ALGORITHMS
        and item.item_id not in {synced_item.item_id for synced_item in synced}
    ]
    for existing in existing_distance_items:
        synced.append(
            InspectionItem(
                item_id=existing.item_id or existing.display_name or "line_distance",
                display_name=existing.display_name or existing.item_id or "Line Distance",
                camera_id=existing.camera_id if existing.camera_id in SUPPORTED_CAMERA_IDS else default_camera_id,
                roi_label=existing.roi_label,
                algorithm_code=existing.algorithm_code,
                enabled=bool(existing.enabled),
                params=dict(existing.params or {}),
            )
        )
    return synced


__all__ = [
    "InspectionItem",
    "SUPPORTED_CAMERA_IDS",
    "build_default_item",
    "load_inspection_items",
    "save_inspection_items",
    "sync_items_with_labels",
]
