"""
inspection_items.py

检测项配置层：
  - 每个 ROI 对应一个 InspectionItem
  - 负责 inspection_items.json 的读写
  - 负责按当前 ROI 标签集合生成/同步默认检测项
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping


SUPPORTED_CAMERA_IDS = ("cam1", "cam2")


@dataclass
class InspectionItem:
    item_id: str
    display_name: str
    camera_id: str
    roi_label: str
    algorithm_type: str = "inherit_product"
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionItem":
        roi_label = str(data.get("roi_label", "")).strip()
        camera_id = str(data.get("camera_id", "cam1")).strip() or "cam1"
        if camera_id not in SUPPORTED_CAMERA_IDS:
            camera_id = "cam1"
        display_name = str(data.get("display_name", "")).strip() or roi_label or "roi"
        item_id = str(data.get("item_id", "")).strip() or roi_label or display_name
        algorithm_type = str(data.get("algorithm_type", "inherit_product")).strip() or "inherit_product"
        return cls(
            item_id=item_id,
            display_name=display_name,
            camera_id=camera_id,
            roi_label=roi_label,
            algorithm_type=algorithm_type,
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_inspection_items(path: str) -> List[InspectionItem]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    items: List[InspectionItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = InspectionItem.from_dict(entry)
        if item.roi_label:
            items.append(item)
    return items


def save_inspection_items(items: Iterable[InspectionItem], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = [item.to_dict() for item in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_default_item(
    roi_label: str,
    *,
    camera_id: str = "cam1",
    display_name: str = "",
) -> InspectionItem:
    roi_label = str(roi_label).strip()
    if camera_id not in SUPPORTED_CAMERA_IDS:
        camera_id = "cam1"
    display_name = str(display_name).strip() or roi_label or "roi"
    return InspectionItem(
        item_id=roi_label or "roi",
        display_name=display_name,
        camera_id=camera_id,
        roi_label=roi_label or "roi",
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
      - 已存在且 roi_label 相同的项：保留 display_name / camera_id / enabled / algorithm_type
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
                    algorithm_type=existing.algorithm_type or "inherit_product",
                    enabled=bool(existing.enabled),
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
    return synced


__all__ = [
    "InspectionItem",
    "SUPPORTED_CAMERA_IDS",
    "build_default_item",
    "load_inspection_items",
    "save_inspection_items",
    "sync_items_with_labels",
]
