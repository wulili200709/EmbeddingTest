
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from common.algorithm_codes import (
    SHARED_BACKBONE_ALGORITHM_CODE,
    normalize_tool_algorithm_code,
)
from common.camera_roles import CAMERA_ROLES, DEFAULT_CAMERA_ROLE, normalize_camera_role
from common.safe_io import atomic_write_json, load_json_with_backup
from common.model_library import make_library_model_key
from common.model_groups import OrderModelGroups, make_group_model_key, legacy_group_model_key


SUPPORTED_CAMERA_IDS = CAMERA_ROLES
POST_DISTANCE_ALGORITHMS = {
    "line_distance",
    "line_distance_ref_normal",
    "center_distance",
    "point_line_distance",
}


def _slug_token(value: object, fallback: str = "tool") -> str:
    normalized = re.sub(
        r"[^\w._-]+",
        "_",
        str(value or "").strip(),
        flags=re.UNICODE,
    ).strip("._-")
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
    task_group: str = ""
    model_source: str = "independent"
    legacy_model_key: str = ""
    library_model_id: str = ""
    library_version: str = ""

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
        task_group: str = "",
        algorithm_type: str | None = None,
        model_source: str | None = None,
        legacy_model_key: str = "",
        library_model_id: str = "",
        library_version: str = "",
    ) -> None:
        normalized_camera_id = normalize_camera_role(camera_id, default=DEFAULT_CAMERA_ROLE)
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
        self.task_group = str(task_group or "").strip()
        self._model_source_explicit = model_source is not None
        self.model_source = str(model_source or ("group" if self.task_group else "independent")).strip()
        self.legacy_model_key = str(legacy_model_key or "").strip()
        if model_source is None and self.task_group and not self.legacy_model_key:
            self.legacy_model_key = legacy_group_model_key(self.camera_id, self.task_group)
        self.library_model_id = str(library_model_id or "").strip()
        self.library_version = str(library_version or "").strip()
        self.validate_model_binding()

    def validate_model_binding(self) -> None:
        if self.model_source not in {"independent", "group", "library"}:
            raise ValueError(f"Unsupported model source: {self.model_source}")
        if self.model_source == "group":
            if not self.task_group:
                raise ValueError("A shared model group is required")
            if self.legacy_model_key:
                if self.legacy_model_key != legacy_group_model_key(self.camera_id, self.task_group):
                    raise ValueError("Legacy group model key does not match its camera and group")
            else:
                make_group_model_key(self.camera_id, self.task_group)
        if self.model_source == "library":
            make_library_model_key(self.camera_id, self.library_model_id, self.library_version)

    @property
    def algorithm_type(self) -> str:
        return self.algorithm_code

    @algorithm_type.setter
    def algorithm_type(self, value: str) -> None:
        self.algorithm_code = normalize_tool_algorithm_code(value)

    @property
    def model_key(self) -> str:
        camera = _slug_token(self.camera_id, DEFAULT_CAMERA_ROLE)
        item = _slug_token(self.item_id or self.roi_label or self.display_name, "roi")
        return f"{camera}__{item}"

    @property
    def effective_task_group(self) -> str:
        return str(
            self.task_group
            or self.item_id
            or self.roi_label
            or self.display_name
            or "roi"
        ).strip() or "roi"

    @property
    def task_model_key(self) -> str:
        return self.legacy_model_key or make_group_model_key(self.camera_id, self.effective_task_group)

    @property
    def effective_model_key(self) -> str:
        self.validate_model_binding()
        if self.model_source == "group":
            return self.task_model_key
        if self.model_source == "library":
            return make_library_model_key(self.camera_id, self.library_model_id, self.library_version)
        return self.model_key

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionItem":
        roi_label = str(data.get("roi_label", "")).strip()
        camera_id = normalize_camera_role(data.get("camera_id"), default=DEFAULT_CAMERA_ROLE)
        display_name = str(data.get("display_name", "")).strip() or roi_label or "roi"
        item_id = str(data.get("item_id", "")).strip() or roi_label or display_name
        task_group = str(
            data.get("task_group", data.get("group_key", data.get("shared_group", "")))
            or ""
        ).strip()
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
            task_group=task_group,
            model_source=data.get("model_source"),
            legacy_model_key=data.get("legacy_model_key", ""),
            library_model_id=data.get("library_model_id", ""),
            library_version=data.get("library_version", ""),
        )

    def to_dict(self) -> dict:
        self.validate_model_binding()
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
        if item.roi_label or item.algorithm_code in POST_DISTANCE_ALGORITHMS:
            items.append(item)
    return items


def save_inspection_items(items: Iterable[InspectionItem], path: str, *, traditional_models: dict | None = None) -> None:
    items = list(items)
    payload = [item.to_dict() for item in items]
    previous = load_json_with_backup(path, default=[])
    catalog = OrderModelGroups(Path(path).parent)
    catalog.remember_items(previous if isinstance(previous, list) else [], payload)
    from common.model_group_migration import separate_legacy_group_assets
    changed = separate_legacy_group_assets(items, Path(path).parent, catalog, traditional_models=traditional_models)
    try:
        atomic_write_json(path, [item.to_dict() for item in items], ensure_ascii=False, indent=2)
    except Exception:
        for item, old_key in changed:
            item.legacy_model_key = old_key
        raise


def build_default_item(
    roi_label: str,
    *,
    camera_id: str = DEFAULT_CAMERA_ROLE,
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
    default_camera_id: str = DEFAULT_CAMERA_ROLE,
    display_names_by_label: Mapping[str, str] | None = None,
    task_groups_by_label: Mapping[str, str] | None = None,
) -> List[InspectionItem]:
    """
    以 labels 的顺序为准同步检测项。

    规则：
      - 已存在且 roi_label 相同的项：保留 display_name / camera_id / enabled / algorithm_code / params
      - 新标签：生成默认项
      - 不再存在的标签：移除
    """
    existing_items = list(existing_items)
    normalized_labels = [str(label).strip() for label in labels if str(label).strip()]
    display_names_by_label = {
        str(label).strip(): str(name).strip()
        for label, name in dict(display_names_by_label or {}).items()
        if str(label).strip()
    }
    task_groups_by_label = {
        str(label).strip(): str(group).strip()
        for label, group in dict(task_groups_by_label or {}).items()
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
        default_group = task_groups_by_label.get(label, "")
        if existing is not None:
            existing_display_name = str(existing.display_name or "").strip()
            if display_name == label and existing_display_name and existing_display_name != label:
                display_name = existing_display_name
            existing_group = str(existing.task_group or "").strip()
            source = existing.model_source
            legacy_key = existing.legacy_model_key
            if default_group and not getattr(existing, "_model_source_explicit", True) and (
                not existing_group
                or re.fullmatch(r"roi(?:\d+)?(?:__\d+)?", existing_group, flags=re.IGNORECASE)
            ):
                existing_group = default_group
                source = "group"
                legacy_key = legacy_group_model_key(existing.camera_id, existing_group)
            synced.append(
                InspectionItem(
                    item_id=existing.item_id or label,
                    display_name=display_name or existing.display_name or label,
                    camera_id=existing.camera_id if existing.camera_id in SUPPORTED_CAMERA_IDS else default_camera_id,
                    roi_label=label,
                    algorithm_code=existing.algorithm_code or SHARED_BACKBONE_ALGORITHM_CODE,
                    enabled=bool(existing.enabled),
                    params=dict(existing.params or {}),
                    task_group=existing_group,
                    model_source=source, legacy_model_key=legacy_key,
                    library_model_id=existing.library_model_id, library_version=existing.library_version,
                )
            )
        else:
            item = build_default_item(
                label,
                camera_id=default_camera_id,
                display_name=display_name or label,
            )
            item.task_group = default_group
            if default_group:
                item.model_source = "group"
                peer = next((peer for peer in existing_items
                             if peer.model_source == "group" and peer.task_group == default_group
                             and peer.camera_id == item.camera_id), None)
                item.legacy_model_key = peer.legacy_model_key if peer is not None else ""
            item._model_source_explicit = True
            synced.append(item)
    existing_distance_items = [
        item
        for item in existing_items
        if isinstance(item, InspectionItem)
        and item.algorithm_code in POST_DISTANCE_ALGORITHMS
        and item.item_id not in {synced_item.item_id for synced_item in synced}
    ]
    for existing in existing_distance_items:
        fallback_name = "Center Distance" if existing.algorithm_code == "center_distance" else "Line Distance"
        synced.append(
            InspectionItem(
                item_id=existing.item_id or existing.display_name or existing.algorithm_code or "distance",
                display_name=existing.display_name or existing.item_id or fallback_name,
                camera_id=existing.camera_id if existing.camera_id in SUPPORTED_CAMERA_IDS else default_camera_id,
                roi_label=existing.roi_label,
                algorithm_code=existing.algorithm_code,
                enabled=bool(existing.enabled),
                params=dict(existing.params or {}),
                task_group=str(existing.task_group or "").strip(),
                model_source=existing.model_source, legacy_model_key=existing.legacy_model_key,
                library_model_id=existing.library_model_id, library_version=existing.library_version,
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
