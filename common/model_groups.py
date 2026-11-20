"""Keep order model groups even when no ROI currently references them."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

from common.camera_roles import DEFAULT_CAMERA_ROLE, normalize_camera_role
from common.safe_io import atomic_write_json, load_json_with_backup


def group_slug(value: object) -> str:
    return re.sub(r"[^\w.-]+", "_", str(value or "").strip(), flags=re.UNICODE).strip("._-")


def make_group_model_key(camera_id: object, name: object) -> str:
    camera = normalize_camera_role(camera_id, default=DEFAULT_CAMERA_ROLE)
    group = str(name or "").strip()
    slug = group_slug(group)
    if not slug or "__library__" in slug:
        raise ValueError("Invalid shared model group name")
    identity = hashlib.sha256(group.encode("utf-8")).hexdigest()[:10]
    return f"{camera}__group__{slug}_{identity}"


_GROUP_KEY = re.compile(r"(?P<camera>cam[1-3])__group__(?P<slug>.+)_[0-9a-f]{10}\Z")


class OrderModelGroups:
    def __init__(self, product_dir: str | Path) -> None:
        self.root = Path(product_dir)
        self.path = self.root / "model_groups.json"

    @staticmethod
    def _record(camera_id, name) -> dict | None:
        camera = normalize_camera_role(camera_id)
        group = str(name or "").strip()
        if not camera:
            return None
        try:
            return {"camera_id": camera, "task_group": group,
                    "model_key": make_group_model_key(camera, group)}
        except ValueError:
            return None

    def _saved(self) -> list[dict]:
        payload = load_json_with_backup(self.path, default={})
        groups = payload.get("groups", []) if isinstance(payload, dict) else []
        if not isinstance(groups, list):
            return []
        return [record for entry in groups if isinstance(entry, dict)
                if (record := self._record(entry.get("camera_id"), entry.get("task_group"))) is not None]

    def _remember(self, records: Iterable[dict]) -> list[dict]:
        previous = self._saved()
        merged = {(record["camera_id"], record["task_group"]): record for record in previous}
        for record in records:
            merged[(record["camera_id"], record["task_group"])] = record
        result = sorted(merged.values(), key=lambda row: (row["camera_id"], row["task_group"].casefold(), row["task_group"]))
        if result != previous:
            atomic_write_json(self.path, {"schema_version": 1, "groups": result})
        return result

    def remember_items(self, *item_sets: Iterable[Mapping]) -> None:
        records = []
        for items in item_sets:
            for item in items:
                if not isinstance(item, Mapping) or item.get("model_source") != "group":
                    continue
                record = self._record(item.get("camera_id"), item.get("task_group"))
                if record is not None:
                    records.append(record)
        self._remember(records)

    @staticmethod
    def _recover_key(key: str) -> dict | None:
        matched = _GROUP_KEY.fullmatch(key)
        if matched is None:
            return None
        # Only use a recovered name when its hash reproduces the existing key.
        # Never bind an ambiguous filename to a different model by guessing.
        slug = matched.group("slug")
        for name in (slug, slug.replace("_", " ")):
            record = OrderModelGroups._record(matched.group("camera"), name)
            if record is not None and record["model_key"] == key:
                return record
        return None

    def list_groups(self, camera_id=None, *, public_models: Iterable[Mapping] = ()) -> list[dict]:
        records = []
        for name in ("inspection_items.json", "inspection_items.json.bak"):
            items = load_json_with_backup(self.root / name, default=[])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("model_source") == "group":
                        record = self._record(item.get("camera_id"), item.get("task_group"))
                        if record is not None:
                            records.append(record)
        for path in self.root.glob("*_register_model_*.npz"):
            key = path.name.rsplit("_register_model_", 1)[0]
            record = self._recover_key(key)
            if record is not None:
                records.append(record)
        params = load_json_with_backup(self.root / "product_params.json", default={})
        models = params.get("traditional_models", {}) if isinstance(params, dict) else {}
        if isinstance(models, dict):
            for key in models:
                record = self._recover_key(str(key).partition("::")[2])
                if record is not None:
                    records.append(record)
        for model in public_models:
            source = model.get("source", {}) if isinstance(model, Mapping) else {}
            if not isinstance(source, Mapping) or os.path.normcase(str(source.get("order", ""))) != os.path.normcase(self.root.name):
                continue
            key = str(source.get("model_key", ""))
            matched = _GROUP_KEY.fullmatch(key)
            if matched is not None:
                record = self._record(matched.group("camera"), source.get("group"))
                if record is not None and record["model_key"] == key:
                    records.append(record)
        result = self._remember(records)
        if camera_id is None:
            return result
        camera = normalize_camera_role(camera_id)
        return [record for record in result if record["camera_id"] == camera]
