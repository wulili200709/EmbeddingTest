"""Inspection-item storage partitioned by the active image coordinate system."""
from __future__ import annotations

from pathlib import Path
import threading

from common.camera_roles import CAMERA_ROLES
from domain.inspection_items import (
    load_inspection_items as _load,
    save_inspection_items as _save,
)
from services import lens_correction

_LOCK = threading.RLock()


def _scoped_paths(path):
    root = str(Path(path).parent)
    return {
        role: str(Path(lens_correction.asset_root(root, role)) / "inspection_items.json")
        for role in CAMERA_ROLES
        if lens_correction.active_signature(root, role) != "raw"
    }


def load_inspection_items(path):
    with _LOCK:
        scoped = _scoped_paths(path)
        original = _load(path)
        if not scoped:
            return original
        result = [item for item in original if item.camera_id not in scoped]
        for role, target in scoped.items():
            result.extend(item for item in _load(target) if item.camera_id == role)
        return result


def save_inspection_items(items, path):
    with _LOCK:
        scoped = _scoped_paths(path)
        items = list(items)
        if not scoped:
            _save(items, path)
            return
        original = _load(path)
        old_raw = [item for item in original if item.camera_id not in scoped]
        new_raw = [item for item in items if item.camera_id not in scoped]
        # A corrected-camera edit must leave its old raw configuration bytes
        # intact. Only an actual edit to a raw camera warrants rewriting it.
        if [item.to_dict() for item in old_raw] != [item.to_dict() for item in new_raw]:
            retained = [item for item in original if item.camera_id in scoped]
            _save([*new_raw, *retained], path)
        for role, target in scoped.items():
            selected = [item for item in items if item.camera_id == role]
            if selected or Path(target).exists():
                _save(selected, target)
