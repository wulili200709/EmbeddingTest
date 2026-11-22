"""Fork old group assets when an independent ROI owns the same legacy key."""
from __future__ import annotations

import hashlib
import copy
import os
import shutil
import tempfile
from pathlib import Path

from common.model_groups import make_group_model_key, legacy_ascii_model_key
from common.safe_io import atomic_write_json, load_json_with_backup


def _digest(path: Path) -> bytes:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").digest()


def separate_legacy_group_assets(items, product_dir: Path, catalog, *, traditional_models=None) -> list[tuple]:
    root = product_dir.resolve()
    independent_keys = {item.model_key for item in items if item.model_source == "independent"}
    if not independent_keys:
        return []
    records = catalog._saved()
    archived = {(record["camera_id"], record["task_group"]) for record in records if not record.get("legacy_model_key")}
    candidates = {(record["camera_id"], record["task_group"]): record["legacy_model_key"]
                  for record in records if record.get("legacy_model_key")}
    candidates.update({(item.camera_id, item.task_group): item.legacy_model_key for item in items if item.model_source == "group" and item.legacy_model_key})
    if not any(key in independent_keys for key in candidates.values()):
        return []
    params_path = root / "product_params.json"
    params = load_json_with_backup(params_path, default={})
    models = params.get("traditional_models", {}) if isinstance(params, dict) else {}
    models = dict(models) if isinstance(models, dict) else {}
    if traditional_models is not None:
        models.update(traditional_models)
    params_changed = False
    copied_thresholds = {}
    completed = []
    created = []
    try:
        for (camera, name), old_key in candidates.items():
            if old_key not in independent_keys:
                continue
            new_key = make_group_model_key(camera, name)
            if (camera, name) not in archived:
                old_keys = list(dict.fromkeys([old_key, legacy_ascii_model_key(old_key)]))
                assets = {}
                for candidate in old_keys:
                    paths = list(root.glob(f"{candidate}_register_model_*.npz"))
                    paths += [path for suffix in (".onnx", ".json", ".pt")
                              if (path := root / f"{candidate}_y8ncls{suffix}").is_file()]
                    for source in paths:
                        assets.setdefault(source.name[len(candidate):], source)
                for suffix, source in assets.items():
                    destination = root / (new_key + suffix)
                    if source.resolve().parent != root or destination.resolve().parent != root:
                        raise ValueError("Group asset path escapes the order directory")
                    if destination.exists():
                        if _digest(source) != _digest(destination):
                            raise FileExistsError(f"Group migration would overwrite another model: {destination}")
                        continue
                    fd, temp_name = tempfile.mkstemp(prefix=".group-fork-", dir=root)
                    os.close(fd)
                    temporary = Path(temp_name)
                    try:
                        shutil.copy2(source, temporary)
                        if _digest(source) != _digest(temporary):
                            raise OSError("Group asset changed while copying")
                        # Both checked paths stay in this order; Windows rename
                        # refuses to replace an existing destination file.
                        temporary.rename(destination)
                        created.append(destination)
                    finally:
                        if temporary.resolve().parent != root:
                            raise ValueError("Invalid temporary group asset path")
                        temporary.unlink(missing_ok=True)
                thresholds = {}
                for candidate in old_keys:
                    for key, model in list(models.items()):
                        if key.endswith("::" + candidate):
                            thresholds.setdefault(key[:-len(candidate)] + new_key, model)
                for destination_key, model in thresholds.items():
                    if destination_key in models and models[destination_key] != model:
                        raise FileExistsError("Group migration would overwrite another threshold model")
                    models[destination_key] = copy.deepcopy(model)
                    copied_thresholds[destination_key] = models[destination_key]
                    params_changed = True
            completed.append((camera, name, old_key))
        if params_changed:
            params["traditional_models"] = models
            atomic_write_json(params_path, params)
        for camera, name, _old_key in completed:
            catalog.remember_group(camera, name)
    except Exception:
        for path in created:
            if path.resolve().parent != root:
                raise ValueError("Invalid group asset cleanup path")
            path.unlink(missing_ok=True)
        raise
    if traditional_models is not None:
        traditional_models.update(copied_thresholds)
    changed = []
    for item in items:
        if item.model_source == "group" and (item.camera_id, item.task_group, item.legacy_model_key) in completed:
            changed.append((item, item.legacy_model_key))
            item.legacy_model_key = ""
    return changed
