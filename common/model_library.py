"""Immutable, local model versions shared by product recipes.

This module deliberately depends only on the standard library and common path
helpers. Loading Torch/OpenCV models remains the algorithm controller's job.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from common.algorithm_codes import LEARNING_BACKBONES, EMBEDDING_BACKBONES, YOLO_CLASSIFICATION_ALGORITHMS, storage_code_backbone
from common.app_paths import MODEL_LIBRARY_DIRECTORY, SESSION_DIRECTORY, writable_embedding_test_root


_MODEL_ID = re.compile(r"[0-9a-f]{32}\Z")
_VERSION = re.compile(r"v[0-9]{6}\Z")
_LIBRARY_KEY = re.compile(
    r"(?P<camera>[A-Za-z0-9._-]+)__library__(?P<model_id>[0-9a-f]{32})__(?P<version>v[0-9]{6})\Z"
)
_TRADITIONAL_ALGORITHMS = {"meanintensity", "meanstd", "meanhsv_h", "meanhsv_v", "meanhsv_s"}
_MIGRATION_LOCK = threading.RLock()


def _validate_identity(model_id: object, version: object | None = None) -> tuple[str, str]:
    identifier = str(model_id or "").strip()
    if not _MODEL_ID.fullmatch(identifier):
        raise ValueError("Model ID must contain exactly 32 lowercase hexadecimal characters")
    revision = "" if version is None else str(version or "").strip()
    if version is not None and (not _VERSION.fullmatch(revision) or int(revision[1:]) < 1):
        raise ValueError("Model version must have the form v000001")
    return identifier, revision


def make_library_model_key(camera_id: object, model_id: object, version: object) -> str:
    identifier, revision = _validate_identity(model_id, version)
    camera = str(camera_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", camera) or "__" in camera:
        raise ValueError("Invalid camera ID for a library model key")
    return f"{camera}__library__{identifier}__{revision}"


def parse_library_model_key(model_key: object) -> tuple[str, str] | None:
    key = str(model_key or "").strip()
    if "__library__" not in key:
        return None
    matched = _LIBRARY_KEY.fullmatch(key)
    if matched is None or "__" in matched.group("camera"):
        raise ValueError("Invalid library model reference")
    return _validate_identity(matched.group("model_id"), matched.group("version"))


def _algorithm(value: object) -> str:
    normalized = storage_code_backbone(value)
    if normalized not in LEARNING_BACKBONES and normalized not in _TRADITIONAL_ALGORITHMS:
        raise ValueError(f"Unsupported library model algorithm: {normalized}")
    return normalized


def _model_kind(algorithm: str) -> str:
    if algorithm in YOLO_CLASSIFICATION_ALGORITHMS:
        return "yolo"
    return "embedding" if algorithm in EMBEDDING_BACKBONES else "traditional"


def _json_object(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON-compatible values") from exc


def _parameters(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = {"score_mode": "proto", "margin": 0.02, "topk": 3}
    result.update(_json_object(value, label="Model parameters"))
    if not isinstance(result["score_mode"], str) or result["score_mode"] not in {"proto", "topk"}:
        raise ValueError("Invalid model score mode")
    try:
        result["margin"] = float(result["margin"])
        topk = int(result["topk"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid model scoring parameters") from exc
    if isinstance(result["topk"], bool) or not math.isfinite(result["margin"]) or topk < 1 or topk != result["topk"]:
        raise ValueError("Invalid model scoring parameters")
    result["topk"] = topk
    if "yolo_ng_threshold" in result:
        threshold = float(result["yolo_ng_threshold"])
        if not math.isfinite(threshold) or not 0. <= threshold <= 1.:
            raise ValueError("Invalid classifier NG threshold")
        result["yolo_ng_threshold"] = threshold
    return result


def _classifier_metadata(value) -> dict:
    result = _json_object(value, label="Classifier metadata")
    names = result.get("names", {})
    values = list(names.values()) if isinstance(names, dict) else names if isinstance(names, list) else []
    if not {"ok", "ng"}.issubset({str(name).strip().lower() for name in values}):
        raise ValueError("Classifier metadata must contain OK and NG classes")
    return result


def _traditional_model(value: Mapping[str, Any], algorithm: str) -> dict[str, Any]:
    result = _json_object(value, label="Traditional model")
    if str(result.get("algorithm", "")).strip() != algorithm:
        raise ValueError("Traditional model algorithm does not match its publication")
    if result.get("ok_when") not in {"greater_equal", "less_equal"}:
        raise ValueError("Invalid traditional model comparison")
    try:
        threshold = float(result["threshold"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid traditional model threshold") from exc
    if not math.isfinite(threshold):
        raise ValueError("Invalid traditional model threshold")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelLibrary:
    def __init__(self, root: str | Path | None = None) -> None:
        app_root = writable_embedding_test_root(__file__) if root is None else None
        self.root = Path(root if root is not None else app_root / SESSION_DIRECTORY / MODEL_LIBRARY_DIRECTORY).expanduser().resolve()
        self._lock = threading.RLock()
        self._verified: dict[tuple[str, str], tuple[tuple, dict[str, Any]]] = {}
        if app_root is not None:
            if any((self.root / filename).exists() for filename in (
                "session.json", "inspection_items.json", "product_params.json",
            )):
                raise ValueError(f"公共模型库目录与已有订单冲突，请先重命名该订单：{self.root}")
            self._import_legacy_versions(app_root / MODEL_LIBRARY_DIRECTORY)

    def _import_legacy_versions(self, legacy_root: Path) -> None:
        """Copy verified old versions atomically, retaining the old library.

        Keeping the old location lets an already running older application keep
        working. Its later publications are imported on the next startup too.
        """
        legacy_root = legacy_root.resolve()
        if legacy_root == self.root or not legacy_root.is_dir():
            return
        with _MIGRATION_LOCK:
            legacy = ModelLibrary(legacy_root)
            for model in legacy.list_models():
                identifier, version = model["model_id"], model["version"]
                destination = self._directory(identifier, version)
                manifest = Path(model["manifest_path"])
                if destination.exists():
                    self.resolve(identifier, version)
                    if manifest.read_bytes() != (destination / "manifest.json").read_bytes():
                        raise FileExistsError(
                            f"公共模型迁移冲突，两个位置的 {identifier}/{version} 内容不同，均未覆盖："
                            f"{legacy_root}；{self.root}"
                        )
                    continue
                self.root.mkdir(parents=True, exist_ok=True)
                stage = self._within_root(Path(tempfile.mkdtemp(prefix=".migrate-", dir=self.root)))
                try:
                    staged_version = stage / identifier / version
                    staged_version.mkdir(parents=True)
                    shutil.copy2(manifest, staged_version / "manifest.json")
                    for asset in model["assets"]:
                        entry = str(asset["file"])
                        shutil.copy2(legacy._asset_path(manifest.parent, entry), staged_version / entry)
                    # Check the copied bytes before making this version visible.
                    ModelLibrary(stage).resolve(identifier, version)
                    checked_version = self._within_root(staged_version)
                    checked_destination = self._directory(identifier, version)
                    checked_destination.parent.mkdir(parents=True, exist_ok=True)
                    checked_version.rename(checked_destination)
                finally:
                    checked_stage = self._within_root(stage)
                    if checked_stage.parent != self.root or not checked_stage.name.startswith(".migrate-"):
                        raise ValueError("Invalid temporary migration directory")
                    shutil.rmtree(checked_stage)

    def _within_root(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Model library path escapes the configured library root")
        return resolved

    def _directory(self, model_id: str, version: str) -> Path:
        identifier, revision = _validate_identity(model_id, version)
        return self._within_root(self.root / identifier / revision)

    def _asset_path(self, directory: Path, filename: object) -> Path:
        name = str(filename or "")
        if not name or name in {".", ".."} or any(character in name for character in "/\\:"):
            raise ValueError("Invalid model asset filename")
        path = self._within_root(directory / name)
        if path.parent != directory:
            raise ValueError("Model asset escapes its immutable version directory")
        return path

    @staticmethod
    def _stat_signature(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        if not path.is_file():
            raise ValueError(f"Model asset is not a file: {path.name}")
        return int(stat.st_mtime_ns), int(stat.st_size), int(stat.st_ctime_ns), int(stat.st_ino)

    def _signature(self, directory: Path, manifest: dict[str, Any]) -> tuple:
        manifest_path = self._asset_path(directory, "manifest.json")
        signatures: list[tuple] = [(str(manifest_path), self._stat_signature(manifest_path))]
        for asset in manifest["assets"]:
            path = self._asset_path(directory, asset["file"])
            signatures.append((str(path), self._stat_signature(path)))
        return tuple(signatures)

    def publish(
        self,
        name: str,
        algorithm: str,
        model_file: str | Path | None = None,
        traditional_model: Mapping[str, Any] | None = None,
        classifier_metadata: str | Path | None = None,
        parameters: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        display_name = str(name or "").strip()
        if not display_name:
            raise ValueError("A model name is required")
        normalized_algorithm = _algorithm(algorithm)
        kind = _model_kind(normalized_algorithm)
        if (model_file is None) == (traditional_model is None):
            raise ValueError("Publish exactly one embedding file or traditional model")
        if model_file is not None and normalized_algorithm not in LEARNING_BACKBONES:
            raise ValueError("Only embedding algorithms can publish an NPZ model")
        if traditional_model is not None and normalized_algorithm not in _TRADITIONAL_ALGORITHMS:
            raise ValueError("Only traditional algorithms can publish threshold models")
        identifier, _ = _validate_identity(model_id or uuid.uuid4().hex)
        scoring = _parameters(parameters)
        if kind == "yolo":
            scoring.setdefault("yolo_ng_threshold", .5)
        provenance = _json_object(source, label="Model source")
        source_file = None if model_file is None else Path(model_file).resolve()
        expected_suffix = ".onnx" if kind == "yolo" else ".npz"
        if source_file is not None and (source_file.suffix.lower() != expected_suffix or not source_file.is_file()):
            raise ValueError(f"An existing {expected_suffix} model file is required")
        metadata_file = None if classifier_metadata is None else Path(classifier_metadata).resolve()
        if kind == "yolo":
            if source_file is None or metadata_file is None or not metadata_file.is_file():
                raise ValueError("Classifier publication requires ONNX and metadata files")
            _classifier_metadata(json.loads(metadata_file.read_text(encoding="utf-8")))
        elif metadata_file is not None:
            raise ValueError("Classifier metadata is only valid for classifier models")
        threshold_model = None if traditional_model is None else _traditional_model(traditional_model, normalized_algorithm)

        with self._lock:
            model_directory = self._within_root(self.root / identifier)
            model_directory.mkdir(parents=True, exist_ok=True)
            versions = [int(path.name[1:]) for path in model_directory.iterdir() if path.is_dir() and _VERSION.fullmatch(path.name)]
            next_version = max(versions, default=0) + 1
            if next_version > 999999:
                raise ValueError("Model version limit reached")
            revision = f"v{next_version:06d}"
            destination = self._directory(identifier, revision)
            stage = self._within_root(Path(tempfile.mkdtemp(prefix=".publish-", dir=model_directory)))
            try:
                filename = "model.onnx" if kind == "yolo" else "model.npz" if source_file is not None else "threshold_model.json"
                artifact = self._asset_path(stage, filename)
                if source_file is not None:
                    shutil.copyfile(source_file, artifact)
                else:
                    artifact.write_text(json.dumps(threshold_model, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                assets = [{"file": filename, "size": artifact.stat().st_size, "sha256": _sha256(artifact)}]
                if metadata_file is not None:
                    metadata_target = self._asset_path(stage, "classifier.json")
                    shutil.copyfile(metadata_file, metadata_target)
                    assets.append({"file": "classifier.json", "size": metadata_target.stat().st_size, "sha256": _sha256(metadata_target)})
                manifest = {
                    "schema_version": 1,
                    "model_id": identifier,
                    "version": revision,
                    "name": display_name,
                    "algorithm": normalized_algorithm,
                    "kind": kind,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parameters": scoring,
                    "source": provenance,
                    "entry_file": filename,
                    "assets": assets,
                }
                (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                # rename never replaces a published, nonempty version directory.
                stage.rename(destination)
            finally:
                if stage.exists():
                    checked_stage = self._within_root(stage)
                    if checked_stage.parent != model_directory or not checked_stage.name.startswith(".publish-"):
                        raise ValueError("Invalid temporary publication directory")
                    shutil.rmtree(checked_stage)
            return self.resolve(identifier, revision, normalized_algorithm)

    def resolve(self, model_id: str, version: str, algorithm: str | None = None) -> dict[str, Any]:
        identifier, revision = _validate_identity(model_id, version)
        expected_algorithm = None if algorithm is None else _algorithm(algorithm)
        directory = self._directory(identifier, revision)
        identity = (identifier, revision)
        with self._lock:
            cached = self._verified.get(identity)
            if cached is not None:
                signature, result = cached
                if signature == self._signature(directory, result):
                    if expected_algorithm is not None and result["algorithm"] != expected_algorithm:
                        raise ValueError("Referenced model algorithm does not match the selected algorithm")
                    return copy.deepcopy(result)

            manifest_path = self._asset_path(directory, "manifest.json")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError("Invalid model library manifest") from exc
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
                raise ValueError("Unsupported model library manifest")
            if manifest.get("model_id") != identifier or manifest.get("version") != revision:
                raise ValueError("Model manifest identity does not match its version directory")
            normalized_algorithm = _algorithm(manifest.get("algorithm"))
            kind = _model_kind(normalized_algorithm)
            if manifest.get("kind") != kind:
                raise ValueError("Model kind does not match its algorithm")
            if expected_algorithm is not None and normalized_algorithm != expected_algorithm:
                raise ValueError("Referenced model algorithm does not match the selected algorithm")
            if not str(manifest.get("name", "")).strip():
                raise ValueError("Model manifest is missing its name")
            manifest["parameters"] = _parameters(manifest.get("parameters"))
            manifest["source"] = _json_object(manifest.get("source"), label="Model source")
            assets = manifest.get("assets")
            expected_file = "model.onnx" if kind == "yolo" else "model.npz" if kind == "embedding" else "threshold_model.json"
            expected_files = {expected_file, "classifier.json"} if kind == "yolo" else {expected_file}
            if not isinstance(assets, list) or len(assets) != len(expected_files) or not all(isinstance(asset, dict) for asset in assets):
                raise ValueError("Model manifest describes incorrect model assets")
            if manifest.get("entry_file") != expected_file or {asset.get("file") for asset in assets} != expected_files:
                raise ValueError("Invalid model entry filename")
            for asset in assets:
                if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256", ""))):
                    raise ValueError("Invalid model asset checksum")
                if type(asset.get("size")) is not int or asset["size"] < 0:
                    raise ValueError("Invalid model asset size")
            before = self._signature(directory, manifest)
            artifact = self._asset_path(directory, expected_file)
            for asset in assets:
                asset_path = self._asset_path(directory, asset["file"])
                if asset_path.stat().st_size != asset["size"] or _sha256(asset_path) != asset["sha256"]:
                    raise ValueError("Model asset checksum verification failed")
            threshold_model = None
            if kind == "traditional":
                try:
                    threshold_model = _traditional_model(json.loads(artifact.read_text(encoding="utf-8")), normalized_algorithm)
                except (json.JSONDecodeError, UnicodeError) as exc:
                    raise ValueError("Invalid traditional model asset") from exc
            if kind == "yolo":
                _classifier_metadata(json.loads(self._asset_path(directory, "classifier.json").read_text(encoding="utf-8")))
                manifest["parameters"].setdefault("yolo_ng_threshold", .5)
            after = self._signature(directory, manifest)
            if before != after:
                raise ValueError("Model files changed during verification")
            result = dict(manifest)
            result.update({
                "directory": str(directory),
                "manifest_path": str(manifest_path),
                "model_file": str(artifact) if kind in {"embedding", "yolo"} else "",
                "classifier_metadata": str(self._asset_path(directory, "classifier.json")) if kind == "yolo" else "",
                "traditional_model": threshold_model,
            })
            self._verified[identity] = (after, result)
            return copy.deepcopy(result)

    def list_models(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        models: list[dict[str, Any]] = []
        for model_directory in self.root.iterdir():
            if not _MODEL_ID.fullmatch(model_directory.name) or not model_directory.is_dir():
                continue
            try:
                self._within_root(model_directory)
                versions = list(model_directory.iterdir())
            except (OSError, ValueError):
                continue
            for version_directory in versions:
                if not _VERSION.fullmatch(version_directory.name) or not version_directory.is_dir():
                    continue
                try:
                    models.append(self.resolve(model_directory.name, version_directory.name))
                except (OSError, TypeError, ValueError):
                    # A damaged version remains unavailable; never substitute a
                    # different version for recipes already referencing it.
                    continue
        return sorted(models, key=lambda model: (model["name"].casefold(), model["model_id"], model["version"]))


__all__ = ["ModelLibrary", "make_library_model_key", "parse_library_model_key"]
