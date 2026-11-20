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

from common.algorithm_codes import LEARNING_BACKBONES, storage_code_backbone
from common.app_paths import writable_embedding_test_root


_MODEL_ID = re.compile(r"[0-9a-f]{32}\Z")
_VERSION = re.compile(r"v[0-9]{6}\Z")
_LIBRARY_KEY = re.compile(
    r"(?P<camera>[A-Za-z0-9._-]+)__library__(?P<model_id>[0-9a-f]{32})__(?P<version>v[0-9]{6})\Z"
)
_TRADITIONAL_ALGORITHMS = {"meanintensity", "meanstd", "meanhsv_h", "meanhsv_v", "meanhsv_s"}


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
        self.root = Path(root if root is not None else writable_embedding_test_root(__file__) / "model_library").expanduser().resolve()
        self._lock = threading.RLock()
        self._verified: dict[tuple[str, str], tuple[tuple, dict[str, Any]]] = {}

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
        parameters: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        display_name = str(name or "").strip()
        if not display_name:
            raise ValueError("A model name is required")
        normalized_algorithm = _algorithm(algorithm)
        if (model_file is None) == (traditional_model is None):
            raise ValueError("Publish exactly one embedding file or traditional model")
        if model_file is not None and normalized_algorithm not in LEARNING_BACKBONES:
            raise ValueError("Only embedding algorithms can publish an NPZ model")
        if traditional_model is not None and normalized_algorithm not in _TRADITIONAL_ALGORITHMS:
            raise ValueError("Only traditional algorithms can publish threshold models")
        identifier, _ = _validate_identity(model_id or uuid.uuid4().hex)
        scoring = _parameters(parameters)
        provenance = _json_object(source, label="Model source")
        source_file = None if model_file is None else Path(model_file).resolve()
        if source_file is not None and (source_file.suffix.lower() != ".npz" or not source_file.is_file()):
            raise ValueError("An existing NPZ model file is required")
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
                filename = "model.npz" if source_file is not None else "threshold_model.json"
                artifact = self._asset_path(stage, filename)
                if source_file is not None:
                    shutil.copyfile(source_file, artifact)
                else:
                    artifact.write_text(json.dumps(threshold_model, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                manifest = {
                    "schema_version": 1,
                    "model_id": identifier,
                    "version": revision,
                    "name": display_name,
                    "algorithm": normalized_algorithm,
                    "kind": "embedding" if source_file is not None else "traditional",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parameters": scoring,
                    "source": provenance,
                    "entry_file": filename,
                    "assets": [{"file": filename, "size": artifact.stat().st_size, "sha256": _sha256(artifact)}],
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
            kind = "embedding" if normalized_algorithm in LEARNING_BACKBONES else "traditional"
            if manifest.get("kind") != kind:
                raise ValueError("Model kind does not match its algorithm")
            if expected_algorithm is not None and normalized_algorithm != expected_algorithm:
                raise ValueError("Referenced model algorithm does not match the selected algorithm")
            if not str(manifest.get("name", "")).strip():
                raise ValueError("Model manifest is missing its name")
            manifest["parameters"] = _parameters(manifest.get("parameters"))
            manifest["source"] = _json_object(manifest.get("source"), label="Model source")
            assets = manifest.get("assets")
            expected_file = "model.npz" if kind == "embedding" else "threshold_model.json"
            if not isinstance(assets, list) or len(assets) != 1 or not isinstance(assets[0], dict):
                raise ValueError("Model manifest must describe exactly one model asset")
            if manifest.get("entry_file") != expected_file or assets[0].get("file") != expected_file:
                raise ValueError("Invalid model entry filename")
            asset = assets[0]
            if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256", ""))):
                raise ValueError("Invalid model asset checksum")
            if type(asset.get("size")) is not int or asset["size"] < 0:
                raise ValueError("Invalid model asset size")
            before = self._signature(directory, manifest)
            artifact = self._asset_path(directory, expected_file)
            if artifact.stat().st_size != asset["size"] or _sha256(artifact) != asset["sha256"]:
                raise ValueError("Model asset checksum verification failed")
            threshold_model = None
            if kind == "traditional":
                try:
                    threshold_model = _traditional_model(json.loads(artifact.read_text(encoding="utf-8")), normalized_algorithm)
                except (json.JSONDecodeError, UnicodeError) as exc:
                    raise ValueError("Invalid traditional model asset") from exc
            after = self._signature(directory, manifest)
            if before != after:
                raise ValueError("Model files changed during verification")
            result = dict(manifest)
            result.update({
                "directory": str(directory),
                "manifest_path": str(manifest_path),
                "model_file": str(artifact) if kind == "embedding" else "",
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
