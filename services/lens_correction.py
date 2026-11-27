"""Opt-in, versioned lens correction shared by offline and live inspection.

No calibration module is imported on the disabled path. Imported profiles are
immutable product-local snapshots; changing the original file never updates an
active camera. Corrected assets and images live in a separate coordinate space.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Mapping

from common.camera_roles import normalize_camera_role
from common.safe_io import atomic_write_json


SETTINGS_SCHEMA = "lc-lens-settings-v1"
VERSION_SCHEMA = "lc-lens-import-v1"
SIDECAR_SCHEMA = "lc-lens-image-v1"
_LOCK = threading.RLock()
_MAP_LOCK = threading.RLock()
_MAP_CACHE: OrderedDict = OrderedDict()
_VALIDATED_CACHE: OrderedDict = OrderedDict()
_MAX_MAPS = 4
_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class LensCorrectionError(ValueError):
    """A correction/configuration failure must not fall back to raw inspection."""


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = False
    profile_id: str = ""
    profile_name: str = ""
    mode: str = ""
    image_size: tuple[int, int] = ()
    camera_serial: str = ""
    profile_path: str = ""
    alpha: float = 1.0


@dataclass(frozen=True)
class CorrectionResult:
    image: Any
    correction_ms: float
    profile_id: str
    corrected: bool


def _role(value) -> str:
    role = normalize_camera_role(value)
    if not role:
        raise LensCorrectionError(f"无效的物理相机角色：{value!r}")
    return role


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("JSON 顶层必须是对象")
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise LensCorrectionError(f"无法读取镜头矫正文件 {path}：{exc}") from exc


def _settings(product_dir) -> dict:
    path = Path(product_dir) / "lens_correction_settings.json"
    if not path.exists():
        return {"schema": SETTINGS_SCHEMA, "cameras": {}}
    data = _read_json(path)
    if data.get("schema") != SETTINGS_SCHEMA or not isinstance(data.get("cameras"), dict):
        raise LensCorrectionError("镜头矫正设置格式或版本无效。")
    return data


def _config(product_dir, data: Mapping | None) -> CorrectionConfig:
    if data is None:
        return CorrectionConfig()
    if not isinstance(data, Mapping) or type(data.get("enabled", False)) is not bool:
        raise LensCorrectionError("镜头矫正开关必须为布尔值。")
    try:
        alpha = float(data.get("alpha", 1.0))
        identifier = str(data.get("profile_id", ""))
        size = tuple(data.get("image_size", ()))
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("视野参数必须在 0 到 1 之间")
        if identifier and not _ID_RE.fullmatch(identifier):
            raise ValueError("参数版本标识无效")
        if identifier and (len(size) != 2 or any(type(v) is not int or v < 2 for v in size)):
            raise ValueError("标定分辨率无效")
        if data.get("enabled", False) and not identifier:
            raise ValueError("尚未显式导入标定参数，不能启用矫正")
        # Only the version ID selects a file; a hand-edited path cannot escape the product.
        path = str((Path(product_dir) / "calibration" / "profiles" / f"{identifier}.json").resolve()) if identifier else ""
        return CorrectionConfig(
            enabled=data.get("enabled", False), profile_id=identifier,
            profile_name=str(data.get("profile_name", "")), mode=str(data.get("mode", "")),
            image_size=size, camera_serial=str(data.get("camera_serial", "")).strip(),
            profile_path=path, alpha=alpha,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise LensCorrectionError(f"镜头矫正设置无效：{exc}") from exc


def get_config(product_dir, camera_role, *, physical=False) -> CorrectionConfig:
    """Read metadata only; missing settings mean disabled, never an auto-selection."""
    role = _role(camera_role)
    with _LOCK:
        cameras = _settings(product_dir)["cameras"]
        if not cameras:
            return CorrectionConfig()
        if not physical:
            from services.lens_camera_roles import resolve_correction_role
            role = _role(resolve_correction_role(product_dir, role))
        return _config(product_dir, cameras.get(role))


def _profile(data: dict, alpha: float):
    try:
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("视野参数必须在 0 到 1 之间")
        if data.get("schema") in ("lc-lens-single-v1", "lc-lens-single-v2"):
            from algorithms.lens_single_calibration import SingleCalibrationProfile
            value = SingleCalibrationProfile.from_dict(data)
            if abs(alpha - 1.0) > 1e-12:
                raise ValueError("单张标定只支持固定画布，alpha 必须为 1")
        else:
            from algorithms.lens_calibration import CalibrationProfile
            value = CalibrationProfile.from_dict(data)
        canonical = value.to_dict()
        # Reject NaN anywhere, including diagnostic metadata retained in the snapshot.
        json.dumps(canonical, allow_nan=False)
        return value, canonical
    except Exception as exc:
        raise LensCorrectionError(f"标定参数无效：{exc}") from exc


def _geometry_id(data: dict, alpha: float) -> str:
    """Fingerprint the actual image mapping, excluding dates, views and RMS."""
    model = {"engine": "lc-float-remap-linear-zero-v1", "image_size": data["image_size"], "alpha": float(alpha)}
    if data["schema"].startswith("lc-lens-single-"):
        model.update(kind="single", mode=data["mode"], center=[float(v) for v in data["center"]],
                     radial_coeffs=data["radial_coeffs"])
        if data["mode"] == "radial_perspective":
            model["homography"] = data["homography"]
    else:
        distortion = list(data["dist_coeffs"])
        if len(distortion) == 4:
            distortion.append(0.0)
        model.update(kind="pinhole", camera_matrix=data["camera_matrix"], dist_coeffs=distortion)
    raw = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _make_rectifier(profile, alpha):
    try:
        if hasattr(profile, "mode"):
            from algorithms.lens_single_calibration import SingleUndistorter
            return SingleUndistorter(profile, alpha=alpha)
        from algorithms.lens_calibration import Undistorter
        return Undistorter(profile, alpha=alpha)
    except Exception as exc:
        raise LensCorrectionError(f"无法生成镜头矫正映射：{exc}") from exc


def _inspect(path, alpha):
    profile, payload = _profile(_read_json(Path(path)), alpha)
    identifier = _geometry_id(payload, alpha)
    config = CorrectionConfig(profile_id=identifier, profile_name=Path(path).stem,
                              mode=getattr(profile, "mode", "multi_view"),
                              image_size=tuple(int(v) for v in profile.image_size),
                              profile_path=str(Path(path).resolve()), alpha=float(alpha))
    return config, payload, profile


def inspect_profile(path, *, alpha=1.0) -> CorrectionConfig:
    """Validate an explicit import without changing any product settings."""
    config, _, profile = _inspect(path, alpha)
    _make_rectifier(profile, config.alpha)
    return config


def save_configs(product_dir, configs: Mapping[str, Mapping]) -> dict[str, CorrectionConfig]:
    """Validate every selected camera before one atomic settings commit."""
    if not isinstance(configs, Mapping):
        raise LensCorrectionError("相机矫正配置必须是映射。")
    with _LOCK:
        settings = _settings(product_dir)
        prepared = {}
        for raw_role, options in configs.items():
            role = _role(raw_role)
            if role in prepared or not isinstance(options, Mapping) or type(options.get("enabled")) is not bool:
                raise LensCorrectionError("相机配置重复或矫正开关无效。")
            old = _config(product_dir, settings["cameras"].get(role))
            serial = old.camera_serial if options.get("camera_serial") is None else str(options["camera_serial"]).strip()
            alpha = options.get("alpha", old.alpha)
            path = options.get("profile_path")
            if path:
                selected, payload, profile = _inspect(path, alpha)
                _make_rectifier(profile, selected.alpha)  # Fail before any active setting changes.
            elif not options["enabled"]:
                # Switching off preserves the exact selected version and does
                # not load calibration dependencies, even if its file is absent.
                selected, payload = old, None
            elif old.profile_id:
                selected, payload, profile = _inspect(old.profile_path, alpha)
                selected = replace(selected, profile_name=old.profile_name)
                # A change to the selected snapshot is an error, not an implicit new import.
                if _geometry_id(payload, old.alpha) != old.profile_id:
                    raise LensCorrectionError("已导入的标定版本内容被修改；请重新显式导入。")
                if options["enabled"]:
                    _make_rectifier(profile, selected.alpha)
            else:
                if options["enabled"]:
                    raise LensCorrectionError("请先为该相机显式导入标定参数。")
                selected, payload = CorrectionConfig(), None
            selected = replace(selected, enabled=options["enabled"], camera_serial=serial)
            prepared[role] = (selected, payload)
        result = {}
        for role, (selected, payload) in prepared.items():
            if payload is not None:
                destination = Path(product_dir) / "calibration" / "profiles" / f"{selected.profile_id}.json"
                version = dict(payload)
                version["_lens_correction"] = {"schema": VERSION_SCHEMA, "profile_id": selected.profile_id, "alpha": selected.alpha}
                if destination.exists():
                    _version(destination, selected.profile_id, selected.alpha)
                else:
                    atomic_write_json(destination, version, create_backup=False)
                selected = replace(selected, profile_path=str(destination.resolve()))
            stored = asdict(selected)
            stored["profile_path"] = f"calibration/profiles/{selected.profile_id}.json" if selected.profile_id else ""
            settings["cameras"][role] = stored
            result[role] = selected
        atomic_write_json(Path(product_dir) / "lens_correction_settings.json", settings, create_backup=False)
        return result


def save_config(product_dir, camera_role, *, enabled, profile_path=None, camera_serial=None, alpha=1.0) -> CorrectionConfig:
    role = _role(camera_role)
    return save_configs(product_dir, {role: dict(enabled=enabled, profile_path=profile_path,
                                               camera_serial=camera_serial, alpha=alpha)})[role]


def active_signature(product_dir, role) -> str:
    config = get_config(product_dir, role)
    return config.profile_id if config.enabled else "raw"


def asset_root(product_dir, role) -> str:
    role = _role(role)
    config = get_config(product_dir, role)
    return str(Path(product_dir) / "lens_assets" / role / config.profile_id) if config.enabled else str(product_dir)


def _version(path: Path, identifier: str, alpha: float | None = None):
    data = _read_json(path)
    metadata = data.get("_lens_correction", {})
    if not isinstance(metadata, dict):
        raise LensCorrectionError("标定版本导入信息格式无效。")
    if alpha is None:
        if not isinstance(metadata, dict) or metadata.get("schema") != VERSION_SCHEMA:
            raise LensCorrectionError("标定版本缺少导入信息，无法确认历史输出视野。")
        alpha = metadata.get("alpha")
    profile, payload = _profile(data, alpha)
    if _geometry_id(payload, alpha) != identifier:
        raise LensCorrectionError("标定版本与当前设置不符，禁止自动切换参数。")
    if metadata and (metadata.get("profile_id") != identifier or float(metadata.get("alpha", -1)) != float(alpha)):
        raise LensCorrectionError("标定版本导入信息与几何参数不一致。")
    return profile, payload, float(alpha)


def _cached(config):
    path = Path(config.profile_path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise LensCorrectionError(f"已选择的标定版本不存在：{path}") from exc
    key = (str(path.resolve()), config.profile_id, config.alpha, config.image_size,
           stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _MAP_LOCK:
        entry = _MAP_CACHE.get(key)
        if entry is None:
            profile, payload, _ = _version(path, config.profile_id, config.alpha)
            if tuple(profile.image_size) != config.image_size:
                raise LensCorrectionError("设置中的分辨率与标定版本不一致。")
            entry = (_make_rectifier(profile, config.alpha), payload)
            _MAP_CACHE[key] = entry
            while len(_MAP_CACHE) > _MAX_MAPS:
                _MAP_CACHE.popitem(last=False)
        _MAP_CACHE.move_to_end(key)
        return entry


def warmup(product_dir, role) -> None:
    config = get_config(product_dir, role)
    if config.enabled:
        _cached(config)


def _sidecar(path) -> dict | None:
    marker = Path(path).with_suffix(".undistortion.json")
    if not marker.exists():
        return None
    return _read_json(marker)


def is_managed_image(path) -> bool:
    """Distinguish this integration's assets from legacy, manually selected Demo images."""
    marker = _sidecar(path)
    return bool(marker and marker.get("schema") == SIDECAR_SCHEMA and marker.get("already_undistorted") is True)


def _known_serial(expected, observed):
    if expected and observed and str(expected) != str(observed):
        raise LensCorrectionError(f"相机序列号不符：参数绑定 {expected}，当前为 {observed}。")


def _check_marker(marker, config, role, camera_serial=""):
    if not marker or marker.get("already_undistorted") is not True:
        raise LensCorrectionError("该图缺少已矫正版本记录，不能作为当前矫正空间的参考图。")
    if not isinstance(marker.get("profile"), dict):
        raise LensCorrectionError("已矫正图片缺少完整标定参数，禁止再次矫正。")
    _, payload = _profile(marker["profile"], marker.get("alpha", 1.0))
    identifier = _geometry_id(payload, marker.get("alpha", 1.0))
    if identifier != config.profile_id or (marker.get("profile_id") and marker["profile_id"] != identifier):
        raise LensCorrectionError("图片已使用不同版本的标定参数矫正，请选择原图或匹配版本。")
    if marker.get("camera_role") and marker["camera_role"] != role:
        raise LensCorrectionError("已矫正图片属于另一台相机，不能混用。")
    _known_serial(config.camera_serial, marker.get("camera_serial", ""))
    _known_serial(camera_serial, marker.get("camera_serial", ""))


def _image_size(image, config):
    import numpy as np
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise LensCorrectionError("请输入灰度或彩色图像。")
    if (image.shape[1], image.shape[0]) != config.image_size:
        raise LensCorrectionError(f"图像分辨率与标定分辨率 {config.image_size} 不一致，不能缩放套用。")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise LensCorrectionError("图像数据类型不受支持。")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise LensCorrectionError("图像通道数不受支持。")


def _prepare(image, config, role, source_path="", camera_serial="") -> CorrectionResult:
    _known_serial(config.camera_serial, camera_serial)
    _image_size(image, config)
    marker = _sidecar(source_path) if source_path else None
    if marker and marker.get("already_undistorted"):
        _check_marker(marker, config, role, camera_serial)
        return CorrectionResult(image.copy(), 0.0, config.profile_id, True)
    rectifier, _ = _cached(config)
    started = time.perf_counter()
    try:
        corrected = rectifier.apply(image)
    except Exception as exc:
        raise LensCorrectionError(f"镜头矫正失败：{exc}") from exc
    return CorrectionResult(corrected, (time.perf_counter() - started) * 1000.0, config.profile_id, True)


def prepare_array(image, product_dir, role, *, source_path="", camera_serial="") -> CorrectionResult:
    role = _role(role)
    config = get_config(product_dir, role)
    if not config.enabled:
        return CorrectionResult(image, 0.0, "raw", False)
    return _prepare(image, config, role, source_path, camera_serial)


def _read_image(path):
    import cv2
    from algorithms.image_io import imread
    image = imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise LensCorrectionError(f"无法读取图像：{path}")
    return image


def validate_prepared_path(path, product_dir, role) -> None:
    """Validate a saved model's reference without silently changing its pixels."""
    role = _role(role)
    config = get_config(product_dir, role)
    if not config.enabled:
        marker = _sidecar(path)
        if marker and marker.get("schema") == SIDECAR_SCHEMA and marker.get("already_undistorted"):
            raise LensCorrectionError("当前为原图模式，不能混用已矫正参考图。")
        return
    image_path = Path(path)
    marker_path = image_path.with_suffix(".undistortion.json")
    try:
        stamps = []
        for item in (image_path, marker_path):
            stat = item.stat()
            stamps.append((str(item.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    except OSError as exc:
        raise LensCorrectionError("参考图或其矫正版本记录不存在。") from exc
    key = (tuple(stamps), role, config.profile_id, config.alpha, config.image_size, config.camera_serial)
    with _LOCK:
        if key in _VALIDATED_CACHE:
            _VALIDATED_CACHE.move_to_end(key)
            return
    marker = _sidecar(path)
    _check_marker(marker, config, role)
    _image_size(_read_image(path), config)
    with _LOCK:
        _VALIDATED_CACHE[key] = True
        while len(_VALIDATED_CACHE) > 128:
            _VALIDATED_CACHE.popitem(last=False)


def source_image_path(path) -> str:
    """Follow sidecar provenance to an existing raw image; never guess a name."""
    original, current = str(path), Path(path)
    seen = set()
    for _ in range(16):
        resolved = str(current.resolve()).casefold()
        if resolved in seen:
            raise LensCorrectionError("已矫正图片的原图记录存在循环。")
        seen.add(resolved)
        marker = _sidecar(current)
        if not marker or not marker.get("already_undistorted"):
            return str(current) if current.exists() else original
        source = marker.get("source_path") or marker.get("source_image")
        if not source or source == "synthetic_demo":
            return original
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = current.parent / candidate
        if not candidate.is_file():
            return original
        current = candidate
    raise LensCorrectionError("已矫正图片的原图记录层级过深。")


def _marker(payload, alpha, identifier, role, serial, source_path, **extra):
    return dict(schema=SIDECAR_SCHEMA, already_undistorted=True, profile_id=identifier,
                camera_role=role, camera_serial=serial, source_path=str(source_path),
                source_image=str(source_path), profile=payload, alpha=alpha, **extra)


def mark_corrected_path(path, product_dir, role, *, profile_id, camera_serial="", source_path="") -> None:
    """Mark an exported frame using its explicit historical version, not today's switch."""
    role = _role(role)
    if not _ID_RE.fullmatch(str(profile_id)):
        raise LensCorrectionError("导出帧缺少有效的矫正版本。")
    version = Path(product_dir) / "calibration" / "profiles" / f"{profile_id}.json"
    profile, payload, alpha = _version(version, profile_id)
    config = CorrectionConfig(profile_id=profile_id, image_size=tuple(profile.image_size))
    _image_size(_read_image(path), config)
    atomic_write_json(Path(path).with_suffix(".undistortion.json"),
                      _marker(payload, alpha, profile_id, role, camera_serial, source_path), create_backup=False)


def prepare_path(path, product_dir, role) -> str:
    """Create a version-isolated PNG; never copy annotations or alter the source."""
    role = _role(role)
    config = get_config(product_dir, role)
    if not config.enabled:
        return str(path)
    source = Path(path)
    marker = _sidecar(source)
    if marker and marker.get("already_undistorted"):
        _check_marker(marker, config, role)
        _image_size(_read_image(source), config)
        return str(path)
    try:
        digest = hashlib.sha256()
        digest.update(str(source.resolve()).encode("utf-8"))
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()
    except OSError as exc:
        raise LensCorrectionError(f"无法读取原图：{source}") from exc
    cache_directory = Path(product_dir) / ".cache" / "lens_images" / config.profile_id
    filename = f"{role}_{fingerprint[:24]}"
    # Include the longer sidecar extension and atomic writer's temporary-file
    # suffix in the Windows path budget. Full hashes remain in metadata.
    stem_budget = max(0, min(24, 245 - len(str(cache_directory.resolve())) - 1
                              - len(filename) - len(".undistortion.json") - 1))
    safe_stem = re.sub(r"[^\w.-]", "_", source.stem)[:stem_budget]
    if safe_stem:
        filename += f"_{safe_stem}"
    destination = cache_directory / f"{filename}.png"
    # Serialize the two-file cache publication within this process. A sidecar is
    # the commit marker; an interrupted PNG without one will be regenerated.
    with _LOCK:
        cached_marker = _sidecar(destination)
        if destination.exists() and cached_marker:
            _check_marker(cached_marker, config, role)
            if cached_marker.get("source_fingerprint") != fingerprint:
                raise LensCorrectionError("校正图片缓存与原图指纹不符。")
            _image_size(_read_image(destination), config)
            return str(destination)
        image = _read_image(source)
        result = _prepare(image, config, role)
        import cv2
        import numpy as np
        if result.image.dtype not in (np.uint8, np.uint16):
            raise LensCorrectionError("PNG 缓存只支持 uint8/uint16 原图；浮点图请使用数组入口，不能隐式转换。")
        ok, encoded = cv2.imencode(".png", result.image)
        if not ok:
            raise LensCorrectionError("无法编码矫正图片。")
        _, payload = _cached(config)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".lens-", suffix=".png", dir=str(destination.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        atomic_write_json(destination.with_suffix(".undistortion.json"),
                          _marker(payload, config.alpha, config.profile_id, role, config.camera_serial,
                                  str(source.resolve()), source_fingerprint=fingerprint,
                                  correction_ms=result.correction_ms), create_backup=False)
        return str(destination)


__all__ = ["CorrectionConfig", "CorrectionResult", "LensCorrectionError", "get_config", "save_config",
           "save_configs", "inspect_profile", "active_signature", "asset_root", "prepare_array", "prepare_path",
           "source_image_path", "validate_prepared_path", "mark_corrected_path", "is_managed_image", "warmup"]
