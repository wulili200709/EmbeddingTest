from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


BACKUP_DIR_NAME = ".json_backups"
MAX_BACKUPS_PER_FILE = 10
PRODUCT_JSON_TYPES: dict[str, type] = {
    "session.json": dict,
    "product_params.json": dict,
    "inspection_items.json": list,
    "sample_annotations.json": dict,
}


def _positive_int(value: object) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _validate_payload(path: Path, payload: Any) -> None:
    if path.name.lower() != "session.json":
        return
    if not isinstance(payload, dict):
        raise ValueError("session.json root must be an object")
    cam1_serial = str(payload.get("runtime_cam1_serial", "") or "").strip()
    cam2_serial = str(payload.get("runtime_cam2_serial", "") or "").strip()
    if not cam1_serial and not cam2_serial:
        raise ValueError("session.json has no runtime camera serial")
    if not _positive_int(payload.get("foot_trigger_delay_ms")):
        raise ValueError("session.json foot_trigger_delay_ms must be greater than 0")
    if not _positive_int(payload.get("ng_stop_delay_ms")):
        raise ValueError("session.json ng_stop_delay_ms must be greater than 0")


def _decode_json(path: Path, raw: bytes, expected_type: type | None) -> Any:
    payload = json.loads(raw.decode("utf-8-sig"))
    if expected_type is not None and not isinstance(payload, expected_type):
        raise ValueError(
            f"Expected JSON root type {expected_type.__name__}, got {type(payload).__name__}"
        )
    return payload


def _backup_dir(path: Path) -> Path:
    return path.parent / BACKUP_DIR_NAME


def _backup_candidates(path: Path) -> list[Path]:
    backup_dir = _backup_dir(path)
    if not backup_dir.is_dir():
        return []
    return sorted(
        backup_dir.glob(f"{path.name}.*.bak"),
        key=lambda item: item.name,
        reverse=True,
    )


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temp_path = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


def _record_recovery(path: Path, backup_path: Path, error: Exception) -> None:
    try:
        backup_dir = _backup_dir(path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        message = (
            f"[{timestamp}] restored {path.name} from {backup_path.name}; "
            f"primary error: {type(error).__name__}: {error}\n"
        )
        with (backup_dir / "recovery.log").open("a", encoding="utf-8") as handle:
            handle.write(message)
    except Exception:
        pass


def _prune_backups(path: Path, max_backups: int) -> None:
    for stale_path in _backup_candidates(path)[max(1, int(max_backups)) :]:
        try:
            stale_path.unlink()
        except OSError:
            pass


def backup_valid_json(
    path: str | os.PathLike[str],
    *,
    expected_type: type | None = None,
    max_backups: int = MAX_BACKUPS_PER_FILE,
) -> Path | None:
    json_path = Path(path)
    if not json_path.is_file():
        return None

    raw = json_path.read_bytes()
    payload = _decode_json(json_path, raw, expected_type)
    _validate_payload(json_path, payload)

    candidates = _backup_candidates(json_path)
    if candidates:
        try:
            if candidates[0].read_bytes() == raw:
                return candidates[0]
        except OSError:
            pass

    backup_dir = _backup_dir(json_path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = backup_dir / f"{json_path.name}.{stamp}-{uuid4().hex[:8]}.bak"
    _atomic_write_bytes(backup_path, raw)
    _prune_backups(json_path, max_backups)
    return backup_path


def load_json_with_recovery(
    path: str | os.PathLike[str],
    *,
    expected_type: type | None = None,
    max_backups: int = MAX_BACKUPS_PER_FILE,
) -> Any:
    json_path = Path(path)
    raw = json_path.read_bytes()
    try:
        payload = _decode_json(json_path, raw, expected_type)
    except Exception as primary_error:
        for backup_path in _backup_candidates(json_path):
            try:
                backup_raw = backup_path.read_bytes()
                payload = _decode_json(json_path, backup_raw, expected_type)
                _validate_payload(json_path, payload)
            except Exception:
                continue
            _atomic_write_bytes(json_path, backup_raw)
            _record_recovery(json_path, backup_path, primary_error)
            return payload
        raise

    try:
        _validate_payload(json_path, payload)
    except ValueError as primary_error:
        for backup_path in _backup_candidates(json_path):
            try:
                backup_raw = backup_path.read_bytes()
                backup_payload = _decode_json(json_path, backup_raw, expected_type)
                _validate_payload(json_path, backup_payload)
            except Exception:
                continue
            _atomic_write_bytes(json_path, backup_raw)
            _record_recovery(json_path, backup_path, primary_error)
            return backup_payload
        return payload

    backup_valid_json(
        json_path,
        expected_type=expected_type,
        max_backups=max_backups,
    )
    return payload


def write_json_with_backup(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    expected_type: type | None = None,
    max_backups: int = MAX_BACKUPS_PER_FILE,
) -> None:
    if expected_type is not None and not isinstance(payload, expected_type):
        raise ValueError(
            f"Expected JSON root type {expected_type.__name__}, got {type(payload).__name__}"
        )

    json_path = Path(path)
    if json_path.is_file():
        try:
            backup_valid_json(
                json_path,
                expected_type=expected_type,
                max_backups=max_backups,
            )
        except Exception:
            pass

    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if expected_type is not None:
        decoded_payload = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded_payload, expected_type):
            raise ValueError(
                f"Expected JSON root type {expected_type.__name__}, "
                f"got {type(decoded_payload).__name__}"
            )
    _atomic_write_bytes(json_path, raw)
    try:
        backup_valid_json(
            json_path,
            expected_type=expected_type,
            max_backups=max_backups,
        )
    except ValueError:
        pass


def _remove_invalid_backups(path: Path, expected_type: type | None) -> None:
    for backup_path in _backup_candidates(path):
        try:
            payload = _decode_json(path, backup_path.read_bytes(), expected_type)
            _validate_payload(path, payload)
        except Exception:
            try:
                backup_path.unlink()
            except OSError:
                pass


def protect_product_json_files(session_dir: str | os.PathLike[str]) -> None:
    root = Path(session_dir)
    if not root.is_dir():
        return
    for product_dir in root.iterdir():
        if not product_dir.is_dir():
            continue
        for file_name, expected_type in PRODUCT_JSON_TYPES.items():
            json_path = product_dir / file_name
            if not json_path.is_file():
                continue
            if file_name == "session.json":
                _remove_invalid_backups(json_path, expected_type)
            try:
                load_json_with_recovery(json_path, expected_type=expected_type)
            except Exception:
                continue


__all__ = [
    "BACKUP_DIR_NAME",
    "MAX_BACKUPS_PER_FILE",
    "PRODUCT_JSON_TYPES",
    "backup_valid_json",
    "load_json_with_recovery",
    "protect_product_json_files",
    "write_json_with_backup",
]
