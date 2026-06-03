from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def backup_path_for(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".bak")


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    update_backup: bool = False,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if update_backup and p.exists():
        try:
            shutil.copy2(p, backup_path_for(p))
        except Exception:
            pass

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{p.name}.",
        suffix=".tmp",
        dir=str(p.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, p)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    encoding: str = "utf-8",
) -> None:
    p = Path(path)
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    update_backup = p.exists()
    atomic_write_text(p, text, encoding=encoding, update_backup=update_backup)


def load_json_with_backup(
    path: str | Path,
    *,
    default: Any = None,
    encoding: str = "utf-8",
) -> Any:
    p = Path(path)
    backup = backup_path_for(p)

    for candidate, should_restore in ((p, False), (backup, True)):
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding=encoding)
            payload = json.loads(text)
        except Exception:
            continue
        if should_restore:
            try:
                atomic_write_text(p, text, encoding=encoding, update_backup=False)
            except Exception:
                pass
        return payload

    return default
