from __future__ import annotations

import contextlib
import ctypes
import os
from pathlib import Path
import sys
from typing import Iterator

from common.app_paths import packaged_embedding_test_root


_MVS_EXECUTABLE_ENV = "HIK_MVS_EXECUTABLE"
_PATH_LIST_ENV_NAMES = (
    "PATH",
    "GENICAM_GENTL64_PATH",
    "GENICAM_GENTL32_PATH",
    "QT_PLUGIN_PATH",
    "QML2_IMPORT_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
)


def find_mvs_executable() -> Path | None:
    """Return the installed or bundled Hikrobot MVS executable, when available."""
    configured_path = str(os.environ.get(_MVS_EXECUTABLE_ENV, "")).strip()
    root = packaged_embedding_test_root(__file__)
    candidates = [
        Path(configured_path) if configured_path else None,
        Path(r"C:\Program Files (x86)\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files (x86)\MVS\Applications\Win32\MVS.exe"),
        Path(r"C:\Program Files\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files\MVS\Applications\Win32\MVS.exe"),
        Path(r"C:\Program Files (x86)\HIKROBOT\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files (x86)\HIKROBOT\MVS\Applications\Win32\MVS.exe"),
        Path(r"C:\Program Files\HIKROBOT\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files\HIKROBOT\MVS\Applications\Win32\MVS.exe"),
        root / "third_party" / "MVS" / "Applications" / "Win64" / "MVS.exe",
        root / "third_party" / "MVS" / "Applications" / "Win32" / "MVS.exe",
        root / "third_party" / "MVS" / "Applications" / "MVS.exe",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _frozen_bundle_roots() -> list[str]:
    if not getattr(sys, "frozen", False):
        return []
    values = [getattr(sys, "_MEIPASS", ""), Path(sys.executable).resolve().parent]
    roots: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = os.path.normcase(os.path.abspath(text))
        if normalized not in roots:
            roots.append(normalized)
    return roots


def _is_bundled_path(value: str, roots: list[str]) -> bool:
    text = str(value or "").strip().strip('"')
    if not text or not os.path.isabs(text):
        return False
    candidate = os.path.normcase(os.path.abspath(text))
    for root in roots:
        try:
            if os.path.commonpath([candidate, root]) == root:
                return True
        except ValueError:
            continue
    return False


def mvs_launch_environment() -> dict[str, str]:
    """Return an environment without PyInstaller/PySide bundle paths."""
    environment = dict(os.environ)
    roots = _frozen_bundle_roots()
    if not roots:
        return environment

    for name in _PATH_LIST_ENV_NAMES:
        raw_value = str(environment.get(name, "") or "")
        if not raw_value:
            continue
        entries = [
            entry
            for entry in raw_value.split(os.pathsep)
            if entry.strip() and not _is_bundled_path(entry, roots)
        ]
        if entries:
            environment[name] = os.pathsep.join(entries)
        else:
            environment.pop(name, None)
    return environment


def _windows_dll_directory() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        get_dll_directory = ctypes.windll.kernel32.GetDllDirectoryW
        get_dll_directory.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p]
        get_dll_directory.restype = ctypes.c_uint32
        required = int(get_dll_directory(0, None))
        if required <= 0:
            return None
        buffer = ctypes.create_unicode_buffer(required)
        copied = int(get_dll_directory(required, buffer))
        return buffer.value if copied > 0 else None
    except Exception:
        return None


def _set_windows_dll_directory(path: str | None) -> bool:
    if sys.platform != "win32":
        return False
    try:
        set_dll_directory = ctypes.windll.kernel32.SetDllDirectoryW
        set_dll_directory.argtypes = [ctypes.c_wchar_p]
        set_dll_directory.restype = ctypes.c_int
        return bool(set_dll_directory(path))
    except Exception:
        return False


@contextlib.contextmanager
def isolated_mvs_dll_search() -> Iterator[None]:
    """Temporarily restore Windows' default DLL search for MVS startup."""
    if sys.platform != "win32":
        yield
        return

    previous_directory = _windows_dll_directory()
    changed = _set_windows_dll_directory(None)
    try:
        yield
    finally:
        if changed:
            _set_windows_dll_directory(previous_directory)


__all__ = [
    "find_mvs_executable",
    "isolated_mvs_dll_search",
    "mvs_launch_environment",
]
