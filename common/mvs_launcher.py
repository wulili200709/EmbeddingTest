from __future__ import annotations

import os
from pathlib import Path

from common.app_paths import packaged_embedding_test_root


_MVS_EXECUTABLE_ENV = "HIK_MVS_EXECUTABLE"
# MVS must not inherit SDK search paths injected later by services.camera.
# This module is imported before the runtime camera service, so keep the
# original desktop-like process environment for launching the external tool.
_MVS_LAUNCH_ENVIRONMENT = dict(os.environ)


def find_mvs_executable() -> Path | None:
    """Return the installed or bundled Hikrobot MVS executable, when available."""
    configured_path = str(os.environ.get(_MVS_EXECUTABLE_ENV, "")).strip()
    root = packaged_embedding_test_root(__file__)
    candidates = [
        Path(configured_path) if configured_path else None,
        root / "third_party" / "MVS" / "Applications" / "Win64" / "MVS.exe",
        root / "third_party" / "MVS" / "Applications" / "Win32" / "MVS.exe",
        root / "third_party" / "MVS" / "Applications" / "MVS.exe",
        Path(r"C:\Program Files (x86)\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files (x86)\MVS\Applications\Win32\MVS.exe"),
        Path(r"C:\Program Files\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files\MVS\Applications\Win32\MVS.exe"),
        Path(r"C:\Program Files (x86)\HIKROBOT\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files (x86)\HIKROBOT\MVS\Applications\Win32\MVS.exe"),
        Path(r"C:\Program Files\HIKROBOT\MVS\Applications\Win64\MVS.exe"),
        Path(r"C:\Program Files\HIKROBOT\MVS\Applications\Win32\MVS.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def mvs_launch_environment() -> dict[str, str]:
    """Return an isolated environment matching the one used at app startup."""
    return dict(_MVS_LAUNCH_ENVIRONMENT)


__all__ = ["find_mvs_executable", "mvs_launch_environment"]
