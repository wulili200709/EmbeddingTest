from __future__ import annotations

import os
import re
from typing import List


_CAMERA_ROLE_RE = re.compile(r"(?:^|[_-])(cam[12])(?=[_.-]|$)", re.IGNORECASE)


def normalize_camera_role(camera_id: object) -> str:
    text = str(camera_id or "").strip().lower()
    if text in {"cam1", "cam2"}:
        return text
    return ""


def camera_role_from_path(path: str) -> str:
    name = os.path.basename(str(path or "")).lower()
    match = _CAMERA_ROLE_RE.search(name)
    if not match:
        return ""
    return normalize_camera_role(match.group(1))


def selected_image_list_camera_role(tool_page) -> str:
    getter = getattr(tool_page, "current_camera_role", None)
    if callable(getter):
        role = normalize_camera_role(getter())
        if role:
            return role
    role_getter = getattr(tool_page, "_selected_debug_camera_role", None)
    if callable(role_getter):
        role = normalize_camera_role(role_getter())
        if role:
            return role
    return "cam1"


def filter_paths_for_camera(tool_page, paths: List[str], camera_id: object) -> List[str]:
    role = normalize_camera_role(camera_id)
    if not role:
        return list(paths)
    if not any(camera_role_from_path(path) for path in paths):
        return list(paths)
    return [path for path in paths if camera_role_from_path(path) == role]
