from __future__ import annotations

import os
from typing import List

from common.camera_roles import (
    DEFAULT_CAMERA_ROLE,
    camera_role_from_text,
    normalize_camera_role as _normalize_camera_role,
)


def normalize_camera_role(camera_id: object) -> str:
    return _normalize_camera_role(camera_id)


def camera_role_from_path(path: str) -> str:
    name = os.path.basename(str(path or "")).lower()
    return camera_role_from_text(name)


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
    return DEFAULT_CAMERA_ROLE


def filter_paths_for_camera(tool_page, paths: List[str], camera_id: object) -> List[str]:
    role = normalize_camera_role(camera_id)
    if not role:
        return list(paths)
    if not any(camera_role_from_path(path) for path in paths):
        return list(paths)
    return [path for path in paths if camera_role_from_path(path) == role]
