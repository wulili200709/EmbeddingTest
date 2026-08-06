from __future__ import annotations

import re
from typing import Iterable


CAMERA_ROLES: tuple[str, ...] = ("cam1", "cam2", "cam3")
DEFAULT_CAMERA_ROLE = "cam1"
ROLE_TO_CAMERA_INDEX: dict[str, int] = {
    role: index for index, role in enumerate(CAMERA_ROLES, start=1)
}
CAMERA_INDEX_TO_ROLE: dict[int, str] = {
    index: role for role, index in ROLE_TO_CAMERA_INDEX.items()
}

_CAMERA_ROLE_RE = re.compile(r"(?:^|[_-])(cam\d+)(?=[_.-]|$)", re.IGNORECASE)


def normalize_camera_role(camera_id: object, *, default: str = "") -> str:
    role = str(camera_id or "").strip().lower()
    if role in ROLE_TO_CAMERA_INDEX:
        return role
    return default


def camera_index_for_role(role: object, *, default: int = 0) -> int:
    return ROLE_TO_CAMERA_INDEX.get(normalize_camera_role(role), int(default))


def camera_role_for_index(index: object, *, default: str = "") -> str:
    try:
        return CAMERA_INDEX_TO_ROLE.get(int(index), default)
    except Exception:
        return default


def configured_camera_roles(roles: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for role in roles or ():
        role_text = normalize_camera_role(role)
        if role_text and role_text not in normalized:
            normalized.append(role_text)
    return normalized


def camera_role_from_text(text: object, *, default: str = "") -> str:
    match = _CAMERA_ROLE_RE.search(str(text or ""))
    if not match:
        return default
    return normalize_camera_role(match.group(1), default=default)


__all__ = [
    "CAMERA_INDEX_TO_ROLE",
    "CAMERA_ROLES",
    "DEFAULT_CAMERA_ROLE",
    "ROLE_TO_CAMERA_INDEX",
    "camera_index_for_role",
    "camera_role_for_index",
    "camera_role_from_text",
    "configured_camera_roles",
    "normalize_camera_role",
]
