"""Runtime capture retention helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

DEFAULT_RELEASE_PASSWORD = "1234"
DEFAULT_LIGHT_STABLE_MS = 20
RUNTIME_CAPTURE_POLICY_ALL = "all"
RUNTIME_CAPTURE_POLICY_NG_ONLY = "ng_only"


def normalize_capture_retention_policy(policy: object) -> str:
    return (
        RUNTIME_CAPTURE_POLICY_ALL
        if str(policy or "").strip().lower() == RUNTIME_CAPTURE_POLICY_ALL
        else RUNTIME_CAPTURE_POLICY_NG_ONLY
    )


def retained_capture_paths_for_policy(
    policy: object,
    final_result: object,
    capture_paths: Dict[str, str] | None,
) -> Dict[str, str]:
    normalized = normalize_capture_retention_policy(policy)
    sanitized = {
        str(role): str(path).strip()
        for role, path in dict(capture_paths or {}).items()
        if str(path or "").strip()
    }
    if normalized == RUNTIME_CAPTURE_POLICY_ALL:
        return sanitized
    if str(final_result or "").strip().upper() == "NG":
        return sanitized
    return {}


def delete_capture_artifacts(capture_paths: Dict[str, str] | None) -> None:
    for raw_path in dict(capture_paths or {}).values():
        image_text = str(raw_path or "").strip()
        if not image_text:
            continue
        image_path = Path(image_text)
        json_path = image_path.with_suffix(".json")
        for path in (image_path, json_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
