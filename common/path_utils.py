from __future__ import annotations

import os
from typing import Iterable, List, Optional


def _clean_path_text(path: object) -> str:
    return str(path or "").strip()


def _split_path_parts(path: str) -> List[str]:
    text = _clean_path_text(path).replace("\\", "/")
    return [part for part in text.split("/") if part and part != "."]


def product_relative_path(path: object, *, base_dir: str) -> str:
    text = _clean_path_text(path)
    if not text:
        return ""
    normalized = os.path.normpath(text)
    if not os.path.isabs(normalized):
        return normalized.replace("\\", "/")
    try:
        rel = os.path.relpath(normalized, start=base_dir)
    except ValueError:
        return normalized
    return os.path.normpath(rel).replace("\\", "/")


def rebase_path_under_anchor(path: object, *, anchor_dir: str) -> str:
    text = _clean_path_text(path)
    anchor = _clean_path_text(anchor_dir)
    if not text or not anchor:
        return ""
    raw_parts = _split_path_parts(text)
    anchor_parts = _split_path_parts(anchor)
    if not raw_parts or not anchor_parts:
        return ""
    anchor_name = anchor_parts[-1].casefold()
    for idx in range(len(raw_parts) - 1, -1, -1):
        if raw_parts[idx].casefold() != anchor_name:
            continue
        suffix = raw_parts[idx + 1 :]
        return os.path.normpath(os.path.join(anchor, *suffix))
    return ""


def resolve_product_path(
    path: object,
    *,
    base_dir: str,
    anchor_dir: str = "",
    prefer_existing: bool = True,
) -> str:
    text = _clean_path_text(path)
    if not text:
        return ""
    candidate = os.path.normpath(text if os.path.isabs(text) else os.path.join(base_dir, text))
    if not prefer_existing or os.path.exists(candidate):
        return candidate
    rebased = rebase_path_under_anchor(text, anchor_dir=anchor_dir)
    if rebased and os.path.exists(rebased):
        return rebased
    return candidate


def resolve_existing_product_paths(
    paths: Iterable[object],
    *,
    base_dir: str,
    anchor_dir: str = "",
) -> List[str]:
    resolved: List[str] = []
    for item in paths:
        path = resolve_product_path(item, base_dir=base_dir, anchor_dir=anchor_dir, prefer_existing=True)
        if path and os.path.exists(path):
            resolved.append(path)
    return resolved


def resolve_existing_product_path(
    path: object,
    *,
    base_dir: str,
    anchor_dir: str = "",
) -> Optional[str]:
    resolved = resolve_product_path(path, base_dir=base_dir, anchor_dir=anchor_dir, prefer_existing=True)
    if resolved and os.path.exists(resolved):
        return resolved
    return None
