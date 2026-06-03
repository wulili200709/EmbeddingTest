from __future__ import annotations

import sys
from pathlib import Path


def _source_embedding_test_root(anchor: str | Path | None = None) -> Path:
    current = Path(anchor if anchor is not None else __file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "EmbeddingTest":
            return parent
    return current.parent


def is_frozen_runtime() -> bool:
    return bool(getattr(sys, "frozen", False))


def distribution_root() -> Path:
    if is_frozen_runtime():
        return Path(sys.executable).resolve().parent
    return _source_embedding_test_root(__file__).parent


def packaged_embedding_test_root(anchor: str | Path | None = None) -> Path:
    if not is_frozen_runtime():
        return _source_embedding_test_root(anchor)

    exe_root = distribution_root()
    candidates = [exe_root / "EmbeddingTest"]

    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(Path(meipass) / "EmbeddingTest")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def writable_embedding_test_root(anchor: str | Path | None = None) -> Path:
    if not is_frozen_runtime():
        return _source_embedding_test_root(anchor)

    exe_root = distribution_root()
    candidate = exe_root / "EmbeddingTest"
    if candidate.exists():
        return candidate
    return exe_root


def packaged_repo_root(anchor: str | Path | None = None) -> Path:
    root = packaged_embedding_test_root(anchor)
    return root.parent if root.name == "EmbeddingTest" else root
