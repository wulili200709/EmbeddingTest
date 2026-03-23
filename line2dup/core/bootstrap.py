from __future__ import annotations

import sys
from pathlib import Path


EMBEDDING_TEST_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = EMBEDDING_TEST_DIR.parent


def ensure_repo_root_on_path() -> Path:
    repo = REPO_ROOT
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


__all__ = ["EMBEDDING_TEST_DIR", "REPO_ROOT", "ensure_repo_root_on_path"]
