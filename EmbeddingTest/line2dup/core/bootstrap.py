from __future__ import annotations

import sys

from app_paths import packaged_embedding_test_root, packaged_repo_root

EMBEDDING_TEST_DIR = packaged_embedding_test_root(__file__)
REPO_ROOT = packaged_repo_root(__file__)


def ensure_repo_root_on_path() -> Path:
    repo = REPO_ROOT
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


__all__ = ["EMBEDDING_TEST_DIR", "REPO_ROOT", "ensure_repo_root_on_path"]
