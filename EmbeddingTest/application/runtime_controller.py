"""Compatibility wrapper for the runtime controller."""

from application.runtime import controller as _impl
from application.runtime.controller import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
