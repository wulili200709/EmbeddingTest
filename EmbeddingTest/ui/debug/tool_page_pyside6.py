"""Compatibility wrapper for the debug tool page."""

from ui.debug.tool_page import page as _impl
from ui.debug.tool_page.page import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
