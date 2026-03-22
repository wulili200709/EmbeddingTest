from __future__ import annotations

from importlib import import_module
from threading import RLock
from types import ModuleType


_MODULE: ModuleType | None = None
_LAST_ERROR: Exception | None = None
_LOCK = RLock()


def load_qr_core() -> ModuleType:
    global _MODULE, _LAST_ERROR
    if _MODULE is not None:
        return _MODULE
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        try:
            _MODULE = import_module("qr_core")
            _LAST_ERROR = None
        except Exception as exc:
            _LAST_ERROR = exc
            raise
        return _MODULE


def preload() -> None:
    load_qr_core()


def is_ready() -> bool:
    return _MODULE is not None


def last_error() -> Exception | None:
    return _LAST_ERROR


def __getattr__(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(load_qr_core(), name)
