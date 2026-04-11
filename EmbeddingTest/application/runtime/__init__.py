"""Runtime application package with lazy exports."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "DEFAULT_LIGHT_STABLE_MS": ".capture_policy",
    "DEFAULT_RELEASE_PASSWORD": ".capture_policy",
    "RUNTIME_CAPTURE_POLICY_ALL": ".capture_policy",
    "RUNTIME_CAPTURE_POLICY_NG_ONLY": ".capture_policy",
    "delete_capture_artifacts": ".capture_policy",
    "normalize_capture_retention_policy": ".capture_policy",
    "retained_capture_paths_for_policy": ".capture_policy",
    "RuntimeController": ".controller",
}

__all__ = sorted(_SYMBOL_TO_MODULE.keys())


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
