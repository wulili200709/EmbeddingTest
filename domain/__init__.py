"""Domain layer package with lazy symbol exports."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "InspectionItem": ".inspection_items",
    "SUPPORTED_CAMERA_IDS": ".inspection_items",
    "build_default_item": ".inspection_items",
    "load_inspection_items": ".inspection_items",
    "save_inspection_items": ".inspection_items",
    "sync_items_with_labels": ".inspection_items",
    "CameraRuntimeResult": ".inspection_models",
    "InspectionItemResult": ".inspection_models",
    "RuntimeInspectionResult": ".inspection_models",
    "build_task_id": ".inspection_models",
    "aggregate_runtime_outcome": ".result_aggregator",
    "build_pending_result": ".result_aggregator",
    "recipe_name_from_path": ".result_aggregator",
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
