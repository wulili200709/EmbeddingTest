"""Application layer package with lazy symbol exports."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "AlgorithmController": ".algorithm_controller",
    "SUPPORTED_ALGORITHMS": ".algorithm_controller",
    "SUPPORTED_EMBEDDING_ALGORITHMS": ".algorithm_controller",
    "SUPPORTED_SCORE_MODES": ".algorithm_controller",
    "TrainResult": ".algorithm_controller",
    "InspectionExecutionRequest": ".inspection_executor",
    "InspectionExecutionResponse": ".inspection_executor",
    "InspectionExecutor": ".inspection_executor",
    "ProductRuntimeContext": ".runtime_context",
    "RuntimeContextProtocol": ".runtime_context",
    "RuntimePredictorProtocol": ".runtime_context",
    "ToolPageRuntimeContext": ".runtime_context",
    "ProductPaths": ".product_session",
    "ProductSession": ".product_session",
    "SessionData": ".product_session",
    "DEFAULT_LIGHT_STABLE_MS": ".runtime.controller",
    "DEFAULT_RELEASE_PASSWORD": ".runtime.controller",
    "RuntimeController": ".runtime.controller",
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
