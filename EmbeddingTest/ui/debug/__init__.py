"""Debug UI package with lazy symbol exports."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "AnomalyHeatmapDialog": ".anomaly_heatmap_dialog",
    "EmbeddingAnalysisDialog": ".embedding_analysis_dialog",
    "Line2DupTemplateDialog": "line2dup.ui.template_page_pyside6",
    "OverlayShape": ".roi_canvas_pyside6",
    "RoiCanvas": ".roi_canvas_pyside6",
    "pixmap_from_path": ".roi_canvas_pyside6",
    "ToolPage": ".tool_page_pyside6",
}

__all__ = sorted(_SYMBOL_TO_MODULE.keys())


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if module_name.startswith("."):
        module = import_module(module_name, __name__)
    else:
        module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
