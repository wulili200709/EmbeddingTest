"""Debug UI package with lazy symbol exports."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "EmbeddingAnalysisDialog": ".embedding_analysis_dialog",
    "Line2DupTemplateDialog": ".line2dup_template_page_pyside6",
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
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
