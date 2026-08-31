from __future__ import annotations

from common.algorithm_catalog import algorithm_display_key, default_algorithm_display_name
from ui.i18n import tr


def algorithm_display_name(code: object) -> str:
    fallback = default_algorithm_display_name(code)
    key = algorithm_display_key(code)
    return str(tr(key) or fallback) if key else fallback


__all__ = ["algorithm_display_name"]
