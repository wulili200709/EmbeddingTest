from __future__ import annotations

from ui.i18n_resources import merge_resources
from .access import TRANSLATIONS as ACCESS_TRANSLATIONS
from .debug import TRANSLATIONS as DEBUG_TRANSLATIONS
from .ncc import TRANSLATIONS as NCC_TRANSLATIONS
from .runtime import TRANSLATIONS as RUNTIME_TRANSLATIONS
from .shell import TRANSLATIONS as SHELL_TRANSLATIONS
from .template import TRANSLATIONS as TEMPLATE_TRANSLATIONS


TRANSLATIONS = merge_resources(
    ACCESS_TRANSLATIONS,
    DEBUG_TRANSLATIONS,
    NCC_TRANSLATIONS,
    RUNTIME_TRANSLATIONS,
    SHELL_TRANSLATIONS,
    TEMPLATE_TRANSLATIONS,
)

__all__ = ["TRANSLATIONS"]
