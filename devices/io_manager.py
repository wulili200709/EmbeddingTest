from __future__ import annotations

from .io_controller import IoController


class IoManager(IoController):
    """Compatibility alias with a clearer business-facing name."""


__all__ = ["IoManager"]
