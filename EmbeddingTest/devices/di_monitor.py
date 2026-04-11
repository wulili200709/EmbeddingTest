from __future__ import annotations

from .di_poller import DiEvent, DiPoller


class DiMonitor(DiPoller):
    """Compatibility alias for the production DI monitor name."""


__all__ = ["DiEvent", "DiMonitor"]
