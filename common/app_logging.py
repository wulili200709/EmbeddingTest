from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def default_log_dir() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent.parent
    return root / "logs"


def configure_app_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir = default_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_dir / "LC_System_app.log", encoding="utf-8")
    except Exception:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("lc_system")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_app_logger(name: str) -> logging.Logger:
    configure_app_logging()
    return logging.getLogger(f"lc_system.{name}")
