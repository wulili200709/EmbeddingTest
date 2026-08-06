from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .app_logging import default_log_dir


_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 4  # current file + four backups = at most about 50 MiB
_DEFAULT_DETAIL_SECONDS = 30 * 60
_ERROR_MARKERS = (
    "error",
    "failed",
    "failure",
    "exception",
    "timeout",
    "precheck_failed",
    "rejected",
    "异常",
    "失败",
    "错误",
    "超时",
    "拒绝",
)


class RuntimeCameraLogService:
    """Small asynchronous rotating log dedicated to runtime camera diagnosis."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        backup_count: int = _DEFAULT_BACKUP_COUNT,
        queue_size: int = 2048,
    ) -> None:
        self.path = Path(path) if path is not None else default_log_dir() / "runtime_camera.log"
        self.max_bytes = max(1024, int(max_bytes))
        self.backup_count = max(0, int(backup_count))
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max(32, int(queue_size)))
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._handler: RotatingFileHandler | None = None
        self._disabled = False
        self._stopped = False
        self._detail_until = 0.0

    def set_detailed(self, enabled: bool, *, duration_seconds: int = _DEFAULT_DETAIL_SECONDS) -> None:
        with self._lock:
            self._detail_until = (
                time.monotonic() + max(1, int(duration_seconds))
                if enabled
                else 0.0
            )
        state = "enabled" if enabled else "disabled"
        self.record(
            f"[camera-diagnostic] detailed={state} duration_seconds={int(duration_seconds) if enabled else 0}",
            force=True,
        )

    def is_detailed(self) -> bool:
        with self._lock:
            enabled = self._detail_until > time.monotonic()
            if not enabled:
                self._detail_until = 0.0
            return enabled

    def record(self, message: object, *, force: bool = False) -> None:
        text = str(message or "").strip().replace("\r", " ").replace("\n", " ")
        if not text or (not force and not self._should_record(text)):
            return
        if not self._ensure_started():
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Runtime inspection must never wait for diagnostic disk I/O.
            return

    def _should_record(self, message: str) -> bool:
        if self.is_detailed() or message.startswith("[trigger-summary]"):
            return True
        lowered = message.casefold()
        return any(marker in lowered for marker in _ERROR_MARKERS)

    def _ensure_started(self) -> bool:
        with self._lock:
            if self._disabled or self._stopped:
                return False
            if self._thread is not None:
                return True
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handler = RotatingFileHandler(
                    self.path,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                    encoding="utf-8",
                )
                self._handler.setFormatter(
                    logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
                )
            except Exception:
                self._disabled = True
                self._handler = None
                return False
            self._thread = threading.Thread(
                target=self._run_writer,
                name="runtime-camera-log",
                daemon=True,
            )
            self._thread.start()
            return True

    def _run_writer(self) -> None:
        while True:
            message = self._queue.get()
            if message is None:
                self._queue.task_done()
                break
            try:
                handler = self._handler
                if handler is not None:
                    record = logging.LogRecord(
                        name="runtime_camera",
                        level=logging.INFO,
                        pathname="",
                        lineno=0,
                        msg=message,
                        args=(),
                        exc_info=None,
                    )
                    handler.emit(record)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def shutdown(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
        if thread is not None:
            while True:
                try:
                    self._queue.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        break
            thread.join(timeout=3.0)
            if thread.is_alive():
                return
        with self._lock:
            handler = self._handler
            self._handler = None
            self._thread = None
        if handler is not None:
            try:
                handler.close()
            except Exception:
                pass


_SERVICE = RuntimeCameraLogService()


def record_runtime_camera_message(message: object, *, force: bool = False) -> None:
    _SERVICE.record(message, force=force)


def set_detailed_camera_diagnostics(
    enabled: bool,
    *,
    duration_seconds: int = _DEFAULT_DETAIL_SECONDS,
) -> None:
    _SERVICE.set_detailed(enabled, duration_seconds=duration_seconds)


def detailed_camera_diagnostics_enabled() -> bool:
    return _SERVICE.is_detailed()


def runtime_camera_log_path() -> Path:
    return _SERVICE.path


def shutdown_runtime_camera_logging() -> None:
    _SERVICE.shutdown()


atexit.register(shutdown_runtime_camera_logging)


__all__ = [
    "RuntimeCameraLogService",
    "detailed_camera_diagnostics_enabled",
    "record_runtime_camera_message",
    "runtime_camera_log_path",
    "set_detailed_camera_diagnostics",
    "shutdown_runtime_camera_logging",
]
