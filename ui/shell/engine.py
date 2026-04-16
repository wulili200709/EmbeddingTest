from __future__ import annotations

from algorithms.proxy import is_ready as is_qr_core_ready
from ui.i18n import tr

from .support import AlgorithmEngineWarmupThread


def start_algorithm_engine_warmup(window) -> None:
    if is_qr_core_ready():
        window._set_algorithm_engine_status_key("status.engine_ready")
        window._preload_current_embedding_model()
        return
    if window._engine_warmup_thread is not None and window._engine_warmup_thread.isRunning():
        return
    window._set_algorithm_engine_status_key("status.engine_loading")
    worker = AlgorithmEngineWarmupThread()
    worker.warmupFinished.connect(window._on_algorithm_engine_warmup_finished)
    worker.finished.connect(window._on_algorithm_engine_warmup_thread_finished)
    worker.finished.connect(worker.deleteLater)
    window._engine_warmup_thread = worker
    worker.start()


def on_algorithm_engine_warmup_finished(window, success: bool, message: str) -> None:
    if success:
        window._set_algorithm_engine_status_key("status.engine_ready")
        window._preload_current_embedding_model()
        return
    window._set_algorithm_engine_status_key("status.engine_failed", tooltip=message)


def on_algorithm_engine_warmup_thread_finished(window) -> None:
    window._engine_warmup_thread = None


def preload_current_embedding_model(window) -> None:
    algorithm = window.tool_page.current_algorithm()
    if not window.algo.is_embedding_algorithm(algorithm):
        return
    try:
        window.tool_page.load_embedding_model(algorithm)
    except Exception:
        window.algo.model = None


def reload_debug_session(window) -> None:
    window.tool_page.load_session()
    if is_qr_core_ready():
        window._preload_current_embedding_model()
    window.runtime_ctrl.refresh_all_status(tr("action.reload_debug"))
    window._sync_shell_status()
