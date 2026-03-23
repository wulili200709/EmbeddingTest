from __future__ import annotations

from algorithms.proxy import is_ready as is_qr_core_ready

from .support import AlgorithmEngineWarmupThread


def start_algorithm_engine_warmup(window) -> None:
    if is_qr_core_ready():
        window._set_algorithm_engine_status("算法引擎：已就绪")
        window._preload_current_embedding_model()
        return
    if window._engine_warmup_thread is not None and window._engine_warmup_thread.isRunning():
        return
    window._set_algorithm_engine_status("算法引擎：加载中...")
    worker = AlgorithmEngineWarmupThread(window)
    worker.warmupFinished.connect(window._on_algorithm_engine_warmup_finished)
    worker.finished.connect(window._on_algorithm_engine_warmup_thread_finished)
    window._engine_warmup_thread = worker
    worker.start()


def on_algorithm_engine_warmup_finished(window, success: bool, message: str) -> None:
    if success:
        window._set_algorithm_engine_status("算法引擎：已就绪")
        window._preload_current_embedding_model()
        return
    window._set_algorithm_engine_status("算法引擎：加载失败", tooltip=message)


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
    window.runtime_ctrl.refresh_all_status("已重新加载调试会话")
    window._sync_shell_status()
