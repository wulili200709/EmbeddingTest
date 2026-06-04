from __future__ import annotations

import ctypes
import atexit
import faulthandler
import tempfile
from datetime import datetime
from pathlib import Path
import subprocess
import sys

try:
    import msvcrt
except Exception:
    msvcrt = None  # type: ignore[assignment]


_SINGLE_INSTANCE_LOCK = None
_STANDARD_STREAM_LOG = None
_FAULT_HANDLER_LOG = None


def _log_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _write_log_header(log_file, label: str) -> None:
    mode = "exe" if getattr(sys, "frozen", False) else "source"
    log_file.write(f"\n--- LC System {label}: {_log_timestamp()} ---\n")
    log_file.write(f"mode: {mode}\n")
    log_file.write(f"executable: {sys.executable}\n")
    log_file.write(f"cwd: {Path.cwd()}\n")


def _open_standard_stream_log():
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.extend([Path(__file__).resolve().parent, Path(tempfile.gettempdir())])

    for root in roots:
        try:
            log_dir = root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = (log_dir / "LC_System_runtime.log").open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            _write_log_header(log_file, "start")
            return log_file
        except OSError:
            continue
    return None


def _close_standard_stream_log() -> None:
    global _STANDARD_STREAM_LOG
    log_file = _STANDARD_STREAM_LOG
    _STANDARD_STREAM_LOG = None
    if log_file is None:
        return
    try:
        log_file.close()
    except Exception:
        pass


def _close_fault_handler_log() -> None:
    global _FAULT_HANDLER_LOG
    log_file = _FAULT_HANDLER_LOG
    _FAULT_HANDLER_LOG = None
    if log_file is None:
        return
    try:
        faulthandler.disable()
    except Exception:
        pass
    try:
        log_file.close()
    except Exception:
        pass


def _enable_fault_handler_log() -> None:
    global _FAULT_HANDLER_LOG
    if _FAULT_HANDLER_LOG is not None:
        return
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.extend([Path(__file__).resolve().parent, Path(tempfile.gettempdir())])

    for root in roots:
        try:
            log_dir = root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = (log_dir / "LC_System_crash.log").open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            _write_log_header(log_file, "fault handler start")
            faulthandler.enable(file=log_file, all_threads=True)
            _FAULT_HANDLER_LOG = log_file
            atexit.register(_close_fault_handler_log)
            return
        except Exception:
            continue


def _ensure_standard_streams() -> None:
    global _STANDARD_STREAM_LOG
    stdout_ready = callable(getattr(getattr(sys, "stdout", None), "write", None))
    stderr_ready = callable(getattr(getattr(sys, "stderr", None), "write", None))
    if stdout_ready and stderr_ready:
        return

    if _STANDARD_STREAM_LOG is None:
        _STANDARD_STREAM_LOG = _open_standard_stream_log()
        if _STANDARD_STREAM_LOG is not None:
            atexit.register(_close_standard_stream_log)

    if _STANDARD_STREAM_LOG is None:
        return
    if not stdout_ready:
        sys.stdout = _STANDARD_STREAM_LOG
    if not stderr_ready:
        sys.stderr = _STANDARD_STREAM_LOG


def _is_windows_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _quote_windows_arg(value: str) -> str:
    return subprocess.list2cmdline([str(value)])


def _relaunch_as_admin() -> bool:
    if sys.platform != "win32" or _is_windows_admin():
        return False

    if getattr(sys, "frozen", False):
        executable = sys.executable
        parameters = " ".join(_quote_windows_arg(arg) for arg in sys.argv[1:])
    else:
        executable = sys.executable
        parameters = " ".join(_quote_windows_arg(arg) for arg in [__file__, *sys.argv[1:]])

    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        executable,
        parameters,
        None,
        1,
    )
    if result <= 32:
        raise RuntimeError(f"Failed to request administrator privileges: ShellExecuteW={result}")
    return True


class _SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None

    def acquire(self) -> bool:
        if msvcrt is None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a+b")
            self._file.seek(0)
            if not self._file.read(1):
                self._file.write(b"0")
                self._file.flush()
            self._file.seek(0)
            msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
        except (OSError, PermissionError):
            self.close()
            return False
        return True

    def close(self) -> None:
        file_obj = self._file
        self._file = None
        if file_obj is None:
            return
        try:
            if msvcrt is not None:
                file_obj.seek(0)
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        try:
            file_obj.close()
        except Exception:
            pass


def _single_instance_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "LC_System_single_instance.lock"


def _show_already_running_message() -> None:
    message = "程序已经开启，请检查"
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "LC System", 0x40)
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def _acquire_single_instance_lock() -> _SingleInstanceLock | None:
    lock = _SingleInstanceLock(_single_instance_lock_path())
    try:
        acquired = lock.acquire()
    except (OSError, PermissionError):
        lock.close()
        acquired = False
    if not acquired:
        _show_already_running_message()
        return None
    atexit.register(lock.close)
    return lock


if __name__ == "__main__":
    _ensure_standard_streams()
    _enable_fault_handler_log()
    _SINGLE_INSTANCE_LOCK = _acquire_single_instance_lock()
    if _SINGLE_INSTANCE_LOCK is None:
        sys.exit(0)
    if _relaunch_as_admin():
        _SINGLE_INSTANCE_LOCK.close()
        sys.exit(0)
    from ui.shell.main_window import main

    main()
