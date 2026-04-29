from __future__ import annotations

import ctypes
import subprocess
import sys

from ui.shell.main_window import main


_SINGLE_INSTANCE_MUTEX_NAME = "Local\\LCSystem.EmbeddingTest.SingleInstance"
_SINGLE_INSTANCE_MUTEX_HANDLE = None


def _is_windows_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _show_already_running_message() -> None:
    try:
        ctypes.windll.user32.MessageBoxW(
            None,
            "LC System 已经打开，请不要重复启动。",
            "LC System",
            0x30,
        )
    except Exception:
        pass


def _another_instance_exists() -> bool:
    if sys.platform != "win32":
        return False
    try:
        synchronize_access = 0x00100000
        handle = ctypes.windll.kernel32.OpenMutexW(
            synchronize_access,
            False,
            _SINGLE_INSTANCE_MUTEX_NAME,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        return False


def _acquire_single_instance_mutex() -> bool:
    global _SINGLE_INSTANCE_MUTEX_HANDLE
    if sys.platform != "win32":
        return True
    try:
        _SINGLE_INSTANCE_MUTEX_HANDLE = ctypes.windll.kernel32.CreateMutexW(
            None,
            False,
            _SINGLE_INSTANCE_MUTEX_NAME,
        )
        if not _SINGLE_INSTANCE_MUTEX_HANDLE:
            raise ctypes.WinError()
        already_exists = ctypes.windll.kernel32.GetLastError() == 183
        if already_exists:
            ctypes.windll.kernel32.CloseHandle(_SINGLE_INSTANCE_MUTEX_HANDLE)
            _SINGLE_INSTANCE_MUTEX_HANDLE = None
            return False
        return True
    except Exception:
        return True


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


if __name__ == "__main__":
    if _another_instance_exists():
        _show_already_running_message()
        sys.exit(0)
    if _relaunch_as_admin():
        sys.exit(0)
    if not _acquire_single_instance_mutex():
        _show_already_running_message()
        sys.exit(0)
    main()
