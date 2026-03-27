from __future__ import annotations

import ctypes
import subprocess
import sys

from ui.shell.main_window import main


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


if __name__ == "__main__":
    if _relaunch_as_admin():
        sys.exit(0)
    main()
