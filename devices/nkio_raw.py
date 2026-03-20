from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_int, c_ubyte, c_ushort
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_candidate_dll_paths(dll_name: str) -> Iterable[Path]:
    repo_root = _repo_root()
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "Lib" / "x64" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "Release" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "Debug" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Bin" / dll_name
    yield Path(dll_name)


def find_default_nkio_dll_path() -> Path:
    for path in _iter_candidate_dll_paths("NKIOLIBx64.dll"):
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in _iter_candidate_dll_paths("NKIOLIBx64.dll"))
    raise FileNotFoundError(f"Could not locate NKIOLIBx64.dll. Searched:\n{searched}")


class NkioRawLib:
    """Thin ctypes wrapper around NKIOLIBx64.dll."""

    def __init__(self, dll_path: str | Path | None = None) -> None:
        resolved_path = Path(dll_path) if dll_path is not None else find_default_nkio_dll_path()
        if not resolved_path.exists():
            raise FileNotFoundError(resolved_path)
        self.dll_path = resolved_path
        self._dll = ctypes.CDLL(str(self.dll_path))
        self._bind()

    def _bind(self) -> None:
        self._dll.NKDIO_LibraryInit.argtypes = [c_char_p]
        self._dll.NKDIO_LibraryInit.restype = c_int

        self._dll.NKDIO_LibraryDeinit.argtypes = []
        self._dll.NKDIO_LibraryDeinit.restype = None

        self._dll.NKDIO_PollingReadDiByte.argtypes = [c_ubyte, POINTER(c_ubyte)]
        self._dll.NKDIO_PollingReadDiByte.restype = c_int

        self._dll.NKDIO_PollingReadDiWord.argtypes = [c_ubyte, POINTER(c_ushort)]
        self._dll.NKDIO_PollingReadDiWord.restype = c_int

        self._dll.NKDIO_PollingWriteDoByte.argtypes = [c_ubyte, c_ubyte]
        self._dll.NKDIO_PollingWriteDoByte.restype = c_int

        self._dll.NKDIO_PollingWriteDoWord.argtypes = [c_ubyte, c_ushort]
        self._dll.NKDIO_PollingWriteDoWord.restype = c_int

        self._dll.NKDIO_PollingReadDoByte.argtypes = [c_ubyte, POINTER(c_ubyte)]
        self._dll.NKDIO_PollingReadDoByte.restype = c_int

        self._dll.NKDIO_PollingReadDoWord.argtypes = [c_ubyte, POINTER(c_ushort)]
        self._dll.NKDIO_PollingReadDoWord.restype = c_int

    def library_init(self, config_file: str | Path) -> int:
        config_bytes = str(Path(config_file)).encode("utf-8")
        return int(self._dll.NKDIO_LibraryInit(config_bytes))

    def library_deinit(self) -> None:
        self._dll.NKDIO_LibraryDeinit()

    def read_di_byte(self, index: int) -> tuple[int, int]:
        value = c_ubyte()
        ret = int(self._dll.NKDIO_PollingReadDiByte(c_ubyte(index), byref(value)))
        return ret, int(value.value)

    def read_di_word(self, index: int = 0) -> tuple[int, int]:
        value = c_ushort()
        ret = int(self._dll.NKDIO_PollingReadDiWord(c_ubyte(index), byref(value)))
        return ret, int(value.value)

    def write_do_byte(self, index: int, value: int) -> int:
        return int(self._dll.NKDIO_PollingWriteDoByte(c_ubyte(index), c_ubyte(value)))

    def write_do_word(self, index: int, value: int) -> int:
        return int(self._dll.NKDIO_PollingWriteDoWord(c_ubyte(index), c_ushort(value)))

    def read_do_byte(self, index: int) -> tuple[int, int]:
        value = c_ubyte()
        ret = int(self._dll.NKDIO_PollingReadDoByte(c_ubyte(index), byref(value)))
        return ret, int(value.value)

    def read_do_word(self, index: int = 0) -> tuple[int, int]:
        value = c_ushort()
        ret = int(self._dll.NKDIO_PollingReadDoWord(c_ubyte(index), byref(value)))
        return ret, int(value.value)
