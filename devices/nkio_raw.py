from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from ctypes import POINTER, byref, c_char_p, c_int, c_ubyte, c_ushort
from pathlib import Path
from typing import Iterable

from common.app_paths import packaged_embedding_test_root, packaged_repo_root


def _repo_root() -> Path:
    return packaged_repo_root(__file__)


def _iter_candidate_dll_paths(dll_name: str) -> Iterable[Path]:
    repo_root = _repo_root()
    embedding_root = packaged_embedding_test_root(__file__)
    yield repo_root / "NKDIOLC_SDK" / "Bin" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Lib" / "x64" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Debug" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Release" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "Release" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "Debug" / dll_name
    yield repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "Lib" / "x64" / dll_name
    yield embedding_root / "third_party" / "nkio" / dll_name
    yield repo_root / "EmbeddingTest" / "third_party" / "nkio" / dll_name
    yield Path(dll_name)


def _iter_candidate_search_dirs(dll_path: Path, extra_dirs: Iterable[Path] | None = None) -> Iterable[Path]:
    repo_root = _repo_root()
    embedding_root = packaged_embedding_test_root(__file__)
    seen: set[Path] = set()
    candidates = [
        dll_path.parent,
        repo_root / "NKDIOLC_SDK" / "Bin",
        repo_root / "NKDIOLC_SDK" / "Lib" / "x64",
        repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Debug",
        repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "x64" / "Release",
        repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "Release",
        repo_root / "NKDIOLC_SDK" / "Sample" / "C#" / "NK_IO_LC_TEST_CSharp" / "bin" / "Debug",
        embedding_root / "third_party" / "nkio",
        repo_root / "EmbeddingTest" / "third_party" / "nkio",
    ]
    if extra_dirs is not None:
        candidates = [*list(extra_dirs), *candidates]
    for path in candidates:
        resolved = Path(path)
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        yield resolved


def _load_cdll_with_search_dirs(
    dll_path: Path,
    *,
    extra_dirs: Iterable[Path] | None = None,
) -> tuple[ctypes.CDLL, list[object]]:
    handles: list[object] = []
    add_dir = getattr(os, "add_dll_directory", None)
    search_dirs = list(_iter_candidate_search_dirs(dll_path, extra_dirs=extra_dirs))
    try:
        for search_dir in search_dirs:
            if add_dir is None:
                continue
            handles.append(add_dir(str(search_dir)))
        with _temporary_path_prefix(search_dirs):
            return ctypes.CDLL(str(dll_path)), handles
    except Exception:
        for handle in reversed(handles):
            try:
                handle.close()
            except Exception:
                pass
        raise


@contextmanager
def _temporary_cwd(path: Path):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


@contextmanager
def _temporary_path_prefix(paths: Iterable[Path]):
    original_path = os.environ.get("PATH", "")
    normalized: list[str] = []
    seen: set[str] = set()
    for path in paths:
        candidate = str(Path(path))
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if normalized:
        os.environ["PATH"] = os.pathsep.join([*normalized, original_path] if original_path else normalized)
    try:
        yield
    finally:
        os.environ["PATH"] = original_path


def find_default_nkio_dll_path() -> Path:
    for path in _iter_candidate_dll_paths("NKIOLIBx64.dll"):
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in _iter_candidate_dll_paths("NKIOLIBx64.dll"))
    raise FileNotFoundError(f"Could not locate NKIOLIBx64.dll. Searched:\n{searched}")


class NkioRawLib:
    """Thin ctypes wrapper around NKIOLIBx64.dll."""

    def __init__(self, dll_path: str | Path | None = None) -> None:
        self._dll_search_handles: list[object] = []
        if dll_path is not None:
            resolved_path = Path(dll_path)
            if not resolved_path.exists():
                raise FileNotFoundError(resolved_path)
            self.dll_path = resolved_path
            self._dll, self._dll_search_handles = _load_cdll_with_search_dirs(self.dll_path)
        else:
            self.dll_path, self._dll, self._dll_search_handles = self._load_first_available_default_dll()
        self._bind()

    @staticmethod
    def _runtime_extra_dirs_for_config(config_file: str | Path) -> tuple[list[Path], Path]:
        config_path = Path(config_file)
        extra_dirs: list[Path] = [config_path.parent]
        working_dir = config_path.parent
        parent = config_path.parent
        if parent.name.upper().startswith("NP-"):
            extra_dirs.insert(0, parent.parent)
            working_dir = parent.parent
        return extra_dirs, working_dir

    @staticmethod
    def _config_init_argument(config_path: Path, working_dir: Path) -> bytes:
        try:
            relative = config_path.resolve().relative_to(working_dir.resolve())
        except Exception:
            absolute_text = str(config_path)
            try:
                return absolute_text.encode("ascii")
            except UnicodeEncodeError:
                return absolute_text.encode("utf-8")
        relative_text = str(relative).replace("/", "\\")
        if not relative_text.startswith(".\\"):
            relative_text = f".\\{relative_text}"
        try:
            return relative_text.encode("ascii")
        except UnicodeEncodeError:
            return str(config_path).encode("utf-8")

    @staticmethod
    def _load_first_available_default_dll() -> tuple[Path, ctypes.CDLL, list[object]]:
        errors: list[str] = []
        for path in _iter_candidate_dll_paths("NKIOLIBx64.dll"):
            if not path.exists():
                continue
            try:
                dll, handles = _load_cdll_with_search_dirs(path)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            return path, dll, handles
        searched = "\n".join(str(p) for p in _iter_candidate_dll_paths("NKIOLIBx64.dll"))
        detail = "\n".join(errors)
        if detail:
            raise OSError(f"Could not load NKIOLIBx64.dll from any candidate path.\nSearched:\n{searched}\nErrors:\n{detail}")
        raise FileNotFoundError(f"Could not locate NKIOLIBx64.dll. Searched:\n{searched}")

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
        config_path = Path(config_file)
        extra_dirs, working_dir = self._runtime_extra_dirs_for_config(config_path)
        config_bytes = self._config_init_argument(config_path, working_dir)
        temp_handles: list[object] = []
        add_dir = getattr(os, "add_dll_directory", None)
        search_dirs = list(_iter_candidate_search_dirs(self.dll_path, extra_dirs=extra_dirs))
        try:
            if add_dir is not None:
                for search_dir in search_dirs:
                    temp_handles.append(add_dir(str(search_dir)))
            with _temporary_path_prefix(search_dirs):
                with _temporary_cwd(working_dir):
                    return int(self._dll.NKDIO_LibraryInit(config_bytes))
        finally:
            for handle in reversed(temp_handles):
                try:
                    handle.close()
                except Exception:
                    pass

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
