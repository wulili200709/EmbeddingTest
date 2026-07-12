from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from ctypes import POINTER, byref, c_char_p, c_int, c_uint8, c_void_p
from pathlib import Path
from typing import Iterable

import numpy as np

from common.app_paths import packaged_embedding_test_root, packaged_repo_root


def _repo_root() -> Path:
    return packaged_repo_root(__file__)


def _iter_candidate_dll_paths(dll_name: str) -> Iterable[Path]:
    repo_root = _repo_root()
    embedding_root = packaged_embedding_test_root(__file__)
    yield embedding_root / dll_name
    yield embedding_root / "third_party" / "ncc" / dll_name
    yield repo_root / dll_name
    yield repo_root / "EmbeddingTest" / "third_party" / "ncc" / dll_name
    yield Path(dll_name)


def _iter_candidate_search_dirs(dll_path: Path) -> Iterable[Path]:
    repo_root = _repo_root()
    embedding_root = packaged_embedding_test_root(__file__)
    seen: set[Path] = set()
    candidates = [
        dll_path.parent,
        embedding_root,
        embedding_root / "third_party" / "ncc",
        repo_root,
        repo_root / "EmbeddingTest" / "third_party" / "ncc",
    ]
    for path in candidates:
        resolved = Path(path)
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        yield resolved


@contextmanager
def _temporary_path_prefix(paths: Iterable[Path]):
    original_path = os.environ.get("PATH", "")
    prefix = os.pathsep.join(str(path) for path in paths if str(path))
    if prefix:
        os.environ["PATH"] = os.pathsep.join([prefix, original_path] if original_path else [prefix])
    try:
        yield
    finally:
        os.environ["PATH"] = original_path


def _load_cdll_with_search_dirs(dll_path: Path) -> tuple[ctypes.CDLL, list[object]]:
    search_dirs = list(_iter_candidate_search_dirs(dll_path))
    add_dir = getattr(os, "add_dll_directory", None)
    handles: list[object] = []
    try:
        for search_dir in search_dirs:
            if add_dir is not None:
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


def find_default_ncc_dll_path() -> Path:
    for path in _iter_candidate_dll_paths("NccMatchNative.dll"):
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in _iter_candidate_dll_paths("NccMatchNative.dll"))
    raise FileNotFoundError(f"Could not locate NccMatchNative.dll. Searched:\n{searched}")


class NccNativeApi:
    _instance: "NccNativeApi | None" = None

    def __init__(self, dll_path: str | Path | None = None) -> None:
        self._dll_search_handles: list[object] = []
        if dll_path is None:
            self.dll_path = find_default_ncc_dll_path()
        else:
            self.dll_path = Path(dll_path).resolve()
        self._dll, self._dll_search_handles = _load_cdll_with_search_dirs(self.dll_path)
        self._bind()

    @classmethod
    def load(cls) -> "NccNativeApi":
        if cls._instance is None:
            cls._instance = NccNativeApi()
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        try:
            api = cls.load()
            error = c_void_p()
            try:
                return bool(api._dll.ncc_is_native_available(byref(error)))
            finally:
                api.free_native_string(error)
        except Exception:
            return False

    def _bind(self) -> None:
        self._dll.ncc_is_native_available.argtypes = [POINTER(c_void_p)]
        self._dll.ncc_is_native_available.restype = c_int

        self._dll.ncc_create_matcher.argtypes = [
            POINTER(c_uint8),
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            POINTER(c_void_p),
        ]
        self._dll.ncc_create_matcher.restype = c_void_p

        self._dll.ncc_match_json.argtypes = [
            c_void_p,
            POINTER(c_uint8),
            c_int,
            c_int,
            c_int,
            c_int,
            c_char_p,
            POINTER(c_void_p),
            POINTER(c_void_p),
        ]
        self._dll.ncc_match_json.restype = c_int

        self._dll.ncc_destroy_matcher.argtypes = [c_void_p]
        self._dll.ncc_destroy_matcher.restype = None

        self._dll.ncc_free_string.argtypes = [c_void_p]
        self._dll.ncc_free_string.restype = None

    @staticmethod
    def _ensure_gray_uint8(image_gray: np.ndarray) -> np.ndarray:
        if image_gray is None or image_gray.size == 0:
            raise ValueError("image_gray is required")
        prepared = np.ascontiguousarray(image_gray)
        if prepared.ndim == 2:
            return prepared.astype(np.uint8, copy=False)
        if prepared.ndim == 3 and prepared.shape[2] == 1:
            return prepared[:, :, 0].astype(np.uint8, copy=False)
        raise ValueError("NCC native matcher expects a single-channel uint8 image")

    @staticmethod
    def _read_native_string(pointer: c_void_p) -> str:
        if not pointer:
            return ""
        return ctypes.cast(pointer, c_char_p).value.decode("utf-8", errors="replace")

    def free_native_string(self, pointer: c_void_p) -> None:
        if pointer:
            self._dll.ncc_free_string(pointer)

    def create_matcher(self, template_gray: np.ndarray, min_reduced_area: int) -> c_void_p:
        template = self._ensure_gray_uint8(template_gray)
        error = c_void_p()
        handle = self._dll.ncc_create_matcher(
            template.ctypes.data_as(POINTER(c_uint8)),
            int(template.shape[1]),
            int(template.shape[0]),
            int(template.strides[0]),
            1,
            max(16, int(min_reduced_area)),
            byref(error),
        )
        try:
            if not handle:
                detail = self._read_native_string(error) or "Failed to create NCC matcher."
                raise RuntimeError(detail)
            return handle
        finally:
            self.free_native_string(error)

    def match_json(self, matcher: c_void_p, source_gray: np.ndarray, payload: dict) -> dict:
        source = self._ensure_gray_uint8(source_gray)
        options_json = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        json_out = c_void_p()
        error = c_void_p()
        try:
            ok = self._dll.ncc_match_json(
                matcher,
                source.ctypes.data_as(POINTER(c_uint8)),
                int(source.shape[1]),
                int(source.shape[0]),
                int(source.strides[0]),
                1,
                options_json,
                byref(json_out),
                byref(error),
            )
            if not ok:
                detail = self._read_native_string(error) or "Failed to execute NCC match."
                raise RuntimeError(detail)
            text = self._read_native_string(json_out) or "{\"matches\": []}"
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"matches": []}
        finally:
            self.free_native_string(json_out)
            self.free_native_string(error)

    def destroy_matcher(self, matcher: c_void_p) -> None:
        if matcher:
            self._dll.ncc_destroy_matcher(matcher)


class NccNativeMatcher:
    def __init__(self, template_gray: np.ndarray, min_reduced_area: int = 256) -> None:
        self._api = NccNativeApi.load()
        self._handle = self._api.create_matcher(template_gray, min_reduced_area)

    def match(self, source_gray: np.ndarray, options_payload: dict) -> dict:
        if not self._handle:
            raise RuntimeError("NCC native matcher has been closed")
        return self._api.match_json(self._handle, source_gray, options_payload)

    def close(self) -> None:
        if self._handle:
            self._api.destroy_matcher(self._handle)
            self._handle = c_void_p()

    def __enter__(self) -> "NccNativeMatcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["NccNativeApi", "NccNativeMatcher", "find_default_ncc_dll_path"]
