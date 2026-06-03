from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Sequence


_DEFAULT_OPENCV_BUILD_ROOT = Path(r"C:\Users\ADMIN\tools\opencv\build")
NATIVE_BACKEND_TO_MODULE = {
    "original": "shape_original",
    "fusion": "shape_fusion",
    "fusionv2": "shape_fusionv2",
    "sim3": "shape_sim3",
}
NATIVE_BACKEND_LEGACY_MODULES: dict[str, tuple[str, ...]] = {}
_NATIVE_MODULES: dict[str, Any] = {}
_OPENCV_DLL_HANDLE: Any = None
_NATIVE_MODULE_ERRORS: dict[str, BaseException] = {}
_NATIVE_FALLBACK_WARNED: set[str] = set()


def _opencv_build_root() -> Path:
    override = os.environ.get("LINE2DUP_OPENCV_BUILD", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_OPENCV_BUILD_ROOT


def _native_build_instructions() -> str:
    return (
        "To enable the OpenCV-backed accelerators:\n"
        "py -3 -m pip install -U setuptools wheel pybind11\n"
        "py -3 EmbeddingTest\\setup.py build_ext --inplace\n"
        f"Set LINE2DUP_OPENCV_BUILD if OpenCV is not installed at {_DEFAULT_OPENCV_BUILD_ROOT}.\n"
        "Optional: set LINE2DUP_OPENCV_WORLD_LIB when your OpenCV world library name is not auto-detected."
    )


def _native_fallback_warn_enabled() -> bool:
    value = os.environ.get("LINE2DUP_WARN_NATIVE_FALLBACK", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def normalize_backend_name(backend: str) -> str:
    key = str(backend or "original").strip().lower()
    if key in {"orig", "original", "native"}:
        return "original"
    if key in {"fusion", "fused"}:
        return "fusion"
    if key in {"fusionv2", "fusion_v2", "fused_v2"}:
        return "fusionv2"
    if key in {"sim3", "icp", "icp(sim3)", "sim3_icp"}:
        return "sim3"
    raise ValueError(f"Unsupported backend: {backend}")


def _warn_native_fallback(backend: str, exc: BaseException) -> None:
    backend = normalize_backend_name(backend)
    if backend in _NATIVE_FALLBACK_WARNED:
        return
    if not _native_fallback_warn_enabled():
        _NATIVE_FALLBACK_WARNED.add(backend)
        return
    print(
        f"{NATIVE_BACKEND_TO_MODULE[backend]} is unavailable; falling back to the slower Python matcher.\n"
        f"{_native_build_instructions()}\n"
        f"Original import error: {exc}",
        file=sys.stderr,
    )
    _NATIVE_FALLBACK_WARNED.add(backend)


def native_module_candidates(backend: str) -> tuple[str, ...]:
    backend = normalize_backend_name(backend)
    preferred = NATIVE_BACKEND_TO_MODULE[backend]
    legacy = tuple(NATIVE_BACKEND_LEGACY_MODULES.get(backend, ()))
    return (preferred, *(name for name in legacy if name != preferred))


def load_native_matcher(backend: str = "original", required: bool = True) -> Any:
    backend = normalize_backend_name(backend)
    module_names = native_module_candidates(backend)
    module_name = module_names[0]
    global _OPENCV_DLL_HANDLE
    if backend in _NATIVE_MODULES:
        return _NATIVE_MODULES[backend]
    if backend in _NATIVE_MODULE_ERRORS:
        if required:
            raise RuntimeError(
                f"{module_name} is unavailable.\n{_native_build_instructions()}\n"
                f"Original import error: {_NATIVE_MODULE_ERRORS[backend]}"
            )
        return None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_dir = _opencv_build_root() / "x64" / "vc16" / "bin"
        if dll_dir.exists() and _OPENCV_DLL_HANDLE is None:
            _OPENCV_DLL_HANDLE = os.add_dll_directory(str(dll_dir))
    errors: list[BaseException] = []
    for candidate in module_names:
        try:
            _NATIVE_MODULES[backend] = importlib.import_module(candidate)
            return _NATIVE_MODULES[backend]
        except Exception as exc:  # pragma: no cover - exercised via runtime import failures
            errors.append(exc)
    exc = errors[-1]
    _NATIVE_MODULE_ERRORS[backend] = exc
    if required:
        attempted = ", ".join(module_names)
        raise RuntimeError(
            f"{module_name} is unavailable. Tried: {attempted}.\n"
            f"{_native_build_instructions()}\nOriginal import error: {exc}"
        ) from exc
    return None


def ensure_native_backends_available(backends: Sequence[str] = ("original", "fusion", "fusionv2", "sim3")) -> None:
    for backend in backends:
        load_native_matcher(backend=backend, required=True)


def create_native_detector(
    num_features: int,
    T_levels: Sequence[int],
    weak_threshold: float,
    strong_threshold: float,
    *,
    backend: str = "original",
) -> Any:
    native_matcher = load_native_matcher(backend=backend, required=True)
    return native_matcher.NativeDetector(
        int(num_features),
        [int(t) for t in T_levels],
        float(weak_threshold),
        float(strong_threshold),
    )


_normalize_backend_name = normalize_backend_name
_native_module_candidates = native_module_candidates
_load_native_matcher = load_native_matcher


__all__ = [
    "NATIVE_BACKEND_LEGACY_MODULES",
    "NATIVE_BACKEND_TO_MODULE",
    "_NATIVE_MODULE_ERRORS",
    "_NATIVE_MODULES",
    "_native_build_instructions",
    "_load_native_matcher",
    "_native_module_candidates",
    "_normalize_backend_name",
    "_warn_native_fallback",
    "create_native_detector",
    "ensure_native_backends_available",
    "load_native_matcher",
    "native_module_candidates",
    "normalize_backend_name",
]
