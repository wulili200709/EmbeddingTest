from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import numpy
import pybind11


ROOT = Path(__file__).resolve().parent
OPENCV_ROOT = Path(r"C:\Users\ADMIN\tools\opencv\build")
OPENCV_INCLUDE_DIR = OPENCV_ROOT / "include"
OPENCV_LIB_DIR = OPENCV_ROOT / "x64" / "vc16" / "lib"
OPENCV_BIN_DIR = OPENCV_ROOT / "x64" / "vc16" / "bin"
OPENCV_WORLD_LIB = "opencv_world4130"

ORIGINAL_ROOT = ROOT / "_third_party_shape_based_matching"
FUSION_ROOT = ROOT / "_third_party_shape_based_matching_fusion_fix_memo"
SIM3_ROOT = ROOT / "_third_party_shape_based_matching_sim3"


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing {description}: {path}")
    return path


def common_compile_args() -> list[str]:
    return ["/O2", "/EHsc", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]


def build_backend_extension(
    *,
    module_name: str,
    backend_root: Path,
    extra_sources: list[str] | None = None,
    extra_include_dirs: list[Path] | None = None,
    extra_define_macros: list[tuple[str, str | None]] | None = None,
) -> Extension:
    require_path(OPENCV_INCLUDE_DIR, "OpenCV include dir")
    require_path(OPENCV_LIB_DIR, "OpenCV lib dir")
    require_path(OPENCV_BIN_DIR, "OpenCV bin dir")
    require_path(OPENCV_LIB_DIR / f"{OPENCV_WORLD_LIB}.lib", "OpenCV world library")
    require_path(backend_root / "line2Dup.cpp", f"{module_name} line2Dup.cpp")
    require_path(backend_root / "line2Dup.h", f"{module_name} line2Dup.h")
    require_path(backend_root / "MIPP", f"{module_name} MIPP dir")

    sources = [
        "native/line2dup_native.cpp",
        str(backend_root / "line2Dup.cpp"),
    ]
    if extra_sources:
        sources.extend(extra_sources)

    define_macros: list[tuple[str, str | None]] = [
        ("_CRT_SECURE_NO_WARNINGS", "1"),
        ("LINE2DUP_PYMODULE_NAME", module_name),
    ]
    if extra_define_macros:
        define_macros.extend(extra_define_macros)

    include_dirs = [
        pybind11.get_include(),
        numpy.get_include(),
        str(backend_root),
        str(backend_root / "MIPP"),
        str(OPENCV_INCLUDE_DIR),
    ]
    if extra_include_dirs:
        include_dirs.extend(str(path) for path in extra_include_dirs)

    return Extension(
        name=module_name,
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=[str(OPENCV_LIB_DIR)],
        libraries=[OPENCV_WORLD_LIB],
        language="c++",
        extra_compile_args=common_compile_args(),
        define_macros=define_macros,
    )


def build_extensions() -> list[Extension]:
    extensions = [
        build_backend_extension(
            module_name="line2dup_native",
            backend_root=ORIGINAL_ROOT,
        ),
        build_backend_extension(
            module_name="line2dup_fusion_native",
            backend_root=FUSION_ROOT,
        ),
        build_backend_extension(
            module_name="line2dup_sim3_native",
            backend_root=SIM3_ROOT,
            extra_sources=[
                str(SIM3_ROOT / "cuda_icp" / "icp.cpp"),
                str(SIM3_ROOT / "cuda_icp" / "scene" / "edge_scene" / "edge_scene.cpp"),
            ],
            extra_include_dirs=[
                SIM3_ROOT / "cuda_icp",
                SIM3_ROOT / "third_party" / "eigen",
            ],
            extra_define_macros=[
                ("LINE2DUP_ENABLE_SIM3_ICP", "1"),
            ],
        ),
    ]
    return extensions


class BuildExtWithOpenCVDll(build_ext):
    def run(self) -> None:
        super().run()
        dll_name = f"{OPENCV_WORLD_LIB}.dll"
        dll_src = require_path(OPENCV_BIN_DIR / dll_name, "OpenCV runtime DLL")
        for ext in self.extensions:
            ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            target_dir = ext_path.parent
            shutil.copy2(dll_src, target_dir / dll_name)


setup(
    name="line2dup_native_backends",
    version="0.0.3",
    ext_modules=build_extensions(),
    cmdclass={"build_ext": BuildExtWithOpenCVDll},
)
