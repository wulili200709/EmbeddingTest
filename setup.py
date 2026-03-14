from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import numpy
import pybind11


ROOT = Path(__file__).resolve().parent
OPENCV_ROOT = Path(os.environ.get("LINE2DUP_OPENCV_BUILD", r"C:\Users\ADMIN\tools\opencv\build")).expanduser()
OPENCV_INCLUDE_DIR = OPENCV_ROOT / "include"
OPENCV_LIB_DIR = OPENCV_ROOT / "x64" / "vc16" / "lib"
OPENCV_BIN_DIR = OPENCV_ROOT / "x64" / "vc16" / "bin"


def resolve_opencv_world_lib(lib_dir: Path) -> str:
    override = os.environ.get("LINE2DUP_OPENCV_WORLD_LIB", "").strip()
    if override:
        return override
    candidates = sorted(path.stem for path in lib_dir.glob("opencv_world*.lib"))
    if len(candidates) == 1:
        return candidates[0]
    return "opencv_world4130"


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing {description}: {path}")
    return path


def build_extension() -> Extension:
    require_path(OPENCV_INCLUDE_DIR, "OpenCV include dir")
    require_path(OPENCV_LIB_DIR, "OpenCV lib dir")
    require_path(OPENCV_BIN_DIR, "OpenCV bin dir")
    opencv_world_lib = resolve_opencv_world_lib(OPENCV_LIB_DIR)
    require_path(OPENCV_LIB_DIR / f"{opencv_world_lib}.lib", "OpenCV world library")

    extra_compile_args = ["/O2", "/EHsc", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    define_macros = [("_CRT_SECURE_NO_WARNINGS", "1")]

    return Extension(
        name="line2dup_native",
        sources=[
            "native/line2dup_native.cpp",
            "_third_party_shape_based_matching/line2Dup.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            str(ROOT / "_third_party_shape_based_matching"),
            str(ROOT / "_third_party_shape_based_matching" / "MIPP"),
            str(OPENCV_INCLUDE_DIR),
        ],
        library_dirs=[str(OPENCV_LIB_DIR)],
        libraries=[opencv_world_lib],
        language="c++",
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
    )


class BuildExtWithOpenCVDll(build_ext):
    def run(self) -> None:
        super().run()
        dll_name = f"{resolve_opencv_world_lib(OPENCV_LIB_DIR)}.dll"
        dll_src = require_path(OPENCV_BIN_DIR / dll_name, "OpenCV runtime DLL")
        for ext in self.extensions:
            ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            target_dir = ext_path.parent
            shutil.copy2(dll_src, target_dir / dll_name)


setup(
    name="line2dup_native",
    version="0.0.2",
    ext_modules=[build_extension()],
    cmdclass={"build_ext": BuildExtWithOpenCVDll},
)
