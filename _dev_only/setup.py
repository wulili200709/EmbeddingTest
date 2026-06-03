from __future__ import annotations

import os
import stat
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import numpy
import pybind11


SCRIPT_ROOT = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_ROOT.parent
PATCHED_BACKEND_ROOT = SCRIPT_ROOT / "build" / "_patched_backends"
OPENCV_ROOT = Path(os.environ.get("LINE2DUP_OPENCV_BUILD", r"C:\Users\ADMIN\tools\opencv\build")).expanduser()
OPENCV_INCLUDE_DIR = OPENCV_ROOT / "include"
OPENCV_LIB_DIR = OPENCV_ROOT / "x64" / "vc16" / "lib"
OPENCV_BIN_DIR = OPENCV_ROOT / "x64" / "vc16" / "bin"
OPENCV_WORLD_LIB = os.environ.get("LINE2DUP_OPENCV_WORLD_LIB", "opencv_world4130")

ORIGINAL_ROOT = SCRIPT_ROOT / "vendor" / "_third_party_shape_based_matching"
FUSION_ROOT = SCRIPT_ROOT / "vendor" / "_third_party_shape_based_matching_fusion_fix_memo"
SIM3_ROOT = SCRIPT_ROOT / "vendor" / "_third_party_shape_based_matching_sim3"
EIGEN_VENDOR_ROOT = SCRIPT_ROOT / "vendor" / "eigen"


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing {description}: {path}")
    return path


def common_compile_args() -> list[str]:
    return ["/O2", "/EHsc", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]


def openmp_compile_args() -> list[str]:
    return ["/openmp"] if os.name == "nt" else ["-fopenmp"]


def openmp_link_args() -> list[str]:
    return [] if os.name == "nt" else ["-fopenmp"]


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise RuntimeError(f"Failed to apply backend patch: {label}")


def _remove_readonly(func, path: str, _exc) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


FUSION_PATCH_MODULES = {
    "shape_fusion",
    "shape_fusionv2",
    "line2dup_fusion_native",
    "line2dup_fusionv2_native",
    "match_fusionv2",
}
SIM3_PATCH_MODULES = {
    "shape_sim3",
    "line2dup_sim3_native",
}


def prepare_backend_root(module_name: str, backend_root: Path) -> Path:
    if module_name not in FUSION_PATCH_MODULES | SIM3_PATCH_MODULES:
        return backend_root

    patched_root = PATCHED_BACKEND_ROOT / module_name
    if patched_root.exists():
        shutil.rmtree(patched_root, onerror=_remove_readonly)
    shutil.copytree(backend_root, patched_root, ignore=shutil.ignore_patterns(".git", "build_bench"))

    if module_name in FUSION_PATCH_MODULES:
        line2dup_cpp = patched_root / "line2Dup.cpp"
        text = line2dup_cpp.read_text(encoding="utf-8")
        openmp_old = """#pragma omp declare reduction \\
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(omp_insert:matches)
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
"""
        openmp_new = """#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp declare reduction \\
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(omp_insert:matches)
#elif defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int template_id = 0; template_id < static_cast<int>(template_pyramids.size()); ++template_id)
"""
        if openmp_new in text:
            pass
        elif openmp_old in text:
            text = text.replace(openmp_old, openmp_new, 1)
        elif "#pragma omp declare reduction \\" not in text or "template_pyramids.size()" not in text:
            raise RuntimeError("Failed to apply backend patch: fusion openmp loop")
        plain_insert = "        matches.insert(matches.end(), candidates.begin(), candidates.end());\n"
        critical_named = """#if defined(_OPENMP) && defined(_MSC_VER)
#pragma omp critical(line2dup_match_collect)
#endif
        matches.insert(matches.end(), candidates.begin(), candidates.end());
"""
        critical_plain = """#if defined(_OPENMP) && defined(_MSC_VER)
#pragma omp critical
#endif
        matches.insert(matches.end(), candidates.begin(), candidates.end());
"""
        critical_double = """#if defined(_OPENMP) && defined(_MSC_VER)
#pragma omp critical
#endif
#if defined(_OPENMP) && defined(_MSC_VER)
#pragma omp critical
#endif
        matches.insert(matches.end(), candidates.begin(), candidates.end());
"""
        if critical_named in text:
            pass
        elif critical_double in text:
            text = text.replace(critical_double, critical_named, 1)
        elif critical_plain in text:
            text = text.replace(critical_plain, critical_named, 1)
        else:
            text = _replace_once(text, plain_insert, critical_named, label="fusion msvc critical section")
        line2dup_cpp.write_text(text, encoding="utf-8")

    if module_name in SIM3_PATCH_MODULES:
        edge_scene_cpp = patched_root / "cuda_icp" / "scene" / "edge_scene" / "edge_scene.cpp"
        text = edge_scene_cpp.read_text(encoding="utf-8")
        text = _replace_once(
            text,
            "        cv::cvtColor(img, gray, CV_BGR2GRAY);\n",
            "        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);\n",
            label="sim3 cvtColor enum",
        )
        edge_scene_cpp.write_text(text, encoding="utf-8")

    return patched_root


def build_backend_extension(
    *,
    module_name: str,
    backend_root: Path,
    extra_sources: list[str] | None = None,
    extra_include_dirs: list[Path] | None = None,
    extra_define_macros: list[tuple[str, str | None]] | None = None,
    extra_compile_args: list[str] | None = None,
    extra_link_args: list[str] | None = None,
) -> Extension:
    backend_root = prepare_backend_root(module_name, backend_root)
    require_path(OPENCV_INCLUDE_DIR, "OpenCV include dir")
    require_path(OPENCV_LIB_DIR, "OpenCV lib dir")
    require_path(OPENCV_BIN_DIR, "OpenCV bin dir")
    require_path(OPENCV_LIB_DIR / f"{OPENCV_WORLD_LIB}.lib", "OpenCV world library")
    require_path(backend_root / "line2Dup.cpp", f"{module_name} line2Dup.cpp")
    require_path(backend_root / "line2Dup.h", f"{module_name} line2Dup.h")
    require_path(backend_root / "MIPP", f"{module_name} MIPP dir")

    sources = [
        str(SCRIPT_ROOT / "native" / "line2dup_native.cpp"),
        str(backend_root / "line2Dup.cpp"),
    ]
    if extra_sources:
        sources.extend(str((backend_root / source) if not Path(source).is_absolute() else Path(source)) for source in extra_sources)

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
        include_dirs.extend(str((backend_root / path) if not Path(path).is_absolute() else Path(path)) for path in extra_include_dirs)

    compile_args = common_compile_args()
    if extra_compile_args:
        compile_args.extend(extra_compile_args)

    return Extension(
        name=module_name,
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=[str(OPENCV_LIB_DIR)],
        libraries=[OPENCV_WORLD_LIB],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=list(extra_link_args or []),
        define_macros=define_macros,
    )


def build_extensions() -> list[Extension]:
    require_path(EIGEN_VENDOR_ROOT / "Eigen", "vendored Eigen headers")
    extensions = [
        build_backend_extension(
            module_name="shape_original",
            backend_root=ORIGINAL_ROOT,
        ),
        build_backend_extension(
            module_name="shape_fusion",
            backend_root=FUSION_ROOT,
            extra_compile_args=openmp_compile_args(),
            extra_link_args=openmp_link_args(),
        ),
        build_backend_extension(
            module_name="shape_fusionv2",
            backend_root=FUSION_ROOT,
            extra_compile_args=openmp_compile_args(),
            extra_link_args=openmp_link_args(),
            extra_define_macros=[
                ("LINE2DUP_ENABLE_FUSION_V2", "1"),
            ],
        ),
        build_backend_extension(
            module_name="shape_sim3",
            backend_root=SIM3_ROOT,
            extra_sources=[
                "cuda_icp/icp.cpp",
                "cuda_icp/scene/edge_scene/edge_scene.cpp",
            ],
            extra_include_dirs=[
                "cuda_icp",
                EIGEN_VENDOR_ROOT,
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
            runtime_ext_path = APP_ROOT / ext_path.name
            if ext_path != runtime_ext_path:
                shutil.copy2(ext_path, runtime_ext_path)
            shutil.copy2(dll_src, APP_ROOT / dll_name)


setup(
    name="shape_native_backends",
    version="0.0.5",
    ext_modules=build_extensions(),
    cmdclass={"build_ext": BuildExtWithOpenCVDll},
)
