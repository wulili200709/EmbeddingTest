from __future__ import annotations

import os
from pathlib import Path

import numpy
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


SCRIPT_ROOT = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_ROOT.parent
SOURCE = SCRIPT_ROOT / "native" / "embedding_core_native.cpp"


def compile_args() -> list[str]:
    return ["/O2", "/EHsc", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]


class InplaceBuildExt(build_ext):
    def run(self) -> None:
        APP_ROOT.joinpath("algorithms").mkdir(parents=True, exist_ok=True)
        super().run()


os.chdir(APP_ROOT)

setup(
    name="embedding_core_native",
    version="0.0.1",
    ext_modules=[
        Extension(
            name="algorithms.embedding_core",
            sources=[str(SOURCE)],
            include_dirs=[pybind11.get_include(), numpy.get_include()],
            language="c++",
            extra_compile_args=compile_args(),
        )
    ],
    cmdclass={"build_ext": InplaceBuildExt},
)
