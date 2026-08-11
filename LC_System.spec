from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(globals().get("SPECPATH", Path.cwd())).resolve()
APP_ROOT = ROOT
REPO_ROOT = APP_ROOT.parent
SDK_ROOT = REPO_ROOT / "NKDIOLC_SDK"
RES_ROOT = APP_ROOT / "res"
CONFIG_ROOT = APP_ROOT / "config"
RECORDS_ROOT = APP_ROOT / "records"
SESSION_ROOT = APP_ROOT / ".qr_session"
ORT_BACKBONE_CACHE_ROOT = APP_ROOT / ".cache" / "ort_backbones"
MVIMPORT_ROOT = APP_ROOT / "third_party" / "MvImport"
NKIO_ROOT = APP_ROOT / "third_party" / "nkio"
NKIO_DLL = NKIO_ROOT / "NKIOLIBx64.dll"
ICON_PATH = RES_ROOT / "logo.ico"
NKIO_BIN_ROOT = SDK_ROOT / "Bin"
NKIO_SELECT_INI = NKIO_BIN_ROOT / "select.ini"
MVS_RUNTIME_ENV_VALUE = str(os.environ.get("HIK_MVS_RUNTIME_DIR", "")).strip()
MVS_RUNTIME_DIR_CANDIDATES = (
    ([Path(MVS_RUNTIME_ENV_VALUE)] if MVS_RUNTIME_ENV_VALUE else [])
    + [
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
        Path(r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64"),
    ]
)
MVS_RUNTIME_DIR = next((path for path in MVS_RUNTIME_DIR_CANDIDATES if str(path).strip() and path.exists()), None)
MVIMPORT_DATA_FILES = [
    MVIMPORT_ROOT / "CameraParams_const.py",
    MVIMPORT_ROOT / "CameraParams_header.py",
    MVIMPORT_ROOT / "MvCameraControl_class.py",
    MVIMPORT_ROOT / "MvErrorDefine_const.py",
    MVIMPORT_ROOT / "PixelType_header.py",
]
CONFIG_DATA_FILES = [
    CONFIG_ROOT / "camera_settings.json",
    CONFIG_ROOT / "system_passwords.json",
    CONFIG_ROOT / "tower_light_settings.json",
    CONFIG_ROOT / "runtime_record_settings.json",
    CONFIG_ROOT / "runtime_mode_settings.json",
    CONFIG_ROOT / "ui_settings.json",
]
CONFIG_DATA_DIRS = [
    (CONFIG_ROOT / "defaults", "EmbeddingTest/config/defaults"),
]
ROOT_LEVEL_BINARY_FILES = [
    SDK_ROOT / "Lib" / "x64" / "WinRing0.dll",
    SDK_ROOT / "Lib" / "x64" / "WinRing0x64.dll",
    SDK_ROOT / "Lib" / "x64" / "NKLCLIBx64.dll",
    SDK_ROOT / "Lib" / "x64" / "NKIOLIBx64.dll",
]
def _first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


WINRING0X64_SYS = _first_existing_path(
    SDK_ROOT / "Sample" / "CPP" / "NK_IO_LC_TEST_Console" / "x64" / "Release" / "WinRing0x64.sys",
    SDK_ROOT / "Sample" / "CPP" / "NK_IO_LC_TEST_Console" / "x64" / "Debug" / "WinRing0x64.sys",
    SDK_ROOT / "Sample" / "Qt" / "NK_IO_LC_TEST_Qt" / "x64" / "Release" / "WinRing0x64.sys",
    SDK_ROOT / "Sample" / "Qt" / "NK_IO_LC_TEST_Qt" / "x64" / "Debug" / "WinRing0x64.sys",
    SDK_ROOT / "Sample" / "CPP" / "NK_IO_LC_TEST_Console" / "SDKLib" / "Lib" / "x64" / "WinRing0x64.sys",
    SDK_ROOT / "Sample" / "Qt" / "NK_IO_LC_TEST_Qt" / "SDKLib" / "Lib" / "x64" / "WinRing0x64.sys",
    SDK_ROOT / "Lib" / "x64" / "WinRing0x64.sys",
    SDK_ROOT / "Bin" / "WinRing0x64.sys",
)

ROOT_LEVEL_DATA_FILES = [
    SDK_ROOT / "Lib" / "x64" / "WinRing0.sys",
]
if WINRING0X64_SYS is not None:
    ROOT_LEVEL_DATA_FILES.append(WINRING0X64_SYS)
PYD_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"


def _selected_nkio_config_dir(select_ini_path: Path) -> Path | None:
    if not select_ini_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.optionxform = str
    try:
        parser.read(select_ini_path, encoding="utf-8")
    except Exception:
        return None
    if not parser.has_section("SELECTED"):
        return None
    config_path = str(parser.get("SELECTED", "ConfigPath", fallback="") or "").strip()
    if not config_path:
        return None
    relative_path = Path(config_path.lstrip("/\\").replace("/", "\\"))
    if not relative_path.parts:
        return None
    candidate = NKIO_BIN_ROOT / relative_path
    return candidate.parent if candidate.exists() else None


NKIO_SELECTED_CONFIG_DIR = _selected_nkio_config_dir(NKIO_SELECT_INI)


def _pair(src: Path, dest: str = "."):
    return (str(src), dest)


datas = []
for src, dest in (
    (RES_ROOT, "EmbeddingTest/res"),
    (SESSION_ROOT, "EmbeddingTest/.qr_session"),
    (ORT_BACKBONE_CACHE_ROOT, "EmbeddingTest/.cache/ort_backbones"),
):
    if src.exists():
        datas.append(_pair(src, dest))

if MVS_RUNTIME_DIR is not None:
    datas.append(_pair(MVS_RUNTIME_DIR, "EmbeddingTest/third_party/MVS/Runtime/Win64_x64"))

if NKIO_SELECT_INI.exists():
    datas.append(_pair(NKIO_SELECT_INI, "NKDIOLC_SDK/Bin"))

for path in (NKIO_BIN_ROOT / "config.ini",):
    if path.exists():
        datas.append(_pair(path, "NKDIOLC_SDK/Bin"))

if NKIO_SELECTED_CONFIG_DIR is not None:
    datas.append(_pair(NKIO_SELECTED_CONFIG_DIR, f"NKDIOLC_SDK/Bin/{NKIO_SELECTED_CONFIG_DIR.name}"))

for src, dest in CONFIG_DATA_DIRS:
    if src.exists():
        datas.append(_pair(src, dest))

for path in CONFIG_DATA_FILES:
    if path.exists():
        datas.append(_pair(path, "EmbeddingTest/config"))

# The build script creates this clean seed database. It contains only the
# default accounts and permissions—never development-machine audit or runtime
# history.
seed_audit_db = RECORDS_ROOT / ".package-seed" / "audit.db"
if not seed_audit_db.exists():
    raise RuntimeError("Missing package seed database; build with build_py312.ps1.")
datas.append(_pair(seed_audit_db, "EmbeddingTest/records"))

for path in MVIMPORT_DATA_FILES:
    if path.exists():
        datas.append(_pair(path, "EmbeddingTest/third_party/MvImport"))

binaries = []
for path in APP_ROOT.glob("*.dll"):
    binaries.append(_pair(path))
for path in APP_ROOT.glob("*.pyd"):
    if PYD_TAG in path.name:
        binaries.append(_pair(path))
for path in ROOT_LEVEL_BINARY_FILES:
    if path.exists():
        binaries.append(_pair(path))
if NKIO_DLL.exists():
    binaries.append(_pair(NKIO_DLL, "EmbeddingTest/third_party/nkio"))
for path in ROOT_LEVEL_DATA_FILES:
    if path.exists():
        datas.append(_pair(path))


hiddenimports = sorted(
    set(
        collect_submodules("algorithms")
        + collect_submodules("application")
        + collect_submodules("application.runtime")
        + collect_submodules("common")
        + collect_submodules("config")
        + collect_submodules("devices")
        + collect_submodules("domain")
        + collect_submodules("infrastructure")
        + collect_submodules("ncc")
        + collect_submodules("shape")
        + collect_submodules("services")
        + collect_submodules("ui")
        + [
            "algorithms.api",
            "algorithms.embedding",
            "algorithms.embedding_core",
            "algorithms.localization",
            "algorithms.traditional",
            "application.algorithm_controller",
            "application.inspection_executor",
            "application.product_session",
            "application.runtime",
            "application.runtime.controller",
            "application.runtime_context",
            "common.algorithm_codes",
            "common.app_paths",
            "common.labelme_io",
            "common.path_utils",
            "common.safe_io",
            "domain.inspection_items",
            "domain.inspection_models",
            "domain.result_aggregator",
            "matplotlib.backends.backend_qtagg",
            "matplotlib.figure",
            "ncc.authoring",
            "ncc.ui",
            "ncc.ui.workbench_dialog",
            "ui.debug.embedding_analysis",
            "ui.debug.embedding_analysis_dialog",
            "ui.debug.roi_canvas_pyside6",
            "ui.debug.tool_page",
            "ui.debug.tool_page.camera_debug_view",
            "ui.debug.tool_page.io_debug_view",
            "ui.debug.tool_page.page",
            "ui.runtime.runtime_mode_pyside6",
            "ui.runtime.run_main_window",
            "shape.core.recipe_labels",
            "ui.shape_template",
            "ui.shape_template.template_page_pyside6",
            "CameraParams_const",
            "CameraParams_header",
            "MvCameraControl_class",
            "MvErrorDefine_const",
            "PixelType_header",
        ]
    )
)


a = Analysis(
    [str(APP_ROOT / "Main.py")],
    pathex=[str(APP_ROOT), str(MVIMPORT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jinja2",
        "matplotlib_inline",
        "tensorboard",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LC System",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory=".",
    icon=str(ICON_PATH) if ICON_PATH.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LC System",
)
