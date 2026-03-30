from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(globals().get("SPECPATH", Path.cwd())).resolve()
APP_ROOT = ROOT
REPO_ROOT = APP_ROOT.parent
SDK_ROOT = REPO_ROOT / "NKDIOLC_SDK"
RES_ROOT = APP_ROOT / "res"
CONFIG_ROOT = APP_ROOT / "config"
SESSION_ROOT = APP_ROOT / ".qr_session"
MVIMPORT_ROOT = APP_ROOT / "third_party" / "MvImport"
NKIO_ROOT = APP_ROOT / "third_party" / "nkio"
NKIO_DLL = NKIO_ROOT / "NKIOLIBx64.dll"
ICON_PATH = RES_ROOT / "logo.ico"
ROOT_LEVEL_BINARY_FILES = [
    SDK_ROOT / "Lib" / "x64" / "WinRing0.dll",
    SDK_ROOT / "Lib" / "x64" / "WinRing0x64.dll",
    SDK_ROOT / "Lib" / "x64" / "NKLCLIBx64.dll",
    SDK_ROOT / "Lib" / "x64" / "NKIOLIBx64.dll",
]
ROOT_LEVEL_DATA_FILES = [
    SDK_ROOT / "Lib" / "x64" / "WinRing0.sys",
    SDK_ROOT / "Lib" / "x64" / "WinRing0x64.sys",
]


def _pair(src: Path, dest: str = "."):
    return (str(src), dest)


datas = []
for src, dest in (
    (RES_ROOT, "EmbeddingTest/res"),
    (CONFIG_ROOT, "EmbeddingTest/config"),
    (SESSION_ROOT, "EmbeddingTest/.qr_session"),
    (MVIMPORT_ROOT, "EmbeddingTest/third_party/MvImport"),
):
    if src.exists():
        datas.append(_pair(src, dest))

if (APP_ROOT / "setup.py").exists():
    datas.append(_pair(APP_ROOT / "setup.py", "EmbeddingTest"))


binaries = []
for pattern in ("*.pyd", "*.dll"):
    for path in APP_ROOT.glob(pattern):
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
        + collect_submodules("config")
        + collect_submodules("devices")
        + collect_submodules("domain")
        + collect_submodules("infrastructure")
        + collect_submodules("line2dup")
        + collect_submodules("services")
        + collect_submodules("ui")
        + [
            "application.algorithm_controller",
            "application.inspection_executor",
            "application.product_session",
            "application.runtime",
            "application.runtime.controller",
            "application.runtime_context",
            "application.runtime_controller",
            "domain.inspection_items",
            "domain.inspection_models",
            "domain.recipe_manager",
            "domain.result_aggregator",
            "ui.debug.roi_canvas_pyside6",
            "ui.debug.tool_page_pyside6",
            "ui.debug.tool_page",
            "ui.debug.tool_page.page",
            "ui.runtime.runtime_mode_pyside6",
            "ui.runtime.run_main_window",
            "line2dup.ui.template_page_pyside6",
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
    excludes=[],
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
