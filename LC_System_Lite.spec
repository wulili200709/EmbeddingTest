from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(globals().get("SPECPATH", Path.cwd())).resolve()
APP_ROOT = ROOT
RES_ROOT = APP_ROOT / "res"
CONFIG_ROOT = APP_ROOT / "config"
SESSION_ROOT = APP_ROOT / ".qr_session"
ICON_PATH = RES_ROOT / "logo.ico"
PYD_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"


def _pair(src: Path, dest: str = "."):
    return (str(src), dest)


datas = []
for src, dest in (
    (RES_ROOT, "EmbeddingTest/res"),
    (SESSION_ROOT, "EmbeddingTest/.qr_session"),
):
    if src.exists():
        datas.append(_pair(src, dest))

for src, dest in (
    (CONFIG_ROOT / "defaults", "EmbeddingTest/config/defaults"),
):
    if src.exists():
        datas.append(_pair(src, dest))

for path in (
    CONFIG_ROOT / "camera_settings.json",
    CONFIG_ROOT / "system_passwords.json",
    CONFIG_ROOT / "runtime_record_settings.json",
    CONFIG_ROOT / "runtime_mode_settings.json",
    CONFIG_ROOT / "ui_settings.json",
):
    if path.exists():
        datas.append(_pair(path, "EmbeddingTest/config"))


binaries = []
for path in APP_ROOT.glob("*.dll"):
    binaries.append(_pair(path))
for path in APP_ROOT.glob("*.pyd"):
    if PYD_TAG in path.name:
        binaries.append(_pair(path))


hiddenimports = sorted(
    set(
        collect_submodules("common")
        + collect_submodules("config")
        + collect_submodules("shape")
        + collect_submodules("ui.shape_template")
        + [
            "algorithms.lazy_api",
            "algorithms.localization",
            "algorithms.measurement",
            "algorithms.measurement_lines",
            "algorithms.measurement_pin_center",
            "algorithms.measurement_types",
            "algorithms.registry",
            "algorithms.traditional",
            "application.algorithm_controller",
            "application.auto_roi_service",
            "application.inspection_executor",
            "application.product_session",
            "application.runtime.preview_frame",
            "application.runtime_context",
            "domain.inspection_items",
            "domain.inspection_models",
            "domain.result_aggregator",
            "infrastructure.camera_settings_store",
            "infrastructure.product_params",
            "shape_fusion",
            "shape_fusionv2",
            "shape_original",
            "shape_sim3",
            "ui.algorithm_labels",
            "ui.debug",
            "ui.debug_main_window",
            "ui.debug.roi_canvas_pyside6",
            "ui.debug.tool_page",
            "ui.debug.tool_page.action_panel_view",
            "ui.debug.tool_page.algorithm_catalog",
            "ui.debug.tool_page.analysis_tools",
            "ui.debug.tool_page.auto_roi",
            "ui.debug.tool_page.auto_roi_flow",
            "ui.debug.tool_page.auto_roi_reference_flow",
            "ui.debug.tool_page.bindings",
            "ui.debug.tool_page.camera_debug",
            "ui.debug.tool_page.camera_roles",
            "ui.debug.tool_page.debug_camera_flow",
            "ui.debug.tool_page.debug_camera_runtime",
            "ui.debug.tool_page.debug_io_flow",
            "ui.debug.tool_page.measurement_algorithms",
            "ui.debug.tool_page.measurement_tool_config",
            "ui.debug.tool_page.measurement_tool_options",
            "ui.debug.tool_page.page",
            "ui.debug.tool_page.product_session_controller",
            "ui.debug.tool_page.roi_annotation_controller",
            "ui.debug.tool_page.roi_measurement_overlays",
            "ui.debug.tool_page.roi_ops",
            "ui.debug.tool_page.sample_annotation_canvas",
            "ui.debug.tool_page.sample_annotation_dialog",
            "ui.debug.tool_page.sample_annotation_overlay",
            "ui.debug.tool_page.sample_auto_roi_dialog",
            "ui.debug.tool_page.sample_list_controller",
            "ui.debug.tool_page.sample_panel_view",
            "ui.debug.tool_page.test_execution_controller",
            "ui.debug.tool_page.test_runner",
            "ui.debug.tool_page.tool_config",
            "ui.debug.tool_page.tool_config_view",
            "ui.debug.tool_page.training_controller",
            "ui.debug.tool_page.training_roi_review",
            "ui.debug.tool_page.training_task_builder",
            "ui.debug.tool_page.training_worker",
            "ui.debug.tool_page.view_builders",
            "ui.i18n",
            "ui.roi_overlay_colors",
            "ui.window_common",
        ]
    )
)


a = Analysis(
    [str(APP_ROOT / "MainLite.py")],
    pathex=[str(APP_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "CameraParams_const",
        "CameraParams_header",
        "IPython",
        "MvCameraControl_class",
        "MvErrorDefine_const",
        "PixelType_header",
        "application.runtime.bindings",
        "application.runtime.controller",
        "application.runtime.execution",
        "application.runtime.hardware",
        "devices",
        "jinja2",
        "matplotlib",
        "matplotlib_inline",
        "onnx",
        "onnxruntime",
        "pandas",
        "scipy",
        "services",
        "sklearn",
        "tensorboard",
        "third_party.MvImport",
        "torch",
        "torch.utils.tensorboard",
        "torchvision",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LC System Lite",
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
    name="LC System Lite",
)
