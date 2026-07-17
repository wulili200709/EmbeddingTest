from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

from PIL import Image, ImageDraw, ImageFont
from PySide6 import QtCore, QtGui, QtWidgets


CANVAS_SIZE = (1600, 900)
BACKGROUND = (31, 35, 41)
ACCENT = (239, 68, 68)
OUTPUT_FILES = (
    "01-main-overview.png",
    "02-workspace-navigation.png",
    "03-product-overview.png",
    "04-roi-editor.png",
    "05-train-samples.png",
    "06-test-samples.png",
    "07-inspection-items.png",
    "08-algorithm-parameters.png",
    "09-location-template.png",
    "10-status-bar.png",
    "11-annotation-overview.png",
    "12-annotation-actions.png",
    "13-annotation-test-samples.png",
    "14-auto-roi-overview.png",
    "15-auto-roi-actions.png",
    "16-camera-debug.png",
    "17-camera-capture-channels.png",
    "18-io-debug.png",
    "19-io-output-controls.png",
    "20-shape-template.png",
    "21-ncc-workbench.png",
    "22-runtime-overview.png",
    "23-runtime-layout.png",
    "24-runtime-result.png",
    "25-login.png",
    "26-user-permissions.png",
    "27-role-permissions.png",
    "28-audit-log.png",
    "29-runtime-records.png",
    "30-software-version.png",
    "31-tools-menu.png",
    "32-control-menu.png",
    "33-path-menu.png",
    "34-language-menu.png",
)


@dataclass(frozen=True)
class Scenario:
    filename: str
    capture: Callable[["CaptureContext"], None]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("msyhbd.ttc" if bold else "msyh.ttc"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _copy_isolated_runtime(temp_root: Path, product: str) -> None:
    source_session = ROOT / ".qr_session"
    source_product = source_session / product
    if not source_product.is_dir():
        raise FileNotFoundError(f"Product not found: {source_product}")

    target_session = temp_root / ".qr_session"
    target_session.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_product, target_session / product)
    products_payload = (
        '{\n  "products": ["%s"],\n  "current_product": "%s"\n}\n' % (product, product)
    )
    (target_session / "products.json").write_text(products_payload, encoding="utf-8")

    source_config = ROOT / "config"
    if source_config.exists():
        shutil.copytree(
            source_config,
            temp_root / "config",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    source_records = ROOT / "records"
    if source_records.exists():
        shutil.copytree(source_records, temp_root / "records")
    else:
        (temp_root / "records").mkdir(parents=True, exist_ok=True)


def _patch_writable_root(temp_root: Path) -> None:
    import common.app_paths as app_paths

    app_paths.writable_embedding_test_root = lambda _anchor=None: temp_root


def _process_events(rounds: int = 4) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for _ in range(rounds):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)


class CaptureContext:
    def __init__(self, *, output: Path, product: str, temp_root: Path) -> None:
        self.output = output
        self.product = product
        self.temp_root = temp_root
        self.window: QtWidgets.QMainWindow | None = None
        self.dialogs: list[QtWidgets.QWidget] = []

    @property
    def tool_page(self):
        assert self.window is not None
        return self.window.tool_page

    @property
    def runtime_page(self):
        assert self.window is not None
        return self.window.runtime_page

    def keep(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.dialogs.append(widget)
        return widget

    def save(
        self,
        filename: str,
        widget: QtWidgets.QWidget,
        targets: Iterable[QtWidgets.QWidget] = (),
        *,
        title: str = "",
    ) -> None:
        widget.show()
        widget.raise_()
        widget.activateWindow()
        _process_events()
        pixmap = widget.grab()
        raw_path = self.temp_root / f"raw-{filename}"
        if not pixmap.save(str(raw_path), "PNG"):
            raise RuntimeError(f"Unable to grab widget for {filename}")
        source = Image.open(raw_path).convert("RGB")

        max_w, max_h = 1540, 830
        scale = min(max_w / source.width, max_h / source.height, 1.0)
        if scale < 1.0:
            source = source.resize(
                (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
                Image.Resampling.LANCZOS,
            )
        canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
        x0 = (CANVAS_SIZE[0] - source.width) // 2
        y0 = (CANVAS_SIZE[1] - source.height) // 2 + (14 if title else 0)
        canvas.paste(source, (x0, y0))
        draw = ImageDraw.Draw(canvas)
        if title:
            draw.text((30, 18), title, font=_font(24, bold=True), fill=(238, 241, 245))

        for index, target in enumerate(targets, start=1):
            if target is None or not target.isVisible():
                continue
            try:
                top_left = target.mapTo(widget, QtCore.QPoint(0, 0))
                rect = target.rect()
                left = x0 + int(top_left.x() * scale)
                top = y0 + int(top_left.y() * scale)
                right = left + max(4, int(rect.width() * scale))
                bottom = top + max(4, int(rect.height() * scale))
            except RuntimeError:
                continue
            pad = 4
            draw.rounded_rectangle(
                (left - pad, top - pad, right + pad, bottom + pad),
                radius=7,
                outline=ACCENT,
                width=4,
            )
            circle_x = max(10, min(CANVAS_SIZE[0] - 42, left - 18))
            circle_y = max(10, min(CANVAS_SIZE[1] - 42, top - 18))
            draw.ellipse((circle_x, circle_y, circle_x + 36, circle_y + 36), fill=ACCENT)
            label = str(index)
            box = draw.textbbox((0, 0), label, font=_font(20, bold=True))
            tw = box[2] - box[0]
            th = box[3] - box[1]
            draw.text(
                (circle_x + (36 - tw) / 2, circle_y + (36 - th) / 2 - 2),
                label,
                font=_font(20, bold=True),
                fill="white",
            )

        destination = self.output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG", optimize=True)
        print(f"captured {destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination}")

    def close(self) -> None:
        for dialog in reversed(self.dialogs):
            try:
                dialog.close()
            except RuntimeError:
                pass
        if self.window is not None:
            self.window.close()
        _process_events()
        self.dialogs.clear()
        self.window = None
        for attr in ("_annotation", "_auto_roi"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()


def _main_scenario(filename: str, targets: tuple[str, ...], title: str = ""):
    def capture(ctx: CaptureContext) -> None:
        assert ctx.window is not None
        ctx.window._switch_workspace("debug")
        widgets = [getattr(ctx.window, name, None) or getattr(ctx.tool_page, name, None) for name in targets]
        ctx.save(filename, ctx.window, [w for w in widgets if isinstance(w, QtWidgets.QWidget)], title=title)

    return capture


def _capture_train(ctx: CaptureContext) -> None:
    ctx.window._switch_workspace("debug")
    ctx.tool_page.tabs.setCurrentIndex(0)
    ctx.save("05-train-samples.png", ctx.window, (ctx.tool_page.tabs, ctx.tool_page.ok_list, ctx.tool_page.btn_import_train), title="训练样本")


def _capture_test(ctx: CaptureContext) -> None:
    ctx.window._switch_workspace("debug")
    ctx.tool_page.tabs.setCurrentIndex(1)
    ctx.save("06-test-samples.png", ctx.window, (ctx.tool_page.tabs, ctx.tool_page.test_list, ctx.tool_page.btn_add_test), title="测试样本")


def _capture_algorithm(ctx: CaptureContext) -> None:
    ctx.window._switch_workspace("debug")
    ctx.tool_page._toggle_algorithm_section(True)
    ctx.save("08-algorithm-parameters.png", ctx.window, (ctx.tool_page.btn_toggle_algo, ctx.tool_page.algorithm_params_frame), title="算法参数")


def _annotation_dialog(ctx: CaptureContext):
    cached = getattr(ctx, "_annotation", None)
    if cached is None:
        from ui.debug.tool_page.sample_annotation_dialog import _SampleAnnotationPreviewDialog

        ctx.tool_page.tabs.setCurrentIndex(0)
        cached = ctx.keep(_SampleAnnotationPreviewDialog(ctx.tool_page, parent=ctx.window))
        cached.resize(1180, 760)
        cached.show()
        _process_events()
        setattr(ctx, "_annotation", cached)
    return cached


def _capture_annotation_overview(ctx: CaptureContext) -> None:
    dialog = _annotation_dialog(ctx)
    ctx.save("11-annotation-overview.png", dialog, (dialog.sample_list, dialog.preview_canvas, dialog.roi_table), title="样本标注")


def _capture_annotation_actions(ctx: CaptureContext) -> None:
    dialog = _annotation_dialog(ctx)
    ctx.save("12-annotation-actions.png", dialog, (dialog.btn_mark_all_ok, dialog.btn_mark_all_ng, dialog.btn_clear_current, dialog.btn_open_autogen), title="标注状态与批量操作")


def _capture_annotation_test(ctx: CaptureContext) -> None:
    dialog = _annotation_dialog(ctx)
    index = dialog.cmb_sample_kind.findData("test")
    if index >= 0:
        dialog.cmb_sample_kind.setCurrentIndex(index)
    _process_events()
    ctx.save("13-annotation-test-samples.png", dialog, (dialog.cmb_sample_kind, dialog.sample_list, dialog.roi_table), title="测试样本标注")


def _auto_roi_dialog(ctx: CaptureContext):
    cached = getattr(ctx, "_auto_roi", None)
    if cached is None:
        from ui.debug.tool_page.sample_auto_roi_dialog import _SampleAnnotationAutoRoiDialog

        preview = _annotation_dialog(ctx)
        train_index = preview.cmb_sample_kind.findData("train")
        if train_index >= 0:
            preview.cmb_sample_kind.setCurrentIndex(train_index)
        _process_events()
        cached = ctx.keep(_SampleAnnotationAutoRoiDialog(preview))
        cached.resize(900, 300)
        cached.show()
        _process_events()
        setattr(ctx, "_auto_roi", cached)
    return cached


def _capture_auto_roi(ctx: CaptureContext, filename: str, action_only: bool) -> None:
    dialog = _auto_roi_dialog(ctx)
    targets = (
        (dialog.btn_autogen_current, dialog.btn_autogen_current_image, dialog.btn_clear_current)
        if action_only
        else (dialog.cmb_location_template, dialog.lbl_ref, dialog.chk_only_missing)
    )
    ctx.save(filename, dialog, targets, title="自动生成 ROI")


def _capture_tool_page(ctx: CaptureContext, filename: str, attr: str, targets: tuple[str, ...], size: tuple[int, int]) -> None:
    page = getattr(ctx.tool_page, attr)
    dialog = ctx.keep(QtWidgets.QDialog(ctx.window))
    dialog.setWindowTitle("LC System")
    dialog.resize(*size)
    layout = QtWidgets.QVBoxLayout(dialog)
    page.setParent(dialog)
    layout.addWidget(page)
    target_widgets = [getattr(ctx.tool_page, name) for name in targets if hasattr(ctx.tool_page, name)]
    ctx.save(filename, dialog, target_widgets, title=dialog.windowTitle())


def _capture_shape(ctx: CaptureContext) -> None:
    from ui.shape_template.template_page_pyside6 import ShapeTemplateDialog

    dialog = ctx.keep(
        ShapeTemplateDialog(
            product_name=ctx.product,
            product_dir=ctx.tool_page.session.product_dir,
            camera_role=ctx.tool_page.current_camera_role(),
            initial_image_path=str(ctx.tool_page.canvas.image_path() or ""),
            parent=ctx.window,
        )
    )
    dialog.resize(1450, 820)
    targets = tuple(w for w in (getattr(dialog, "tabs", None), getattr(dialog, "canvas", None)) if isinstance(w, QtWidgets.QWidget))
    ctx.save("20-shape-template.png", dialog, targets, title="Shape 定位模板")


def _capture_ncc(ctx: CaptureContext) -> None:
    from ncc.ui.workbench_dialog import NccMatchWorkbenchDialog

    dialog = ctx.keep(
        NccMatchWorkbenchDialog(
            product_name=ctx.product,
            product_dir=ctx.tool_page.session.product_dir,
            camera_role=ctx.tool_page.current_camera_role(),
            initial_image_path=str(ctx.tool_page.canvas.image_path() or ""),
            parent=ctx.window,
        )
    )
    dialog.resize(1450, 820)
    targets = tuple(w for w in (getattr(dialog, "tabs", None), getattr(dialog, "find_canvas", None)) if isinstance(w, QtWidgets.QWidget))
    ctx.save("21-ncc-workbench.png", dialog, targets, title="NCC 位置修正工具")


def _capture_runtime(ctx: CaptureContext, filename: str, mode: str) -> None:
    ctx.window.main_pages.setCurrentWidget(ctx.runtime_page)
    ctx.window.btn_workspace_debug.setChecked(False)
    ctx.window.btn_workspace_runtime.setChecked(True)
    page = ctx.runtime_page
    page.set_current_product(ctx.product)
    if mode == "overview":
        targets = (page.lbl_current_product, page._camera_grid_host, page.lbl_final_result)
    elif mode == "layout":
        targets = (page.cmb_camera_layout, page._camera_grid_host)
    else:
        page.set_final_result("NG", "离线文档示例")
        page._ok_count_total = 12
        page._ng_count_total = 1
        page._refresh_count_labels()
        targets = (page.lbl_final_result, page.lbl_ok_count, page.lbl_ng_count, page.btn_release)
    ctx.save(filename, ctx.window, targets, title="运行检测")


def _capture_login(ctx: CaptureContext) -> None:
    from ui.shell.audit_dialogs import LoginDialog

    dialog = ctx.keep(LoginDialog(ctx.window))
    dialog.edit_user.setText("operator")
    dialog.edit_password.clear()
    ctx.save("25-login.png", dialog, (dialog.edit_user, dialog.edit_password), title="用户登录")


def _capture_admin_dialog(ctx: CaptureContext, filename: str, kind: str) -> None:
    from ui.shell.audit_dialogs import (
        AuditLogDialog,
        RuntimeRecordsDialog,
        SoftwareVersionDialog,
        UserPermissionDialog,
    )

    if kind == "users" or kind == "roles":
        dialog = ctx.keep(UserPermissionDialog(ctx.window, ctx.window._audit_store))
        tabs = dialog.findChild(QtWidgets.QTabWidget)
        if kind == "roles" and tabs is not None:
            tabs.setCurrentIndex(1)
        targets = (tabs,) if tabs is not None else ()
    elif kind == "audit":
        dialog = ctx.keep(AuditLogDialog(ctx.window, ctx.window._audit_store, can_export=False))
        dialog._query()
        targets = (dialog.table, dialog.cmb_product_filter)
    elif kind == "records":
        dialog = ctx.keep(RuntimeRecordsDialog(ctx.window, ctx.window._runtime_results_store, can_export=False))
        dialog._query()
        targets = (dialog.table_runs, dialog.table_roi_results)
    else:
        from ui.shell.support import APP_VERSION

        dialog = ctx.keep(
            SoftwareVersionDialog(
                ctx.window,
                ctx.window._audit_store,
                can_edit=False,
                current_user="operator",
                software_version=APP_VERSION,
            )
        )
        targets = (dialog.table, dialog.edit_version)
    ctx.save(filename, dialog, targets, title=dialog.windowTitle())


def _capture_menu(ctx: CaptureContext, filename: str, key: str) -> None:
    menu = ctx.window._shell_i18n_refs["menus"][key]
    menu.adjustSize()
    ctx.save(filename, menu, (menu,), title=menu.title())


SCENARIOS = (
    Scenario("01-main-overview.png", _main_scenario("01-main-overview.png", ("btn_workspace_debug", "btn_workspace_runtime", "cmb_product", "canvas", "tabs"), "LC System 主界面")),
    Scenario("02-workspace-navigation.png", _main_scenario("02-workspace-navigation.png", ("btn_workspace_debug", "btn_workspace_runtime", "sidebar_runtime_result_frame"), "工作区切换")),
    Scenario("03-product-overview.png", _main_scenario("03-product-overview.png", ("cmb_product", "btn_new_product", "btn_delete_product"), "产品管理")),
    Scenario("04-roi-editor.png", _main_scenario("04-roi-editor.png", ("canvas", "cmb_shape", "cmb_label", "btn_save", "btn_clear"), "ROI 编辑")),
    Scenario("05-train-samples.png", _capture_train),
    Scenario("06-test-samples.png", _capture_test),
    Scenario("07-inspection-items.png", _main_scenario("07-inspection-items.png", ("btn_toggle_tools", "inspection_items_table", "lbl_current_tool_sample_stats"), "检测项目")),
    Scenario("08-algorithm-parameters.png", _capture_algorithm),
    Scenario("09-location-template.png", _main_scenario("09-location-template.png", ("lbl_ref", "btn_set_ref", "btn_pick_ref", "cmb_loc", "btn_autogen", "btn_autogen_all"), "定位模板与批量 ROI")),
    Scenario("10-status-bar.png", _main_scenario("10-status-bar.png", ("lbl_status_workspace", "lbl_status_product", "lbl_status_engine", "lbl_status_io_text"), "状态栏")),
    Scenario("11-annotation-overview.png", _capture_annotation_overview),
    Scenario("12-annotation-actions.png", _capture_annotation_actions),
    Scenario("13-annotation-test-samples.png", _capture_annotation_test),
    Scenario("14-auto-roi-overview.png", lambda ctx: _capture_auto_roi(ctx, "14-auto-roi-overview.png", False)),
    Scenario("15-auto-roi-actions.png", lambda ctx: _capture_auto_roi(ctx, "15-auto-roi-actions.png", True)),
    Scenario("16-camera-debug.png", lambda ctx: _capture_tool_page(ctx, "16-camera-debug.png", "camera_debug_page", ("cmb_debug_role", "cmb_debug_camera", "btn_debug_connect", "btn_debug_capture"), (1200, 760))),
    Scenario("17-camera-capture-channels.png", lambda ctx: ctx.save("17-camera-capture-channels.png", ctx.dialogs[-1], (ctx.tool_page.capture_channel_table,), title="相机采集通道")),
    Scenario("18-io-debug.png", lambda ctx: _capture_tool_page(ctx, "18-io-debug.png", "io_debug_page", ("btn_debug_open_io", "lbl_debug_di_snapshot", "lbl_debug_do_snapshot"), (1100, 620))),
    Scenario("19-io-output-controls.png", lambda ctx: ctx.save("19-io-output-controls.png", ctx.dialogs[-1], tuple(ctx.tool_page._debug_do_channel_buttons.values()), title="I/O 输出控制")),
    Scenario("20-shape-template.png", _capture_shape),
    Scenario("21-ncc-workbench.png", _capture_ncc),
    Scenario("22-runtime-overview.png", lambda ctx: _capture_runtime(ctx, "22-runtime-overview.png", "overview")),
    Scenario("23-runtime-layout.png", lambda ctx: _capture_runtime(ctx, "23-runtime-layout.png", "layout")),
    Scenario("24-runtime-result.png", lambda ctx: _capture_runtime(ctx, "24-runtime-result.png", "result")),
    Scenario("25-login.png", _capture_login),
    Scenario("26-user-permissions.png", lambda ctx: _capture_admin_dialog(ctx, "26-user-permissions.png", "users")),
    Scenario("27-role-permissions.png", lambda ctx: _capture_admin_dialog(ctx, "27-role-permissions.png", "roles")),
    Scenario("28-audit-log.png", lambda ctx: _capture_admin_dialog(ctx, "28-audit-log.png", "audit")),
    Scenario("29-runtime-records.png", lambda ctx: _capture_admin_dialog(ctx, "29-runtime-records.png", "records")),
    Scenario("30-software-version.png", lambda ctx: _capture_admin_dialog(ctx, "30-software-version.png", "version")),
    Scenario("31-tools-menu.png", lambda ctx: _capture_menu(ctx, "31-tools-menu.png", "tools")),
    Scenario("32-control-menu.png", lambda ctx: _capture_menu(ctx, "32-control-menu.png", "runtime")),
    Scenario("33-path-menu.png", lambda ctx: _capture_menu(ctx, "33-path-menu.png", "path")),
    Scenario("34-language-menu.png", lambda ctx: _capture_menu(ctx, "34-language-menu.png", "language")),
)


def _build_window(ctx: CaptureContext) -> None:
    from ui.i18n import set_language
    from ui.shell.main_window import MainWindow

    set_language("zh")
    app = QtWidgets.QApplication.instance()
    if app is not None:
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path)) if font_path.exists() else -1
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        app.setFont(QtGui.QFont(families[0] if families else "Microsoft YaHei UI", 10))
    original_single_shot = QtCore.QTimer.singleShot
    QtCore.QTimer.singleShot = lambda *_args, **_kwargs: None
    try:
        window = MainWindow()
    finally:
        QtCore.QTimer.singleShot = original_single_shot
    window._allow_initial_tool_session_load = True
    window.tool_page.load_session()
    if getattr(window.tool_page, "ok_list", None) is not None and window.tool_page.ok_list.count():
        window.tool_page.ok_list.setCurrentRow(0)
    window.resize(*CANVAS_SIZE)
    window.show()
    _process_events(8)
    ctx.window = window


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture LC System beginner-guide screenshots safely.")
    parser.add_argument("--product", default="022618Cover")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "beginner-guide" / "images")
    parser.add_argument("--scenario", default="all", help="all, a filename, or a comma-separated filename list")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    requested = {item.strip() for item in args.scenario.split(",") if item.strip()}
    selected = list(SCENARIOS) if "all" in requested else [s for s in SCENARIOS if s.filename in requested]
    unknown = requested - {"all"} - {s.filename for s in SCENARIOS}
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {', '.join(sorted(unknown))}")
    if not selected:
        raise SystemExit("No screenshot scenarios selected")

    with tempfile.TemporaryDirectory(prefix="lc-system-manual-", ignore_cleanup_errors=True) as temp_dir:
        temp_root = Path(temp_dir)
        _copy_isolated_runtime(temp_root, args.product)
        _patch_writable_root(temp_root)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        app.setApplicationName("LC System Manual Capture")
        ctx = CaptureContext(output=output, product=args.product, temp_root=temp_root)
        try:
            _build_window(ctx)
            for scenario in selected:
                scenario.capture(ctx)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
