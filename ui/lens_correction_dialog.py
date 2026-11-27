"""Product/camera lens settings and coordinate-space guards for image editors."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
from contextlib import nullcontext

from PySide6 import QtCore, QtWidgets

from ui.i18n import language_code
from ui.lens_calibration_theme import apply_lens_calibration_theme
from ui.lens_calibration_text import diagnostic_text


_TEXT = {
    "title": ("镜头矫正设置…", "Lens correction settings…"),
    "product": ("产品 · {name}", "Product · {name}"),
    "camera": ("相机", "Camera"), "enabled": ("启用", "Enabled"),
    "profile": ("标定文件", "Calibration file"), "mode": ("模式 / 尺寸", "Mode / size"),
    "serial": ("相机序列号", "Camera serial"),
    "import": ("导入参数…", "Import profile…"),
    "demo": ("标定与预览…", "Calibrate and preview…"),
    "save": ("保存设置", "Save settings"), "cancel": ("取消", "Cancel"),
    "close": ("关闭", "Close"),
    "none": ("未选择参数", "No profile selected"),
    "radial": ("径向", "Radial"), "radial_perspective": ("径向 + 透视", "Radial + perspective"),
    "pinhole": ("相机内参", "Camera intrinsics"),
    "multi_view": ("多图相机标定", "Multi-view calibration"),
    "reopen": ("镜头矫正设置已改变。请关闭并重新打开此编辑器，再创建或测试模板；本窗口的旧坐标不能保存到新设置。", "Lens correction settings changed. Close and reopen this editor before creating or testing templates; coordinates from this window cannot be saved under the new settings."),
    "busy": ("运行监控或后台任务仍在进行。请先停止/断开采集监控，并等待检测、训练、标定或图片加载完成，再修改镜头矫正设置。", "Runtime monitoring or a background task is active. Stop/disconnect capture monitoring and wait for inspection, training, calibration or image loading to finish before changing lens correction settings."),
    "choose_product": ("请先选择产品。", "Select a product first."),
    "missing": ("{role} 尚未导入标定参数，不能启用。", "Import a calibration profile before enabling {role}."),
    "model_space": ("该模型不属于当前镜头矫正版本。请在当前校正图上创建模板，再使用产品模型进行测试。", "This model does not belong to the current lens correction version. Create a template on the corrected image and test with the product model."),
    "error": ("镜头矫正设置失败", "Lens correction settings failed"),
    "reload_error": ("设置已保存，但检测上下文刷新失败；运行触发保持停用，请重启程序后再运行。\n{error}", "Settings were saved, but the inspection context failed to refresh. Runtime triggering remains disabled; restart the application before running.\n{error}"),
    "target_changed": ("该相机的绑定信息已改变，请关闭标定窗口后从对应相机重新打开。", "The camera binding changed. Close this calibration window and reopen it from the intended camera."),
}


def lens_text(key: str, **values) -> str:
    text = _TEXT[key][1 if language_code().lower().startswith("en") else 0]
    return text.format(**values)


def ensure_lens_context(owner, product_dir: str, role: str, expected: str) -> bool:
    from services.lens_correction import active_signature
    try:
        if active_signature(product_dir, role) == expected:
            return True
        message = lens_text("reopen")
    except Exception as exc:
        message = diagnostic_text(exc)
    QtWidgets.QMessageBox.warning(owner, lens_text("title"), message)
    return False


def prepare_editor_image(path: str, product_dir: str, role: str) -> str:
    """External browsing must never mix a managed corrected image with raw assets."""
    from services.lens_correction import SIDECAR_SCHEMA, prepare_path, source_image_path, validate_prepared_path
    source = str(path)
    marker = Path(source).with_suffix(".undistortion.json")
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding="utf-8-sig"))
        if isinstance(metadata, dict) and metadata.get("schema") == SIDECAR_SCHEMA:
            source = source_image_path(source)
    prepared = prepare_path(source, product_dir, role)
    validate_prepared_path(prepared, product_dir, role)
    return prepared


def _can_change_settings(owner) -> bool:
    window = owner.window() if isinstance(owner, QtWidgets.QWidget) else owner
    permission = getattr(window, "_require_permission", None)
    if callable(permission) and not permission("camera.edit_params", lens_text("title")):
        return False
    runtime = getattr(window, "runtime_ctrl", None)
    tool_page = getattr(window, "tool_page", None)
    if tool_page is None and hasattr(owner, "session") and hasattr(owner, "current_camera_role"):
        tool_page = owner
    trigger_future = getattr(runtime, "_trigger_future", None)
    if (bool(getattr(runtime, "_busy", False)) or getattr(runtime, "_di_poller", None) is not None or
            (trigger_future is not None and not trigger_future.done())):
        QtWidgets.QMessageBox.warning(owner, lens_text("title"), lens_text("busy"))
        return False
    if any(bool(getattr(window, name, False)) for name in (
        "_lite_warmup_running", "_lite_reconcile_running", "_lite_file_job_running", "_lite_image_loading_path",
    )):
        QtWidgets.QMessageBox.warning(owner, lens_text("title"), lens_text("busy"))
        return False
    if tool_page is None:
        return True
    if any(bool(getattr(tool_page, name, False)) for name in ("_training_in_progress", "_test_in_progress")):
        QtWidgets.QMessageBox.warning(owner, lens_text("title"), lens_text("busy"))
        return False
    if getattr(tool_page, "_sample_auto_roi_jobs", []):
        QtWidgets.QMessageBox.warning(owner, lens_text("title"), lens_text("busy"))
        return False
    preview = getattr(tool_page, "_sample_annotation_preview_dialog", None)
    autogen = getattr(preview, "_sample_annotation_autogen_dialog", None) if preview is not None else None
    editors = [autogen, preview] + [getattr(tool_page, name, None) for name in ("_template_editor_dialog", "_ncc_workbench_dialog")]
    for editor in editors:
        if editor is not None and (getattr(editor, "_build_thread", None) is not None or
                                   bool(getattr(editor, "_find_running", False)) or
                                   bool(getattr(editor, "_autogen_running", False))):
            QtWidgets.QMessageBox.warning(owner, lens_text("title"), lens_text("busy"))
            return False
    for editor in editors:
        if editor is not None and not editor.close():
            return False
    return True


def _runtime_has_active_work(runtime) -> bool:
    future = getattr(runtime, "_trigger_future", None)
    return bool(getattr(runtime, "_busy", False)) or getattr(runtime, "_di_poller", None) is not None or (future is not None and not future.done())


class LensCorrectionSettingsDialog(QtWidgets.QDialog):
    saved = QtCore.Signal(str, list)

    def __init__(self, owner, product_dir: str, camera_roles, camera_serials=None):
        super().__init__(owner)
        from services.lens_correction import get_config
        from services.lens_camera_roles import resolve_correction_role
        self.product_dir = str(product_dir)
        self.changed = False
        self._owner = owner
        logical_roles = list(dict.fromkeys(str(role) for role in camera_roles)) or ["cam1"]
        physical_roles = {role: resolve_correction_role(self.product_dir, role) for role in logical_roles}
        self._roles = list(dict.fromkeys(physical_roles.values()))
        self._configs = {role: get_config(self.product_dir, role, physical=True) for role in self._roles}
        self._pending_profiles = {}
        self._demo_windows = []
        self.setWindowTitle(lens_text("title"))
        apply_lens_calibration_theme(self)
        self.resize(800, 150 + 34 * min(len(self._roles), 8))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        label = QtWidgets.QLabel(lens_text("product", name=Path(product_dir).name))
        layout.addWidget(label)
        self.table = QtWidgets.QTableWidget(len(self._roles), 5)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels([lens_text(key) for key in ("camera", "enabled", "profile", "mode", "serial")])
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 75)
        self.table.setColumnWidth(1, 65)
        self.table.setColumnWidth(4, 175)
        serials = dict(camera_serials or {})
        for logical, physical in physical_roles.items():
            if not serials.get(physical) and serials.get(logical):
                serials[physical] = serials[logical]
        for row, role in enumerate(self._roles):
            config = self._configs[role]
            role_item = QtWidgets.QTableWidgetItem(role)
            role_item.setFlags(role_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, role_item)
            enabled = QtWidgets.QTableWidgetItem()
            enabled.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable)
            enabled.setCheckState(QtCore.Qt.CheckState.Checked if config.enabled else QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(row, 1, enabled)
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(config.camera_serial or str(serials.get(role, ""))))
            self._show_profile(row, config)
        current_page = getattr(owner, "tool_page", owner)
        selected_role = getattr(current_page, "current_camera_role", lambda: "")()
        selected_physical = physical_roles.get(selected_role, selected_role)
        self.table.selectRow(self._roles.index(selected_physical) if selected_physical in self._roles else 0)
        layout.addWidget(self.table)
        actions = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton(lens_text("import"))
        self.import_button.clicked.connect(self._choose_profile)
        self.demo_button = QtWidgets.QPushButton(lens_text("demo"))
        self.demo_button.clicked.connect(self._open_demo)
        actions.addWidget(self.demo_button)
        actions.addWidget(self.import_button)
        actions.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox()
        self.save_button = buttons.addButton(lens_text("save"), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.save_button.setObjectName("primary")
        self.close_button = buttons.addButton(lens_text("cancel"), QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        actions.addWidget(buttons)
        layout.addLayout(actions)

    def _show_profile(self, row: int, config) -> None:
        for column, text in ((2, config.profile_name or lens_text("none")), (3, self._profile_description(config))):
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(str(config.profile_path or ""))
            self.table.setItem(row, column, item)

    @staticmethod
    def _profile_description(config) -> str:
        if not config.profile_id:
            return "—"
        mode = lens_text(config.mode) if config.mode in _TEXT else str(config.mode)
        size = config.image_size
        return f"{mode} · {size[0]} × {size[1]} px"

    def _choose_profile(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, lens_text("import"), "", "JSON (*.json)")
        if path:
            self.import_profile(row, path)

    def import_profile(self, row: int, path: str) -> bool:
        from services.lens_correction import inspect_profile
        try:
            config = inspect_profile(path, alpha=1.0)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, lens_text("error"), diagnostic_text(exc))
            return False
        self._pending_profiles[self._roles[row]] = str(Path(path).resolve())
        self._show_profile(row, config)
        return True

    def _open_demo(self) -> None:
        from tools.lens_distortion_demo import LensDemoWindow
        row = self.table.currentRow()
        if row < 0:
            return
        role = self._roles[row]
        serial = self.table.item(row, 4).text().strip()
        target = (str(Path(self.product_dir).resolve()), role, serial)
        for existing in self._demo_windows:
            if getattr(existing, "_camera_target", None) == target:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
        window = LensDemoWindow()
        window.setParent(self, QtCore.Qt.WindowType.Window)
        window.set_camera_context(
            product_dir=self.product_dir, camera_role=role, camera_serial=serial,
            apply_callback=lambda profile, alpha, selected=role, bound=serial:
                self._apply_demo_profile(selected, bound, profile, alpha),
        )
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._demo_windows.append(window)
        window.destroyed.connect(lambda *_: self._demo_windows.remove(window) if window in self._demo_windows else None)
        window.show()

    def _apply_demo_profile(self, role, serial, profile, alpha) -> bool:
        from algorithms.lens_calibration import save_profile

        row = self._roles.index(role)
        if self.table.item(row, 4).text().strip() != serial:
            QtWidgets.QMessageBox.warning(self, lens_text("title"), lens_text("target_changed"))
            return False
        # The service snapshots this explicitly selected result into the product.
        # Temporary interchange is internal; no manual JSON export/import is needed.
        with tempfile.TemporaryDirectory(prefix="lc-camera-calibration-") as folder:
            path = Path(folder) / f"{role}_calibration.json"
            save_profile(profile, path)
            return self._commit_payload({role: {
                "enabled": True, "profile_path": str(path), "camera_serial": serial, "alpha": alpha,
            }})

    def done(self, result: int) -> None:
        for window in list(self._demo_windows):
            worker = getattr(window, "worker", None)
            if worker is not None and worker.isRunning():
                QtWidgets.QMessageBox.warning(self, lens_text("title"), lens_text("busy"))
                return
        for window in list(self._demo_windows):
            window.close()
        super().done(result)

    def _save(self) -> None:
        if any(getattr(window, "worker", None) is not None and window.worker.isRunning() for window in self._demo_windows):
            QtWidgets.QMessageBox.warning(self, lens_text("title"), lens_text("busy"))
            return
        payload = {}
        for row, role in enumerate(self._roles):
            config = self._configs[role]
            enabled = self.table.item(row, 1).checkState() == QtCore.Qt.CheckState.Checked
            profile_path = self._pending_profiles.get(role)
            serial = self.table.item(row, 4).text().strip()
            if enabled and not (profile_path or config.profile_id):
                QtWidgets.QMessageBox.warning(self, lens_text("title"), lens_text("missing", role=role))
                return
            if enabled != config.enabled or profile_path or serial != str(config.camera_serial or ""):
                payload[role] = {"enabled": enabled, "profile_path": profile_path,
                                 "camera_serial": serial, "alpha": 1.0 if profile_path else config.alpha}
        if not payload:
            self.accept()
            return
        if self._commit_payload(payload):
            self.accept()

    def _commit_payload(self, payload) -> bool:
        from services.lens_correction import save_configs

        if any(getattr(window, "worker", None) is not None and window.worker.isRunning() for window in self._demo_windows):
            QtWidgets.QMessageBox.warning(self, lens_text("title"), lens_text("busy"))
            return False
        if not _can_change_settings(self._owner):
            return False
        window = self._owner.window() if isinstance(self._owner, QtWidgets.QWidget) else self._owner
        runtime = getattr(window, "runtime_ctrl", None)
        tool_page = getattr(window, "tool_page", None)
        if tool_page is None and hasattr(self._owner, "_reload_lens_correction_context"):
            tool_page = self._owner
        gate = getattr(runtime, "_trigger_lock", None)
        previous_accept = None
        with gate if gate is not None else nullcontext():
            busy = _runtime_has_active_work(runtime)
            if not busy and runtime is not None:
                previous_accept = bool(getattr(runtime, "_accept_trigger_jobs", True))
                runtime._accept_trigger_jobs = False
        if busy:
            QtWidgets.QMessageBox.warning(self, lens_text("title"), lens_text("busy"))
            return False
        committed = False
        refreshed = False
        try:
            configs = save_configs(self.product_dir, payload)
            committed = True
            for role, config in configs.items():
                self._configs[role] = config
                self._pending_profiles.pop(role, None)
                row = self._roles.index(role)
                self.table.item(row, 1).setCheckState(QtCore.Qt.CheckState.Checked if config.enabled else QtCore.Qt.CheckState.Unchecked)
                self.table.item(row, 4).setText(config.camera_serial)
                self._show_profile(row, config)
            if tool_page is not None:
                tool_page._reload_lens_correction_context()
            refreshed = True
        except Exception as exc:
            message = diagnostic_text(exc)
            if committed:
                message = lens_text("reload_error", error=message)
            QtWidgets.QMessageBox.warning(self, lens_text("error"), message)
            return False
        finally:
            if previous_accept is not None and (not committed or refreshed):
                with gate if gate is not None else nullcontext():
                    runtime._accept_trigger_jobs = previous_accept
        self.changed = True
        self.close_button.setText(lens_text("close"))
        self.saved.emit(self.product_dir, list(payload))
        return True


def open_lens_correction_settings(owner, product_dir: str, camera_roles, camera_serials=None) -> bool:
    if not str(product_dir or "").strip():
        QtWidgets.QMessageBox.warning(owner, lens_text("title"), lens_text("choose_product"))
        return False
    if not _can_change_settings(owner):
        return False
    try:
        dialog = LensCorrectionSettingsDialog(owner, product_dir, camera_roles, camera_serials)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(owner, lens_text("error"), diagnostic_text(exc))
        return False
    dialog.exec()
    return dialog.changed
