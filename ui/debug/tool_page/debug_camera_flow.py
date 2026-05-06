"""Debug camera and IO workflow helpers for ToolPage."""

from __future__ import annotations

from . import page as page_module
from ui.i18n import tr

os = page_module.os
cv2 = page_module.cv2
datetime = page_module.datetime
QtCore = page_module.QtCore
QtGui = page_module.QtGui
QtWidgets = page_module.QtWidgets
IoController = page_module.IoController
FrameGrabService = page_module.FrameGrabService
HikCameraManager = page_module.HikCameraManager
HikCameraSettings = page_module.HikCameraSettings
frame_to_bgr_image = page_module.frame_to_bgr_image
hik_settings_kwargs_from_mapping = page_module.hik_settings_kwargs_from_mapping
_DebugCameraPreviewThread = page_module._DebugCameraPreviewThread
_qimage_from_hik_frame = page_module._qimage_from_hik_frame

_DEBUG_IO_NAME_LABELS = {
    "foot_switch": ("debug.io_name.foot_switch", "DI_FOOT_SWITCH"),
    "reject_signal": ("debug.io_name.reject_signal", "DI_REJECT_SIGNAL"),
    "reserved_in_1": ("debug.io_name.reserved_in_1", "DI_RESERVED_1"),
    "reserved_in_2": ("debug.io_name.reserved_in_2", "DI_RESERVED_2"),
    "tower_red": ("debug.io_name.tower_red", "DO_TOWER_RED"),
    "tower_green": ("debug.io_name.tower_green", "DO_TOWER_GREEN"),
    "tower_blue": ("debug.io_name.tower_blue", "DO_TOWER_BLUE"),
    "light_cam1": ("debug.io_name.light_cam1", "DO_LIGHT_CAM1"),
    "light_cam2": ("debug.io_name.light_cam2", "DO_LIGHT_CAM2"),
    "buzzer": ("debug.io_name.buzzer", "DO_BUZZER"),
    "reserved_out_1": ("debug.io_name.reserved_out_1", "DO_RESERVED_1"),
    "reserved_out_2": ("debug.io_name.reserved_out_2", "DO_RESERVED_2"),
}


def _on_debug_camera_param_editing_finished(self) -> None:
    if self._debug_camera_block_spin_apply:
        return
    serial = self._selected_debug_camera_serial()
    if self._debug_camera_device() is None:
        if serial:
            self._save_debug_camera_settings(
                serial, self._debug_camera_settings_payload_from_ui()
            )
        return
    self._apply_debug_camera_settings(quiet=True)


def _on_debug_camera_trigger_activated(self, _index: int) -> None:
    if self._debug_camera_block_spin_apply:
        return
    serial = self._selected_debug_camera_serial()
    if self._debug_camera_device() is None:
        if serial:
            self._save_debug_camera_settings(
                serial, self._debug_camera_settings_payload_from_ui()
            )
        return
    self._apply_debug_camera_settings(quiet=True)


def _save_debug_camera_image(self) -> None:
    pixmap = self.view_debug_camera._pixmap
    if pixmap is None or pixmap.isNull():
        self.lbl_debug_camera_status.setText("Camera status: no image to save")
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        tr("debug.save_image").strip(),
        "",
        "BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)",
    )
    if path:
        if pixmap.save(path):
            self.lbl_debug_camera_status.setText(f"Camera status: image saved to {path}")
        else:
            self.lbl_debug_camera_status.setText("Camera status: failed to save image")


def _toggle_debug_camera_preview(self, checked: bool) -> None:
    if checked:
        self._start_debug_camera_preview()
        return
    self._stop_debug_camera_preview()


def _start_debug_camera_preview(self) -> None:
    if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
        self._set_debug_preview_running(False)
        QtWidgets.QMessageBox.information(self, "Camera Debug", "Connect the debug camera first")
        return
    self._stop_debug_camera_preview()
    worker = _DebugCameraPreviewThread(self._debug_frame_grab_service, self)
    worker.frameReady.connect(self._on_debug_preview_frame_ready)
    worker.errorOccurred.connect(self._on_debug_preview_error)
    worker.finished.connect(self._on_debug_preview_finished)
    self._debug_preview_thread = worker
    self._set_debug_preview_running(True)
    self._set_debug_preview_placeholder("Starting live preview...")
    worker.start()
    self._set_debug_camera_status("Live preview running")


def _stop_debug_camera_preview(self, *, clear_view: bool = True) -> None:
    worker = self._debug_preview_thread
    self._debug_preview_thread = None
    if worker is not None:
        worker.stop()
    device = self._debug_camera_device()
    if device is not None:
        try:
            device.stop_grabbing()
        except Exception:
            pass
    self._set_debug_preview_running(False)
    if clear_view:
        self._set_debug_preview_placeholder(tr("debug.preview_closed"))


@QtCore.Slot(QtGui.QImage)
def _on_debug_preview_frame_ready(self, image: QtGui.QImage) -> None:
    self._show_debug_preview_image(image)


@QtCore.Slot(str)
def _on_debug_preview_error(self, message: str) -> None:
    self._set_debug_camera_status(f"Live preview error: {message}")


@QtCore.Slot()
def _on_debug_preview_finished(self) -> None:
    if self._debug_preview_thread is not None:
        return
    self._set_debug_preview_running(False)


def _set_debug_camera_status(self, message: str) -> None:
    self.lbl_debug_camera_status.setText(f"Camera status: {message}")


def _ensure_debug_camera_services(self) -> bool:
    if FrameGrabService is None or HikCameraManager is None or HikCameraSettings is None:
        QtWidgets.QMessageBox.warning(self, "Camera Debug", "Hik camera debug service is unavailable in the current environment")
        self._set_debug_camera_status("Service unavailable")
        return False
    if self._debug_camera_manager is None:
        self._debug_camera_manager = HikCameraManager()
        self._debug_frame_grab_service = FrameGrabService(self._debug_camera_manager)
    return True


def _refresh_debug_camera_list(self) -> None:
    if not self._ensure_debug_camera_services():
        return
    current_serial = str(self.cmb_debug_camera.currentData() or "").strip()
    preferred_serial = str(
        self._load_debug_role_binding(self._selected_debug_camera_role()) or ""
    ).strip()
    try:
        infos = self._debug_camera_manager.enumerate_cameras()
    except Exception as exc:
        QtWidgets.QMessageBox.warning(self, "Camera Debug", f"Failed to scan cameras: {exc}")
        self._set_debug_camera_status(f"Scan failed: {exc}")
        return
    self._debug_camera_infos = list(infos)
    self.cmb_debug_camera.blockSignals(True)
    try:
        self.cmb_debug_camera.clear()
        for info in infos:
            label = f"{info.serial_number} / {info.model_name or 'UnknownModel'} / {info.transport_layer}"
            self.cmb_debug_camera.addItem(label, info.serial_number)
        target_serial = preferred_serial or current_serial
        if target_serial:
            index = self.cmb_debug_camera.findData(target_serial)
            if index >= 0:
                self.cmb_debug_camera.setCurrentIndex(index)
    finally:
        self.cmb_debug_camera.blockSignals(False)
    self._refresh_debug_camera_info()
    self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())
    self._set_debug_camera_status(f"Scanned {len(infos)} camera(s)")


def _on_debug_camera_role_changed(self) -> None:
    self._set_current_camera_role(self._selected_debug_camera_role(), sync_debug_role=False)
    connected_serial = str(getattr(self._debug_camera_device(), "serial_number", "") or "").strip()
    self._refresh_debug_role_status()
    self._apply_debug_role_binding_to_camera_combo()
    self._refresh_debug_camera_info()
    self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())
    selected_serial = self._selected_debug_camera_serial()
    if connected_serial and connected_serial != selected_serial:
        self._disconnect_debug_camera()
        self._set_debug_camera_status(f"Switched to {self._selected_debug_camera_role()}, reconnect required")


def _on_debug_camera_selected(self) -> None:
    self._refresh_debug_camera_info()
    serial = self._selected_debug_camera_serial()
    if serial:
        self._save_debug_role_binding(self._selected_debug_camera_role(), serial)
    self._load_saved_debug_camera_settings_to_ui(serial)


def _refresh_debug_camera_info(self) -> None:
    info = self._selected_debug_camera_info()
    if info is None:
        self.lbl_debug_camera_info.setText(f"{tr('debug.camera_info')} -")
        self._refresh_debug_role_status()
        return
    self.lbl_debug_camera_info.setText(
        f"{tr('debug.camera_info')} "
        + f"Serial={info.serial_number}  "
        + f"Model={info.model_name or '-'}  "
        + f"Name={info.user_defined_name or '-'}  "
        + f"Vendor={info.manufacturer_name or '-'}  "
        + f"Transport={info.transport_layer}"
    )
    self._refresh_debug_role_status()


def _connect_debug_camera(self) -> None:
    if not self._ensure_debug_camera_services():
        return
    serial = self._selected_debug_camera_serial()
    if not serial:
        QtWidgets.QMessageBox.information(self, "Camera Debug", "Scan and select a camera first")
        return
    role = self._selected_debug_camera_role()
    self.debugCameraConnectRequested.emit(serial)
    try:
        saved_settings = self._camera_settings_store.load_for_role(role, serial=serial)
        settings = {
            "debug": HikCameraSettings(
                **hik_settings_kwargs_from_mapping(
                    saved_settings,
                    default_trigger_mode="continuous",
                )
            )
        }
        self._debug_frame_grab_service.open_bound_cameras({"debug": serial}, settings_by_role=settings)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "Camera Debug", f"Failed to connect debug camera: {exc}")
        self._set_debug_camera_status(f"Connection failed: {exc}")
        return
    self._save_debug_role_binding(role, serial)
    self._refresh_debug_camera_settings()
    self._set_debug_preview_placeholder("Debug camera connected; live preview is available")
    self._set_debug_camera_status(f"Connected {role} debug camera: {serial}")
    self.debugCameraConnected.emit(role, serial)


def _disconnect_debug_camera(self) -> None:
    self._stop_debug_camera_preview()
    if self._debug_frame_grab_service is not None:
        try:
            self._debug_frame_grab_service.close_all()
        except Exception:
            pass
    if self._debug_camera_manager is not None:
        try:
            self._debug_camera_manager.close()
        except Exception:
            pass
    self._debug_camera_manager = None
    self._debug_frame_grab_service = None
    self.lbl_debug_camera_info.setText(f"{tr('debug.camera_info')} -")
    self._set_debug_preview_placeholder("Debug camera disconnected")
    self._set_debug_camera_status("Disconnected")
    self._refresh_debug_role_status()


def _refresh_debug_camera_settings(self) -> None:
    device = self._debug_camera_device()
    if device is None:
        return
    self._debug_camera_block_spin_apply = True
    try:
        try:
            exposure = float(device.get_float_value("ExposureTime"))
        except Exception:
            exposure = float(self.spin_debug_exposure.value())
        try:
            gain = float(device.get_float_value("Gain"))
        except Exception:
            gain = float(self.spin_debug_gain.value())
        try:
            digital_shift_enable = bool(device.get_bool_value("DigitalShiftEnable"))
        except Exception:
            digital_shift_enable = bool(self.chk_debug_digital_shift_enable.isChecked())
        try:
            digital_shift = float(device.get_float_value("DigitalShift"))
        except Exception:
            digital_shift = float(self.spin_debug_digital_shift.value())
        self.spin_debug_exposure.setValue(exposure)
        self.spin_debug_gain.setValue(gain)
        self.chk_debug_digital_shift_enable.setChecked(digital_shift_enable)
        self.spin_debug_digital_shift.setValue(digital_shift)
        try:
            trigger_mode_int = int(device.get_int_value("TriggerMode"))
            self.cmb_debug_trigger_mode.setCurrentText(
                "software" if trigger_mode_int else "continuous"
            )
        except Exception:
            pass
        serial = getattr(device, "serial_number", self._selected_debug_camera_serial())
        self._save_debug_camera_settings(
            serial,
            self._debug_camera_settings_payload_from_ui(),
        )
        self._save_debug_role_binding(
            self._selected_debug_camera_role(),
            serial,
        )
    finally:
        self._debug_camera_block_spin_apply = False


def _apply_debug_camera_settings(self, *, quiet: bool = False) -> None:
    device = self._debug_camera_device()
    if device is None:
        if not quiet:
            QtWidgets.QMessageBox.information(self, "Camera Debug", "Connect the debug camera first")
        return
    try:
        device.apply_settings(
            HikCameraSettings(
                trigger_mode=str(self.cmb_debug_trigger_mode.currentText() or "continuous"),
                exposure_time_us=float(self.spin_debug_exposure.value()),
                gain=float(self.spin_debug_gain.value()),
                digital_shift_enable=bool(self.chk_debug_digital_shift_enable.isChecked()),
                digital_shift=float(self.spin_debug_digital_shift.value()),
            )
        )
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "Camera Debug", f"Failed to apply camera parameters: {exc}")
        return
    serial = str(getattr(device, "serial_number", self._selected_debug_camera_serial()) or "").strip()
    payload = dict(self._debug_camera_settings_payload_from_ui())
    self._save_debug_camera_settings(serial, payload)
    self._save_debug_role_binding(self._selected_debug_camera_role(), serial)
    if not quiet:
        self.cameraSettingsApplied.emit(serial, payload)
    self._refresh_debug_camera_settings()
    self._set_debug_camera_status("Parameters written to camera" if quiet else "Camera parameters applied")


def _grab_debug_camera_once(self) -> None:
    if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
        QtWidgets.QMessageBox.information(self, "Camera Debug", "Connect the debug camera first")
        return
    try:
        frame = self._debug_frame_grab_service.capture_once("debug", timeout_ms=1000)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "Camera Debug", f"Capture failed: {exc}")
        self._set_debug_camera_status(f"Capture failed: {exc}")
        return
    try:
        self._show_debug_preview_image(_qimage_from_hik_frame(frame))
    except Exception:
        pass

    capture_dir = os.path.join(self.session.product_dir, "debug_capture")
    os.makedirs(capture_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    role = self._selected_debug_camera_role()
    image_path = os.path.join(capture_dir, f"{role}_debug_cam_{stamp}.png")
    if frame_to_bgr_image is None:
        QtWidgets.QMessageBox.critical(self, "Camera Debug", "Camera image conversion service is unavailable")
        return
    image = frame_to_bgr_image(frame)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if not cv2.imwrite(image_path, image):
        QtWidgets.QMessageBox.critical(self, "Camera Debug", f"Failed to save captured image: {image_path}")
        return
    if image_path not in self.test_files:
        self.test_files.append(image_path)
        self._refresh_lists()
        self._save_session()
    self.tabs.setCurrentIndex(1)
    self._load_canvas_image(image_path)
    self._set_debug_camera_status(f"{role} capture saved: {os.path.basename(image_path)}")
    self.lbl_status.setText(f"Status: {role} debug capture saved to test samples -> {os.path.basename(image_path)}")


def _debug_io_display_name(name: str, channel: int | None = None) -> str:
    if str(name) == "reserved_out_1" and channel == 5:
        return f"{tr('debug.io_name.buzzer')}\nDO_BUZZER"
    entry = _DEBUG_IO_NAME_LABELS.get(str(name))
    if entry is None:
        return str(name)
    key, code = entry
    return f"{tr(key)}\n{code}"


def _debug_io_channel_maps(io_controller):
    di_map = {}
    do_map = {}
    mapping = getattr(io_controller, "mapping", None)
    if mapping is None:
        return di_map, do_map
    for name in mapping.di_names():
        cfg = mapping.get_input(name)
        di_map[int(cfg.channel)] = (name, cfg)
    for name in mapping.do_names():
        cfg = mapping.get_output(name)
        do_map[int(cfg.channel)] = (name, cfg)
    return di_map, do_map


def _set_debug_di_indicator(indicator, active: bool) -> None:
    indicator.setStyleSheet(
        "background:#2fbf71;border:2px solid #86efac;border-radius:16px;"
        if active
        else "background:#7a7a7a;border:2px solid #9a9a9a;border-radius:16px;"
    )


def _reset_debug_io_panels(self) -> None:
    self._debug_output_buttons.clear()
    for channel, card in self._debug_di_cards.items():
        card.setVisible(False)
    for channel, indicator in self._debug_di_indicators.items():
        _set_debug_di_indicator(indicator, False)
        indicator.setToolTip(f"DI_{channel}\nDisconnected")
    for channel, hint in self._debug_di_hints.items():
        hint.setText(tr("debug.unmapped"))
        hint.setToolTip(f"DI_{channel}\nUnmapped")
    for channel, card in self._debug_do_cards.items():
        card.setVisible(False)
    for channel, button in self._debug_do_channel_buttons.items():
        button.blockSignals(True)
        button.setChecked(False)
        button.setEnabled(False)
        button.blockSignals(False)
        button.setToolTip(f"DO_{channel}\nDisconnected")
    for channel, hint in self._debug_do_hints.items():
        hint.setText(tr("debug.unmapped"))
        hint.setToolTip(f"DO_{channel}\nUnmapped")
    self.lbl_debug_io_mapping_summary.setText(tr("debug.mapping_not_loaded"))


def _update_debug_io_panels(self, di_word: int, do_word: int) -> None:
    di_map, do_map = _debug_io_channel_maps(self._debug_io_controller)
    self._debug_output_buttons.clear()

    for channel, indicator in self._debug_di_indicators.items():
        mapped = di_map.get(channel)
        self._debug_di_cards[channel].setVisible(mapped is not None)
        if mapped is None:
            continue

        name, cfg = mapped
        raw_state = bool(int(di_word) & (1 << channel))
        display_state = raw_state if cfg.active_high else not raw_state
        tooltip = (
            f"DI_{channel}\n"
            f"name: {name}\n"
            f"active: {'high' if cfg.active_high else 'low'}\n"
            f"level: {'HIGH' if raw_state else 'LOW'}\n"
            f"logic: {'ON' if display_state else 'OFF'}"
        )
        _set_debug_di_indicator(indicator, display_state)
        indicator.setToolTip(tooltip)
        hint = self._debug_di_hints[channel]
        hint.setText(_debug_io_display_name(name, channel))
        hint.setToolTip(tooltip)

    for channel, button in self._debug_do_channel_buttons.items():
        mapped = do_map.get(channel)
        self._debug_do_cards[channel].setVisible(mapped is not None)
        if mapped is None:
            button.blockSignals(True)
            button.setEnabled(False)
            button.setChecked(False)
            button.blockSignals(False)
            continue

        name, cfg = mapped
        raw_state = bool(int(do_word) & (1 << channel))
        display_state = raw_state if cfg.active_high else not raw_state
        tooltip = (
            f"DO_{channel}\n"
            f"name: {name}\n"
            f"active: {'high' if cfg.active_high else 'low'}\n"
            f"level: {'HIGH' if raw_state else 'LOW'}\n"
            f"logic: {'ON' if display_state else 'OFF'}"
        )
        self._debug_output_buttons[name] = button
        button.blockSignals(True)
        button.setEnabled(True)
        button.setChecked(display_state)
        button.blockSignals(False)
        button.setToolTip(tooltip)
        hint = self._debug_do_hints[channel]
        hint.setText(_debug_io_display_name(name, channel))
        hint.setToolTip(tooltip)

    input_states = []
    for channel, (name, cfg) in sorted(di_map.items()):
        raw_state = bool(int(di_word) & (1 << channel))
        state = raw_state if cfg.active_high else not raw_state
        input_states.append(f"{name}={'ON' if state else 'OFF'}")

    output_states = []
    for channel, (name, cfg) in sorted(do_map.items()):
        raw_state = bool(int(do_word) & (1 << channel))
        state = raw_state if cfg.active_high else not raw_state
        output_states.append(f"{name}={'ON' if state else 'OFF'}")

    self.lbl_debug_di_snapshot.setText(
        f"DI 0x{int(di_word):04X}" + (f"  {'  '.join(input_states)}" if input_states else "")
    )
    self.lbl_debug_do_snapshot.setText(
        f"DO 0x{int(do_word):04X}" + (f"  {'  '.join(output_states)}" if output_states else "")
    )
    enabled_channels = ", ".join(f"DO_{channel}" for channel in sorted(do_map)) or "-"
    self.lbl_debug_io_mapping_summary.setText(
        f"Mapping: DI {len(di_map)} / DO {len(do_map)}; enabled output channels {enabled_channels}"
    )


def _open_debug_io(self) -> None:
    runtime_ctrl = self.runtime_controller()
    if runtime_ctrl is None:
        QtWidgets.QMessageBox.warning(self, "DI/DO Debug", "Runtime IO service is unavailable in the current environment")
        return

    if getattr(self, "_runtime_io_ready", False) and getattr(self, "_runtime_io_controller", None) is not None:
        self._apply_runtime_io_debug_state()
        self.lbl_status.setText("Status: DI/DO debug attached to runtime IO")
        return

    if not runtime_ctrl.initialize_startup_io(force=True):
        detail = getattr(self, "_runtime_io_status_detail", "") or "unknown error"
        QtWidgets.QMessageBox.critical(self, "DI/DO Debug", f"Failed to open IO debug: {detail}")
        return

    self._apply_runtime_io_debug_state()
    self.lbl_status.setText("Status: DI/DO debug reloaded from runtime IO")


def _close_debug_io(self, *, silent: bool = False) -> None:
    self._debug_io_timer.stop()
    if self._debug_io_controller is not None and not getattr(self, "_debug_io_uses_runtime_controller", False):
        try:
            self._debug_io_controller.clear_outputs()
        except Exception:
            pass
        try:
            self._debug_io_controller.close()
        except Exception:
            pass
    if getattr(self, "_debug_io_uses_runtime_controller", False):
        self._debug_io_uses_runtime_controller = False
    self._debug_io_controller = None
    self.lbl_debug_di_snapshot.setText(tr("debug.di_disconnected"))
    self.lbl_debug_do_snapshot.setText(tr("debug.do_disconnected"))
    _reset_debug_io_panels(self)
    if getattr(self, "_runtime_io_ready", False) and getattr(self, "_runtime_io_controller", None) is not None:
        self._apply_runtime_io_debug_state()
    if not silent:
        self.lbl_status.setText("Status: DI/DO debug closed")


def _refresh_debug_io_snapshot(self) -> None:
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        return
    try:
        di_word = self._debug_io_controller.board.read_di_word()
        do_word = self._debug_io_controller.board.read_do_word()
    except Exception as exc:
        self.lbl_debug_di_snapshot.setText(f"DI read failed ({exc})")
        self.lbl_debug_do_snapshot.setText(f"DO read failed ({exc})")
        self.lbl_debug_io_mapping_summary.setText("Mapping: read failed")
        return
    _update_debug_io_panels(self, di_word, do_word)


def _set_debug_output_channel(self, channel: int, on: bool) -> None:
    button = self._debug_do_channel_buttons.get(int(channel))
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        QtWidgets.QMessageBox.information(self, "DI/DO Debug", "Open IO debug first")
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        return

    _di_map, do_map = _debug_io_channel_maps(self._debug_io_controller)
    mapped = do_map.get(int(channel))
    if mapped is None:
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        QtWidgets.QMessageBox.information(self, "DI/DO Debug", f"DO_{channel} is not mapped and cannot be written")
        return

    name, _cfg = mapped
    try:
        self._debug_io_controller.set_output(name, on)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "DI/DO Debug", f"{name} output failed: {exc}")
    self._refresh_debug_io_snapshot()


def _set_debug_output(self, name: str, on: bool) -> None:
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        QtWidgets.QMessageBox.information(self, "DI/DO Debug", "Open IO debug first")
        return
    _di_map, do_map = _debug_io_channel_maps(self._debug_io_controller)
    for channel, (mapped_name, _cfg) in do_map.items():
        if mapped_name == name:
            self._set_debug_output_channel(channel, on)
            return
    QtWidgets.QMessageBox.information(self, "DI/DO Debug", f"{name} has no mapped output channel")
