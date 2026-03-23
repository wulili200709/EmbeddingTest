"""Debug camera and IO workflow helpers for ToolPage."""

from __future__ import annotations

from . import page as page_module

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


def _on_debug_camera_param_editing_finished(self) -> None:
    """曝光/增益编辑结束：已连接则写入相机，否则仅写入本地缓存。"""
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
        self.lbl_debug_camera_status.setText("相机状态：没有可保存的图片")
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self, "保存图片", "", "BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件 (*)"
    )
    if path:
        if pixmap.save(path):
            self.lbl_debug_camera_status.setText(f"相机状态：图片已保存到 {path}")
        else:
            self.lbl_debug_camera_status.setText("相机状态：保存图片失败")

def _toggle_debug_camera_preview(self, checked: bool) -> None:
    if checked:
        self._start_debug_camera_preview()
        return
    self._stop_debug_camera_preview()

def _start_debug_camera_preview(self) -> None:
    if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
        self._set_debug_preview_running(False)
        QtWidgets.QMessageBox.information(self, "相机调试", "请先连接调试相机")
        return
    self._stop_debug_camera_preview()
    worker = _DebugCameraPreviewThread(self._debug_frame_grab_service, self)
    worker.frameReady.connect(self._on_debug_preview_frame_ready)
    worker.errorOccurred.connect(self._on_debug_preview_error)
    worker.finished.connect(self._on_debug_preview_finished)
    self._debug_preview_thread = worker
    self._set_debug_preview_running(True)
    self._set_debug_preview_placeholder("实时预览启动中...")
    worker.start()
    self._set_debug_camera_status("实时预览中")

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
        self._set_debug_preview_placeholder("未开启实时预览")

@QtCore.Slot(QtGui.QImage)
def _on_debug_preview_frame_ready(self, image: QtGui.QImage) -> None:
    self._show_debug_preview_image(image)

@QtCore.Slot(str)
def _on_debug_preview_error(self, message: str) -> None:
    self._set_debug_camera_status(f"实时预览异常：{message}")

@QtCore.Slot()
def _on_debug_preview_finished(self) -> None:
    if self._debug_preview_thread is not None:
        return
    self._set_debug_preview_running(False)

def _set_debug_camera_status(self, message: str) -> None:
    self.lbl_debug_camera_status.setText(f"相机状态：{message}")

def _ensure_debug_camera_services(self) -> bool:
    if FrameGrabService is None or HikCameraManager is None or HikCameraSettings is None:
        QtWidgets.QMessageBox.warning(self, "相机调试", "当前环境未启用海康相机调试服务")
        self._set_debug_camera_status("服务不可用")
        return False
    if self._debug_camera_manager is None:
        self._debug_camera_manager = HikCameraManager()
        self._debug_frame_grab_service = FrameGrabService(self._debug_camera_manager)
    return True

def _refresh_debug_camera_list(self) -> None:
    if not self._ensure_debug_camera_services():
        return
    current_serial = str(self.cmb_debug_camera.currentData() or "")
    try:
        infos = self._debug_camera_manager.enumerate_cameras()
    except Exception as exc:
        QtWidgets.QMessageBox.warning(self, "相机调试", f"扫描相机失败：{exc}")
        self._set_debug_camera_status(f"扫描失败：{exc}")
        return
    self._debug_camera_infos = list(infos)
    self.cmb_debug_camera.clear()
    for info in infos:
        label = f"{info.serial_number} / {info.model_name or 'UnknownModel'} / {info.transport_layer}"
        self.cmb_debug_camera.addItem(label, info.serial_number)
    if current_serial:
        index = self.cmb_debug_camera.findData(current_serial)
        if index >= 0:
            self.cmb_debug_camera.setCurrentIndex(index)
    self._refresh_debug_camera_info()
    self._set_debug_camera_status(f"已扫描到 {len(infos)} 台相机")

def _on_debug_camera_selected(self) -> None:
    self._refresh_debug_camera_info()
    self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())

def _refresh_debug_camera_info(self) -> None:
    info = self._selected_debug_camera_info()
    if info is None:
        self.lbl_debug_camera_info.setText("相机信息：-")
        return
    self.lbl_debug_camera_info.setText(
        "相机信息："
        + f"序列号={info.serial_number}  "
        + f"型号={info.model_name or '-'}  "
        + f"名称={info.user_defined_name or '-'}  "
        + f"厂商={info.manufacturer_name or '-'}  "
        + f"传输层={info.transport_layer}"
    )

def _connect_debug_camera(self) -> None:
    if not self._ensure_debug_camera_services():
        return
    serial = self._selected_debug_camera_serial()
    if not serial:
        QtWidgets.QMessageBox.information(self, "相机调试", "请先扫描并选择一台相机")
        return
    self.debugCameraConnectRequested.emit(serial)
    try:
        saved_settings = self._camera_settings_store.load_for_serial(serial)
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
        QtWidgets.QMessageBox.critical(self, "相机调试", f"连接调试相机失败：{exc}")
        self._set_debug_camera_status(f"连接失败：{exc}")
        return
    self._refresh_debug_camera_settings()
    self._set_debug_preview_placeholder("已连接调试相机，可开启实时预览")
    self._set_debug_camera_status(f"已连接调试相机：{serial}")

    self.debugCameraConnected.emit(serial)

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
    self.lbl_debug_camera_info.setText("相机信息：-")
    self._set_debug_preview_placeholder("未连接调试相机")
    self._set_debug_camera_status("已断开")

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
        self.spin_debug_exposure.setValue(exposure)
        self.spin_debug_gain.setValue(gain)
        try:
            trigger_mode_int = int(device.get_int_value("TriggerMode"))
            self.cmb_debug_trigger_mode.setCurrentText(
                "software" if trigger_mode_int else "continuous"
            )
        except Exception:
            pass
        self._save_debug_camera_settings(
            getattr(device, "serial_number", self._selected_debug_camera_serial()),
            self._debug_camera_settings_payload_from_ui(),
        )
    finally:
        self._debug_camera_block_spin_apply = False

def _apply_debug_camera_settings(self, *, quiet: bool = False) -> None:
    device = self._debug_camera_device()
    if device is None:
        if not quiet:
            QtWidgets.QMessageBox.information(self, "相机调试", "请先连接调试相机")
        return
    try:
        device.apply_settings(
            HikCameraSettings(
                trigger_mode=str(self.cmb_debug_trigger_mode.currentText() or "continuous"),
                exposure_time_us=float(self.spin_debug_exposure.value()),
                gain=float(self.spin_debug_gain.value()),
            )
        )
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "相机调试", f"应用相机参数失败：{exc}")
        return
    self._save_debug_camera_settings(
        getattr(device, "serial_number", self._selected_debug_camera_serial()),
        self._debug_camera_settings_payload_from_ui(),
    )
    if not quiet:
        self.cameraSettingsApplied.emit(
            str(getattr(device, "serial_number", self._selected_debug_camera_serial()) or "").strip(),
            dict(self._debug_camera_settings_payload_from_ui()),
        )
    self._refresh_debug_camera_settings()
    self._set_debug_camera_status(
        "参数已写入相机" if quiet else "相机参数已应用"
    )

def _grab_debug_camera_once(self) -> None:
    if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
        QtWidgets.QMessageBox.information(self, "相机调试", "请先连接调试相机")
        return
    try:
        frame = self._debug_frame_grab_service.capture_once("debug", timeout_ms=1000)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "相机调试", f"拍照失败：{exc}")
        self._set_debug_camera_status(f"拍照失败：{exc}")
        return
    try:
        self._show_debug_preview_image(_qimage_from_hik_frame(frame))
    except Exception:
        pass

    capture_dir = os.path.join(self.session.product_dir, "debug_capture")
    os.makedirs(capture_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(capture_dir, f"debug_cam_{stamp}.png")
    if frame_to_bgr_image is None:
        QtWidgets.QMessageBox.critical(self, "鐩告満璋冭瘯", "鐩告満褰╄壊杞崲鏈嶅姟涓嶅彲鐢?")
        return
    image = frame_to_bgr_image(frame)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if not cv2.imwrite(image_path, image):
        QtWidgets.QMessageBox.critical(self, "相机调试", f"保存采图失败：{image_path}")
        return
    if image_path not in self.test_files:
        self.test_files.append(image_path)
        self._refresh_lists()
        self._save_session()
    self.tabs.setCurrentIndex(2)
    self._load_canvas_image(image_path)
    self._set_debug_camera_status(f"拍照成功：{os.path.basename(image_path)}")
    self.lbl_status.setText(f"状态：调试拍照已保存到 TEST -> {os.path.basename(image_path)}")

def _open_debug_io(self) -> None:
    if IoController is None:
        QtWidgets.QMessageBox.warning(self, "DI/DO 调试", "当前环境未启用 IO 调试服务")
        return
    mapping_path = self._default_io_mapping_path()
    board_config = self._find_debug_nkio_config_path()
    if not board_config:
        QtWidgets.QMessageBox.warning(self, "DI/DO 调试", "未找到 nkio_config.ini")
        return
    try:
        self._close_debug_io(silent=True)
        self._debug_io_controller = IoController.from_config_file(board_config, mapping_path)
        self._debug_io_controller.open()
    except Exception as exc:
        self._debug_io_controller = None
        QtWidgets.QMessageBox.critical(self, "DI/DO 调试", f"打开 IO 调试失败：{exc}")
        return
    self._debug_io_timer.start()
    self._refresh_debug_io_snapshot()
    self.lbl_status.setText("状态：DI/DO 调试已打开")

def _close_debug_io(self, *, silent: bool = False) -> None:
    self._debug_io_timer.stop()
    if self._debug_io_controller is not None:
        try:
            self._debug_io_controller.clear_outputs()
        except Exception:
            pass
        try:
            self._debug_io_controller.close()
        except Exception:
            pass
    self._debug_io_controller = None
    self.lbl_debug_di_snapshot.setText("DI：未连接")
    self.lbl_debug_do_snapshot.setText("DO：未连接")
    for button in self._debug_output_buttons.values():
        button.blockSignals(True)
        button.setChecked(False)
        button.blockSignals(False)
    if not silent:
        self.lbl_status.setText("状态：DI/DO 调试已关闭")

def _refresh_debug_io_snapshot(self) -> None:
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        return
    try:
        inputs = self._debug_io_controller.snapshot_inputs()
        outputs = self._debug_io_controller.snapshot_outputs()
    except Exception as exc:
        self.lbl_debug_di_snapshot.setText(f"DI：读取失败 ({exc})")
        self.lbl_debug_do_snapshot.setText(f"DO：读取失败 ({exc})")
        return
    self.lbl_debug_di_snapshot.setText(
        "DI：" + "  ".join(f"{name}={'ON' if state else 'OFF'}" for name, state in sorted(inputs.items()))
    )
    self.lbl_debug_do_snapshot.setText(
        "DO：" + "  ".join(f"{name}={'ON' if state else 'OFF'}" for name, state in sorted(outputs.items()))
    )
    for name, button in self._debug_output_buttons.items():
        state = bool(outputs.get(name, False))
        button.blockSignals(True)
        button.setChecked(state)
        button.blockSignals(False)

def _set_debug_output(self, name: str, on: bool) -> None:
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        QtWidgets.QMessageBox.information(self, "DI/DO 调试", "请先打开 IO 调试")
        button = self._debug_output_buttons.get(name)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        return
    try:
        self._debug_io_controller.set_output(name, on)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "DI/DO 调试", f"{name} 输出失败：{exc}")
    self._refresh_debug_io_snapshot()
