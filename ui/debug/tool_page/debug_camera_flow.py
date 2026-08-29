"""Debug camera workflow helpers for ToolPage."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import Iterator

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from infrastructure.camera_settings_store import hik_settings_kwargs_from_mapping
from ui.debug.tool_page.debug_camera_runtime import (
    DebugCameraPreviewThread,
    FrameGrabService,
    HikCameraManager,
    HikCameraSettings,
    frame_to_bgr_image,
    qimage_from_hik_frame,
)
from ui.i18n import tr


_DEBUG_CAMERA_LIGHT_OUTPUT_NAMES = tuple(f"light_cam{index}" for index in range(1, 4))


@contextmanager
def _temporary_debug_camera_light(io_controller, light_index: int) -> Iterator[None]:
    """Select one capture light, then restore every camera light to its prior state."""
    previous_states = {
        name: bool(io_controller.read_output(name))
        for name in _DEBUG_CAMERA_LIGHT_OUTPUT_NAMES
    }
    try:
        io_controller.set_outputs(
            {
                name: index == int(light_index)
                for index, name in enumerate(_DEBUG_CAMERA_LIGHT_OUTPUT_NAMES, start=1)
            }
        )
        yield
    finally:
        io_controller.set_outputs(previous_states)


def _require_debug_camera_param_permission(self) -> bool:
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission):
        return bool(require_permission("camera.edit_params", "修改相机参数"))
    return True


def _on_debug_camera_param_editing_finished(self) -> None:
    if self._debug_camera_block_spin_apply:
        return
    if not _require_debug_camera_param_permission(self):
        self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())
        return
    serial = self._selected_debug_camera_serial()
    payload = self._debug_camera_settings_payload_from_ui()
    if self._debug_camera_device() is None:
        if serial:
            self._save_debug_camera_settings(serial, payload)
            self.cameraSettingsApplied.emit(serial, payload)
        return
    self._apply_debug_camera_settings(quiet=True)
    serial = self._selected_debug_camera_serial()
    if serial:
        self.cameraSettingsApplied.emit(serial, self._debug_camera_settings_payload_from_ui())


def _on_debug_camera_trigger_activated(self, _index: int) -> None:
    if self._debug_camera_block_spin_apply:
        return
    if not _require_debug_camera_param_permission(self):
        self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())
        return
    serial = self._selected_debug_camera_serial()
    payload = self._debug_camera_settings_payload_from_ui()
    if self._debug_camera_device() is None:
        if serial:
            self._save_debug_camera_settings(serial, payload)
            self.cameraSettingsApplied.emit(serial, payload)
        return
    self._apply_debug_camera_settings(quiet=True)
    serial = self._selected_debug_camera_serial()
    if serial:
        self.cameraSettingsApplied.emit(serial, self._debug_camera_settings_payload_from_ui())


def _save_debug_camera_image(self) -> None:
    pixmap = self.view_debug_camera._pixmap
    if pixmap is None or pixmap.isNull():
        self.lbl_debug_camera_status.setText(tr("debug.camera_dialog.status_no_image"))
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        tr("debug.save_image").strip(),
        "",
        "BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)",
    )
    if path:
        if pixmap.save(path):
            self.lbl_debug_camera_status.setText(
                tr("debug.camera_dialog.status_saved", path=path)
            )
        else:
            self.lbl_debug_camera_status.setText(tr("debug.camera_dialog.status_save_failed"))


def _toggle_debug_camera_preview(self, checked: bool) -> None:
    if checked:
        self._start_debug_camera_preview()
        return
    self._stop_debug_camera_preview()


def _start_debug_camera_preview(self) -> None:
    if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
        self._set_debug_preview_running(False)
        QtWidgets.QMessageBox.information(
            self, tr("debug.camera_dialog.title"), tr("debug.camera_dialog.connect_first")
        )
        return
    self._stop_debug_camera_preview()
    worker = DebugCameraPreviewThread(self._debug_frame_grab_service, self)
    worker.frameReady.connect(self._on_debug_preview_frame_ready)
    worker.errorOccurred.connect(self._on_debug_preview_error)
    worker.finished.connect(self._on_debug_preview_finished)
    self._debug_preview_thread = worker
    self._set_debug_preview_running(True)
    self._set_debug_preview_placeholder(tr("debug.camera_dialog.preview_starting"))
    worker.start()
    self._set_debug_camera_status(tr("debug.camera_dialog.preview_running"))


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
    self._set_debug_camera_status(
        tr("debug.camera_dialog.preview_error", error=message)
    )


@QtCore.Slot()
def _on_debug_preview_finished(self) -> None:
    if self._debug_preview_thread is not None:
        return
    self._set_debug_preview_running(False)


def _set_debug_camera_status(self, message: str) -> None:
    self.lbl_debug_camera_status.setText(
        tr("debug.camera_dialog.status", message=message)
    )


def _ensure_debug_camera_services(self) -> bool:
    if FrameGrabService is None or HikCameraManager is None or HikCameraSettings is None:
        QtWidgets.QMessageBox.warning(
            self, tr("debug.camera_dialog.title"), tr("debug.camera_dialog.unavailable")
        )
        self._set_debug_camera_status(tr("debug.camera_dialog.service_unavailable"))
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
        QtWidgets.QMessageBox.warning(
            self,
            tr("debug.camera_dialog.title"),
            tr("debug.camera_dialog.scan_failed", error=exc),
        )
        self._set_debug_camera_status(
            tr("debug.camera_dialog.scan_failed_status", error=exc)
        )
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
    self._set_debug_camera_status(
        tr("debug.camera_dialog.scanned", count=len(infos))
    )


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
        self._set_debug_camera_status(
            tr(
                "debug.camera_dialog.switched_reconnect",
                role=self._selected_debug_camera_role(),
            )
        )


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
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("runtime.connect_camera", "连接调试相机"):
        return
    if not self._ensure_debug_camera_services():
        return
    serial = self._selected_debug_camera_serial()
    if not serial:
        QtWidgets.QMessageBox.information(
            self, tr("debug.camera_dialog.title"), tr("debug.camera_dialog.scan_first")
        )
        return
    role = self._selected_debug_camera_role()
    physical_role = self._debug_physical_camera_role(role)
    channel = self._debug_capture_channel_for_role(role)
    self.debugCameraConnectRequested.emit(serial)
    try:
        saved_settings = dict(
            self._camera_settings_store.load_for_role(physical_role, serial=serial) or {}
        )
        if channel:
            saved_settings["exposure_time_us"] = float(channel.get("exposure_time_us", 5000.0) or 5000.0)
            saved_settings["gain"] = float(channel.get("gain", 0.0) or 0.0)
            saved_settings["light_source_mode"] = "board_io"
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
        QtWidgets.QMessageBox.critical(
            self,
            tr("debug.camera_dialog.title"),
            tr("debug.camera_dialog.connect_failed", error=exc),
        )
        self._set_debug_camera_status(
            tr("debug.camera_dialog.connection_failed", error=exc)
        )
        return
    self._save_debug_role_binding(role, serial)
    self._refresh_debug_camera_settings()
    self._set_debug_preview_placeholder(tr("debug.camera_dialog.connected_preview"))
    self._set_debug_camera_status(
        tr(
            "debug.camera_dialog.connected",
            role=role,
            physical=physical_role,
            serial=serial,
        )
    )
    self.debugCameraConnected.emit(physical_role, serial)


def _disconnect_debug_camera_requested(self) -> None:
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("runtime.connect_camera", "断开调试相机"):
        return
    self._disconnect_debug_camera()


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
    self._set_debug_preview_placeholder(tr("debug.camera_dialog.disconnected_preview"))
    self._set_debug_camera_status(tr("debug.camera_dialog.disconnected"))
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
        channel = self._debug_capture_channel_for_role()
        self.spin_debug_exposure.setValue(
            float(channel.get("exposure_time_us", exposure) or exposure)
            if channel
            else exposure
        )
        self.spin_debug_gain.setValue(
            float(channel.get("gain", gain) or gain)
            if channel
            else gain
        )
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
        if not channel:
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


def _apply_debug_camera_settings(self, *, quiet: bool = False) -> bool:
    if not quiet and not _require_debug_camera_param_permission(self):
        return
    device = self._debug_camera_device()
    if device is None:
        if not quiet:
            QtWidgets.QMessageBox.information(
                self, tr("debug.camera_dialog.title"), tr("debug.camera_dialog.connect_first")
            )
        return False
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
        QtWidgets.QMessageBox.critical(
            self,
            tr("debug.camera_dialog.title"),
            tr("debug.camera_dialog.apply_failed", error=exc),
        )
        return False
    serial = str(getattr(device, "serial_number", self._selected_debug_camera_serial()) or "").strip()
    payload = dict(self._debug_camera_settings_payload_from_ui())
    self._save_debug_camera_settings(serial, payload)
    self._save_debug_role_binding(self._selected_debug_camera_role(), serial)
    if not quiet:
        self.cameraSettingsApplied.emit(serial, payload)
    self._refresh_debug_camera_settings()
    self._set_debug_camera_status(
        tr("debug.camera_dialog.params_written")
        if quiet
        else tr("debug.camera_dialog.params_applied")
    )
    return True


def _grab_debug_camera_once(self) -> None:
    if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
        QtWidgets.QMessageBox.information(
            self, tr("debug.camera_dialog.title"), tr("debug.camera_dialog.connect_first")
        )
        return
    role = self._selected_debug_camera_role()
    physical_role = self._debug_physical_camera_role(role)
    channel = self._debug_capture_channel_for_role(role)
    device = self._debug_camera_device()
    connected_serial = str(getattr(device, "serial_number", "") or "").strip()
    expected_serial = self._selected_debug_camera_serial()
    if expected_serial and connected_serial and connected_serial != expected_serial:
        QtWidgets.QMessageBox.information(
            self,
            tr("debug.camera_dialog.title"),
            tr(
                "debug.camera_dialog.mapped_camera_first",
                role=role,
                physical=physical_role,
            ),
        )
        self._set_debug_camera_status(
            tr("debug.camera_dialog.mapped_camera_changed")
        )
        return
    if channel and not self._apply_debug_camera_settings(quiet=True):
        return

    light_index = None
    io_controller = getattr(self, "_debug_io_controller", None) or getattr(self, "_runtime_io_controller", None)
    if channel:
        if io_controller is None or not bool(getattr(io_controller, "is_open", False)):
            QtWidgets.QMessageBox.information(
                self,
                tr("debug.camera_dialog.title"),
                tr("debug.camera_dialog.mapped_light_first"),
            )
            self._set_debug_camera_status(
                tr("debug.camera_dialog.mapped_light_requires_io")
            )
            return
        light_index = self._debug_capture_light_index(role)
    capture_light_context = (
        _temporary_debug_camera_light(io_controller, light_index)
        if light_index is not None
        else nullcontext()
    )
    try:
        with capture_light_context:
            if light_index is not None:
                stable_delay_ms = max(0, int(float(channel.get("stable_delay_ms", 50) or 50)))
                if stable_delay_ms:
                    time.sleep(stable_delay_ms / 1000.0)
            frame = self._debug_frame_grab_service.capture_once("debug", timeout_ms=1000)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            self,
            tr("debug.camera_dialog.title"),
            tr("debug.camera_dialog.capture_failed", error=exc),
        )
        self._set_debug_camera_status(
            tr("debug.camera_dialog.capture_failed", error=exc)
        )
        return
    try:
        self._show_debug_preview_image(qimage_from_hik_frame(frame))
    except Exception:
        pass

    capture_dir = os.path.join(self.session.product_dir, "debug_capture")
    os.makedirs(capture_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(capture_dir, f"{role}_debug_cam_{stamp}.png")
    if frame_to_bgr_image is None:
        QtWidgets.QMessageBox.critical(
            self,
            tr("debug.camera_dialog.title"),
            tr("debug.camera_dialog.convert_unavailable"),
        )
        return
    image = frame_to_bgr_image(frame)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if not cv2.imwrite(image_path, image):
        QtWidgets.QMessageBox.critical(
            self,
            tr("debug.camera_dialog.title"),
            tr("debug.camera_dialog.save_failed", path=image_path),
        )
        return
    if image_path not in self.test_files:
        self.test_files.append(image_path)
        self._refresh_lists()
        self._save_session()
    self.tabs.setCurrentIndex(1)
    self._load_canvas_image(image_path)
    self._set_debug_camera_status(
        tr(
            "debug.camera_dialog.captured",
            role=role,
            physical=physical_role,
            image=os.path.basename(image_path),
        )
    )
    self.lbl_status.setText(
        tr(
            "debug.camera_dialog.saved_to_samples",
            role=role,
            image=os.path.basename(image_path),
        )
    )
