from __future__ import annotations

import ctypes
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app_paths import packaged_embedding_test_root


SDK_OK = 0

_EMBEDDING_ROOT = packaged_embedding_test_root(__file__)
_MVIMPORT_CANDIDATES = [
    _EMBEDDING_ROOT / "third_party" / "MvImport",
    _EMBEDDING_ROOT.parent / "Python" / "MvImport",
]
_MVIMPORT_DIR = next((path for path in _MVIMPORT_CANDIDATES if path.exists()), _MVIMPORT_CANDIDATES[0])
if str(_MVIMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_MVIMPORT_DIR))

try:
    from CameraParams_const import MV_ACCESS_Exclusive, MV_GIGE_DEVICE, MV_USB_DEVICE
    from CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT,
        MV_TRIGGER_MODE_OFF,
        MV_TRIGGER_MODE_ON,
        MV_TRIGGER_SOURCE_SOFTWARE,
        MVCC_FLOATVALUE,
        MVCC_INTVALUE_EX,
        MVCC_STRINGVALUE,
    )
    from MvCameraControl_class import MvCamera
    from PixelType_header import (
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BGR8_Packed,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_RGB8_Packed,
    )
except Exception as exc:  # pragma: no cover - depends on local Hikvision SDK runtime
    raise RuntimeError(
        "Failed to import Hikvision Python SDK modules from 'EmbeddingTest/third_party/MvImport'. "
        "Please confirm the Hikvision SDK runtime is installed and the SDK Python files are present."
    ) from exc


class HikCameraError(RuntimeError):
    pass


def _raise_for_code(code: int, operation: str) -> None:
    if int(code) != SDK_OK:
        raise HikCameraError(f"{operation} failed with sdk code=0x{int(code):08X}")


def _decode_bytes(raw: Any) -> str:
    try:
        data = bytes(raw)
    except TypeError:
        data = bytes(bytearray(raw))
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _copy_device_info(source: MV_CC_DEVICE_INFO) -> MV_CC_DEVICE_INFO:
    copied = MV_CC_DEVICE_INFO()
    ctypes.memmove(ctypes.byref(copied), ctypes.byref(source), ctypes.sizeof(MV_CC_DEVICE_INFO))
    return copied


def _transport_name(tlayer_type: int) -> str:
    if int(tlayer_type) == MV_GIGE_DEVICE:
        return "GigE"
    if int(tlayer_type) == MV_USB_DEVICE:
        return "USB"
    return f"Transport({int(tlayer_type)})"


@dataclass(frozen=True)
class HikCameraInfo:
    index: int
    serial_number: str
    user_defined_name: str
    model_name: str
    manufacturer_name: str
    transport_layer: str


@dataclass(frozen=True)
class HikCameraSettings:
    exposure_time_us: float | None = None
    gain: float | None = None
    trigger_mode: str = "software"  # "software" | "continuous"
    trigger_source: str = "Software"
    acquisition_frame_rate_enable: bool | None = None
    acquisition_frame_rate: float | None = None


@dataclass(frozen=True)
class HikFrame:
    camera_serial: str
    width: int
    height: int
    pixel_type: int
    frame_num: int
    host_timestamp: int
    data: bytes

    def as_numpy(self):
        import numpy as np

        array = np.frombuffer(self.data, dtype=np.uint8)
        if self.pixel_type == PixelType_Gvsp_Mono8:
            return array.reshape((self.height, self.width))
        if self.pixel_type in (
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerBG8,
        ):
            return array.reshape((self.height, self.width))
        if self.pixel_type == PixelType_Gvsp_BGR8_Packed:
            return array.reshape((self.height, self.width, 3))
        if self.pixel_type == PixelType_Gvsp_RGB8_Packed:
            return array.reshape((self.height, self.width, 3))
        raise ValueError(f"unsupported pixel type for direct numpy reshape: {self.pixel_type}")


def frame_to_rgb_image(frame: HikFrame):
    import cv2
    import numpy as np

    pixel_type = int(frame.pixel_type)
    array = np.frombuffer(frame.data, dtype=np.uint8)

    if pixel_type == PixelType_Gvsp_Mono8:
        return array.reshape((frame.height, frame.width))
    if pixel_type == PixelType_Gvsp_RGB8_Packed:
        return array.reshape((frame.height, frame.width, 3))
    if pixel_type == PixelType_Gvsp_BGR8_Packed:
        bgr = array.reshape((frame.height, frame.width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    bayer_convert_codes = {
        # Match the color appearance of Hikvision MVS preview.
        int(PixelType_Gvsp_BayerRG8): cv2.COLOR_BayerBG2RGB,
        int(PixelType_Gvsp_BayerGR8): cv2.COLOR_BayerGB2RGB,
        int(PixelType_Gvsp_BayerGB8): cv2.COLOR_BayerGR2RGB,
        int(PixelType_Gvsp_BayerBG8): cv2.COLOR_BayerRG2RGB,
    }
    if pixel_type in bayer_convert_codes:
        raw = array.reshape((frame.height, frame.width))
        return cv2.cvtColor(raw, bayer_convert_codes[pixel_type])

    raise ValueError(f"unsupported pixel type for RGB conversion: {frame.pixel_type}")


def frame_to_bgr_image(frame: HikFrame):
    import cv2

    image = frame_to_rgb_image(frame)
    if getattr(image, "ndim", 0) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class HikCameraDevice:
    """Wrapper around one Hikvision camera handle."""

    def __init__(
        self,
        camera_info: HikCameraInfo,
        device_info: MV_CC_DEVICE_INFO,
        *,
        settings: HikCameraSettings | None = None,
    ) -> None:
        self.camera_info = camera_info
        self._device_info = _copy_device_info(device_info)
        self._settings = settings or HikCameraSettings()
        self._camera: MvCamera | None = None
        self._lock = threading.RLock()
        self._is_open = False
        self._is_grabbing = False

    @property
    def serial_number(self) -> str:
        return self.camera_info.serial_number

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_grabbing(self) -> bool:
        return self._is_grabbing

    def open(self) -> None:
        with self._lock:
            if self._is_open:
                return
            camera = MvCamera()
            _raise_for_code(camera.MV_CC_CreateHandle(self._device_info), "MV_CC_CreateHandle")
            try:
                _raise_for_code(camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "MV_CC_OpenDevice")
                if self.camera_info.transport_layer == "GigE":
                    packet_size = int(camera.MV_CC_GetOptimalPacketSize())
                    if packet_size > 0:
                        camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                self._camera = camera
                self._is_open = True
                self.apply_settings(self._settings)
            except Exception:
                try:
                    camera.MV_CC_CloseDevice()
                finally:
                    camera.MV_CC_DestroyHandle()
                raise

    def close(self) -> None:
        with self._lock:
            camera = self._require_camera(allow_closed=True)
            if camera is None:
                return
            try:
                if self._is_grabbing:
                    camera.MV_CC_StopGrabbing()
            finally:
                try:
                    if self._is_open:
                        camera.MV_CC_CloseDevice()
                finally:
                    camera.MV_CC_DestroyHandle()
                    self._camera = None
                    self._is_grabbing = False
                    self._is_open = False

    def apply_settings(self, settings: HikCameraSettings) -> None:
        with self._lock:
            camera = self._require_camera()
            self._settings = settings
            was_grabbing = self._is_grabbing

            if was_grabbing:
                _raise_for_code(
                    camera.MV_CC_StopGrabbing(),
                    "MV_CC_StopGrabbing before applying settings",
                )
                self._is_grabbing = False

            try:
                self._apply_trigger_mode_only(settings.trigger_mode)

                camera.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
                camera.MV_CC_SetEnumValueByString("GainAuto", "Off")

                if settings.exposure_time_us is not None:
                    _raise_for_code(
                        camera.MV_CC_SetFloatValue("ExposureTime", float(settings.exposure_time_us)),
                        "set ExposureTime",
                    )
                if settings.gain is not None:
                    _raise_for_code(camera.MV_CC_SetFloatValue("Gain", float(settings.gain)), "set Gain")
                if settings.acquisition_frame_rate_enable is not None:
                    _raise_for_code(
                        camera.MV_CC_SetBoolValue(
                            "AcquisitionFrameRateEnable",
                            bool(settings.acquisition_frame_rate_enable),
                        ),
                        "set AcquisitionFrameRateEnable",
                    )
                if settings.acquisition_frame_rate is not None:
                    _raise_for_code(
                        camera.MV_CC_SetFloatValue("AcquisitionFrameRate", float(settings.acquisition_frame_rate)),
                        "set AcquisitionFrameRate",
                    )
            finally:
                if was_grabbing:
                    _raise_for_code(
                        camera.MV_CC_StartGrabbing(),
                        "MV_CC_StartGrabbing after applying settings",
                    )
                    self._is_grabbing = True

    def set_trigger_mode(self, trigger_mode: str) -> None:
        with self._lock:
            camera = self._require_camera()
            was_grabbing = self._is_grabbing
            normalized = str(trigger_mode or "").strip().lower()

            if was_grabbing:
                _raise_for_code(
                    camera.MV_CC_StopGrabbing(),
                    "MV_CC_StopGrabbing before setting trigger mode",
                )
                self._is_grabbing = False

            try:
                self._apply_trigger_mode_only(normalized)
                self._settings = replace(
                    self._settings,
                    trigger_mode=normalized,
                    trigger_source="Software",
                )
            finally:
                if was_grabbing:
                    _raise_for_code(
                        camera.MV_CC_StartGrabbing(),
                        "MV_CC_StartGrabbing after setting trigger mode",
                    )
                    self._is_grabbing = True

    def start_grabbing(self) -> None:
        with self._lock:
            camera = self._require_camera()
            if self._is_grabbing:
                return
            _raise_for_code(camera.MV_CC_StartGrabbing(), "MV_CC_StartGrabbing")
            self._is_grabbing = True

    def stop_grabbing(self) -> None:
        with self._lock:
            camera = self._require_camera()
            if not self._is_grabbing:
                return
            _raise_for_code(camera.MV_CC_StopGrabbing(), "MV_CC_StopGrabbing")
            self._is_grabbing = False

    def trigger_once(self) -> None:
        with self._lock:
            camera = self._require_camera()
            _raise_for_code(camera.MV_CC_SetCommandValue("TriggerSoftware"), "TriggerSoftware")

    def capture_one_frame(self, timeout_ms: int = 1000) -> HikFrame:
        with self._lock:
            camera = self._require_camera()
            self.start_grabbing()
            if self._settings.trigger_mode.lower() == "software":
                self.trigger_once()

            frame = MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(MV_FRAME_OUT))
            _raise_for_code(camera.MV_CC_GetImageBuffer(frame, int(timeout_ms)), "MV_CC_GetImageBuffer")
            try:
                frame_len = int(frame.stFrameInfo.nFrameLen)
                payload = ctypes.string_at(frame.pBufAddr, frame_len)
                return HikFrame(
                    camera_serial=self.serial_number,
                    width=int(frame.stFrameInfo.nWidth or frame.stFrameInfo.nExtendWidth),
                    height=int(frame.stFrameInfo.nHeight or frame.stFrameInfo.nExtendHeight),
                    pixel_type=int(frame.stFrameInfo.enPixelType),
                    frame_num=int(frame.stFrameInfo.nFrameNum),
                    host_timestamp=int(frame.stFrameInfo.nHostTimeStamp),
                    data=payload,
                )
            finally:
                camera.MV_CC_FreeImageBuffer(frame)

    def get_float_value(self, key: str) -> float:
        with self._lock:
            camera = self._require_camera()
            value = MVCC_FLOATVALUE()
            ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(MVCC_FLOATVALUE))
            _raise_for_code(camera.MV_CC_GetFloatValue(key, value), f"get {key}")
            return float(value.fCurValue)

    def get_int_value(self, key: str) -> int:
        with self._lock:
            camera = self._require_camera()
            value = MVCC_INTVALUE_EX()
            ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(MVCC_INTVALUE_EX))
            _raise_for_code(camera.MV_CC_GetIntValueEx(key, value), f"get {key}")
            return int(value.nCurValue)

    def get_string_value(self, key: str) -> str:
        with self._lock:
            camera = self._require_camera()
            value = MVCC_STRINGVALUE()
            ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(MVCC_STRINGVALUE))
            _raise_for_code(camera.MV_CC_GetStringValue(key, value), f"get {key}")
            return _decode_bytes(value.chCurValue)

    def _require_camera(self, *, allow_closed: bool = False) -> MvCamera | None:
        camera = self._camera
        if camera is None:
            if allow_closed:
                return None
            raise HikCameraError(f"camera {self.serial_number} is not open")
        return camera

    def _apply_trigger_mode_only(self, trigger_mode: str) -> None:
        camera = self._require_camera()
        normalized = str(trigger_mode or "").strip().lower()
        if normalized == "software":
            _raise_for_code(camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON), "set TriggerMode On")
            _raise_for_code(
                camera.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE),
                "set TriggerSource Software",
            )
            return
        if normalized == "continuous":
            _raise_for_code(camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF), "set TriggerMode Off")
            return
        raise ValueError(f"unsupported trigger_mode: {trigger_mode}")


class HikCameraManager:
    """Owns SDK lifecycle and device enumeration/binding."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._initialized = False
        self._device_info_by_serial: dict[str, MV_CC_DEVICE_INFO] = {}
        self._camera_info_by_serial: dict[str, HikCameraInfo] = {}
        self._opened_devices: list[HikCameraDevice] = []

    def open(self) -> None:
        with self._lock:
            if self._initialized:
                return
            _raise_for_code(MvCamera.MV_CC_Initialize(), "MV_CC_Initialize")
            self._initialized = True

    def close(self) -> None:
        with self._lock:
            for device in list(self._opened_devices):
                try:
                    device.close()
                except Exception:
                    pass
            self._opened_devices.clear()
            self._device_info_by_serial.clear()
            self._camera_info_by_serial.clear()
            if self._initialized:
                _raise_for_code(MvCamera.MV_CC_Finalize(), "MV_CC_Finalize")
                self._initialized = False

    def enumerate_cameras(self) -> list[HikCameraInfo]:
        with self._lock:
            self.open()
            device_list = MV_CC_DEVICE_INFO_LIST()
            ctypes.memset(ctypes.byref(device_list), 0, ctypes.sizeof(MV_CC_DEVICE_INFO_LIST))
            tlayer_types = MV_GIGE_DEVICE | MV_USB_DEVICE
            _raise_for_code(MvCamera.MV_CC_EnumDevices(tlayer_types, device_list), "MV_CC_EnumDevices")

            results: list[HikCameraInfo] = []
            self._device_info_by_serial.clear()
            self._camera_info_by_serial.clear()

            for index in range(int(device_list.nDeviceNum)):
                device_info = ctypes.cast(
                    device_list.pDeviceInfo[index],
                    ctypes.POINTER(MV_CC_DEVICE_INFO),
                ).contents
                info = self._make_camera_info(index, device_info)
                copied = _copy_device_info(device_info)
                self._device_info_by_serial[info.serial_number] = copied
                self._camera_info_by_serial[info.serial_number] = info
                results.append(info)
            return results

    def get_camera_info(self, serial_number: str) -> HikCameraInfo:
        serial_number = str(serial_number).strip()
        if not serial_number:
            raise ValueError("serial_number must not be empty")
        if serial_number not in self._camera_info_by_serial:
            self.enumerate_cameras()
        try:
            return self._camera_info_by_serial[serial_number]
        except KeyError as exc:
            raise HikCameraError(f"camera with serial '{serial_number}' not found") from exc

    def open_camera(
        self,
        serial_number: str,
        *,
        settings: HikCameraSettings | None = None,
    ) -> HikCameraDevice:
        with self._lock:
            info = self.get_camera_info(serial_number)
            device_info = self._device_info_by_serial[serial_number]
            device = HikCameraDevice(info, device_info, settings=settings)
            device.open()
            self._opened_devices.append(device)
            return device

    def open_bound_cameras(
        self,
        serial_bindings: dict[str, str],
        *,
        settings_by_role: dict[str, HikCameraSettings] | None = None,
    ) -> dict[str, HikCameraDevice]:
        opened: dict[str, HikCameraDevice] = {}
        for role, serial_number in serial_bindings.items():
            settings = settings_by_role.get(role) if settings_by_role is not None else None
            opened[role] = self.open_camera(serial_number, settings=settings)
        return opened

    def _make_camera_info(self, index: int, device_info: MV_CC_DEVICE_INFO) -> HikCameraInfo:
        tlayer_type = int(device_info.nTLayerType)
        transport_layer = _transport_name(tlayer_type)

        if tlayer_type == MV_GIGE_DEVICE:
            special = device_info.SpecialInfo.stGigEInfo
            serial_number = _decode_bytes(special.chSerialNumber)
            user_defined_name = _decode_bytes(special.chUserDefinedName)
            model_name = _decode_bytes(special.chModelName)
            manufacturer_name = _decode_bytes(special.chManufacturerName)
        elif tlayer_type == MV_USB_DEVICE:
            special = device_info.SpecialInfo.stUsb3VInfo
            serial_number = _decode_bytes(special.chSerialNumber)
            user_defined_name = _decode_bytes(special.chUserDefinedName)
            model_name = _decode_bytes(special.chModelName)
            manufacturer_name = _decode_bytes(special.chManufacturerName)
        else:
            serial_number = f"unknown-{index}"
            user_defined_name = ""
            model_name = ""
            manufacturer_name = ""

        return HikCameraInfo(
            index=index,
            serial_number=serial_number,
            user_defined_name=user_defined_name,
            model_name=model_name,
            manufacturer_name=manufacturer_name,
            transport_layer=transport_layer,
        )


class FrameGrabService:
    """Higher-level camera binding and single-frame acquisition service."""

    def __init__(self, manager: HikCameraManager) -> None:
        self.manager = manager
        self._devices_by_role: dict[str, HikCameraDevice] = {}
        self._lock = threading.RLock()

    def open_bound_cameras(
        self,
        serial_bindings: dict[str, str],
        *,
        settings_by_role: dict[str, HikCameraSettings] | None = None,
    ) -> None:
        with self._lock:
            self.close_all()
            self._devices_by_role = self.manager.open_bound_cameras(
                serial_bindings,
                settings_by_role=settings_by_role,
            )

    def roles(self) -> list[str]:
        with self._lock:
            return list(self._devices_by_role.keys())

    def get_device(self, role: str) -> HikCameraDevice:
        with self._lock:
            try:
                return self._devices_by_role[role]
            except KeyError as exc:
                raise HikCameraError(f"camera role '{role}' is not opened") from exc

    def capture_once(self, role: str, *, timeout_ms: int = 1000) -> HikFrame:
        return self.get_device(role).capture_one_frame(timeout_ms=timeout_ms)

    def capture_many(
        self,
        roles: list[str],
        *,
        timeout_ms: int = 1000,
        capture_interval_s: float = 0.0,
    ) -> dict[str, HikFrame]:
        frames: dict[str, HikFrame] = {}
        for index, role in enumerate(roles):
            frames[role] = self.capture_once(role, timeout_ms=timeout_ms)
            if capture_interval_s > 0 and index < len(roles) - 1:
                time.sleep(float(capture_interval_s))
        return frames

    def close_all(self) -> None:
        with self._lock:
            for device in list(self._devices_by_role.values()):
                device.close()
            self._devices_by_role.clear()
