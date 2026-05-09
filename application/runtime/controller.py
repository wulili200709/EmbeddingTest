"""
runtime_controller.py

运行链路业务控制器，零 UI 直接调用，通过 Signal 驱动 RuntimeModePage 和 MainWindow。

职责：
  - 持有并管理所有运行服务对象（HikCameraManager / FrameGrabService /
    InspectionRuntime / InspectionScheduler / PermissionManager / TestRecordService）
  - 提供公开操作方法：refresh_cameras / connect_cameras / disconnect / trigger / release
  - 通过细粒度 Signal 把状态变化推给 UI（RuntimeModePage）

不负责：
  - 任何 QWidget / QDialog 操作（由 MainWindow 监听 warningOccurred 等信号弹框）
  - 工具页业务（ROI / 训练 / 产品切换），通过构造时传入的 ToolPage 引用调用
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from PySide6 import QtCore

from . import bindings
from .capture_policy import (
    DEFAULT_LIGHT_STABLE_MS,
    DEFAULT_RELEASE_PASSWORD,
    RUNTIME_CAPTURE_POLICY_ALL,
    RUNTIME_CAPTURE_POLICY_NG_ONLY,
    delete_capture_artifacts,
    normalize_capture_retention_policy,
    retained_capture_paths_for_policy,
)

_RUN_STATE_ZH_FOR_STATUS = {
    "WaitingTrigger": "等待触发",
    "ReleasedPendingConsume": "已放行，待消耗",
    "CapturingCam1": "采集中（相机1）",
    "CapturingCam2": "采集中（相机2）",
    "Inspecting": "检测中",
    "Aggregating": "汇总结果",
    "CompletedOk": "本轮完成 OK",
    "CompletedNg": "本轮 NG",
    "LockedByNg": "NG 锁定",
    "Error": "运行异常",
    "Unavailable": "服务不可用",
}

from ..algorithm_controller import AlgorithmController
from ..inspection_executor import InspectionExecutionRequest, InspectionExecutor
from ..product_session import ProductSession
from ..runtime_context import RuntimeContextProtocol

from domain import (
    RuntimeInspectionResult,
    aggregate_runtime_outcome,
    build_pending_result,
    recipe_name_from_path,
)
from infrastructure.camera_settings_store import (
    CameraSettingsStore,
    hik_settings_kwargs_from_mapping,
    light_source_mode_from_mapping,
)

try:
    from services import (
        CameraInspectionOutcome,
        CsvRecordWriter,
        CsvReleaseLogWriter,
        FrameGrabService,
        HikCameraManager,
        HikCameraSettings,
        InspectionRuntime,
        InspectionScheduler,
        PermissionManager,
        ReleaseLogService,
        RunStateMachine,
        TestRecordService,
        frame_to_bgr_image,
    )
except Exception:
    CameraInspectionOutcome = None  # type: ignore[assignment,misc]
    CsvRecordWriter = None          # type: ignore[assignment,misc]
    CsvReleaseLogWriter = None      # type: ignore[assignment,misc]
    FrameGrabService = None         # type: ignore[assignment,misc]
    HikCameraManager = None         # type: ignore[assignment,misc]
    HikCameraSettings = None        # type: ignore[assignment,misc]
    InspectionRuntime = None        # type: ignore[assignment,misc]
    InspectionScheduler = None      # type: ignore[assignment,misc]
    PermissionManager = None        # type: ignore[assignment,misc]
    ReleaseLogService = None        # type: ignore[assignment,misc]
    RunStateMachine = None          # type: ignore[assignment,misc]
    TestRecordService = None        # type: ignore[assignment,misc]
    frame_to_bgr_image = None       # type: ignore[assignment,misc]

try:
    from devices import DiMonitor, IoManager, LightController, TowerLightController
except Exception:
    DiMonitor = None               # type: ignore[assignment,misc]
    IoManager = None               # type: ignore[assignment,misc]
    LightController = None          # type: ignore[assignment,misc]
    TowerLightController = None     # type: ignore[assignment,misc]


DEFAULT_TOWER_LIGHT_OK_FLASH_MS = 200
DEFAULT_TOWER_LIGHT_NG_FLASH_MS = 200
DEFAULT_TOWER_LIGHT_NG_BUZZER_MS = 500
DEFAULT_TOWER_LIGHT_IDLE_BLUE_DELAY_MS = 30000


# ---------------------------------------------------------------------------
# 无硬件时的占位控制器
# ---------------------------------------------------------------------------

class _UiOnlyLightController:
    def __init__(self) -> None:
        self.active_camera: Optional[int] = None
        self._camera_light_modes: Dict[int, str] = {}

    def set_camera_light_mode(self, camera_index: int, mode: str) -> None:
        self._camera_light_modes[int(camera_index)] = str(mode or "board_io").strip() or "board_io"
        if self._camera_light_modes.get(int(camera_index)) != "board_io" and self.active_camera == int(camera_index):
            self.active_camera = None

    def requires_stable_delay(self, camera_index: int) -> bool:
        return self._camera_light_modes.get(int(camera_index), "board_io") == "board_io"

    def prepare_capture(self, camera_index: int) -> None:
        if self.requires_stable_delay(camera_index):
            self.active_camera = int(camera_index)
        else:
            self.active_camera = None

    def finish_capture(self, camera_index: int) -> None:
        if self.active_camera == int(camera_index):
            self.active_camera = None


class _UiOnlyTowerLightController:
    def __init__(self) -> None:
        self.state = "waiting"

    def enter_waiting(self) -> None:
        self.state = "waiting"

    def enter_inspecting(self) -> None:
        self.state = "inspecting"

    def show_ok(self) -> None:
        self.state = "ok"

    def show_ng(self) -> None:
        self.state = "ng"


# ---------------------------------------------------------------------------
# RuntimeController
# ---------------------------------------------------------------------------

class RuntimeController(QtCore.QObject):
    """
    运行链路业务控制器。

    使用方式（MainWindow 中）::

        self.runtime_ctrl = RuntimeController(
            session=self.session,
            algo=self.algo,
            tool_page=self.tool_page,
            import_error=_RUNTIME_IMPORT_ERROR,
            parent=self,
        )
        # 然后在 _connect_signals 里把各 Signal 连到 RuntimeModePage
    """

    # ── 状态类 Signal（逐字段更新，替代原来的整体刷新） ──────────────────
    runtimeStateChanged     = QtCore.Signal(str)   # → runtime_page.set_runtime_state
    productNameChanged      = QtCore.Signal(str)   # → runtime_page.set_current_product
    permissionStatusChanged = QtCore.Signal(str)   # → runtime_page.set_permission_status
    connectionStatusChanged = QtCore.Signal(str)   # → runtime_page.set_connection_status
    towerLightStatusChanged = QtCore.Signal(str)   # → runtime_page.set_tower_light_status
    statusMessageChanged    = QtCore.Signal(str)   # → runtime_page.set_runtime_status
    recordPathChanged       = QtCore.Signal(str)   # → runtime_page.set_record_path
    ioStatusChanged         = QtCore.Signal(bool, str, object)  # (ready, detail, io_controller)

    # ── 动作类 Signal ─────────────────────────────────────────────────────
    camerasEnumerated = QtCore.Signal(list)        # → runtime_page.set_available_cameras
    logAppended       = QtCore.Signal(str)         # → runtime_page.append_log
    busyChanged       = QtCore.Signal(bool)        # → runtime_page.set_busy
    triggerResultReady = QtCore.Signal(str, str)   # (result, detail) → runtime_page.set_final_result
    previewUpdated    = QtCore.Signal(str, object)    # (role, preview source) → MainWindow 转发
    cameraViewsCleared = QtCore.Signal()           # → runtime_page.clear_camera_views
    activeCameraRolesChanged = QtCore.Signal(list) # → runtime_page.set_active_camera_roles
    inspectionItemsChanged = QtCore.Signal(list)   # → runtime_page.set_inspection_items
    cameraResultsChanged = QtCore.Signal(dict)     # → runtime_page.set_camera_results
    durationChanged = QtCore.Signal(int)           # → runtime_page.set_duration_ms
    timingBreakdownChanged = QtCore.Signal(dict)   # → runtime_page.set_timing_breakdown

    # ── 对话框类 Signal（由 MainWindow 弹框） ─────────────────────────────
    warningOccurred = QtCore.Signal(str)           # → QMessageBox.warning
    errorOccurred   = QtCore.Signal(str)           # → QMessageBox.critical
    infoOccurred    = QtCore.Signal(str)           # → QMessageBox.information

    def __init__(
        self,
        session: ProductSession,
        algo: AlgorithmController,
        runtime_context: RuntimeContextProtocol,
        import_error: Optional[Exception] = None,
        release_password: str = DEFAULT_RELEASE_PASSWORD,
        lock_on_ng: bool = True,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._algo = algo
        self._runtime_context = runtime_context
        self._import_error = import_error
        self._release_password = release_password
        self._lock_on_ng = bool(lock_on_ng)

        self._frame_lock = threading.RLock()
        self._inspect_lock = threading.RLock()
        self._persistence_lock = threading.RLock()
        self._persistence_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="runtime-persist",
        )
        self._persistence_futures: set[Future] = set()
        self._last_capture_paths: Dict[str, str] = {}
        self._last_preview_frames: Dict[str, object] = {}
        self._last_record_path: Optional[str] = None
        self._last_runtime_result: Optional[RuntimeInspectionResult] = None
        self._capture_retention_policy = RUNTIME_CAPTURE_POLICY_ALL
        session_product_dir = Path(str(getattr(self._session, "product_dir", ".") or "."))
        camera_settings_path = str(
            getattr(self._session, "camera_settings_path", session_product_dir / "camera_settings.json")
        )
        self._camera_settings_store = CameraSettingsStore(camera_settings_path)
        self._runtime_records_dir = session_product_dir / "runtime_records"
        self._runtime_capture_dir = session_product_dir / "runtime_capture"

        self._camera_manager = None
        self._frame_grab_service = None
        self._permission_manager = None
        self._scheduler = None
        self._record_service = None
        self._release_log_service = None
        self._runner = None
        self._io_controller = None
        self._io_ready = False
        self._io_status_detail = "IO not initialized"
        self._di_poller = None
        self._inspection_executor = InspectionExecutor(self._runtime_context)
        self._last_item_results_by_camera: Dict[str, list] = {}
        self._light_controller: _UiOnlyLightController = _UiOnlyLightController()
        self._tower_light_controller: _UiOnlyTowerLightController = _UiOnlyTowerLightController()
        self._busy = False
        self._pending_camera_settings_by_serial: Dict[str, dict] = {}
        self._tower_light_settings = {
            "ok_flash_ms": DEFAULT_TOWER_LIGHT_OK_FLASH_MS,
            "ng_flash_ms": DEFAULT_TOWER_LIGHT_NG_FLASH_MS,
            "ng_buzzer_ms": DEFAULT_TOWER_LIGHT_NG_BUZZER_MS,
            "idle_blue_delay_ms": DEFAULT_TOWER_LIGHT_IDLE_BLUE_DELAY_MS,
        }

    def set_capture_retention_policy(self, policy: object) -> None:
        self._capture_retention_policy = normalize_capture_retention_policy(policy)

    def capture_retention_policy(self) -> str:
        return self._capture_retention_policy

    def _submit_persistence_task(
        self,
        callback: Callable[..., Any],
        *args,
        description: str = "",
        **kwargs,
    ) -> Future | None:
        executor = self._persistence_executor
        if executor is None:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                self.logAppended.emit(
                    f"[runtime] persistence task failed ({description or 'sync-fallback'}): {exc}"
                )
            return None

        try:
            future = executor.submit(callback, *args, **kwargs)
        except RuntimeError:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                self.logAppended.emit(
                    f"[runtime] persistence task failed ({description or 'sync-fallback'}): {exc}"
                )
            return None

        with self._persistence_lock:
            self._persistence_futures.add(future)

        def _on_done(done: Future) -> None:
            with self._persistence_lock:
                self._persistence_futures.discard(done)
            try:
                done.result()
            except Exception as exc:
                self.logAppended.emit(
                    f"[runtime] persistence task failed ({description or 'async'}): {exc}"
                )

        future.add_done_callback(_on_done)
        return future

    def shutdown_persistence(self, *, wait: bool = True) -> None:
        with self._persistence_lock:
            executor = self._persistence_executor
            self._persistence_executor = None
        if executor is None:
            return
        executor.shutdown(wait=wait)

    def initialize_startup_io(self, *, force: bool = False) -> bool:
        return self._initialize_startup_io(force=force)

    def tower_light_settings(self) -> dict[str, int]:
        return dict(self._tower_light_settings)

    def runtime_records_directory(self) -> str:
        return str(self._runtime_records_dir)

    def update_runtime_records_directory(self, directory: str | Path) -> None:
        target_text = str(directory or "").strip()
        target_dir = Path(target_text) if target_text else Path(self._session.product_dir) / "runtime_records"
        self._runtime_records_dir = target_dir
        self._runtime_records_dir.mkdir(parents=True, exist_ok=True)
        if self._record_service is not None:
            self._record_service.writer.base_directory = self._runtime_records_dir
        self._last_record_path = str(
            self._runtime_records_dir / f"{datetime.now().strftime('%Y-%m-%d')}.csv"
        )
        self.recordPathChanged.emit(self._last_record_path or "-")

    def runtime_capture_directory(self) -> str:
        return str(self._runtime_capture_dir)

    def update_runtime_capture_directory(self, directory: str | Path) -> None:
        target_text = str(directory or "").strip()
        target_dir = Path(target_text) if target_text else Path(self._session.product_dir) / "runtime_capture"
        self._runtime_capture_dir = target_dir
        self._runtime_capture_dir.mkdir(parents=True, exist_ok=True)

    def update_tower_light_settings(self, settings: dict[str, object]) -> None:
        normalized = {
            "ok_flash_ms": max(10, int(settings.get("ok_flash_ms", DEFAULT_TOWER_LIGHT_OK_FLASH_MS))),
            "ng_flash_ms": max(10, int(settings.get("ng_flash_ms", DEFAULT_TOWER_LIGHT_NG_FLASH_MS))),
            "ng_buzzer_ms": max(0, int(settings.get("ng_buzzer_ms", DEFAULT_TOWER_LIGHT_NG_BUZZER_MS))),
            "idle_blue_delay_ms": max(0, int(settings.get("idle_blue_delay_ms", DEFAULT_TOWER_LIGHT_IDLE_BLUE_DELAY_MS))),
        }
        self._tower_light_settings = normalized

        if self._io_controller is None or not getattr(self._io_controller, "is_open", False):
            self._update_status("\u5854\u706f\u65f6\u5e8f\u53c2\u6570\u5df2\u66f4\u65b0")
            return

        previous_state = str(getattr(self._tower_light_controller, "state", "waiting") or "waiting")
        try:
            if hasattr(self._tower_light_controller, "close"):
                self._tower_light_controller.close()
        except Exception:
            pass

        if TowerLightController is None:
            self._tower_light_controller = _UiOnlyTowerLightController()
        else:
            self._tower_light_controller = TowerLightController(
                self._io_controller,
                ok_flash_ms=normalized["ok_flash_ms"],
                ng_flash_ms=normalized["ng_flash_ms"],
                ng_buzzer_ms=normalized["ng_buzzer_ms"],
                idle_blue_delay_s=float(normalized["idle_blue_delay_ms"]) / 1000.0,
            )

        if self._runner is not None:
            try:
                self._runner.tower_light_controller = self._tower_light_controller
            except Exception:
                pass

        if previous_state == "inspecting":
            self._tower_light_controller.enter_inspecting()
        elif previous_state == "off" and hasattr(self._tower_light_controller, "all_off"):
            self._tower_light_controller.all_off()
        elif previous_state == "ok":
            self._tower_light_controller.show_ok()
        elif previous_state in {"ng", "ng_buzzer"}:
            self._tower_light_controller.show_ng()
        elif previous_state == "post_result" and hasattr(self._tower_light_controller, "schedule_idle_waiting"):
            self._tower_light_controller.schedule_idle_waiting()
        else:
            self._tower_light_controller.enter_waiting()

        self._update_status("\u5854\u706f\u65f6\u5e8f\u53c2\u6570\u5df2\u66f4\u65b0")

    def _sync_camera_settings_store_path(self) -> None:
        self._camera_settings_store.set_path(self._session.camera_settings_path)

    # ------------------------------------------------------------------
    # 公开操作方法（RuntimeModePage Signal 直接连到这里）
    # ------------------------------------------------------------------

    def refresh_cameras(self) -> None:
        """枚举可用相机，枚举完成后发射 camerasEnumerated。"""
        if self._import_error is not None:
            self._update_status(f"运行服务不可用：{self._import_error}")
            return

        manager = self._camera_manager
        owns_manager = False
        if manager is None:
            manager = HikCameraManager()
            owns_manager = True

        try:
            infos = manager.enumerate_cameras()
        except Exception as exc:
            self.camerasEnumerated.emit([])
            self.logAppended.emit(f"[相机] 刷新失败：{exc}")
            self._update_status(f"刷新相机失败：{exc}")
            return
        finally:
            if owns_manager:
                try:
                    manager.close()
                except Exception:
                    pass

        descriptions = [
            f"{info.serial_number} / {info.model_name or 'UnknownModel'} / {info.transport_layer}"
            for info in infos
        ]
        self.camerasEnumerated.emit(descriptions)
        self.logAppended.emit(f"[相机] 枚举完成，共 {len(descriptions)} 台")
        self._update_status("已刷新相机列表")

    def connect_cameras(self, bindings_obj) -> None:
        """解析绑定关系并连接相机。"""
        self._sync_camera_settings_store_path()
        bindings = dict(bindings_obj or {})
        cam1 = str(bindings.get("cam1", "")).strip()
        cam2 = str(bindings.get("cam2", "")).strip()
        bindings = {}
        if cam1:
            bindings["cam1"] = cam1
        if cam2:
            bindings["cam2"] = cam2

        if not bindings:
            self.warningOccurred.emit("请先填写至少一个相机序列号")
            return

        if "cam2" in bindings and "cam1" not in bindings:
            self.warningOccurred.emit("最小运行链路需要先配置 Cam1，再决定是否接入 Cam2")
            return

        if not self._rebuild_runner():
            return

        settings_by_role = {}
        for role, serial in bindings.items():
            saved_settings = self._camera_settings_store.load_for_role(role, serial=serial)
            settings_by_role[role] = HikCameraSettings(
                **hik_settings_kwargs_from_mapping(
                    saved_settings,
                    default_trigger_mode="software",
                    force_trigger_mode="software",
                )
            )
            camera_index = 1 if role == "cam1" else 2 if role == "cam2" else 0
            if camera_index > 0 and hasattr(self._light_controller, "set_camera_light_mode"):
                self._light_controller.set_camera_light_mode(
                    camera_index,
                    light_source_mode_from_mapping(saved_settings),
                )
        try:
            self._frame_grab_service.open_bound_cameras(bindings, settings_by_role=settings_by_role)
        except Exception as exc:
            self.disconnect(silent=True, close_io=False)
            self.errorOccurred.emit(f"连接相机失败：{exc}")
            self._update_status(f"连接相机失败：{exc}")
            return

        self._start_di_poller_if_available()

        self.cameraViewsCleared.emit()
        self.logAppended.emit(
            "[相机] 已连接：" + ", ".join(f"{role}={serial}" for role, serial in bindings.items())
        )
        self._update_status("相机已连接，可开始触发")

    def try_connect_cameras(self, bindings_obj) -> bool:
        self._sync_camera_settings_store_path()
        bindings = dict(bindings_obj or {})
        cam1 = str(bindings.get("cam1", "")).strip()
        cam2 = str(bindings.get("cam2", "")).strip()
        bindings = {}
        if cam1:
            bindings["cam1"] = cam1
        if cam2:
            bindings["cam2"] = cam2

        if not bindings:
            return False

        if "cam2" in bindings and "cam1" not in bindings:
            self.logAppended.emit("[camera] startup auto-connect skipped: Cam2 requires Cam1")
            self._update_status("startup auto-connect skipped: missing Cam1 binding")
            return False

        if not self._rebuild_runner():
            return False

        settings_by_role = {}
        for role, serial in bindings.items():
            saved_settings = self._camera_settings_store.load_for_role(role, serial=serial)
            settings_by_role[role] = HikCameraSettings(
                **hik_settings_kwargs_from_mapping(
                    saved_settings,
                    default_trigger_mode="software",
                    force_trigger_mode="software",
                )
            )
            camera_index = 1 if role == "cam1" else 2 if role == "cam2" else 0
            if camera_index > 0 and hasattr(self._light_controller, "set_camera_light_mode"):
                self._light_controller.set_camera_light_mode(
                    camera_index,
                    light_source_mode_from_mapping(saved_settings),
                )
        try:
            self._frame_grab_service.open_bound_cameras(bindings, settings_by_role=settings_by_role)
        except Exception as exc:
            self.disconnect(silent=True, close_io=False)
            self.logAppended.emit(f"[camera] startup auto-connect failed: {exc}")
            self._update_status(f"startup auto-connect failed: {exc}")
            return False

        self._start_di_poller_if_available()
        self.cameraViewsCleared.emit()
        self.logAppended.emit(
            "[camera] startup auto-connect success: "
            + ", ".join(f"{role}={serial}" for role, serial in bindings.items())
        )
        self._update_status("startup auto-connect success")
        return True

    def disconnect(self, *, silent: bool = False, close_io: bool = True) -> None:
        """断开所有相机并释放运行链路资源。"""
        self._stop_di_poller()
        if close_io:
            self._close_io_controller()
        if self._frame_grab_service is not None:
            try:
                self._frame_grab_service.close_all()
            except Exception:
                pass
        if self._camera_manager is not None:
            try:
                self._camera_manager.close()
            except Exception:
                pass

        self._camera_manager = None
        self._frame_grab_service = None
        self._permission_manager = None
        self._scheduler = None
        self._record_service = None
        self._release_log_service = None
        self._runner = None
        self._di_poller = None
        self._last_capture_paths = {}
        self._last_preview_frames = {}
        self._last_record_path = None
        self._last_item_results_by_camera = {}
        self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
        if close_io or self._io_controller is None:
            self._io_controller = None
            self._light_controller = _UiOnlyLightController()
            self._tower_light_controller = _UiOnlyTowerLightController()
        self._busy = False
        self._pending_camera_settings_by_serial = {}

        self.cameraViewsCleared.emit()
        if not silent:
            self.logAppended.emit("[相机] 已断开")
        self._update_status("相机已断开")

    def apply_camera_settings_for_serial(self, serial: str, settings_payload) -> None:
        self._sync_camera_settings_store_path()
        serial_text = str(serial).strip()
        if not serial_text or HikCameraSettings is None:
            return

        payload = dict(settings_payload or {})
        if not payload:
            payload = self._camera_settings_store.load_for_serial(serial_text) or {}
        if not payload:
            return

        matched_roles = self._matching_runtime_roles_by_serial(serial_text)
        if not matched_roles:
            return

        if self._busy:
            self._pending_camera_settings_by_serial[serial_text] = payload
            self.logAppended.emit(f"[camera] queued runtime settings sync for {serial_text}")
            self._update_status(f"camera settings queued: {serial_text}")
            return

        if hasattr(self._light_controller, "set_camera_light_mode"):
            for role in matched_roles:
                camera_index = 1 if role == "cam1" else 2 if role == "cam2" else 0
                if camera_index <= 0:
                    continue
                self._light_controller.set_camera_light_mode(
                    camera_index,
                    light_source_mode_from_mapping(payload),
                )

        try:
            self._apply_camera_settings_now(serial_text, payload, matched_roles=matched_roles)
        except Exception as exc:
            self.errorOccurred.emit(f"Failed to sync runtime camera settings: {exc}")
            self.logAppended.emit(f"[camera] runtime settings sync failed for {serial_text}: {exc}")
            self._update_status(f"camera settings sync failed: {exc}")
            return

        self.logAppended.emit(f"[camera] runtime settings synced for {serial_text}")
        self._update_status(f"camera settings synced: {serial_text}")


    def trigger_camera(self, cam_index: int) -> None:
        """手动触发指定物理序号相机的检测流程（仅该路采图+检测，调试用）。"""
        self._sync_camera_settings_store_path()
        self._reload_runtime_context()
        if self._runner is None:
            self.logAppended.emit("[运行] 忽略触发：请先刷新并连接相机")
            self._write_release_log(
                event_type="invalid_trigger",
                result="ignored",
                message="运行链路未初始化",
            )
            self._update_status("尚未建立运行链路")
            return

        release_status_before = (
            self._permission_manager.status()
            if self._permission_manager is not None
            else None
        )
        self._set_busy(True)
        self._last_capture_paths = {}
        self._last_preview_frames = {}
        self._last_item_results_by_camera = {}
        self._last_runtime_result = self._build_pending_runtime_result(status="RUNNING")
        self._emit_runtime_context()
        self._update_status(f"手动调试：正在触发相机{cam_index}…")
        self.logAppended.emit(f"[运行] 手动触发相机{cam_index}")

        role_text = f"cam{int(cam_index)}"
        original_precheck = getattr(self._runner, "precheck_callback", None)
        try:
            self._runner.precheck_callback = lambda: self._precheck_for_roles([role_text])
            outcome = self._runner.on_single_camera_debug_trigger(cam_index)
        except Exception as exc:
            self.logAppended.emit(f"[运行] 触发异常：{exc}")
            self.triggerResultReady.emit("ERROR", str(exc))
            self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
            self._emit_runtime_context()
            self._update_status(f"运行异常：{exc}")
            return
        finally:
            self._runner.precheck_callback = original_precheck
            self._set_busy(False)

        if outcome is None:
            reason = (
                self._scheduler.can_accept_trigger().reason
                if self._scheduler is not None
                else "当前状态不允许触发"
            )
            self.logAppended.emit(f"[运行] 触发被拒绝：{reason}")
            self._write_release_log(
                event_type="invalid_trigger",
                result="blocked",
                message=reason or "当前状态不允许触发",
            )
            self.triggerResultReady.emit("BLOCKED", reason or "当前状态不允许触发")
            self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
            self._emit_runtime_context()
            self._update_status(reason or "当前状态不允许触发")
            return

        self._finalize_trigger_outcome(outcome, release_status_before)

    def trigger(self) -> None:
        """执行脚踏触发检测流程。"""
        self._reload_runtime_context()
        if self._runner is None:
            self.logAppended.emit("[运行] 忽略触发：请先刷新并连接相机")
            self._write_release_log(
                event_type="invalid_trigger",
                result="ignored",
                message="运行链路未初始化",
            )
            self._update_status("尚未建立运行链路")
            return

        release_status_before = (
            self._permission_manager.status()
            if self._permission_manager is not None
            else None
        )
        self._set_busy(True)
        self._last_capture_paths = {}
        self._last_preview_frames = {}
        self._last_item_results_by_camera = {}
        self._last_runtime_result = self._build_pending_runtime_result(status="RUNNING")
        self._emit_runtime_context()
        self._update_status("开始执行脚踏触发链路")

        try:
            outcome = self._runner.on_foot_trigger()
        except Exception as exc:
            self.logAppended.emit(f"[运行] 触发异常：{exc}")
            self.triggerResultReady.emit("ERROR", str(exc))
            self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
            self._emit_runtime_context()
            self._update_status(f"运行异常：{exc}")
            return
        finally:
            self._set_busy(False)

        if outcome is None:
            reason = (
                self._scheduler.can_accept_trigger().reason
                if self._scheduler is not None
                else "当前状态不允许触发"
            )
            self.logAppended.emit(f"[运行] 触发被拒绝：{reason}")
            self._write_release_log(
                event_type="invalid_trigger",
                result="blocked",
                message=reason or "当前状态不允许触发",
            )
            self.triggerResultReady.emit("BLOCKED", reason or "当前状态不允许触发")
            self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
            self._emit_runtime_context()
            self._update_status(reason or "当前状态不允许触发")
            return

        self._finalize_trigger_outcome(outcome, release_status_before)

    def release(self, password: str) -> None:
        """尝试放行 NG 锁定。"""
        if self._scheduler is None or self._permission_manager is None:
            self.infoOccurred.emit("请先连接相机并初始化运行链路")
            self._write_release_log(
                event_type="release_request",
                result="ignored",
                message="运行链路未初始化",
            )
            self._update_status("运行链路未初始化")
            return

        if not self._permission_manager.is_locked:
            self.logAppended.emit("[放行] 当前未锁定，无需放行")
            self._write_release_log(
                event_type="release_request",
                result="ignored",
                message="当前未锁定，无需放行",
            )
            self._update_status("当前未锁定，无需放行")
            return

        if self._scheduler.try_release_ng_lock(password):
            self.logAppended.emit("[放行] 密码正确，已放行一次，等待下一次有效检测消耗")
            self._write_release_log(
                event_type="release_request",
                result="granted",
                message="密码正确，已放行一次",
            )
            self._update_status("已放行一次，等待下一次有效检测")
            return

        self.logAppended.emit("[放行] 密码错误")
        self._write_release_log(
            event_type="release_request",
            result="rejected",
            message="密码错误",
        )
        self._update_status("放行失败：密码错误")

    def update_release_password(self, password: str) -> None:
        password_text = str(password).strip()
        if not password_text:
            return
        self._release_password = password_text
        if self._permission_manager is not None:
            self._permission_manager.update_password(password_text)
        self.logAppended.emit("[放行] 放行密码已更新")
        self._update_status("放行密码已更新")

    def refresh_all_status(self, message: Optional[str] = None) -> None:
        """
        主动推送所有状态 Signal（产品切换 / 会话清空后 MainWindow 调用）。
        """
        self._sync_camera_settings_store_path()
        self._reload_runtime_context()
        self._update_status(message)

    def connected_roles(self) -> list[str]:
        return self._connected_roles()

    # ------------------------------------------------------------------
    # 内部：构建运行链路
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 内部：检测回调
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 内部：统一状态推送（替代原 _refresh_runtime_status_ui）
    # ------------------------------------------------------------------

bindings.bind_runtime_controller(RuntimeController)
