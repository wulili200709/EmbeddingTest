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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import cv2

from PySide6 import QtCore

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

from .algorithm_controller import AlgorithmController
from .inspection_executor import InspectionExecutionRequest, InspectionExecutor
from .product_session import ProductSession
from .runtime_context import RuntimeContextProtocol

from domain import (
    RuntimeInspectionResult,
    aggregate_runtime_outcome,
    build_pending_result,
    recipe_name_from_path,
)
from infrastructure.camera_settings_store import (
    CameraSettingsStore,
    hik_settings_kwargs_from_mapping,
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


DEFAULT_RELEASE_PASSWORD = "1234"
DEFAULT_LIGHT_STABLE_MS = 20
RUNTIME_CAPTURE_POLICY_ALL = "all"
RUNTIME_CAPTURE_POLICY_NG_ONLY = "ng_only"


def normalize_capture_retention_policy(policy: object) -> str:
    return (
        RUNTIME_CAPTURE_POLICY_ALL
        if str(policy or "").strip().lower() == RUNTIME_CAPTURE_POLICY_ALL
        else RUNTIME_CAPTURE_POLICY_NG_ONLY
    )


def retained_capture_paths_for_policy(
    policy: object,
    final_result: object,
    capture_paths: Dict[str, str] | None,
) -> Dict[str, str]:
    normalized = normalize_capture_retention_policy(policy)
    sanitized = {
        str(role): str(path).strip()
        for role, path in dict(capture_paths or {}).items()
        if str(path or "").strip()
    }
    if normalized == RUNTIME_CAPTURE_POLICY_ALL:
        return sanitized
    if str(final_result or "").strip().upper() == "NG":
        return sanitized
    return {}


def delete_capture_artifacts(capture_paths: Dict[str, str] | None) -> None:
    for raw_path in dict(capture_paths or {}).values():
        image_text = str(raw_path or "").strip()
        if not image_text:
            continue
        image_path = Path(image_text)
        json_path = image_path.with_suffix(".json")
        for path in (image_path, json_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 无硬件时的占位控制器
# ---------------------------------------------------------------------------

class _UiOnlyLightController:
    def __init__(self) -> None:
        self.active_camera: Optional[int] = None

    def prepare_capture(self, camera_index: int) -> None:
        self.active_camera = int(camera_index)

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

    # ── 动作类 Signal ─────────────────────────────────────────────────────
    camerasEnumerated = QtCore.Signal(list)        # → runtime_page.set_available_cameras
    logAppended       = QtCore.Signal(str)         # → runtime_page.append_log
    busyChanged       = QtCore.Signal(bool)        # → runtime_page.set_busy
    triggerResultReady = QtCore.Signal(str, str)   # (result, detail) → runtime_page.set_final_result
    previewUpdated    = QtCore.Signal(str, str)    # (role, image_path) → MainWindow 转发
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
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._algo = algo
        self._runtime_context = runtime_context
        self._import_error = import_error
        self._release_password = release_password

        self._frame_lock = threading.RLock()
        self._inspect_lock = threading.RLock()
        self._last_capture_paths: Dict[str, str] = {}
        self._last_record_path: Optional[str] = None
        self._last_runtime_result: Optional[RuntimeInspectionResult] = None
        self._capture_retention_policy = RUNTIME_CAPTURE_POLICY_NG_ONLY
        self._camera_settings_store = CameraSettingsStore()

        self._camera_manager = None
        self._frame_grab_service = None
        self._permission_manager = None
        self._scheduler = None
        self._record_service = None
        self._release_log_service = None
        self._runner = None
        self._io_controller = None
        self._di_poller = None
        self._inspection_executor = InspectionExecutor(self._runtime_context)
        self._last_item_results_by_camera: Dict[str, list] = {}
        self._light_controller: _UiOnlyLightController = _UiOnlyLightController()
        self._tower_light_controller: _UiOnlyTowerLightController = _UiOnlyTowerLightController()
        self._busy = False
        self._pending_camera_settings_by_serial: Dict[str, dict] = {}

    def set_capture_retention_policy(self, policy: object) -> None:
        self._capture_retention_policy = normalize_capture_retention_policy(policy)

    def capture_retention_policy(self) -> str:
        return self._capture_retention_policy

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
            saved_settings = self._camera_settings_store.load_for_serial(serial)
            settings_by_role[role] = HikCameraSettings(
                **hik_settings_kwargs_from_mapping(
                    saved_settings,
                    default_trigger_mode="continuous",
                )
            )
        try:
            self._frame_grab_service.open_bound_cameras(bindings, settings_by_role=settings_by_role)
        except Exception as exc:
            self.disconnect(silent=True)
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
            saved_settings = self._camera_settings_store.load_for_serial(serial)
            settings_by_role[role] = HikCameraSettings(
                **hik_settings_kwargs_from_mapping(
                    saved_settings,
                    default_trigger_mode="continuous",
                )
            )
        try:
            self._frame_grab_service.open_bound_cameras(bindings, settings_by_role=settings_by_role)
        except Exception as exc:
            self.disconnect(silent=True)
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

    def disconnect(self, *, silent: bool = False) -> None:
        """断开所有相机并释放运行链路资源。"""
        self._stop_di_poller()
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
        self._io_controller = None
        self._di_poller = None
        self._last_capture_paths = {}
        self._last_record_path = None
        self._last_item_results_by_camera = {}
        self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
        self._light_controller = _UiOnlyLightController()
        self._tower_light_controller = _UiOnlyTowerLightController()
        self._busy = False
        self._pending_camera_settings_by_serial = {}

        self.cameraViewsCleared.emit()
        if not silent:
            self.logAppended.emit("[相机] 已断开")
        self._update_status("相机已断开")

    def apply_camera_settings_for_serial(self, serial: str, settings_payload) -> None:
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

        try:
            self._apply_camera_settings_now(serial_text, payload, matched_roles=matched_roles)
        except Exception as exc:
            self.errorOccurred.emit(f"Failed to sync runtime camera settings: {exc}")
            self.logAppended.emit(f"[camera] runtime settings sync failed for {serial_text}: {exc}")
            self._update_status(f"camera settings sync failed: {exc}")
            return

        self.logAppended.emit(f"[camera] runtime settings synced for {serial_text}")
        self._update_status(f"camera settings synced: {serial_text}")

    def _finalize_trigger_outcome(self, outcome, release_status_before) -> None:
        """脚踏 / 单相机调试触发成功后，统一写记录、刷新预览并发射结果信号。"""
        self._last_record_path = ""
        if self._record_service is not None:
            self._last_record_path = str(self._record_service.writer.file_path_for_date())
        current_capture_paths = dict(self._last_capture_paths)

        for role in ("cam1", "cam2"):
            path = current_capture_paths.get(role, "")
            self.previewUpdated.emit(role, path)

        self.recordPathChanged.emit(self._last_record_path or "-")
        retained_capture_paths = retained_capture_paths_for_policy(
            self._capture_retention_policy,
            outcome.final_result,
            current_capture_paths,
        )

        self._last_runtime_result = aggregate_runtime_outcome(
            product_name=self._session.current_product,
            recipe_name=recipe_name_from_path(self._session.line2dup_recipe_path),
            items=self._runtime_context.inspection_items,
            active_roles=self._connected_roles(),
            camera_outcomes=outcome.camera_outcomes,
            final_result=outcome.final_result,
            duration_ms=outcome.duration_ms,
            error_message=outcome.error_message,
            capture_paths=retained_capture_paths,
            item_results_by_camera=self._last_item_results_by_camera,
        )
        self._write_runtime_record(self._last_runtime_result)
        transient_capture_paths = {
            role: path
            for role, path in current_capture_paths.items()
            if path and retained_capture_paths.get(role) != path
        }
        delete_capture_artifacts(transient_capture_paths)
        self._last_capture_paths = dict(retained_capture_paths)
        if self._record_service is not None:
            self._last_record_path = str(self._record_service.writer.file_path_for_date())
            self.recordPathChanged.emit(self._last_record_path or "-")
        detail_text = self._last_runtime_result.summary_text()

        if (
            release_status_before is not None
            and release_status_before.has_pending_release
            and outcome.final_result != "PRECHECK_FAILED"
        ):
            self._write_release_log(
                event_type="release_consumed",
                result="consumed",
                message=f"放行已在有效检测开始时消耗，结果={outcome.final_result}",
            )
        if outcome.final_result == "NG":
            self._write_release_log(
                event_type="ng_lock",
                result="locked",
                message=detail_text,
            )
        elif outcome.error_message:
            self._write_release_log(
                event_type="runtime_error_lock",
                result="locked",
                message=outcome.error_message,
            )

        self.triggerResultReady.emit(outcome.final_result, detail_text)
        self.logAppended.emit(f"[运行] 结果={outcome.final_result}，{detail_text}")
        self._emit_runtime_context()
        self._update_status(f"本次结果：{outcome.final_result}")

    def trigger_camera(self, cam_index: int) -> None:
        """手动触发指定物理序号相机的检测流程（仅该路采图+检测，调试用）。"""
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
        self._last_runtime_result = self._build_pending_runtime_result(status="RUNNING")
        self._emit_runtime_context()
        self._update_status(f"手动调试：正在触发相机{cam_index}…")
        self.logAppended.emit(f"[运行] 手动触发相机{cam_index}")

        try:
            outcome = self._runner.on_single_camera_debug_trigger(cam_index)
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
        self._reload_runtime_context()
        self._update_status(message)

    def connected_roles(self) -> list[str]:
        return self._connected_roles()

    # ------------------------------------------------------------------
    # 内部：构建运行链路
    # ------------------------------------------------------------------

    def _rebuild_runner(self) -> bool:
        if self._import_error is not None:
            self._update_status(f"运行服务不可用：{self._import_error}")
            return False

        self._close_io_controller()
        self._last_record_path = None
        self._last_capture_paths = {}
        self._light_controller = _UiOnlyLightController()
        self._tower_light_controller = _UiOnlyTowerLightController()
        self._io_controller = self._try_create_io_controller()
        if self._io_controller is not None:
            self._light_controller = LightController(self._io_controller)
            self._tower_light_controller = TowerLightController(self._io_controller)
        self._camera_manager = HikCameraManager()
        self._frame_grab_service = FrameGrabService(self._camera_manager)
        self._permission_manager = PermissionManager(self._release_password)
        self._scheduler = InspectionScheduler(
            state_machine=RunStateMachine(),
            permission_manager=self._permission_manager,
        )
        records_dir = os.path.join(self._session.product_dir, "runtime_records")
        release_logs_dir = os.path.join(self._session.product_dir, "release_logs")
        os.makedirs(records_dir, exist_ok=True)
        os.makedirs(release_logs_dir, exist_ok=True)
        self._record_service = TestRecordService(CsvRecordWriter(records_dir))
        self._release_log_service = ReleaseLogService(CsvReleaseLogWriter(release_logs_dir))
        self._runner = InspectionRuntime(
            scheduler=self._scheduler,
            permission_manager=self._permission_manager,
            frame_grab_service=self._frame_grab_service,
            light_controller=self._light_controller,
            tower_light_controller=self._tower_light_controller,
            inspect_callback=self._inspect_frame,
            precheck_callback=self._precheck,
            role_to_camera_index={"cam1": 1, "cam2": 2},
            light_stable_ms=DEFAULT_LIGHT_STABLE_MS,
        )
        self._tower_light_controller.enter_waiting()
        return True

    def _try_create_io_controller(self):
        if DiMonitor is None or IoManager is None or LightController is None or TowerLightController is None:
            self.logAppended.emit("[IO] 真实 IO 控制器不可用，已降级为 UI 占位模式")
            return None

        mapping_path = Path(__file__).resolve().parent / "config" / "defaults" / "io_mapping.json"
        if not mapping_path.exists():
            self.logAppended.emit(f"[IO] 未找到 IO 映射配置：{mapping_path}，已降级为 UI 占位模式")
            return None

        board_config_path = self._find_nkio_config_path()
        if board_config_path is None:
            self.logAppended.emit("[IO] 未找到 nkio_config.ini，已降级为 UI 占位模式")
            return None

        try:
            controller = IoManager.from_config_file(board_config_path, mapping_path)
            controller.open()
        except Exception as exc:
            self.logAppended.emit(f"[IO] 初始化真实 IO 失败：{exc}，已降级为 UI 占位模式")
            return None

        self.logAppended.emit(f"[IO] 已启用真实 IO：{board_config_path}")
        return controller

    def _close_io_controller(self) -> None:
        if hasattr(self._tower_light_controller, "close"):
            try:
                self._tower_light_controller.close()
            except Exception:
                pass
        if hasattr(self._light_controller, "turn_off_all"):
            try:
                self._light_controller.turn_off_all()
            except Exception:
                pass
        if self._io_controller is not None:
            try:
                if hasattr(self._tower_light_controller, "all_off"):
                    self._tower_light_controller.all_off()
            except Exception:
                pass
            try:
                self._io_controller.close()
            except Exception:
                pass

    def _start_di_poller_if_available(self) -> None:
        self._stop_di_poller()
        if self._io_controller is None or DiMonitor is None:
            return
        try:
            poller = DiMonitor(self._io_controller, input_names=["foot_switch"], poll_interval_ms=20, debounce_ms=50)
            poller.add_rising_callback(self._on_foot_switch_rising)
            poller.start()
        except Exception as exc:
            self.logAppended.emit(f"[IO] 启动脚踏 DI 监听失败：{exc}")
            return
        self._di_poller = poller
        self.logAppended.emit("[IO] 已启动脚踏 DI 上升沿监听")

    def _stop_di_poller(self) -> None:
        if self._di_poller is None:
            return
        try:
            self._di_poller.stop()
        except Exception:
            pass
        self._di_poller = None

    def _on_foot_switch_rising(self, event) -> None:
        self.logAppended.emit(f"[脚踏] 检测到上升沿：{event.name}")
        QtCore.QMetaObject.invokeMethod(self, "_trigger_from_di", QtCore.Qt.QueuedConnection)

    @QtCore.Slot()
    def _trigger_from_di(self) -> None:
        self.trigger()

    def _find_nkio_config_path(self) -> Optional[Path]:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
            repo_root / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
            repo_root / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    # ------------------------------------------------------------------
    # 内部：检测回调
    # ------------------------------------------------------------------

    def _precheck(self) -> Tuple[bool, str]:
        if self._frame_grab_service is None or not self._frame_grab_service.roles():
            return False, "未连接相机"

        if self._runtime_context.loc_method != "line2dup":
            return False, "运行页当前仅支持 line2dup 定位链路"

        if not os.path.exists(self._session.line2dup_recipe_path):
            return False, "请先在工具页生成并保存 line2dup 模板"

        algorithm = self._runtime_context.current_algorithm()
        if not algorithm:
            return False, "请先在工具页选择工具"
        if self._algo.is_embedding_algorithm(algorithm):
            try:
                self._runtime_context.load_embedding_model(algorithm)
                if self._algo.model is not None:
                    self._algo.get_feat_net(
                        self._algo.model.backbone,
                        getattr(self._algo.model, "device", None),
                    )
            except Exception as exc:
                return False, f"加载模型失败：{exc}"
            if self._algo.model is None:
                return False, f"当前算法 {algorithm} 还没有训练好的模型"
        else:
            model_dict = self._algo.product_params.traditional_models.get(algorithm)
            if not isinstance(model_dict, dict):
                return False, f"传统算法 {algorithm} 尚未训练"

        return True, ""

    def _save_frame(self, role: str, frame) -> str:
        capture_dir = os.path.join(self._session.product_dir, "runtime_capture")
        os.makedirs(capture_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(capture_dir, f"{stamp}_{role}.png")
        if frame_to_bgr_image is None:
            raise RuntimeError("camera frame conversion service is unavailable")
        image = frame_to_bgr_image(frame)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if not cv2.imwrite(path, image):
            raise RuntimeError(f"保存运行采图失败：{path}")
        with self._frame_lock:
            self._last_capture_paths[role] = path
        return path

    def _inspect_frame(self, role: str, frame) -> "CameraInspectionOutcome":
        path = self._save_frame(role, frame)
        with self._inspect_lock:
            response = self._inspection_executor.execute(
                InspectionExecutionRequest(
                    camera_id=role,
                    image_path=path,
                    items=[item for item in self._runtime_context.inspection_items if item.camera_id == role],
                )
            )
            self._last_item_results_by_camera[role] = list(response.item_results)
        message = f"{os.path.basename(path)} pred={response.result}"
        if response.detail:
            message += f" {response.detail}"
        return CameraInspectionOutcome(
            role=role,
            result=response.result,
            message=message,
            match_ms=float(response.match_ms or 0.0),
            infer_ms=float(response.infer_ms or 0.0),
        )

    # ------------------------------------------------------------------
    # 内部：统一状态推送（替代原 _refresh_runtime_status_ui）
    # ------------------------------------------------------------------

    def _update_status(self, message: Optional[str] = None) -> None:
        """读取各服务对象的当前状态，逐字段发射 Signal。"""
        self._emit_runtime_context()
        if self._import_error is not None:
            self.runtimeStateChanged.emit("Unavailable")
            self.permissionStatusChanged.emit("-")
            self.connectionStatusChanged.emit("服务导入失败")
            self.towerLightStatusChanged.emit("-")
            self.statusMessageChanged.emit(
                message or f"运行服务不可用：{self._import_error}"
            )
            self.recordPathChanged.emit("-")
            return

        # 运行状态
        state_text = "WaitingTrigger"
        if self._scheduler is not None:
            state_value = self._scheduler.state
            state_text = state_value.value if hasattr(state_value, "value") else str(state_value)
        self.runtimeStateChanged.emit(state_text)

        # 放行状态
        if self._permission_manager is not None:
            release_status = self._permission_manager.status()
            if release_status.is_locked:
                permission_text = "NG锁定"
            elif release_status.has_pending_release:
                permission_text = "已放行，待消耗"
            elif release_status.release_consumed:
                permission_text = "已消耗一次放行"
            else:
                permission_text = "未锁定"
        else:
            permission_text = "未初始化"
        self.permissionStatusChanged.emit(permission_text)

        # 连接状态
        roles = self._connected_roles()
        connection_text = "已连接：" + ", ".join(roles) if roles else "未连接相机"
        self.connectionStatusChanged.emit(connection_text)

        # 塔灯状态
        tower_state = getattr(self._tower_light_controller, "state", "waiting")
        tower_text_map = {
            "waiting": "蓝灯等待",
            "inspecting": "检测中",
            "ok": "绿灯结果",
            "ng": "红灯结果",
        }
        self.towerLightStatusChanged.emit(tower_text_map.get(tower_state, str(tower_state)))

        # 主状态文字
        zh = _RUN_STATE_ZH_FOR_STATUS.get(state_text, state_text)
        self.statusMessageChanged.emit(message or zh)

        # 记录路径
        self.recordPathChanged.emit(self._last_record_path or "-")

    def _connected_roles(self) -> list[str]:
        roles: list[str] = []
        if self._frame_grab_service is not None:
            try:
                roles = [str(role) for role in self._frame_grab_service.roles()]
            except Exception:
                roles = []
        return roles

    def _matching_runtime_roles_by_serial(self, serial: str) -> list[str]:
        serial_text = str(serial).strip()
        if not serial_text or self._frame_grab_service is None:
            return []
        matched_roles: list[str] = []
        for role in self._connected_roles():
            try:
                device = self._frame_grab_service.get_device(role)
            except Exception:
                continue
            if str(getattr(device, "serial_number", "") or "").strip() == serial_text:
                matched_roles.append(role)
        return matched_roles

    def _apply_camera_settings_now(
        self,
        serial: str,
        settings_payload,
        *,
        matched_roles: Optional[list[str]] = None,
    ) -> None:
        if self._frame_grab_service is None or HikCameraSettings is None:
            return
        roles = matched_roles if matched_roles is not None else self._matching_runtime_roles_by_serial(serial)
        if not roles:
            return
        settings = HikCameraSettings(
            **hik_settings_kwargs_from_mapping(
                settings_payload,
                default_trigger_mode="continuous",
            )
        )
        for role in roles:
            device = self._frame_grab_service.get_device(role)
            device.apply_settings(settings)

    def reset_all_camera_triggers_off(self) -> None:
        if self._import_error is not None or HikCameraManager is None or HikCameraSettings is None:
            return

        manager = HikCameraManager()
        failures: list[str] = []
        try:
            infos = manager.enumerate_cameras()
            for info in infos:
                serial_text = str(info.serial_number or "").strip()
                if not serial_text:
                    continue
                saved_settings = self._camera_settings_store.load_for_serial(serial_text) or {}
                settings = HikCameraSettings(
                    **hik_settings_kwargs_from_mapping(
                        saved_settings,
                        default_trigger_mode="continuous",
                        force_trigger_mode="continuous",
                    )
                )
                try:
                    device = manager.open_camera(serial_text, settings=settings)
                    device.close()
                except Exception as exc:
                    failures.append(f"{serial_text}: {exc}")
        except Exception as exc:
            self.logAppended.emit(f"[camera] startup trigger reset failed: {exc}")
            self._update_status(f"startup trigger reset failed: {exc}")
            return
        finally:
            try:
                manager.close()
            except Exception:
                pass

        if failures:
            self.logAppended.emit("[camera] startup trigger reset partial failure")
            self._update_status("startup trigger reset partial failure")
            return

        self.logAppended.emit("[camera] startup trigger reset complete")
        self._update_status("startup camera trigger=off")

    """
    def _flush_pending_camera_settings(self) -> None:
        if not self._pending_camera_settings_by_serial:
            return
        pending = dict(self._pending_camera_settings_by_serial)
        self._pending_camera_settings_by_serial.clear()
        for serial_text, payload in pending.items():
            matched_roles = self._matching_runtime_roles_by_serial(serial_text)
            if not matched_roles:
                continue
            try:
                self._apply_camera_settings_now(serial_text, payload, matched_roles=matched_roles)
            except Exception as exc:
                self.errorOccurred.emit(f"鍚屾杩愯鐩告満鍙傛暟澶辫触锛歿exc}")
                self.logAppended.emit(f"[鐩告満] 鍚屾鍙傛暟澶辫触锛歿serial_text}: {exc}")
                self._update_status(f"鐩告満鍙傛暟鍚屾澶辫触锛歿exc}")
                continue
            self.logAppended.emit(f"[鐩告満] 宸插悓姝ヨ繍琛岀浉鏈哄弬鏁帮細{serial_text}")
            self._update_status(f"鐩告満鍙傛暟宸插悓姝ワ細{serial_text}")

    """
    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.busyChanged.emit(self._busy)
        if not self._busy:
            self._flush_pending_camera_settings()

    def _flush_pending_camera_settings(self) -> None:
        if not self._pending_camera_settings_by_serial:
            return
        pending = dict(self._pending_camera_settings_by_serial)
        self._pending_camera_settings_by_serial.clear()
        for serial_text, payload in pending.items():
            matched_roles = self._matching_runtime_roles_by_serial(serial_text)
            if not matched_roles:
                continue
            try:
                self._apply_camera_settings_now(serial_text, payload, matched_roles=matched_roles)
            except Exception as exc:
                self.errorOccurred.emit(f"Failed to sync runtime camera settings: {exc}")
                self.logAppended.emit(f"[camera] runtime settings sync failed for {serial_text}: {exc}")
                self._update_status(f"camera settings sync failed: {exc}")
                continue
            self.logAppended.emit(f"[camera] runtime settings synced for {serial_text}")
            self._update_status(f"camera settings synced: {serial_text}")

    def _current_item_signature(self) -> list[tuple[str, str, str, str, bool]]:
        return [
            (
                str(item.item_id),
                str(item.display_name),
                str(item.camera_id),
                str(item.roi_label),
                bool(item.enabled),
            )
            for item in self._runtime_context.inspection_items
        ]

    def _result_item_signature(self) -> list[tuple[str, str, str, str, bool]]:
        if self._last_runtime_result is None:
            return []
        return [
            (
                str(item.item_id),
                str(item.display_name),
                str(item.camera_id),
                str(item.roi_label),
                bool(item.enabled),
            )
            for item in self._last_runtime_result.item_results
        ]

    def _runtime_result_is_stale(self) -> bool:
        if self._last_runtime_result is None:
            return True
        if str(self._last_runtime_result.product_name or "") != str(self._session.current_product or ""):
            return True
        if set(self._last_runtime_result.camera_results.keys()) != set(self._connected_roles()):
            return True
        return self._current_item_signature() != self._result_item_signature()

    def _emit_runtime_context(self) -> None:
        if self._runtime_result_is_stale():
            self._last_runtime_result = self._build_pending_runtime_result(status="PENDING")
        self.productNameChanged.emit(self._session.current_product)
        self.activeCameraRolesChanged.emit(self._connected_roles())
        self.inspectionItemsChanged.emit(self._last_runtime_result.item_rows())
        self.cameraResultsChanged.emit(self._last_runtime_result.camera_result_map())
        self.durationChanged.emit(int(getattr(self._last_runtime_result, "duration_ms", 0) or 0))
        self.timingBreakdownChanged.emit(self._last_runtime_result.timing_breakdown())

    def _build_pending_runtime_result(self, *, status: str) -> RuntimeInspectionResult:
        return build_pending_result(
            product_name=self._session.current_product,
            recipe_name=recipe_name_from_path(self._session.line2dup_recipe_path),
            items=self._runtime_context.inspection_items,
            active_roles=self._connected_roles(),
            status=status,
        )

    def _write_release_log(self, *, event_type: str, result: str, message: str = "") -> None:
        if self._release_log_service is None:
            return
        try:
            self._release_log_service.write_event(
                product_name=self._session.current_product,
                recipe_name=recipe_name_from_path(self._session.line2dup_recipe_path),
                event_type=event_type,
                result=result,
                message=message,
                runtime_state=self._current_runtime_state_text(),
            )
        except Exception as exc:
            self.logAppended.emit(f"[放行] 写入独立日志失败：{exc}")

    def _write_runtime_record(self, runtime_result: RuntimeInspectionResult) -> None:
        if self._record_service is None:
            return
        try:
            self._record_service.write_product_result(
                product_name=runtime_result.product_name,
                recipe_name=runtime_result.recipe_name,
                final_result=runtime_result.final_result,
                camera1_result=runtime_result.camera_results.get("cam1", None).result
                if runtime_result.camera_results.get("cam1") is not None
                else "",
                camera2_result=runtime_result.camera_results.get("cam2", None).result
                if runtime_result.camera_results.get("cam2") is not None
                else "",
                duration_ms=runtime_result.duration_ms,
                is_error=runtime_result.is_system_error,
                error_message=runtime_result.error_message,
                lock_required=(runtime_result.final_result == "NG"),
                release_required=(runtime_result.final_result == "NG"),
                release_result="pending" if runtime_result.final_result == "NG" else "",
                extra_fields=runtime_result.to_record_extra_fields(),
            )
        except Exception as exc:
            self.logAppended.emit(f"[运行] 写入运行记录失败：{exc}")

    def _reload_runtime_context(self) -> None:
        try:
            self._runtime_context.reload()
        except Exception as exc:
            self.logAppended.emit(f"[运行] 刷新运行上下文失败：{exc}")

    def _current_runtime_state_text(self) -> str:
        if self._scheduler is None:
            return "Uninitialized"
        state_value = self._scheduler.state
        return state_value.value if hasattr(state_value, "value") else str(state_value)
