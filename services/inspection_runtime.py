from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

from .camera import FrameGrabService, HikFrame
from .inspection_scheduler import InspectionScheduler
from .permission_manager import PermissionManager
from .run_state import RunState


@dataclass(frozen=True)
class CameraInspectionOutcome:
    role: str
    result: str
    message: str = ""
    capture_ms: float = 0.0
    match_ms: float = 0.0
    infer_ms: float = 0.0


@dataclass(frozen=True)
class FinalInspectionOutcome:
    final_result: str
    camera_outcomes: dict[str, CameraInspectionOutcome]
    duration_ms: int
    error_message: str = ""


InspectionCallback = Callable[[str, HikFrame], CameraInspectionOutcome]
PrecheckCallback = Callable[[], tuple[bool, str]]


class InspectionRuntime:
    """Minimal V1 runtime chain without UI dependency."""

    def __init__(
        self,
        *,
        scheduler: InspectionScheduler,
        permission_manager: PermissionManager,
        frame_grab_service: FrameGrabService,
        light_controller,
        tower_light_controller,
        inspect_callback: InspectionCallback,
        precheck_callback: PrecheckCallback | None = None,
        role_to_camera_index: dict[str, int] | None = None,
        light_stable_ms: int = 0,
    ) -> None:
        self.scheduler = scheduler
        self.permission_manager = permission_manager
        self.frame_grab_service = frame_grab_service
        self.light_controller = light_controller
        self.tower_light_controller = tower_light_controller
        self.inspect_callback = inspect_callback
        self.precheck_callback = precheck_callback
        self.role_to_camera_index = role_to_camera_index or {"cam1": 1, "cam2": 2}
        self.light_stable_ms = max(0, int(light_stable_ms))

    def on_foot_trigger(self) -> FinalInspectionOutcome | None:
        return self._run_inspection_trigger(self._ordered_roles())

    def on_single_camera_debug_trigger(self, camera_index: int) -> FinalInspectionOutcome | None:
        """仅采集并检测指定物理序号相机对应的路（用于运行页手动调试）。"""
        wanted = int(camera_index)
        roles = [
            r
            for r in self._ordered_roles()
            if self.role_to_camera_index.get(r) == wanted
        ]
        return self._run_inspection_trigger(
            roles,
            no_roles_message=(
                f"当前未连接相机{wanted}或该路未在运行链路中启用"
            ),
        )

    def _run_inspection_trigger(
        self,
        roles: list[str],
        *,
        no_roles_message: str | None = None,
    ) -> FinalInspectionOutcome | None:
        decision = self.scheduler.begin_precheck()
        if not decision.allowed:
            return None

        if self.precheck_callback is not None:
            ok, reason = self.precheck_callback()
            if not ok:
                self.scheduler.on_precheck_failed()
                return FinalInspectionOutcome(
                    final_result="PRECHECK_FAILED",
                    camera_outcomes={},
                    duration_ms=0,
                    error_message=reason,
                )

        if not roles:
            return FinalInspectionOutcome(
                final_result="PRECHECK_FAILED",
                camera_outcomes={},
                duration_ms=0,
                error_message=no_roles_message or "未连接相机或角色映射不存在",
            )

        started_at = time.perf_counter()
        try:
            self.tower_light_controller.enter_inspecting()
            camera_outcomes = self._capture_and_inspect_for_roles(
                roles, timeout_ms=1000
            )
            self.scheduler.on_aggregating_started()

            final_ok = all(outcome.result == "OK" for outcome in camera_outcomes.values())
            duration_ms = int((time.perf_counter() - started_at) * 1000.0)

            self.scheduler.on_completed(final_ok=final_ok)
            if final_ok:
                self.tower_light_controller.show_ok()
            else:
                self.tower_light_controller.show_ng()

            final_outcome = FinalInspectionOutcome(
                final_result="OK" if final_ok else "NG",
                camera_outcomes=camera_outcomes,
                duration_ms=duration_ms,
            )
            return final_outcome
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000.0)
            self.scheduler.on_error(lock_as_ng=True)
            self.tower_light_controller.show_ng()
            final_outcome = FinalInspectionOutcome(
                final_result="NG",
                camera_outcomes={},
                duration_ms=duration_ms,
                error_message=str(exc),
            )
            return final_outcome

    @property
    def run_state(self) -> RunState:
        return self.scheduler.state

    def _capture_and_inspect_pipeline(
        self,
        *,
        timeout_ms: int,
    ) -> dict[str, CameraInspectionOutcome]:
        """
        第一版双相机推荐时序：
          1. cam1 采图完成后立即提交检测线程
          2. 不等待 cam1 检测完成，直接继续 cam2 采图
          3. 全部采图结束后，再统一等待所有检测结果

        对单相机场景同样成立：
          - 只有 cam1 时，拍完后立即提交检测，然后直接进入等待结果
        """
        return self._capture_and_inspect_for_roles(
            self._ordered_roles(), timeout_ms=timeout_ms
        )

    def _capture_and_inspect_for_roles(
        self,
        roles: list[str],
        *,
        timeout_ms: int,
    ) -> dict[str, CameraInspectionOutcome]:
        if not roles:
            return {}

        futures: dict[str, Future[CameraInspectionOutcome]] = {}
        capture_ms_by_role: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(roles))) as executor:
            for role in roles:
                camera_index = self.role_to_camera_index.get(role)
                if camera_index is None:
                    raise RuntimeError(f"missing camera index mapping for role={role}")

                capture_t0 = time.perf_counter()
                self.light_controller.prepare_capture(camera_index)
                requires_stable_delay = True
                requires_stable_delay_getter = getattr(
                    self.light_controller,
                    "requires_stable_delay",
                    None,
                )
                if callable(requires_stable_delay_getter):
                    requires_stable_delay = bool(
                        requires_stable_delay_getter(camera_index)
                    )
                if self.light_stable_ms > 0 and requires_stable_delay:
                    time.sleep(self.light_stable_ms / 1000.0)
                self.scheduler.on_capture_started(camera_index)
                frame = self.frame_grab_service.capture_once(role, timeout_ms=timeout_ms)
                self.light_controller.finish_capture(camera_index)
                capture_ms_by_role[role] = (time.perf_counter() - capture_t0) * 1000.0
                futures[role] = executor.submit(self.inspect_callback, role, frame)

            self.scheduler.on_inspecting_started()
            outcomes: dict[str, CameraInspectionOutcome] = {}
            for role, future in futures.items():
                outcomes[role] = replace(
                    future.result(),
                    capture_ms=float(capture_ms_by_role.get(role, 0.0) or 0.0),
                )
            return outcomes

    def _ordered_roles(self) -> list[str]:
        roles = [str(role) for role in self.frame_grab_service.roles()]
        return sorted(
            roles,
            key=lambda role: (
                self.role_to_camera_index.get(role, 999),
                role,
            ),
        )

