from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

from PySide6 import QtCore, QtGui, QtWidgets

from app_paths import packaged_embedding_test_root, writable_embedding_test_root
from application import AlgorithmController, ProductSession
from line2dup.core import locator as line2dup_locator
import algorithms.proxy as qr_core
from ui.roi_overlay_colors import overlay_style_for_label, search_region_style


_RUNTIME_OVERLAY_WIDTH_MULTIPLIER = 3.0


def embedding_test_root(anchor_file: str) -> Path:
    return packaged_embedding_test_root(anchor_file)


_CAMERA_ROLE_RE = re.compile(r"(?:^|[_-])(cam[12])(?=[_.-]|$)", re.IGNORECASE)


def _camera_role_from_path(path: str) -> str:
    match = _CAMERA_ROLE_RE.search(Path(path).name)
    if not match:
        return "cam1"
    return str(match.group(1) or "cam1").lower()


def default_session_dir(anchor_file: str) -> str:
    return str(writable_embedding_test_root(anchor_file) / ".qr_session")


def build_default_session_and_algo(anchor_file: str) -> tuple[ProductSession, AlgorithmController]:
    session = ProductSession(default_session_dir(anchor_file))
    session.load()
    session.switch_product(session.current_product)
    algo = AlgorithmController()
    return session, algo


def detect_runtime_import_error() -> Optional[Exception]:
    try:
        from services import CameraInspectionOutcome  # noqa: F401

        return None
    except Exception as exc:
        return exc


def connect_runtime_refresh_sources(tool_page, runtime_ctrl, *, session_loaded_message: str) -> None:
    tool_page.sessionLoaded.connect(
        lambda: runtime_ctrl.refresh_all_status(session_loaded_message)
    )
    tool_page.inspectionItemsChanged.connect(
        lambda: runtime_ctrl.refresh_all_status("检测项配置已同步")
    )


def connect_runtime_page(runtime_page, runtime_ctrl) -> None:
    runtime_page.refreshCamerasRequested.connect(runtime_ctrl.refresh_cameras)
    runtime_page.connectCamerasRequested.connect(runtime_ctrl.connect_cameras)
    runtime_page.disconnectCamerasRequested.connect(runtime_ctrl.disconnect)
    runtime_page.triggerRequested.connect(runtime_ctrl.trigger)
    runtime_page.triggerCameraRequested.connect(runtime_ctrl.trigger_camera)
    runtime_page.releaseRequested.connect(runtime_ctrl.release)

    runtime_ctrl.runtimeStateChanged.connect(runtime_page.set_runtime_state)
    runtime_ctrl.productNameChanged.connect(runtime_page.set_current_product)
    runtime_ctrl.permissionStatusChanged.connect(runtime_page.set_permission_status)
    runtime_ctrl.connectionStatusChanged.connect(runtime_page.set_connection_status)
    runtime_ctrl.towerLightStatusChanged.connect(runtime_page.set_tower_light_status)
    runtime_ctrl.statusMessageChanged.connect(runtime_page.set_runtime_status)
    runtime_ctrl.recordPathChanged.connect(runtime_page.set_record_path)
    runtime_ctrl.camerasEnumerated.connect(runtime_page.set_available_cameras)
    runtime_ctrl.logAppended.connect(runtime_page.append_log)
    runtime_ctrl.busyChanged.connect(runtime_page.set_busy)
    runtime_ctrl.triggerResultReady.connect(runtime_page.set_final_result)
    runtime_ctrl.cameraResultsChanged.connect(runtime_page.set_camera_results)
    runtime_ctrl.durationChanged.connect(runtime_page.set_duration_ms)
    runtime_ctrl.timingBreakdownChanged.connect(runtime_page.set_timing_breakdown)
    runtime_ctrl.cameraViewsCleared.connect(runtime_page.clear_camera_views)
    runtime_ctrl.activeCameraRolesChanged.connect(runtime_page.set_active_camera_roles)
    runtime_ctrl.inspectionItemsChanged.connect(runtime_page.set_inspection_items)


def connect_runtime_dialogs(window: QtWidgets.QWidget, runtime_ctrl) -> None:
    runtime_ctrl.warningOccurred.connect(
        lambda msg: QtWidgets.QMessageBox.warning(window, "运行页", msg)
    )
    runtime_ctrl.errorOccurred.connect(
        lambda msg: QtWidgets.QMessageBox.critical(window, "运行错误", msg)
    )
    runtime_ctrl.infoOccurred.connect(
        lambda msg: QtWidgets.QMessageBox.information(window, "运行页", msg)
    )


def update_runtime_preview(runtime_page, role: str, path: str) -> None:
    if hasattr(runtime_page, "set_camera_source_path"):
        runtime_page.set_camera_source_path(role, path)
    if path and Path(path).exists():
        roi_statuses = {}
        if hasattr(runtime_page, "roi_statuses_for_camera"):
            roi_statuses = dict(runtime_page.roi_statuses_for_camera(role) or {})
        runtime_page.set_camera_pixmap(
            role,
            _render_runtime_overlay_pixmap(path, roi_statuses=roi_statuses),
        )
        return
    runtime_page.set_camera_pixmap(
        role,
        None,
        placeholder=f"{role.upper()} 画面占位",
    )


def _render_runtime_overlay_pixmap(
    path: str,
    *,
    roi_statuses: Optional[dict[str, str]] = None,
) -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(path)
    if pixmap.isNull():
        return pixmap

    canvas = QtGui.QPixmap(pixmap)
    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    _draw_runtime_search_region(painter, path)
    _draw_runtime_roi_shapes(painter, path, roi_statuses=roi_statuses)

    painter.end()
    return canvas


def _draw_runtime_search_region(painter: QtGui.QPainter, path: str) -> None:
    try:
        product_dir = str(Path(path).resolve().parent.parent)
        recipe = line2dup_locator.load_recipe_for_product(product_dir, _camera_role_from_path(path))
    except Exception:
        return

    points = [
        (float(pt[0]), float(pt[1]))
        for pt in (getattr(recipe, "search_points", None) or [])
        if isinstance(pt, (list, tuple)) and len(pt) >= 2
    ]
    if len(points) < 2:
        return

    color, width, dash = search_region_style()
    pen = QtGui.QPen(color)
    pen.setWidthF(width)
    pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)

    if str(getattr(recipe, "search_shape_type", "rectangle") or "rectangle") == "rectangle" and len(points) == 2:
        (x0, y0), (x1, y1) = points[:2]
        rect = QtCore.QRectF(
            min(x0, x1),
            min(y0, y1),
            abs(x1 - x0),
            abs(y1 - y0),
        )
        painter.drawRect(rect)
        return

    polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])
    painter.drawPolygon(polygon)


def _draw_runtime_roi_shapes(
    painter: QtGui.QPainter,
    path: str,
    *,
    roi_statuses: Optional[dict[str, str]] = None,
) -> None:
    jpath = qr_core.labelme_json_of_image(path)
    if not Path(jpath).exists():
        return
    roi_statuses = {
        str(label).strip(): str(status or "").strip().lower()
        for label, status in dict(roi_statuses or {}).items()
        if str(label).strip()
    }

    def draw_shape(label: str, color: QtGui.QColor, *, width: float = 2.0, dash: bool = False) -> bool:
        runtime_width = max(1.0, float(width) * _RUNTIME_OVERLAY_WIDTH_MULTIPLIER)
        polygon = qr_core.try_read_polygon_points_from_labelme(jpath, label)
        if polygon and len(polygon) >= 3:
            pen = QtGui.QPen(color)
            pen.setWidthF(runtime_width)
            pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in polygon]))
            return True

        xywh = qr_core.try_read_xywh_from_labelme(jpath, label)
        if xywh:
            x, y, w, h = xywh
            pen = QtGui.QPen(color)
            pen.setWidthF(runtime_width)
            pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(QtCore.QRectF(float(x), float(y), float(w), float(h)))
            return True
        return False

    seen_labels: set[str] = set()
    for label in qr_core.sorted_label_names_from_labelme(jpath, label_prefix="roi"):
        seen_labels.add(label)
        color, width, dash = overlay_style_for_label(label, status=roi_statuses.get(label, ""))
        draw_shape(label, color, width=width, dash=dash)

    for label in ["anchor", "roi", "anchor_mask"]:
        if label in seen_labels:
            continue
        color, width, dash = overlay_style_for_label(label, status=roi_statuses.get(label, ""))
        draw_shape(label, color, width=width, dash=dash)
