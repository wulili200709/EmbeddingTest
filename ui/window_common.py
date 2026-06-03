from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from common.app_paths import packaged_embedding_test_root, writable_embedding_test_root
from application import AlgorithmController, ProductSession
from application.runtime.preview_frame import RuntimePreviewFrame, RuntimePreviewShape
from application.runtime.preview_frame import read_exported_runtime_preview_measurements
from shape.core import locator as shape_locator
import algorithms.proxy as qr_core
from ui.i18n import tr
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
        lambda: runtime_ctrl.refresh_all_status(tr("status.inspection_items_synced"))
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
        lambda msg: QtWidgets.QMessageBox.warning(window, tr("workspace.runtime"), msg)
    )
    runtime_ctrl.errorOccurred.connect(
        lambda msg: QtWidgets.QMessageBox.critical(window, tr("runtime.state.Error"), msg)
    )
    runtime_ctrl.infoOccurred.connect(
        lambda msg: QtWidgets.QMessageBox.information(window, tr("workspace.runtime"), msg)
    )


def update_runtime_preview(runtime_page, role: str, source: object) -> None:
    display_size = _runtime_preview_display_size(runtime_page, role)
    if hasattr(runtime_page, "set_camera_preview_source"):
        runtime_page.set_camera_preview_source(role, source)
    if isinstance(source, RuntimePreviewFrame):
        roi_statuses = {}
        if hasattr(runtime_page, "roi_statuses_for_camera"):
            roi_statuses = dict(runtime_page.roi_statuses_for_camera(role) or {})
        runtime_page.set_camera_pixmap(
            role,
            _render_runtime_overlay_pixmap(
                source,
                roi_statuses=roi_statuses,
                display_size=display_size,
            ),
        )
        return

    path = str(source or "").strip()
    if path and Path(path).exists():
        roi_statuses = {}
        if hasattr(runtime_page, "roi_statuses_for_camera"):
            roi_statuses = dict(runtime_page.roi_statuses_for_camera(role) or {})
        runtime_page.set_camera_pixmap(
            role,
            _render_runtime_overlay_pixmap(
                path,
                roi_statuses=roi_statuses,
                display_size=display_size,
            ),
        )
        return
    runtime_page.set_camera_pixmap(
        role,
        None,
        placeholder=tr("runtime.camera_placeholder", role=role.upper()),
    )


def _runtime_preview_display_size(runtime_page, role: str) -> QtCore.QSize | None:
    role_text = str(role or "cam1").strip()
    view_name = "view_cam2" if role_text == "cam2" else "view_cam1"
    view = getattr(runtime_page, view_name, None)
    if view is None or not hasattr(view, "size"):
        return None
    size = view.size()
    if size.width() <= 0 or size.height() <= 0:
        return None
    return QtCore.QSize(size)


def _render_runtime_overlay_pixmap(
    source: str | RuntimePreviewFrame,
    *,
    roi_statuses: Optional[dict[str, str]] = None,
    display_size: QtCore.QSize | None = None,
) -> QtGui.QPixmap:
    pixmap = _runtime_source_pixmap(source)
    if pixmap.isNull():
        return pixmap

    canvas = QtGui.QPixmap(pixmap)
    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    _draw_runtime_search_region(painter, source)
    _draw_runtime_roi_shapes(painter, source, roi_statuses=roi_statuses)
    _draw_runtime_measurements(painter, source, canvas.size(), display_size=display_size)

    painter.end()
    return canvas


def _runtime_source_pixmap(source: str | RuntimePreviewFrame) -> QtGui.QPixmap:
    if isinstance(source, RuntimePreviewFrame):
        image = np.asarray(source.image_bgr)
        if image.ndim == 2:
            gray = np.ascontiguousarray(image)
            qimage = QtGui.QImage(
                gray.data,
                int(gray.shape[1]),
                int(gray.shape[0]),
                int(gray.strides[0]),
                QtGui.QImage.Format_Grayscale8,
            )
            return QtGui.QPixmap.fromImage(qimage.copy())
        rgb = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
        qimage = QtGui.QImage(
            rgb.data,
            int(rgb.shape[1]),
            int(rgb.shape[0]),
            int(rgb.strides[0]),
            QtGui.QImage.Format_RGB888,
        )
        return QtGui.QPixmap.fromImage(qimage.copy())
    return QtGui.QPixmap(str(source or ""))


def _draw_runtime_search_region(painter: QtGui.QPainter, source: str | RuntimePreviewFrame) -> None:
    if isinstance(source, RuntimePreviewFrame):
        product_dir = str(source.product_dir or "").strip()
        camera_role = str(source.camera_role or source.role or "cam1").strip() or "cam1"
    else:
        path = str(source or "").strip()
        product_dir = str(Path(path).resolve().parent.parent) if path else ""
        camera_role = _camera_role_from_path(path)
    if not product_dir:
        return
    try:
        recipe = shape_locator.load_recipe_for_product(product_dir, camera_role)
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
    source: str | RuntimePreviewFrame,
    *,
    roi_statuses: Optional[dict[str, str]] = None,
) -> None:
    roi_statuses = {
        str(label).strip(): str(status or "").strip().lower()
        for label, status in dict(roi_statuses or {}).items()
        if str(label).strip()
    }
    shapes = _runtime_source_shapes(source)
    if not shapes:
        return

    shape_by_label = {shape.label: shape for shape in shapes if str(shape.label).strip()}

    def draw_shape(label: str, color: QtGui.QColor, *, width: float = 2.0, dash: bool = False) -> bool:
        shape = shape_by_label.get(label)
        if shape is None:
            return False
        runtime_width = max(1.0, float(width) * _RUNTIME_OVERLAY_WIDTH_MULTIPLIER)
        if shape.shape_type == "polygon" and len(shape.points) >= 3:
            pen = QtGui.QPen(color)
            pen.setWidthF(runtime_width)
            pen.setStyle(QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in shape.points]))
            return True

        xywh = _runtime_shape_xywh(shape)
        if xywh is not None:
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
    for label in _sorted_runtime_shape_labels(shapes):
        seen_labels.add(label)
        color, width, dash = overlay_style_for_label(label, status=roi_statuses.get(label, ""))
        draw_shape(label, color, width=width, dash=dash)

    for label in ["anchor", "roi", "anchor_mask"]:
        if label in seen_labels:
            continue
        color, width, dash = overlay_style_for_label(label, status=roi_statuses.get(label, ""))
        draw_shape(label, color, width=width, dash=dash)


def _point_tuple(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _segment_tuple(value: object) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    p0 = _point_tuple(value[0])
    p1 = _point_tuple(value[1])
    if p0 is None or p1 is None:
        return None
    return p0, p1


def _draw_runtime_measurements(
    painter: QtGui.QPainter,
    source: str | RuntimePreviewFrame,
    source_size: QtCore.QSize,
    *,
    display_size: QtCore.QSize | None = None,
) -> None:
    overlay_scale = _runtime_measurement_overlay_scale(source_size, display_size)
    for measurement in _runtime_source_measurements(source):
        measurement_type = str(measurement.get("type", "") or "").strip()
        pred = str(measurement.get("pred", "") or "").strip().upper()
        color = QtGui.QColor("#22c55e" if pred == "OK" else "#ff4040" if pred == "NG" else "#f97316")
        if measurement_type in {"line_distance", "line_distance_ref_normal"}:
            segment_a = _segment_tuple(measurement.get("line_a_segment"))
            if segment_a is None:
                line_a = measurement.get("line_a")
                if isinstance(line_a, dict):
                    segment_a = _segment_tuple(line_a.get("line_segment"))
            if segment_a is not None:
                _draw_runtime_segment(painter, segment_a, color, overlay_scale=overlay_scale)
            segment_b = _segment_tuple(measurement.get("line_b_segment"))
            if segment_b is None:
                line_b = measurement.get("line_b")
                if isinstance(line_b, dict):
                    segment_b = _segment_tuple(line_b.get("line_segment"))
            if segment_b is not None:
                _draw_runtime_segment(painter, segment_b, color, overlay_scale=overlay_scale)
            dimension_segment = _segment_tuple(measurement.get("dimension_segment"))
            if dimension_segment is not None:
                _draw_runtime_dimension(
                    painter,
                    dimension_segment,
                    color,
                    str(measurement.get("label", "") or ""),
                    overlay_scale=overlay_scale,
                )
            continue
        segment = _segment_tuple(measurement.get("line_segment"))
        if segment is not None:
            _draw_runtime_segment(painter, segment, color, overlay_scale=overlay_scale)


def _runtime_measurement_overlay_scale(
    source_size: QtCore.QSize,
    display_size: QtCore.QSize | None,
) -> float:
    source_w = max(1.0, float(source_size.width()))
    source_h = max(1.0, float(source_size.height()))
    if display_size is not None and display_size.width() > 0 and display_size.height() > 0:
        display_scale = min(float(display_size.width()) / source_w, float(display_size.height()) / source_h)
        if display_scale > 0:
            return max(1.0, min(12.0, 1.0 / display_scale))
    return max(1.0, min(8.0, max(source_w / 700.0, source_h / 500.0)))


def _draw_runtime_segment(
    painter: QtGui.QPainter,
    segment: tuple[tuple[float, float], tuple[float, float]],
    color: QtGui.QColor,
    *,
    overlay_scale: float = 1.0,
) -> None:
    pen = QtGui.QPen(color)
    pen.setWidthF(max(3.0, 3.0 * float(overlay_scale)))
    pen.setStyle(QtCore.Qt.SolidLine)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    (x0, y0), (x1, y1) = segment
    painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))


def _draw_runtime_dimension(
    painter: QtGui.QPainter,
    segment: tuple[tuple[float, float], tuple[float, float]],
    color: QtGui.QColor,
    text: str,
    *,
    overlay_scale: float = 1.0,
) -> None:
    (x0, y0), (x1, y1) = segment
    scale = max(1.0, float(overlay_scale))
    painter.setPen(QtGui.QPen(color, max(3.0, 3.0 * scale), QtCore.Qt.SolidLine))
    painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

    dx = x1 - x0
    dy = y1 - y0
    length = max(1.0, float((dx * dx + dy * dy) ** 0.5))
    ux = dx / length
    uy = dy / length
    nx = -uy
    ny = ux
    head = max(12.0 * scale, min(24.0 * scale, length * 0.2))
    painter.setBrush(QtGui.QBrush(color))
    for x, y, sign in ((x0, y0, 1.0), (x1, y1, -1.0)):
        back_x = x + sign * ux * head
        back_y = y + sign * uy * head
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(x, y),
                    QtCore.QPointF(back_x + nx * head * 0.45, back_y + ny * head * 0.45),
                    QtCore.QPointF(back_x - nx * head * 0.45, back_y - ny * head * 0.45),
                ]
            )
        )

    if not text:
        return
    tx = (x0 + x1) * 0.5 + nx * 16.0 * scale
    ty = (y0 + y1) * 0.5 + ny * 16.0 * scale
    font = painter.font()
    font.setPixelSize(max(18, int(round(18.0 * scale))))
    font.setBold(True)
    painter.setFont(font)
    metrics = QtGui.QFontMetrics(font)
    rect = metrics.boundingRect(text)
    box = QtCore.QRectF(
        tx - rect.width() / 2 - 7 * scale,
        ty - rect.height() / 2 - 5 * scale,
        rect.width() + 14 * scale,
        rect.height() + 10 * scale,
    )
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 180)))
    painter.drawRoundedRect(box, 4, 4)
    painter.setPen(QtGui.QPen(color))
    painter.drawText(box, QtCore.Qt.AlignCenter, text)


def _runtime_source_measurements(source: str | RuntimePreviewFrame) -> tuple[dict, ...]:
    if isinstance(source, RuntimePreviewFrame):
        return tuple(dict(item) for item in tuple(getattr(source, "measurements", ()) or ()) if isinstance(item, dict))
    path = str(source or "").strip()
    if not path:
        return ()
    return read_exported_runtime_preview_measurements(path)


def _runtime_source_shapes(source: str | RuntimePreviewFrame) -> tuple[RuntimePreviewShape, ...]:
    if isinstance(source, RuntimePreviewFrame):
        return tuple(source.roi_shapes or ())
    path = str(source or "").strip()
    if not path:
        return ()
    jpath = qr_core.labelme_json_of_image(path)
    if not Path(jpath).exists():
        return ()
    shapes: list[RuntimePreviewShape] = []
    for shape in qr_core.list_shapes_from_labelme(jpath):
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        points = []
        for point in shape.get("points", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except Exception:
                continue
        if not points:
            continue
        shapes.append(
            RuntimePreviewShape(
                label=label,
                shape_type=str(shape.get("shape_type", "rectangle") or "rectangle"),
                points=tuple(points),
            )
        )
    return tuple(shapes)


def _runtime_shape_xywh(shape: RuntimePreviewShape) -> tuple[float, float, float, float] | None:
    if not shape.points:
        return None
    xs = [float(point[0]) for point in shape.points]
    ys = [float(point[1]) for point in shape.points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        return None
    return (x_min, y_min, width, height)


def _sorted_runtime_shape_labels(shapes: tuple[RuntimePreviewShape, ...]) -> list[str]:
    roi_labels = sorted(
        [shape.label for shape in shapes if str(shape.label).startswith("roi")],
        key=_runtime_shape_label_sort_key,
    )
    other_labels = [
        shape.label
        for shape in shapes
        if shape.label not in roi_labels and shape.label not in {"anchor", "roi", "anchor_mask"}
    ]
    return roi_labels + other_labels


def _runtime_shape_label_sort_key(name: str) -> tuple[int, int | str]:
    suffix = str(name or "")[3:]
    if suffix.isdigit():
        return (0, int(suffix))
    if name == "roi":
        return (0, 0)
    return (1, name)
