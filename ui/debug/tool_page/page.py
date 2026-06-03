"""
Tool page UI for template setup, ROI review, training, and testing.

Main workflow groups:
  1. ROI editing: _load_canvas_image / _save_current_rect / _on_select_ok
  2. Shape localization: _autogen_roi_for_images / _open_shape_template_page
  3. Testing: _predict_image / _run_test / _populate_results_table / _append_test_log
  4. Validation: _suggest_margin_from_rows / _run_margin_validation / _run_traditional_baseline_debug

Signals consumed by MainWindow:
  productChangeRequested(str): ask MainWindow to switch products through apply_product_switch()
  sessionClearRequested(): ask MainWindow to clear state through reset_for_clear()
  sessionLoaded(): notify MainWindow that session state was loaded
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import algorithms.proxy as qr_core
from algorithms.registry import learning_backbone_storage_code

from infrastructure.camera_settings_store import (
    CameraSettingsStore,
    hik_settings_kwargs_from_mapping,
)
from application import (
    AlgorithmController,
    InspectionExecutionRequest,
    InspectionExecutor,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_EMBEDDING_ALGORITHMS,
    SUPPORTED_SCORE_MODES,
    ProductSession,
    SessionData,
    TrainResult,
)
from domain import (
    InspectionItem,
    clearable_roi_labels,
    inspection_item_specs_from_shape_recipe,
    load_inspection_items,
    save_inspection_items,
    sync_items_with_labels,
    output_labels_from_shape_recipe,
)
from shape.core import locator as shape_locator
from shape.core.recipe import ShapeRecipe
from ui.debug import (
    OverlayShape,
    RoiCanvas,
)
from ui.i18n import language_code, tr
from ui.roi_overlay_colors import is_roi_label, overlay_style_for_label
from ui.runtime import RuntimeImageView
from common.path_utils import product_relative_path, resolve_product_path
from common.safe_io import atomic_write_json, backup_path_for, load_json_with_backup


try:
    from devices import IoController
except Exception:
    IoController = None  # type: ignore[assignment]

try:
    from services import (
        FrameGrabService,
        HikCameraManager,
        HikCameraSettings,
        HikFrame,
        frame_to_bgr_image,
        frame_to_rgb_image,
    )
except Exception:
    FrameGrabService = None  # type: ignore[assignment]
    HikCameraManager = None  # type: ignore[assignment]
    HikCameraSettings = None  # type: ignore[assignment]
    HikFrame = None  # type: ignore[assignment]
    frame_to_bgr_image = None  # type: ignore[assignment]
    frame_to_rgb_image = None  # type: ignore[assignment]


SUPPORTED_LOC_MODES = ["shape"]
SUPPORTED_SHAPES = ["rect", "polygon"]
ROI_OVERLAY_PALETTE = [
    QtGui.QColor(255, 215, 0),
    QtGui.QColor(255, 64, 128),
    QtGui.QColor(0, 0, 255),
    QtGui.QColor(0, 255, 128),
    QtGui.QColor(255, 128, 0),
    QtGui.QColor(128, 255, 0),
]

_CAMERA_ROLE_RE = re.compile(r"(?:^|[_-])(cam[12])(?=[_.-]|$)", re.IGNORECASE)


def _normalize_camera_role(camera_id: object) -> str:
    text = str(camera_id or "").strip().lower()
    if text in {"cam1", "cam2"}:
        return text
    return ""


def _camera_role_from_path(path: str) -> str:
    name = os.path.basename(str(path or "")).lower()
    match = _CAMERA_ROLE_RE.search(name)
    if not match:
        return ""
    return _normalize_camera_role(match.group(1))


def _selected_image_list_camera_role(tool_page) -> str:
    getter = getattr(tool_page, "current_camera_role", None)
    if callable(getter):
        role = _normalize_camera_role(getter())
        if role:
            return role
    role_getter = getattr(tool_page, "_selected_debug_camera_role", None)
    if callable(role_getter):
        role = _normalize_camera_role(role_getter())
        if role:
            return role
    return "cam1"


def _filter_paths_for_camera(tool_page, paths: List[str], camera_id: object) -> List[str]:
    role = _normalize_camera_role(camera_id)
    if not role:
        return list(paths)
    if not any(_camera_role_from_path(path) for path in paths):
        return list(paths)
    return [path for path in paths if _camera_role_from_path(path) == role]

class _SampleAnnotationCanvas(QtWidgets.QWidget):
    imagePressed = QtCore.Signal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._pixmap = QtGui.QPixmap()
        self._scaled_pixmap = QtGui.QPixmap()
        self._overlays: list[OverlayShape] = []
        self._scale = 1.0
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 12.0
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._image_offset = QtCore.QPointF(0.0, 0.0)
        self._pressed_button = 0
        self._press_pos = QtCore.QPointF()
        self._pan_anchor = QtCore.QPointF()
        self._is_panning = False

    def clear_image(self) -> None:
        self._pixmap = QtGui.QPixmap()
        self._scaled_pixmap = QtGui.QPixmap()
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._zoom = 1.0
        self.update()

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = QtGui.QPixmap(pixmap)
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._zoom = 1.0
        self._update_scaled_pixmap()
        self.update()

    def set_overlays(self, overlays: Optional[List[OverlayShape]]) -> None:
        self._overlays = list(overlays or [])
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._pixmap.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        old_scale = self._scale
        anchor_widget = event.position()
        anchor_image = self._widget_to_image(anchor_widget)
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._zoom = max(self._zoom_min, min(self._zoom_max, self._zoom * factor))
        self._update_scaled_pixmap()
        if anchor_image is not None and old_scale > 0.0:
            new_scale = self._scale
            new_offset_x = float(anchor_widget.x()) - (anchor_image[0] * new_scale)
            new_offset_y = float(anchor_widget.y()) - (anchor_image[1] * new_scale)
            self._pan_offset = QtCore.QPointF(
                new_offset_x - self._base_image_offset().x(),
                new_offset_y - self._base_image_offset().y(),
            )
            self._pan_offset = self._clamp_pan_offset(self._pan_offset)
            self._image_offset = QtCore.QPointF(
                self._base_image_offset().x() + self._pan_offset.x(),
                self._base_image_offset().y() + self._pan_offset.y(),
            )
        self.update()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull():
            return
        self._pressed_button = int(getattr(event.button(), "value", event.button()))
        self._press_pos = event.position()
        self._pan_anchor = QtCore.QPointF(self._pan_offset)
        self._is_panning = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull() or self._pressed_button == 0:
            return
        if not self._can_pan():
            return
        if not self._is_panning:
            if (event.position() - self._press_pos).manhattanLength() < 6:
                return
            self._is_panning = True
        delta = event.position() - self._press_pos
        self._pan_offset = self._clamp_pan_offset(
            QtCore.QPointF(
                self._pan_anchor.x() + delta.x(),
                self._pan_anchor.y() + delta.y(),
            )
        )
        self._image_offset = QtCore.QPointF(
            self._base_image_offset().x() + self._pan_offset.x(),
            self._base_image_offset().y() + self._pan_offset.y(),
        )
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap.isNull():
            return
        released_button = int(getattr(event.button(), "value", event.button()))
        image_xy = self._widget_to_image(event.position())
        if released_button == self._pressed_button and not self._is_panning and image_xy is not None:
            self.imagePressed.emit(released_button, int(round(image_xy[0])), int(round(image_xy[1])))
        self._pressed_button = 0
        self._is_panning = False

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#111111"))
        if self._pixmap.isNull() or self._scaled_pixmap.isNull():
            painter.setPen(QtGui.QColor("#8a8a8a"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, tr("sample.no_selected_image"))
            return
        top_left = self._image_offset
        painter.drawPixmap(int(round(top_left.x())), int(round(top_left.y())), self._scaled_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for overlay in self._overlays:
            pen = QtGui.QPen(QtGui.QColor(overlay.color))
            pen.setWidthF(float(overlay.width))
            pen.setStyle(QtCore.Qt.DashLine if overlay.dash else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            if overlay.shape_type == "rect" and overlay.xywh is not None:
                x, y, w, h = overlay.xywh
                painter.drawRect(
                    QtCore.QRectF(
                        top_left.x() + float(x) * self._scale,
                        top_left.y() + float(y) * self._scale,
                        float(w) * self._scale,
                        float(h) * self._scale,
                    )
                )
            elif overlay.shape_type == "polygon" and overlay.points:
                polygon = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(
                            top_left.x() + float(x) * self._scale,
                            top_left.y() + float(y) * self._scale,
                        )
                        for x, y in overlay.points
                    ]
                )
                painter.drawPolygon(polygon)

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap.isNull():
            self._scaled_pixmap = QtGui.QPixmap()
            self._scale = 1.0
            self._image_offset = QtCore.QPointF(0.0, 0.0)
            return
        base_scale = min(
            max(1, self.width()) / max(1, self._pixmap.width()),
            max(1, self.height()) / max(1, self._pixmap.height()),
        )
        self._scale = max(0.01, float(base_scale) * float(self._zoom))
        target_size = QtCore.QSize(
            max(1, int(round(self._pixmap.width() * self._scale))),
            max(1, int(round(self._pixmap.height() * self._scale))),
        )
        self._scaled_pixmap = self._pixmap.scaled(
            target_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        base_offset = self._base_image_offset()
        self._pan_offset = self._clamp_pan_offset(self._pan_offset)
        self._image_offset = QtCore.QPointF(base_offset.x() + self._pan_offset.x(), base_offset.y() + self._pan_offset.y())

    def _base_image_offset(self) -> QtCore.QPointF:
        return QtCore.QPointF(
            float(max(0, (self.width() - self._scaled_pixmap.width()) / 2.0)),
            float(max(0, (self.height() - self._scaled_pixmap.height()) / 2.0)),
        )

    def _widget_to_image(self, widget_pos: QtCore.QPointF) -> tuple[float, float] | None:
        if self._pixmap.isNull() or self._scale <= 0.0:
            return None
        image_x = (float(widget_pos.x()) - self._image_offset.x()) / self._scale
        image_y = (float(widget_pos.y()) - self._image_offset.y()) / self._scale
        if image_x < 0.0 or image_y < 0.0 or image_x > self._pixmap.width() or image_y > self._pixmap.height():
            return None
        return (image_x, image_y)

    def _can_pan(self) -> bool:
        return (
            not self._scaled_pixmap.isNull()
            and (self._scaled_pixmap.width() > self.width() or self._scaled_pixmap.height() > self.height())
        )

    def _clamp_pan_offset(self, pan_offset: QtCore.QPointF) -> QtCore.QPointF:
        if self._scaled_pixmap.isNull():
            return QtCore.QPointF(0.0, 0.0)
        max_abs_x = max(0.0, (self._scaled_pixmap.width() - self.width()) / 2.0)
        max_abs_y = max(0.0, (self._scaled_pixmap.height() - self.height()) / 2.0)
        return QtCore.QPointF(
            max(-max_abs_x, min(max_abs_x, float(pan_offset.x()))),
            max(-max_abs_y, min(max_abs_y, float(pan_offset.y()))),
        )


class _SampleAnnotationAutoRoiDialog(QtWidgets.QDialog):
    def __init__(self, preview_dialog: "_SampleAnnotationPreviewDialog") -> None:
        super().__init__(preview_dialog)
        self._preview_dialog = preview_dialog
        self._tool_page = preview_dialog._tool_page
        self.setWindowTitle(tr("sample.auto_roi_tool"))
        self.setModal(False)
        self.resize(760, 180)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.lbl_scope = QtWidgets.QLabel("")
        self.lbl_scope.setStyleSheet("color:#d0d0d0;font-size:12px;")
        root.addWidget(self.lbl_scope)

        self.lbl_ref = QtWidgets.QLabel("")
        self.lbl_ref.setStyleSheet("color:#d0d0d0;font-size:12px;")
        root.addWidget(self.lbl_ref)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color:#86efac;font-size:12px;")
        root.addWidget(self.lbl_status)

        self.chk_only_missing = QtWidgets.QCheckBox(tr("debug.only_missing_roi"))
        self.chk_only_missing.setChecked(True)
        root.addWidget(self.chk_only_missing, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.btn_autogen_current = QtWidgets.QPushButton(tr("debug.batch_roi_current"))
        self.btn_autogen_current.clicked.connect(self._run_autogen_current_list)
        row.addWidget(self.btn_autogen_current)
        self.btn_autogen_current_image = QtWidgets.QPushButton(tr("debug.batch_roi_current_image_missing"))
        self.btn_autogen_current_image.clicked.connect(self._run_autogen_current_image)
        row.addWidget(self.btn_autogen_current_image)
        self.btn_clear_current = QtWidgets.QPushButton(tr("debug.clear_roi_current"))
        self.btn_clear_current.clicked.connect(self._run_clear_current_list)
        row.addWidget(self.btn_clear_current)
        root.addLayout(row)

        self._tool_page.roiGeometryChanged.connect(self._refresh_scope)
        self._tool_page.inspectionItemsChanged.connect(self._refresh_scope)
        self._autogen_thread: QtCore.QThread | None = None
        self._autogen_worker: _SampleAutoRoiWorker | None = None
        self._autogen_running = False
        self._refresh_scope()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_scope()

    def _camera_role(self) -> str:
        return str(self._preview_dialog.cmb_camera.currentData() or "cam1")

    def _sample_kind(self) -> str:
        return str(self._preview_dialog.cmb_sample_kind.currentData() or "train")

    def _scope_paths(self) -> List[str]:
        return self._tool_page._sample_paths_for_kind(self._sample_kind(), self._camera_role())

    def _current_path(self) -> str:
        path, _role = self._preview_dialog._current_path_and_role()
        return str(path or "").strip()

    def _refresh_scope(self) -> None:
        camera_role = self._camera_role()
        sample_kind = self._sample_kind()
        paths = self._scope_paths()
        current_path = self._current_path()
        sample_text = tr("debug.train_samples") if sample_kind == "train" else tr("debug.test_samples")
        self.lbl_scope.setText(
            tr("sample.current_scope", role=camera_role, sample=sample_text, count=len(paths))
            + (tr("sample.current_image_suffix", image=os.path.basename(current_path)) if current_path else "")
        )
        recipe = self._tool_page.shape_recipe_for_role(camera_role, force_reload=False)
        ref_image = ""
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = str(recipe.reference_image)
        self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(ref_image) if ref_image else '-'}")
        self.lbl_ref.setToolTip(ref_image)
        has_scope = bool(paths)
        if self._autogen_running:
            self.btn_autogen_current.setEnabled(False)
            self.btn_clear_current.setEnabled(False)
            self.btn_autogen_current_image.setEnabled(False)
            self.chk_only_missing.setEnabled(False)
        else:
            self.btn_autogen_current.setEnabled(has_scope)
            self.btn_clear_current.setEnabled(has_scope)
            self.btn_autogen_current_image.setEnabled(bool(current_path))
            self.chk_only_missing.setEnabled(True)

    def _sync_tool_page_role(self) -> None:
        self._tool_page._set_current_camera_role(self._camera_role(), sync_debug_role=True)

    def _run_autogen_current_list(self) -> None:
        paths = self._scope_paths()
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("sample.no_processable_images"))
            return
        self._sync_tool_page_role()
        self._start_autogen_job(paths, only_missing=self.chk_only_missing.isChecked())

    def _run_autogen_current_image(self) -> None:
        path = self._current_path()
        if not path:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("sample.no_selected_image"))
            return
        self._sync_tool_page_role()
        self._start_autogen_job([path], only_missing=True)

    def _set_autogen_running(self, running: bool, message: str = "") -> None:
        self._autogen_running = bool(running)
        self.lbl_status.setText(message)
        self._refresh_scope()

    def _prepare_autogen_payload(
        self,
        paths: List[str],
        *,
        only_missing: bool,
    ) -> dict | None:
        tool_page = self._tool_page
        ref_image = tool_page.ref_image
        method = tool_page.loc_method
        role = self._camera_role()
        labels: List[str]
        pre_resolved = False

        if method == "shape":
            try:
                recipe = tool_page.shape_recipe_for_role(role, force_reload=True)
                if role == tool_page.current_camera_role():
                    tool_page.shape_recipe = recipe
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Info", f"Failed to load template recipe: {exc}")
                return None
            if recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
                if tool_page.ref_image != ref_image:
                    if role == tool_page.current_camera_role():
                        tool_page._set_reference(ref_image)
                    else:
                        tool_page.ref_image = ref_image
                        if getattr(tool_page, "lbl_ref", None) is not None:
                            tool_page.lbl_ref.setText(
                                f"{tr('debug.reference_image')}: {os.path.basename(ref_image)}"
                            )
                            tool_page.lbl_ref.setToolTip(ref_image)
            if not os.path.exists(tool_page.shape_model_path_for_role(role)):
                QtWidgets.QMessageBox.warning(self, "Info", "Current product has no template model. Create a template first.")
                return None
            labels = tool_page._shape_output_labels(role)
            recipe_region_labels = {
                str(region.get("output_label") or region.get("reference_label") or "").strip()
                for region in (recipe.reference_regions or [])
                if isinstance(region, dict)
            }
            recipe_region_labels.discard("")
            if (not ref_image or not os.path.exists(ref_image)) and not recipe_region_labels:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.need_reference_or_saved"))
                return None
            if labels:
                missing_labels = [label for label in labels if label not in recipe_region_labels]
                if missing_labels:
                    ref_json = qr_core.labelme_json_of_image(ref_image) if ref_image else ""
                    if not ref_json or not os.path.exists(ref_json):
                        QtWidgets.QMessageBox.warning(
                            self,
                            tr("common.info"),
                            tr("auto.missing_reference_json", labels=", ".join(missing_labels)),
                        )
                        return None
                    missing_labels = [
                        label for label in missing_labels
                        if qr_core.read_shape_from_labelme(ref_json, label) is None
                    ]
                    if missing_labels:
                        QtWidgets.QMessageBox.warning(
                            self,
                            tr("common.info"),
                            tr("auto.missing_reference_roi", labels=", ".join(missing_labels)),
                        )
                        return None
        else:
            labels = ["roi"]
            if not ref_image or not os.path.exists(ref_image):
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.set_reference_first"))
                return None
            ref_json = qr_core.labelme_json_of_image(ref_image)
            if not os.path.exists(ref_json):
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.reference_missing_json"))
                return None
            if qr_core.try_read_xywh_from_labelme(ref_json, "anchor") is None:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.reference_missing_anchor"))
                return None
            if qr_core.try_read_xywh_from_labelme(ref_json, "roi") is None:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.reference_missing_roi"))
                return None

        resolved_paths = list(paths)
        if not only_missing:
            resolved_paths = tool_page._resolve_autogen_targets(
                paths,
                only_missing=False,
                silent=False,
                camera_role=role,
            )
            pre_resolved = True
            if not resolved_paths:
                return None

        return {
            "paths": resolved_paths,
            "labels": labels,
            "method": method,
            "ref_image": ref_image,
            "product_dir": tool_page.session.product_dir,
            "camera_role": role,
            "only_missing": bool(only_missing),
            "pre_resolved": pre_resolved,
        }

    def _start_autogen_job(self, paths: List[str], *, only_missing: bool) -> None:
        if self._autogen_running:
            return
        if getattr(self._tool_page, "_sample_auto_roi_jobs", []):
            QtWidgets.QMessageBox.information(self, tr("common.info"), "自动 ROI 正在处理，请等待")
            return
        payload = self._prepare_autogen_payload(paths, only_missing=only_missing)
        if payload is None:
            return

        thread = QtCore.QThread(self._tool_page)
        worker = _SampleAutoRoiWorker(payload)
        worker.moveToThread(thread)
        self._autogen_thread = thread
        self._autogen_worker = worker

        jobs = getattr(self._tool_page, "_sample_auto_roi_jobs", None)
        if jobs is None:
            jobs = []
            setattr(self._tool_page, "_sample_auto_roi_jobs", jobs)
        jobs.append((thread, worker))

        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_autogen_progress)
        worker.finished.connect(self._on_autogen_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._forget_autogen_job(thread, worker))
        self._set_autogen_running(True, "处理中...")
        thread.start()

    def _forget_autogen_job(self, thread: QtCore.QThread, worker: _SampleAutoRoiWorker) -> None:
        jobs = getattr(self._tool_page, "_sample_auto_roi_jobs", [])
        try:
            jobs.remove((thread, worker))
        except ValueError:
            pass
        if self._autogen_thread is thread:
            self._autogen_thread = None
            self._autogen_worker = None

    def _on_autogen_progress(self, message: str) -> None:
        if self._autogen_running:
            self.lbl_status.setText(f"处理中... {message}")

    def _on_autogen_finished(self, result: dict) -> None:
        tool_page = self._tool_page
        result = dict(result or {})
        timings = dict(result.get("timings", {}) or {})
        for path, elapsed_ms in timings.items():
            try:
                value = float(elapsed_ms)
            except Exception:
                continue
            tool_page._shape_match_ms_by_image[str(path)] = value
            tool_page._shape_autogen_ms_by_image[str(path)] = value

        ok = int(result.get("ok", 0) or 0)
        errs = [str(err) for err in result.get("errs", []) or [] if str(err)]
        changed_paths = {str(path) for path in result.get("ok_paths", []) or [] if str(path)}
        todo_paths = {str(path) for path in result.get("todo_paths", []) or [] if str(path)}

        if ok:
            previous = bool(getattr(tool_page, "_suppress_sample_preview_reload", False))
            tool_page._suppress_sample_preview_reload = True
            try:
                tool_page._reload_inspection_items()
                tool_page.roiGeometryChanged.emit()
            finally:
                tool_page._suppress_sample_preview_reload = previous

            cur = tool_page.canvas.image_path()
            if cur and cur in todo_paths:
                tool_page._load_canvas_image(cur)
                tool_page._set_status_for_current_image(cur)
            self._preview_dialog._refresh_after_auto_roi_change(changed_paths, self._camera_role())

        if result.get("fatal"):
            message = str(result.get("fatal") or "")
            if self.isVisible():
                QtWidgets.QMessageBox.warning(self, tr("common.error"), message)
            self.lbl_status.setText(message)
        elif result.get("no_work"):
            if self.isVisible():
                QtWidgets.QMessageBox.information(self, tr("common.info"), tr("auto.images_already_have_roi"))
            self.lbl_status.setText(tr("auto.images_already_have_roi"))
        else:
            msg = tr("auto.finished", ok=ok, failed=len(errs))
            if errs:
                msg += "\n\n" + tr("auto.failed_examples") + "\n" + "\n".join(errs[:10])
            if self.isVisible():
                QtWidgets.QMessageBox.information(self, tr("common.done"), msg)
            tool_page.lbl_status.setText(tr("auto.status_generated", ok=ok, failed=len(errs)))
            self.lbl_status.setText(tr("auto.status_generated", ok=ok, failed=len(errs)))

        self._set_autogen_running(False, self.lbl_status.text())

    def _run_clear_current_list(self) -> None:
        paths = self._scope_paths()
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("sample.no_processable_images"))
            return
        sample_text = tr("debug.train_samples") if self._sample_kind() == "train" else tr("debug.test_samples")
        reply = QtWidgets.QMessageBox.question(
            self,
            tr("auto.clear_title"),
            tr("sample.clear_roi_confirm", role=self._camera_role(), sample=sample_text),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._sync_tool_page_role()
        self._tool_page._clear_roi_for_images(paths, silent=False, camera_role=self._camera_role())


class _SampleAnnotationPreviewDialog(QtWidgets.QDialog):
    def __init__(self, tool_page: "ToolPage", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent or tool_page)
        self._tool_page = tool_page
        self.setWindowTitle(tr("sample.annotation_title"))
        self.resize(1100, 720)
        self.setModal(False)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QtWidgets.QLabel(tr("sample.product")))
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.addItem(tool_page.current_product_name())
        self.cmb_product.setEnabled(False)
        top_row.addWidget(self.cmb_product, 1)
        top_row.addWidget(QtWidgets.QLabel(tr("sample.camera")))
        self.cmb_camera = QtWidgets.QComboBox()
        self.sync_camera_roles(tool_page.configured_camera_roles())
        top_row.addWidget(self.cmb_camera)
        top_row.addWidget(QtWidgets.QLabel(tr("sample.sample")))
        self.cmb_sample_kind = QtWidgets.QComboBox()
        self.cmb_sample_kind.addItem(tr("debug.train_samples"), "train")
        self.cmb_sample_kind.addItem(tr("debug.test_samples"), "test")
        top_row.addWidget(self.cmb_sample_kind)
        root.addLayout(top_row)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(8)

        left_panel = QtWidgets.QFrame()
        left_panel.setStyleSheet("QFrame{background:#2f2f2f;border:1px solid #505050;}")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        left_layout.addWidget(QtWidgets.QLabel(tr("sample.image_list")))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setStyleSheet(
            "QListWidget{background:#333333;color:#e0e0e0;border:1px solid #404040;}"
            "QListWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
        )
        left_layout.addWidget(self.sample_list, 1)
        body.addWidget(left_panel, 1)

        center_panel = QtWidgets.QFrame()
        center_panel.setStyleSheet("QFrame{background:#1f1f1f;border:1px solid #505050;}")
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(6)
        center_layout.addWidget(QtWidgets.QLabel(tr("sample.current_image")))
        self.lbl_canvas_hint = QtWidgets.QLabel(tr("sample.canvas_hint"))
        self.lbl_canvas_hint.setStyleSheet("color:#a0a0a0;font-size:12px;")
        center_layout.addWidget(self.lbl_canvas_hint)
        self.preview_canvas = _SampleAnnotationCanvas()
        self.preview_canvas.setStyleSheet("QWidget{background:#111111;border:1px solid #303030;}")
        self.preview_canvas.imagePressed.connect(self._on_canvas_image_pressed)
        center_layout.addWidget(self.preview_canvas, 1)
        self.lbl_image_status = QtWidgets.QLabel(tr("sample.status_none"))
        self.lbl_image_status.setStyleSheet("color:#bcbcbc;font-size:12px;")
        center_layout.addWidget(self.lbl_image_status)
        body.addWidget(center_panel, 2)
        self._canvas_shapes: list[dict[str, object]] = []
        self._active_roi_label = ""
        self._suppress_tool_page_context_sync = False

        right_panel = QtWidgets.QFrame()
        right_panel.setStyleSheet("QFrame{background:#2f2f2f;border:1px solid #505050;}")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)
        right_layout.addWidget(QtWidgets.QLabel(tr("sample.current_roi_labels")))
        self.roi_table = QtWidgets.QTableWidget(0, 3)
        self.roi_table.setHorizontalHeaderLabels(["ROI", tr("sample.geometry"), tr("sample.label")])
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.roi_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.roi_table.setStyleSheet(
            "QTableWidget{background:#333333;color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;}"
            "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:4px;}"
        )
        right_layout.addWidget(self.roi_table, 1)
        self.lbl_dialog_hint = QtWidgets.QLabel(tr("sample.annotation_tip"))
        self.lbl_dialog_hint.setWordWrap(True)
        self.lbl_dialog_hint.setStyleSheet("color:#a0a0a0;font-size:12px;")
        right_layout.addWidget(self.lbl_dialog_hint)
        body.addWidget(right_panel, 1)
        root.addLayout(body, 1)

        footer = QtWidgets.QHBoxLayout()
        self.btn_mark_all_ok = QtWidgets.QPushButton(tr("sample.mark_all_ok"))
        self.btn_mark_all_ok.clicked.connect(self._mark_current_image_all_ok)
        footer.addWidget(self.btn_mark_all_ok)
        self.btn_clear_current = QtWidgets.QPushButton(tr("sample.clear_current_labels"))
        self.btn_clear_current.clicked.connect(self._clear_current_image_annotations)
        footer.addWidget(self.btn_clear_current)
        self.btn_open_autogen = QtWidgets.QPushButton(tr("sample.auto_roi"))
        self.btn_open_autogen.clicked.connect(self._open_autogen_dialog)
        footer.addWidget(self.btn_open_autogen)
        footer.addStretch(1)
        self.btn_prev = QtWidgets.QPushButton(tr("sample.prev"))
        self.btn_prev.clicked.connect(lambda: self._step_selection(-1))
        footer.addWidget(self.btn_prev)
        self.btn_next = QtWidgets.QPushButton(tr("sample.next"))
        self.btn_next.clicked.connect(lambda: self._step_selection(1))
        footer.addWidget(self.btn_next)
        btn_close = QtWidgets.QPushButton(tr("sample.close"))
        btn_close.clicked.connect(self.close)
        footer.addWidget(btn_close)
        root.addLayout(footer)

        camera_index = self.cmb_camera.findData(tool_page.current_camera_role())
        if camera_index >= 0:
            self.cmb_camera.setCurrentIndex(camera_index)
        sample_kind = tool_page._current_sample_tab_kind()
        sample_index = self.cmb_sample_kind.findData(sample_kind)
        if sample_index >= 0:
            self.cmb_sample_kind.setCurrentIndex(sample_index)

        tool_page.roiGeometryChanged.connect(self._on_tool_page_roi_geometry_changed)
        tool_page.inspectionItemsChanged.connect(self._on_tool_page_roi_geometry_changed)
        self.cmb_camera.currentIndexChanged.connect(lambda *_: self._reload_samples())
        self.cmb_sample_kind.currentIndexChanged.connect(lambda *_: self._reload_samples())
        self.sample_list.itemSelectionChanged.connect(self._on_sample_selected)
        self._reload_samples()

    def sync_camera_roles(self, roles: List[str]) -> None:
        normalized: List[str] = []
        for role in roles:
            role_text = _normalize_camera_role(role)
            if role_text and role_text not in normalized:
                normalized.append(role_text)
        if not normalized:
            normalized = ["cam1"]
        current_role = str(self.cmb_camera.currentData() or "cam1")
        blocker = QtCore.QSignalBlocker(self.cmb_camera)
        self.cmb_camera.clear()
        for role in normalized:
            self.cmb_camera.addItem(role, role)
        index = self.cmb_camera.findData(current_role if current_role in normalized else normalized[0])
        self.cmb_camera.setCurrentIndex(index if index >= 0 else 0)
        self.cmb_camera.setEnabled(len(normalized) > 1)
        del blocker
        if hasattr(self, "cmb_sample_kind"):
            self._reload_samples()

    def _reload_samples(self, preferred_path: Optional[str] = None) -> None:
        tool_page = self._tool_page
        camera_role = str(self.cmb_camera.currentData() or "cam1")
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        current_path = (
            str(preferred_path or "").strip()
            or self._current_dialog_selected_path()
            or tool_page._current_selected_path()
            or ""
        )
        paths = tool_page._sample_paths_for_kind(sample_kind, camera_role)
        blocker = QtCore.QSignalBlocker(self.sample_list)
        self.sample_list.clear()
        selected_row = -1
        for index, path in enumerate(paths):
            item = QtWidgets.QListWidgetItem(tool_page._sample_item_display_text(path, sample_kind, camera_role))
            item.setToolTip(path)
            item.setData(QtCore.Qt.UserRole, path)
            self.sample_list.addItem(item)
            if current_path and path == current_path:
                selected_row = index
        del blocker
        if self.sample_list.count() == 0:
            self.preview_canvas.clear_image()
            self.lbl_image_status.setText(tr("sample.list_empty"))
            self.roi_table.setRowCount(0)
            self._sync_navigation_buttons()
            self._sync_tool_page_context("")
            return
        if selected_row < 0:
            selected_row = 0
        self.sample_list.setCurrentRow(selected_row)
        self._on_sample_selected()

    def _on_sample_selected(self) -> None:
        tool_page = self._tool_page
        item = self.sample_list.currentItem()
        if item is None:
            return
        path = str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "")
        camera_role = str(self.cmb_camera.currentData() or "cam1")
        if not path:
            return
        self._active_roi_label = ""
        self._load_canvas_preview(path, camera_role)
        usage_text = tool_page._sample_usage_text(path)
        annotation_state = tool_page._sample_annotation_state_for_path(path, camera_role)
        self.lbl_image_status.setText(f"Status: {usage_text} / {annotation_state}")
        self._populate_roi_table(path, camera_role)
        self._sync_navigation_buttons()
        self._sync_tool_page_context(path)

    def _populate_roi_table(self, path: str, camera_role: str) -> None:
        tool_page = self._tool_page
        labels = tool_page._inspection_label_names_for_role(camera_role)
        self.roi_table.setRowCount(0)
        for row_index, label in enumerate(labels):
            self.roi_table.insertRow(row_index)
            self.roi_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(label))
            has_geometry = tool_page._path_has_roi_geometry(path, label)
            geometry_item = QtWidgets.QTableWidgetItem(tr("sample.generated") if has_geometry else tr("sample.missing_roi"))
            self.roi_table.setItem(row_index, 1, geometry_item)
            combo = QtWidgets.QComboBox()
            combo.addItem(tr("sample.unset"), "")
            combo.addItem("OK", "OK")
            combo.addItem("NG", "NG")
            current_status = tool_page._sample_roi_status_for_path(path, camera_role, label)
            combo_index = combo.findData(current_status)
            if combo_index < 0:
                combo_index = 0
            combo.setCurrentIndex(combo_index)
            combo.setEnabled(has_geometry)
            combo.currentIndexChanged.connect(
                lambda _index, image_path=path, role=camera_role, roi_label=label, widget=combo: self._on_roi_status_changed(
                    image_path,
                    role,
                    roi_label,
                    str(widget.currentData() or ""),
                )
            )
            self.roi_table.setCellWidget(row_index, 2, combo)

    def _on_roi_status_changed(self, path: str, camera_role: str, label: str, status: str) -> None:
        self._tool_page._set_sample_roi_status_for_path(path, camera_role, label, status)
        self._refresh_current_row_text(path, camera_role)
        self._tool_page._refresh_lists()
        self._tool_page._update_sample_panel_widgets()
        self._active_roi_label = label
        self._refresh_canvas_overlays(path, camera_role)
        self.lbl_image_status.setText(
            f"Status: {self._tool_page._sample_usage_text(path)} / "
            f"{self._tool_page._sample_annotation_state_for_path(path, camera_role)}"
        )

    def _refresh_current_row_text(self, path: str, camera_role: str) -> None:
        current_item = self.sample_list.currentItem()
        if current_item is None:
            return
        item_path = str(current_item.data(QtCore.Qt.UserRole) or current_item.toolTip() or "")
        if item_path != path:
            return
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        current_item.setText(self._tool_page._sample_item_display_text(path, sample_kind, camera_role))

    def _current_path_and_role(self) -> tuple[str, str]:
        item = self.sample_list.currentItem()
        if item is None:
            return "", str(self.cmb_camera.currentData() or "cam1")
        path = str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "")
        role = str(self.cmb_camera.currentData() or "cam1")
        return path, role

    def _current_dialog_selected_path(self) -> str:
        item = self.sample_list.currentItem()
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.UserRole) or item.toolTip() or "").strip()

    def _sync_tool_page_context(self, preferred_path: str = "") -> None:
        if getattr(self, "_suppress_tool_page_context_sync", False):
            return
        camera_role = str(self.cmb_camera.currentData() or "cam1")
        sample_kind = str(self.cmb_sample_kind.currentData() or "train")
        try:
            self._tool_page._set_current_camera_role(camera_role, sync_debug_role=True)
        except Exception:
            pass
        target_index = 0 if sample_kind == "train" else 1
        tabs = getattr(self._tool_page, "tabs", None)
        if tabs is not None and tabs.currentIndex() != target_index:
            tabs.setCurrentIndex(target_index)
        path = str(preferred_path or "").strip()
        if not path:
            return
        try:
            self._tool_page._select_path_in_current_tab(path)
        except Exception:
            pass

    def _open_autogen_dialog(self) -> None:
        dialog = getattr(self, "_sample_annotation_autogen_dialog", None)
        if dialog is None:
            dialog = _SampleAnnotationAutoRoiDialog(self)
            self._sample_annotation_autogen_dialog = dialog
            dialog.finished.connect(lambda *_: setattr(self, "_sample_annotation_autogen_dialog", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_tool_page_roi_geometry_changed(self) -> None:
        if bool(getattr(self._tool_page, "_suppress_sample_preview_reload", False)):
            return
        previous = bool(getattr(self, "_suppress_tool_page_context_sync", False))
        self._suppress_tool_page_context_sync = True
        try:
            self._reload_samples(preferred_path=self._current_dialog_selected_path())
        finally:
            self._suppress_tool_page_context_sync = previous

    def _refresh_after_auto_roi_change(self, changed_paths: set[str], camera_role: str) -> None:
        path = self._current_dialog_selected_path()
        if not path:
            return
        role = str(camera_role or self.cmb_camera.currentData() or "cam1")
        self._refresh_current_row_text(path, role)
        if path in set(changed_paths or set()):
            self._populate_roi_table(path, role)
            self._refresh_canvas_overlays(path, role)
            self.lbl_image_status.setText(
                f"Status: {self._tool_page._sample_usage_text(path)} / "
                f"{self._tool_page._sample_annotation_state_for_path(path, role)}"
            )

    def _mark_current_image_all_ok(self) -> None:
        path, camera_role = self._current_path_and_role()
        if not path:
            return
        self._tool_page._mark_sample_path_all_ok(path, camera_role)
        self._refresh_after_annotation_change(path, camera_role)

    def _clear_current_image_annotations(self) -> None:
        path, camera_role = self._current_path_and_role()
        if not path:
            return
        self._tool_page._clear_sample_path_annotations(path, camera_role)
        self._refresh_after_annotation_change(path, camera_role)

    def _step_selection(self, direction: int) -> None:
        count = self.sample_list.count()
        if count <= 0:
            return
        current_row = self.sample_list.currentRow()
        if current_row < 0:
            current_row = 0
        next_row = max(0, min(count - 1, current_row + int(direction)))
        if next_row == current_row:
            return
        self.sample_list.setCurrentRow(next_row)

    def _sync_navigation_buttons(self) -> None:
        count = self.sample_list.count()
        row = self.sample_list.currentRow()
        has_selection = count > 0 and row >= 0
        self.btn_mark_all_ok.setEnabled(has_selection)
        self.btn_clear_current.setEnabled(has_selection)
        self.btn_prev.setEnabled(has_selection and row > 0)
        self.btn_next.setEnabled(has_selection and row >= 0 and row < count - 1)

    def _refresh_after_annotation_change(self, path: str, camera_role: str) -> None:
        self._refresh_current_row_text(path, camera_role)
        self._tool_page._refresh_lists()
        self._tool_page._update_sample_panel_widgets()
        self._populate_roi_table(path, camera_role)
        self._refresh_canvas_overlays(path, camera_role)
        self.lbl_image_status.setText(
            tr(
                "debug.current_image_state",
                usage=self._tool_page._sample_usage_text(path),
                state=self._tool_page._sample_annotation_state_for_path(path, camera_role),
            )
        )

    def _load_canvas_preview(self, path: str, camera_role: str) -> None:
        pixmap = _pixmap_from_path(path)
        if pixmap.isNull():
            self.preview_canvas.clear_image()
            return
        self.preview_canvas.set_image(pixmap)
        self._refresh_canvas_overlays(path, camera_role)

    def _refresh_canvas_overlays(self, path: str, camera_role: str) -> None:
        jpath = qr_core.labelme_json_of_image(path)
        labels = self._tool_page._inspection_label_names_for_role(camera_role)
        overlays: list[OverlayShape] = []
        shape_entries: list[dict[str, object]] = []
        if not os.path.exists(jpath):
            self._canvas_shapes = []
            self.preview_canvas.set_overlays([])
            return
        for label in labels:
            poly_points = qr_core.try_read_polygon_points_from_labelme(jpath, label)
            xywh = qr_core.try_read_xywh_from_labelme(jpath, label)
            if poly_points and len(poly_points) >= 3:
                status = self._tool_page._sample_roi_status_for_path(path, camera_role, label).lower()
                color, width, dash = overlay_style_for_label(label, status=status)
                if label == self._active_roi_label:
                    width = max(float(width), 4.0)
                overlays.append(
                    OverlayShape(
                        shape_type="polygon",
                        points=[(float(x), float(y)) for x, y in poly_points],
                        color=QtGui.QColor(color),
                        width=float(width),
                        dash=bool(dash),
                    )
                )
                shape_entries.append(
                    {
                        "label": label,
                        "shape_type": "polygon",
                        "points": [(float(x), float(y)) for x, y in poly_points],
                    }
                )
                continue
            if xywh:
                status = self._tool_page._sample_roi_status_for_path(path, camera_role, label).lower()
                color, width, dash = overlay_style_for_label(label, status=status)
                if label == self._active_roi_label:
                    width = max(float(width), 4.0)
                overlays.append(
                    OverlayShape(
                        shape_type="rect",
                        xywh=tuple(int(v) for v in xywh),
                        color=QtGui.QColor(color),
                        width=float(width),
                        dash=bool(dash),
                    )
                )
                shape_entries.append(
                    {
                        "label": label,
                        "shape_type": "rect",
                        "xywh": tuple(int(v) for v in xywh),
                    }
                )
        self._canvas_shapes = shape_entries
        self.preview_canvas.set_overlays(overlays)

    def _on_canvas_image_pressed(self, button: int, image_x: int, image_y: int) -> None:
        button_value = int(getattr(QtCore.Qt.MouseButton.LeftButton, "value", QtCore.Qt.MouseButton.LeftButton))
        right_value = int(getattr(QtCore.Qt.MouseButton.RightButton, "value", QtCore.Qt.MouseButton.RightButton))
        if button not in {button_value, right_value}:
            return
        path, camera_role = self._current_path_and_role()
        if not path:
            return
        label = self._find_roi_label_at_point(float(image_x), float(image_y))
        if not label:
            return
        self._active_roi_label = label
        self._refresh_canvas_overlays(path, camera_role)
        self._focus_roi_row(label)
        self._show_roi_label_menu(path, camera_role, label)

    def _focus_roi_row(self, label: str) -> None:
        for row in range(self.roi_table.rowCount()):
            item = self.roi_table.item(row, 0)
            if item is None:
                continue
            if str(item.text()).strip() != str(label).strip():
                continue
            self.roi_table.setCurrentCell(row, 0)
            self.roi_table.scrollToItem(item)
            break

    def _show_roi_label_menu(self, path: str, camera_role: str, label: str) -> None:
        menu = QtWidgets.QMenu(self)
        action_ok = menu.addAction(f"{label} -> OK")
        action_ng = menu.addAction(f"{label} -> NG")
        action_clear = menu.addAction(tr("sample.clear_label", label=label))
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen is None:
            return
        if chosen == action_ok:
            status = "OK"
        elif chosen == action_ng:
            status = "NG"
        else:
            status = ""
        self._set_roi_status_from_canvas(path, camera_role, label, status)

    def _set_roi_status_from_canvas(self, path: str, camera_role: str, label: str, status: str) -> None:
        self._tool_page._set_sample_roi_status_for_path(path, camera_role, label, status)
        self._refresh_after_annotation_change(path, camera_role)

    def _find_roi_label_at_point(self, image_x: float, image_y: float) -> str:
        for entry in reversed(self._canvas_shapes):
            label = str(entry.get("label", "") or "").strip()
            if not label:
                continue
            shape_type = str(entry.get("shape_type", "") or "")
            if shape_type == "polygon":
                points = entry.get("points") or []
                polygon = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in points])
                if len(polygon) >= 3 and polygon.containsPoint(
                    QtCore.QPointF(float(image_x), float(image_y)),
                    QtCore.Qt.FillRule.OddEvenFill,
                ):
                    return label
                continue
            xywh = entry.get("xywh")
            if not xywh:
                continue
            x, y, w, h = [float(v) for v in xywh]
            if x <= float(image_x) <= x + w and y <= float(image_y) <= y + h:
                return label
        return ""

ALGORITHM_GROUPS = [
    (
        "learning",
        [
            ("High-Accuracy Learning Tool", "efficientnet_b0", True),
            ("Lightweight Learning Tool", "mobilenet_v3_small", True),
            ("Balanced Learning Tool", "mobilenet_v3_large", True),
        ],
    ),
    (
        "traditional",
        [
            ("Hue Tool", "meanhsv_h", True),
            ("Grayscale Tool", "meanintensity", True),
            ("Deviation Tool", "meanstd", True),
            ("Value Tool", "meanhsv_v", True),
            ("Saturation Tool", "meanhsv_s", True),
        ],
    ),
    (
        "measurement",
        [
            ("Find Line", "find_line", True),
            ("Find Circle", "find_circle", False),
        ],
    ),
]

ALGORITHM_DISPLAY_NAMES = {
    code: label
    for _group_name, items in ALGORITHM_GROUPS
    for label, code, enabled in items
    if enabled
}

ALGORITHM_GROUP_KEYS = {
    "learning": "debug.algorithm_group.learning",
    "traditional": "debug.algorithm_group.traditional",
    "measurement": "debug.algorithm_group.measurement",
}

ALGORITHM_DISPLAY_KEYS = {
    "efficientnet_b0": "debug.algorithm.efficientnet_b0",
    "mobilenet_v3_small": "debug.algorithm.mobilenet_v3_small",
    "mobilenet_v3_large": "debug.algorithm.mobilenet_v3_large",
    "meanhsv_h": "debug.algorithm.meanhsv_h",
    "meanintensity": "debug.algorithm.meanintensity",
    "meanstd": "debug.algorithm.meanstd",
    "meanhsv_v": "debug.algorithm.meanhsv_v",
    "meanhsv_s": "debug.algorithm.meanhsv_s",
    "find_circle": "debug.algorithm.find_circle",
    "find_line": "debug.algorithm.find_line",
    "find_line_subpix": "debug.algorithm.find_line_subpix",
    "line_distance": "debug.algorithm.line_distance",
    "line_distance_ref_normal": "debug.algorithm.line_distance_ref_normal",
}


def _algorithm_group_display_name(group_name: str) -> str:
    key = ALGORITHM_GROUP_KEYS.get(str(group_name or ""))
    return tr(key) if key else str(group_name or "")


def _algorithm_display_name(label: str, code: str) -> str:
    key = ALGORITHM_DISPLAY_KEYS.get(str(code or ""))
    return tr(key) if key else str(label or code or "")


def _pixmap_from_path(path: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(path)


def _qimage_from_hik_frame(frame: "HikFrame") -> QtGui.QImage:
    if frame_to_rgb_image is None:
        raise RuntimeError("camera frame conversion service is unavailable")
    image = frame_to_rgb_image(frame)
    if image.ndim == 2:
        height, width = image.shape
        return QtGui.QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()

    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = np.ascontiguousarray(image[:, :, :3])
        height, width, _ = rgb.shape
        return QtGui.QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        ).copy()

    raise ValueError("unsupported frame shape for debug preview")


class _DebugCameraPreviewThread(QtCore.QThread):
    frameReady = QtCore.Signal(QtGui.QImage)
    errorOccurred = QtCore.Signal(str)

    def __init__(self, frame_grab_service, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._frame_grab_service = frame_grab_service
        self._running = False
        self._last_error = ""

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                frame = self._frame_grab_service.capture_once("debug", timeout_ms=300)
            except Exception as exc:
                if self._running:
                    message = str(exc)
                    if message != self._last_error:
                        self.errorOccurred.emit(message)
                        self._last_error = message
                    self.msleep(120)
                continue

            self._last_error = ""
            try:
                image = _qimage_from_hik_frame(frame)
            except Exception as exc:
                if self._running:
                    self.errorOccurred.emit(tr("debug.preview_convert_failed", error=exc))
                    self.msleep(120)
                continue

            self.frameReady.emit(image)
            self.msleep(30)


class _SampleAutoRoiWorker(QtCore.QObject):
    progressChanged = QtCore.Signal(str)
    finished = QtCore.Signal(dict)

    def __init__(self, payload: dict, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._payload = dict(payload or {})

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._run_autogen()
        except Exception as exc:
            result = {
                "ok": 0,
                "errs": [str(exc)],
                "ok_paths": [],
                "todo_paths": [],
                "timings": {},
                "fatal": str(exc),
            }
        self.finished.emit(result)

    def _run_autogen(self) -> dict:
        paths = [str(path) for path in self._payload.get("paths", []) if str(path)]
        labels = [str(label).strip() for label in self._payload.get("labels", []) if str(label).strip()] or ["roi"]
        method = str(self._payload.get("method", "") or "").strip()
        only_missing = bool(self._payload.get("only_missing", True))
        pre_resolved = bool(self._payload.get("pre_resolved", False))

        if pre_resolved:
            todo = list(paths)
        else:
            missing = self._missing_roi_files(paths, labels)
            todo = list(missing) if only_missing else list(paths)

        if not todo:
            return {
                "ok": 0,
                "errs": [],
                "ok_paths": [],
                "todo_paths": [],
                "timings": {},
                "no_work": True,
            }

        ok = 0
        ok_paths: list[str] = []
        timings: dict[str, float] = {}
        errs: list[str] = []
        total = len(todo)
        for index, path in enumerate(todo, start=1):
            self.progressChanged.emit(f"{index}/{total} {os.path.basename(path)}")
            try:
                if method == "shape":
                    run = shape_locator.autogen_roi_json_from_shape_timed(
                        tgt_img_path=path,
                        ref_img_path=str(self._payload.get("ref_image", "") or ""),
                        product_dir=str(self._payload.get("product_dir", "") or ""),
                        camera_role=str(self._payload.get("camera_role", "cam1") or "cam1"),
                    )
                    timings[path] = float(run.total_ms)
                else:
                    qr_core.autogen_roi_json_from_reference(
                        tgt_img_path=path,
                        ref_img_path=str(self._payload.get("ref_image", "") or ""),
                        method=method,
                        anchor_label="anchor",
                        roi_label="roi",
                    )
                ok += 1
                ok_paths.append(path)
            except Exception as exc:
                errs.append(f"{os.path.basename(path)}: {exc}")

        return {
            "ok": ok,
            "errs": errs,
            "ok_paths": ok_paths,
            "todo_paths": todo,
            "timings": timings,
            "no_work": False,
        }

    @staticmethod
    def _missing_roi_files(paths: list[str], labels: list[str]) -> list[str]:
        missing: list[str] = []
        for path in paths:
            json_path = qr_core.labelme_json_of_image(path)
            if not os.path.exists(json_path):
                missing.append(path)
                continue
            try:
                existing_labels = {
                    str(shape.get("label", "") or "").strip()
                    for shape in qr_core.list_shapes_from_labelme(json_path)
                    if isinstance(shape, dict)
                }
            except Exception:
                missing.append(path)
                continue
            if any(label not in existing_labels for label in labels):
                missing.append(path)
        return missing


class _TrainingJobWorker(QtCore.QObject):
    progressChanged = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(
        self,
        algo: AlgorithmController,
        payload: dict,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._algo = algo
        self._payload = dict(payload or {})

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._run_training()
        except Exception as exc:
            detail = traceback.format_exc()
            result = {
                "mode": str(self._payload.get("mode", "") or ""),
                "success_names": [],
                "failure_messages": [f"{exc}\n\n{detail}"],
                "display_rows": [],
                "last_status_message": "",
                "fatal": f"{exc}\n\n{detail}",
            }
        self.finished.emit(result)

    def _run_training(self) -> dict:
        tasks = [dict(task or {}) for task in self._payload.get("tasks", [])]
        mode = str(self._payload.get("mode", "") or "")
        product_dir = str(self._payload.get("product_dir", "") or "")
        score_mode = str(self._payload.get("score_mode", "proto") or "proto")
        margin = float(self._payload.get("margin", 0.02))
        topk = int(self._payload.get("topk", 3))
        selected_item_id = str(self._payload.get("selected_item_id", "") or "")

        success_names: list[str] = []
        failure_messages: list[str] = [
            str(message)
            for message in self._payload.get("failure_messages", [])
            if str(message)
        ]
        display_rows: list[dict[str, object]] = []
        last_status_message = ""
        last_dialog_message = ""
        total = len(tasks)

        for index, task in enumerate(tasks, start=1):
            display_name = str(task.get("display_name", "") or "tool").strip() or "tool"
            algorithm = str(task.get("algorithm", "") or "").strip()
            item_id = str(task.get("item_id", "") or "").strip()
            self.progressChanged.emit(f"training {index}/{total} {display_name}")

            def _progress(message: str, *, _index: int = index, _total: int = total, _name: str = display_name) -> None:
                self.progressChanged.emit(f"training {_index}/{_total} {_name}: {message}")

            try:
                self._algo.product_params.algorithm = algorithm
                self._algo.product_params.score_mode = score_mode
                self._algo.product_params.margin = margin
                self._algo.product_params.topk = topk
                result = self._algo.train(
                    list(task.get("ok_files", []) or []),
                    list(task.get("ng_files", []) or []),
                    algorithm=algorithm,
                    product_dir=product_dir,
                    label_names=list(task.get("label_names", []) or []),
                    model_key=task.get("model_key", ""),
                    progress_callback=_progress,
                    embedding_cache_dir=str(task.get("embedding_cache_dir", "") or ""),
                )
                success_names.append(display_name)
                last_status_message = str(result.status_message or "")
                last_dialog_message = str(result.dialog_message or "")
                if not result.is_embedding and result.result_rows:
                    if item_id == selected_item_id:
                        display_rows = list(result.result_rows)
                    elif not display_rows:
                        display_rows = list(result.result_rows)
            except Exception as exc:
                failure_messages.append(f"{display_name}: {exc}\n{traceback.format_exc()}")

        return {
            "mode": mode,
            "success_names": success_names,
            "failure_messages": failure_messages,
            "display_rows": display_rows,
            "last_status_message": last_status_message,
            "last_dialog_message": last_dialog_message,
            "fatal": "",
        }


# ---------------------------------------------------------------------------
# ToolPage
# ---------------------------------------------------------------------------

class ToolPage(QtWidgets.QWidget):
    """
    Main debug tool page widget.

    Typical MainWindow integration:

        self.tool_page = ToolPage(self.session, self.algo)
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(lambda: self._refresh_runtime_status_ui("session loaded"))
        self.main_pages.addTab(self.tool_page, "Debug")
        self.tool_page.load_session()
    """

    productChangeRequested = QtCore.Signal(str)   # new product name
    productDeleteRequested = QtCore.Signal(str)
    sessionClearRequested = QtCore.Signal()
    sessionLoaded = QtCore.Signal()
    inspectionItemsChanged = QtCore.Signal()
    roiGeometryChanged = QtCore.Signal()
    debugCameraConnectRequested = QtCore.Signal(str)
    debugCameraConnected = QtCore.Signal(str, str)
    cameraSettingsApplied = QtCore.Signal(str, object)


    def __init__(
        self,
        session: ProductSession,
        algo: AlgorithmController,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.algo = algo
        self._configured_camera_roles: List[str] = ["cam1", "cam2"]

        self.train_files: List[str] = []
        self.ok_files: List[str] = []
        self.ng_files: List[str] = []
        self.test_files: List[str] = []
        self.ref_image: Optional[str] = None
        self.loc_method: str = "shape"
        self.shape_recipe: Optional[ShapeRecipe] = None
        self._shape_recipes_by_role: Dict[str, Optional[ShapeRecipe]] = {}
        self._training_roi_ready_signatures: Dict[str, str] = {}
        self._training_roi_pending_actions: Dict[str, str] = {}
        self.inspection_items: List[InspectionItem] = []
        self._visible_inspection_item_indexes: List[int] = []
        self._inspection_items_table_loading = False
        self._shape_match_ms_by_image: Dict[str, float] = {}
        self._shape_autogen_ms_by_image: Dict[str, float] = {}
        self._current_result_rows: List[Dict[str, object]] = []
        self._roi_results_by_image: Dict[str, Dict[str, str]] = {}
        self._sample_roi_annotations_by_path: Dict[str, Dict[str, str]] = {}
        self._updating_runtime_params = False
        self._skip_empty_autogen_message = False
        self._tool_dialogs: Dict[str, QtWidgets.QDialog] = {}
        self._template_editor_dialog: Optional[QtWidgets.QDialog] = None
        self._sample_annotation_preview_dialog: Optional[QtWidgets.QDialog] = None
        self._debug_camera_manager = None
        self._debug_frame_grab_service = None
        self._debug_camera_infos: List[object] = []
        self._debug_preview_thread: Optional[_DebugCameraPreviewThread] = None
        self._debug_io_controller = None
        self._runtime_io_controller = None
        self._runtime_io_ready = False
        self._runtime_io_status_detail = "IO not initialized"
        self._debug_io_uses_runtime_controller = False
        self._debug_output_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._debug_di_cards: Dict[int, QtWidgets.QFrame] = {}
        self._debug_di_indicators: Dict[int, QtWidgets.QLabel] = {}
        self._debug_di_hints: Dict[int, QtWidgets.QLabel] = {}
        self._debug_do_cards: Dict[int, QtWidgets.QFrame] = {}
        self._debug_do_channel_buttons: Dict[int, QtWidgets.QPushButton] = {}
        self._debug_do_hints: Dict[int, QtWidgets.QLabel] = {}
        self._debug_io_timer = QtCore.QTimer(self)
        self._debug_io_timer.setInterval(500)
        self._debug_io_timer.timeout.connect(self._refresh_debug_io_snapshot)
        self._camera_settings_store = CameraSettingsStore(self.session.camera_settings_path)
        self._current_camera_role = "cam1"
        # Guard against recursive setValue/apply callbacks in camera controls.
        self._debug_camera_block_spin_apply = False
        self._main_right_panel: Optional[QtWidgets.QFrame] = None
        self._algorithm_picker_style_default = ""
        self._algorithm_picker_style_compact = ""
        self._defer_initial_session_load = True
        self._deferred_session_load_scheduled = False
        self._training_thread: Optional[QtCore.QThread] = None
        self._training_worker: Optional[_TrainingJobWorker] = None
        self._training_in_progress = False
        self._training_roi_confirmed_signatures: Dict[str, str] = {}

        self._build_ui()
        self._set_current_camera_role(self._current_camera_role)
        self._apply_configured_camera_roles_to_ui()
        QtCore.QTimer.singleShot(0, self._update_responsive_layout)
        self.destroyed.connect(lambda *_: self._cleanup_debug_hardware())

    # ------------------------------------------------------------------
    # Public API used by MainWindow
    # ------------------------------------------------------------------

    def retranslate_ui(self) -> None:
        for attr, text in (
            ("lbl_product_caption", tr("debug.product")),
            ("lbl_current_camera_caption", tr("debug.current_camera")),
            ("lbl_images_section", tr("debug.image_list")),
            ("lbl_algo_tool", tr("debug.tool")),
            ("lbl_algo_decision", tr("debug.decision")),
            ("lbl_algo_threshold", tr("debug.threshold")),
            ("lbl_action_section", tr("debug.actions")),
            ("lbl_roi_shape_caption", tr("debug.shape")),
            ("lbl_roi_label_caption", tr("debug.label")),
            ("lbl_loc_method", tr("debug.location_method")),
            ("lbl_cam_left_title", tr("debug.device_list")),
            ("lbl_debug_role", tr("debug.debug_role")),
            ("lbl_cam_info_title", tr("debug.device_info")),
            ("lbl_cam_right_title", tr("debug.param_settings")),
            ("lbl_io_ctrl_title", tr("debug.connection_control")),
            ("lbl_io_status_title", tr("debug.io_status")),
            ("lbl_io_panel_title", tr("debug.dido_panel")),
            ("lbl_di_title", tr("debug.di_monitor")),
            ("lbl_do_title", tr("debug.do_control")),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(text)

        for attr, text in (
            ("btn_new_product", tr("debug.new")),
            ("btn_delete_product", tr("debug.delete_product")),
            ("btn_import_train", tr("debug.add_external_images")),
            ("btn_train_to_test", tr("debug.move_to_test")),
            ("btn_sample_annotation", tr("debug.sample_annotation")),
            ("btn_del_ok", tr("debug.remove")),
            ("btn_test_to_train", tr("debug.move_to_train")),
            ("btn_add_test", tr("debug.add_external_images")),
            ("btn_del_test", tr("debug.remove")),
            ("btn_sample_annotation_test", tr("debug.clear_current_test_list")),
            ("btn_toggle_algo", tr("debug.algorithm_params")),
            ("btn_toggle_tools", tr("debug.inspection_tools")),
            ("btn_train", tr("debug.train_all_tools")),
            ("btn_train_current", tr("debug.calibrate_current_tool")),
            ("btn_test", tr("debug.test_current_image")),
            ("btn_export_test", tr("debug.export_report")),
            ("btn_clear_session", tr("debug.test_all_test_samples")),
            ("btn_save", tr("debug.save_annotation")),
            ("btn_clear", tr("debug.clear_annotation")),
            ("btn_set_ref", tr("debug.set_as_reference")),
            ("btn_pick_ref", tr("debug.pick_reference")),
            ("chk_only_missing", tr("debug.only_missing_roi")),
            ("btn_autogen", tr("debug.batch_roi_current")),
            ("btn_autogen_all", tr("debug.batch_roi_all")),
            ("btn_clear_roi_batch", tr("debug.clear_roi_current")),
            ("btn_debug_refresh_camera", tr("debug.scan_camera")),
            ("btn_debug_connect_camera", tr("debug.connect")),
            ("btn_debug_disconnect_camera", tr("debug.disconnect")),
            ("btn_debug_live_preview", tr("debug.live_preview")),
            ("btn_debug_grab_once", tr("debug.grab_to_test")),
            ("btn_debug_save_image", tr("debug.save_image")),
            ("chk_debug_digital_shift_enable", tr("debug.enable")),
            ("btn_debug_read_camera_settings", tr("debug.read_camera_params")),
            ("btn_debug_apply_camera_settings", tr("debug.apply_camera_params")),
            ("btn_debug_open_io", tr("debug.open_io_debug")),
            ("btn_debug_close_io", tr("debug.close_io_debug")),
            ("btn_debug_refresh_io", tr("debug.refresh_dido")),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(text)

        if hasattr(self, "cmb_algorithm"):
            current_algorithm = self.current_algorithm()
            blocker = QtCore.QSignalBlocker(self.cmb_algorithm)
            self._populate_algorithm_combo()
            index = self._find_algorithm_combo_index(current_algorithm)
            self.cmb_algorithm.setCurrentIndex(index if index >= 0 else -1)
            del blocker
        if hasattr(self, "btn_algorithm_picker"):
            self.btn_algorithm_picker.setMenu(self._build_algorithm_picker_menu())
            self._sync_algorithm_picker()
        if hasattr(self, "tabs"):
            self._update_sample_panel_widgets()
        if hasattr(self, "inspection_items_table"):
            self.inspection_items_table.setHorizontalHeaderLabels([
                tr("debug.tools_table.enabled"),
                tr("debug.tools_table.name"),
                tr("debug.tools_table.camera"),
                tr("debug.tools_table.algorithm"),
                tr("debug.tools_table.status"),
            ])
            self._refresh_inspection_items_table()
        if hasattr(self, "template_match_box"):
            self.template_match_box.setTitle(tr("debug.auto_roi"))
        if hasattr(self, "btn_train_cancel"):
            self.btn_train_cancel.setToolTip(tr("debug.cancel_train_confirm"))
        if hasattr(self, "btn_train_current_cancel"):
            self.btn_train_current_cancel.setToolTip(tr("debug.cancel_current_confirm"))
        if hasattr(self, "cmb_debug_light_source_mode"):
            current_data = self.cmb_debug_light_source_mode.currentData()
            blocker = QtCore.QSignalBlocker(self.cmb_debug_light_source_mode)
            self.cmb_debug_light_source_mode.clear()
            self.cmb_debug_light_source_mode.addItem(tr("debug.board_do_light"), "board_io")
            self.cmb_debug_light_source_mode.addItem(tr("debug.camera_line1_strobe"), "camera_line1_strobe")
            index = self.cmb_debug_light_source_mode.findData(current_data)
            self.cmb_debug_light_source_mode.setCurrentIndex(max(0, index))
            self.cmb_debug_light_source_mode.setToolTip(tr("debug.camera_line1_tip"))
            del blocker
        form = getattr(self, "cam_params_form", None)
        if form is not None:
            row_labels = {
                "spin_debug_exposure": tr("debug.exposure"),
                "spin_debug_gain": tr("debug.gain"),
                "chk_debug_digital_shift_enable": tr("debug.digital_shift_enable"),
                "spin_debug_digital_shift": tr("debug.digital_shift"),
                "cmb_debug_trigger_mode": tr("debug.trigger_mode"),
                "cmb_debug_light_source_mode": tr("debug.light_source"),
            }
            for field_attr, label_text in row_labels.items():
                field = getattr(self, field_attr, None)
                label = form.labelForField(field) if field is not None else None
                if label is not None:
                    label.setText(label_text)
        measurement_form = getattr(self, "measurement_form", None)
        if measurement_form is not None:
            row_labels = {
                "cmb_measurement_line_a_tool": tr("debug.measurement.line_a_tool"),
                "cmb_measurement_line_b_tool": tr("debug.measurement.line_b_tool"),
                "cmb_measurement_line_a_direction": tr("debug.measurement.line_a_direction"),
                "cmb_measurement_line_b_direction": tr("debug.measurement.line_b_direction"),
                "cmb_measurement_polarity": tr("debug.measurement.polarity"),
                "spin_measurement_edge_threshold": tr("debug.measurement.edge_threshold"),
                "spin_measurement_scan_step": tr("debug.measurement.scan_step"),
                "spin_measurement_min_points": tr("debug.measurement.min_points"),
                "cmb_measurement_unit": tr("debug.measurement.unit"),
                "spin_measurement_pixel_size": tr("debug.measurement.pixel_size"),
            }
            for field_attr, label_text in row_labels.items():
                field = getattr(self, field_attr, None)
                label = measurement_form.labelForField(field) if field is not None else None
                if label is not None:
                    label.setText(label_text)
        if hasattr(self, "chk_measurement_lower"):
            self.chk_measurement_lower.setText(tr("debug.measurement.use_lower"))
        if hasattr(self, "chk_measurement_upper"):
            self.chk_measurement_upper.setText(tr("debug.measurement.use_upper"))
        if hasattr(self, "btn_add_line_distance_tool"):
            self.btn_add_line_distance_tool.setText(tr("debug.measurement.add_line_distance_tool"))
        if hasattr(self, "btn_delete_line_distance_tool"):
            self.btn_delete_line_distance_tool.setText(tr("debug.measurement.delete_line_distance_tool"))
        if hasattr(self, "cmb_measurement_line_a_direction"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_line_a_direction)
            for index, key in enumerate((
                "debug.measurement.direction.left_right",
                "debug.measurement.direction.right_left",
                "debug.measurement.direction.top_down",
                "debug.measurement.direction.bottom_up",
            )):
                if index < self.cmb_measurement_line_a_direction.count():
                    self.cmb_measurement_line_a_direction.setItemText(index, tr(key))
            del blocker
        if hasattr(self, "cmb_measurement_line_b_direction"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_line_b_direction)
            for index, key in enumerate((
                "debug.measurement.direction.left_right",
                "debug.measurement.direction.right_left",
                "debug.measurement.direction.top_down",
                "debug.measurement.direction.bottom_up",
            )):
                if index < self.cmb_measurement_line_b_direction.count():
                    self.cmb_measurement_line_b_direction.setItemText(index, tr(key))
            del blocker
        if hasattr(self, "cmb_measurement_polarity"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_polarity)
            for index, key in enumerate((
                "debug.measurement.polarity.any",
                "debug.measurement.polarity.dark_to_bright",
                "debug.measurement.polarity.bright_to_dark",
            )):
                if index < self.cmb_measurement_polarity.count():
                    self.cmb_measurement_polarity.setItemText(index, tr(key))
            del blocker
        dialog = getattr(self, "_template_editor_dialog", None)
        if dialog is not None and hasattr(dialog, "retranslate_ui"):
            dialog.retranslate_ui()
        self._retranslate_tool_dialogs()
        if hasattr(self, "lbl_debug_di_snapshot") and getattr(self, "_debug_io_controller", None) is None:
            self.lbl_debug_di_snapshot.setText(tr("debug.di_disconnected"))
            self.lbl_debug_do_snapshot.setText(tr("debug.do_disconnected"))
            self.lbl_debug_io_mapping_summary.setText(tr("debug.mapping_not_loaded"))
        self._sync_footer()

    def _tool_dialog_title(self, key: str) -> str:
        title_keys = {
            "camera_debug": "action.camera_tool",
            "io_debug": "action.io_tool",
            "template_match": "action.auto_region",
        }
        return tr(title_keys.get(str(key), str(key)))

    def _retranslate_tool_dialogs(self) -> None:
        for key, dialog in getattr(self, "_tool_dialogs", {}).items():
            if dialog is not None:
                dialog.setWindowTitle(self._tool_dialog_title(key))

    def current_algorithm(self) -> str:
        value = self.cmb_algorithm.currentData() if hasattr(self, "cmb_algorithm") else None
        if value is None:
            return ""
        return str(value).strip()

    def configured_camera_roles(self) -> List[str]:
        roles: List[str] = []
        for role in list(getattr(self, "_configured_camera_roles", []) or []):
            normalized = _normalize_camera_role(role)
            if normalized and normalized not in roles:
                roles.append(normalized)
        if not roles:
            roles = ["cam1"]
        if "cam1" not in roles:
            roles.insert(0, "cam1")
        return roles

    def set_configured_camera_roles(self, roles: List[str]) -> None:
        normalized: List[str] = []
        for role in roles:
            role_text = _normalize_camera_role(role)
            if role_text and role_text not in normalized:
                normalized.append(role_text)
        if not normalized:
            normalized = ["cam1"]
        if "cam1" not in normalized:
            normalized.insert(0, "cam1")
        self._configured_camera_roles = normalized
        self._apply_configured_camera_roles_to_ui()

    def current_camera_role(self) -> str:
        combo = getattr(self, "cmb_current_camera_role", None)
        if combo is None:
            return _normalize_camera_role(getattr(self, "_current_camera_role", "cam1")) or "cam1"
        return _normalize_camera_role(combo.currentData() or combo.currentText() or self._current_camera_role) or "cam1"

    def _apply_camera_role_options_to_combo(self, combo: object) -> None:
        if combo is None:
            return
        allowed_roles = set(self.configured_camera_roles())
        model = combo.model() if hasattr(combo, "model") else None
        for role in ("cam1", "cam2"):
            index = combo.findData(role) if hasattr(combo, "findData") else -1
            if index < 0 or model is None or not hasattr(model, "item"):
                continue
            item = model.item(index)
            if item is not None:
                item.setEnabled(role in allowed_roles)
        if hasattr(combo, "setEnabled"):
            combo.setEnabled(len(allowed_roles) > 1)

    def _apply_configured_camera_roles_to_ui(self) -> None:
        allowed_roles = self.configured_camera_roles()
        self._apply_camera_role_options_to_combo(getattr(self, "cmb_current_camera_role", None))
        self._apply_camera_role_options_to_combo(getattr(self, "cmb_debug_camera_role", None))
        if self.current_camera_role() not in set(allowed_roles):
            self._set_current_camera_role(allowed_roles[0], sync_debug_role=True)
        else:
            refresh_role_status = getattr(self, "_refresh_debug_role_status", None)
            if callable(refresh_role_status):
                refresh_role_status()
        dialog = getattr(self, "_sample_annotation_preview_dialog", None)
        if dialog is not None and hasattr(dialog, "sync_camera_roles"):
            dialog.sync_camera_roles(allowed_roles)

    def shape_paths_for_role(self, camera_role: object = None):
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return self.session.shape_paths_for_role(role)

    def load_shape_recipe_for_role(
        self,
        camera_role: object = None,
        *,
        force_reload: bool = False,
    ) -> Optional[ShapeRecipe]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        if not force_reload and role in self._shape_recipes_by_role:
            recipe = self._shape_recipes_by_role.get(role)
        else:
            paths = self.shape_paths_for_role(role)
            if not (os.path.exists(paths.recipe_path) or os.path.exists(paths.legacy_recipe_path)):
                recipe = None
            else:
                try:
                    recipe = shape_locator.load_recipe_for_product(self.session.product_dir, role)
                except Exception:
                    recipe = None
            self._shape_recipes_by_role[role] = recipe
        if role == self.current_camera_role():
            self.shape_recipe = recipe
        return recipe

    def shape_recipe_for_role(
        self,
        camera_role: object = None,
        *,
        force_reload: bool = False,
    ) -> Optional[ShapeRecipe]:
        return self.load_shape_recipe_for_role(camera_role, force_reload=force_reload)

    def shape_model_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        paths = self.shape_paths_for_role(role)
        return shape_locator.resolved_model_path_for_product(self.session.product_dir, role)

    def shape_recipe_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return shape_locator.resolved_recipe_path_for_product(self.session.product_dir, role)

    def _apply_current_role_recipe_state(self) -> None:
        recipe = self.load_shape_recipe_for_role(self.current_camera_role())
        self.shape_recipe = recipe
        ref_image = ""
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = str(recipe.reference_image)
        self.ref_image = ref_image or None
        if self.ref_image:
            self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        else:
            self.lbl_ref.setText(tr("debug.reference_image_not_set"))
            self.lbl_ref.setToolTip("")

    def _train_sample_paths_for_role(self, camera_role: object = None) -> List[str]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        train_paths = list(getattr(self, "train_files", []) or [])
        if not train_paths:
            train_paths = list(getattr(self, "ok_files", []) or []) + list(getattr(self, "ng_files", []) or [])
        return _filter_paths_for_camera(self, list(dict.fromkeys(train_paths)), role)

    def _training_sample_groups_for_role(
        self,
        camera_role: object = None,
        *,
        roi_label: object = None,
    ) -> tuple[List[str], List[str], List[str]]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        candidate_paths = self._train_sample_paths_for_role(role)
        label = str(roi_label or "").strip()
        if not label:
            return [], [], candidate_paths
        training_ok_files: List[str] = []
        training_ng_files: List[str] = []
        for path in candidate_paths:
            if not self._path_has_roi_geometry(path, label):
                continue
            status = self._sample_roi_status_for_path(path, role, label)
            if status == "OK":
                training_ok_files.append(path)
            elif status == "NG":
                training_ng_files.append(path)
        return training_ok_files, training_ng_files, candidate_paths

    def _training_roi_ready_signature(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        candidate_paths = self._train_sample_paths_for_role(role)
        recipe_path = self.shape_recipe_path_for_role(role)
        model_path = self.shape_model_path_for_role(role)
        recipe_mtime = os.path.getmtime(recipe_path) if recipe_path and os.path.exists(recipe_path) else -1.0
        model_mtime = os.path.getmtime(model_path) if model_path and os.path.exists(model_path) else -1.0
        return "|".join(
            [
                role,
                str(recipe_mtime),
                str(model_mtime),
                str(self.ref_image or ""),
                *sorted(str(path) for path in candidate_paths),
            ]
        )

    def _refresh_current_image_after_roi_update(self, candidate_paths: List[str]) -> None:
        current_path = self.canvas.image_path()
        if not current_path or current_path not in set(candidate_paths):
            return
        self._load_canvas_image(current_path)
        self._set_status_for_current_image(current_path)

    def _clear_training_roi_review_state(self, camera_role: object = None) -> None:
        if camera_role is None:
            self._training_roi_ready_signatures = {}
            self._training_roi_pending_actions = {}
            self._training_roi_confirmed_signatures = {}
        else:
            role = _normalize_camera_role(camera_role) or "cam1"
            self._training_roi_ready_signatures.pop(role, None)
            self._training_roi_pending_actions.pop(role, None)
            self._training_roi_confirmed_signatures.pop(role, None)
        self._update_runtime_widgets()

    def _sync_training_action_buttons(self) -> None:
        train_button = getattr(self, "btn_train", None)
        train_current_button = getattr(self, "btn_train_current", None)
        cancel_train_button = getattr(self, "btn_train_cancel", None)
        cancel_current_button = getattr(self, "btn_train_current_cancel", None)
        if train_button is None or train_current_button is None:
            return

        current_role = self.current_camera_role()
        pending_action = getattr(self, "_training_roi_pending_actions", {}).get(current_role, "")
        default_train_text = tr("debug.train_all_tools")
        default_current_text = tr("debug.calibrate_current_tool")
        default_train_style = getattr(self, "_train_action_btn_style", "")
        default_current_style = getattr(self, "_train_current_btn_style", "")
        confirm_style = getattr(self, "_train_confirm_btn_style", default_train_style)

        if pending_action == "all":
            train_button.setText(tr("debug.confirm_train_all"))
            train_button.setStyleSheet(confirm_style)
            if cancel_train_button is not None:
                cancel_train_button.setVisible(True)
        else:
            train_button.setText(default_train_text)
            train_button.setStyleSheet(default_train_style)
            if cancel_train_button is not None:
                cancel_train_button.setVisible(False)

        if pending_action == "current":
            train_current_button.setText(tr("debug.confirm_current_tool"))
            train_current_button.setStyleSheet(confirm_style)
            if cancel_current_button is not None:
                cancel_current_button.setVisible(True)
            return

        train_current_button.setText(default_current_text)
        train_current_button.setStyleSheet(default_current_style)
        if cancel_current_button is not None:
            cancel_current_button.setVisible(False)

    def _cancel_training_pending_action(self, action_key: str | None = None) -> None:
        role = self.current_camera_role()
        pending_action = self._training_roi_pending_actions.get(role, "")
        if action_key and pending_action != action_key:
            return
        if not pending_action:
            return
        self._training_roi_pending_actions.pop(role, None)
        self._update_runtime_widgets()
        action_text = tr("debug.train_all_tools") if pending_action == "all" else tr("debug.calibrate_current_tool")
        self.lbl_status.setText(tr("debug.cancelled_confirm", action=action_text))

    def _ensure_training_roi_reviewed(self, camera_role: object, *, action_name: str, action_key: str) -> bool:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        if self.loc_method != "shape":
            return True
        candidate_paths = self._train_sample_paths_for_role(role)
        if not candidate_paths:
            return True

        signature = self._training_roi_ready_signature(role)
        if self._training_roi_confirmed_signatures.get(role) == signature:
            return True
        if (
            self._training_roi_ready_signatures.get(role) == signature
            and self._training_roi_pending_actions.get(role) == action_key
        ):
            self._training_roi_ready_signatures.pop(role, None)
            self._training_roi_pending_actions.pop(role, None)
            self._training_roi_confirmed_signatures[role] = signature
            self._update_runtime_widgets()
            return True

        recipe = self.shape_recipe_for_role(role, force_reload=True)
        if recipe is None:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.recipe_missing", role=role))
            return False
        ref_image = self.ref_image
        if recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
        if not ref_image or not os.path.exists(ref_image):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.reference_missing", role=role))
            return False

        ok_count = 0
        errors: List[str] = []
        for path in candidate_paths:
            try:
                run = shape_locator.autogen_roi_json_from_shape_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=role,
                )
                self._shape_match_ms_by_image[path] = float(run.total_ms)
                self._shape_autogen_ms_by_image[path] = float(run.total_ms)
                ok_count += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        self._refresh_current_image_after_roi_update(candidate_paths)

        if errors:
            self._clear_training_roi_review_state(role)
            QtWidgets.QMessageBox.warning(
                self,
                tr("debug.roi_generate_failed_title"),
                tr("debug.roi_generate_failed_message", errors="\n".join(errors[:20])),
            )
            return False

        self._training_roi_ready_signatures[role] = signature
        self._training_roi_pending_actions[role] = action_key
        self._update_runtime_widgets()
        self.lbl_status.setText(tr("debug.roi_updated_status", role=role, action=action_name))
        QtWidgets.QMessageBox.information(
            self,
            tr("debug.roi_updated_title"),
            tr("debug.roi_updated_message", role=role, count=ok_count, action=action_name),
        )
        return False

    def _set_current_camera_role(self, role: object, *, sync_debug_role: bool = True) -> None:
        normalized = _normalize_camera_role(role) or "cam1"
        previous = getattr(self, "_current_camera_role", "cam1")
        self._current_camera_role = normalized

        combo = getattr(self, "cmb_current_camera_role", None)
        if combo is not None:
            index = combo.findData(normalized)
            if index >= 0 and combo.currentIndex() != index:
                blocker = QtCore.QSignalBlocker(combo)
                combo.setCurrentIndex(index)
                del blocker

        if sync_debug_role:
            debug_combo = getattr(self, "cmb_debug_camera_role", None)
            if debug_combo is not None:
                index = debug_combo.findData(normalized)
                if index >= 0 and debug_combo.currentIndex() != index:
                    blocker = QtCore.QSignalBlocker(debug_combo)
                    debug_combo.setCurrentIndex(index)
                    del blocker

        if previous != normalized:
            self._clear_image_view_for_role_switch()
        self._apply_current_role_recipe_state()
        self._refresh_lists()
        self._refresh_inspection_items_table()
        self._update_runtime_widgets()
        refresh_role_status = getattr(self, "_refresh_debug_role_status", None)
        if callable(refresh_role_status):
            refresh_role_status()

    def _on_current_camera_role_changed(self, value: str) -> None:
        self._set_current_camera_role(value, sync_debug_role=True)
        connected_serial = str(getattr(self._debug_camera_device(), "serial_number", "") or "").strip()
        self._apply_debug_role_binding_to_camera_combo()
        self._refresh_debug_camera_info()
        self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())
        selected_serial = self._selected_debug_camera_serial()
        if connected_serial and connected_serial != selected_serial:
            self._disconnect_debug_camera()
            self._set_debug_camera_status(tr("debug.status_switched_camera_reconnect", role=self.current_camera_role()))

    def current_algorithm_display_name(self) -> str:
        algorithm = self.current_algorithm()
        if not algorithm:
            return ""
        return _algorithm_display_name(ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm), algorithm)

    def _populate_algorithm_combo(self) -> None:
        self.cmb_algorithm.clear()
        model = self.cmb_algorithm.model()
        for group_name, items in ALGORITHM_GROUPS:
            self.cmb_algorithm.addItem(_algorithm_group_display_name(group_name), None)
            header_index = self.cmb_algorithm.count() - 1
            header_item = model.item(header_index) if hasattr(model, "item") else None
            if header_item is not None:
                header_item.setEnabled(False)
                header_font = QtGui.QFont(header_item.font())
                if header_font.pointSizeF() <= 0:
                    fallback_font = QtWidgets.QApplication.font(self.cmb_algorithm)
                    fallback_point_size = int(round(fallback_font.pointSizeF()))
                    if fallback_point_size > 0:
                        header_font.setPointSize(fallback_point_size)
                    elif fallback_font.pixelSize() > 0:
                        header_font.setPointSize(max(1, int(round(fallback_font.pixelSize() * 0.75))))
                    elif header_font.pixelSize() > 0:
                        header_font.setPointSize(max(1, int(round(header_font.pixelSize() * 0.75))))
                    else:
                        header_font.setPointSize(10)
                header_font.setBold(True)
                header_item.setFont(header_font)
                header_item.setForeground(QtGui.QColor("#9fd2ff"))

            for label, code, enabled in items:
                self.cmb_algorithm.addItem(_algorithm_display_name(label, code), code if enabled else None)
                item_index = self.cmb_algorithm.count() - 1
                item = model.item(item_index) if hasattr(model, "item") else None
                if item is not None and not enabled:
                    item.setEnabled(False)
                    item.setForeground(QtGui.QColor("#707070"))
                    item.setToolTip(tr("debug.tool_unimplemented"))

        self.cmb_algorithm.setCurrentIndex(-1)

    def _find_algorithm_combo_index(self, algorithm: str) -> int:
        for index in range(self.cmb_algorithm.count()):
            value = self.cmb_algorithm.itemData(index)
            if value == algorithm:
                return index
        return -1

    def _set_current_algorithm(self, algorithm: str) -> None:
        algorithm = str(algorithm or "").strip()
        if not algorithm:
            self.cmb_algorithm.setCurrentIndex(-1)
            self._sync_algorithm_picker()
            return
        index = self._find_algorithm_combo_index(algorithm)
        if index >= 0:
            self.cmb_algorithm.setCurrentIndex(index)
        else:
            self.cmb_algorithm.setCurrentIndex(-1)
        self._sync_algorithm_picker()
        return

    def _build_algorithm_picker_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#3a3a3a;color:#e0e0e0;border:1px solid #505050;}"
            "QMenu::item{padding:6px 24px;}"
            "QMenu::item:selected{background:#3794ff;}"
        )
        self._algorithm_actions: Dict[str, QtGui.QAction] = {}
        self._algorithm_action_group = QtGui.QActionGroup(self)
        self._algorithm_action_group.setExclusive(True)
        for group_name, items in ALGORITHM_GROUPS:
            submenu = menu.addMenu(_algorithm_group_display_name(group_name))
            for label, code, enabled in items:
                action = submenu.addAction(_algorithm_display_name(label, code))
                if not enabled:
                    action.setEnabled(False)
                    action.setToolTip(tr("debug.tool_unimplemented"))
                    continue
                action.setCheckable(True)
                self._algorithm_action_group.addAction(action)
                action.triggered.connect(
                    lambda checked=False, algorithm=code: self._set_current_algorithm(algorithm)
                )
                self._algorithm_actions[code] = action
        return menu

    def _sync_algorithm_picker(self) -> None:
        button = getattr(self, "btn_algorithm_picker", None)
        if button is not None:
            text = self.current_algorithm_display_name() or tr("debug.select_tool")
            button.setText(text)
            button.setToolTip(text)
        actions = getattr(self, "_algorithm_actions", {})
        current_algorithm = self.current_algorithm()
        for code, action in actions.items():
            action.setChecked(code == current_algorithm)

    def open_camera_debug_dialog(self) -> None:
        self._show_tool_dialog(
            "camera_debug",
            self.camera_debug_page,
            size=(1100, 700),
        )
        self._refresh_debug_role_status()
        self._refresh_debug_camera_list()

    def open_io_debug_dialog(self) -> None:
        self._show_tool_dialog(
            "io_debug",
            self.io_debug_page,
            size=(900, 480),
        )
        self._apply_runtime_io_debug_state()

    def open_template_editor_dialog(self) -> None:
        self._open_shape_template_page()

    def runtime_controller(self):
        parent = self.parent()
        while parent is not None:
            runtime_ctrl = getattr(parent, "runtime_ctrl", None)
            if runtime_ctrl is not None:
                return runtime_ctrl
            parent = parent.parent() if hasattr(parent, "parent") else None
        return None

    def set_runtime_io_state(self, ready: bool, detail: str, controller: object = None) -> None:
        self._runtime_io_ready = bool(ready)
        self._runtime_io_status_detail = str(detail or "")
        self._runtime_io_controller = controller if bool(ready) else None
        self._apply_runtime_io_debug_state()

    def _apply_runtime_io_debug_state(self) -> None:
        if self._runtime_io_ready and self._runtime_io_controller is not None:
            if (
                self._debug_io_controller is not None
                and self._debug_io_controller is not self._runtime_io_controller
                and not self._debug_io_uses_runtime_controller
            ):
                self._close_debug_io(silent=True)
            self._debug_io_controller = self._runtime_io_controller
            self._debug_io_uses_runtime_controller = True
            self._debug_io_timer.start()
            self.btn_debug_open_io.setEnabled(False)
            self.btn_debug_close_io.setEnabled(False)
            self.btn_debug_refresh_io.setEnabled(True)
            self.lbl_debug_io_mapping_summary.setToolTip(self._runtime_io_status_detail)
            self._refresh_debug_io_snapshot()
            return

        if self._debug_io_uses_runtime_controller:
            self._debug_io_timer.stop()
            self._debug_io_controller = None
            self._debug_io_uses_runtime_controller = False
            self.lbl_debug_di_snapshot.setText(tr("debug.di_disconnected"))
            self.lbl_debug_do_snapshot.setText(tr("debug.do_disconnected"))
            self.lbl_debug_io_mapping_summary.setText(tr("debug.mapping_not_loaded"))

        self.btn_debug_open_io.setEnabled(True)
        self.btn_debug_close_io.setEnabled(self._debug_io_controller is not None and not self._debug_io_uses_runtime_controller)
        self.btn_debug_refresh_io.setEnabled(self._debug_io_controller is not None)
        self.lbl_debug_io_mapping_summary.setToolTip(self._runtime_io_status_detail)

    def open_template_match_dialog(self) -> None:
        self._show_tool_dialog(
            "template_match",
            self.template_match_box,
            size=(880, 170),
        )

    def open_margin_validation_tool(self) -> None:
        self._run_margin_validation()

    def open_embedding_analysis_tool(self) -> None:
        self._open_embedding_analysis_dialog()

    def open_baseline_debug_tool(self) -> None:
        self._run_traditional_baseline_debug()

    def current_product_name(self) -> str:
        return self.session.current_product

    def refresh_product_selector(self) -> None:
        if not hasattr(self, "cmb_product"):
            return
        blocker = QtCore.QSignalBlocker(self.cmb_product)
        try:
            self.cmb_product.clear()
            self.cmb_product.addItems(self.session.product_names)
            self.cmb_product.setCurrentText(self.session.current_product)
        finally:
            del blocker

    def _sync_camera_settings_store_path(self) -> None:
        self._camera_settings_store.set_path(self.session.camera_settings_path)

    def connected_debug_camera_serial(self) -> str:
        device = self._debug_camera_device()
        if device is not None:
            return str(getattr(device, "serial_number", "") or "").strip()
        return ""

    def release_debug_camera_for_runtime(self) -> str:
        serial = self.connected_debug_camera_serial()
        if serial:
            self._disconnect_debug_camera()
        return serial

    def inspection_item_rows(
        self,
        *,
        status_kind: str = "pending",
        status_text: str = "",
    ) -> List[Dict[str, object]]:
        effective_status_text = status_text or tr("runtime.untested")
        return [
            {
                "item_id": item.item_id,
                "display_name": item.display_name,
                "camera_id": item.camera_id,
                "roi_label": item.roi_label,
                "algorithm_code": item.algorithm_code,
                "algorithm_type": item.algorithm_type,
                "params": dict(item.params or {}),
                "enabled": bool(item.enabled),
                "status_kind": status_kind if item.enabled else "disabled",
                "status_text": effective_status_text if item.enabled else tr("debug.status.disabled"),
            }
            for item in self.inspection_items
        ]

    def _roi_status_for_path(self, path: str, label_name: str) -> str:
        if not path:
            return ""
        label = str(label_name or "").strip()
        if not is_roi_label(label):
            return ""
        return str(self._roi_results_by_image.get(path, {}).get(label, "") or "").strip().lower()

    def _record_roi_result(self, path: str, label_name: str, status: object) -> None:
        if not path:
            return
        label = str(label_name or "").strip()
        if not is_roi_label(label):
            return
        status_text = str(status or "").strip().lower()
        if not status_text:
            return
        self._roi_results_by_image.setdefault(path, {})[label] = status_text

    def load_embedding_model(self, algorithm: str, model_key: Optional[str] = None) -> None:
        # Load the embedding model for the given algorithm and update lbl_status.
        _, msg = self.algo.load_model_for_algorithm(
            algorithm,
            self.session.product_dir,
            model_key=model_key or "",
        )
        self.lbl_status.setText(msg)

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: Optional[List[str]] = None,
        algorithm_override: Optional[str] = None,
        model_key_override: Optional[str] = None,
        params_override: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        # Used by MainWindow runtime callback.
        return self._predict_image(
            path,
            feat_net=feat_net,
            labels_override=labels_override,
            algorithm_override=algorithm_override,
            model_key_override=model_key_override,
            params_override=params_override,
        )

    def load_session(self) -> None:
        # Load algorithm params and session data, then refresh UI.
        # Emits sessionLoaded after the session is applied.
        if self._defer_initial_session_load:
            top_level = self.window()
            load_allowed = True
            if isinstance(top_level, QtWidgets.QWidget):
                load_allowed = bool(
                    getattr(top_level, "_allow_initial_tool_session_load", top_level.isVisible())
                )
            if not load_allowed:
                if not self._deferred_session_load_scheduled:
                    self._deferred_session_load_scheduled = True
                    QtCore.QTimer.singleShot(80, self._run_deferred_initial_session_load)
                return
            self._defer_initial_session_load = False
            self._deferred_session_load_scheduled = False

        self._sync_camera_settings_store_path()

        self.algo.load_params(self.session.product_params_path)
        self.algo.model = None
        self._roi_results_by_image = {}
        self._apply_runtime_params_to_ui()

        sd = self.session.load_session()
        self.train_files = list(dict.fromkeys(sd.train_files or (sd.ok_files + sd.ng_files)))
        self.ok_files = []
        self.ng_files = []
        self.test_files = sd.test_files
        self._load_sample_roi_annotations()
        self.loc_method = sd.loc_method
        self._shape_recipes_by_role = {}
        self._clear_training_roi_review_state()
        self.cmb_loc.setCurrentText(self.loc_method)
        self._apply_current_role_recipe_state()
        if not self.ref_image and sd.ref_image and os.path.exists(sd.ref_image):
            self.ref_image = sd.ref_image
            self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        self._refresh_lists()

        self._reload_inspection_items()
        self._sync_footer()

        self.sessionLoaded.emit()

    def _run_deferred_initial_session_load(self) -> None:
        self._deferred_session_load_scheduled = False
        if self._defer_initial_session_load:
            self.load_session()

    def apply_product_switch(self, name: str) -> None:
        # Switch product after runtime cameras are disconnected.
        # Update session paths, clear transient state, and reload the session.


        self.session.switch_product(name)
        self.session.save_products()
        self._sync_camera_settings_store_path()

        self.algo.model = None
        self.shape_recipe = None
        self._shape_recipes_by_role = {}
        self._clear_training_roi_review_state()
        self.ref_image = None
        self._shape_match_ms_by_image = {}
        self._shape_autogen_ms_by_image = {}
        self.train_files = []
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self._sample_roi_annotations_by_path = {}
        self._current_result_rows = []
        self._roi_results_by_image = {}
        self.inspection_items = []

        self.table.setRowCount(0)
        self.canvas.clear_image()
        self.lbl_ref.setText(tr("debug.reference_image_not_set"))
        self.lbl_ref.setToolTip("")
        self.lbl_status.setText(tr("debug.status_switched_product"))

        self.load_session()
        self._refresh_lists()

    def reset_for_clear(self) -> None:
        # Clear current debug session after runtime cameras are disconnected.
        # Reset image lists, cached models, and related UI state.


        self.train_files = []
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self._sample_roi_annotations_by_path = {}
        self.algo.model = None
        self.shape_recipe = None
        self._shape_recipes_by_role = {}
        self._clear_training_roi_review_state()
        self.ref_image = None
        self._shape_match_ms_by_image = {}
        self._shape_autogen_ms_by_image = {}
        self.lbl_ref.setText(tr("debug.reference_image_not_set"))
        self.lbl_status.setText(tr("debug.status_untrained"))
        self.table.setRowCount(0)
        self._current_result_rows = []
        self._roi_results_by_image = {}
        self._refresh_lists()
        self.session.delete_session_file()
        self._delete_sample_annotation_file()
        self._reload_inspection_items()
        self._sync_footer()

    # ------------------------------------------------------------------
    # UI 鏋勫缓
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        def _si(sp):
            return self.style().standardIcon(sp)
        SP = QtWidgets.QStyle.StandardPixmap
        _DARK_BG = "#2d2d2d"
        _PANEL_BG = "#363636"
        _HEADER_BG = "#3a3a3a"
        _TEXT_LIGHT = "#e0e0e0"
        _TEXT_DIM = "#888888"
        _compact_btn = (
            "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
            "padding:4px 8px;border-radius:3px;font-size:12px;}"
            "QPushButton:hover{background:#505050;}"
        )
        _input_style = (
            "QComboBox,QDoubleSpinBox,QSpinBox{"
            "background:#404040;color:#e0e0e0;border:1px solid #5a5a5a;padding:2px 4px;border-radius:3px;font-size:12px;}"
            "QComboBox:disabled,QDoubleSpinBox:disabled,QSpinBox:disabled{"
            "background:#353535;color:#8a8a8a;border:1px solid #4a4a4a;}"
            "QSpinBox::up-button:disabled,QSpinBox::down-button:disabled{"
            "background:#353535;border-left:1px solid #444444;}"
        )
        _section_style = (
            f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;"
            "border-bottom:1px solid #505050;padding:6px 10px;"
        )

        self.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QtWidgets.QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(f"background:{_HEADER_BG};border-bottom:1px solid #505050;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 16, 0)
        header_layout.setSpacing(4)

        self.lbl_product_caption = QtWidgets.QLabel(tr("debug.product"))
        header_layout.addWidget(self.lbl_product_caption)
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.setFixedWidth(180)
        self.cmb_product.addItems(self.session.product_names)
        self.cmb_product.setCurrentText(self.session.current_product)
        self.cmb_product.currentTextChanged.connect(self._on_product_changed)
        self.cmb_product.setStyleSheet(_input_style)
        header_layout.addWidget(self.cmb_product)

        self.btn_new_product = QtWidgets.QPushButton(_si(SP.SP_FileDialogNewFolder), tr("debug.new"))
        self.btn_new_product.setFixedWidth(60)
        self.btn_new_product.setStyleSheet(_compact_btn)
        self.btn_new_product.clicked.connect(self._new_product)
        header_layout.addWidget(self.btn_new_product)

        self.btn_delete_product = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), tr("debug.delete_product"))
        self.btn_delete_product.setFixedWidth(60)
        self.btn_delete_product.setStyleSheet(_compact_btn)
        self.btn_delete_product.clicked.connect(self._request_delete_product)
        header_layout.addWidget(self.btn_delete_product)

        header_layout.addSpacing(10)
        self.lbl_current_camera_caption = QtWidgets.QLabel(tr("debug.current_camera"))
        header_layout.addWidget(self.lbl_current_camera_caption)
        self.cmb_current_camera_role = QtWidgets.QComboBox()
        self.cmb_current_camera_role.setFixedWidth(84)
        self.cmb_current_camera_role.addItem("cam1", "cam1")
        self.cmb_current_camera_role.addItem("cam2", "cam2")
        self.cmb_current_camera_role.setStyleSheet(_input_style)
        self.cmb_current_camera_role.currentTextChanged.connect(self._on_current_camera_role_changed)
        header_layout.addWidget(self.cmb_current_camera_role)

        header_layout.addStretch(1)

        self.lbl_status = QtWidgets.QLabel(tr("debug.status_untrained"))
        self.lbl_status.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;")
        self.lbl_status.hide()
        header_layout.addWidget(self.lbl_status)

        root.addWidget(header)

        # Main canvas and side panels
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left side: canvas and result table
        canvas_frame = QtWidgets.QWidget()
        canvas_frame.setStyleSheet(f"background:{_DARK_BG};")
        canvas_vbox = QtWidgets.QVBoxLayout(canvas_frame)
        canvas_vbox.setContentsMargins(2, 2, 2, 2)
        canvas_vbox.setSpacing(2)

        self.canvas = RoiCanvas()
        self.canvas.setMinimumSize(480, 360)
        self.canvas.shapesChanged.connect(self._on_shapes_changed)
        canvas_vbox.addWidget(self.canvas, 3)

        self.table = QtWidgets.QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            ["\u6587\u4ef6", "GT", "Pred", "diff", "sim_ok", "sim_ng", "value", "threshold", "match_ms", "total_ms", "json"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_table_click)
        self.table.setStyleSheet(
            "QTableWidget{background:#333333;color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;}"
            "QTableWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
            "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:4px;}"
        )
        self.table.setMaximumHeight(180)
        canvas_vbox.addWidget(self.table, 1)

        body.addWidget(canvas_frame, 3)

        # 鍙充晶闈㈡澘
        right_panel = QtWidgets.QFrame()
        self._main_right_panel = right_panel
        right_panel.setStyleSheet(f"background:{_PANEL_BG};border-left:1px solid #505050;")
        right_vbox = QtWidgets.QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        # --- Image list ---
        self.lbl_images_section = QtWidgets.QLabel(tr("debug.image_list"))
        self.lbl_images_section.setFixedHeight(28)
        self.lbl_images_section.setStyleSheet(_section_style)
        right_vbox.addWidget(self.lbl_images_section)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane{border:none;}"
            f"QTabBar::tab{{background:#3a3a3a;color:{_TEXT_DIM};padding:4px 14px;border:none;font-size:12px;}}"
            f"QTabBar::tab:selected{{background:#4a4a4a;color:{_TEXT_LIGHT};border-bottom:2px solid #3794ff;}}"
        )
        self.tabs.currentChanged.connect(self._on_tab_changed)

        _lw_css = (
            f"QListWidget{{background:#333333;color:{_TEXT_LIGHT};border:none;font-size:12px;outline:0;}}"
            "QListWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
            "QListWidget::item:hover:!selected{background:#4a4a4a;}"
        )

        self.ok_list = QtWidgets.QListWidget()
        self.ok_list.setStyleSheet(_lw_css)
        self.ok_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ok_list.itemSelectionChanged.connect(self._on_select_ok)
        train_tab = QtWidgets.QWidget()
        train_layout = QtWidgets.QVBoxLayout(train_tab)
        train_layout.setContentsMargins(4, 4, 4, 4)
        train_layout.setSpacing(4)
        train_layout.addWidget(self.ok_list, 1)
        train_actions = QtWidgets.QGridLayout()
        train_actions.setHorizontalSpacing(4)
        train_actions.setVerticalSpacing(4)
        self.btn_import_train = QtWidgets.QPushButton(tr("debug.add_external_images"))
        self.btn_import_train.setStyleSheet(_compact_btn)
        self.btn_import_train.clicked.connect(lambda: self._add_images_to("TRAIN"))
        self.btn_train_to_test = QtWidgets.QPushButton(tr("debug.move_to_test"))
        self.btn_train_to_test.setStyleSheet(_compact_btn)
        self.btn_train_to_test.clicked.connect(lambda: self._move_selected_sample_to("TEST"))
        self.btn_sample_annotation = QtWidgets.QPushButton(tr("debug.sample_annotation"))
        self.btn_sample_annotation.setStyleSheet(_compact_btn)
        self.btn_sample_annotation.clicked.connect(self._open_sample_annotation_dialog)
        self.btn_del_ok = QtWidgets.QPushButton(tr("debug.remove"))
        self.btn_del_ok.setStyleSheet(_compact_btn)
        self.btn_del_ok.clicked.connect(lambda: self._remove_selected_from("TRAIN"))
        train_actions.addWidget(self.btn_import_train, 0, 0)
        train_actions.addWidget(self.btn_train_to_test, 0, 1)
        train_actions.addWidget(self.btn_sample_annotation, 1, 0)
        train_actions.addWidget(self.btn_del_ok, 1, 1)
        train_layout.addLayout(train_actions)
        self.tabs.addTab(train_tab, tr("debug.train_samples"))

        self.ng_list = QtWidgets.QListWidget(self)
        self.ng_list.setStyleSheet(_lw_css)
        self.ng_list.hide()

        self.test_list = QtWidgets.QListWidget()
        self.test_list.setStyleSheet(_lw_css)
        self.test_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.test_list.itemSelectionChanged.connect(self._on_select_test)
        test_tab = QtWidgets.QWidget()
        test_layout = QtWidgets.QVBoxLayout(test_tab)
        test_layout.setContentsMargins(4, 4, 4, 4)
        test_layout.setSpacing(4)
        test_layout.addWidget(self.test_list, 1)
        test_actions = QtWidgets.QGridLayout()
        test_actions.setHorizontalSpacing(4)
        test_actions.setVerticalSpacing(4)
        self.btn_test_to_train = QtWidgets.QPushButton(tr("debug.move_to_train"))
        self.btn_test_to_train.setStyleSheet(_compact_btn)
        self.btn_test_to_train.clicked.connect(lambda: self._move_selected_sample_to("TRAIN"))
        self.btn_add_test = QtWidgets.QPushButton(tr("debug.add_external_images"))
        self.btn_add_test.setStyleSheet(_compact_btn)
        self.btn_add_test.clicked.connect(lambda: self._add_images_to("TEST"))
        self.btn_del_test = QtWidgets.QPushButton(tr("debug.remove"))
        self.btn_del_test.setStyleSheet(_compact_btn)
        self.btn_del_test.clicked.connect(lambda: self._remove_selected_from("TEST"))
        self.btn_sample_annotation_test = QtWidgets.QPushButton(tr("debug.clear_current_test_list"))
        self.btn_sample_annotation_test.setStyleSheet(_compact_btn)
        self.btn_sample_annotation_test.clicked.connect(self._clear_current_test_list)
        test_actions.addWidget(self.btn_test_to_train, 0, 0)
        test_actions.addWidget(self.btn_add_test, 0, 1)
        test_actions.addWidget(self.btn_sample_annotation_test, 1, 0)
        test_actions.addWidget(self.btn_del_test, 1, 1)
        test_layout.addLayout(test_actions)
        self.tabs.addTab(test_tab, tr("debug.test_samples"))
        right_vbox.addWidget(self.tabs, 1)

        self.lbl_current_image_sample_state = QtWidgets.QLabel(f"  {tr('debug.current_image_state_none')}")
        self.lbl_current_image_sample_state.setWordWrap(True)
        self.lbl_current_image_sample_state.setStyleSheet(
            f"color:{_TEXT_DIM};font-size:11px;padding:4px 10px 8px 10px;border-bottom:1px solid #505050;"
        )
        right_vbox.addWidget(self.lbl_current_image_sample_state)

        # --- 绠楁硶鍙傛暟 ---
        self.btn_toggle_algo = QtWidgets.QToolButton()
        self.btn_toggle_algo.setText(tr("debug.algorithm_params"))
        self.btn_toggle_algo.setCheckable(True)
        self.btn_toggle_algo.setChecked(True)
        self.btn_toggle_algo.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_toggle_algo.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_toggle_algo.setStyleSheet(
            (
                f"QToolButton{{background:#404040;color:{_TEXT_LIGHT};font-size:12px;"
                f"font-weight:bold;border:none;border-bottom:1px solid #505050;padding:6px 10px;}}"
                "QToolButton:hover{background:#474747;}"
            )
        )
        self.btn_toggle_algo.toggled.connect(self._toggle_algorithm_section)
        right_vbox.addWidget(self.btn_toggle_algo)

        algo_frame = QtWidgets.QWidget()
        algo_form = QtWidgets.QFormLayout(algo_frame)
        algo_form.setContentsMargins(10, 6, 10, 6)
        algo_form.setSpacing(4)
        algo_form.setHorizontalSpacing(6)
        algo_form.setLabelAlignment(QtCore.Qt.AlignRight)
        algo_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.cmb_algorithm = QtWidgets.QComboBox()
        self._populate_algorithm_combo()
        self.cmb_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        self.cmb_algorithm.hide()
        self.cmb_backbone = self.cmb_algorithm
        self.btn_algorithm_picker = QtWidgets.QToolButton()
        self.btn_algorithm_picker.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_algorithm_picker.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_algorithm_picker.setMenu(self._build_algorithm_picker_menu())
        self._algorithm_picker_style_default = (
            "QToolButton{background:#2f2f2f;color:#e0e0e0;border:1px solid #555;"
            "padding:5px 28px 5px 8px;border-radius:3px;font-size:12px;}"
            "QToolButton:hover{background:#3a3a3a;}"
        )
        self._algorithm_picker_style_compact = (
            "QToolButton{background:#2f2f2f;color:#e0e0e0;border:1px solid #555;"
            "padding:4px 22px 4px 6px;border-radius:3px;font-size:11px;}"
            "QToolButton:hover{background:#3a3a3a;}"
        )
        self.btn_algorithm_picker.setStyleSheet(self._algorithm_picker_style_default)
        self.btn_algorithm_picker.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_algorithm_picker.setMinimumWidth(240)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(SUPPORTED_SCORE_MODES)
        self.cmb_mode.currentTextChanged.connect(self._on_runtime_params_changed)
        self.cmb_mode.setStyleSheet(_input_style)
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setDecimals(4)
        self.spin_margin.setSingleStep(0.005)
        self.spin_margin.setRange(-1.0, 1.0)
        self.spin_margin.setValue(0.02)
        self.spin_margin.valueChanged.connect(self._on_runtime_params_changed)
        self.spin_margin.setStyleSheet(_input_style)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setRange(1, 50)
        self.spin_topk.setValue(3)
        self.spin_topk.valueChanged.connect(self._on_runtime_params_changed)
        self.spin_topk.setStyleSheet(_input_style)

        _lbl_s = f"color:{_TEXT_DIM};font-size:12px;"
        _lbl_disabled_s = "color:#7a7a7a;font-size:12px;"
        self._algo_param_label_style = _lbl_s
        self._algo_param_label_disabled_style = _lbl_disabled_s
        self.lbl_algo_tool = QtWidgets.QLabel(tr("debug.tool")); self.lbl_algo_tool.setStyleSheet(_lbl_s)
        self.lbl_algo_decision = QtWidgets.QLabel(tr("debug.decision")); self.lbl_algo_decision.setStyleSheet(_lbl_s)
        self.lbl_algo_threshold = QtWidgets.QLabel(tr("debug.threshold")); self.lbl_algo_threshold.setStyleSheet(_lbl_s)
        self.lbl_topk = QtWidgets.QLabel("TopK")
        self.lbl_topk.setStyleSheet(self._algo_param_label_style)
        algo_form.addRow(self.lbl_algo_tool, self.btn_algorithm_picker)
        algo_form.addRow(self.lbl_algo_decision, self.cmb_mode)
        algo_form.addRow(self.lbl_algo_threshold, self.spin_margin)
        algo_form.addRow(self.lbl_topk, self.spin_topk)
        self.algorithm_params_frame = algo_frame
        right_vbox.addWidget(algo_frame)
        self._sync_algorithm_picker()

        tool_gap = QtWidgets.QWidget()
        tool_gap.setFixedHeight(14)
        right_vbox.addWidget(tool_gap)

        self.btn_toggle_tools = QtWidgets.QToolButton()
        self.btn_toggle_tools.setText(tr("debug.inspection_tools"))
        self.btn_toggle_tools.setCheckable(True)
        self.btn_toggle_tools.setChecked(True)
        self.btn_toggle_tools.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_toggle_tools.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_toggle_tools.setStyleSheet(
            (
                f"QToolButton{{background:#404040;color:{_TEXT_LIGHT};font-size:12px;"
                f"font-weight:bold;border:none;border-bottom:1px solid #505050;padding:6px 10px;}}"
                "QToolButton:hover{background:#474747;}"
            )
        )
        self.btn_toggle_tools.toggled.connect(self._toggle_tool_config_section)
        right_vbox.addWidget(self.btn_toggle_tools)

        tool_frame = QtWidgets.QWidget()
        tool_vbox = QtWidgets.QVBoxLayout(tool_frame)
        lbl_mg = QtWidgets.QLabel("Threshold") ; lbl_mg.setStyleSheet(_lbl_s)
        tool_vbox.setContentsMargins(0, 0, 0, 0)
        tool_vbox.setSpacing(4)

        self.lbl_tool_config_hint = QtWidgets.QLabel("")
        self.lbl_tool_config_hint.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        tool_vbox.addWidget(self.lbl_tool_config_hint)

        self.measurement_params_frame = QtWidgets.QWidget()
        measurement_form = QtWidgets.QFormLayout(self.measurement_params_frame)
        self.measurement_form = measurement_form
        measurement_form.setContentsMargins(8, 6, 8, 6)
        measurement_form.setSpacing(6)
        self._measurement_params_loading = False
        self.chk_measurement_lower = QtWidgets.QCheckBox(tr("debug.measurement.use_lower"))
        self.chk_measurement_upper = QtWidgets.QCheckBox(tr("debug.measurement.use_upper"))
        self.spin_measurement_lower = QtWidgets.QDoubleSpinBox()
        self.spin_measurement_upper = QtWidgets.QDoubleSpinBox()
        self.spin_measurement_pixel_size = QtWidgets.QDoubleSpinBox()
        self.cmb_measurement_line_a_tool = QtWidgets.QComboBox()
        self.cmb_measurement_line_b_tool = QtWidgets.QComboBox()
        for combo in (self.cmb_measurement_line_a_tool, self.cmb_measurement_line_b_tool):
            combo.setStyleSheet(_input_style)
            combo.currentIndexChanged.connect(self._on_measurement_params_changed)
        self.cmb_measurement_line_a_direction = QtWidgets.QComboBox()
        self.cmb_measurement_line_b_direction = QtWidgets.QComboBox()
        for combo in (self.cmb_measurement_line_a_direction, self.cmb_measurement_line_b_direction):
            combo.addItem(tr("debug.measurement.direction.left_right"), "left_right")
            combo.addItem(tr("debug.measurement.direction.right_left"), "right_left")
            combo.addItem(tr("debug.measurement.direction.top_down"), "top_down")
            combo.addItem(tr("debug.measurement.direction.bottom_up"), "bottom_up")
            combo.setStyleSheet(_input_style)
            combo.currentIndexChanged.connect(self._on_measurement_params_changed)
        self.cmb_measurement_polarity = QtWidgets.QComboBox()
        self.cmb_measurement_polarity.addItem(tr("debug.measurement.polarity.any"), "any")
        self.cmb_measurement_polarity.addItem(tr("debug.measurement.polarity.dark_to_bright"), "dark_to_bright")
        self.cmb_measurement_polarity.addItem(tr("debug.measurement.polarity.bright_to_dark"), "bright_to_dark")
        self.cmb_measurement_polarity.setStyleSheet(_input_style)
        self.cmb_measurement_polarity.currentIndexChanged.connect(self._on_measurement_params_changed)
        self.spin_measurement_edge_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_measurement_edge_threshold.setDecimals(3)
        self.spin_measurement_edge_threshold.setRange(0.0, 255.0)
        self.spin_measurement_edge_threshold.setSingleStep(1.0)
        self.spin_measurement_edge_threshold.setValue(10.0)
        self.spin_measurement_scan_step = QtWidgets.QSpinBox()
        self.spin_measurement_scan_step.setRange(1, 1000)
        self.spin_measurement_scan_step.setValue(2)
        self.spin_measurement_min_points = QtWidgets.QSpinBox()
        self.spin_measurement_min_points.setRange(2, 100000)
        self.spin_measurement_min_points.setValue(10)
        for spin in (
            self.spin_measurement_edge_threshold,
            self.spin_measurement_scan_step,
            self.spin_measurement_min_points,
        ):
            spin.setKeyboardTracking(False)
            spin.setStyleSheet(_input_style)
            spin.valueChanged.connect(self._on_measurement_params_changed)
        for spin in (self.spin_measurement_lower, self.spin_measurement_upper):
            spin.setDecimals(4)
            spin.setRange(-1000000.0, 1000000.0)
            spin.setSingleStep(0.01)
            spin.setKeyboardTracking(False)
            spin.setStyleSheet(_input_style)
            spin.valueChanged.connect(self._on_measurement_params_changed)
        for checkbox in (self.chk_measurement_lower, self.chk_measurement_upper):
            checkbox.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:12px;")
            checkbox.toggled.connect(self._on_measurement_params_changed)
        self.spin_measurement_pixel_size.setDecimals(9)
        self.spin_measurement_pixel_size.setRange(0.0, 1000000.0)
        self.spin_measurement_pixel_size.setSingleStep(0.000001)
        self.spin_measurement_pixel_size.setKeyboardTracking(False)
        self.spin_measurement_pixel_size.setStyleSheet(_input_style)
        self.spin_measurement_pixel_size.valueChanged.connect(self._on_measurement_params_changed)
        self.cmb_measurement_unit = QtWidgets.QComboBox()
        self.cmb_measurement_unit.addItems(["px", "mm"])
        self.cmb_measurement_unit.setStyleSheet(_input_style)
        self.cmb_measurement_unit.currentTextChanged.connect(self._on_measurement_params_changed)
        measurement_form.addRow(tr("debug.measurement.line_a_tool"), self.cmb_measurement_line_a_tool)
        measurement_form.addRow(tr("debug.measurement.line_b_tool"), self.cmb_measurement_line_b_tool)
        measurement_form.addRow(tr("debug.measurement.line_a_direction"), self.cmb_measurement_line_a_direction)
        measurement_form.addRow(tr("debug.measurement.line_b_direction"), self.cmb_measurement_line_b_direction)
        measurement_form.addRow(tr("debug.measurement.polarity"), self.cmb_measurement_polarity)
        measurement_form.addRow(tr("debug.measurement.edge_threshold"), self.spin_measurement_edge_threshold)
        measurement_form.addRow(tr("debug.measurement.scan_step"), self.spin_measurement_scan_step)
        measurement_form.addRow(tr("debug.measurement.min_points"), self.spin_measurement_min_points)
        measurement_form.addRow(self.chk_measurement_lower, self.spin_measurement_lower)
        measurement_form.addRow(self.chk_measurement_upper, self.spin_measurement_upper)
        measurement_form.addRow(tr("debug.measurement.unit"), self.cmb_measurement_unit)
        measurement_form.addRow(tr("debug.measurement.pixel_size"), self.spin_measurement_pixel_size)
        self.measurement_params_frame.hide()
        tool_vbox.addWidget(self.measurement_params_frame)

        measurement_action_row = QtWidgets.QHBoxLayout()
        measurement_action_row.setContentsMargins(0, 0, 0, 0)
        measurement_action_row.setSpacing(4)

        self.btn_add_line_distance_tool = QtWidgets.QPushButton(tr("debug.measurement.add_line_distance_tool"))
        self.btn_add_line_distance_tool.setStyleSheet(_compact_btn)
        self.btn_add_line_distance_tool.clicked.connect(self._add_line_distance_tool)
        measurement_action_row.addWidget(self.btn_add_line_distance_tool, 1)

        self.btn_delete_line_distance_tool = QtWidgets.QPushButton(tr("debug.measurement.delete_line_distance_tool"))
        self.btn_delete_line_distance_tool.setStyleSheet(_compact_btn)
        self.btn_delete_line_distance_tool.setEnabled(False)
        self.btn_delete_line_distance_tool.hide()
        self.btn_delete_line_distance_tool.clicked.connect(self._delete_selected_line_distance_tool)
        measurement_action_row.addWidget(self.btn_delete_line_distance_tool, 1)
        tool_vbox.addLayout(measurement_action_row)

        self.tool_config_frame = tool_frame
        self.inspection_items_table = QtWidgets.QTableWidget(0, 5)
        self.inspection_items_table.setHorizontalHeaderLabels([
            tr("debug.tools_table.enabled"),
            tr("debug.tools_table.name"),
            tr("debug.tools_table.camera"),
            tr("debug.tools_table.algorithm"),
            tr("debug.tools_table.status"),
        ])
        self.inspection_items_table.verticalHeader().setVisible(False)
        self.inspection_items_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.inspection_items_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        header = self.inspection_items_table.horizontalHeader()
        header.setStretchLastSection(False)
        self.btn_toggle_tools.setText(tr("debug.inspection_tools"))
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.inspection_items_table.setStyleSheet(
            "QTableWidget{background:#333333;color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;font-size:12px;}"
            "QTableWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
            "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:4px;}"
        )
        self.inspection_items_table.setMinimumHeight(170)
        self.inspection_items_table.setColumnWidth(0, 52)
        self.inspection_items_table.itemChanged.connect(self._on_inspection_items_table_item_changed)
        self.inspection_items_table.itemSelectionChanged.connect(self._on_inspection_items_selection_changed)
        tool_vbox.addWidget(self.inspection_items_table)
        self.tool_config_scroll = QtWidgets.QScrollArea()
        self.tool_config_scroll.setWidgetResizable(True)
        self.tool_config_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.tool_config_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tool_config_scroll.setStyleSheet(
            "QScrollArea{background:#2f2f2f;border:none;}"
            "QScrollArea > QWidget > QWidget{background:#2f2f2f;}"
            "QScrollBar:vertical{background:#2f2f2f;width:10px;margin:0;}"
            "QScrollBar::handle:vertical{background:#5a5a5a;min-height:28px;border-radius:5px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        self.tool_config_scroll.setWidget(tool_frame)
        self.tool_config_scroll.setMinimumHeight(220)
        right_vbox.addWidget(self.tool_config_scroll, 1)
        self.lbl_current_tool_sample_stats = QtWidgets.QLabel(f"  {tr('debug.current_tool_stats_select')}")
        self.lbl_current_tool_sample_stats.setWordWrap(True)
        self.lbl_current_tool_sample_stats.setStyleSheet(
            f"color:{_TEXT_DIM};font-size:11px;padding:6px 10px;border-top:1px solid #505050;border-bottom:1px solid #505050;"
        )
        right_vbox.addWidget(self.lbl_current_tool_sample_stats)
        self._update_learning_backbone_hint()

        # --- 鎿嶄綔鎸夐挳 ---
        self.lbl_action_section = QtWidgets.QLabel(tr("debug.actions"))
        sec_action = self.lbl_action_section
        sec_action.setFixedHeight(28)
        sec_action.setStyleSheet(_section_style)
        right_vbox.addWidget(sec_action)

        action_frame = QtWidgets.QWidget()
        action_vbox = QtWidgets.QVBoxLayout(action_frame)
        action_vbox.setContentsMargins(8, 6, 8, 6)
        action_vbox.setSpacing(4)
        self.lbl_training_validation = QtWidgets.QLabel(tr("debug.training_validation_select"))
        self.lbl_training_validation.setWordWrap(True)
        self.lbl_training_validation.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;padding:0 2px 4px 2px;")
        action_vbox.addWidget(self.lbl_training_validation)

        _action_btn = (
            "QPushButton{background:#2d5aa0;color:white;border:none;"
            "padding:6px 12px;border-radius:3px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#3a6abf;}"
            "QPushButton:pressed{background:#244a85;}"
        )
        _confirm_action_btn = (
            "QPushButton{background:#b36a19;color:white;border:none;"
            "padding:6px 12px;border-radius:3px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#ca7b22;}"
            "QPushButton:pressed{background:#985914;}"
        )
        _cancel_action_btn = (
            "QPushButton{background:#4a4a4a;color:#e0e0e0;border:1px solid #666666;"
            "padding:0px;border-radius:3px;font-size:13px;font-weight:bold;min-width:24px;max-width:24px;min-height:34px;max-height:34px;}"
            "QPushButton:hover{background:#5a5a5a;color:white;}"
            "QPushButton:pressed{background:#3d3d3d;}"
        )

        train_row = QtWidgets.QHBoxLayout()
        train_row.setContentsMargins(0, 0, 0, 0)
        train_row.setSpacing(4)

        self.btn_train = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), tr("debug.train_all_tools"))
        self._train_action_btn_style = _action_btn
        self._train_current_btn_style = _compact_btn
        self._train_confirm_btn_style = _confirm_action_btn
        self.btn_train.setStyleSheet(self._train_action_btn_style)
        self.btn_train.clicked.connect(self._train_all_tools)
        train_row.addWidget(self.btn_train, 1)

        self.btn_train_cancel = QtWidgets.QPushButton("脳")
        self.btn_train_cancel.setToolTip(tr("debug.cancel_train_confirm"))
        self.btn_train_cancel.setStyleSheet(_cancel_action_btn)
        self.btn_train_cancel.setVisible(False)
        self.btn_train_cancel.clicked.connect(lambda: self._cancel_training_pending_action("all"))
        train_row.addWidget(self.btn_train_cancel, 0)
        action_vbox.addLayout(train_row)

        train_current_row = QtWidgets.QHBoxLayout()
        train_current_row.setContentsMargins(0, 0, 0, 0)
        train_current_row.setSpacing(4)

        self.btn_train_current = QtWidgets.QPushButton(tr("debug.calibrate_current_tool"))
        self.btn_train_current.setStyleSheet(self._train_current_btn_style)
        self.btn_train_current.clicked.connect(self._train)
        train_current_row.addWidget(self.btn_train_current, 1)

        self.btn_train_current_cancel = QtWidgets.QPushButton("脳")
        self.btn_train_current_cancel.setToolTip(tr("debug.cancel_current_confirm"))
        self.btn_train_current_cancel.setStyleSheet(_cancel_action_btn)
        self.btn_train_current_cancel.setVisible(False)
        self.btn_train_current_cancel.clicked.connect(lambda: self._cancel_training_pending_action("current"))
        train_current_row.addWidget(self.btn_train_current_cancel, 0)
        action_vbox.addLayout(train_current_row)

        act_row = QtWidgets.QHBoxLayout()
        act_row.setSpacing(4)
        self.btn_test = QtWidgets.QPushButton(_si(SP.SP_MediaPlay), tr("debug.test_current_image"))
        self.btn_test.setStyleSheet(_compact_btn)
        self.btn_test.clicked.connect(self._run_test)
        self.btn_export_test = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), tr("debug.export_report"))
        self.btn_export_test.setStyleSheet(_compact_btn)
        self.btn_export_test.clicked.connect(self._export_current_results_csv)
        self.btn_clear_session = QtWidgets.QPushButton(_si(SP.SP_MediaPlay), tr("debug.test_all_test_samples"))
        self.btn_clear_session.setStyleSheet(_compact_btn)
        self.btn_clear_session.clicked.connect(self._run_all_test_samples)
        act_row.addWidget(self.btn_test)
        act_row.addWidget(self.btn_export_test)
        act_row.addWidget(self.btn_clear_session)
        action_vbox.addLayout(act_row)
        right_vbox.addWidget(action_frame)
        self._sync_training_action_buttons()

        body.addWidget(right_panel, 0)
        root.addLayout(body, 1)

        # Footer
        footer = QtWidgets.QFrame()
        footer.setFixedHeight(28)
        footer.setStyleSheet(f"background:{_HEADER_BG};border-top:1px solid #505050;")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 0, 16, 0)
        footer_layout.setSpacing(20)

        self.lbl_footer_ref = QtWidgets.QLabel(tr("debug.reference_image_not_set"))
        self.lbl_footer_ref.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_ref)
        self.lbl_footer_algo = QtWidgets.QLabel("")
        self.lbl_footer_algo.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_algo)
        footer_layout.addStretch(1)
        self.lbl_footer_product_dir = QtWidgets.QLabel("")
        self.lbl_footer_product_dir.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_product_dir)
        root.addWidget(footer)

        # Hidden controls kept for dialog/API compatibility

        # Hidden ROI annotation toolbar
        roi_bar_w = QtWidgets.QWidget(self)
        roi_bar = QtWidgets.QHBoxLayout(roi_bar_w)
        self._manual_roi_bar = roi_bar
        self.lbl_roi_shape_caption = QtWidgets.QLabel(tr("debug.shape"))
        roi_bar.addWidget(self.lbl_roi_shape_caption)
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(SUPPORTED_SHAPES)
        self.cmb_shape.setCurrentText("rect")
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        roi_bar.addWidget(self.cmb_shape)
        self.lbl_roi_label_caption = QtWidgets.QLabel(tr("debug.label"))
        roi_bar.addWidget(self.lbl_roi_label_caption)
        self.cmb_label = QtWidgets.QComboBox()
        self.cmb_label.addItems(["roi", "anchor", "anchor_mask"])
        self.cmb_label.setCurrentText("roi")
        self.cmb_label.currentTextChanged.connect(self._on_label_changed)
        roi_bar.addWidget(self.cmb_label)
        self.btn_save = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), tr("debug.save_annotation"))
        self.btn_save.clicked.connect(self._save_current_rect)
        self.btn_clear = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), tr("debug.clear_annotation"))
        self.btn_clear.clicked.connect(self._clear_current_rect)
        roi_bar.addWidget(self.btn_save)
        roi_bar.addWidget(self.btn_clear)
        roi_bar_w.hide()

        # 自动 ROI 对话框
        auto_box = QtWidgets.QGroupBox(tr("debug.auto_roi"))
        auto_l = QtWidgets.QGridLayout(auto_box)
        self._auto_roi_layout = auto_l
        auto_l.setHorizontalSpacing(10)
        auto_l.setVerticalSpacing(10)
        auto_l.setColumnStretch(0, 1)
        auto_l.setColumnStretch(1, 1)
        auto_l.setColumnStretch(2, 1)
        self.lbl_ref = QtWidgets.QLabel(tr("debug.reference_not_set"))
        self.btn_set_ref = QtWidgets.QPushButton(_si(SP.SP_ArrowRight), tr("debug.set_as_reference"))
        self.btn_set_ref.clicked.connect(self._set_ref_from_current)
        self.btn_pick_ref = QtWidgets.QPushButton(_si(SP.SP_DirOpenIcon), tr("debug.pick_reference"))
        self.btn_pick_ref.clicked.connect(self._pick_ref_image)
        auto_l.addWidget(self.lbl_ref, 0, 0, 1, 3)
        auto_l.addWidget(self.btn_set_ref, 1, 0)
        auto_l.addWidget(self.btn_pick_ref, 1, 1)
        self.lbl_loc_method = QtWidgets.QLabel(tr("debug.location_method"))
        auto_l.addWidget(self.lbl_loc_method, 2, 0)
        self.cmb_loc = QtWidgets.QComboBox()
        self.cmb_loc.addItems(SUPPORTED_LOC_MODES)
        self.cmb_loc.setCurrentText(self.loc_method)
        self.cmb_loc.currentTextChanged.connect(self._on_loc_method_changed)
        auto_l.addWidget(self.cmb_loc, 2, 1)
        self.chk_only_missing = QtWidgets.QCheckBox(tr("debug.only_missing_roi"))
        self.chk_only_missing.setChecked(True)
        auto_l.addWidget(self.chk_only_missing, 0, 0, 1, 3, QtCore.Qt.AlignmentFlag.AlignRight)
        self.btn_autogen = QtWidgets.QPushButton(_si(SP.SP_FileDialogListView), tr("debug.batch_roi_current"))
        self.btn_autogen.clicked.connect(self._autogen_roi_current_tab)
        self.btn_autogen_all = QtWidgets.QPushButton(_si(SP.SP_FileDialogListView), tr("debug.batch_roi_all"))
        self.btn_autogen_all.clicked.connect(self._autogen_roi_all)
        self.btn_clear_roi_batch = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), tr("debug.clear_roi_current"))
        self.btn_clear_roi_batch.clicked.connect(self._clear_roi_current_tab)
        auto_l.addWidget(self.btn_autogen, 1, 0)
        auto_l.addWidget(self.btn_autogen_all, 1, 1)
        auto_l.addWidget(self.btn_clear_roi_batch, 1, 2)
        self.template_match_box = auto_box
        self._update_loc_ui()

        # Camera debug page, laid out in the same style as the MVS page
        self.camera_debug_page = QtWidgets.QWidget()
        self.camera_debug_page.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")
        cam_main = QtWidgets.QHBoxLayout(self.camera_debug_page)
        cam_main.setContentsMargins(0, 0, 0, 0)
        cam_main.setSpacing(0)

        # Left side: device list and device information
        cam_left = QtWidgets.QFrame()
        cam_left.setFixedWidth(220)
        cam_left.setStyleSheet(f"QFrame{{background:{_PANEL_BG};border-right:1px solid #505050;}}")
        cam_left_vbox = QtWidgets.QVBoxLayout(cam_left)
        cam_left_vbox.setContentsMargins(0, 0, 0, 0)
        cam_left_vbox.setSpacing(0)

        self.lbl_cam_left_title = QtWidgets.QLabel(tr("debug.device_list"))
        cam_left_title = self.lbl_cam_left_title
        cam_left_title.setFixedHeight(28)
        cam_left_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        cam_left_vbox.addWidget(cam_left_title)

        role_row = QtWidgets.QWidget()
        role_layout = QtWidgets.QHBoxLayout(role_row)
        role_layout.setContentsMargins(8, 6, 8, 2)
        role_layout.setSpacing(6)
        self.lbl_debug_role = QtWidgets.QLabel(tr("debug.debug_role"))
        self.lbl_debug_role.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        role_layout.addWidget(self.lbl_debug_role)
        self.cmb_debug_camera_role = QtWidgets.QComboBox()
        self.cmb_debug_camera_role.setStyleSheet(_input_style)
        self.cmb_debug_camera_role.addItem("cam1", "cam1")
        self.cmb_debug_camera_role.addItem("cam2", "cam2")
        self.cmb_debug_camera_role.currentIndexChanged.connect(self._on_debug_camera_role_changed)
        role_layout.addWidget(self.cmb_debug_camera_role, 1)
        cam_left_vbox.addWidget(role_row)

        self.cmb_debug_camera = QtWidgets.QComboBox()
        self.cmb_debug_camera.setStyleSheet(_input_style)
        self.cmb_debug_camera.currentIndexChanged.connect(self._on_debug_camera_selected)
        cam_left_vbox.addWidget(self.cmb_debug_camera)

        self.btn_debug_refresh_camera = QtWidgets.QPushButton(_si(SP.SP_BrowserReload), tr("debug.scan_camera"))
        self.btn_debug_refresh_camera.setStyleSheet(_compact_btn)
        self.btn_debug_refresh_camera.clicked.connect(self._refresh_debug_camera_list)
        cam_left_vbox.addWidget(self.btn_debug_refresh_camera)

        cam_left_vbox.addSpacing(8)
        self.lbl_cam_info_title = QtWidgets.QLabel(tr("debug.device_info"))
        cam_info_title = self.lbl_cam_info_title
        cam_info_title.setFixedHeight(28)
        cam_info_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;border-top:1px solid #505050;padding-left:8px;")
        cam_left_vbox.addWidget(cam_info_title)

        self.lbl_debug_camera_info = QtWidgets.QLabel(tr("debug.camera_info"))
        self.lbl_debug_camera_info.setWordWrap(True)
        self.lbl_debug_camera_info.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;padding:8px;")
        cam_left_vbox.addWidget(self.lbl_debug_camera_info)
        self.lbl_debug_current_role = QtWidgets.QLabel(tr("debug.current_debug_role", role="cam1"))
        self.lbl_debug_current_role.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;padding:0 8px 8px 8px;")
        cam_left_vbox.addWidget(self.lbl_debug_current_role)
        cam_left_vbox.addStretch(1)
        cam_main.addWidget(cam_left)

        # Preview area, toolbar, and camera controls
        cam_center = QtWidgets.QWidget()
        cam_center_vbox = QtWidgets.QVBoxLayout(cam_center)
        cam_center_vbox.setContentsMargins(0, 0, 0, 0)
        cam_center_vbox.setSpacing(0)

        cam_toolbar = QtWidgets.QFrame()
        cam_toolbar.setFixedHeight(36)
        cam_toolbar.setStyleSheet(f"QFrame{{background:{_HEADER_BG};border-bottom:1px solid #505050;}}")
        cam_tb_layout = QtWidgets.QHBoxLayout(cam_toolbar)
        cam_tb_layout.setContentsMargins(8, 2, 8, 2)
        cam_tb_layout.setSpacing(6)

        self.btn_debug_connect_camera = QtWidgets.QPushButton(_si(SP.SP_DriveNetIcon), tr("debug.connect"))
        self.btn_debug_connect_camera.setStyleSheet(_compact_btn)
        self.btn_debug_connect_camera.clicked.connect(self._connect_debug_camera)
        cam_tb_layout.addWidget(self.btn_debug_connect_camera)

        self.btn_debug_disconnect_camera = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), tr("debug.disconnect"))
        self.btn_debug_disconnect_camera.setStyleSheet(_compact_btn)
        self.btn_debug_disconnect_camera.clicked.connect(self._disconnect_debug_camera)
        cam_tb_layout.addWidget(self.btn_debug_disconnect_camera)

        cam_tb_layout.addSpacing(12)

        self.btn_debug_live_preview = QtWidgets.QPushButton(_si(SP.SP_MediaPlay), tr("debug.live_preview"))
        self.btn_debug_live_preview.setCheckable(True)
        self.btn_debug_live_preview.setStyleSheet(_compact_btn)
        self.btn_debug_live_preview.toggled.connect(self._toggle_debug_camera_preview)
        cam_tb_layout.addWidget(self.btn_debug_live_preview)

        self.btn_debug_grab_once = QtWidgets.QPushButton(_si(SP.SP_DesktopIcon), tr("debug.grab_to_test"))
        self.btn_debug_grab_once.setStyleSheet(_compact_btn)
        self.btn_debug_grab_once.clicked.connect(self._grab_debug_camera_once)
        cam_tb_layout.addWidget(self.btn_debug_grab_once)

        cam_tb_layout.addSpacing(12)

        self.btn_debug_save_image = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), tr("debug.save_image"))
        self.btn_debug_save_image.setStyleSheet(_compact_btn)
        self.btn_debug_save_image.clicked.connect(self._save_debug_camera_image)
        cam_tb_layout.addWidget(self.btn_debug_save_image)

        cam_tb_layout.addStretch(1)
        cam_center_vbox.addWidget(cam_toolbar)

        self.view_debug_camera = RuntimeImageView(tr("debug.preview_title"))
        self.view_debug_camera.setMinimumSize(640, 400)
        self.view_debug_camera.set_runtime_pixmap(None, placeholder=tr("debug.preview_closed"))
        cam_center_vbox.addWidget(self.view_debug_camera, 1)

        cam_statusbar = QtWidgets.QFrame()
        cam_statusbar.setFixedHeight(26)
        cam_statusbar.setStyleSheet(f"QFrame{{background:{_HEADER_BG};border-top:1px solid #505050;}}")
        cam_sb_layout = QtWidgets.QHBoxLayout(cam_statusbar)
        cam_sb_layout.setContentsMargins(10, 0, 10, 0)
        self.lbl_debug_camera_status = QtWidgets.QLabel(tr("debug.camera_status_unscanned"))
        self.lbl_debug_camera_status.setWordWrap(False)
        self.lbl_debug_camera_status.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        cam_sb_layout.addWidget(self.lbl_debug_camera_status)
        cam_center_vbox.addWidget(cam_statusbar)
        cam_main.addWidget(cam_center, 1)

        # Right side: camera parameter controls
        cam_right = QtWidgets.QFrame()
        cam_right.setFixedWidth(240)
        cam_right.setStyleSheet(f"QFrame{{background:{_PANEL_BG};border-left:1px solid #505050;}}")
        cam_right_vbox = QtWidgets.QVBoxLayout(cam_right)
        cam_right_vbox.setContentsMargins(0, 0, 0, 0)
        cam_right_vbox.setSpacing(0)

        self.lbl_cam_right_title = QtWidgets.QLabel(tr("debug.param_settings"))
        cam_right_title = self.lbl_cam_right_title
        cam_right_title.setFixedHeight(28)
        cam_right_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        cam_right_vbox.addWidget(cam_right_title)

        cam_params = QtWidgets.QWidget()
        cam_params_form = QtWidgets.QFormLayout(cam_params)
        self.cam_params_form = cam_params_form
        cam_params_form.setContentsMargins(12, 12, 12, 12)
        cam_params_form.setSpacing(10)
        self.view_debug_camera.set_runtime_pixmap(None, placeholder=tr("debug.preview_closed"))

        self.spin_debug_exposure = QtWidgets.QDoubleSpinBox()
        self.spin_debug_exposure.setDecimals(1)
        self.spin_debug_exposure.setRange(1.0, 1000000.0)
        self.spin_debug_exposure.setValue(20000.0)
        self.spin_debug_exposure.setStyleSheet(_input_style)
        self._debug_exposure_row_label = tr("debug.exposure")
        cam_params_form.addRow(self._debug_exposure_row_label, self.spin_debug_exposure)
        self.lbl_debug_camera_status = QtWidgets.QLabel(tr("debug.camera_status_unscanned"))
        self.spin_debug_gain = QtWidgets.QDoubleSpinBox()
        self.spin_debug_gain.setDecimals(2)
        self.spin_debug_gain.setRange(0.0, 48.0)
        self.spin_debug_gain.setValue(0.0)
        self.spin_debug_gain.setStyleSheet(_input_style)
        self._debug_gain_row_label = tr("debug.gain")
        cam_params_form.addRow(self._debug_gain_row_label, self.spin_debug_gain)

        self.chk_debug_digital_shift_enable = QtWidgets.QCheckBox(tr("debug.enable"))
        self.chk_debug_digital_shift_enable.setStyleSheet(f"color:{_TEXT_LIGHT};")
        self._debug_shift_enable_row_label = tr("debug.digital_shift_enable")
        cam_params_form.addRow(self._debug_shift_enable_row_label, self.chk_debug_digital_shift_enable)

        self.spin_debug_digital_shift = QtWidgets.QDoubleSpinBox()
        self.spin_debug_digital_shift.setDecimals(4)
        self.spin_debug_digital_shift.setRange(0.0, 16.0)
        self.spin_debug_digital_shift.setValue(0.0)
        self.spin_debug_digital_shift.setStyleSheet(_input_style)
        self.spin_debug_digital_shift.setEnabled(False)
        self.spin_debug_digital_shift.setToolTip("Digital Shift")
        self._debug_shift_row_label = tr("debug.digital_shift")
        cam_params_form.addRow(self._debug_shift_row_label, self.spin_debug_digital_shift)

        # Avoid QPushButton auto-default behavior stealing Enter from spin boxes.
        self.spin_debug_exposure.setKeyboardTracking(False)
        self.spin_debug_gain.setKeyboardTracking(False)
        self.spin_debug_digital_shift.setKeyboardTracking(False)
        self.spin_debug_exposure.editingFinished.connect(self._on_debug_camera_param_editing_finished)
        self.spin_debug_gain.editingFinished.connect(self._on_debug_camera_param_editing_finished)
        self.spin_debug_digital_shift.editingFinished.connect(self._on_debug_camera_param_editing_finished)
        self.chk_debug_digital_shift_enable.toggled.connect(self.spin_debug_digital_shift.setEnabled)
        self.chk_debug_digital_shift_enable.toggled.connect(
            lambda _checked: self._on_debug_camera_param_editing_finished()
        )

        self.cmb_debug_trigger_mode = QtWidgets.QComboBox()
        self.cmb_debug_trigger_mode.addItems(["software", "continuous"])
        self.cmb_debug_trigger_mode.setCurrentText("continuous")
        self.cmb_debug_trigger_mode.setStyleSheet(_input_style)
        self._debug_trigger_row_label = tr("debug.trigger_mode")
        cam_params_form.addRow(self._debug_trigger_row_label, self.cmb_debug_trigger_mode)
        # Use activated so programmatic setCurrentIndex calls do not apply hardware changes.
        self.cmb_debug_trigger_mode.activated.connect(self._on_debug_camera_trigger_activated)

        self.cmb_debug_light_source_mode = QtWidgets.QComboBox()
        self.cmb_debug_light_source_mode.addItem(tr("debug.board_do_light"), "board_io")
        self.cmb_debug_light_source_mode.addItem(tr("debug.camera_line1_strobe"), "camera_line1_strobe")
        self.cmb_debug_light_source_mode.setCurrentIndex(0)
        self.cmb_debug_light_source_mode.setStyleSheet(_input_style)
        self.cmb_debug_light_source_mode.setToolTip(tr("debug.camera_line1_tip"))
        self._debug_light_row_label = tr("debug.light_source")
        cam_params_form.addRow(self._debug_light_row_label, self.cmb_debug_light_source_mode)
        self.cmb_debug_light_source_mode.activated.connect(self._on_debug_camera_trigger_activated)

        cam_right_vbox.addWidget(cam_params)
        cam_right_vbox.addSpacing(8)

        cam_btns_w = QtWidgets.QWidget()
        cam_btns_layout = QtWidgets.QVBoxLayout(cam_btns_w)
        cam_btns_layout.setContentsMargins(12, 0, 12, 12)
        cam_btns_layout.setSpacing(6)

        self.btn_debug_read_camera_settings = QtWidgets.QPushButton(_si(SP.SP_FileDialogInfoView), tr("debug.read_camera_params"))
        self.btn_debug_read_camera_settings.setStyleSheet(_compact_btn)
        self.btn_debug_read_camera_settings.clicked.connect(self._refresh_debug_camera_settings)
        cam_btns_layout.addWidget(self.btn_debug_read_camera_settings)

        self.btn_debug_apply_camera_settings = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), tr("debug.apply_camera_params"))
        self.btn_debug_apply_camera_settings.setStyleSheet(_compact_btn)
        self.btn_debug_apply_camera_settings.clicked.connect(self._apply_debug_camera_settings)
        cam_btns_layout.addWidget(self.btn_debug_apply_camera_settings)

        cam_right_vbox.addWidget(cam_btns_w)
        cam_right_vbox.addStretch(1)
        cam_main.addWidget(cam_right)

        # IO debug page, laid out in the same style as the MVS page
        self.io_debug_page = QtWidgets.QWidget()
        self.io_debug_page.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")
        io_main = QtWidgets.QHBoxLayout(self.io_debug_page)
        io_main.setContentsMargins(0, 0, 0, 0)
        io_main.setSpacing(0)

        io_left = QtWidgets.QFrame()
        io_left.setFixedWidth(260)
        io_left.setStyleSheet(f"QFrame{{background:{_PANEL_BG};border-right:1px solid #505050;}}")
        io_left_vbox = QtWidgets.QVBoxLayout(io_left)
        io_left_vbox.setContentsMargins(0, 0, 0, 0)
        io_left_vbox.setSpacing(0)

        self.lbl_io_ctrl_title = QtWidgets.QLabel(tr("debug.connection_control"))
        io_ctrl_title = self.lbl_io_ctrl_title
        io_ctrl_title.setFixedHeight(28)
        io_ctrl_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        io_left_vbox.addWidget(io_ctrl_title)

        io_ctrl_w = QtWidgets.QWidget()
        io_ctrl_layout = QtWidgets.QVBoxLayout(io_ctrl_w)
        io_ctrl_layout.setContentsMargins(10, 10, 10, 10)
        io_ctrl_layout.setSpacing(6)

        self.btn_debug_open_io = QtWidgets.QPushButton(_si(SP.SP_DriveNetIcon), tr("debug.open_io_debug"))
        self.btn_debug_open_io.setStyleSheet(_compact_btn)
        self.btn_debug_open_io.clicked.connect(self._open_debug_io)
        io_ctrl_layout.addWidget(self.btn_debug_open_io)

        self.btn_debug_close_io = QtWidgets.QPushButton(_si(SP.SP_DialogCloseButton), tr("debug.close_io_debug"))
        self.btn_debug_close_io.setStyleSheet(_compact_btn)
        self.btn_debug_close_io.clicked.connect(self._close_debug_io)
        io_ctrl_layout.addWidget(self.btn_debug_close_io)

        self.btn_debug_refresh_io = QtWidgets.QPushButton(_si(SP.SP_BrowserReload), tr("debug.refresh_dido"))
        self.btn_debug_refresh_io.setStyleSheet(_compact_btn)
        self.btn_debug_refresh_io.clicked.connect(self._refresh_debug_io_snapshot)
        io_ctrl_layout.addWidget(self.btn_debug_refresh_io)

        io_left_vbox.addWidget(io_ctrl_w)
        io_left_vbox.addSpacing(4)

        self.lbl_io_status_title = QtWidgets.QLabel(tr("debug.io_status"))
        io_status_title = self.lbl_io_status_title
        io_status_title.setFixedHeight(28)
        io_status_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;border-top:1px solid #505050;padding-left:8px;")
        io_left_vbox.addWidget(io_status_title)

        io_status_w = QtWidgets.QWidget()
        io_status_layout = QtWidgets.QVBoxLayout(io_status_w)
        io_status_layout.setContentsMargins(10, 10, 10, 10)
        io_status_layout.setSpacing(6)

        self.lbl_debug_di_snapshot = QtWidgets.QLabel(tr("debug.di_disconnected"))
        self.lbl_debug_di_snapshot.setWordWrap(True)
        self.lbl_debug_di_snapshot.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        io_status_layout.addWidget(self.lbl_debug_di_snapshot)

        self.lbl_debug_do_snapshot = QtWidgets.QLabel(tr("debug.do_disconnected"))
        self.lbl_debug_do_snapshot.setWordWrap(True)
        self.lbl_debug_do_snapshot.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        io_status_layout.addWidget(self.lbl_debug_do_snapshot)

        self.lbl_debug_io_mapping_summary = QtWidgets.QLabel(tr("debug.mapping_not_loaded"))
        self.lbl_debug_io_mapping_summary.setWordWrap(True)
        self.lbl_debug_io_mapping_summary.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        io_status_layout.addWidget(self.lbl_debug_io_mapping_summary)

        io_left_vbox.addWidget(io_status_w)
        io_left_vbox.addStretch(1)
        io_main.addWidget(io_left)

        io_right = QtWidgets.QWidget()
        io_right_vbox = QtWidgets.QVBoxLayout(io_right)
        io_right_vbox.setContentsMargins(0, 0, 0, 0)
        io_right_vbox.setSpacing(12)

        self.lbl_io_panel_title = QtWidgets.QLabel(tr("debug.dido_panel"))
        io_panel_title = self.lbl_io_panel_title
        io_panel_title.setFixedHeight(28)
        io_panel_title.setStyleSheet(
            f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;"
            "border-bottom:1px solid #505050;padding-left:8px;"
        )
        io_right_vbox.addWidget(io_panel_title)

        io_panel_w = QtWidgets.QWidget()
        io_panel_w.setStyleSheet(f"background:{_DARK_BG};")
        io_panel_layout = QtWidgets.QVBoxLayout(io_panel_w)
        io_panel_layout.setContentsMargins(16, 16, 16, 16)
        io_panel_layout.setSpacing(16)

        _channel_card_css = (
            f"QFrame{{background:{_PANEL_BG};border:1px solid #4f4f4f;border-radius:8px;}}"
        )
        _di_indicator_off = "background:#7a7a7a;border:2px solid #9a9a9a;border-radius:16px;"
        _do_button_css = (
            "QPushButton{background:#4a4a4a;color:#d8d8d8;border:1px solid #666666;"
            "border-radius:6px;padding:8px 6px;font-size:12px;font-weight:bold;}"
            "QPushButton:hover:!disabled{background:#5b5b5b;}"
            "QPushButton:checked{background:#1f9d55;color:white;border:1px solid #1f9d55;}"
            "QPushButton:disabled{background:#363636;color:#737373;border:1px solid #474747;}"
        )

        self.lbl_di_title = QtWidgets.QLabel(tr("debug.di_monitor"))
        di_title = self.lbl_di_title
        di_title.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:13px;font-weight:bold;")
        io_panel_layout.addWidget(di_title)

        di_grid = QtWidgets.QGridLayout()
        di_grid.setHorizontalSpacing(10)
        di_grid.setVerticalSpacing(10)
        for channel in range(16):
            card = QtWidgets.QFrame()
            card.setStyleSheet(_channel_card_css)
            card.setVisible(False)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(6)

            title = QtWidgets.QLabel(f"DI_{channel}")
            title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:11px;font-weight:bold;border:none;")
            card_layout.addWidget(title)

            indicator = QtWidgets.QLabel()
            indicator.setFixedSize(32, 32)
            indicator.setStyleSheet(_di_indicator_off)
            indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(indicator, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

            hint = QtWidgets.QLabel(tr("debug.unmapped"))
            hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color:{_TEXT_DIM};font-size:9px;border:none;")
            card_layout.addWidget(hint)

            self._debug_di_cards[channel] = card
            self._debug_di_indicators[channel] = indicator
            self._debug_di_hints[channel] = hint
            di_grid.addWidget(card, channel // 8, channel % 8)

        io_panel_layout.addLayout(di_grid)

        self.lbl_do_title = QtWidgets.QLabel(tr("debug.do_control"))
        do_title = self.lbl_do_title
        do_title.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:13px;font-weight:bold;")
        io_panel_layout.addWidget(do_title)

        do_grid = QtWidgets.QGridLayout()
        do_grid.setHorizontalSpacing(10)
        do_grid.setVerticalSpacing(10)
        for channel in range(16):
            card = QtWidgets.QFrame()
            card.setStyleSheet(_channel_card_css)
            card.setVisible(False)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(6)

            button = QtWidgets.QPushButton(f"DO_{channel}")
            button.setCheckable(True)
            button.setEnabled(False)
            button.setMinimumHeight(34)
            button.setStyleSheet(_do_button_css)
            button.toggled.connect(
                lambda checked, do_channel=channel: self._set_debug_output_channel(do_channel, checked)
            )
            card_layout.addWidget(button)

            hint = QtWidgets.QLabel(tr("debug.unmapped"))
            hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color:{_TEXT_DIM};font-size:9px;border:none;")
            card_layout.addWidget(hint)

            self._debug_do_cards[channel] = card
            self._debug_do_channel_buttons[channel] = button
            self._debug_do_hints[channel] = hint
            do_grid.addWidget(card, channel // 8, channel % 8)

        io_panel_layout.addLayout(do_grid)
        io_panel_layout.addStretch(1)

        io_right_vbox.addWidget(io_panel_w)
        io_right_vbox.addStretch(1)
        io_main.addWidget(io_right, 1)

        self.lbl_template_tool_hint = QtWidgets.QLabel("")
        self.lbl_template_tool_hint.hide()
        self._normalize_stylesheet_font_units()
        self._update_responsive_layout()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._update_responsive_layout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _toggle_tool_config_section(self, checked: bool) -> None:
        frame = getattr(self, "tool_config_scroll", None) or getattr(self, "tool_config_frame", None)
        if frame is not None:
            frame.setVisible(bool(checked))
        toggle = getattr(self, "btn_toggle_tools", None)
        if toggle is not None:
            toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )

    def _toggle_algorithm_section(self, checked: bool) -> None:
        frame = getattr(self, "algorithm_params_frame", None)
        if frame is not None:
            frame.setVisible(bool(checked))
        toggle = getattr(self, "btn_toggle_algo", None)
        if toggle is not None:
            toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )

    @staticmethod
    def _normalize_font_size_units(style_sheet: str) -> str:
        def replace(match: re.Match[str]) -> str:
            px = int(match.group(1))
            pt = max(1, int(round(px * 0.75)))
            return f"font-size:{pt}pt"

        return re.sub(r"font-size\s*:\s*(\d+)px", replace, style_sheet)

    def _normalize_stylesheet_font_units(self) -> None:
        for widget in [self, *self.findChildren(QtWidgets.QWidget)]:
            style_sheet = widget.styleSheet()
            if not style_sheet or "font-size" not in style_sheet or "px" not in style_sheet:
                continue
            normalized = self._normalize_font_size_units(style_sheet)
            if normalized != style_sheet:
                widget.setStyleSheet(normalized)

    def _update_responsive_layout(self) -> None:
        width = max(self.width(), 1)
        compact = width <= 1366
        right_panel = getattr(self, "_main_right_panel", None)
        if right_panel is not None:
            panel_width = 660 if compact else 500
            if width >= 1700:
                panel_width = 540
            right_panel.setFixedWidth(panel_width)
        if hasattr(self, "cmb_product"):
            self.cmb_product.setFixedWidth(160 if compact else 180)
        if hasattr(self, "btn_new_product"):
            self.btn_new_product.setFixedWidth(56 if compact else 60)
        if hasattr(self, "btn_delete_product"):
            self.btn_delete_product.setFixedWidth(56 if compact else 60)
        if hasattr(self, "cmb_current_camera_role"):
            self.cmb_current_camera_role.setFixedWidth(72 if compact else 84)
        if hasattr(self, "btn_algorithm_picker"):
            style = self._algorithm_picker_style_compact if compact else self._algorithm_picker_style_default
            if style and self.btn_algorithm_picker.styleSheet() != style:
                self.btn_algorithm_picker.setStyleSheet(style)


    # ------------------------------------------------------------------
    # 搴曟爮鍚屾
    # ------------------------------------------------------------------

    def _sync_footer(self) -> None:
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "-"
        self.lbl_footer_ref.setText(f"{tr('debug.reference_image')}: {ref_name}")
        algo = self.current_algorithm_display_name() if hasattr(self, "cmb_algorithm") else ""
        self.lbl_footer_algo.setText(f"{tr('debug.algorithm')}: {algo}" if algo else "")
        self.lbl_footer_product_dir.setText(f"{tr('debug.product_dir')}: {self.session.product_dir}")

    # ------------------------------------------------------------------
    # 列表刷新
    # ------------------------------------------------------------------

    def _refresh_lists(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        if hasattr(self, "lbl_images_section"):
            title = tr("debug.image_list")
            if current_role:
                if language_code().lower().startswith("zh"):
                    title = f"{title}（{current_role}）"
                else:
                    title = f"{title} ({current_role})"
            self.lbl_images_section.setText(title)

        def fill(
            listw: QtWidgets.QListWidget,
            files: List[str],
            *,
            sample_kind: str,
        ) -> None:
            current_item = listw.currentItem()
            current_path = None
            if current_item is not None:
                current_path = current_item.data(QtCore.Qt.UserRole) or current_item.toolTip()
            blocker = QtCore.QSignalBlocker(listw)
            listw.clear()
            selected_row = -1
            for index, p in enumerate(files):
                it = QtWidgets.QListWidgetItem(self._sample_item_display_text(p, sample_kind, current_role))
                it.setToolTip(p)
                it.setData(QtCore.Qt.UserRole, p)
                listw.addItem(it)
                if current_path and p == current_path:
                    selected_row = index
            if selected_row >= 0:
                listw.setCurrentRow(selected_row)
            del blocker

        fill(self.ok_list, self._sample_paths_for_kind("train", current_role), sample_kind="train")
        fill(self.ng_list, [], sample_kind="ng")
        fill(self.test_list, self._sample_paths_for_kind("test", current_role), sample_kind="test")
        self._update_sample_panel_widgets()

    def _save_session(self) -> None:
        self.session.save_session(SessionData(
            train_files=list(self.train_files),
            ok_files=list(self.ok_files),
            ng_files=list(self.ng_files),
            test_files=list(self.test_files),
            ref_image=self.ref_image,
            loc_method=self.loc_method,
        ))

    # ------------------------------------------------------------------
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "Not Set"
    # ------------------------------------------------------------------

    def _current_sample_tab_kind(self) -> str:
        return "train" if self.tabs.currentIndex() == 0 else "test"

    def _sample_paths_for_kind(
        self,
        kind: str,
        camera_role: object = None,
    ) -> List[str]:
        role = _normalize_camera_role(camera_role or _selected_image_list_camera_role(self)) or "cam1"
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind in {"train", "training"}:
            return self._train_sample_paths_for_role(role)
        return _filter_paths_for_camera(self, self.test_files, role)

    def _sample_usage_text(self, path: str) -> str:
        if path in self.train_files:
            return tr("debug.train_samples")
        if path in self.test_files:
            return tr("debug.test_samples")
        return "Uncategorized"

    def _inspection_label_names_for_role(self, camera_role: object = None) -> List[str]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        labels: List[str] = []
        seen: set[str] = set()
        for item in self.inspection_items:
            item_role = _normalize_camera_role(getattr(item, "camera_id", "")) or "cam1"
            if item_role != role:
                continue
            label = str(getattr(item, "roi_label", "") or "").strip()
            if not is_roi_label(label) or label in seen:
                continue
            labels.append(label)
            seen.add(label)
        if not labels:
            labels_getter = getattr(self, "_shape_output_labels", None)
            if callable(labels_getter):
                for label in labels_getter():
                    text = str(label or "").strip()
                    if not is_roi_label(text) or text in seen:
                        continue
                    labels.append(text)
                    seen.add(text)
        return labels

    def _sample_annotation_store_path(self) -> str:
        return os.path.join(self.session.product_dir, "sample_annotations.json")

    def _sample_annotation_path_key(self, path: object) -> str:
        return product_relative_path(path, base_dir=self.session.product_dir)

    def _sample_roi_annotation_key(self, camera_role: object, label_name: object) -> str:
        role = _normalize_camera_role(camera_role) or "cam1"
        label = str(label_name or "").strip()
        return f"{role}::{label}" if label else role

    def _load_sample_roi_annotations(self) -> None:
        self._sample_roi_annotations_by_path = {}
        store_path = self._sample_annotation_store_path()
        if not store_path:
            return
        raw_payload = load_json_with_backup(store_path, default=None)
        if raw_payload is None:
            return
        image_payload = raw_payload.get("images", raw_payload) if isinstance(raw_payload, dict) else {}
        if not isinstance(image_payload, dict):
            return
        for stored_path, payload in image_payload.items():
            resolved_path = resolve_product_path(
                stored_path,
                base_dir=self.session.product_dir,
                anchor_dir=self.session.product_dir,
                prefer_existing=False,
            )
            if not resolved_path:
                continue
            if isinstance(payload, dict):
                raw_labels = payload.get("roi_status", payload)
            else:
                raw_labels = {}
            if not isinstance(raw_labels, dict):
                continue
            normalized_labels: Dict[str, str] = {}
            for key, value in raw_labels.items():
                annotation_key = str(key or "").strip()
                annotation_value = str(value or "").strip().upper()
                if not annotation_key or annotation_value not in {"OK", "NG"}:
                    continue
                normalized_labels[annotation_key] = annotation_value
            if normalized_labels:
                self._sample_roi_annotations_by_path[os.path.normpath(resolved_path)] = normalized_labels

    def _save_sample_roi_annotations(self) -> None:
        store_path = self._sample_annotation_store_path()
        if not store_path:
            return
        images_payload: Dict[str, Dict[str, Dict[str, str]]] = {}
        for path, labels in sorted(self._sample_roi_annotations_by_path.items()):
            normalized_path = os.path.normpath(str(path or ""))
            if not normalized_path or not labels:
                continue
            key = self._sample_annotation_path_key(normalized_path)
            if not key:
                continue
            images_payload[key] = {"roi_status": dict(sorted(labels.items()))}
        if not images_payload:
            self._delete_sample_annotation_file()
            return
        atomic_write_json(store_path, {"images": images_payload}, ensure_ascii=False, indent=2)

    def _delete_sample_annotation_file(self) -> None:
        store_path = self._sample_annotation_store_path()
        try:
            if store_path and os.path.exists(store_path):
                os.remove(store_path)
            backup_path = backup_path_for(store_path) if store_path else None
            if backup_path is not None and backup_path.exists():
                backup_path.unlink()
        except Exception:
            pass

    def _path_has_roi_geometry(self, path: str, label_name: str) -> bool:
        if not path or not label_name:
            return False
        json_path = qr_core.labelme_json_of_image(path)
        if not os.path.exists(json_path):
            return False
        try:
            return qr_core.read_shape_from_labelme(json_path, label_name) is not None
        except Exception:
            return False

    def _path_has_roi_label(self, path: str, label_name: str) -> bool:
        return self._path_has_roi_geometry(path, label_name)

    def _sample_roi_status_for_path(
        self,
        path: str,
        camera_role: object,
        label_name: str,
    ) -> str:
        normalized_path = os.path.normpath(str(path or ""))
        if not normalized_path:
            return ""
        annotation_key = self._sample_roi_annotation_key(camera_role, label_name)
        return str(self._sample_roi_annotations_by_path.get(normalized_path, {}).get(annotation_key, "") or "").strip().upper()

    def _set_sample_roi_status_for_path(
        self,
        path: str,
        camera_role: object,
        label_name: str,
        status: object,
    ) -> None:
        normalized_path = os.path.normpath(str(path or ""))
        if not normalized_path:
            return
        annotation_key = self._sample_roi_annotation_key(camera_role, label_name)
        status_text = str(status or "").strip().upper()
        annotations = dict(self._sample_roi_annotations_by_path.get(normalized_path, {}))
        if status_text in {"OK", "NG"}:
            annotations[annotation_key] = status_text
        else:
            annotations.pop(annotation_key, None)
        if annotations:
            self._sample_roi_annotations_by_path[normalized_path] = annotations
        else:
            self._sample_roi_annotations_by_path.pop(normalized_path, None)
        self._save_sample_roi_annotations()

    def _mark_sample_path_all_ok(self, path: str, camera_role: object = None) -> None:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        for label in self._inspection_label_names_for_role(role):
            if not self._path_has_roi_geometry(path, label):
                continue
            self._set_sample_roi_status_for_path(path, role, label, "OK")

    def _clear_sample_path_annotations(self, path: str, camera_role: object = None) -> None:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        for label in self._inspection_label_names_for_role(role):
            self._set_sample_roi_status_for_path(path, role, label, "")

    def _sample_annotation_counts_for_roi(
        self,
        roi_label: str,
        camera_role: object = None,
        *,
        paths: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        target_paths = list(paths or self._sample_paths_for_kind("train", role))
        ok_count = 0
        ng_count = 0
        unset_count = 0
        for path in target_paths:
            if not self._path_has_roi_geometry(path, roi_label):
                unset_count += 1
                continue
            status = self._sample_roi_status_for_path(path, role, roi_label)
            if status == "OK":
                ok_count += 1
            elif status == "NG":
                ng_count += 1
            else:
                unset_count += 1
        return ok_count, ng_count, unset_count

    def _sample_annotation_progress_for_path(
        self,
        path: str,
        camera_role: object = None,
    ) -> Tuple[int, int]:
        labels = self._inspection_label_names_for_role(camera_role)
        if not labels:
            return 0, 0
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        present_count = sum(
            1
            for label in labels
            if self._path_has_roi_geometry(path, label)
            and self._sample_roi_status_for_path(path, role, label) in {"OK", "NG"}
        )
        return present_count, len(labels)

    def _sample_annotation_state_for_path(
        self,
        path: str,
        camera_role: object = None,
    ) -> str:
        labels = self._inspection_label_names_for_role(camera_role)
        if not labels:
            return tr("sample.unset")
        geometry_missing = sum(1 for label in labels if not self._path_has_roi_geometry(path, label))
        if geometry_missing:
            return tr("sample.missing_roi")
        present_count, total_count = self._sample_annotation_progress_for_path(path, camera_role)
        if total_count <= 0 or present_count <= 0:
            return tr("sample.unset")
        if present_count < total_count:
            return tr("sample.partial")
        return tr("sample.complete")

    def _sample_item_display_text(
        self,
        path: str,
        sample_kind: str,
        camera_role: object = None,
    ) -> str:
        status = self._sample_annotation_state_for_path(path, camera_role)
        name = os.path.basename(path)
        if str(sample_kind or "").strip().lower() in {"train", "training"}:
            return f"{name}    [{status}]"
        return f"{name}    [{status}]"

    def _current_image_sample_state_text(self) -> str:
        path = self.canvas.image_path()
        if not path:
            return tr("debug.current_image_state_none")
        return tr(
            "debug.current_image_state",
            usage=self._sample_usage_text(path),
            state=self._sample_annotation_state_for_path(path, self.current_camera_role()),
        )

    def _current_tool_sample_stats_text(self) -> str:
        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            return tr("debug.current_tool_stats_select")
        camera_role = _normalize_camera_role(getattr(inspection_item, "camera_id", "")) or self.current_camera_role()
        roi_label = str(getattr(inspection_item, "roi_label", "") or "").strip() or "roi"
        ok_count, ng_count, unset_count = self._sample_annotation_counts_for_roi(roi_label, camera_role)
        return tr("debug.current_tool_samples", roi=roi_label, ok=ok_count, ng=ng_count, unset=unset_count)

    def _training_validation_text(self) -> str:
        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            return tr("debug.training_validation_select")
        camera_role = _normalize_camera_role(getattr(inspection_item, "camera_id", "")) or self.current_camera_role()
        roi_label = str(getattr(inspection_item, "roi_label", "") or "").strip() or "roi"
        ok_files, ng_files, candidate_paths = self._training_sample_groups_for_role(camera_role, roi_label=roi_label)
        if not candidate_paths:
            return tr("debug.training_validation_no_samples", role=camera_role)
        _ok_count, _ng_count, missing_count = self._sample_annotation_counts_for_roi(roi_label, camera_role, paths=candidate_paths)
        if missing_count > 0:
            return tr("debug.training_validation_missing_annotations", roi=roi_label, count=missing_count)
        if not ok_files or not ng_files:
            missing_groups: List[str] = []
            if not ok_files:
                missing_groups.append("OK")
            if not ng_files:
                missing_groups.append("NG")
            return tr("debug.training_validation_missing_groups", roi=roi_label, groups="/".join(missing_groups))
        return tr("debug.training_validation_ready", roi=roi_label)

    def _update_sample_panel_widgets(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        train_count = len(self._sample_paths_for_kind("train", current_role))
        test_count = len(self._sample_paths_for_kind("test", current_role))
        if hasattr(self, "tabs"):
            self.tabs.setTabText(0, f"{tr('debug.train_samples')} ({train_count})")
            self.tabs.setTabText(1, f"{tr('debug.test_samples')} ({test_count})")
        current_image_label = getattr(self, "lbl_current_image_sample_state", None)
        if current_image_label is not None:
            current_image_label.setText(f"  {self._current_image_sample_state_text()}")
        tool_stats_label = getattr(self, "lbl_current_tool_sample_stats", None)
        if tool_stats_label is not None:
            tool_stats_label.setText(f"  {self._current_tool_sample_stats_text()}")
        validation_label = getattr(self, "lbl_training_validation", None)
        if validation_label is not None and not getattr(self, "_training_in_progress", False):
            validation_label.setText(self._training_validation_text())

        selected_path = self._current_selected_path()
        current_tab_kind = self._current_sample_tab_kind()
        for attr_name, enabled in (
            ("btn_train_to_test", current_tab_kind == "train" and bool(selected_path)),
            ("btn_del_ok", current_tab_kind == "train" and bool(selected_path)),
            ("btn_test_to_train", current_tab_kind == "test" and bool(selected_path)),
            ("btn_del_test", current_tab_kind == "test" and bool(selected_path)),
            ("btn_sample_annotation_test", current_tab_kind == "test" and test_count > 0),
        ):
            button = getattr(self, attr_name, None)
            if button is not None:
                button.setEnabled(enabled)

    def _select_path_in_current_tab(self, path: str) -> None:
        if not path:
            return
        list_widget = self.ok_list if self._current_sample_tab_kind() == "train" else self.test_list
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            item_path = item.data(QtCore.Qt.UserRole) or item.toolTip()
            if str(item_path or "") == str(path):
                blocker = QtCore.QSignalBlocker(list_widget)
                list_widget.setCurrentRow(row)
                del blocker
                self._show_selected_image_path(path)
                return

    def _move_selected_sample_to(self, target_kind: str) -> None:
        path = self._current_selected_path()
        if not path:
            return
        normalized_target = str(target_kind or "").strip().upper()
        for collection in (self.train_files, self.test_files, self.ok_files, self.ng_files):
            while path in collection:
                collection.remove(path)
        if normalized_target == "TRAIN":
            self.train_files.append(path)
            self.train_files = sorted(list(dict.fromkeys(self.train_files)))
            self.tabs.setCurrentIndex(0)
        else:
            self.test_files.append(path)
            self.test_files = sorted(list(dict.fromkeys(self.test_files)))
            self.tabs.setCurrentIndex(1)
        self._refresh_lists()
        self._clear_training_roi_review_state()
        self._save_session()
        self._select_path_in_current_tab(path)

    def _open_sample_annotation_dialog(self) -> None:
        dialog = getattr(self, "_sample_annotation_preview_dialog", None)
        if dialog is None:
            dialog = _SampleAnnotationPreviewDialog(self, self)
            self._sample_annotation_preview_dialog = dialog
            dialog.finished.connect(lambda *_: setattr(self, "_sample_annotation_preview_dialog", None))
        dialog.sync_camera_roles(self.configured_camera_roles())
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _current_selected_path(self) -> Optional[str]:
        tab = self.tabs.currentIndex()
        if tab == 0:
            items = self.ok_list.selectedItems()
            if not items:
                return None
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if path:
                return str(path)
            visible = self._sample_paths_for_kind("train", _selected_image_list_camera_role(self))
            row = self.ok_list.row(items[0])
            return visible[row] if row < len(visible) else None
        items = self.test_list.selectedItems()
        if not items:
            return None
        path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
        if path:
            return str(path)
        visible = _filter_paths_for_camera(self, self.test_files, _selected_image_list_camera_role(self))
        row = self.test_list.row(items[0])
        return visible[row] if row < len(visible) else None

    def _show_selected_image_path(self, path: Optional[str]) -> None:
        if not path:
            self._update_sample_panel_widgets()
            return
        self._clear_selected_inspection_item()
        if self.canvas.image_path() != path:
            self._load_canvas_image(path)
        self._set_status_for_current_image(path)
        self._update_sample_panel_widgets()

    def _clear_image_view_for_role_switch(self) -> None:
        for listw in (self.ok_list, self.ng_list, self.test_list):
            blocker = QtCore.QSignalBlocker(listw)
            listw.clearSelection()
            listw.setCurrentItem(None)
            del blocker
        self.canvas.clear_image()
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        self._current_result_rows = []
        self.lbl_status.setText(tr("debug.status_switched_camera_select", role=self.current_camera_role()))
        self._update_sample_panel_widgets()

    def _clear_selected_inspection_item(self) -> None:
        table = getattr(self, "inspection_items_table", None)
        if table is None:
            return
        table.clearSelection()
        table.setCurrentItem(None)

    def _on_select_ok(self) -> None:
        self._show_selected_image_path(self._current_selected_path())

    def _on_select_ng(self) -> None:
        self._show_selected_image_path(self._current_selected_path())

    def _on_select_test(self) -> None:
        self._show_selected_image_path(self._current_selected_path())

    def _add_images_to(self, kind: str) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            tr("debug.add_images_title", kind=kind),
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return
        normalized_kind = str(kind or "").strip().upper()
        if normalized_kind in {"TRAIN", "OK", "NG", "TRAIN_OK", "TRAIN_NG"}:
            self.train_files.extend(files)
            self.train_files = sorted(list(dict.fromkeys(self.train_files)))
        else:
            self.test_files.extend(files)
            self.test_files = sorted(list(dict.fromkeys(self.test_files)))
        self._refresh_lists()
        self._clear_training_roi_review_state()
        self._save_session()

    def _remove_selected_from(self, kind: str) -> None:
        normalized_kind = str(kind or "").strip().upper()
        if normalized_kind == "TRAIN":
            path = self._current_selected_path()
            if not path:
                return
            if path in self.train_files:
                self.train_files.remove(str(path))
        else:
            items = self.test_list.selectedItems()
            if not items:
                return
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if not path:
                visible = _filter_paths_for_camera(self, self.test_files, _selected_image_list_camera_role(self))
                idx = self.test_list.row(items[0])
                path = visible[idx] if idx < len(visible) else None
            if not path or path not in self.test_files:
                return
            self.test_files.remove(str(path))
        self._refresh_lists()
        self._clear_training_roi_review_state()
        self._save_session()

    def _clear_current_test_list(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        visible = self._sample_paths_for_kind("test", current_role)
        if not visible:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.clear_current_test_list_empty"))
            return
        role_text = tr("runtime.camera1") if current_role == "cam1" else tr("runtime.camera2")
        confirm = QtWidgets.QMessageBox.question(
            self,
            tr("debug.clear_current_test_list"),
            tr("debug.clear_current_test_list_confirm", role=role_text),
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        remove_set = {str(path) for path in visible if str(path)}
        self.test_files = [path for path in self.test_files if str(path) not in remove_set]
        self._refresh_lists()
        self._clear_training_roi_review_state()
        self._save_session()

    def _on_tab_changed(self, index: int) -> None:
        if index == 0:
            listw = self.ok_list
            files = self._sample_paths_for_kind("train", _selected_image_list_camera_role(self))
        else:
            listw = self.test_list
            files = self._sample_paths_for_kind("test", _selected_image_list_camera_role(self))

        self._update_sample_panel_widgets()
        if not files:
            return

        row = listw.currentRow()
        if row < 0 or row >= len(files):
            blocker = QtCore.QSignalBlocker(listw)
            listw.setCurrentRow(0)
            del blocker
            row = 0
        self._show_selected_image_path(files[row])

    # ------------------------------------------------------------------
    # 产品管理
    # ------------------------------------------------------------------

    def _on_product_changed(self, product_name: str) -> None:
        if not product_name or product_name == self.session.current_product:
            return
        self._save_session()
        self.productChangeRequested.emit(product_name)

    def _new_product(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            tr("debug.new_product_title"),
            tr("debug.new_product_prompt"),
        )
        if not ok or not name.strip():
            return
        error = self.session.create_product(name.strip())
        if error:
            QtWidgets.QMessageBox.warning(self, tr("common.error"), error)
            return
        self.cmb_product.addItem(name.strip())
        self.cmb_product.setCurrentText(name.strip())

    def _request_delete_product(self) -> None:
        name = str(self.cmb_product.currentText() or self.session.current_product or "").strip()
        if not name:
            return
        self.productDeleteRequested.emit(name)

    def _clear_session(self) -> None:
        ret = QtWidgets.QMessageBox.question(
            self,
            tr("debug.clear_session_title"),
            tr("debug.clear_session_confirm"),
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return
        self.sessionClearRequested.emit()

    # ------------------------------------------------------------------
    # Auto ROI
    # ------------------------------------------------------------------


    def _on_shape_changed(self) -> None:
        if self._current_label() == "anchor_mask" and self.cmb_shape.currentText() != "polygon":
            self.cmb_shape.setCurrentText("polygon")
            return
        self.canvas.draw_shape = self.cmb_shape.currentText()
        self.canvas._poly_pts = []
        self.canvas.update()
        self._on_shapes_changed()

    def _on_label_changed(self) -> None:
        label = self._current_label()
        if label == "anchor_mask":
            self.cmb_shape.setCurrentText("polygon")
            self.cmb_shape.setEnabled(False)
        else:
            self.cmb_shape.setEnabled(True)
        self._update_save_label_text()
        p = self.canvas.image_path()
        if p:
            self._load_shape_for_label(p, label)

    def _clear_current_rect(self) -> None:
        ret = QtWidgets.QMessageBox.question(
            self,
            tr("debug.clear_annotation_title"),
            tr("debug.clear_annotation_confirm"),
        )
        p = self.canvas.image_path()
        if p is not None:
            try:
                deleted = qr_core.delete_labelme_shape(p, label_name=self._current_label())
            except Exception:
                deleted = False
            if deleted:
                self._load_canvas_image(p)
                return
        self.canvas.update()
        self._on_shapes_changed()


    def _save_current_rect(self) -> None:
        p = self.canvas.image_path()
        if p is None:
            return
        st = self.canvas.roi
        label_name = self._current_label()

        if label_name == "anchor_mask" and st.shape_type != "polygon":
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.anchor_polygon_only"))
            return

        if st.shape_type == "rect":
            if st.xywh is None:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.draw_rect_first"))
                return
            jpath = qr_core.upsert_labelme_rect(p, st.xywh, label_name=label_name)
        else:
            if not st.points or len(st.points) < 3:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.polygon_min_points"))
                return
            jpath = qr_core.upsert_labelme_polygon(p, st.points, label_name=label_name)

        QtWidgets.QMessageBox.information(
            self,
            tr("debug.annotation_saved_title"),
            tr("debug.annotation_saved_message", path=jpath, label=label_name, shape=st.shape_type),
        )
        self._load_canvas_image(p)

    # ------------------------------------------------------------------
    # Training / calibration
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Auto ROI
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 绠楁硶鍙傛暟

    def _is_embedding_algorithm(self, algorithm: Optional[str] = None) -> bool:
        normalized = str(algorithm or self.current_algorithm() or "").strip()
        if not normalized:
            return False
        return self.algo.is_embedding_algorithm(normalized)

    def _embedding_model_path(self, algorithm: str) -> str:
        return self.algo.embedding_model_path(algorithm, self.session.product_dir)

    def _save_runtime_params(self) -> None:
        self.algo.save_params(self.session.product_params_path)

    def _apply_runtime_params_to_ui(self) -> None:
        self._updating_runtime_params = True
        try:
            algorithm = (
                self.algo.product_params.algorithm
                if self.algo.product_params.algorithm in SUPPORTED_ALGORITHMS
                else ""
            )
            score_mode = (
                self.algo.product_params.score_mode
                if self.algo.product_params.score_mode in SUPPORTED_SCORE_MODES
                else SUPPORTED_SCORE_MODES[0]
            )
            self._set_current_algorithm(algorithm)
            self.cmb_mode.setCurrentText(score_mode)
            self.spin_margin.setValue(float(self.algo.product_params.margin))
            self.spin_topk.setValue(max(1, int(self.algo.product_params.topk)))
        finally:
            self._updating_runtime_params = False
        self._update_runtime_widgets()
        self._update_learning_backbone_hint()

    def _update_runtime_widgets(self) -> None:
        algorithm_selected = bool(self.current_algorithm())
        embedding = algorithm_selected and self._is_embedding_algorithm()
        topk_enabled = embedding and self.cmb_mode.currentText() == "topk"
        inspection_items = list(getattr(self, "inspection_items", []) or [])
        current_role = self.current_camera_role()
        has_enabled_items = any(
            getattr(item, "enabled", False)
            and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
            for item in inspection_items
        )
        selected_item_fn = getattr(self, "_selected_inspection_item", None)
        selected_item = selected_item_fn() if callable(selected_item_fn) else None
        selected_tool_enabled = bool(
            algorithm_selected
            and selected_item is not None
            and selected_item.enabled
        )
        self.cmb_mode.setEnabled(embedding)
        self.spin_margin.setEnabled(embedding)
        self.spin_topk.setEnabled(topk_enabled)
        topk_label = getattr(self, "lbl_topk", None)
        if topk_label is not None:
            enabled_style = getattr(self, "_algo_param_label_style", "")
            disabled_style = getattr(self, "_algo_param_label_disabled_style", enabled_style)
            topk_label.setEnabled(topk_enabled)
            topk_label.setStyleSheet(enabled_style if topk_enabled else disabled_style)
        self.btn_train.setEnabled(has_enabled_items)
        train_current_button = getattr(self, "btn_train_current", None)
        if train_current_button is not None:
            train_current_button.setEnabled(selected_tool_enabled)
        self.btn_test.setEnabled(algorithm_selected)
        margin_button = getattr(self, "btn_validate_margin", None)
        if margin_button is not None:
            margin_button.setEnabled(embedding)
        embedding_button = getattr(self, "btn_embedding_analysis", None)
        if embedding_button is not None:
            embedding_button.setEnabled(embedding)
        if getattr(self, "_training_in_progress", False):
            self._set_training_running(True)
        self._sync_training_action_buttons()
        self._update_sample_panel_widgets()

    def _on_runtime_params_changed(self, *args) -> None:
        if self._updating_runtime_params:
            return
        self.algo.product_params.algorithm = self.current_algorithm()
        self.algo.product_params.score_mode = self.cmb_mode.currentText()
        self.algo.product_params.margin = float(self.spin_margin.value())
        self.algo.product_params.topk = int(self.spin_topk.value())
        if self._is_embedding_algorithm():
            self.algo.apply_params_to_model()
        self._save_runtime_params()
        self._update_runtime_widgets()
        self._update_learning_backbone_hint()

    def _on_algorithm_changed(self, *args) -> None:
        algorithm = self.current_algorithm()
        if self._updating_runtime_params:
            return
        selected_item = self._selected_inspection_item()
        if algorithm in SUPPORTED_EMBEDDING_ALGORITHMS:
            self.algo.set_learning_backbone(algorithm)
        if not algorithm:
            self.algo.product_params.algorithm = ""
            self.algo.model = None
            self._save_runtime_params()
            self._update_runtime_widgets()
            self.lbl_status.setText(tr("debug.training_validation_select"))
            return
        if selected_item is not None:
            selected_item.algorithm_code = (
                "shared_backbone_register"
                if algorithm in SUPPORTED_EMBEDDING_ALGORITHMS
                else algorithm
            )
            self._persist_inspection_items()
            self._refresh_inspection_items_table()
        self.algo.product_params.algorithm = algorithm
        self._save_runtime_params()
        self._update_runtime_widgets()
        self._update_learning_backbone_hint()
        try:
            _, msg = self.algo.load_model_for_algorithm(
                algorithm,
                self.session.product_dir,
                model_key=selected_item.model_key if selected_item is not None else "",
            )
            self.lbl_status.setText(msg)
        except Exception as exc:
            self.algo.model = None
            display_name = self.algo.algorithm_display_name(algorithm) or algorithm
            self.lbl_status.setText(f"Status: failed to load tool {display_name} - {exc}")

    # ------------------------------------------------------------------
    # 璁粌
    # ------------------------------------------------------------------

    def _resolve_training_algorithm(self, inspection_item: InspectionItem) -> str:
        if self.algo.is_learning_tool(inspection_item.algorithm_code):
            return self.algo.current_learning_backbone()
        return self.algo.resolve_tool_algorithm(inspection_item.algorithm_code)

    def _training_camera_roles_in_lists(self, camera_id: object | None = None) -> List[str]:
        if camera_id is None:
            candidate_paths = list(getattr(self, "train_files", []) or [])
            if not candidate_paths:
                candidate_paths = list(getattr(self, "ok_files", []) or []) + list(getattr(self, "ng_files", []) or [])
        else:
            candidate_paths = self._train_sample_paths_for_role(camera_id)
        roles = {
            _camera_role_from_path(path)
            for path in candidate_paths
        }
        roles.discard("")
        return sorted(roles)

    def _warn_mixed_training_camera_samples(self, camera_id: object | None = None) -> bool:
        roles = self._training_camera_roles_in_lists(camera_id)
        if len(roles) < 2:
            return False
        role_text = _normalize_camera_role(camera_id) if camera_id is not None else ""
        suffix = tr("debug.current_role_suffix", role=role_text) if role_text else ""
        QtWidgets.QMessageBox.warning(
            self,
            tr("debug.mixed_training_title"),
            tr("debug.mixed_training_message", suffix=suffix),
        )
        return True

    def _missing_training_roi_paths(self, roi_label: str, candidate_paths: List[str]) -> List[str]:
        missing_paths: List[str] = []
        for path in candidate_paths:
            json_path = qr_core.labelme_json_of_image(path)
            if not os.path.exists(json_path):
                missing_paths.append(path)
                continue
            if qr_core.read_shape_from_labelme(json_path, roi_label) is None:
                missing_paths.append(path)
        if missing_paths and self.loc_method == "shape":
            try:
                self._autogen_roi_for_images(missing_paths, only_missing=False, silent=True)
                refreshed_missing_paths: List[str] = []
                for path in candidate_paths:
                    json_path = qr_core.labelme_json_of_image(path)
                    if not os.path.exists(json_path) or qr_core.read_shape_from_labelme(json_path, roi_label) is None:
                        refreshed_missing_paths.append(path)
                missing_paths = refreshed_missing_paths
            except Exception:
                pass
        return missing_paths

    @staticmethod
    def _training_item_display_name(inspection_item: InspectionItem) -> str:
        return str(
            inspection_item.display_name
            or inspection_item.roi_label
            or inspection_item.item_id
            or "tool"
        ).strip() or "tool"

    def _build_training_task_for_item(self, inspection_item: InspectionItem) -> dict:
        if not inspection_item.enabled:
            raise RuntimeError("selected tool is disabled")

        algorithm = self._resolve_training_algorithm(inspection_item)
        if not algorithm:
            if self.algo.is_learning_tool(inspection_item.algorithm_code):
                raise RuntimeError("please choose a learning tool subtype first")
            raise RuntimeError("please select an inspection tool")

        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        training_ok_files, training_ng_files, candidate_paths = self._training_sample_groups_for_role(
            inspection_item.camera_id,
            roi_label=roi_label,
        )
        if not candidate_paths:
            camera_id = _normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing training images for {camera_id}")
        missing_groups: List[str] = []
        if not training_ok_files:
            missing_groups.append("OK")
        if not training_ng_files:
            missing_groups.append("NG")
        if missing_groups:
            camera_id = _normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing {'/'.join(missing_groups)} images for {camera_id}")

        missing_paths = self._missing_training_roi_paths(roi_label, candidate_paths)
        if missing_paths:
            missing = [os.path.basename(path) for path in missing_paths[:50]]
            raise RuntimeError(
                f"missing ROI label '{roi_label}' in some training sample jsons:\n" + "\n".join(missing)
            )

        embedding_cache_dir = ""
        is_embedding = False
        embedding_checker = getattr(self.algo, "is_embedding_algorithm", None)
        if callable(embedding_checker):
            is_embedding = bool(embedding_checker(algorithm))
        if is_embedding:
            cache_dir_getter = getattr(self.algo, "embedding_cache_dir", None)
            if callable(cache_dir_getter):
                embedding_cache_dir = str(cache_dir_getter(algorithm, self.session.product_dir) or "")

        return {
            "item_id": str(inspection_item.item_id or ""),
            "display_name": self._training_item_display_name(inspection_item),
            "camera_id": _normalize_camera_role(inspection_item.camera_id) or "cam1",
            "algorithm": algorithm,
            "ok_files": list(training_ok_files),
            "ng_files": list(training_ng_files),
            "label_names": [roi_label],
            "model_key": inspection_item.model_key,
            "embedding_cache_dir": embedding_cache_dir,
        }

    def _training_payload(self, mode: str, tasks: List[dict], *, selected_item_id: str = "", failures: Optional[List[str]] = None) -> dict:
        return {
            "mode": mode,
            "tasks": list(tasks),
            "product_dir": self.session.product_dir,
            "score_mode": self.cmb_mode.currentText(),
            "margin": float(self.spin_margin.value()),
            "topk": int(self.spin_topk.value()),
            "selected_item_id": str(selected_item_id or ""),
            "failure_messages": list(failures or []),
        }

    def _set_training_running(self, running: bool) -> None:
        self._training_in_progress = bool(running)
        for attr in ("btn_train", "btn_train_current", "btn_test", "btn_export_test", "btn_clear_session"):
            button = getattr(self, attr, None)
            if button is not None:
                button.setEnabled(not running)

    def _on_training_progress(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        status_text = f"Status: {text}"
        status_label = getattr(self, "lbl_status", None)
        if status_label is not None:
            status_label.setText(status_text)
        validation_label = getattr(self, "lbl_training_validation", None)
        if validation_label is not None:
            validation_label.setText(status_text)

    def _start_training_worker(self, payload: dict) -> None:
        if getattr(self, "_training_in_progress", False):
            QtWidgets.QMessageBox.information(self, tr("common.info"), "Training is already running.")
            return

        self._set_training_running(True)
        self._on_training_progress("training queued")

        if not isinstance(self, QtCore.QObject):
            worker = _TrainingJobWorker(self.algo, payload)
            result = worker._run_training()
            self._on_training_finished(result)
            return

        thread = QtCore.QThread(self)
        worker = _TrainingJobWorker(self.algo, payload)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_training_progress)
        worker.finished.connect(self._on_training_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._training_thread = thread
        self._training_worker = worker
        thread.start()

    def _on_training_finished(self, payload: object) -> None:
        result = dict(payload or {}) if isinstance(payload, dict) else {}
        self._training_thread = None
        self._training_worker = None
        self._set_training_running(False)

        last_status_message = str(result.get("last_status_message", "") or "")
        if last_status_message:
            self.lbl_status.setText(last_status_message)
            if hasattr(self, "lbl_training_validation"):
                self.lbl_training_validation.setText(last_status_message)

        success_names = [str(name) for name in result.get("success_names", []) if str(name)]
        failure_messages = [str(message) for message in result.get("failure_messages", []) if str(message)]
        mode = str(result.get("mode", "") or "")

        display_rows = list(result.get("display_rows", []) or [])
        if display_rows:
            self._populate_results_table(display_rows)

        if success_names:
            try:
                self._save_runtime_params()
            except Exception as exc:
                self.lbl_status.setText(f"Status: training done, but saving runtime params failed: {exc}")
            self._save_session()
        self._refresh_inspection_items_table()
        self._update_runtime_widgets()

        if failure_messages:
            summary_lines: List[str] = []
            if success_names:
                summary_lines.append(
                    f"Succeeded: {len(success_names)} tool(s) - " + ", ".join(success_names)
                )
            summary_lines.append(f"Failed: {len(failure_messages)} tool(s)")
            summary_lines.extend(failure_messages[:20])
            self.lbl_status.setText(
                f"Status: partial train done, success={len(success_names)}, failed={len(failure_messages)}"
            )
            QtWidgets.QMessageBox.warning(self, tr("debug.train_result_title"), "\n".join(summary_lines))
            return

        if mode == "current":
            dialog_message = str(result.get("last_dialog_message", "") or "").strip()
            QtWidgets.QMessageBox.information(
                self,
                tr("debug.train_done_title"),
                dialog_message or "OK/NG training finished. You can start testing.",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            tr("debug.train_result_title"),
            f"Finished training/calibrating {len(success_names)} enabled tool(s).",
        )

    def _train_inspection_item(self, inspection_item: InspectionItem) -> TrainResult:
        if not inspection_item.enabled:
            raise RuntimeError("selected tool is disabled")

        algorithm = self._resolve_training_algorithm(inspection_item)
        if not algorithm:
            if self.algo.is_learning_tool(inspection_item.algorithm_code):
                raise RuntimeError("please choose a learning tool subtype first")
            raise RuntimeError("please select an inspection tool")

        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        training_ok_files, training_ng_files, candidate_paths = self._training_sample_groups_for_role(
            inspection_item.camera_id,
            roi_label=roi_label,
        )
        if not candidate_paths:
            camera_id = _normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing training images for {camera_id}")
        missing_groups: List[str] = []
        if not training_ok_files:
            missing_groups.append("OK")
        if not training_ng_files:
            missing_groups.append("NG")
        if missing_groups:
            camera_id = _normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing {'/'.join(missing_groups)} images for {camera_id}")
        missing_paths = self._missing_training_roi_paths(
            roi_label,
            candidate_paths,
        )
        if missing_paths:
            missing = [os.path.basename(path) for path in missing_paths[:50]]
            raise RuntimeError(
                f"missing ROI label '{roi_label}' in some training sample jsons:\n" + "\n".join(missing)
            )

        self.algo.product_params.algorithm = algorithm
        self.algo.product_params.score_mode = self.cmb_mode.currentText()
        self.algo.product_params.margin = float(self.spin_margin.value())
        self.algo.product_params.topk = int(self.spin_topk.value())
        return self.algo.train(
            training_ok_files,
            training_ng_files,
            algorithm=algorithm,
            product_dir=self.session.product_dir,
            label_names=[roi_label],
            model_key=inspection_item.model_key,
        )

    def _train_all_tools(self) -> None:
        self.algo.model = None
        self.table.setRowCount(0)
        self._current_result_rows = []
        if getattr(self, "_training_in_progress", False):
            QtWidgets.QMessageBox.information(self, tr("common.info"), "Training is already running.")
            return
        if self._warn_mixed_training_camera_samples(self.current_camera_role()):
            return
        if not self._ensure_training_roi_reviewed(
            self.current_camera_role(),
            action_name=tr("debug.train_all_tools"),
            action_key="all",
        ):
            return

        current_role = self.current_camera_role()
        enabled_items = [
            item
            for item in self.inspection_items
            if item.enabled and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
        ]
        if not enabled_items:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.enable_one_tool", role=current_role))
            return

        selected_item = self._selected_inspection_item()
        selected_item_id = str(selected_item.item_id or "") if selected_item is not None else ""
        tasks: List[dict] = []
        failure_messages: List[str] = []

        for inspection_item in enabled_items:
            display_name = self._training_item_display_name(inspection_item)
            try:
                tasks.append(self._build_training_task_for_item(inspection_item))
            except Exception as exc:
                failure_messages.append(f"{display_name}: {exc}")

        if not tasks:
            if failure_messages:
                self.lbl_status.setText("Status: train failed")
                QtWidgets.QMessageBox.warning(
                    self,
                    tr("debug.train_result_title"),
                    "\n".join([f"Failed: {len(failure_messages)} tool(s)", *failure_messages[:20]]),
                )
            else:
                QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.enable_one_tool", role=current_role))
            return

        self._start_training_worker(
            self._training_payload(
                "all",
                tasks,
                selected_item_id=selected_item_id,
                failures=failure_messages,
            )
        )

    def _train(self) -> None:
        self.algo.model = None
        self.table.setRowCount(0)
        self._current_result_rows = []
        if getattr(self, "_training_in_progress", False):
            QtWidgets.QMessageBox.information(self, tr("common.info"), "Training is already running.")
            return
        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_inspection_tool_in_table"))
            return
        if self._warn_mixed_training_camera_samples(inspection_item.camera_id):
            return
        if not self._ensure_training_roi_reviewed(
            inspection_item.camera_id,
            action_name=tr("debug.calibrate_current_tool"),
            action_key="current",
        ):
            return
        if not inspection_item.enabled:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.tool_disabled"))
            return
        try:
            task = self._build_training_task_for_item(inspection_item)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("debug.train_failed_title"), str(exc))
            return

        selected_item_id = str(inspection_item.item_id or "")
        self._start_training_worker(
            self._training_payload(
                "current",
                [task],
                selected_item_id=selected_item_id,
            )
        )

    # ------------------------------------------------------------------
    # 棰勬祴 / 娴嬭瘯
    # ------------------------------------------------------------------

    def _test_target_inspection_items(self) -> List[InspectionItem]:
        current_role = self.current_camera_role()
        return [
            item
            for item in self.inspection_items
            if item.enabled and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
        ]

    def _execute_test_image(self, path: str) -> Dict[str, object]:
        from application import ToolPageRuntimeContext

        p = str(path or "").strip()
        if p is None or not os.path.exists(p):
            raise FileNotFoundError(p or tr("debug.open_test_image_first"))
        self.canvas.set_overlays([])

        target_items = self._test_target_inspection_items()
        camera_id = (
            str(target_items[0].camera_id or "").strip()
            if target_items
            else self.current_camera_role()
        ) or "cam1"

        executor = InspectionExecutor(ToolPageRuntimeContext(self))
        try:
            response = executor.execute(
                InspectionExecutionRequest(
                    camera_id=camera_id,
                    image_path=p,
                    items=target_items,
                )
            )
        except Exception as ex:
            raise RuntimeError(str(ex)) from ex

        rows: List[Dict[str, object]] = []
        log_names: List[str] = []
        raw_rows = []
        match_ms = float(response.match_ms or 0.0)
        infer_ms = float(response.infer_ms or 0.0)
        total_ms = float(response.total_ms or 0.0)
        if total_ms <= 0.0 and (match_ms > 0.0 or infer_ms > 0.0):
            total_ms = match_ms + infer_ms
        if isinstance(response.raw_row, dict):
            if isinstance(response.raw_row.get("item_rows"), list):
                raw_rows = [dict(row) for row in response.raw_row.get("item_rows", [])]
            elif response.raw_row:
                raw_rows = [dict(response.raw_row)]

        if target_items:
            for index, item_result in enumerate(response.item_results):
                row = dict(raw_rows[index]) if index < len(raw_rows) else {}
                display_name = str(item_result.display_name or item_result.item_id or "tool").strip()
                roi_label = str(item_result.roi_label or "").strip()
                algorithm = (
                    self.algo.current_learning_backbone()
                    if self.algo.is_learning_tool(item_result.algorithm_code)
                    else self.algo.resolve_tool_algorithm(item_result.algorithm_code)
                )
                if row.get("pred") is not None and str(row.get("pred", "")).strip() != str(item_result.result or "").strip():
                    row.setdefault("raw_pred", row.get("pred"))
                row["pred"] = item_result.result
                row["match_ms"] = match_ms if match_ms > 0.0 else row.get("match_ms")
                row["total_ms"] = total_ms if total_ms > 0.0 else row.get("total_ms")
                row["tool_name"] = display_name
                row["camera_id"] = item_result.camera_id
                row["roi_label"] = roi_label
                row["algorithm"] = algorithm
                row["file_name"] = f"{os.path.basename(p)} [{display_name}]"
                if roi_label:
                    self._record_roi_result(p, roi_label, item_result.result)
                rows.append(row)
                log_names.append(os.path.basename(self._append_test_log(row)))
        else:
            row = dict(raw_rows[0]) if raw_rows else {}
            row.setdefault("pred", response.result)
            row["match_ms"] = match_ms if match_ms > 0.0 else row.get("match_ms")
            row["total_ms"] = total_ms if total_ms > 0.0 else row.get("total_ms")
            labels_override = (
                self._shape_output_labels()
                if self.loc_method == "shape"
                else ["roi"]
            )
            for roi_label in labels_override:
                self._record_roi_result(p, roi_label, response.result)
            rows.append(row)
            log_names.append(os.path.basename(self._append_test_log(row)))

        self._populate_results_table(rows)

        overall_pred = str(response.result or "NG")
        ng_names = [
            str(row.get("tool_name", row.get("roi_label", "")) or "").strip()
            for row in rows
            if str(row.get("pred", "NG") or "NG").strip().upper() == "NG"
        ]
        status_text = (
            f"Status: TEST={os.path.basename(p)}  overall={overall_pred}"
            f"  tools={len(rows)}"
        )
        if ng_names:
            status_text += "  NG=" + ", ".join(ng_names[:5])
        if match_ms > 0.0:
            status_text += f"  match={match_ms:.1f}ms"
        status_text += f"  infer={infer_ms:.1f}ms"
        if log_names:
            status_text += f"  log={log_names[-1]}"
        self.lbl_status.setText(status_text)
        self._load_canvas_image(p)
        self._update_sample_panel_widgets()
        return {
            "result": overall_pred,
            "rows": rows,
            "log_names": log_names,
            "status_text": status_text,
        }

    def _run_test(self) -> None:
        p = self.canvas.image_path()
        if p is None or not os.path.exists(p):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.open_test_image_first"))
            return
        try:
            self._execute_test_image(p)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, tr("debug.test_failed_title"), str(ex))

    def _run_all_test_samples(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        paths = list(self._sample_paths_for_kind("test", current_role))
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.test_all_test_samples_empty"))
            return

        total = len(paths)
        ok_count = 0
        ng_count = 0
        failed_count = 0
        failures: List[str] = []
        buttons = [
            getattr(self, "btn_clear_session", None),
            getattr(self, "btn_test", None),
            getattr(self, "btn_export_test", None),
        ]
        old_enabled = {
            button: bool(button.isEnabled())
            for button in buttons
            if button is not None
        }
        for button in old_enabled:
            button.setEnabled(False)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for index, path in enumerate(paths, start=1):
                self.lbl_status.setText(
                    f"Status: batch test {index}/{total}  {os.path.basename(path)}"
                )
                QtWidgets.QApplication.processEvents()
                try:
                    result = self._execute_test_image(path)
                except Exception as ex:
                    failed_count += 1
                    if len(failures) < 5:
                        failures.append(f"{os.path.basename(path)}: {ex}")
                    continue
                if str(result.get("result", "") or "").strip().upper() == "OK":
                    ok_count += 1
                else:
                    ng_count += 1
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            for button, enabled in old_enabled.items():
                button.setEnabled(enabled)

        message = tr(
            "debug.test_all_test_samples_done_message",
            total=total,
            ok=ok_count,
            ng=ng_count,
            failed=failed_count,
        )
        if failures:
            message += "\n\n" + "\n".join(failures)
        QtWidgets.QMessageBox.information(
            self,
            tr("debug.test_all_test_samples_done_title"),
            message,
        )
        self.lbl_status.setText(
            f"Status: batch test done total={total} OK={ok_count} NG={ng_count} failed={failed_count}"
        )


    def _show_tool_dialog(
        self,
        key: str,
        widget: QtWidgets.QWidget,
        *,
        size: Tuple[int, int],
    ) -> None:
        dialog = self._tool_dialogs.get(key)
        if dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setModal(False)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addWidget(widget)
            dialog.resize(*size)
            if key == "camera_debug":
                dialog.finished.connect(lambda *_: self._stop_debug_camera_preview())
            self._tool_dialogs[key] = dialog
        dialog.setWindowTitle(self._tool_dialog_title(key))
        if key == "camera_debug":
            # Prevent Enter in parameter editors from triggering dialog buttons.
            for _btn in dialog.findChildren(QtWidgets.QPushButton):
                _btn.setAutoDefault(False)
                _btn.setDefault(False)
        widget.show()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


    def _export_current_results_csv(self) -> None:
        if not self._current_result_rows:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.no_export_results"))
            return
        json_path, csv_path = self._save_test_result_report(
            self._current_result_rows,
            report_prefix="test_result",
        )
        QtWidgets.QMessageBox.information(
            self,
            tr("debug.export_done_title"),
            tr("debug.export_done_message", json_path=json_path, csv_path=csv_path),
        )

    def _on_table_click(self, row: int, _col: int) -> None:
        it = self.table.item(row, 0)
        if it is None:
            return
        p = it.data(QtCore.Qt.UserRole)
        if isinstance(p, str) and os.path.exists(p):
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)
            self._update_sample_panel_widgets()

    # ------------------------------------------------------------------
    # 分析 / 验证
    # ------------------------------------------------------------------


    def _save_margin_report(
        self, rows: List[Dict[str, object]], summary: Dict[str, object]
    ) -> Tuple[str, str]:
        report_dir = os.path.join(self.session.product_dir, "margin_reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm_code = learning_backbone_storage_code(self.current_algorithm())
        base = f"margin_report_{algorithm_code}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {
            "product": self.session.current_product,
            "algorithm": algorithm_code,
            "score_mode": self.cmb_mode.currentText(),
            "topk": int(self.spin_topk.value()),
            "margin": float(self.spin_margin.value()),
            "loc_method": self.loc_method,
            "summary": summary,
            "rows": rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self._write_test_rows_csv(csv_path, rows)
        return json_path, csv_path

    def _run_margin_validation(self) -> None:
        inspection_item = self._selected_inspection_item()
        if inspection_item is not None and inspection_item.enabled:
            algorithm = (
                self.algo.current_learning_backbone()
                if self.algo.is_learning_tool(inspection_item.algorithm_code)
                else self.algo.resolve_tool_algorithm(inspection_item.algorithm_code)
            )
            labels_override = [str(inspection_item.roi_label or "").strip() or "roi"]
            algorithm_override = inspection_item.algorithm_code
            model_key_override = inspection_item.model_key
            validation_ok_files, validation_ng_files, _candidate_paths = self._training_sample_groups_for_role(
                inspection_item.camera_id,
                roi_label=labels_override[0],
            )
        else:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_inspection_tool_first"))
            return
        if not self._is_embedding_algorithm(algorithm):
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.margin_traditional_unsupported"))
            return
        if not self.algo._loaded_embedding_matches(
            algorithm,
            labels=labels_override,
            model_key=model_key_override or "",
        ):
            try:
                self.load_embedding_model(algorithm, model_key=model_key_override)
            except Exception:
                pass
        if self.algo.model is None:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.train_register_first"))
            return
        if not validation_ok_files or not validation_ng_files:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.margin_need_ok_ng"))
            return

        feat_net = self.algo.get_feat_net(
            self.algo.model.backbone,
            getattr(self.algo.model, "device", None),
        )
        rows: List[Dict[str, object]] = []
        try:
            for path in validation_ok_files:
                row = self._predict_image(
                    path,
                    feat_net=feat_net,
                    prefer_canvas_roi=False,
                    labels_override=labels_override,
                    algorithm_override=algorithm_override,
                    model_key_override=model_key_override,
                )
                row["gt"] = "OK"
                rows.append(row)
            for path in validation_ng_files:
                row = self._predict_image(
                    path,
                    feat_net=feat_net,
                    prefer_canvas_roi=False,
                    labels_override=labels_override,
                    algorithm_override=algorithm_override,
                    model_key_override=model_key_override,
                )
                row["gt"] = "NG"
                rows.append(row)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, tr("debug.margin_failed_title"), str(ex))
            return

        self._populate_results_table(rows)
        summary = self._suggest_margin_from_rows(rows)
        json_path, csv_path = self._save_margin_report(rows, summary)

        safe_range = summary.get("safe_range")
        safe_text = ""
        if isinstance(safe_range, tuple):
            safe_text = "\n" + tr("debug.safe_range", low=safe_range[0], high=safe_range[1])

        self.lbl_status.setText(
            "Status: "
            + f"current margin={summary['current_margin']:.4f} acc={summary['current_accuracy']:.4f}  "
            + f"suggested margin={summary['suggested_margin']:.4f} acc={summary['suggested_accuracy']:.4f}"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Margin suggestion",
            f"Current margin: {summary['current_margin']:.4f}\n"
            f"Current accuracy: {summary['current_accuracy']:.4f}\n"
            f"Suggested margin: {summary['suggested_margin']:.4f}\n"
            f"Suggested accuracy: {summary['suggested_accuracy']:.4f}"
            + safe_text
            + f"\n\nSaved reports:\n{json_path}\n{csv_path}",
        )

    def _open_embedding_analysis_dialog(self) -> None:
        try:
            from ui.debug import EmbeddingAnalysisDialog
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                tr("debug.embedding_analysis_failed_title"),
                tr("debug.embedding_analysis_import_failed", detail=f"{exc}\n\n{traceback.format_exc()}"),
            )
            return

        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_learning_tool_first"))
            return
        if not inspection_item.enabled:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.tool_disabled"))
            return
        if not self.algo.is_learning_tool(inspection_item.algorithm_code):
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.not_learning_tool"))
            return
        current_role = self.current_camera_role()
        allowed_learning_items = [
            item
            for item in list(getattr(self, "inspection_items", []) or [])
            if bool(getattr(item, "enabled", True))
            and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
            and self.algo.is_learning_tool(getattr(item, "algorithm_code", ""))
        ]
        allowed_model_keys = list(
            dict.fromkeys(
                str(getattr(item, "model_key", "") or "").strip()
                for item in allowed_learning_items
                if str(getattr(item, "model_key", "") or "").strip()
            )
        )
        allowed_backbones = []
        current_backbone = str(self.algo.current_learning_backbone() or "").strip()
        if current_backbone:
            allowed_backbones.append(current_backbone)
        try:
            dialog = EmbeddingAnalysisDialog(
                session_root=self.session.session_dir,
                initial_product=self.session.current_product,
                initial_backbone=current_backbone,
                initial_model_key=inspection_item.model_key,
                allowed_model_keys=allowed_model_keys,
                allowed_backbones=allowed_backbones,
                parent=self,
            )
            dialog.exec()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                tr("debug.embedding_analysis_failed_title"),
                tr("debug.embedding_analysis_init_failed", detail=f"{exc}\n\n{traceback.format_exc()}"),
            )

    # ------------------------------------------------------------------
    # 传统基线调试
    # ------------------------------------------------------------------


    def _run_traditional_baseline_debug(self) -> None:
        paths, tab_name = self._current_tab_paths_and_name()
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.current_list_empty"))
            return

        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_inspection_tool_first"))
            return
        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        display_name = str(
            inspection_item.display_name or inspection_item.roi_label or inspection_item.item_id or roi_label
        ).strip()

        rows: List[Dict[str, object]] = []
        ok = 0
        for path in paths:
            try:
                row = self._compute_traditional_baseline_metrics(path, preferred_label=roi_label)
                ok += 1
            except Exception as exc:
                row = {
                    "file_path": path, "file_name": os.path.basename(path),
                    "roi_label": "", "bbox_xywh": "", "mean_intensity": "", "mean_std": "",
                    "hsv_h_mean": "", "hsv_h_std": "", "hsv_s_mean": "", "hsv_s_std": "",
                    "hsv_v_mean": "", "hsv_v_std": "", "roi_area": "", "error": str(exc),
                }
            rows.append(row)

        json_path, csv_path = self._save_traditional_baseline_report(rows, tab_name=tab_name, roi_label=roi_label)
        self.lbl_status.setText(f"Status: baseline debug done for {display_name}, success {ok}/{len(paths)}")
        QtWidgets.QMessageBox.information(
            self,
            "Traditional Baseline Debug",
            f"Completed baseline metrics for {display_name} ({roi_label}) in {tab_name}.\n"
            f"Success: {ok}/{len(paths)}\n\nJSON:\n{json_path}\n\nCSV:\n{csv_path}",
        )





