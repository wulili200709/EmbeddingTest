"""
tool_page_pyside6.py

??? QWidget??? ROI ???????????????????? UI ??????

????
  1. ROI ??: _load_canvas_image / _save_current_rect / _on_select_ok
  2. ?? ROI / ??: _autogen_roi_for_images / _open_line2dup_template_page
  3. ?? / ??: _predict_image / _run_test / _populate_results_table / _append_test_log
  4. ?? / ??: _suggest_margin_from_rows / _run_margin_validation / _run_traditional_baseline_debug

?? Signal ? MainWindow ????????
  productChangeRequested(str): ????????? MainWindow ??????????? apply_product_switch()
  sessionClearRequested(): ????????????? MainWindow ??????????? reset_for_clear()
  sessionLoaded(): ???????????? MainWindow ???????

????
  - ????????????????
  - RuntimeModePage ???
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import algorithms.proxy as qr_core

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
    ToolPageRuntimeContext,
    TrainResult,
)
from domain import (
    InspectionItem,
    clearable_roi_labels,
    inspection_item_specs_from_line2dup_recipe,
    load_inspection_items,
    save_inspection_items,
    sync_items_with_labels,
    output_labels_from_line2dup_recipe,
)
from line2dup.core import locator as line2dup_locator
from line2dup.core.recipe import Line2DupRecipe
from ui.debug import (
    OverlayShape,
    RoiCanvas,
)
from ui.roi_overlay_colors import is_roi_label
from ui.runtime import RuntimeImageView


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


SUPPORTED_LOC_MODES = ["line2dup"]
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

ALGORITHM_GROUPS = [
    (
        "学习工具",
        [
            ("高精度学习工具", "efficientnet_b0", True),
            ("轻量学习工具", "mobilenet_v3_small", True),
            ("均衡学习工具", "mobilenet_v3_large", True),
        ],
    ),
    (
        "传统工具",
        [
            ("色相工具", "meanhsv_h", True),
            ("灰度工具", "meanintensity", True),
            ("偏差工具", "meanstd", True),
            ("明度工具", "meanhsv_v", True),
            ("饱和度工具", "meanhsv_s", True),
        ],
    ),
    (
        "测量工具",
        [
            ("找圆", "find_circle", False),
            ("找直线", "find_line", False),
        ],
    ),
]

ALGORITHM_DISPLAY_NAMES = {
    code: label
    for _group_name, items in ALGORITHM_GROUPS
    for label, code, enabled in items
    if enabled
}


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
                    self.errorOccurred.emit(f"预览转换失败: {exc}")
                    self.msleep(120)
                continue

            self.frameReady.emit(image)
            self.msleep(30)


# ---------------------------------------------------------------------------
# ToolPage
# ---------------------------------------------------------------------------

class ToolPage(QtWidgets.QWidget):
    """
    ?????? QWidget?

    ?????MainWindow ??::

        self.tool_page = ToolPage(self.session, self.algo)
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(lambda: self._refresh_runtime_status_ui("????????"))
        self.main_pages.addTab(self.tool_page, "???")
        self.tool_page.load_session()
    """

    productChangeRequested = QtCore.Signal(str)   # new product name
    sessionClearRequested = QtCore.Signal()
    sessionLoaded = QtCore.Signal()
    inspectionItemsChanged = QtCore.Signal()
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

        self.ok_files: List[str] = []
        self.ng_files: List[str] = []
        self.test_files: List[str] = []
        self.ref_image: Optional[str] = None
        self.loc_method: str = "line2dup"
        self.line2dup_recipe: Optional[Line2DupRecipe] = None
        self._line2dup_recipes_by_role: Dict[str, Optional[Line2DupRecipe]] = {}
        self._training_roi_ready_signatures: Dict[str, str] = {}
        self._training_roi_pending_actions: Dict[str, str] = {}
        self.inspection_items: List[InspectionItem] = []
        self._visible_inspection_item_indexes: List[int] = []
        self._inspection_items_table_loading = False
        self._line2dup_match_ms_by_image: Dict[str, float] = {}
        self._line2dup_autogen_ms_by_image: Dict[str, float] = {}
        self._current_result_rows: List[Dict[str, object]] = []
        self._roi_results_by_image: Dict[str, Dict[str, str]] = {}
        self._updating_runtime_params = False
        self._skip_empty_autogen_message = False
        self._tool_dialogs: Dict[str, QtWidgets.QDialog] = {}
        self._template_editor_dialog: Optional[QtWidgets.QDialog] = None
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
        # ?? setValue/????????????????????
        self._debug_camera_block_spin_apply = False
        self._main_right_panel: Optional[QtWidgets.QFrame] = None
        self._algorithm_picker_style_default = ""
        self._algorithm_picker_style_compact = ""

        self._build_ui()
        self._set_current_camera_role(self._current_camera_role)
        QtCore.QTimer.singleShot(0, self._update_responsive_layout)
        self.destroyed.connect(lambda *_: self._cleanup_debug_hardware())

    # ------------------------------------------------------------------
    # 鍏紑鎺ュ彛锛圡ainWindow 璋冪敤锛?
    # ------------------------------------------------------------------

    def current_algorithm(self) -> str:
        value = self.cmb_algorithm.currentData() if hasattr(self, "cmb_algorithm") else None
        if value is None:
            return ""
        return str(value).strip()

    def current_camera_role(self) -> str:
        combo = getattr(self, "cmb_current_camera_role", None)
        if combo is None:
            return _normalize_camera_role(getattr(self, "_current_camera_role", "cam1")) or "cam1"
        return _normalize_camera_role(combo.currentData() or combo.currentText() or self._current_camera_role) or "cam1"

    def line2dup_paths_for_role(self, camera_role: object = None):
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return self.session.line2dup_paths_for_role(role)

    def load_line2dup_recipe_for_role(
        self,
        camera_role: object = None,
        *,
        force_reload: bool = False,
    ) -> Optional[Line2DupRecipe]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        if not force_reload and role in self._line2dup_recipes_by_role:
            recipe = self._line2dup_recipes_by_role.get(role)
        else:
            paths = self.line2dup_paths_for_role(role)
            if not (os.path.exists(paths.recipe_path) or os.path.exists(paths.legacy_recipe_path)):
                recipe = None
            else:
                try:
                    recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir, role)
                except Exception:
                    recipe = None
            self._line2dup_recipes_by_role[role] = recipe
        if role == self.current_camera_role():
            self.line2dup_recipe = recipe
        return recipe

    def line2dup_recipe_for_role(
        self,
        camera_role: object = None,
        *,
        force_reload: bool = False,
    ) -> Optional[Line2DupRecipe]:
        return self.load_line2dup_recipe_for_role(camera_role, force_reload=force_reload)

    def line2dup_model_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        paths = self.line2dup_paths_for_role(role)
        return line2dup_locator.resolved_model_path_for_product(self.session.product_dir, role)

    def line2dup_recipe_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return line2dup_locator.resolved_recipe_path_for_product(self.session.product_dir, role)

    def _apply_current_role_recipe_state(self) -> None:
        recipe = self.load_line2dup_recipe_for_role(self.current_camera_role())
        self.line2dup_recipe = recipe
        ref_image = ""
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = str(recipe.reference_image)
        self.ref_image = ref_image or None
        if self.ref_image:
            self.lbl_ref.setText(f"参考图: {os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        else:
            self.lbl_ref.setText("参考图：未设置")
            self.lbl_ref.setToolTip("")

    def _training_sample_groups_for_role(self, camera_role: object = None) -> tuple[List[str], List[str], List[str]]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        training_ok_files = _filter_paths_for_camera(self, self.ok_files, role)
        training_ng_files = _filter_paths_for_camera(self, self.ng_files, role)
        return training_ok_files, training_ng_files, list(training_ok_files) + list(training_ng_files)

    def _training_roi_ready_signature(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        _training_ok_files, _training_ng_files, candidate_paths = self._training_sample_groups_for_role(role)
        recipe_path = self.line2dup_recipe_path_for_role(role)
        model_path = self.line2dup_model_path_for_role(role)
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
        else:
            role = _normalize_camera_role(camera_role) or "cam1"
            self._training_roi_ready_signatures.pop(role, None)
            self._training_roi_pending_actions.pop(role, None)
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
        default_train_text = "训练 / 标定全部启用工具"
        default_current_text = "标定当前工具"
        default_train_style = getattr(self, "_train_action_btn_style", "")
        default_current_style = getattr(self, "_train_current_btn_style", "")
        confirm_style = getattr(self, "_train_confirm_btn_style", default_train_style)

        if pending_action == "all":
            train_button.setText("确认开始训练 / 标定全部启用工具")
            train_button.setStyleSheet(confirm_style)
            if cancel_train_button is not None:
                cancel_train_button.setVisible(True)
        else:
            train_button.setText(default_train_text)
            train_button.setStyleSheet(default_train_style)
            if cancel_train_button is not None:
                cancel_train_button.setVisible(False)

        if pending_action == "current":
            train_current_button.setText("确认开始标定当前工具")
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
        action_text = "训练 / 标定全部启用工具" if pending_action == "all" else "标定当前工具"
        self.lbl_status.setText(f"状态：已取消“{action_text}”确认")

    def _ensure_training_roi_reviewed(self, camera_role: object, *, action_name: str, action_key: str) -> bool:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        if self.loc_method != "line2dup":
            return True
        training_ok_files, training_ng_files, candidate_paths = self._training_sample_groups_for_role(role)
        if not training_ok_files or not training_ng_files:
            return True

        signature = self._training_roi_ready_signature(role)
        if (
            self._training_roi_ready_signatures.get(role) == signature
            and self._training_roi_pending_actions.get(role) == action_key
        ):
            self._training_roi_ready_signatures.pop(role, None)
            self._training_roi_pending_actions.pop(role, None)
            self._update_runtime_widgets()
            return True

        recipe = self.line2dup_recipe_for_role(role, force_reload=True)
        if recipe is None:
            QtWidgets.QMessageBox.warning(self, "提示", f"{role} 尚未加载 line2dup 配方，请先创建模板。")
            return False
        ref_image = self.ref_image
        if recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
        if not ref_image or not os.path.exists(ref_image):
            QtWidgets.QMessageBox.warning(self, "提示", f"{role} 缺少参考图，请先确认位置修正模板。")
            return False

        ok_count = 0
        errors: List[str] = []
        for path in candidate_paths:
            try:
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=role,
                )
                self._line2dup_match_ms_by_image[path] = float(run.total_ms)
                self._line2dup_autogen_ms_by_image[path] = float(run.total_ms)
                ok_count += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        self._refresh_current_image_after_roi_update(candidate_paths)

        if errors:
            self._clear_training_roi_review_state(role)
            QtWidgets.QMessageBox.warning(
                self,
                "ROI 生成失败",
                "训练前自动更新 ROI 失败，请先检查模板或图片。\n\n"
                + "\n".join(errors[:20]),
            )
            return False

        self._training_roi_ready_signatures[role] = signature
        self._training_roi_pending_actions[role] = action_key
        self._update_runtime_widgets()
        self.lbl_status.setText(f"状态：已更新 {role} 的 ROI，请检查后再次点击{action_name}")
        QtWidgets.QMessageBox.information(
            self,
            "ROI 已更新",
            f"已更新 {role} 的 OK/NG ROI，共 {ok_count} 张。\n请检查当前图片上的 ROI 后，再次点击“{action_name}”。",
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
            self._set_debug_camera_status(f"已切换到 {self.current_camera_role()}，请重新连接")

    def current_algorithm_display_name(self) -> str:
        algorithm = self.current_algorithm()
        if not algorithm:
            return ""
        return ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)

    def _populate_algorithm_combo(self) -> None:
        self.cmb_algorithm.clear()
        model = self.cmb_algorithm.model()
        for group_name, items in ALGORITHM_GROUPS:
            self.cmb_algorithm.addItem(group_name, None)
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
                self.cmb_algorithm.addItem(label, code if enabled else None)
                item_index = self.cmb_algorithm.count() - 1
                item = model.item(item_index) if hasattr(model, "item") else None
                if item is not None and not enabled:
                    item.setEnabled(False)
                    item.setForeground(QtGui.QColor("#707070"))
                    item.setToolTip("暂未实现")

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
        sec_tools = QtWidgets.QLabel("  检测工具")
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
            submenu = menu.addMenu(group_name)
            for label, code, enabled in items:
                action = submenu.addAction(label)
                if not enabled:
                    action.setEnabled(False)
                    action.setToolTip("暂未实现")
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
            text = self.current_algorithm_display_name() or "请选择工具"
            button.setText(text)
            button.setToolTip(text)
        actions = getattr(self, "_algorithm_actions", {})
        current_algorithm = self.current_algorithm()
        for code, action in actions.items():
            action.setChecked(code == current_algorithm)

    def open_camera_debug_dialog(self) -> None:
        self._show_tool_dialog(
            "camera_debug",
            "相机取图 / 参数工具",
            self.camera_debug_page,
            size=(1100, 700),
        )
        self._refresh_debug_role_status()
        self._refresh_debug_camera_list()

    def open_io_debug_dialog(self) -> None:
        self._show_tool_dialog(
            "io_debug",
            "DI / DO 调试工具",
            self.io_debug_page,
            size=(900, 480),
        )
        self._apply_runtime_io_debug_state()

    def open_template_editor_dialog(self) -> None:
        self._open_line2dup_template_page()

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
            self.lbl_debug_di_snapshot.setText("DI：未连接")
            self.lbl_debug_do_snapshot.setText("DO：未连接")
            self.lbl_debug_io_mapping_summary.setText("映射：未加载")

        self.btn_debug_open_io.setEnabled(True)
        self.btn_debug_close_io.setEnabled(self._debug_io_controller is not None and not self._debug_io_uses_runtime_controller)
        self.btn_debug_refresh_io.setEnabled(self._debug_io_controller is not None)
        self.lbl_debug_io_mapping_summary.setToolTip(self._runtime_io_status_detail)

    def open_template_match_dialog(self) -> None:
        self._show_tool_dialog(
            "template_match",
            "自动生成 ROI 工具",
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
        status_text: str = "未检测",
    ) -> List[Dict[str, object]]:
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
                "status_text": status_text if item.enabled else "已禁用",
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
    ) -> Dict[str, object]:
        # Used by MainWindow runtime callback.
        return self._predict_image(
            path,
            feat_net=feat_net,
            labels_override=labels_override,
            algorithm_override=algorithm_override,
            model_key_override=model_key_override,
        )

    def load_session(self) -> None:
        # Load algorithm params and session data, then refresh UI.
        # Emits sessionLoaded after the session is applied.

        self._sync_camera_settings_store_path()

        self.algo.load_params(self.session.product_params_path)
        self.algo.model = None
        self._roi_results_by_image = {}
        self._apply_runtime_params_to_ui()

        sd = self.session.load_session()
        self.ok_files = sd.ok_files
        self.ng_files = sd.ng_files
        self.test_files = sd.test_files
        self.loc_method = sd.loc_method
        self._line2dup_recipes_by_role = {}
        self._clear_training_roi_review_state()
        self.cmb_loc.setCurrentText(self.loc_method)
        self._apply_current_role_recipe_state()
        if not self.ref_image and sd.ref_image and os.path.exists(sd.ref_image):
            self.ref_image = sd.ref_image
            self.lbl_ref.setText(f"参考图: {os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        self._refresh_lists()

        self._reload_inspection_items()
        self._sync_footer()

        self.sessionLoaded.emit()

    def apply_product_switch(self, name: str) -> None:
        # Switch product after runtime cameras are disconnected.
        # Update session paths, clear transient state, and reload the session.


        self.session.switch_product(name)
        self.session.save_products()
        self._sync_camera_settings_store_path()

        self.algo.model = None
        self.line2dup_recipe = None
        self._line2dup_recipes_by_role = {}
        self._clear_training_roi_review_state()
        self.ref_image = None
        self._line2dup_match_ms_by_image = {}
        self._line2dup_autogen_ms_by_image = {}
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self._current_result_rows = []
        self._roi_results_by_image = {}
        self.inspection_items = []

        self.table.setRowCount(0)
        self.canvas.clear_image()
        self.lbl_ref.setText("参考图：未设置")
        self.lbl_ref.setToolTip("")
        self.lbl_status.setText("状态：已切换产品")

        self.load_session()
        self._refresh_lists()

    def reset_for_clear(self) -> None:
        # Clear current debug session after runtime cameras are disconnected.
        # Reset image lists, cached models, and related UI state.


        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self.algo.model = None
        self.line2dup_recipe = None
        self._line2dup_recipes_by_role = {}
        self._clear_training_roi_review_state()
        self.ref_image = None
        self._line2dup_match_ms_by_image = {}
        self._line2dup_autogen_ms_by_image = {}
        self.lbl_ref.setText("参考图：未设置")
        self.lbl_status.setText("状态：未训练")
        self.table.setRowCount(0)
        self._current_result_rows = []
        self._roi_results_by_image = {}
        self._refresh_lists()
        self.session.delete_session_file()
        self._reload_inspection_items()
        self._sync_footer()

    # ------------------------------------------------------------------
    # UI 构建
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

        # 鈹€鈹€ 椤舵爮 鈹€鈹€
        header = QtWidgets.QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(f"background:{_HEADER_BG};border-bottom:1px solid #505050;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 16, 0)
        header_layout.setSpacing(4)

        header_layout.addWidget(QtWidgets.QLabel("\u4ea7\u54c1:"))
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.setFixedWidth(180)
        self.cmb_product.addItems(self.session.product_names)
        self.cmb_product.setCurrentText(self.session.current_product)
        self.cmb_product.currentTextChanged.connect(self._on_product_changed)
        self.cmb_product.setStyleSheet(_input_style)
        header_layout.addWidget(self.cmb_product)

        self.btn_new_product = QtWidgets.QPushButton(_si(SP.SP_FileDialogNewFolder), "\u65b0\u5efa")
        self.btn_new_product.setFixedWidth(60)
        self.btn_new_product.setStyleSheet(_compact_btn)
        self.btn_new_product.clicked.connect(self._new_product)
        header_layout.addWidget(self.btn_new_product)

        header_layout.addSpacing(10)
        header_layout.addWidget(QtWidgets.QLabel("当前相机:"))
        self.cmb_current_camera_role = QtWidgets.QComboBox()
        self.cmb_current_camera_role.setFixedWidth(84)
        self.cmb_current_camera_role.addItem("cam1", "cam1")
        self.cmb_current_camera_role.addItem("cam2", "cam2")
        self.cmb_current_camera_role.setStyleSheet(_input_style)
        self.cmb_current_camera_role.currentTextChanged.connect(self._on_current_camera_role_changed)
        header_layout.addWidget(self.cmb_current_camera_role)

        header_layout.addStretch(1)

        self.lbl_status = QtWidgets.QLabel("\u72b6\u6001\uff1a\u672a\u8bad\u7ec3")
        self.lbl_status.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;")
        self.lbl_status.hide()
        header_layout.addWidget(self.lbl_status)

        root.addWidget(header)

        # ???Canvas + ????
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # 左侧：Canvas + 结果表格
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

        # 右侧面板
        right_panel = QtWidgets.QFrame()
        self._main_right_panel = right_panel
        right_panel.setStyleSheet(f"background:{_PANEL_BG};border-left:1px solid #505050;")
        right_vbox = QtWidgets.QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        # --- 图片列表 ---
        self.lbl_images_section = QtWidgets.QLabel("  \u56fe\u7247\u5217\u8868")
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

        self.ok_list = QtWidgets.QListWidget()
        _lw_css = (
            f"QListWidget{{background:#333333;color:{_TEXT_LIGHT};border:none;font-size:12px;outline:0;}}"
            "QListWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
            "QListWidget::item:hover:!selected{background:#4a4a4a;}"
        )
        self.ok_list.setStyleSheet(_lw_css)
        self.ok_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ok_list.itemSelectionChanged.connect(self._on_select_ok)
        ok_tab = QtWidgets.QWidget()
        ok_l = QtWidgets.QVBoxLayout(ok_tab)
        ok_l.setContentsMargins(4, 4, 4, 4)
        ok_l.setSpacing(4)
        ok_l.addWidget(self.ok_list, 1)
        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(4)
        self.btn_add_ok = QtWidgets.QPushButton(_si(SP.SP_FileDialogStart), "\u6dfb\u52a0")
        self.btn_add_ok.setStyleSheet(_compact_btn)
        self.btn_add_ok.clicked.connect(lambda: self._add_images_to("OK"))
        self.btn_del_ok = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), "\u79fb\u9664")
        self.btn_del_ok.setStyleSheet(_compact_btn)
        self.btn_del_ok.clicked.connect(lambda: self._remove_selected_from("OK"))
        btns.addWidget(self.btn_add_ok)
        btns.addWidget(self.btn_del_ok)
        ok_l.addLayout(btns)
        self.tabs.addTab(ok_tab, "OK")

        self.ng_list = QtWidgets.QListWidget()
        self.ng_list.setStyleSheet(_lw_css)
        self.ng_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ng_list.itemSelectionChanged.connect(self._on_select_ng)
        ng_tab = QtWidgets.QWidget()
        ng_l = QtWidgets.QVBoxLayout(ng_tab)
        ng_l.setContentsMargins(4, 4, 4, 4)
        ng_l.setSpacing(4)
        ng_l.addWidget(self.ng_list, 1)
        btns2 = QtWidgets.QHBoxLayout()
        btns2.setSpacing(4)
        self.btn_add_ng = QtWidgets.QPushButton(_si(SP.SP_FileDialogStart), "\u6dfb\u52a0")
        self.btn_add_ng.setStyleSheet(_compact_btn)
        self.btn_add_ng.clicked.connect(lambda: self._add_images_to("NG"))
        self.btn_del_ng = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), "\u79fb\u9664")
        self.btn_del_ng.setStyleSheet(_compact_btn)
        self.btn_del_ng.clicked.connect(lambda: self._remove_selected_from("NG"))
        btns2.addWidget(self.btn_add_ng)
        btns2.addWidget(self.btn_del_ng)
        ng_l.addLayout(btns2)
        self.tabs.addTab(ng_tab, "NG")

        self.test_list = QtWidgets.QListWidget()
        self.test_list.setStyleSheet(_lw_css)
        self.test_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.test_list.itemSelectionChanged.connect(self._on_select_test)
        test_tab = QtWidgets.QWidget()
        t_l = QtWidgets.QVBoxLayout(test_tab)
        t_l.setContentsMargins(4, 4, 4, 4)
        t_l.setSpacing(4)
        t_l.addWidget(self.test_list, 1)
        btns3 = QtWidgets.QHBoxLayout()
        btns3.setSpacing(4)
        self.btn_add_test = QtWidgets.QPushButton(_si(SP.SP_FileDialogStart), "\u6dfb\u52a0")
        self.btn_add_test.setStyleSheet(_compact_btn)
        self.btn_add_test.clicked.connect(lambda: self._add_images_to("TEST"))
        self.btn_del_test = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), "\u79fb\u9664")
        self.btn_del_test.setStyleSheet(_compact_btn)
        self.btn_del_test.clicked.connect(lambda: self._remove_selected_from("TEST"))
        btns3.addWidget(self.btn_add_test)
        btns3.addWidget(self.btn_del_test)
        t_l.addLayout(btns3)
        self.tabs.addTab(test_tab, "TEST")
        right_vbox.addWidget(self.tabs, 1)

        # --- 算法参数 ---
        self.btn_toggle_algo = QtWidgets.QToolButton()
        self.btn_toggle_algo.setText("  算法参数")
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
        lbl_a = QtWidgets.QLabel("工具"); lbl_a.setStyleSheet(_lbl_s)
        lbl_m = QtWidgets.QLabel("\u5224\u5b9a"); lbl_m.setStyleSheet(_lbl_s)
        lbl_mg = QtWidgets.QLabel("阈值"); lbl_mg.setStyleSheet(_lbl_s)
        self.lbl_topk = QtWidgets.QLabel("TopK")
        self.lbl_topk.setStyleSheet(self._algo_param_label_style)
        algo_form.addRow(lbl_a, self.btn_algorithm_picker)
        algo_form.addRow(lbl_m, self.cmb_mode)
        algo_form.addRow(lbl_mg, self.spin_margin)
        algo_form.addRow(self.lbl_topk, self.spin_topk)
        self.algorithm_params_frame = algo_frame
        right_vbox.addWidget(algo_frame)
        self._sync_algorithm_picker()

        tool_gap = QtWidgets.QWidget()
        tool_gap.setFixedHeight(14)
        right_vbox.addWidget(tool_gap)

        self.btn_toggle_tools = QtWidgets.QToolButton()
        self.btn_toggle_tools.setText("  检测工具")
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

        self.tool_config_frame = tool_frame
        self.inspection_items_table = QtWidgets.QTableWidget(0, 5)
        self.inspection_items_table.setHorizontalHeaderLabels(["启用", "名称", "相机", "算法", "状态"])
        self.inspection_items_table.verticalHeader().setVisible(False)
        self.inspection_items_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.inspection_items_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        header = self.inspection_items_table.horizontalHeader()
        header.setStretchLastSection(False)
        self.btn_toggle_tools.setText("  检测工具")
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
        self._update_learning_backbone_hint()

        # --- 操作按钮 ---
        sec_action = QtWidgets.QLabel("  \u64cd\u4f5c")
        sec_action.setFixedHeight(28)
        sec_action.setStyleSheet(_section_style)
        right_vbox.addWidget(sec_action)

        action_frame = QtWidgets.QWidget()
        action_vbox = QtWidgets.QVBoxLayout(action_frame)
        action_vbox.setContentsMargins(8, 6, 8, 6)
        action_vbox.setSpacing(4)

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

        self.btn_train = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), "\u8bad\u7ec3 / \u6807\u5b9a\u5168\u90e8\u542f\u7528\u5de5\u5177")
        self._train_action_btn_style = _action_btn
        self._train_current_btn_style = _compact_btn
        self._train_confirm_btn_style = _confirm_action_btn
        self.btn_train.setStyleSheet(self._train_action_btn_style)
        self.btn_train.clicked.connect(self._train_all_tools)
        train_row.addWidget(self.btn_train, 1)

        self.btn_train_cancel = QtWidgets.QPushButton("×")
        self.btn_train_cancel.setToolTip("取消训练确认")
        self.btn_train_cancel.setStyleSheet(_cancel_action_btn)
        self.btn_train_cancel.setVisible(False)
        self.btn_train_cancel.clicked.connect(lambda: self._cancel_training_pending_action("all"))
        train_row.addWidget(self.btn_train_cancel, 0)
        action_vbox.addLayout(train_row)

        train_current_row = QtWidgets.QHBoxLayout()
        train_current_row.setContentsMargins(0, 0, 0, 0)
        train_current_row.setSpacing(4)

        self.btn_train_current = QtWidgets.QPushButton("\u6807\u5b9a\u5f53\u524d\u5de5\u5177")
        self.btn_train_current.setStyleSheet(self._train_current_btn_style)
        self.btn_train_current.clicked.connect(self._train)
        train_current_row.addWidget(self.btn_train_current, 1)

        self.btn_train_current_cancel = QtWidgets.QPushButton("×")
        self.btn_train_current_cancel.setToolTip("取消当前工具确认")
        self.btn_train_current_cancel.setStyleSheet(_cancel_action_btn)
        self.btn_train_current_cancel.setVisible(False)
        self.btn_train_current_cancel.clicked.connect(lambda: self._cancel_training_pending_action("current"))
        train_current_row.addWidget(self.btn_train_current_cancel, 0)
        action_vbox.addLayout(train_current_row)

        act_row = QtWidgets.QHBoxLayout()
        act_row.setSpacing(4)
        self.btn_test = QtWidgets.QPushButton(_si(SP.SP_MediaPlay), "\u6d4b\u8bd5\u5f53\u524d\u56fe")
        self.btn_test.setStyleSheet(_compact_btn)
        self.btn_test.clicked.connect(self._run_test)
        self.btn_export_test = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), "\u5bfc\u51fa\u62a5\u8868")
        self.btn_export_test.setStyleSheet(_compact_btn)
        self.btn_export_test.clicked.connect(self._export_current_results_csv)
        self.btn_clear_session = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), "\u6e05\u7a7a\u4f1a\u8bdd")
        self.btn_clear_session.setStyleSheet(
            "QPushButton{background:#383838;color:#e06666;border:1px solid #555;"
            "padding:4px 8px;border-radius:3px;font-size:12px;}"
            "QPushButton:hover{background:#4a4a4a;}"
        )
        self.btn_clear_session.clicked.connect(self._clear_session)
        act_row.addWidget(self.btn_test)
        act_row.addWidget(self.btn_export_test)
        act_row.addWidget(self.btn_clear_session)
        action_vbox.addLayout(act_row)
        right_vbox.addWidget(action_frame)
        self._sync_training_action_buttons()

        body.addWidget(right_panel, 0)
        root.addLayout(body, 1)

        # 鈹€鈹€ 搴曟爮 鈹€鈹€
        footer = QtWidgets.QFrame()
        footer.setFixedHeight(28)
        footer.setStyleSheet(f"background:{_HEADER_BG};border-top:1px solid #505050;")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 0, 16, 0)
        footer_layout.setSpacing(20)

        self.lbl_footer_ref = QtWidgets.QLabel("\u53c2\u8003\u56fe: \u672a\u8bbe\u7f6e")
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

        # 鈹€鈹€ 闅愯棌/瀵硅瘽妗嗕笓鐢ㄦ帶浠讹紙淇濇寔鎺ュ彛鍏煎锛?鈹€鈹€

        # ROI 鏍囨敞宸ュ叿鏍忥紙闅愯棌锛?
        roi_bar_w = QtWidgets.QWidget(self)
        roi_bar = QtWidgets.QHBoxLayout(roi_bar_w)
        self._manual_roi_bar = roi_bar
        roi_bar.addWidget(QtWidgets.QLabel("\u5f62\u72b6\uff1a"))
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(SUPPORTED_SHAPES)
        self.cmb_shape.setCurrentText("rect")
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        roi_bar.addWidget(self.cmb_shape)
        roi_bar.addWidget(QtWidgets.QLabel("\u6807\u6ce8\uff1a"))
        self.cmb_label = QtWidgets.QComboBox()
        self.cmb_label.addItems(["roi", "anchor", "anchor_mask"])
        self.cmb_label.setCurrentText("roi")
        self.cmb_label.currentTextChanged.connect(self._on_label_changed)
        roi_bar.addWidget(self.cmb_label)
        self.btn_save = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), "\u4fdd\u5b58\u6807\u6ce8")
        self.btn_save.clicked.connect(self._save_current_rect)
        self.btn_clear = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), "\u6e05\u7a7a\u6807\u6ce8")
        self.btn_clear.clicked.connect(self._clear_current_rect)
        roi_bar.addWidget(self.btn_save)
        roi_bar.addWidget(self.btn_clear)
        roi_bar_w.hide()

        # 自动 ROI（对话框用）
        auto_box = QtWidgets.QGroupBox("\u81ea\u52a8 ROI")
        auto_l = QtWidgets.QGridLayout(auto_box)
        self._auto_roi_layout = auto_l
        auto_l.setHorizontalSpacing(10)
        auto_l.setVerticalSpacing(10)
        auto_l.setColumnStretch(0, 1)
        auto_l.setColumnStretch(1, 1)
        auto_l.setColumnStretch(2, 1)
        self.lbl_ref = QtWidgets.QLabel("\u53c2\u8003\u56fe\uff1a\u672a\u8bbe\u7f6e")
        self.btn_set_ref = QtWidgets.QPushButton(_si(SP.SP_ArrowRight), "\u8bbe\u4e3a\u53c2\u8003\u56fe(\u5f53\u524d)")
        self.btn_set_ref.clicked.connect(self._set_ref_from_current)
        self.btn_pick_ref = QtWidgets.QPushButton(_si(SP.SP_DirOpenIcon), "\u9009\u62e9\u53c2\u8003\u56fe\u2026")
        self.btn_pick_ref.clicked.connect(self._pick_ref_image)
        auto_l.addWidget(self.lbl_ref, 0, 0, 1, 3)
        auto_l.addWidget(self.btn_set_ref, 1, 0)
        auto_l.addWidget(self.btn_pick_ref, 1, 1)
        self.lbl_loc_method = QtWidgets.QLabel("\u5b9a\u4f4d\u65b9\u5f0f\uff1a")
        auto_l.addWidget(self.lbl_loc_method, 2, 0)
        self.cmb_loc = QtWidgets.QComboBox()
        self.cmb_loc.addItems(SUPPORTED_LOC_MODES)
        self.cmb_loc.setCurrentText(self.loc_method)
        self.cmb_loc.currentTextChanged.connect(self._on_loc_method_changed)
        auto_l.addWidget(self.cmb_loc, 2, 1)
        self.chk_only_missing = QtWidgets.QCheckBox("\u4ec5\u7f3a\u5931ROI")
        self.chk_only_missing.setChecked(True)
        auto_l.addWidget(self.chk_only_missing, 0, 0, 1, 3, QtCore.Qt.AlignmentFlag.AlignRight)
        self.btn_autogen = QtWidgets.QPushButton(_si(SP.SP_FileDialogListView), "\u6279\u91cf\u751f\u6210ROI(\u5f53\u524d\u5217\u8868)")
        self.btn_autogen.clicked.connect(self._autogen_roi_current_tab)
        self.btn_autogen_all = QtWidgets.QPushButton(_si(SP.SP_FileDialogListView), "\u6279\u91cf\u751f\u6210ROI(\u5168\u90e8\u5217\u8868)")
        self.btn_autogen_all.clicked.connect(self._autogen_roi_all)
        self.btn_clear_roi_batch = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), "\u6e05\u7a7aROI(\u5f53\u524d\u5217\u8868)")
        self.btn_clear_roi_batch.clicked.connect(self._clear_roi_current_tab)
        auto_l.addWidget(self.btn_autogen, 1, 0)
        auto_l.addWidget(self.btn_autogen_all, 1, 1)
        auto_l.addWidget(self.btn_clear_roi_batch, 1, 2)
        self.template_match_box = auto_box
        self._update_loc_ui()

        # ???????????MVS ??????
        self.camera_debug_page = QtWidgets.QWidget()
        self.camera_debug_page.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")
        cam_main = QtWidgets.QHBoxLayout(self.camera_debug_page)
        cam_main.setContentsMargins(0, 0, 0, 0)
        cam_main.setSpacing(0)

        # 鈹€鈹€ 宸︿晶锛氳澶囧垪琛?+ 璁惧淇℃伅 鈹€鈹€
        cam_left = QtWidgets.QFrame()
        cam_left.setFixedWidth(220)
        cam_left.setStyleSheet(f"QFrame{{background:{_PANEL_BG};border-right:1px solid #505050;}}")
        cam_left_vbox = QtWidgets.QVBoxLayout(cam_left)
        cam_left_vbox.setContentsMargins(0, 0, 0, 0)
        cam_left_vbox.setSpacing(0)

        cam_left_title = QtWidgets.QLabel("  设备列表")
        cam_left_title.setFixedHeight(28)
        cam_left_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        cam_left_vbox.addWidget(cam_left_title)

        role_row = QtWidgets.QWidget()
        role_layout = QtWidgets.QHBoxLayout(role_row)
        role_layout.setContentsMargins(8, 6, 8, 2)
        role_layout.setSpacing(6)
        lbl_debug_role = QtWidgets.QLabel("调试角色")
        lbl_debug_role.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        role_layout.addWidget(lbl_debug_role)
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

        self.btn_debug_refresh_camera = QtWidgets.QPushButton(_si(SP.SP_BrowserReload), " 扫描相机")
        self.btn_debug_refresh_camera.setStyleSheet(_compact_btn)
        self.btn_debug_refresh_camera.clicked.connect(self._refresh_debug_camera_list)
        cam_left_vbox.addWidget(self.btn_debug_refresh_camera)

        cam_left_vbox.addSpacing(8)
        cam_info_title = QtWidgets.QLabel("  设备信息")
        cam_info_title.setFixedHeight(28)
        cam_info_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;border-top:1px solid #505050;padding-left:8px;")
        cam_left_vbox.addWidget(cam_info_title)

        self.lbl_debug_camera_info = QtWidgets.QLabel("相机信息：")
        self.lbl_debug_camera_info.setWordWrap(True)
        self.lbl_debug_camera_info.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;padding:8px;")
        cam_left_vbox.addWidget(self.lbl_debug_camera_info)
        self.lbl_debug_current_role = QtWidgets.QLabel("当前调试角色：cam1")
        self.lbl_debug_current_role.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;padding:0 8px 8px 8px;")
        cam_left_vbox.addWidget(self.lbl_debug_current_role)
        cam_left_vbox.addStretch(1)
        cam_main.addWidget(cam_left)

        # ?????? + ???? + ???
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

        self.btn_debug_connect_camera = QtWidgets.QPushButton(_si(SP.SP_DriveNetIcon), " 连接")
        self.btn_debug_connect_camera.setStyleSheet(_compact_btn)
        self.btn_debug_connect_camera.clicked.connect(self._connect_debug_camera)
        cam_tb_layout.addWidget(self.btn_debug_connect_camera)

        self.btn_debug_disconnect_camera = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), " 断开")
        self.btn_debug_disconnect_camera.setStyleSheet(_compact_btn)
        self.btn_debug_disconnect_camera.clicked.connect(self._disconnect_debug_camera)
        cam_tb_layout.addWidget(self.btn_debug_disconnect_camera)

        cam_tb_layout.addSpacing(12)

        self.btn_debug_live_preview = QtWidgets.QPushButton(_si(SP.SP_MediaPlay), " 实时预览")
        self.btn_debug_live_preview.setCheckable(True)
        self.btn_debug_live_preview.setStyleSheet(_compact_btn)
        self.btn_debug_live_preview.toggled.connect(self._toggle_debug_camera_preview)
        cam_tb_layout.addWidget(self.btn_debug_live_preview)

        self.btn_debug_grab_once = QtWidgets.QPushButton(_si(SP.SP_DesktopIcon), " 拍照到 TEST")
        self.btn_debug_grab_once.setStyleSheet(_compact_btn)
        self.btn_debug_grab_once.clicked.connect(self._grab_debug_camera_once)
        cam_tb_layout.addWidget(self.btn_debug_grab_once)

        cam_tb_layout.addSpacing(12)

        self.btn_debug_save_image = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), " 保存图片")
        self.btn_debug_save_image.setStyleSheet(_compact_btn)
        self.btn_debug_save_image.clicked.connect(self._save_debug_camera_image)
        cam_tb_layout.addWidget(self.btn_debug_save_image)

        cam_tb_layout.addStretch(1)
        cam_center_vbox.addWidget(cam_toolbar)

        self.view_debug_camera = RuntimeImageView("调试预览")
        self.view_debug_camera.setMinimumSize(640, 400)
        self.view_debug_camera.set_runtime_pixmap(None, placeholder="预览关闭")
        cam_center_vbox.addWidget(self.view_debug_camera, 1)

        cam_statusbar = QtWidgets.QFrame()
        cam_statusbar.setFixedHeight(26)
        cam_statusbar.setStyleSheet(f"QFrame{{background:{_HEADER_BG};border-top:1px solid #505050;}}")
        cam_sb_layout = QtWidgets.QHBoxLayout(cam_statusbar)
        cam_sb_layout.setContentsMargins(10, 0, 10, 0)
        self.lbl_debug_camera_status = QtWidgets.QLabel("相机状态：未扫描")
        self.lbl_debug_camera_status.setWordWrap(False)
        self.lbl_debug_camera_status.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        cam_sb_layout.addWidget(self.lbl_debug_camera_status)
        cam_center_vbox.addWidget(cam_statusbar)
        cam_main.addWidget(cam_center, 1)

        # ???????
        cam_right = QtWidgets.QFrame()
        cam_right.setFixedWidth(240)
        cam_right.setStyleSheet(f"QFrame{{background:{_PANEL_BG};border-left:1px solid #505050;}}")
        cam_right_vbox = QtWidgets.QVBoxLayout(cam_right)
        cam_right_vbox.setContentsMargins(0, 0, 0, 0)
        cam_right_vbox.setSpacing(0)

        cam_right_title = QtWidgets.QLabel("  参数设置")
        cam_right_title.setFixedHeight(28)
        cam_right_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        cam_right_vbox.addWidget(cam_right_title)

        cam_params = QtWidgets.QWidget()
        cam_params_form = QtWidgets.QFormLayout(cam_params)
        cam_params_form.setContentsMargins(12, 12, 12, 12)
        cam_params_form.setSpacing(10)
        self.view_debug_camera.set_runtime_pixmap(None, placeholder="预览关闭")

        self.spin_debug_exposure = QtWidgets.QDoubleSpinBox()
        self.spin_debug_exposure.setDecimals(1)
        self.spin_debug_exposure.setRange(1.0, 1000000.0)
        self.spin_debug_exposure.setValue(20000.0)
        self.spin_debug_exposure.setStyleSheet(_input_style)
        cam_params_form.addRow("曝光(us)", self.spin_debug_exposure)
        self.lbl_debug_camera_status = QtWidgets.QLabel("相机状态：未扫描")
        self.spin_debug_gain = QtWidgets.QDoubleSpinBox()
        self.spin_debug_gain.setDecimals(2)
        self.spin_debug_gain.setRange(0.0, 48.0)
        self.spin_debug_gain.setValue(0.0)
        self.spin_debug_gain.setStyleSheet(_input_style)
        cam_params_form.addRow("增益", self.spin_debug_gain)
        # ????????????????????? autoDefault ???Enter ??????????????
        self.spin_debug_exposure.setKeyboardTracking(False)
        self.spin_debug_gain.setKeyboardTracking(False)
        self.spin_debug_exposure.editingFinished.connect(self._on_debug_camera_param_editing_finished)
        self.spin_debug_gain.editingFinished.connect(self._on_debug_camera_param_editing_finished)

        self.cmb_debug_trigger_mode = QtWidgets.QComboBox()
        self.cmb_debug_trigger_mode.addItems(["software", "continuous"])
        self.cmb_debug_trigger_mode.setCurrentText("continuous")
        self.cmb_debug_trigger_mode.setStyleSheet(_input_style)
        cam_params_form.addRow("触发模式", self.cmb_debug_trigger_mode)
        # activated?????????????? setCurrentIndex ???
        self.cmb_debug_trigger_mode.activated.connect(self._on_debug_camera_trigger_activated)

        cam_right_vbox.addWidget(cam_params)
        cam_right_vbox.addSpacing(8)

        cam_btns_w = QtWidgets.QWidget()
        cam_btns_layout = QtWidgets.QVBoxLayout(cam_btns_w)
        cam_btns_layout.setContentsMargins(12, 0, 12, 12)
        cam_btns_layout.setSpacing(6)

        self.btn_debug_read_camera_settings = QtWidgets.QPushButton(_si(SP.SP_FileDialogInfoView), " 读取相机参数")
        self.btn_debug_read_camera_settings.setStyleSheet(_compact_btn)
        self.btn_debug_read_camera_settings.clicked.connect(self._refresh_debug_camera_settings)
        cam_btns_layout.addWidget(self.btn_debug_read_camera_settings)

        self.btn_debug_apply_camera_settings = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), " 应用相机参数")
        self.btn_debug_apply_camera_settings.setStyleSheet(_compact_btn)
        self.btn_debug_apply_camera_settings.clicked.connect(self._apply_debug_camera_settings)
        cam_btns_layout.addWidget(self.btn_debug_apply_camera_settings)

        cam_right_vbox.addWidget(cam_btns_w)
        cam_right_vbox.addStretch(1)
        cam_main.addWidget(cam_right)

        # IO 璋冭瘯锛堝璇濇鐢級鈥?MVS 椋庢牸甯冨眬
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

        io_ctrl_title = QtWidgets.QLabel("  连接控制")
        io_ctrl_title.setFixedHeight(28)
        io_ctrl_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        io_left_vbox.addWidget(io_ctrl_title)

        io_ctrl_w = QtWidgets.QWidget()
        io_ctrl_layout = QtWidgets.QVBoxLayout(io_ctrl_w)
        io_ctrl_layout.setContentsMargins(10, 10, 10, 10)
        io_ctrl_layout.setSpacing(6)

        self.btn_debug_open_io = QtWidgets.QPushButton(_si(SP.SP_DriveNetIcon), " 打开 IO 调试")
        self.btn_debug_open_io.setStyleSheet(_compact_btn)
        self.btn_debug_open_io.clicked.connect(self._open_debug_io)
        io_ctrl_layout.addWidget(self.btn_debug_open_io)

        self.btn_debug_close_io = QtWidgets.QPushButton(_si(SP.SP_DialogCloseButton), " 关闭 IO 调试")
        self.btn_debug_close_io.setStyleSheet(_compact_btn)
        self.btn_debug_close_io.clicked.connect(self._close_debug_io)
        io_ctrl_layout.addWidget(self.btn_debug_close_io)

        self.btn_debug_refresh_io = QtWidgets.QPushButton(_si(SP.SP_BrowserReload), " Refresh DI/DO Status")
        self.btn_debug_refresh_io.setStyleSheet(_compact_btn)
        self.btn_debug_refresh_io.clicked.connect(self._refresh_debug_io_snapshot)
        io_ctrl_layout.addWidget(self.btn_debug_refresh_io)

        self.btn_debug_simulate_trigger = QtWidgets.QPushButton(" 模拟脚踏触发（待补）")
        self.btn_debug_simulate_trigger.setStyleSheet(_compact_btn)
        self.btn_debug_simulate_trigger.setEnabled(False)
        io_ctrl_layout.addWidget(self.btn_debug_simulate_trigger)

        io_left_vbox.addWidget(io_ctrl_w)
        io_left_vbox.addSpacing(4)

        io_status_title = QtWidgets.QLabel("  IO Status")
        io_status_title.setFixedHeight(28)
        io_status_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;border-top:1px solid #505050;padding-left:8px;")
        io_left_vbox.addWidget(io_status_title)

        io_status_w = QtWidgets.QWidget()
        io_status_layout = QtWidgets.QVBoxLayout(io_status_w)
        io_status_layout.setContentsMargins(10, 10, 10, 10)
        io_status_layout.setSpacing(6)

        self.lbl_debug_di_snapshot = QtWidgets.QLabel("DI：未连接")
        self.lbl_debug_di_snapshot.setWordWrap(True)
        self.lbl_debug_di_snapshot.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        io_status_layout.addWidget(self.lbl_debug_di_snapshot)

        self.lbl_debug_do_snapshot = QtWidgets.QLabel("DO：未连接")
        self.lbl_debug_do_snapshot.setWordWrap(True)
        self.lbl_debug_do_snapshot.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        io_status_layout.addWidget(self.lbl_debug_do_snapshot)

        self.lbl_debug_io_mapping_summary = QtWidgets.QLabel("映射：未加载")
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

        io_panel_title = QtWidgets.QLabel("  DI / DO 通道面板")
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

        di_title = QtWidgets.QLabel("DI 输入监视")
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

            hint = QtWidgets.QLabel("未映射")
            hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color:{_TEXT_DIM};font-size:9px;border:none;")
            card_layout.addWidget(hint)

            self._debug_di_cards[channel] = card
            self._debug_di_indicators[channel] = indicator
            self._debug_di_hints[channel] = hint
            di_grid.addWidget(card, channel // 8, channel % 8)

        io_panel_layout.addLayout(di_grid)

        do_title = QtWidgets.QLabel("DO 输出控制")
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

            hint = QtWidgets.QLabel("未映射")
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
        if hasattr(self, "cmb_current_camera_role"):
            self.cmb_current_camera_role.setFixedWidth(72 if compact else 84)
        if hasattr(self, "btn_algorithm_picker"):
            style = self._algorithm_picker_style_compact if compact else self._algorithm_picker_style_default
            if style and self.btn_algorithm_picker.styleSheet() != style:
                self.btn_algorithm_picker.setStyleSheet(style)


    # ------------------------------------------------------------------
    # 底栏同步
    # ------------------------------------------------------------------

    def _sync_footer(self) -> None:
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "未设置"
        self.lbl_footer_ref.setText(f"参考图: {ref_name}")
        algo = self.current_algorithm_display_name() if hasattr(self, "cmb_algorithm") else ""
        self.lbl_footer_algo.setText(f"算法: {algo}" if algo else "")
        self.lbl_footer_product_dir.setText(f"产品目录: {self.session.product_dir}")

    # ------------------------------------------------------------------
    # 列表刷新
    # ------------------------------------------------------------------

    def _refresh_lists(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        if hasattr(self, "lbl_images_section"):
            title = "  图片列表"
            if current_role:
                title = f"{title}（{current_role}）"
            self.lbl_images_section.setText(title)

        def visible_files(files: List[str]) -> List[str]:
            if not current_role:
                return list(files)
            return _filter_paths_for_camera(self, files, current_role)

        def fill(listw: QtWidgets.QListWidget, files: List[str]) -> None:
            current_item = listw.currentItem()
            current_path = None
            if current_item is not None:
                current_path = current_item.data(QtCore.Qt.UserRole) or current_item.toolTip()
            filtered_files = visible_files(files)
            blocker = QtCore.QSignalBlocker(listw)
            listw.clear()
            selected_row = -1
            for index, p in enumerate(filtered_files):
                it = QtWidgets.QListWidgetItem(os.path.basename(p))
                it.setToolTip(p)
                it.setData(QtCore.Qt.UserRole, p)
                listw.addItem(it)
                if current_path and p == current_path:
                    selected_row = index
            if selected_row >= 0:
                listw.setCurrentRow(selected_row)
            del blocker

        fill(self.ok_list, self.ok_files)
        fill(self.ng_list, self.ng_files)
        fill(self.test_list, self.test_files)

    def _save_session(self) -> None:
        self.session.save_session(SessionData(
            ok_files=list(self.ok_files),
            ng_files=list(self.ng_files),
            test_files=list(self.test_files),
            ref_image=self.ref_image,
            loc_method=self.loc_method,
        ))

    # ------------------------------------------------------------------
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "Not Set"
    # ------------------------------------------------------------------

    def _current_selected_path(self) -> Optional[str]:
        tab = self.tabs.currentIndex()
        if tab == 0:
            items = self.ok_list.selectedItems()
            if not items:
                return None
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if path:
                return str(path)
            visible = _filter_paths_for_camera(self, self.ok_files, _selected_image_list_camera_role(self))
            row = self.ok_list.row(items[0])
            return visible[row] if row < len(visible) else None
        if tab == 1:
            items = self.ng_list.selectedItems()
            if not items:
                return None
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if path:
                return str(path)
            visible = _filter_paths_for_camera(self, self.ng_files, _selected_image_list_camera_role(self))
            row = self.ng_list.row(items[0])
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
            return
        self._clear_selected_inspection_item()
        if self.canvas.image_path() != path:
            self._load_canvas_image(path)
        self._set_status_for_current_image(path)

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
        self.lbl_status.setText(f"状态：已切换到 {self.current_camera_role()}，请重新选择图片。")

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
            f"Select images to add into {kind}",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return
        if kind == "OK":
            self.ok_files.extend(files)
            self.ok_files = sorted(list(dict.fromkeys(self.ok_files)))
        elif kind == "NG":
            self.ng_files.extend(files)
            self.ng_files = sorted(list(dict.fromkeys(self.ng_files)))
        else:
            self.test_files.extend(files)
            self.test_files = sorted(list(dict.fromkeys(self.test_files)))
        self._refresh_lists()
        self._clear_training_roi_review_state()
        self._save_session()

    def _remove_selected_from(self, kind: str) -> None:
        if kind == "OK":
            items = self.ok_list.selectedItems()
            if not items:
                return
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if not path:
                visible = _filter_paths_for_camera(self, self.ok_files, _selected_image_list_camera_role(self))
                idx = self.ok_list.row(items[0])
                path = visible[idx] if idx < len(visible) else None
            if not path or path not in self.ok_files:
                return
            self.ok_files.remove(str(path))
        elif kind == "NG":
            items = self.ng_list.selectedItems()
            if not items:
                return
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if not path:
                visible = _filter_paths_for_camera(self, self.ng_files, _selected_image_list_camera_role(self))
                idx = self.ng_list.row(items[0])
                path = visible[idx] if idx < len(visible) else None
            if not path or path not in self.ng_files:
                return
            self.ng_files.remove(str(path))
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

    def _on_tab_changed(self, index: int) -> None:
        if index == 0:
            listw = self.ok_list
            files = _filter_paths_for_camera(self, self.ok_files, _selected_image_list_camera_role(self))
        elif index == 1:
            listw = self.ng_list
            files = _filter_paths_for_camera(self, self.ng_files, _selected_image_list_camera_role(self))
        else:
            listw = self.test_list
            files = _filter_paths_for_camera(self, self.test_files, _selected_image_list_camera_role(self))

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
        name, ok = QtWidgets.QInputDialog.getText(self, "新建产品", "请输入产品名称：")
        if not ok or not name.strip():
            return
        error = self.session.create_product(name.strip())
        if error:
            QtWidgets.QMessageBox.warning(self, "错误", error)
            return
        self.cmb_product.addItem(name.strip())
        self.cmb_product.setCurrentText(name.strip())

    def _clear_session(self) -> None:
        ret = QtWidgets.QMessageBox.question(self, "清空会话", "确认清空当前会话数据（列表 / 参考图 / 缓存）？")
        if ret != QtWidgets.QMessageBox.Yes:
            return
        self.sessionClearRequested.emit()

    # ------------------------------------------------------------------
    # Canvas / ROI
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
        ret = QtWidgets.QMessageBox.question(self, "清空标注", "确认清空当前图片的所选标注？")
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
            QtWidgets.QMessageBox.warning(self, "Info", "anchor_mask only supports polygon annotation.")
            return

        if st.shape_type == "rect":
            if st.xywh is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先拖拽画出矩形标注")
                return
            jpath = qr_core.upsert_labelme_rect(p, st.xywh, label_name=label_name)
        else:
            if not st.points or len(st.points) < 3:
                QtWidgets.QMessageBox.warning(self, "Info", "Polygon needs at least 3 points.")
                return
            jpath = qr_core.upsert_labelme_polygon(p, st.points, label_name=label_name)

        QtWidgets.QMessageBox.information(
            self, "已保存",
            f"已更新到 labelme json：\n{jpath}\n(label={label_name}, type={st.shape_type})",
        )
        self._load_canvas_image(p)

    # ------------------------------------------------------------------
    # ??? / ??
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 自动 ROI
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 算法参数

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
        self._sync_training_action_buttons()

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
            self.lbl_status.setText("状态：请选择工具")
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
    # 训练
    # ------------------------------------------------------------------

    def _resolve_training_algorithm(self, inspection_item: InspectionItem) -> str:
        if self.algo.is_learning_tool(inspection_item.algorithm_code):
            return self.algo.current_learning_backbone()
        return self.algo.resolve_tool_algorithm(inspection_item.algorithm_code)

    def _training_camera_roles_in_lists(self, camera_id: object | None = None) -> List[str]:
        if camera_id is None:
            candidate_paths = list(self.ok_files) + list(self.ng_files)
        else:
            candidate_paths = _filter_paths_for_camera(
                self,
                list(self.ok_files) + list(self.ng_files),
                camera_id,
            )
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
        suffix = f"（当前角色 {role_text}）" if role_text else ""
        QtWidgets.QMessageBox.warning(
            self,
            "训练样本提示",
            f"当前 OK/NG 列表{suffix}同时包含 cam1 和 cam2 图片。\n"
            "请先分开整理样本后，再执行训练/注册/标定。",
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
        if missing_paths and self.loc_method == "line2dup":
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

    def _train_inspection_item(self, inspection_item: InspectionItem) -> TrainResult:
        if not inspection_item.enabled:
            raise RuntimeError("selected tool is disabled")

        algorithm = self._resolve_training_algorithm(inspection_item)
        if not algorithm:
            if self.algo.is_learning_tool(inspection_item.algorithm_code):
                raise RuntimeError("please choose a learning tool subtype first")
            raise RuntimeError("please select an inspection tool")

        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        training_ok_files = _filter_paths_for_camera(self, self.ok_files, inspection_item.camera_id)
        training_ng_files = _filter_paths_for_camera(self, self.ng_files, inspection_item.camera_id)
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
            list(training_ok_files) + list(training_ng_files),
        )
        if missing_paths:
            missing = [os.path.basename(path) for path in missing_paths[:50]]
            raise RuntimeError(
                f"missing ROI label '{roi_label}' in some OK/NG jsons:\n" + "\n".join(missing)
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
        if self._warn_mixed_training_camera_samples(self.current_camera_role()):
            return
        if not self._ensure_training_roi_reviewed(
            self.current_camera_role(),
            action_name="训练 / 标定全部启用工具",
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
            QtWidgets.QMessageBox.information(self, "Info", f"Please enable at least one inspection tool for {current_role}.")
            return

        selected_item = self._selected_inspection_item()
        selected_item_id = str(selected_item.item_id or "") if selected_item is not None else ""
        display_rows: List[Dict[str, object]] = []
        success_names: List[str] = []
        failure_messages: List[str] = []
        last_status_message = ""

        for inspection_item in enabled_items:
            display_name = str(
                inspection_item.display_name or inspection_item.roi_label or inspection_item.item_id or "tool"
            ).strip()
            try:
                result = self._train_inspection_item(inspection_item)
                success_names.append(display_name)
                last_status_message = result.status_message
                if not result.is_embedding and result.result_rows:
                    if inspection_item.item_id == selected_item_id:
                        display_rows = result.result_rows
                    elif not display_rows:
                        display_rows = result.result_rows
            except Exception as exc:
                failure_messages.append(f"{display_name}: {exc}")

        if last_status_message:
            self.lbl_status.setText(last_status_message)
        if display_rows:
            self._populate_results_table(display_rows)

        self._save_runtime_params()
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
            QtWidgets.QMessageBox.warning(self, "Train Result", "\n".join(summary_lines))
            return

        QtWidgets.QMessageBox.information(
            self,
            "Train Result",
            f"Finished training/calibrating {len(success_names)} enabled tool(s).",
        )

    def _train(self) -> None:
        self.algo.model = None
        self.table.setRowCount(0)
        self._current_result_rows = []
        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, "Info", "Please select one inspection tool in the table first.")
            return
        if self._warn_mixed_training_camera_samples(inspection_item.camera_id):
            return
        if not self._ensure_training_roi_reviewed(
            inspection_item.camera_id,
            action_name="标定当前工具",
            action_key="current",
        ):
            return
        if not inspection_item.enabled:
            QtWidgets.QMessageBox.information(self, "提示", "当前选中的检测工具已禁用")
            return
        if self.algo.is_learning_tool(inspection_item.algorithm_code):
            algorithm = self.algo.current_learning_backbone()
        else:
            algorithm = self.algo.resolve_tool_algorithm(inspection_item.algorithm_code)
        if not algorithm:
            QtWidgets.QMessageBox.information(self, "提示", "请先选择工具")
            return
        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        label_names = [roi_label]
        training_ok_files = _filter_paths_for_camera(self, self.ok_files, inspection_item.camera_id)
        training_ng_files = _filter_paths_for_camera(self, self.ng_files, inspection_item.camera_id)
        missing_groups: List[str] = []
        if not training_ok_files:
            missing_groups.append("OK")
        if not training_ng_files:
            missing_groups.append("NG")
        if missing_groups:
            camera_id = _normalize_camera_role(inspection_item.camera_id) or "cam1"
            QtWidgets.QMessageBox.warning(
                self,
                "训练样本不足",
                f"{camera_id} 缺少 {'/'.join(missing_groups)} 图片，请先补齐对应角色的样本。",
            )
            return
        candidate_paths = list(training_ok_files) + list(training_ng_files)
        missing_paths = []
        for path in candidate_paths:
            json_path = qr_core.labelme_json_of_image(path)
            if not os.path.exists(json_path):
                missing_paths.append(path)
                continue
            if qr_core.read_shape_from_labelme(json_path, roi_label) is None:
                missing_paths.append(path)
        if missing_paths and self.loc_method == "line2dup":
            try:
                self._autogen_roi_for_images(missing_paths, only_missing=False, silent=True)
                refreshed_missing_paths = []
                for path in candidate_paths:
                    json_path = qr_core.labelme_json_of_image(path)
                    if not os.path.exists(json_path) or qr_core.read_shape_from_labelme(json_path, roi_label) is None:
                        refreshed_missing_paths.append(path)
                missing_paths = refreshed_missing_paths
            except Exception:
                pass
        missing = [os.path.basename(p) for p in missing_paths]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "缺少 ROI 标注",
                f"每张 OK/NG 图片都需要包含 ROI: {roi_label}\n"
                "请逐张打开图片并保存对应 ROI。\n缺少文件:\n" + "\n".join(missing[:50]),
            )
            return

        self.algo.product_params.algorithm = algorithm
        self.algo.product_params.score_mode = self.cmb_mode.currentText()
        self.algo.product_params.margin = float(self.spin_margin.value())
        self.algo.product_params.topk = int(self.spin_topk.value())

        try:
            result: TrainResult = self.algo.train(
                training_ok_files,
                training_ng_files,
                algorithm=algorithm,
                product_dir=self.session.product_dir,
                label_names=label_names,
                model_key=inspection_item.model_key,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "训练失败", str(e))
            return

        self.lbl_status.setText(result.status_message)
        if not result.is_embedding and result.result_rows:
            self._populate_results_table(result.result_rows)

        self._save_runtime_params()
        self._save_session()
        self._refresh_inspection_items_table()
        self._update_runtime_widgets()
        QtWidgets.QMessageBox.information(self, "训练完成", result.dialog_message)

    # ------------------------------------------------------------------
    # 预测 / 测试
    # ------------------------------------------------------------------

    def _test_target_inspection_items(self) -> List[InspectionItem]:
        current_role = self.current_camera_role()
        return [
            item
            for item in self.inspection_items
            if item.enabled and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
        ]

    def _run_test(self) -> None:
        p = self.canvas.image_path()
        if p is None or not os.path.exists(p):
            QtWidgets.QMessageBox.warning(self, "Info", "Please open a test image first.")
            return
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
            QtWidgets.QMessageBox.critical(self, "测试失败", str(ex))
            return

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
                row.setdefault("pred", item_result.result)
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
                self._line2dup_output_labels()
                if self.loc_method == "line2dup"
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
            if str(row.get("pred", "NG") or "NG") != "OK"
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


    def _show_tool_dialog(
        self,
        key: str,
        title: str,
        widget: QtWidgets.QWidget,
        *,
        size: Tuple[int, int],
    ) -> None:
        dialog = self._tool_dialogs.get(key)
        if dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(title)
            dialog.setModal(False)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addWidget(widget)
            dialog.resize(*size)
            if key == "camera_debug":
                dialog.finished.connect(lambda *_: self._stop_debug_camera_preview())
            self._tool_dialogs[key] = dialog
        if key == "camera_debug":
            # ??????????? Enter ???????????????????????
            for _btn in dialog.findChildren(QtWidgets.QPushButton):
                _btn.setAutoDefault(False)
                _btn.setDefault(False)
        widget.show()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


    def _export_current_results_csv(self) -> None:
        if not self._current_result_rows:
            QtWidgets.QMessageBox.information(self, "提示", "当前没有可导出的测试结果")
            return
        json_path, csv_path = self._save_test_result_report(
            self._current_result_rows,
            report_prefix="test_result",
        )
        QtWidgets.QMessageBox.information(
            self,
            "导出完成",
            f"测试结果已导出到：\n{json_path}\n{csv_path}",
        )

    def _on_table_click(self, row: int, _col: int) -> None:
        it = self.table.item(row, 0)
        if it is None:
            return
        p = it.data(QtCore.Qt.UserRole)
        if isinstance(p, str) and os.path.exists(p):
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    # ------------------------------------------------------------------
    # 分析 / 验证
    # ------------------------------------------------------------------


    def _save_margin_report(
        self, rows: List[Dict[str, object]], summary: Dict[str, object]
    ) -> Tuple[str, str]:
        report_dir = os.path.join(self.session.product_dir, "margin_reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"margin_report_{self.current_algorithm()}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {
            "product": self.session.current_product,
            "algorithm": self.current_algorithm(),
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
        else:
            algorithm = self.current_algorithm()
            labels_override = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
            algorithm_override = None
            model_key_override = None
        if not self._is_embedding_algorithm(algorithm):
            QtWidgets.QMessageBox.information(self, "Info", "Traditional algorithms do not support margin validation.")
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
            QtWidgets.QMessageBox.warning(self, "Info", "Please train/register first (OK + NG).")
            return
        if not self.ok_files or not self.ng_files:
            QtWidgets.QMessageBox.warning(self, "Info", "Need at least one OK and one NG image for margin validation.")
            return

        feat_net = self.algo.get_feat_net(
            self.algo.model.backbone,
            getattr(self.algo.model, "device", None),
        )
        rows: List[Dict[str, object]] = []
        try:
            for path in self.ok_files:
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
            for path in self.ng_files:
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
            QtWidgets.QMessageBox.critical(self, "Margin validation failed", str(ex))
            return

        self._populate_results_table(rows)
        summary = self._suggest_margin_from_rows(rows)
        json_path, csv_path = self._save_margin_report(rows, summary)

        safe_range = summary.get("safe_range")
        safe_text = ""
        if isinstance(safe_range, tuple):
            safe_text = f"\n安全区间: {safe_range[0]:.4f} ~ {safe_range[1]:.4f}"

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
        from ui.debug import EmbeddingAnalysisDialog

        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, "Info", "Please select a learning tool first.")
            return
        if not inspection_item.enabled:
            QtWidgets.QMessageBox.information(self, "提示", "当前选中的检测工具已禁用")
            return
        if not self.algo.is_learning_tool(inspection_item.algorithm_code):
            QtWidgets.QMessageBox.information(self, "Info", "Current selection is not a learning tool.")
            return
        dialog = EmbeddingAnalysisDialog(
            session_root=self.session.session_dir,
            initial_product=self.session.current_product,
            initial_backbone=self.algo.current_learning_backbone(),
            initial_model_key=inspection_item.model_key,
            parent=self,
        )
        dialog.exec()

    # ------------------------------------------------------------------
    # 传统基线调试
    # ------------------------------------------------------------------


    def _run_traditional_baseline_debug(self) -> None:
        paths, tab_name = self._current_tab_paths_and_name()
        if not paths:
            QtWidgets.QMessageBox.information(self, "Info", "Current list has no images.")
            return

        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, "Info", "Please select an inspection tool first.")
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
