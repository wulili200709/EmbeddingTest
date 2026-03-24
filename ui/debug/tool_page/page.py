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
    debugCameraConnected = QtCore.Signal(str)
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
        self.inspection_items: List[InspectionItem] = []
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
        self._debug_output_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._debug_io_timer = QtCore.QTimer(self)
        self._debug_io_timer.setInterval(500)
        self._debug_io_timer.timeout.connect(self._refresh_debug_io_snapshot)
        self._camera_settings_store = CameraSettingsStore(self.session.camera_settings_path)
        # ?? setValue/????????????????????
        self._debug_camera_block_spin_apply = False

        self._build_ui()
        self.destroyed.connect(lambda *_: self._cleanup_debug_hardware())

    # ------------------------------------------------------------------
    # 鍏紑鎺ュ彛锛圡ainWindow 璋冪敤锛?
    # ------------------------------------------------------------------

    def current_algorithm(self) -> str:
        value = self.cmb_algorithm.currentData() if hasattr(self, "cmb_algorithm") else None
        if value is None:
            return ""
        return str(value).strip()

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
            button.setText(self.current_algorithm_display_name() or "请选择工具")
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

    def open_io_debug_dialog(self) -> None:
        self._show_tool_dialog(
            "io_debug",
            "DI / DO 调试工具",
            self.io_debug_page,
            size=(900, 480),
        )

    def open_template_editor_dialog(self) -> None:
        self._open_line2dup_template_page()

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
        self.ref_image = sd.ref_image
        self.loc_method = sd.loc_method

        if self.ref_image:
            self.lbl_ref.setText(f"参考图: {os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        self.cmb_loc.setCurrentText(self.loc_method)
        self._refresh_lists()

        if os.path.exists(self.session.line2dup_recipe_path):
            try:
                self.line2dup_recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
                if (
                    not self.ref_image
                    and self.line2dup_recipe.reference_image
                    and os.path.exists(self.line2dup_recipe.reference_image)
                ):
                    self.ref_image = self.line2dup_recipe.reference_image
                    self.lbl_ref.setText(f"参考图: {os.path.basename(self.ref_image)}")
                    self.lbl_ref.setToolTip(self.ref_image)
            except Exception:
                self.line2dup_recipe = None

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

        header_layout.addStretch(1)

        self.lbl_status = QtWidgets.QLabel("\u72b6\u6001\uff1a\u672a\u8bad\u7ec3")
        self.lbl_status.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;")
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
        right_panel.setFixedWidth(400)
        right_panel.setStyleSheet(f"background:{_PANEL_BG};border-left:1px solid #505050;")
        right_vbox = QtWidgets.QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        # --- 图片列表 ---
        sec_images = QtWidgets.QLabel("  \u56fe\u7247\u5217\u8868")
        sec_images.setFixedHeight(28)
        sec_images.setStyleSheet(_section_style)
        right_vbox.addWidget(sec_images)

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
        sec_algo = QtWidgets.QLabel("  \u7b97\u6cd5\u53c2\u6570")
        sec_algo.setFixedHeight(28)
        sec_algo.setStyleSheet(_section_style)
        right_vbox.addWidget(sec_algo)

        algo_frame = QtWidgets.QWidget()
        algo_form = QtWidgets.QFormLayout(algo_frame)
        algo_form.setContentsMargins(10, 6, 10, 6)
        algo_form.setSpacing(4)
        algo_form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.cmb_algorithm = QtWidgets.QComboBox()
        self._populate_algorithm_combo()
        self.cmb_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        self.cmb_algorithm.hide()
        self.cmb_backbone = self.cmb_algorithm
        self.btn_algorithm_picker = QtWidgets.QToolButton()
        self.btn_algorithm_picker.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_algorithm_picker.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_algorithm_picker.setMenu(self._build_algorithm_picker_menu())
        self.btn_algorithm_picker.setStyleSheet(
            "QToolButton{background:#2f2f2f;color:#e0e0e0;border:1px solid #555;"
            "padding:5px 28px 5px 8px;border-radius:3px;font-size:12px;}"
            "QToolButton:hover{background:#3a3a3a;}"
        )
        self.btn_algorithm_picker.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
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
        self.inspection_items_table.setMaximumHeight(210)
        self.inspection_items_table.setColumnWidth(0, 52)
        self.inspection_items_table.itemChanged.connect(self._on_inspection_items_table_item_changed)
        self.inspection_items_table.itemSelectionChanged.connect(self._on_inspection_items_selection_changed)
        tool_vbox.addWidget(self.inspection_items_table)
        right_vbox.addWidget(tool_frame)
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

        self.btn_train = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), "\u8bad\u7ec3 / \u6807\u5b9a\u5168\u90e8\u542f\u7528\u5de5\u5177")
        self.btn_train.setStyleSheet(_action_btn)
        self.btn_train.clicked.connect(self._train_all_tools)
        action_vbox.addWidget(self.btn_train)

        self.btn_train_current = QtWidgets.QPushButton("\u6807\u5b9a\u5f53\u524d\u5de5\u5177")
        self.btn_train_current.setStyleSheet(_compact_btn)
        self.btn_train_current.clicked.connect(self._train)
        action_vbox.addWidget(self.btn_train_current)

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
        self.btn_debug_refresh_io = QtWidgets.QPushButton(_si(SP.SP_BrowserReload), " Refresh DI/DO Status")
        io_left_vbox.addWidget(io_status_w)
        io_left_vbox.addStretch(1)
        io_main.addWidget(io_left)

        io_right = QtWidgets.QWidget()
        io_right_vbox = QtWidgets.QVBoxLayout(io_right)
        io_right_vbox.setContentsMargins(0, 0, 0, 0)
        io_right_vbox.setSpacing(0)

        io_output_title = QtWidgets.QLabel("  DO 输出点动控制")
        io_output_title.setFixedHeight(28)
        io_output_title.setStyleSheet(f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;border-bottom:1px solid #505050;padding-left:8px;")
        io_status_title = QtWidgets.QLabel("  IO Status")

        io_grid_w = QtWidgets.QWidget()
        io_grid_w.setStyleSheet(f"background:{_DARK_BG};")
        io_grid = QtWidgets.QGridLayout(io_grid_w)
        io_grid.setContentsMargins(16, 16, 16, 16)
        io_grid.setSpacing(10)

        _toggle_btn_css = (
            "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
            "padding:10px 20px;border-radius:4px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#505050;}"
            "QPushButton:checked{background:#3794ff;color:white;border:1px solid #3794ff;}"
        )

        output_specs = [
            ("tower_red", "红灯"), ("tower_green", "绿灯"), ("tower_blue", "蓝灯"),
            ("light_cam1", "光源1"), ("light_cam2", "光源2"),
        ]
        for index, (name, label) in enumerate(output_specs):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.setStyleSheet(_toggle_btn_css)
            button.setMinimumHeight(40)
            button.toggled.connect(lambda checked, output_name=name: self._set_debug_output(output_name, checked))
            self._debug_output_buttons[name] = button
            row = index // 3
            col = index % 3
            io_grid.addWidget(button, row, col)

        io_right_vbox.addWidget(io_grid_w)
        io_right_vbox.addStretch(1)
        io_main.addWidget(io_right, 1)

        self.lbl_template_tool_hint = QtWidgets.QLabel("")
        self.lbl_template_tool_hint.hide()
        self._normalize_stylesheet_font_units()

    def _toggle_tool_config_section(self, checked: bool) -> None:
        frame = getattr(self, "tool_config_frame", None)
        if frame is not None:
            frame.setVisible(bool(checked))
        toggle = getattr(self, "btn_toggle_tools", None)
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
        def fill(listw: QtWidgets.QListWidget, files: List[str]) -> None:
            listw.clear()
            for p in files:
                it = QtWidgets.QListWidgetItem(os.path.basename(p))
                it.setToolTip(p)
                listw.addItem(it)

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
            row = self.ok_list.row(items[0])
            return self.ok_files[row] if row < len(self.ok_files) else None
        if tab == 1:
            items = self.ng_list.selectedItems()
            if not items:
                return None
            row = self.ng_list.row(items[0])
            return self.ng_files[row] if row < len(self.ng_files) else None
        items = self.test_list.selectedItems()
        if not items:
            return None
        row = self.test_list.row(items[0])
        return self.test_files[row] if row < len(self.test_files) else None

    def _show_selected_image_path(self, path: Optional[str]) -> None:
        if not path:
            return
        if self.canvas.image_path() != path:
            self._load_canvas_image(path)
        self._set_status_for_current_image(path)

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
        self._save_session()

    def _remove_selected_from(self, kind: str) -> None:
        if kind == "OK":
            items = self.ok_list.selectedItems()
            if not items:
                return
            idx = self.ok_list.row(items[0])
            self.ok_files.pop(idx)
        elif kind == "NG":
            items = self.ng_list.selectedItems()
            if not items:
                return
            idx = self.ng_list.row(items[0])
            self.ng_files.pop(idx)
        else:
            f"Select images to add into {kind}",
            if not items:
                return
            idx = self.test_list.row(items[0])
            self.test_files.pop(idx)
        self._refresh_lists()
        self._save_session()

    def _on_tab_changed(self, index: int) -> None:
        if index == 0:
            listw = self.ok_list
            files = self.ok_files
        elif index == 1:
            listw = self.ng_list
            files = self.ng_files
        else:
            listw = self.test_list
            files = self.test_files

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
        has_enabled_items = any(getattr(item, "enabled", False) for item in inspection_items)
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
        missing_paths = self._missing_training_roi_paths(
            roi_label,
            list(self.ok_files) + list(self.ng_files),
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
            self.ok_files,
            self.ng_files,
            algorithm=algorithm,
            product_dir=self.session.product_dir,
            label_names=[roi_label],
            model_key=inspection_item.model_key,
        )

    def _train_all_tools(self) -> None:
        self.algo.model = None
        self.table.setRowCount(0)
        self._current_result_rows = []

        enabled_items = [item for item in self.inspection_items if item.enabled]
        if not enabled_items:
            QtWidgets.QMessageBox.information(self, "Info", "Please enable at least one inspection tool.")
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
        candidate_paths = list(self.ok_files) + list(self.ng_files)
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
                self.ok_files, self.ng_files,
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
        enabled_items = [item for item in self.inspection_items if item.enabled]
        if not enabled_items:
            return []

        selected_item = self._selected_inspection_item()
        selected_camera_id = (
            str(selected_item.camera_id or "").strip()
            if selected_item is not None
            else ""
        )
        if not selected_camera_id:
            selected_camera_id = str(enabled_items[0].camera_id or "").strip()
        if selected_camera_id:
            camera_items = [
                item
                for item in enabled_items
                if str(item.camera_id or "").strip() == selected_camera_id
            ]
            if camera_items:
                return camera_items
        return enabled_items

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
            else (
                str((self._selected_inspection_item().camera_id or "")).strip()
                if self._selected_inspection_item() is not None
                else "cam1"
            )
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
        match_ms = float(response.match_ms or 0.0)
        infer_ms = float(response.infer_ms or 0.0)
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
