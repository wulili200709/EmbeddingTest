"""
tool_page_pyside6.py

工具页 QWidget：封装 ROI 标注、自动定位、预测/测试、分析/验证全部 UI 与业务。

职责块：
  ③ ROI 标注          _load_canvas_image / _save_current_rect / _on_select_ok …
  ④ 自动 ROI / 定位   _autogen_roi_for_images / _build_shape_model / _open_line2dup_template_page …
  ⑤ 预测/测试         _predict_image / _run_test / _populate_results_table / _append_test_log …
  ⑥ 分析/验证         _suggest_margin_from_rows / _run_margin_validation / _run_traditional_baseline_debug …

通过 Signal 向 MainWindow 暴露跨边界意图：
  productChangeRequested(str)  — 用户切换了产品，MainWindow 先断运行相机，再调 apply_product_switch()
  sessionClearRequested()      — 用户点"清空会话"，MainWindow 先断运行相机，再调 reset_for_clear()
  sessionLoaded()              — 会话/产品加载完毕，MainWindow 刷新运行页状态显示

不负责：
  - 任何运行链路（相机、触发、记录）
  - RuntimeModePage 的交互
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import qr_core_proxy as qr_core

from camera_settings_store import (
    CameraSettingsStore,
    hik_settings_kwargs_from_mapping,
)
from application import (
    AlgorithmController,
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
    inspection_item_specs_from_line2dup_recipe,
    load_inspection_items,
    save_inspection_items,
    sync_items_with_labels,
    output_labels_from_line2dup_recipe,
)
import line2dup_locator
from line2dup_recipe import Line2DupRecipe
from ui.debug import (
    OverlayShape,
    RoiCanvas,
)
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
                    self.errorOccurred.emit(f"预览转换失败：{exc}")
                    self.msleep(120)
                continue

            self.frameReady.emit(image)
            self.msleep(30)


# ---------------------------------------------------------------------------
# ToolPage
# ---------------------------------------------------------------------------

class ToolPage(QtWidgets.QWidget):
    """
    工具页主体 QWidget。

    使用方式（MainWindow 中）::

        self.tool_page = ToolPage(self.session, self.algo)
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(lambda: self._refresh_runtime_status_ui("工具页会话已加载"))
        self.main_pages.addTab(self.tool_page, "工具页")
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
        self._line2dup_match_ms_by_image: Dict[str, float] = {}
        self._line2dup_autogen_ms_by_image: Dict[str, float] = {}
        self._current_result_rows: List[Dict[str, object]] = []
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
        self._camera_settings_store = CameraSettingsStore()
        # 避免 setValue/从相机读回时误触发「编辑完成→下发」逻辑
        self._debug_camera_block_spin_apply = False

        self._build_ui()
        self.destroyed.connect(lambda *_: self._cleanup_debug_hardware())

    # ------------------------------------------------------------------
    # 公开接口（MainWindow 调用）
    # ------------------------------------------------------------------

    def current_algorithm(self) -> str:
        return str(self.cmb_algorithm.currentText() or "").strip()

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
                "algorithm_type": item.algorithm_type,
                "enabled": bool(item.enabled),
                "status_kind": status_kind if item.enabled else "disabled",
                "status_text": status_text if item.enabled else "已禁用",
            }
            for item in self.inspection_items
        ]

    def load_embedding_model(self, algorithm: str) -> None:
        """加载指定算法的 embedding 模型并把状态写入 lbl_status。"""
        _, msg = self.algo.load_model_for_algorithm(algorithm, self.session.product_dir)
        self.lbl_status.setText(msg)

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """供 MainWindow runtime 检测回调使用。"""
        return self._predict_image(path, feat_net=feat_net, labels_override=labels_override)

    def load_session(self) -> None:
        """
        从磁盘读取算法参数和会话文件，刷新全部 UI。
        完成后发射 sessionLoaded。
        """
        self.algo.load_params(self.session.product_params_path)
        self.algo.model = None
        self._apply_runtime_params_to_ui()

        sd = self.session.load_session()
        self.ok_files = sd.ok_files
        self.ng_files = sd.ng_files
        self.test_files = sd.test_files
        self.ref_image = sd.ref_image
        self.loc_method = sd.loc_method

        if self.ref_image:
            self.lbl_ref.setText(f"参考图：{os.path.basename(self.ref_image)}")
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
                    self.lbl_ref.setText(f"参考图：{os.path.basename(self.ref_image)}")
                    self.lbl_ref.setToolTip(self.ref_image)
            except Exception:
                self.line2dup_recipe = None

        self._reload_inspection_items()
        self._sync_footer()

        self.sessionLoaded.emit()

    def apply_product_switch(self, name: str) -> None:
        """
        MainWindow 在断开运行相机后调用此方法，完成产品切换。
        内部会更新 session 路径、清空状态、重新加载会话。
        """
        self.session.switch_product(name)
        self.session.save_products()

        self.algo.model = None
        self.line2dup_recipe = None
        self.ref_image = None
        self._line2dup_match_ms_by_image = {}
        self._line2dup_autogen_ms_by_image = {}
        self.ok_files = []
        self.ng_files = []
        self.test_files = []
        self._current_result_rows = []
        self.inspection_items = []

        self.table.setRowCount(0)
        self.canvas.clear_image()
        self.lbl_ref.setText("参考图：未设置")
        self.lbl_ref.setToolTip("")
        self.lbl_status.setText("状态：已切换产品")

        self.load_session()
        self._refresh_lists()

    def reset_for_clear(self) -> None:
        """
        MainWindow 在断开运行相机后调用此方法，完成清空会话。
        内部删除 session.json 并重置全部状态。
        """
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
        )
        _section_style = (
            f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;"
            "border-bottom:1px solid #505050;padding:6px 10px;"
        )

        self.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶栏 ──
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

        # ── 主体：Canvas + 右侧面板 ──
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
        right_panel.setFixedWidth(300)
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
        self.cmb_algorithm.addItems(SUPPORTED_ALGORITHMS)
        self.cmb_algorithm.currentTextChanged.connect(self._on_algorithm_changed)
        self.cmb_algorithm.setStyleSheet(_input_style)
        self.cmb_backbone = self.cmb_algorithm
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
        lbl_a = QtWidgets.QLabel("\u7b97\u6cd5"); lbl_a.setStyleSheet(_lbl_s)
        lbl_m = QtWidgets.QLabel("\u5224\u5b9a"); lbl_m.setStyleSheet(_lbl_s)
        lbl_mg = QtWidgets.QLabel("Margin"); lbl_mg.setStyleSheet(_lbl_s)
        lbl_tk = QtWidgets.QLabel("TopK"); lbl_tk.setStyleSheet(_lbl_s)
        algo_form.addRow(lbl_a, self.cmb_algorithm)
        algo_form.addRow(lbl_m, self.cmb_mode)
        algo_form.addRow(lbl_mg, self.spin_margin)
        algo_form.addRow(lbl_tk, self.spin_topk)
        right_vbox.addWidget(algo_frame)

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

        self.btn_train = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), "\u8bad\u7ec3 / \u6ce8\u518c")
        self.btn_train.setStyleSheet(_action_btn)
        self.btn_train.clicked.connect(self._train)
        action_vbox.addWidget(self.btn_train)

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

        # ── 底栏 ──
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

        # ── 隐藏/对话框专用控件（保持接口兼容） ──

        # ROI 标注工具栏（隐藏）
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
        self.btn_build_shape = QtWidgets.QPushButton(_si(SP.SP_DialogApplyButton), "\u751f\u6210\u6a21\u677f")
        self.btn_build_shape.clicked.connect(self._build_shape_model)
        auto_l.addWidget(self.lbl_ref, 0, 0, 1, 3)
        auto_l.addWidget(self.btn_set_ref, 1, 0)
        auto_l.addWidget(self.btn_pick_ref, 1, 1)
        auto_l.addWidget(self.btn_build_shape, 1, 2)
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

        # 相机调试（对话框用）— MVS 风格三栏布局
        self.camera_debug_page = QtWidgets.QWidget()
        self.camera_debug_page.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")
        cam_main = QtWidgets.QHBoxLayout(self.camera_debug_page)
        cam_main.setContentsMargins(0, 0, 0, 0)
        cam_main.setSpacing(0)

        # ── 左侧：设备列表 + 设备信息 ──
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

        self.lbl_debug_camera_info = QtWidgets.QLabel("相机信息：-")
        self.lbl_debug_camera_info.setWordWrap(True)
        self.lbl_debug_camera_info.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;padding:8px;")
        cam_left_vbox.addWidget(self.lbl_debug_camera_info)
        cam_left_vbox.addStretch(1)
        cam_main.addWidget(cam_left)

        # ── 中央：工具栏 + 预览画面 + 状态栏 ──
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
        self.view_debug_camera.set_runtime_pixmap(None, placeholder="未开启实时预览")
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

        # ── 右侧：参数面板 ──
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
        cam_params_form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.spin_debug_exposure = QtWidgets.QDoubleSpinBox()
        self.spin_debug_exposure.setDecimals(1)
        self.spin_debug_exposure.setRange(1.0, 1000000.0)
        self.spin_debug_exposure.setValue(20000.0)
        self.spin_debug_exposure.setStyleSheet(_input_style)
        cam_params_form.addRow("曝光(us)", self.spin_debug_exposure)

        self.spin_debug_gain = QtWidgets.QDoubleSpinBox()
        self.spin_debug_gain.setDecimals(2)
        self.spin_debug_gain.setRange(0.0, 48.0)
        self.spin_debug_gain.setValue(0.0)
        self.spin_debug_gain.setStyleSheet(_input_style)
        cam_params_form.addRow("增益", self.spin_debug_gain)
        # 回车应提交到控件并下发相机；若对话框里存在 autoDefault 按钮，Enter 会先点「读取」导致从相机读回旧值
        self.spin_debug_exposure.setKeyboardTracking(False)
        self.spin_debug_gain.setKeyboardTracking(False)
        self.spin_debug_exposure.editingFinished.connect(self._on_debug_camera_param_editing_finished)
        self.spin_debug_gain.editingFinished.connect(self._on_debug_camera_param_editing_finished)

        self.cmb_debug_trigger_mode = QtWidgets.QComboBox()
        self.cmb_debug_trigger_mode.addItems(["software", "continuous"])
        self.cmb_debug_trigger_mode.setCurrentText("continuous")
        self.cmb_debug_trigger_mode.setStyleSheet(_input_style)
        cam_params_form.addRow("触发模式", self.cmb_debug_trigger_mode)
        # activated：仅用户操作时触发，避免程序 setCurrentIndex 误下发
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

        # IO 调试（对话框用）— MVS 风格布局
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

        self.btn_debug_refresh_io = QtWidgets.QPushButton(_si(SP.SP_BrowserReload), " 刷新 DI/DO 状态")
        self.btn_debug_refresh_io.setStyleSheet(_compact_btn)
        self.btn_debug_refresh_io.clicked.connect(self._refresh_debug_io_snapshot)
        io_ctrl_layout.addWidget(self.btn_debug_refresh_io)

        self.btn_debug_simulate_trigger = QtWidgets.QPushButton(" 模拟脚踏触发（待补）")
        self.btn_debug_simulate_trigger.setStyleSheet(_compact_btn)
        self.btn_debug_simulate_trigger.setEnabled(False)
        io_ctrl_layout.addWidget(self.btn_debug_simulate_trigger)

        io_left_vbox.addWidget(io_ctrl_w)
        io_left_vbox.addSpacing(4)

        io_status_title = QtWidgets.QLabel("  IO 状态")
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
        io_right_vbox.addWidget(io_output_title)

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


    # ------------------------------------------------------------------
    # 底栏同步
    # ------------------------------------------------------------------

    def _sync_footer(self) -> None:
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "未设置"
        self.lbl_footer_ref.setText(f"参考图: {ref_name}")
        algo = self.current_algorithm() if hasattr(self, "cmb_algorithm") else ""
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
    # 图片列表事件
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

    def _on_select_ok(self) -> None:
        p = self._current_selected_path()
        if p:
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    def _on_select_ng(self) -> None:
        p = self._current_selected_path()
        if p:
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    def _on_select_test(self) -> None:
        p = self._current_selected_path()
        if p:
            self._load_canvas_image(p)
            self._set_status_for_current_image(p)

    def _add_images_to(self, kind: str) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            f"选择要加入 {kind} 的图片",
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
            items = self.test_list.selectedItems()
            if not items:
                return
            idx = self.test_list.row(items[0])
            self.test_files.pop(idx)
        self._refresh_lists()
        self._save_session()

    def _on_tab_changed(self, index: int) -> None:
        if index == 0 and self.ok_files:
            self.ok_list.setCurrentRow(0)
            self._load_canvas_image(self.ok_files[0])
        elif index == 1 and self.ng_files:
            self.ng_list.setCurrentRow(0)
            self._load_canvas_image(self.ng_files[0])
        elif index == 2 and self.test_files:
            self.test_list.setCurrentRow(0)
            self._load_canvas_image(self.test_files[0])

    # ------------------------------------------------------------------
    # 产品管理
    # ------------------------------------------------------------------

    def _on_product_changed(self, product_name: str) -> None:
        if not product_name or product_name == self.session.current_product:
            return
        self._save_session()
        self.productChangeRequested.emit(product_name)

    def _new_product(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "新建产品", "请输入产品名称:")
        if not ok or not name.strip():
            return
        error = self.session.create_product(name.strip())
        if error:
            QtWidgets.QMessageBox.warning(self, "错误", error)
            return
        self.cmb_product.addItem(name.strip())
        self.cmb_product.setCurrentText(name.strip())

    def _clear_session(self) -> None:
        ret = QtWidgets.QMessageBox.question(self, "清空会话", "确定清空会话（列表/参考图/模型缓存）吗？")
        if ret != QtWidgets.QMessageBox.Yes:
            return
        self.sessionClearRequested.emit()

    # ------------------------------------------------------------------
    # Canvas / ROI
    # ------------------------------------------------------------------

    def _load_canvas_image(self, path: str) -> None:
        self.canvas.set_image(path, pixmap=_pixmap_from_path(path))
        self._load_shape_for_label(path, self._current_label())

    def _set_status_for_current_image(self, path: str) -> None:
        match_ms = self._line2dup_match_ms_by_image.get(path)
        total_ms = self._line2dup_autogen_ms_by_image.get(path)
        if match_ms is None and total_ms is None:
            return
        parts = [f"当前图：{os.path.basename(path)}"]
        if match_ms is not None:
            parts.append(f"模板匹配={match_ms:.1f}ms")
        if total_ms is not None:
            parts.append(f"生成ROI={total_ms:.1f}ms")
        self.lbl_status.setText("状态：" + "  ".join(parts))

    def _current_label(self) -> str:
        return self.cmb_label.currentText()

    def _update_save_label_text(self) -> None:
        label = self._current_label()
        self.btn_save.setText(f"保存标注({label}) -> labelme json")

    def _set_overlay_shapes(self, img_path: str, current_label: str) -> None:
        j = qr_core.labelme_json_of_image(img_path)
        overlays: List[OverlayShape] = []
        visible_roi_labels: Optional[set[str]] = None

        recipe = self.line2dup_recipe
        if recipe is None and os.path.exists(self.session.line2dup_recipe_path):
            try:
                recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
                self.line2dup_recipe = recipe
            except Exception:
                recipe = None
        if self.loc_method == "line2dup":
            labels = [str(label).strip() for label in output_labels_from_line2dup_recipe(recipe) if str(label).strip()]
            visible_roi_labels = set(labels) if labels else None

        if self.loc_method == "line2dup" and recipe is not None and recipe.search_points:
            points = [
                (float(pt[0]), float(pt[1]))
                for pt in (recipe.search_points or [])
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if len(points) >= 2:
                if str(recipe.search_shape_type or "rectangle") == "rectangle" and len(points) == 2:
                    (x0, y0), (x1, y1) = points[:2]
                    x = int(round(min(x0, x1)))
                    y = int(round(min(y0, y1)))
                    w = max(1, int(round(abs(x1 - x0))))
                    h = max(1, int(round(abs(y1 - y0))))
                    overlays.append(OverlayShape(
                        shape_type="rect", xywh=(x, y, w, h),
                        color=QtGui.QColor(0, 0, 255), width=0.5, dash=False,
                    ))
                elif len(points) >= 3:
                    overlays.append(OverlayShape(
                        shape_type="polygon", points=points,
                        color=QtGui.QColor(0, 0, 255), width=0.5, dash=False,
                    ))

        if not os.path.exists(j):
            self.canvas.set_overlays(overlays)
            return

        def add_shape(label: str, color: QtGui.QColor, *, width: int = 2, dash: bool = False) -> None:
            poly_pts = qr_core.try_read_polygon_points_from_labelme(j, label)
            if poly_pts and len(poly_pts) >= 3:
                overlays.append(OverlayShape(shape_type="polygon", points=poly_pts, color=color, width=width, dash=dash))
                return
            xywh = qr_core.try_read_xywh_from_labelme(j, label)
            if xywh:
                overlays.append(OverlayShape(shape_type="rect", xywh=xywh, color=color, width=width, dash=dash))

        seen_labels: set = set()
        for idx, label in enumerate(qr_core.sorted_label_names_from_labelme(j, label_prefix="roi")):
            if visible_roi_labels is not None and label not in visible_roi_labels:
                continue
            if label == current_label:
                continue
            seen_labels.add(label)
            add_shape(label, ROI_OVERLAY_PALETTE[idx % len(ROI_OVERLAY_PALETTE)], width=2, dash=False)

        for label, color, dash in [
            ("anchor", QtGui.QColor(0, 255, 255), True),
            ("roi", QtGui.QColor(255, 165, 0), False),
            ("anchor_mask", QtGui.QColor(255, 0, 0), True),
        ]:
            if label == current_label or label in seen_labels:
                continue
            add_shape(label, color, width=2, dash=dash)

        self.canvas.set_overlays(overlays)

    def _load_shape_for_label(self, img_path: str, label_name: str) -> None:
        self.canvas.clear_roi()
        j = qr_core.labelme_json_of_image(img_path)
        loaded = False
        if os.path.exists(j):
            poly_pts = qr_core.try_read_polygon_points_from_labelme(j, label_name)
            if poly_pts and len(poly_pts) >= 3:
                self.canvas.set_roi_polygon(poly_pts)
                self.cmb_shape.setCurrentText("polygon")
                loaded = True
            xywh = qr_core.try_read_xywh_from_labelme(j, label_name)
            if xywh:
                self.canvas.set_roi_rect(xywh)
                self.cmb_shape.setCurrentText("rect")
                loaded = True
        self._set_overlay_shapes(img_path, label_name)
        if not loaded:
            self._on_shapes_changed()

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
        self.canvas.clear_roi()
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

    def _on_shapes_changed(self) -> None:
        p = self.canvas.image_path()
        if p is None:
            self.btn_save.setEnabled(False)
            return
        st = self.canvas.roi
        ok = (st.shape_type == "rect" and st.xywh is not None) or (st.shape_type == "polygon" and st.points is not None)
        self.btn_save.setEnabled(ok)

    def _roi_xywh_from_canvas(self) -> Optional[Tuple[int, int, int, int]]:
        roi = self.canvas.roi_xywh()
        if roi is not None:
            return roi
        p = self.canvas.image_path()
        if p:
            j = qr_core.labelme_json_of_image(p)
            if os.path.exists(j):
                xywh = qr_core.try_read_xywh_from_labelme(j, "roi")
                if xywh:
                    return xywh
        return None

    def _save_current_rect(self) -> None:
        p = self.canvas.image_path()
        if p is None:
            return
        st = self.canvas.roi
        label_name = self._current_label()

        if label_name == "anchor_mask" and st.shape_type != "polygon":
            QtWidgets.QMessageBox.warning(self, "提示", "anchor_mask 只能用多边形标注（polygon）")
            return

        if st.shape_type == "rect":
            if st.xywh is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先拖拽画出矩形标注")
                return
            jpath = qr_core.upsert_labelme_rect(p, st.xywh, label_name=label_name)
        else:
            if not st.points or len(st.points) < 3:
                QtWidgets.QMessageBox.warning(self, "提示", "多边形至少需要 3 个点（左键点选加点，右键结束）")
                return
            jpath = qr_core.upsert_labelme_polygon(p, st.points, label_name=label_name)

        QtWidgets.QMessageBox.information(
            self, "已保存",
            f"已更新 labelme json：\n{jpath}\n(label={label_name}, type={st.shape_type})",
        )
        self._load_canvas_image(p)

    # ------------------------------------------------------------------
    # 参考图 / 定位
    # ------------------------------------------------------------------

    def _set_reference(self, path: str) -> None:
        self.ref_image = path
        if self.lbl_ref is not None:
            self.lbl_ref.setText(f"参考图：{os.path.basename(path)}")
            self.lbl_ref.setToolTip(path)
        try:
            recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
            recipe.reference_image = path
            recipe.model_path = self.session.line2dup_model_path
            line2dup_locator.save_recipe_for_product(self.session.product_dir, recipe)
            self.line2dup_recipe = recipe
        except Exception:
            pass
        self._save_session()

    def _set_ref_from_current(self) -> None:
        p = self.canvas.image_path()
        if not p:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在右侧打开一张图片")
            return
        self._set_reference(p)

    def _pick_ref_image(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择参考图", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not p:
            return
        self._set_reference(p)

    def _open_line2dup_template_page(self) -> None:
        from ui.debug import Line2DupTemplateDialog

        if self._template_editor_dialog is not None and self._template_editor_dialog.isVisible():
            self._template_editor_dialog.raise_()
            self._template_editor_dialog.activateWindow()
            return
        initial = self.ref_image or self.canvas.image_path() or ""
        dlg = Line2DupTemplateDialog(
            product_name=self.session.current_product,
            product_dir=self.session.product_dir,
            initial_image_path=initial,
            parent=self.window(),
        )
        dlg.setModal(False)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.modelSaved.connect(self._on_line2dup_model_saved)
        dlg.referenceRegionsChanged.connect(self._on_line2dup_reference_regions_changed)
        dlg.destroyed.connect(self._on_template_editor_dialog_destroyed)
        self._template_editor_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_template_editor_dialog_destroyed(self, *_args) -> None:
        self._template_editor_dialog = None

    def _on_line2dup_model_saved(self, model_path: str, recipe_path: str) -> None:
        try:
            self.line2dup_recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
        except Exception:
            self.line2dup_recipe = None
        self._reload_inspection_items()
        self.lbl_status.setText(f"状态：模板模型已保存 {os.path.basename(model_path)}")

    def _on_line2dup_reference_regions_changed(self) -> None:
        self._sync_line2dup_recipe_and_items()
        current_path = self.canvas.image_path()
        if current_path and os.path.exists(current_path):
            self._load_canvas_image(current_path)
        self.lbl_status.setText("状态：参考ROI已同步到运行界面")

    def _sync_line2dup_recipe_and_items(self) -> None:
        try:
            self.line2dup_recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
        except Exception:
            self.line2dup_recipe = None
        self._reload_inspection_items()

    def _update_loc_ui(self) -> None:
        method = self.loc_method
        if hasattr(self, "btn_build_shape"):
            self.btn_build_shape.setVisible(method == "shape_model")

    def _build_shape_model(self) -> None:
        if not self.ref_image or not os.path.exists(self.ref_image):
            QtWidgets.QMessageBox.warning(self, "提示", "请先设置参考图")
            return
        ref_json = qr_core.labelme_json_of_image(self.ref_image)
        if not os.path.exists(ref_json):
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少标注 json（需要 anchor + roi）")
            return
        if qr_core.try_read_xywh_from_labelme(ref_json, "anchor") is None:
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 anchor 标注")
            return
        if qr_core.try_read_xywh_from_labelme(ref_json, "roi") is None:
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 roi 标注")
            return
        try:
            model_path = qr_core.create_shape_model_from_reference(
                ref_img_path=self.ref_image,
                model_path=self.session.shape_model_path,
                anchor_label="anchor",
                anchor_mask_label="anchor_mask",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "生成模板失败", str(e))
            return
        QtWidgets.QMessageBox.information(self, "生成模板完成", f"已保存模板：\n{model_path}")

    def _on_loc_method_changed(self, method: str) -> None:
        self.loc_method = method
        self._update_loc_ui()
        self._save_session()

    # ------------------------------------------------------------------
    # 自动 ROI
    # ------------------------------------------------------------------

    def _line2dup_output_labels(self) -> List[str]:
        recipe = self.line2dup_recipe
        if recipe is None and os.path.exists(self.session.line2dup_recipe_path):
            try:
                recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
                self.line2dup_recipe = recipe
            except Exception:
                recipe = None
        return output_labels_from_line2dup_recipe(recipe)

    def _inspection_item_labels(self) -> List[str]:
        labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        labels = [str(label).strip() for label in labels if str(label).strip()]
        return labels or ["roi"]

    def _reload_inspection_items(self) -> None:
        path = self.session.inspection_items_path
        existing_items = load_inspection_items(path)
        recipe = None
        if self.loc_method == "line2dup":
            try:
                recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
            except Exception:
                recipe = None
        specs = inspection_item_specs_from_line2dup_recipe(recipe) if self.loc_method == "line2dup" else [
            {"roi_label": "roi", "display_name": "roi"}
        ]
        labels = [str(spec.get("roi_label", "")).strip() for spec in specs if str(spec.get("roi_label", "")).strip()]
        display_names_by_label = {
            str(spec.get("roi_label", "")).strip(): str(spec.get("display_name", "")).strip()
            for spec in specs
            if str(spec.get("roi_label", "")).strip()
        }
        if not labels:
            labels = self._inspection_item_labels()
        self.inspection_items = sync_items_with_labels(
            existing_items,
            labels,
            default_camera_id="cam1",
            display_names_by_label=display_names_by_label,
        )
        save_inspection_items(self.inspection_items, path)
        self.inspectionItemsChanged.emit()

    def _missing_roi_files(self, paths: List[str]) -> List[str]:
        missing: List[str] = []
        labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        for p in paths:
            j = qr_core.labelme_json_of_image(p)
            if not os.path.exists(j):
                missing.append(p)
                continue
            if any(qr_core.read_shape_from_labelme(j, label) is None for label in labels):
                missing.append(p)
        return missing

    def _resolve_autogen_targets(
        self,
        paths: List[str],
        *,
        only_missing: bool,
        silent: bool,
    ) -> List[str]:
        self._skip_empty_autogen_message = False
        if not paths:
            return []
        missing = self._missing_roi_files(paths)
        if not missing:
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "这些图片已经存在 ROI。")
                self._skip_empty_autogen_message = True
            return []

        missing_set = set(missing)
        existing = [p for p in paths if p not in missing_set]
        if not existing or silent:
            return list(missing) if only_missing else list(paths)

        default_button = (
            QtWidgets.QMessageBox.StandardButton.No
            if only_missing
            else QtWidgets.QMessageBox.StandardButton.Yes
        )
        reply = QtWidgets.QMessageBox.question(
            self, "覆盖已存在ROI？",
            (
                f"当前列表中已有 ROI 的图片有 {len(existing)} 张。\n"
                "是否覆盖并重新创建这些 ROI？\n\n"
                '选择"是"将重建整个列表；选择"否"只创建缺失 ROI；选择"取消"终止。'
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            default_button,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
            self._skip_empty_autogen_message = True
            return []
        if reply == QtWidgets.QMessageBox.StandardButton.No:
            return list(missing)
        return list(paths)

    def _autogen_roi_for_images(self, paths: List[str], only_missing: bool, silent: bool = False) -> None:
        if not paths:
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "没有可处理的图片")
            return
        ref_image = self.ref_image
        method = self.loc_method
        if method == "line2dup":
            try:
                recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
                self.line2dup_recipe = recipe
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "提示", f"无法加载模板 recipe：{exc}")
                return
            if recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
                if self.ref_image != ref_image:
                    self._set_reference(ref_image)
            if not os.path.exists(self.session.line2dup_model_path):
                QtWidgets.QMessageBox.warning(self, "提示", "当前产品还没有模板模型，请先创建模板。")
                return
            labels = self._line2dup_output_labels()
            recipe_region_labels = {
                str(region.get("output_label") or region.get("reference_label") or "").strip()
                for region in (recipe.reference_regions or [])
                if isinstance(region, dict)
            }
            recipe_region_labels.discard("")
            if (not ref_image or not os.path.exists(ref_image)) and not recipe_region_labels:
                QtWidgets.QMessageBox.warning(self, "提示", "模板定位需要参考图或已保存的参考 ROI。")
                return
            if labels:
                missing_labels = [label for label in labels if label not in recipe_region_labels]
                if missing_labels:
                    ref_json = qr_core.labelme_json_of_image(ref_image) if ref_image else ""
                    if not ref_json or not os.path.exists(ref_json):
                        QtWidgets.QMessageBox.warning(
                            self, "提示",
                            f"参考图缺少 labelme json，且 recipe 中也没有这些参考ROI：{', '.join(missing_labels)}",
                        )
                        return
                    missing_labels = [
                        label for label in missing_labels
                        if qr_core.read_shape_from_labelme(ref_json, label) is None
                    ]
                    if missing_labels:
                        QtWidgets.QMessageBox.warning(
                            self, "提示",
                            f"参考图缺少参考ROI标注：{', '.join(missing_labels)}",
                        )
                        return
        else:
            if not ref_image or not os.path.exists(ref_image):
                QtWidgets.QMessageBox.warning(self, "提示", "请先设置参考图")
                return
            ref_json = qr_core.labelme_json_of_image(ref_image)
            if not os.path.exists(ref_json):
                QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少标注 json（需要 anchor + roi）")
                return
            if qr_core.try_read_xywh_from_labelme(ref_json, "anchor") is None:
                QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 anchor 标注")
                return
            if qr_core.try_read_xywh_from_labelme(ref_json, "roi") is None:
                QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 roi 标注")
                return

        todo = self._resolve_autogen_targets(paths, only_missing=only_missing, silent=silent)
        if not todo:
            if getattr(self, "_skip_empty_autogen_message", False):
                self._skip_empty_autogen_message = False
                return
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "这些图片已存在 ROI")
            return

        ok = 0
        errs: List[str] = []
        for p in todo:
            try:
                if method == "shape_model":
                    qr_core.autogen_roi_json_from_shape_model(
                        tgt_img_path=p, ref_img_path=ref_image,
                        model_path=self.session.shape_model_path,
                        anchor_label="anchor", roi_label="roi",
                        anchor_mask_label="anchor_mask",
                    )
                elif method == "line2dup":
                    run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                        tgt_img_path=p, ref_img_path=ref_image,
                        product_dir=self.session.product_dir,
                    )
                    self._line2dup_match_ms_by_image[p] = float(run.locate_ms)
                    self._line2dup_autogen_ms_by_image[p] = float(run.total_ms)
                else:
                    qr_core.autogen_roi_json_from_reference(
                        tgt_img_path=p, ref_img_path=ref_image,
                        method=method, anchor_label="anchor", roi_label="roi",
                    )
                ok += 1
            except Exception as e:
                errs.append(f"{os.path.basename(p)}: {e}")

        if not silent:
            msg = f"自动 ROI 完成：成功 {ok} / 失败 {len(errs)}"
            if errs:
                msg += "\n\n失败示例（前10）：\n" + "\n".join(errs[:10])
            QtWidgets.QMessageBox.information(self, "完成", msg)
            if ok:
                self.lbl_status.setText(f"状态：当前列表已生成ROI，成功 {ok} 张，失败 {len(errs)} 张")

        cur = self.canvas.image_path()
        if cur and cur in todo:
            self._load_canvas_image(cur)
            self._set_status_for_current_image(cur)

    def _autogen_roi_current_tab(self) -> None:
        tab = self.tabs.currentIndex()
        if tab == 0:
            paths = list(self.ok_files)
        elif tab == 1:
            paths = list(self.ng_files)
        else:
            paths = list(self.test_files)
        self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

    def _autogen_roi_all(self) -> None:
        paths = list(self.ok_files) + list(self.ng_files) + list(self.test_files)
        self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

    def _existing_roi_like_labels(self, paths: List[str]) -> List[str]:
        labels: List[str] = []
        seen: set[str] = set()
        for path in paths:
            jpath = qr_core.labelme_json_of_image(path)
            if not os.path.exists(jpath):
                continue
            try:
                shapes = qr_core.list_shapes_from_labelme(jpath)
            except Exception:
                continue
            for shape in shapes:
                if not isinstance(shape, dict):
                    continue
                label = str(shape.get("label", "")).strip()
                if not label or label in {"anchor", "anchor_mask"} or label in seen:
                    continue
                labels.append(label)
                seen.add(label)
        return labels

    def _clear_roi_labels_for_paths(self, paths: List[str]) -> Tuple[List[str], str]:
        current_labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        prefer_stale_only = bool(
            self.loc_method == "line2dup"
            and getattr(self, "chk_only_missing", None) is not None
            and self.chk_only_missing.isChecked()
        )
        return clearable_roi_labels(
            current_labels,
            self._existing_roi_like_labels(paths),
            prefer_stale_only=prefer_stale_only,
        )

    def _clear_roi_for_images(
        self,
        paths: List[str],
        *,
        labels: Optional[List[str]] = None,
        silent: bool = False,
    ) -> None:
        if not paths:
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "没有可处理的图片")
            return
        if labels is None:
            labels, _clear_mode = self._clear_roi_labels_for_paths(paths)
        labels = [str(label).strip() for label in (labels or []) if str(label).strip()] or ["roi"]

        removed = 0
        touched = 0
        for path in paths:
            any_removed = False
            for label in labels:
                try:
                    if qr_core.delete_labelme_shape(path, label):
                        removed += 1
                        any_removed = True
                except Exception:
                    pass
            if any_removed:
                touched += 1
                self._line2dup_match_ms_by_image.pop(path, None)
                self._line2dup_autogen_ms_by_image.pop(path, None)

        cur = self.canvas.image_path()
        if cur and cur in paths:
            self._load_canvas_image(cur)
            self._set_status_for_current_image(cur)

        if not silent:
            QtWidgets.QMessageBox.information(
                self, "完成",
                f"已清空 ROI：{touched} 张图片，删除 {removed} 个标签。\n标签: {', '.join(labels)}",
            )
            self.lbl_status.setText(f"状态：已清空 ROI，图片 {touched} 张，标签 {removed} 个")

    def _clear_roi_current_tab(self) -> None:
        tab = self.tabs.currentIndex()
        if tab == 0:
            paths, tab_name = list(self.ok_files), "OK"
        elif tab == 1:
            paths, tab_name = list(self.ng_files), "NG"
        else:
            paths, tab_name = list(self.test_files), "TEST"

        if not paths:
            QtWidgets.QMessageBox.information(self, "提示", "当前列表没有图片")
            return

        labels, clear_mode = self._clear_roi_labels_for_paths(paths)
        if clear_mode == "stale_only":
            action_text = "将删除当前列表中已失效的 ROI 标签: "
        elif clear_mode == "all_existing":
            action_text = "将删除当前列表中的全部相关 ROI 标签: "
        else:
            action_text = "将删除标签: "
        reply = QtWidgets.QMessageBox.question(
            self, "清空ROI",
            f"确定清空当前 {tab_name} 列表中的 ROI 吗？\n{action_text}{', '.join(labels)}",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._clear_roi_for_images(paths, labels=labels, silent=False)

    # ------------------------------------------------------------------
    # 算法参数
    # ------------------------------------------------------------------

    def _is_embedding_algorithm(self, algorithm: Optional[str] = None) -> bool:
        return self.algo.is_embedding_algorithm(algorithm or self.current_algorithm())

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
                else SUPPORTED_ALGORITHMS[0]
            )
            score_mode = (
                self.algo.product_params.score_mode
                if self.algo.product_params.score_mode in SUPPORTED_SCORE_MODES
                else SUPPORTED_SCORE_MODES[0]
            )
            self.cmb_algorithm.setCurrentText(algorithm)
            self.cmb_mode.setCurrentText(score_mode)
            self.spin_margin.setValue(float(self.algo.product_params.margin))
            self.spin_topk.setValue(max(1, int(self.algo.product_params.topk)))
        finally:
            self._updating_runtime_params = False
        self._update_runtime_widgets()

    def _update_runtime_widgets(self) -> None:
        embedding = self._is_embedding_algorithm()
        self.cmb_mode.setEnabled(embedding)
        self.spin_topk.setEnabled(embedding and self.cmb_mode.currentText() == "topk")
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

    def _on_algorithm_changed(self, algorithm: str) -> None:
        if self._updating_runtime_params or not algorithm:
            return
        self.algo.product_params.algorithm = algorithm
        self._save_runtime_params()
        self._update_runtime_widgets()
        try:
            _, msg = self.algo.load_model_for_algorithm(algorithm, self.session.product_dir)
            self.lbl_status.setText(msg)
        except Exception as exc:
            self.algo.model = None
            self.lbl_status.setText(f"状态：加载算法 {algorithm} 失败 - {exc}")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _train(self) -> None:
        self.algo.model = None
        self.table.setRowCount(0)
        self._current_result_rows = []
        label_names = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        candidate_paths = list(self.ok_files) + list(self.ng_files)
        missing_paths = self._missing_roi_files(candidate_paths)
        if missing_paths and self.loc_method == "line2dup":
            try:
                self._autogen_roi_for_images(missing_paths, only_missing=False, silent=True)
                missing_paths = self._missing_roi_files(candidate_paths)
            except Exception:
                pass
        missing = [os.path.basename(p) for p in missing_paths]
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "缺少ROI标注",
                f"需要每张 OK/NG 图都具备这些 ROI：{', '.join(label_names)}。\n"
                "请逐张打开图片 -> 画 ROI -> 保存。\n缺少：\n" + "\n".join(missing[:50]),
            )
            return

        algorithm = self.current_algorithm()
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
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "训练失败", str(e))
            return

        self.lbl_status.setText(result.status_message)
        if not result.is_embedding and result.result_rows:
            self._populate_results_table(result.result_rows)

        self._save_runtime_params()
        self._save_session()
        self._update_runtime_widgets()
        QtWidgets.QMessageBox.information(self, "训练完成", result.dialog_message)

    # ------------------------------------------------------------------
    # 预测 / 测试
    # ------------------------------------------------------------------

    def _predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        prefer_canvas_roi: bool = False,
        labels_override: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """定位（line2dup）→ 委托 AlgorithmController 推理 → 返回结果 dict。"""
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        total_t0 = time.perf_counter()
        algorithm = self.current_algorithm()

        match_ms: Optional[float] = None
        if self.loc_method == "line2dup":
            recipe = self.line2dup_recipe
            if recipe is None and os.path.exists(self.session.line2dup_recipe_path):
                recipe = line2dup_locator.load_recipe_for_product(self.session.product_dir)
                self.line2dup_recipe = recipe
            ref_image = self.ref_image
            if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
            if ref_image and os.path.exists(ref_image):
                run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                    tgt_img_path=path,
                    ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                )
                match_ms = float(run.locate_ms)
                self._line2dup_match_ms_by_image[path] = match_ms
                self._line2dup_autogen_ms_by_image[path] = float(run.total_ms)
        elif self.ref_image and os.path.exists(self.ref_image):
            self._autogen_roi_for_images([path], only_missing=True, silent=True)

        labels = [str(label).strip() for label in (labels_override or []) if str(label).strip()]
        if not labels:
            labels = self._line2dup_output_labels() if self.loc_method == "line2dup" else ["roi"]
        roi = None
        if prefer_canvas_roi and len(labels) == 1 and self.canvas.image_path() == path:
            roi = self._roi_xywh_from_canvas()

        if self.algo.is_embedding_algorithm(algorithm):
            if self.algo.model is None or self.algo.model.backbone != algorithm:
                self.load_embedding_model(algorithm)

        result = self.algo.predict_image(
            path, labels=labels, feat_net=feat_net, roi=roi, match_ms=match_ms,
        )
        payload = result.to_dict()
        payload["infer_ms"] = (
            float(payload.get("total_ms", 0.0))
            if payload.get("total_ms") is not None
            else None
        )
        payload["total_ms"] = float((time.perf_counter() - total_t0) * 1000.0)
        return payload

    def _populate_results_table(self, rows: List[Dict[str, object]]) -> None:
        self._current_result_rows = list(rows)
        self.table.setRowCount(0)
        for row_idx, row in enumerate(rows):
            self.table.insertRow(row_idx)
            values = [
                str(row.get("file_name", "")),
                str(row.get("gt", "")),
                str(row.get("pred", "")),
                f"{float(row.get('diff', 0.0)):.4f}" if row.get("diff") is not None else "",
                f"{float(row.get('sim_ok', 0.0)):.4f}" if row.get("sim_ok") is not None else "",
                f"{float(row.get('sim_ng', 0.0)):.4f}" if row.get("sim_ng") is not None else "",
                f"{float(row.get('value', 0.0)):.4f}" if row.get("value") is not None else "",
                f"{float(row.get('threshold', 0.0)):.4f}" if row.get("threshold") is not None else "",
                f"{float(row.get('match_ms', 0.0)):.1f}" if row.get("match_ms") is not None else "",
                f"{float(row.get('total_ms', 0.0)):.1f}" if row.get("total_ms") is not None else "",
                str(row.get("json_name", "")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.UserRole, str(row.get("file_path", "")))
                gt = str(row.get("gt", ""))
                pred = str(row.get("pred", ""))
                if gt and pred and gt != pred:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(192, 32, 32)))
                self.table.setItem(row_idx, col_idx, item)

    def _run_test(self) -> None:
        p = self.canvas.image_path()
        if p is None or not os.path.exists(p):
            QtWidgets.QMessageBox.warning(self, "提示", "请先打开一张测试图片")
            return
        self.canvas.set_overlays([])

        algorithm = self.current_algorithm()
        if self._is_embedding_algorithm(algorithm):
            if self.algo.model is None or self.algo.model.backbone != algorithm:
                try:
                    self.load_embedding_model(algorithm)
                except Exception:
                    pass
            if self.algo.model is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先训练/注册（OK+NG）")
                return

        try:
            feat_net = None
            if self._is_embedding_algorithm(algorithm) and self.algo.model is not None:
                feat_net, _ = qr_core.load_backbone(self.algo.model.backbone, device=self.algo.model.device)
            row = self._predict_image(p, feat_net=feat_net, prefer_canvas_roi=True)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "测试失败", str(ex))
            return

        self._populate_results_table([row])
        log_path = self._append_test_log(row)
        metric_text = ""
        if row.get("value") is not None:
            metric_text = f"  value={float(row['value']):.4f}  threshold={float(row.get('threshold', 0.0)):.4f}"
        self.lbl_status.setText(
            "状态："
            + f"TEST={os.path.basename(p)}  pred={row['pred']}  diff={float(row['diff']):.4f}"
            + metric_text
            + (f"  模板匹配={float(row['match_ms']):.1f}ms" if row.get("match_ms") is not None else "")
            + f"  总耗时={float(row['total_ms']):.1f}ms  日志={os.path.basename(log_path)}"
        )
        self._load_canvas_image(p)

    def _daily_test_log_path(self) -> str:
        log_dir = os.path.join(self.session.product_dir, "test_logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, datetime.now().strftime("%Y%m%d") + ".csv")

    def _append_test_log(self, row: Dict[str, object]) -> str:
        csv_path = self._daily_test_log_path()
        fields = [
            "timestamp", "product", "algorithm", "score_mode", "margin", "topk",
            "file_name", "gt", "pred", "diff", "sim_ok", "sim_ng",
            "value", "threshold", "match_ms", "total_ms", "json_name",
        ]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "product": self.session.current_product,
                "algorithm": self.current_algorithm(),
                "score_mode": self.cmb_mode.currentText(),
                "margin": float(self.spin_margin.value()),
                "topk": int(self.spin_topk.value()),
                "file_name": row.get("file_name", ""),
                "gt": row.get("gt", ""),
                "pred": row.get("pred", ""),
                "diff": row.get("diff", ""),
                "sim_ok": row.get("sim_ok", ""),
                "sim_ng": row.get("sim_ng", ""),
                "value": row.get("value", ""),
                "threshold": row.get("threshold", ""),
                "match_ms": row.get("match_ms", ""),
                "total_ms": row.get("total_ms", ""),
                "json_name": row.get("json_name", ""),
            })
        return csv_path

    def _embedding_test_root(self) -> Path:
        return Path(__file__).resolve().parent

    def _selected_debug_camera_serial(self) -> str:
        return str(self.cmb_debug_camera.currentData() or "").strip()

    def _debug_camera_settings_payload_from_ui(self) -> dict[str, object]:
        return {
            "trigger_mode": str(self.cmb_debug_trigger_mode.currentText() or "continuous"),
            "exposure_time_us": float(self.spin_debug_exposure.value()),
            "gain": float(self.spin_debug_gain.value()),
        }

    def _load_saved_debug_camera_settings_to_ui(self, serial: str) -> bool:
        self._debug_camera_block_spin_apply = True
        try:
            payload = self._camera_settings_store.load_for_serial(serial)
            if not payload:
                return False
            if payload.get("exposure_time_us") is not None:
                self.spin_debug_exposure.setValue(float(payload["exposure_time_us"]))
            if payload.get("gain") is not None:
                self.spin_debug_gain.setValue(float(payload["gain"]))
            trigger_mode = str(payload.get("trigger_mode") or "").strip()
            if trigger_mode:
                self.cmb_debug_trigger_mode.setCurrentText(trigger_mode)
            return True
        finally:
            self._debug_camera_block_spin_apply = False

    def _on_debug_camera_param_editing_finished(self) -> None:
        """曝光/增益编辑结束：已连接则写入相机，否则仅写入本地缓存。"""
        if self._debug_camera_block_spin_apply:
            return
        serial = self._selected_debug_camera_serial()
        if self._debug_camera_device() is None:
            if serial:
                self._save_debug_camera_settings(
                    serial, self._debug_camera_settings_payload_from_ui()
                )
            return
        self._apply_debug_camera_settings(quiet=True)

    def _on_debug_camera_trigger_activated(self, _index: int) -> None:
        if self._debug_camera_block_spin_apply:
            return
        serial = self._selected_debug_camera_serial()
        if self._debug_camera_device() is None:
            if serial:
                self._save_debug_camera_settings(
                    serial, self._debug_camera_settings_payload_from_ui()
                )
            return
        self._apply_debug_camera_settings(quiet=True)

    def _save_debug_camera_settings(self, serial: str, settings: dict[str, object]) -> None:
        serial_text = str(serial).strip()
        if not serial_text:
            return
        self._camera_settings_store.save_for_serial(serial_text, settings)

    def _set_debug_preview_placeholder(self, text: str) -> None:
        self.view_debug_camera.set_runtime_pixmap(None, placeholder=text)

    def _show_debug_preview_image(self, image: QtGui.QImage) -> None:
        self.view_debug_camera.set_runtime_pixmap(QtGui.QPixmap.fromImage(image))

    def _save_debug_camera_image(self) -> None:
        pixmap = self.view_debug_camera._pixmap
        if pixmap is None or pixmap.isNull():
            self.lbl_debug_camera_status.setText("相机状态：没有可保存的图片")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存图片", "", "BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件 (*)"
        )
        if path:
            if pixmap.save(path):
                self.lbl_debug_camera_status.setText(f"相机状态：图片已保存到 {path}")
            else:
                self.lbl_debug_camera_status.setText("相机状态：保存图片失败")

    def _set_debug_preview_running(self, running: bool) -> None:
        if not hasattr(self, "btn_debug_live_preview"):
            return
        self.btn_debug_live_preview.blockSignals(True)
        self.btn_debug_live_preview.setChecked(running)
        self.btn_debug_live_preview.setText("停止预览" if running else "实时预览")
        self.btn_debug_live_preview.blockSignals(False)

    def _toggle_debug_camera_preview(self, checked: bool) -> None:
        if checked:
            self._start_debug_camera_preview()
            return
        self._stop_debug_camera_preview()

    def _start_debug_camera_preview(self) -> None:
        if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
            self._set_debug_preview_running(False)
            QtWidgets.QMessageBox.information(self, "相机调试", "请先连接调试相机")
            return
        self._stop_debug_camera_preview()
        worker = _DebugCameraPreviewThread(self._debug_frame_grab_service, self)
        worker.frameReady.connect(self._on_debug_preview_frame_ready)
        worker.errorOccurred.connect(self._on_debug_preview_error)
        worker.finished.connect(self._on_debug_preview_finished)
        self._debug_preview_thread = worker
        self._set_debug_preview_running(True)
        self._set_debug_preview_placeholder("实时预览启动中...")
        worker.start()
        self._set_debug_camera_status("实时预览中")

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
            self._set_debug_preview_placeholder("未开启实时预览")

    @QtCore.Slot(QtGui.QImage)
    def _on_debug_preview_frame_ready(self, image: QtGui.QImage) -> None:
        self._show_debug_preview_image(image)

    @QtCore.Slot(str)
    def _on_debug_preview_error(self, message: str) -> None:
        self._set_debug_camera_status(f"实时预览异常：{message}")

    @QtCore.Slot()
    def _on_debug_preview_finished(self) -> None:
        if self._debug_preview_thread is not None:
            return
        self._set_debug_preview_running(False)

    def _set_debug_camera_status(self, message: str) -> None:
        self.lbl_debug_camera_status.setText(f"相机状态：{message}")

    def _ensure_debug_camera_services(self) -> bool:
        if FrameGrabService is None or HikCameraManager is None or HikCameraSettings is None:
            QtWidgets.QMessageBox.warning(self, "相机调试", "当前环境未启用海康相机调试服务")
            self._set_debug_camera_status("服务不可用")
            return False
        if self._debug_camera_manager is None:
            self._debug_camera_manager = HikCameraManager()
            self._debug_frame_grab_service = FrameGrabService(self._debug_camera_manager)
        return True

    def _refresh_debug_camera_list(self) -> None:
        if not self._ensure_debug_camera_services():
            return
        current_serial = str(self.cmb_debug_camera.currentData() or "")
        try:
            infos = self._debug_camera_manager.enumerate_cameras()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "相机调试", f"扫描相机失败：{exc}")
            self._set_debug_camera_status(f"扫描失败：{exc}")
            return
        self._debug_camera_infos = list(infos)
        self.cmb_debug_camera.clear()
        for info in infos:
            label = f"{info.serial_number} / {info.model_name or 'UnknownModel'} / {info.transport_layer}"
            self.cmb_debug_camera.addItem(label, info.serial_number)
        if current_serial:
            index = self.cmb_debug_camera.findData(current_serial)
            if index >= 0:
                self.cmb_debug_camera.setCurrentIndex(index)
        self._refresh_debug_camera_info()
        self._set_debug_camera_status(f"已扫描到 {len(infos)} 台相机")

    def _on_debug_camera_selected(self) -> None:
        self._refresh_debug_camera_info()
        self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())

    def _selected_debug_camera_info(self):
        serial = str(self.cmb_debug_camera.currentData() or "").strip()
        for info in self._debug_camera_infos:
            if str(getattr(info, "serial_number", "")) == serial:
                return info
        return None

    def _refresh_debug_camera_info(self) -> None:
        info = self._selected_debug_camera_info()
        if info is None:
            self.lbl_debug_camera_info.setText("相机信息：-")
            return
        self.lbl_debug_camera_info.setText(
            "相机信息："
            + f"序列号={info.serial_number}  "
            + f"型号={info.model_name or '-'}  "
            + f"名称={info.user_defined_name or '-'}  "
            + f"厂商={info.manufacturer_name or '-'}  "
            + f"传输层={info.transport_layer}"
        )

    def _connect_debug_camera(self) -> None:
        if not self._ensure_debug_camera_services():
            return
        serial = self._selected_debug_camera_serial()
        if not serial:
            QtWidgets.QMessageBox.information(self, "相机调试", "请先扫描并选择一台相机")
            return
        self.debugCameraConnectRequested.emit(serial)
        try:
            saved_settings = self._camera_settings_store.load_for_serial(serial)
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
            QtWidgets.QMessageBox.critical(self, "相机调试", f"连接调试相机失败：{exc}")
            self._set_debug_camera_status(f"连接失败：{exc}")
            return
        self._refresh_debug_camera_settings()
        self._set_debug_preview_placeholder("已连接调试相机，可开启实时预览")
        self._set_debug_camera_status(f"已连接调试相机：{serial}")

        self.debugCameraConnected.emit(serial)

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
        self.lbl_debug_camera_info.setText("相机信息：-")
        self._set_debug_preview_placeholder("未连接调试相机")
        self._set_debug_camera_status("已断开")

    def _debug_camera_device(self):
        if self._debug_frame_grab_service is None:
            return None
        try:
            return self._debug_frame_grab_service.get_device("debug")
        except Exception:
            return None

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
            self.spin_debug_exposure.setValue(exposure)
            self.spin_debug_gain.setValue(gain)
            try:
                trigger_mode_int = int(device.get_int_value("TriggerMode"))
                self.cmb_debug_trigger_mode.setCurrentText(
                    "software" if trigger_mode_int else "continuous"
                )
            except Exception:
                pass
            self._save_debug_camera_settings(
                getattr(device, "serial_number", self._selected_debug_camera_serial()),
                self._debug_camera_settings_payload_from_ui(),
            )
        finally:
            self._debug_camera_block_spin_apply = False

    def _apply_debug_camera_settings(self, *, quiet: bool = False) -> None:
        device = self._debug_camera_device()
        if device is None:
            if not quiet:
                QtWidgets.QMessageBox.information(self, "相机调试", "请先连接调试相机")
            return
        try:
            device.apply_settings(
                HikCameraSettings(
                    trigger_mode=str(self.cmb_debug_trigger_mode.currentText() or "continuous"),
                    exposure_time_us=float(self.spin_debug_exposure.value()),
                    gain=float(self.spin_debug_gain.value()),
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "相机调试", f"应用相机参数失败：{exc}")
            return
        self._save_debug_camera_settings(
            getattr(device, "serial_number", self._selected_debug_camera_serial()),
            self._debug_camera_settings_payload_from_ui(),
        )
        if not quiet:
            self.cameraSettingsApplied.emit(
                str(getattr(device, "serial_number", self._selected_debug_camera_serial()) or "").strip(),
                dict(self._debug_camera_settings_payload_from_ui()),
            )
        self._refresh_debug_camera_settings()
        self._set_debug_camera_status(
            "参数已写入相机" if quiet else "相机参数已应用"
        )

    def _grab_debug_camera_once(self) -> None:
        if self._debug_frame_grab_service is None or "debug" not in self._debug_frame_grab_service.roles():
            QtWidgets.QMessageBox.information(self, "相机调试", "请先连接调试相机")
            return
        try:
            frame = self._debug_frame_grab_service.capture_once("debug", timeout_ms=1000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "相机调试", f"拍照失败：{exc}")
            self._set_debug_camera_status(f"拍照失败：{exc}")
            return
        try:
            self._show_debug_preview_image(_qimage_from_hik_frame(frame))
        except Exception:
            pass

        capture_dir = os.path.join(self.session.product_dir, "debug_capture")
        os.makedirs(capture_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(capture_dir, f"debug_cam_{stamp}.png")
        if frame_to_bgr_image is None:
            QtWidgets.QMessageBox.critical(self, "鐩告満璋冭瘯", "鐩告満褰╄壊杞崲鏈嶅姟涓嶅彲鐢?")
            return
        image = frame_to_bgr_image(frame)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if not cv2.imwrite(image_path, image):
            QtWidgets.QMessageBox.critical(self, "相机调试", f"保存采图失败：{image_path}")
            return
        if image_path not in self.test_files:
            self.test_files.append(image_path)
            self._refresh_lists()
            self._save_session()
        self.tabs.setCurrentIndex(2)
        self._load_canvas_image(image_path)
        self._set_debug_camera_status(f"拍照成功：{os.path.basename(image_path)}")
        self.lbl_status.setText(f"状态：调试拍照已保存到 TEST -> {os.path.basename(image_path)}")

    def _default_io_mapping_path(self) -> str:
        return str(self._embedding_test_root() / "config" / "defaults" / "io_mapping.json")

    def _find_debug_nkio_config_path(self) -> Optional[str]:
        root = self._embedding_test_root().parent
        candidates = [
            root / "NKDIOLC_SDK" / "ConfigFile" / "J1900" / "NP-6133-16I16O" / "nkio_config.ini",
            root / "NKDIOLC_SDK" / "ConfigFile" / "NP-6133-16I16O" / "nkio_config.ini",
            root / "NKDIOLC_SDK" / "Bin" / "NP-61x0-16I16O" / "nkio_config.ini",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _open_debug_io(self) -> None:
        if IoController is None:
            QtWidgets.QMessageBox.warning(self, "DI/DO 调试", "当前环境未启用 IO 调试服务")
            return
        mapping_path = self._default_io_mapping_path()
        board_config = self._find_debug_nkio_config_path()
        if not board_config:
            QtWidgets.QMessageBox.warning(self, "DI/DO 调试", "未找到 nkio_config.ini")
            return
        try:
            self._close_debug_io(silent=True)
            self._debug_io_controller = IoController.from_config_file(board_config, mapping_path)
            self._debug_io_controller.open()
        except Exception as exc:
            self._debug_io_controller = None
            QtWidgets.QMessageBox.critical(self, "DI/DO 调试", f"打开 IO 调试失败：{exc}")
            return
        self._debug_io_timer.start()
        self._refresh_debug_io_snapshot()
        self.lbl_status.setText("状态：DI/DO 调试已打开")

    def _close_debug_io(self, *, silent: bool = False) -> None:
        self._debug_io_timer.stop()
        if self._debug_io_controller is not None:
            try:
                self._debug_io_controller.clear_outputs()
            except Exception:
                pass
            try:
                self._debug_io_controller.close()
            except Exception:
                pass
        self._debug_io_controller = None
        self.lbl_debug_di_snapshot.setText("DI：未连接")
        self.lbl_debug_do_snapshot.setText("DO：未连接")
        for button in self._debug_output_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        if not silent:
            self.lbl_status.setText("状态：DI/DO 调试已关闭")

    def _refresh_debug_io_snapshot(self) -> None:
        if self._debug_io_controller is None or not self._debug_io_controller.is_open:
            return
        try:
            inputs = self._debug_io_controller.snapshot_inputs()
            outputs = self._debug_io_controller.snapshot_outputs()
        except Exception as exc:
            self.lbl_debug_di_snapshot.setText(f"DI：读取失败 ({exc})")
            self.lbl_debug_do_snapshot.setText(f"DO：读取失败 ({exc})")
            return
        self.lbl_debug_di_snapshot.setText(
            "DI：" + "  ".join(f"{name}={'ON' if state else 'OFF'}" for name, state in sorted(inputs.items()))
        )
        self.lbl_debug_do_snapshot.setText(
            "DO：" + "  ".join(f"{name}={'ON' if state else 'OFF'}" for name, state in sorted(outputs.items()))
        )
        for name, button in self._debug_output_buttons.items():
            state = bool(outputs.get(name, False))
            button.blockSignals(True)
            button.setChecked(state)
            button.blockSignals(False)

    def _set_debug_output(self, name: str, on: bool) -> None:
        if self._debug_io_controller is None or not self._debug_io_controller.is_open:
            QtWidgets.QMessageBox.information(self, "DI/DO 调试", "请先打开 IO 调试")
            button = self._debug_output_buttons.get(name)
            if button is not None:
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            return
        try:
            self._debug_io_controller.set_output(name, on)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "DI/DO 调试", f"{name} 输出失败：{exc}")
        self._refresh_debug_io_snapshot()

    def _cleanup_debug_hardware(self) -> None:
        try:
            self._stop_debug_camera_preview()
        except Exception:
            pass
        try:
            self._close_debug_io(silent=True)
        except Exception:
            pass
        try:
            self._disconnect_debug_camera()
        except Exception:
            pass

    def _summarize_test_rows(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        labeled_rows = [row for row in rows if str(row.get("gt", "")) in {"OK", "NG"}]
        matched_rows = [
            row for row in labeled_rows
            if str(row.get("gt", "")) == str(row.get("pred", ""))
        ]
        total_ms_values = [
            float(row["total_ms"])
            for row in rows
            if row.get("total_ms") is not None
        ]
        return {
            "row_count": len(rows),
            "labeled_count": len(labeled_rows),
            "matched_count": len(matched_rows),
            "pred_ok_count": sum(1 for row in rows if str(row.get("pred", "")) == "OK"),
            "pred_ng_count": sum(1 for row in rows if str(row.get("pred", "")) == "NG"),
            "accuracy": (
                float(len(matched_rows)) / float(len(labeled_rows))
                if labeled_rows else None
            ),
            "avg_total_ms": (
                sum(total_ms_values) / float(len(total_ms_values))
                if total_ms_values else None
            ),
        }

    def _write_test_rows_csv(self, csv_path: str, rows: List[Dict[str, object]]) -> None:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file", "file_path", "gt", "pred", "status", "diff", "sim_ok", "sim_ng",
                "value", "threshold", "match_ms", "total_ms", "json",
            ])
            for row in rows:
                gt = str(row.get("gt", ""))
                pred = str(row.get("pred", ""))
                status = ""
                if gt in {"OK", "NG"} and pred:
                    status = "PASS" if gt == pred else "FAIL"
                writer.writerow([
                    row.get("file_name", ""),
                    row.get("file_path", ""),
                    gt,
                    pred,
                    status,
                    row.get("diff", ""),
                    row.get("sim_ok", ""),
                    row.get("sim_ng", ""),
                    row.get("value", ""),
                    row.get("threshold", ""),
                    row.get("match_ms", ""),
                    row.get("total_ms", ""),
                    row.get("json_name", ""),
                ])

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
            # 每次显示都关一次：防止 Enter 触发「读取」等按钮，把 spin 里未下发的值用相机当前值覆盖
            for _btn in dialog.findChildren(QtWidgets.QPushButton):
                _btn.setAutoDefault(False)
                _btn.setDefault(False)
        widget.show()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _save_test_result_report(
        self,
        rows: List[Dict[str, object]],
        *,
        report_prefix: str,
        summary: Dict[str, object] | None = None,
    ) -> Tuple[str, str]:
        report_dir = os.path.join(self.session.product_dir, "test_exports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{report_prefix}_{self.current_algorithm()}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {
            "product": self.session.current_product,
            "algorithm": self.current_algorithm(),
            "score_mode": self.cmb_mode.currentText(),
            "margin": float(self.spin_margin.value()),
            "topk": int(self.spin_topk.value()),
            "loc_method": self.loc_method,
            "summary": summary or self._summarize_test_rows(rows),
            "rows": rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._write_test_rows_csv(csv_path, rows)
        return json_path, csv_path

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

    def _suggest_margin_from_rows(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        labeled = [row for row in rows if str(row.get("gt", "")) in {"OK", "NG"} and row.get("diff") is not None]
        if not labeled:
            raise RuntimeError("no labeled rows for margin suggestion")

        current_margin = float(self.spin_margin.value())
        diffs = sorted({float(row["diff"]) for row in labeled})
        candidates: List[float] = []
        if diffs:
            candidates.append(diffs[0] - 1e-6)
            candidates.extend(diffs)
            candidates.extend((a + b) * 0.5 for a, b in zip(diffs, diffs[1:]))
            candidates.append(diffs[-1] + 1e-6)

        def _accuracy(threshold: float) -> Tuple[float, int, int, int, int]:
            tp = tn = fp = fn = 0
            for row in labeled:
                gt = str(row["gt"])
                pred = "OK" if float(row["diff"]) >= threshold else "NG"
                if gt == "OK" and pred == "OK":
                    tp += 1
                elif gt == "OK" and pred == "NG":
                    fn += 1
                elif gt == "NG" and pred == "NG":
                    tn += 1
                else:
                    fp += 1
            acc = float(tp + tn) / float(len(labeled))
            return acc, tp, tn, fp, fn

        best_margin = current_margin
        best_acc = -1.0
        best_conf = (0, 0, 0, 0)
        for candidate in candidates:
            acc, tp, tn, fp, fn = _accuracy(candidate)
            if acc > best_acc + 1e-12 or (
                abs(acc - best_acc) <= 1e-12
                and abs(candidate - current_margin) < abs(best_margin - current_margin)
            ):
                best_acc = acc
                best_margin = float(candidate)
                best_conf = (tp, tn, fp, fn)

        current_acc, current_tp, current_tn, current_fp, current_fn = _accuracy(current_margin)
        ok_diffs = [float(row["diff"]) for row in labeled if str(row["gt"]) == "OK"]
        ng_diffs = [float(row["diff"]) for row in labeled if str(row["gt"]) == "NG"]
        safe_range = None
        if ok_diffs and ng_diffs:
            lower = max(ng_diffs)
            upper = min(ok_diffs)
            if lower < upper:
                safe_range = (float(lower), float(upper))
                best_margin = float((lower + upper) * 0.5)
                best_acc, *conf = _accuracy(best_margin)
                best_conf = tuple(conf)

        return {
            "current_margin": current_margin,
            "current_accuracy": float(current_acc),
            "current_confusion": {
                "tp_ok": current_tp, "tn_ng": current_tn,
                "fp_ok_as_ng": current_fn, "fp_ng_as_ok": current_fp,
            },
            "suggested_margin": float(best_margin),
            "suggested_accuracy": float(best_acc),
            "suggested_confusion": {
                "tp_ok": best_conf[0], "tn_ng": best_conf[1],
                "fp_ng_as_ok": best_conf[2], "fp_ok_as_ng": best_conf[3],
            },
            "ok_diff_min": float(min(ok_diffs)) if ok_diffs else None,
            "ok_diff_max": float(max(ok_diffs)) if ok_diffs else None,
            "ng_diff_min": float(min(ng_diffs)) if ng_diffs else None,
            "ng_diff_max": float(max(ng_diffs)) if ng_diffs else None,
            "safe_range": safe_range,
        }

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
        if not self._is_embedding_algorithm():
            QtWidgets.QMessageBox.information(self, "提示", "传统算法不支持 Margin 建议，请切回嵌入式算法")
            return
        if self.algo.model is None or self.algo.model.backbone != self.current_algorithm():
            try:
                self.load_embedding_model(self.current_algorithm())
            except Exception:
                pass
        if self.algo.model is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先训练/注册（OK+NG）")
            return
        if not self.ok_files or not self.ng_files:
            QtWidgets.QMessageBox.warning(self, "提示", "需要至少一批 OK 和 NG 图片才能建议 margin。")
            return

        feat_net, _ = qr_core.load_backbone(self.algo.model.backbone, device=self.algo.model.device)
        rows: List[Dict[str, object]] = []
        try:
            for path in self.ok_files:
                row = self._predict_image(path, feat_net=feat_net, prefer_canvas_roi=False)
                row["gt"] = "OK"
                rows.append(row)
            for path in self.ng_files:
                row = self._predict_image(path, feat_net=feat_net, prefer_canvas_roi=False)
                row["gt"] = "NG"
                rows.append(row)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "验证失败", str(ex))
            return

        self._populate_results_table(rows)
        summary = self._suggest_margin_from_rows(rows)
        json_path, csv_path = self._save_margin_report(rows, summary)

        safe_range = summary.get("safe_range")
        safe_text = ""
        if isinstance(safe_range, tuple):
            safe_text = f"\n安全区间: {safe_range[0]:.4f} ~ {safe_range[1]:.4f}"

        self.lbl_status.setText(
            "状态："
            + f"当前margin={summary['current_margin']:.4f} acc={summary['current_accuracy']:.4f}  "
            + f"建议margin={summary['suggested_margin']:.4f} acc={summary['suggested_accuracy']:.4f}"
        )
        QtWidgets.QMessageBox.information(
            self, "Margin 建议",
            f"当前 margin: {summary['current_margin']:.4f}\n"
            f"当前准确率: {summary['current_accuracy']:.4f}\n"
            f"建议 margin: {summary['suggested_margin']:.4f}\n"
            f"建议准确率: {summary['suggested_accuracy']:.4f}"
            + safe_text
            + f"\n\n报告已保存:\n{json_path}\n{csv_path}",
        )

    def _open_embedding_analysis_dialog(self) -> None:
        from ui.debug import EmbeddingAnalysisDialog

        if not self._is_embedding_algorithm():
            QtWidgets.QMessageBox.information(self, "提示", "传统算法没有 embedding 可视化，请切回嵌入式算法")
            return
        dialog = EmbeddingAnalysisDialog(
            session_root=self.session.session_dir,
            initial_product=self.session.current_product,
            initial_backbone=self.current_algorithm(),
            parent=self,
        )
        dialog.exec()

    # ------------------------------------------------------------------
    # 传统基线调试
    # ------------------------------------------------------------------

    def _current_tab_paths_and_name(self) -> Tuple[List[str], str]:
        tab = self.tabs.currentIndex()
        if tab == 0:
            return list(self.ok_files), "OK"
        if tab == 1:
            return list(self.ng_files), "NG"
        return list(self.test_files), "TEST"

    def _load_roi_mask_crop(self, img_path: str, preferred_label: str = "roi1") -> Dict[str, object]:
        jpath = qr_core.labelme_json_of_image(img_path)
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"缺少 labelme json: {jpath}")

        label_name = preferred_label
        shape = qr_core.read_shape_from_labelme(jpath, preferred_label)
        if shape is None:
            label_name = "roi"
            shape = qr_core.read_shape_from_labelme(jpath, label_name)
        if shape is None:
            raise RuntimeError(f"{os.path.basename(img_path)} 缺少 {preferred_label}/roi 标注")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        h_img, w_img = img_bgr.shape[:2]

        pts = np.asarray(shape.get("points", []), dtype=np.float32)
        if pts.size == 0:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI points empty")
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        x = max(0, int(np.floor(float(x_min))))
        y = max(0, int(np.floor(float(y_min))))
        x2 = min(w_img, int(np.ceil(float(x_max))))
        y2 = min(h_img, int(np.ceil(float(y_max))))
        if x2 <= x or y2 <= y:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI bbox invalid")

        crop_bgr = img_bgr[y:y2, x:x2].copy()
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
        rel_pts = pts - np.array([[x, y]], dtype=np.float32)
        if str(shape.get("shape_type", "rectangle")) == "polygon" and len(rel_pts) >= 3:
            cv2.fillPoly(mask, [np.round(rel_pts).astype(np.int32)], 255)
        else:
            p0 = rel_pts.min(axis=0)
            p1 = rel_pts.max(axis=0)
            rx = max(0, int(np.floor(float(p0[0]))))
            ry = max(0, int(np.floor(float(p0[1]))))
            rx2 = min(mask.shape[1], int(np.ceil(float(p1[0]))))
            ry2 = min(mask.shape[0], int(np.ceil(float(p1[1]))))
            mask[ry:ry2, rx:rx2] = 255
        return {
            "label_name": label_name,
            "crop_bgr": crop_bgr,
            "crop_gray": crop_gray,
            "mask": mask,
            "bbox_xywh": (x, y, x2 - x, y2 - y),
        }

    def _compute_traditional_baseline_metrics(self, img_path: str, preferred_label: str = "roi1") -> Dict[str, object]:
        roi = self._load_roi_mask_crop(img_path, preferred_label=preferred_label)
        crop_gray = np.asarray(roi["crop_gray"], dtype=np.float32)
        crop_bgr = np.asarray(roi["crop_bgr"], dtype=np.uint8)
        mask = np.asarray(roi["mask"], dtype=np.uint8)
        valid_gray = crop_gray[mask > 0]
        if valid_gray.size == 0:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI valid pixels empty")
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        valid_hsv = hsv[mask > 0]
        if valid_hsv.size == 0:
            raise RuntimeError(f"{os.path.basename(img_path)} ROI HSV pixels empty")
        valid_hsv = np.asarray(valid_hsv, dtype=np.float32)
        h_vals = valid_hsv[:, 0]
        s_vals = valid_hsv[:, 1]
        v_vals = valid_hsv[:, 2]
        return {
            "file_path": img_path,
            "file_name": os.path.basename(img_path),
            "roi_label": str(roi["label_name"]),
            "bbox_xywh": list(roi["bbox_xywh"]),
            "mean_intensity": float(np.mean(valid_gray)),
            "hsv_h_mean": float(np.mean(h_vals)),
            "hsv_h_std": float(np.std(h_vals)),
            "hsv_s_mean": float(np.mean(s_vals)),
            "hsv_s_std": float(np.std(s_vals)),
            "hsv_v_mean": float(np.mean(v_vals)),
            "hsv_v_std": float(np.std(v_vals)),
            "roi_area": int(valid_gray.size),
        }

    def _save_traditional_baseline_report(
        self, rows: List[Dict[str, object]], tab_name: str
    ) -> Tuple[str, str]:
        report_dir = os.path.join(self.session.product_dir, "traditional_baseline_reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"baseline_roi1_hsv_{tab_name.lower()}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {"product": self.session.current_product, "tab": tab_name, "rows": rows}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file", "roi_label", "bbox_xywh", "mean_intensity",
                "hsv_h_mean", "hsv_h_std", "hsv_s_mean", "hsv_s_std",
                "hsv_v_mean", "hsv_v_std", "roi_area", "error",
            ])
            for row in rows:
                writer.writerow([
                    row.get("file_name", ""), row.get("roi_label", ""),
                    row.get("bbox_xywh", ""), row.get("mean_intensity", ""),
                    row.get("hsv_h_mean", ""), row.get("hsv_h_std", ""),
                    row.get("hsv_s_mean", ""), row.get("hsv_s_std", ""),
                    row.get("hsv_v_mean", ""), row.get("hsv_v_std", ""),
                    row.get("roi_area", ""), row.get("error", ""),
                ])
        return json_path, csv_path

    def _run_traditional_baseline_debug(self) -> None:
        paths, tab_name = self._current_tab_paths_and_name()
        if not paths:
            QtWidgets.QMessageBox.information(self, "提示", "当前列表没有图片")
            return

        rows: List[Dict[str, object]] = []
        ok = 0
        for path in paths:
            try:
                row = self._compute_traditional_baseline_metrics(path, preferred_label="roi1")
                ok += 1
            except Exception as exc:
                row = {
                    "file_path": path, "file_name": os.path.basename(path),
                    "roi_label": "", "bbox_xywh": "", "mean_intensity": "",
                    "hsv_h_mean": "", "hsv_h_std": "", "hsv_s_mean": "", "hsv_s_std": "",
                    "hsv_v_mean": "", "hsv_v_std": "", "roi_area": "", "error": str(exc),
                }
            rows.append(row)

        json_path, csv_path = self._save_traditional_baseline_report(rows, tab_name=tab_name)
        self.lbl_status.setText(f"状态：传统基线调试已完成，成功 {ok}/{len(paths)}，结果已保存")
        QtWidgets.QMessageBox.information(
            self, "传统基线调试",
            f"已完成当前 {tab_name} 列表的 ROI1/ROI 指标计算。\n"
            f"成功: {ok}/{len(paths)}\n\nJSON:\n{json_path}\n\nCSV:\n{csv_path}",
        )
