
from __future__ import annotations

import csv
import re
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from common.camera_roles import (
    CAMERA_ROLES,
    DEFAULT_CAMERA_ROLE,
    camera_index_for_role,
    configured_camera_roles,
    normalize_camera_role,
)
from algorithms.measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    CENTER_DISTANCE_ALGORITHMS,
    FIND_LINE_ALGORITHMS,
    LINE_DISTANCE_ALGORITHMS,
)
from ui.i18n import tr, tr_runtime_state, tr_status_text
from ui.roi_overlay_colors import is_roi_label, merge_roi_statuses


_DARK_BG = "#2d2d2d"
_PANEL_BG = "#363636"
_HEADER_BG = "#3a3a3a"
_TEXT_LIGHT = "#e0e0e0"
_TEXT_DIM = "#888888"
_OK_GREEN = "#379b37"
_NG_RED = "#dc1e1e"
_PASS_YELLOW = "#2f8f46"
_PENDING_GRAY = "#666666"
_RUNNING_YELLOW = "#eab308"
_RUNTIME_ROI_RECORD_COLUMN_RE = re.compile(r"^cam\d+\..+$", re.IGNORECASE)

_DEFAULT_CAMERA_LAYOUT = "two_top_one_bottom"
_CAMERA_LAYOUT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("two_top_one_bottom", "runtime.layout.two_top_one_bottom"),
    ("main_left", "runtime.layout.main_left"),
    ("main_right", "runtime.layout.main_right"),
    ("one_top_two_bottom", "runtime.layout.one_top_two_bottom"),
    ("row", "runtime.layout.row"),
)
_TEMPLATE_MATCH_FAILURE_MARKERS = (
    "match failure",
    "match failed",
    "matching failed",
    "did not find any match",
    "no match",
    "匹配失败",
    "未匹配",
)


def _runtime_record_row_result(row: dict[str, str]) -> str:
    legacy_result = str(row.get("final_result", "") or "").strip().upper()
    if legacy_result in {"OK", "NG"}:
        return legacy_result

    roi_results = [
        str(value or "").strip().upper()
        for key, value in row.items()
        if _RUNTIME_ROI_RECORD_COLUMN_RE.match(str(key or ""))
    ]
    if "NG" in roi_results:
        return "NG"
    if "OK" in roi_results:
        return "OK"
    return ""


def _camera_title(camera_id: str) -> str:
    role = normalize_camera_role(camera_id, default=DEFAULT_CAMERA_ROLE)
    if role == "cam1":
        return tr("runtime.camera1")
    if role == "cam2":
        return tr("runtime.camera2")
    if role == "cam3":
        return tr("runtime.camera3")
    index = camera_index_for_role(role)
    return f"Cam{index}" if index else str(camera_id or DEFAULT_CAMERA_ROLE)


def _status_badge_width(label: QtWidgets.QLabel, text: str, *, minimum: int = 56, maximum: int = 160) -> int:
    metrics = label.fontMetrics()
    width = metrics.horizontalAdvance(str(text or "")) + 24
    return max(minimum, min(maximum, width))

class RuntimeImageView(QtWidgets.QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._pixmap: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_DIM};font-size:14px;")

    def set_runtime_pixmap(self, pixmap: QtGui.QPixmap | None, *, placeholder: str | None = None) -> None:
        self._pixmap = pixmap
        if pixmap is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText(placeholder or tr("runtime.waiting_image"))
            return
        self._refresh_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class _ElidedLabel(QtWidgets.QLabel):
    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("", parent)
        self._full_text = ""
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full_text = str(text or "")
        super().setText(self._elided_text())
        self.setToolTip(self._full_text)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        super().setText(self._elided_text())

    def _elided_text(self) -> str:
        width = max(0, self.contentsRect().width())
        if width <= 0:
            return self._full_text
        return self.fontMetrics().elidedText(self._full_text, QtCore.Qt.ElideRight, width)


class _ItemIndicator(QtWidgets.QFrame):
    def __init__(self, index: int, name: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._status_kind = ""
        self._status_text = ""
        self._display_text = ""
        self.setStyleSheet(
            f"_ItemIndicator{{background:{_PANEL_BG};border-bottom:1px solid #404040;}}"
        )
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.lbl_index = QtWidgets.QLabel(f"{index:02d}")
        self.lbl_index.setFixedWidth(24)
        self.lbl_index.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;font-weight:bold;")
        self.lbl_index.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_index)

        self.lbl_name = _ElidedLabel(name)
        self.lbl_name.setMinimumWidth(0)
        self.lbl_name.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.lbl_name.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:14px;")
        layout.addWidget(self.lbl_name, 1)

        self.lbl_result = QtWidgets.QLabel("")
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result.setFixedSize(56, 32)
        self.lbl_result.setStyleSheet(
            f"background:{_PENDING_GRAY};color:white;font-size:14px;font-weight:bold;"
            "border-radius:4px;"
        )
        layout.addWidget(self.lbl_result)

    def set_result(self, status_kind: str, status_text: str) -> None:
        normalized_kind = str(status_kind or "pending").strip().lower() or "pending"
        normalized_text = str(status_text or "")
        display = tr_status_text(normalized_text.split("(")[0].strip()) if normalized_text else ""
        if (
            normalized_kind == self._status_kind
            and normalized_text == self._status_text
            and display == self._display_text
        ):
            return
        self._status_kind = normalized_kind
        self._status_text = normalized_text
        self._display_text = display
        color_map = {
            "ok": _OK_GREEN,
            "ng": _NG_RED,
            "pass": _PASS_YELLOW,
            "pending": _PENDING_GRAY,
            "running": _RUNNING_YELLOW,
            "measured": "#2563eb",
            "disabled": "#444444",
            "inactive": "#444444",
        }
        bg = color_map.get(normalized_kind, _PENDING_GRAY)
        self.lbl_result.setFixedWidth(_status_badge_width(self.lbl_result, display, maximum=180 if normalized_kind == "measured" else 140))
        self.lbl_result.setText(display)
        self.lbl_result.setToolTip(str(normalized_text or display or ""))
        self.lbl_result.setStyleSheet(
            f"background:{bg};color:white;font-size:14px;font-weight:bold;border-radius:4px;"
        )


class _CameraSectionHeader(QtWidgets.QFrame):
    def __init__(self, camera_id: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._camera_id = str(camera_id).strip() or "cam1"
        self.setStyleSheet(
            f"_CameraSectionHeader{{background:#404040;border-top:1px solid #505050;border-bottom:1px solid #505050;}}"
        )
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        camera_name = _camera_title(self._camera_id)
        self.lbl_title = QtWidgets.QLabel(camera_name)
        self.lbl_title.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:13px;font-weight:bold;")
        layout.addWidget(self.lbl_title)

        layout.addStretch(1)

        self.lbl_result = QtWidgets.QLabel(tr("runtime.untested"))
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result.setFixedSize(56, 28)
        self.lbl_result.setStyleSheet(
            f"background:{_PENDING_GRAY};color:white;font-size:12px;font-weight:bold;border-radius:4px;"
        )
        layout.addWidget(self.lbl_result)

    def set_result(self, result_text: str) -> None:
        result_upper = str(result_text or "").strip().upper()
        if result_upper == "OK":
            bg = _OK_GREEN
            display = "OK"
        elif result_upper == "NG":
            bg = _NG_RED
            display = "NG"
        elif result_upper in {"RUNNING", "INSPECTING"}:
            bg = _RUNNING_YELLOW
            display = tr("runtime.inspecting")
        else:
            bg = _PENDING_GRAY
            display = tr("runtime.untested")
        self.lbl_result.setText(display)
        self.lbl_result.setFixedWidth(_status_badge_width(self.lbl_result, display, maximum=140))
        self.lbl_result.setStyleSheet(
            f"background:{bg};color:white;font-size:12px;font-weight:bold;border-radius:4px;"
        )

    def retranslate_ui(self) -> None:
        self.lbl_title.setText(_camera_title(self._camera_id))
        self.set_result(self.lbl_result.text())


class _CameraSlotFrame(QtWidgets.QFrame):
    def __init__(self, index: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = int(index)
        self._view: RuntimeImageView | None = None
        self.setStyleSheet(
            f"_CameraSlotFrame{{background:{_DARK_BG};border:1px solid #3f3f3f;}}"
        )
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setFixedHeight(30)
        header.setStyleSheet("background:#343434;border-bottom:1px solid #454545;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(8, 3, 8, 3)
        header_layout.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel()
        self.lbl_title.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        header_layout.addWidget(self.lbl_title)

        self.cmb_camera = QtWidgets.QComboBox()
        self.cmb_camera.setMinimumWidth(96)
        self.cmb_camera.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.cmb_camera.setStyleSheet(
            "QComboBox{background:#404040;color:#e0e0e0;border:1px solid #555555;"
            "padding:2px 18px 2px 6px;border-radius:3px;font-size:12px;}"
            "QComboBox::drop-down{border:none;width:18px;}"
            "QComboBox QAbstractItemView{background:#404040;color:#e0e0e0;"
            "selection-background-color:#4f7ecb;}"
        )
        header_layout.addWidget(self.cmb_camera, 1)
        layout.addWidget(header)

        self._content = QtWidgets.QWidget()
        self._content.setStyleSheet(f"background:{_DARK_BG};")
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        layout.addWidget(self._content, 1)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.lbl_title.setText(tr("runtime.slot", index=self._index + 1))
        self.cmb_camera.setToolTip(tr("runtime.slot_camera_tip"))

    def detach_view(self) -> RuntimeImageView | None:
        view = self._view
        self._view = None
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        return view

    def set_view(self, view: RuntimeImageView | None) -> None:
        self.detach_view()
        self._view = view
        if view is None:
            return
        self._content_layout.addWidget(view)


class RuntimeModePage(QtWidgets.QWidget):
    refreshCamerasRequested = QtCore.Signal()
    connectCamerasRequested = QtCore.Signal(object)
    disconnectCamerasRequested = QtCore.Signal()
    triggerRequested = QtCore.Signal()
    triggerCameraRequested = QtCore.Signal(int)
    releaseRequested = QtCore.Signal(str)
    cameraLayoutSettingsChanged = QtCore.Signal(dict)


    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._item_indicators: list[_ItemIndicator] = []
        self._item_indicators_by_item_id: dict[str, _ItemIndicator] = {}
        self._camera_section_headers: dict[str, _CameraSectionHeader] = {}
        self._release_pwd = ""
        self._configured_role_set: set[str] = set(CAMERA_ROLES)
        self._active_role_set: set[str] = set()
        self._busy = False
        self._inspection_rows: list[dict] = []
        self._inspection_structure_signature: tuple[tuple[str, str, str, str], ...] = ()
        self._inspection_structure_initialized = False
        self._camera_preview_sources: dict[str, object | None] = {
            role: None for role in CAMERA_ROLES
        }
        self._current_product_name = ""
        self._current_record_path = ""
        self._available_camera_count: int | None = None
        self._last_runtime_state = "WaitingTrigger"
        self._last_permission_status = ""
        self._last_connection_status = ""
        self._last_runtime_status = ""
        self._last_timing_map: dict[str, object] = {}
        self._last_duration_ms = 0.0
        self._ok_count_total = 0
        self._ng_count_total = 0
        self._camera_layout_id = _DEFAULT_CAMERA_LAYOUT
        self._camera_slot_roles: list[str] = list(CAMERA_ROLES)
        self._camera_views_by_role: dict[str, RuntimeImageView] = {}
        self._camera_slots: list[_CameraSlotFrame] = []
        self._updating_camera_layout_controls = False
        self._build_ui()

    def retranslate_ui(self) -> None:
        self.btn_simulate_foot.setText(tr("runtime.simulate_foot"))
        self.btn_simulate_foot.setToolTip(tr("runtime.simulate_foot_tip"))
        self.btn_trigger_cam1.setText(tr("runtime.trigger_cam1"))
        self.btn_trigger_cam2.setText(tr("runtime.trigger_cam2"))
        self.btn_trigger_cam3.setText(tr("runtime.trigger_cam3"))
        self.lbl_panel_title.setText(tr("runtime.items"))
        self.lbl_total_label.setText(tr("runtime.stats"))
        self.lbl_camera_layout_caption.setText(tr("runtime.view_layout"))
        self._populate_camera_layout_combo()
        self._populate_camera_slot_combos()
        for slot in self._camera_slots:
            slot.retranslate_ui()

        if self._available_camera_count is None:
            self.lbl_header_info.setText(tr("runtime.external_trigger"))
        else:
            self.set_available_cameras([""] * self._available_camera_count)

        self.set_current_product(self._current_product_name)
        self.set_runtime_state(self._last_runtime_state)
        if self._last_runtime_status:
            self.set_runtime_status(self._last_runtime_status)
        if self._last_permission_status:
            self.set_permission_status(self._last_permission_status)
        if self._last_connection_status:
            self.set_connection_status(self._last_connection_status)

        if self._last_timing_map:
            self.set_timing_breakdown(self._last_timing_map)
        else:
            self.lbl_capture_time.setText(self._format_timing_label(tr("runtime.capture"), 0.0))
            self.lbl_match_time.setText(self._format_timing_label(tr("runtime.match"), 0.0))
            self.lbl_infer_time.setText(self._format_timing_label(tr("runtime.infer"), 0.0))
            self._set_total_duration_labels(self._last_duration_ms)

        for header in self._camera_section_headers.values():
            header.retranslate_ui()
        if self._inspection_rows:
            self.set_inspection_items(self._inspection_rows)
        self._refresh_count_labels()

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background:{_DARK_BG};")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶栏 ──
        header = QtWidgets.QFrame()
        header.setMinimumHeight(40)
        header.setMaximumHeight(52)
        header.setStyleSheet(
            f"background:{_HEADER_BG};border-bottom:1px solid #505050;"
        )
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 16, 0)
        header_layout.setSpacing(8)

        self.lbl_run_indicator = QtWidgets.QLabel(tr("runtime.unlocked"))
        self.lbl_run_indicator.setStyleSheet(
            f"color:{_OK_GREEN};font-size:15px;font-weight:bold;"
        )
        header_layout.addWidget(self.lbl_run_indicator)

        self.lbl_header_info = QtWidgets.QLabel(tr("runtime.external_trigger"))
        self.lbl_header_info.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;")
        header_layout.addWidget(self.lbl_header_info)

        header_layout.addSpacing(20)

        _trigger_btn_css = (
            "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
            "padding:4px 12px;border-radius:3px;font-size:12px;}"
            "QPushButton:hover{background:#505050;}"
            "QPushButton:pressed{background:#3794ff;color:white;}"
        )
        self.btn_simulate_foot = QtWidgets.QPushButton(tr("runtime.simulate_foot"))
        self.btn_simulate_foot.setStyleSheet(_trigger_btn_css)
        self.btn_simulate_foot.setAutoDefault(False)
        self.btn_simulate_foot.setDefault(False)
        self.btn_simulate_foot.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_simulate_foot.setEnabled(False)
        self.btn_simulate_foot.setToolTip(tr("runtime.simulate_foot_tip"))
        self.btn_simulate_foot.clicked.connect(self.triggerRequested.emit)
        header_layout.addWidget(self.btn_simulate_foot)
        self.btn_trigger_cam1 = QtWidgets.QPushButton(tr("runtime.trigger_cam1"))
        self.btn_trigger_cam1.setStyleSheet(_trigger_btn_css)
        self.btn_trigger_cam1.setAutoDefault(False)
        self.btn_trigger_cam1.setDefault(False)
        self.btn_trigger_cam1.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_trigger_cam1.setEnabled(False)
        self.btn_trigger_cam1.clicked.connect(lambda: self.triggerCameraRequested.emit(1))
        header_layout.addWidget(self.btn_trigger_cam1)

        self.btn_trigger_cam2 = QtWidgets.QPushButton(tr("runtime.trigger_cam2"))
        self.btn_trigger_cam2.setStyleSheet(_trigger_btn_css)
        self.btn_trigger_cam2.setAutoDefault(False)
        self.btn_trigger_cam2.setDefault(False)
        self.btn_trigger_cam2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_trigger_cam2.setEnabled(False)
        self.btn_trigger_cam2.clicked.connect(lambda: self.triggerCameraRequested.emit(2))
        header_layout.addWidget(self.btn_trigger_cam2)

        self.btn_trigger_cam3 = QtWidgets.QPushButton(tr("runtime.trigger_cam3"))
        self.btn_trigger_cam3.setStyleSheet(_trigger_btn_css)
        self.btn_trigger_cam3.setAutoDefault(False)
        self.btn_trigger_cam3.setDefault(False)
        self.btn_trigger_cam3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_trigger_cam3.setEnabled(False)
        self.btn_trigger_cam3.clicked.connect(lambda: self.triggerCameraRequested.emit(3))
        header_layout.addWidget(self.btn_trigger_cam3)

        header_layout.addStretch(1)

        self.lbl_current_product = QtWidgets.QLabel("-")
        self.lbl_current_product.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:14px;font-weight:bold;")
        header_layout.addWidget(self.lbl_current_product)

        root.addWidget(header)

        # ── 主体：画面 + 右侧面板 ──
        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(4)
        body.setStyleSheet(
            "QSplitter::handle{background:#404040;}"
            "QSplitter::handle:hover{background:#505050;}"
        )

        camera_frame = QtWidgets.QFrame()
        camera_frame.setStyleSheet(f"background:{_DARK_BG};")
        camera_layout = QtWidgets.QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(2, 2, 2, 2)
        camera_layout.setSpacing(2)

        camera_toolbar = QtWidgets.QFrame()
        camera_toolbar.setFixedHeight(34)
        camera_toolbar.setStyleSheet("background:#333333;border-bottom:1px solid #444444;")
        camera_toolbar_layout = QtWidgets.QHBoxLayout(camera_toolbar)
        camera_toolbar_layout.setContentsMargins(8, 3, 8, 3)
        camera_toolbar_layout.setSpacing(8)

        self.lbl_camera_layout_caption = QtWidgets.QLabel(tr("runtime.view_layout"))
        self.lbl_camera_layout_caption.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        camera_toolbar_layout.addWidget(self.lbl_camera_layout_caption)

        self.cmb_camera_layout = QtWidgets.QComboBox()
        self.cmb_camera_layout.setMinimumWidth(128)
        self.cmb_camera_layout.setStyleSheet(
            "QComboBox{background:#404040;color:#e0e0e0;border:1px solid #555555;"
            "padding:3px 20px 3px 8px;border-radius:3px;font-size:12px;}"
            "QComboBox::drop-down{border:none;width:18px;}"
            "QComboBox QAbstractItemView{background:#404040;color:#e0e0e0;"
            "selection-background-color:#4f7ecb;}"
        )
        camera_toolbar_layout.addWidget(self.cmb_camera_layout)
        camera_toolbar_layout.addStretch(1)
        camera_layout.addWidget(camera_toolbar)

        self._camera_grid_host = QtWidgets.QWidget()
        self._camera_grid_host.setStyleSheet(f"background:{_DARK_BG};")
        self._camera_grid = QtWidgets.QGridLayout(self._camera_grid_host)
        self._camera_grid.setContentsMargins(0, 0, 0, 0)
        self._camera_grid.setSpacing(2)

        self.view_cam1 = RuntimeImageView("Cam1")
        self.view_cam2 = RuntimeImageView("Cam2")
        self.view_cam3 = RuntimeImageView("Cam3")
        self._camera_views_by_role = {
            "cam1": self.view_cam1,
            "cam2": self.view_cam2,
            "cam3": self.view_cam3,
        }
        self._camera_slots = [_CameraSlotFrame(index) for index in range(len(CAMERA_ROLES))]
        self._populate_camera_layout_combo()
        self._populate_camera_slot_combos()
        self.cmb_camera_layout.currentIndexChanged.connect(self._on_camera_layout_changed)
        for slot_index, slot in enumerate(self._camera_slots):
            slot.cmb_camera.currentIndexChanged.connect(
                lambda _index, slot_index=slot_index: self._on_camera_slot_changed(slot_index)
            )
        camera_layout.addWidget(self._camera_grid_host, 1)
        self._apply_camera_layout()

        right_panel = QtWidgets.QFrame()
        right_panel.setMinimumWidth(240)
        right_panel.setMaximumWidth(380)
        right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        right_panel.setStyleSheet(
            f"background:{_PANEL_BG};border-left:1px solid #505050;"
        )
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.lbl_panel_title = QtWidgets.QLabel(tr("runtime.items"))
        panel_title = self.lbl_panel_title
        panel_title.setMinimumHeight(30)
        panel_title.setStyleSheet(
            f"background:#404040;color:{_TEXT_LIGHT};font-size:13px;font-weight:bold;"
            "border-bottom:1px solid #505050;padding-left:10px;"
        )
        right_layout.addWidget(panel_title)

        self._items_scroll = QtWidgets.QScrollArea()
        self._items_scroll.setWidgetResizable(True)
        self._items_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._items_scroll.setStyleSheet("QScrollArea{border:none;}")
        self._items_container = QtWidgets.QWidget()
        self._items_container.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._items_container.setStyleSheet(f"background:{_PANEL_BG};")
        self._items_vbox = QtWidgets.QVBoxLayout(self._items_container)
        self._items_vbox.setContentsMargins(0, 0, 0, 0)
        self._items_vbox.setSpacing(0)
        self._items_vbox.addStretch(1)
        self._items_scroll.setWidget(self._items_container)
        right_layout.addWidget(self._items_scroll, 1)

        total_frame = QtWidgets.QFrame()
        self._right_total_frame = total_frame
        total_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        total_frame.setStyleSheet(f"background:#404040;border-top:1px solid #5a5a5a;")
        total_layout = QtWidgets.QVBoxLayout(total_frame)
        total_layout.setContentsMargins(12, 8, 12, 8)
        total_layout.setSpacing(6)

        total_header = QtWidgets.QHBoxLayout()
        total_header.setContentsMargins(0, 0, 0, 0)
        total_header.setSpacing(8)

        self.lbl_total_label = QtWidgets.QLabel(tr("runtime.stats"))
        total_label = self.lbl_total_label
        total_label.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:14px;font-weight:bold;")
        total_header.addWidget(total_label)

        total_layout.addLayout(total_header)

        timing_grid = QtWidgets.QGridLayout()
        timing_grid.setContentsMargins(0, 0, 0, 0)
        timing_grid.setHorizontalSpacing(10)
        timing_grid.setVerticalSpacing(4)

        self.lbl_capture_time = QtWidgets.QLabel(self._format_timing_label(tr("runtime.capture"), 0.0))
        self.lbl_capture_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_capture_time, 0, 0)

        self.lbl_match_time = QtWidgets.QLabel(self._format_timing_label(tr("runtime.match"), 0.0))
        self.lbl_match_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_match_time, 0, 1)

        self.lbl_infer_time = QtWidgets.QLabel(self._format_timing_label(tr("runtime.infer"), 0.0))
        self.lbl_infer_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_infer_time, 1, 0)

        self.lbl_duration = QtWidgets.QLabel(self._format_timing_label(tr("runtime.total_flow"), 0.0))
        self.lbl_duration.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_duration, 1, 1)
        self.lbl_capture_time.hide()
        self.lbl_match_time.hide()
        self.lbl_infer_time.hide()
        self.lbl_duration.hide()
        total_layout.addLayout(timing_grid)

        count_grid = QtWidgets.QGridLayout()
        count_grid.setContentsMargins(0, 0, 0, 0)
        count_grid.setHorizontalSpacing(10)
        count_grid.setVerticalSpacing(4)

        self.lbl_ok_count = QtWidgets.QLabel("OK: 0")
        self.lbl_ok_count.setStyleSheet(f"color:{_OK_GREEN};font-size:18px;font-weight:bold;")
        count_grid.addWidget(self.lbl_ok_count, 0, 0)

        self.lbl_ng_count = QtWidgets.QLabel("NG: 0")
        self.lbl_ng_count.setStyleSheet(f"color:{_NG_RED};font-size:18px;font-weight:bold;")
        count_grid.addWidget(self.lbl_ng_count, 0, 1)

        total_layout.addLayout(count_grid)

        self.lbl_cam1_timing = QtWidgets.QLabel("Cam1: -")
        self.lbl_cam1_timing.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_cam1_timing.setWordWrap(True)
        self.lbl_cam1_timing.setMinimumWidth(0)
        total_layout.addWidget(self.lbl_cam1_timing)
        self.lbl_cam1_timing.hide()

        self.lbl_cam2_timing = QtWidgets.QLabel("Cam2: -")
        self.lbl_cam2_timing.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_cam2_timing.setWordWrap(True)
        self.lbl_cam2_timing.setMinimumWidth(0)
        total_layout.addWidget(self.lbl_cam2_timing)
        self.lbl_cam2_timing.hide()

        self.lbl_cam3_timing = QtWidgets.QLabel("Cam3: -")
        self.lbl_cam3_timing.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_cam3_timing.setWordWrap(True)
        self.lbl_cam3_timing.setMinimumWidth(0)
        total_layout.addWidget(self.lbl_cam3_timing)
        self.lbl_cam3_timing.hide()
        self._refresh_camera_timing_visibility()

        self.lbl_final_result = QtWidgets.QLabel("-")
        self.lbl_final_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_final_result.setMinimumHeight(36)
        self.lbl_final_result.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.lbl_final_result.setStyleSheet(
            f"background:{_PENDING_GRAY};color:white;font-size:16px;font-weight:bold;border-radius:4px;"
        )
        total_layout.addWidget(self.lbl_final_result, 1)
        self.lbl_final_result.hide()

        right_layout.addWidget(total_frame)
        body.addWidget(camera_frame)
        body.addWidget(right_panel)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        body.setSizes([1100, 280])
        self._body_splitter = body

        root.addWidget(body, 1)

        # ── 底栏 ──
        footer = QtWidgets.QFrame()
        footer.setMinimumHeight(28)
        footer.setStyleSheet(
            f"background:{_HEADER_BG};border-top:1px solid #505050;"
        )
        footer_layout = QtWidgets.QVBoxLayout(footer)
        footer_layout.setContentsMargins(12, 4, 12, 4)
        footer_layout.setSpacing(2)

        footer_status_layout = QtWidgets.QHBoxLayout()
        footer_status_layout.setContentsMargins(0, 0, 0, 0)
        footer_status_layout.setSpacing(12)

        self.lbl_footer_time = QtWidgets.QLabel(self._format_timing_label(tr("runtime.process"), 0.0))
        self.lbl_footer_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_status_layout.addWidget(self.lbl_footer_time)
        self.lbl_footer_time.hide()

        self.lbl_footer_state = QtWidgets.QLabel(
            f"{tr('runtime.status')}: {tr_runtime_state('WaitingTrigger')}"
        )
        self.lbl_footer_state.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_status_layout.addWidget(self.lbl_footer_state)

        self.lbl_footer_connection = QtWidgets.QLabel(
            f"{tr('runtime.camera')}: {tr('runtime.not_connected')}"
        )
        self.lbl_footer_connection.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_status_layout.addWidget(self.lbl_footer_connection)

        self.lbl_footer_permission = QtWidgets.QLabel("")
        self.lbl_footer_permission.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_status_layout.addWidget(self.lbl_footer_permission)

        self.lbl_footer_record = _ElidedLabel("")
        self.lbl_footer_record.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_footer_record.setMinimumWidth(0)
        self.lbl_footer_record.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        footer_status_layout.addWidget(self.lbl_footer_record, 1)
        footer_layout.addLayout(footer_status_layout)

        self.ng_summary_bar = QtWidgets.QFrame()
        self.ng_summary_bar.setObjectName("runtimeNgSummaryBar")
        self.ng_summary_bar.setMinimumHeight(24)
        self.ng_summary_bar.setStyleSheet(
            "QFrame#runtimeNgSummaryBar{background:transparent;border:none;}"
        )
        ng_summary_layout = QtWidgets.QHBoxLayout(self.ng_summary_bar)
        ng_summary_layout.setContentsMargins(0, 2, 0, 2)
        ng_summary_layout.setSpacing(16)

        self.lbl_ng_timing_summary = _ElidedLabel("")
        self.lbl_ng_timing_summary.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lbl_ng_timing_summary.setMinimumWidth(0)
        self.lbl_ng_timing_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.lbl_ng_timing_summary.setStyleSheet(
            f"color:{_TEXT_DIM};font-size:11px;background:transparent;"
        )
        ng_summary_layout.addWidget(self.lbl_ng_timing_summary, 1)

        self.lbl_ng_summary = QtWidgets.QLabel("")
        self.lbl_ng_summary.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_ng_summary.setWordWrap(False)
        self.lbl_ng_summary.setMinimumWidth(0)
        self.lbl_ng_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.lbl_ng_summary.setStyleSheet(
            f"color:{_NG_RED};font-size:12px;font-weight:bold;background:transparent;"
        )
        ng_summary_layout.addWidget(self.lbl_ng_summary)

        footer_layout.addWidget(self.ng_summary_bar)
        self.ng_summary_bar.hide()

        root.addWidget(footer)

        # ── 隐藏控件（保持接口兼容） ──
        self.edit_cam1_serial = QtWidgets.QLineEdit()
        self.edit_cam1_serial.hide()
        self.edit_cam2_serial = QtWidgets.QLineEdit()
        self.edit_cam2_serial.hide()
        self.edit_cam3_serial = QtWidgets.QLineEdit()
        self.edit_cam3_serial.hide()
        self.edit_release_password = QtWidgets.QLineEdit()
        self.edit_release_password.hide()
        self.btn_refresh_cameras = QtWidgets.QPushButton()
        self.btn_refresh_cameras.hide()
        self.btn_connect_cameras = QtWidgets.QPushButton()
        self.btn_connect_cameras.hide()
        self.btn_disconnect_cameras = QtWidgets.QPushButton()
        self.btn_disconnect_cameras.hide()
        self.btn_trigger = QtWidgets.QPushButton()
        self.btn_trigger.hide()
        self.btn_release = QtWidgets.QPushButton()
        self.btn_release.hide()
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.hide()

        self.btn_refresh_cameras.clicked.connect(self.refreshCamerasRequested.emit)
        self.btn_connect_cameras.clicked.connect(self._emit_connect_requested)
        self.btn_disconnect_cameras.clicked.connect(self.disconnectCamerasRequested.emit)
        self.btn_trigger.clicked.connect(self.triggerRequested.emit)
        self.btn_release.clicked.connect(self._emit_release_requested)
        self._refresh_count_labels()

    # ------------------------------------------------------------------
    # 公开接口（保持与旧版完全一致的方法签名）
    # ------------------------------------------------------------------

    def set_camera_layout_settings(self, settings: dict[str, object]) -> None:
        self._camera_layout_id = self._normalize_camera_layout_id(
            dict(settings or {}).get("camera_layout", self._camera_layout_id)
        )
        self._camera_slot_roles = self._normalize_camera_slot_roles(
            dict(settings or {}).get("camera_slots", self._camera_slot_roles)
        )
        self._apply_camera_layout(refresh_previews=True)

    def camera_layout_settings(self) -> dict[str, object]:
        return {
            "camera_layout": self._camera_layout_id,
            "camera_slots": list(self._camera_slot_roles),
        }

    @staticmethod
    def _normalize_camera_layout_id(layout_id: object) -> str:
        value = str(layout_id or "").strip()
        valid = {option_id for option_id, _label_key in _CAMERA_LAYOUT_OPTIONS}
        return value if value in valid else _DEFAULT_CAMERA_LAYOUT

    @staticmethod
    def _normalize_camera_slot_roles(roles: object) -> list[str]:
        normalized: list[str] = []
        raw_roles = list(roles) if isinstance(roles, (list, tuple)) else []
        for role in raw_roles:
            role_text = normalize_camera_role(role)
            if role_text and role_text not in normalized:
                normalized.append(role_text)
        for role in CAMERA_ROLES:
            if role not in normalized:
                normalized.append(role)
        return normalized[: len(CAMERA_ROLES)]

    def _populate_camera_layout_combo(self) -> None:
        if not hasattr(self, "cmb_camera_layout"):
            return
        current = self._camera_layout_id
        self._updating_camera_layout_controls = True
        try:
            self.cmb_camera_layout.blockSignals(True)
            self.cmb_camera_layout.clear()
            for layout_id, label_key in _CAMERA_LAYOUT_OPTIONS:
                self.cmb_camera_layout.addItem(tr(label_key), layout_id)
            self._set_combo_current_data(self.cmb_camera_layout, current)
        finally:
            self.cmb_camera_layout.blockSignals(False)
            self._updating_camera_layout_controls = False

    def _populate_camera_slot_combos(self) -> None:
        if not self._camera_slots:
            return
        available_roles = [
            role for role in CAMERA_ROLES if role in self._display_role_set()
        ] or [DEFAULT_CAMERA_ROLE]
        self._updating_camera_layout_controls = True
        try:
            for slot_index, slot in enumerate(self._camera_slots):
                current = self._camera_slot_roles[slot_index]
                combo = slot.cmb_camera
                combo.blockSignals(True)
                combo.clear()
                for role in available_roles:
                    combo.addItem(_camera_title(role), role)
                self._set_combo_current_data(combo, current)
                if combo.currentIndex() < 0:
                    combo.setCurrentIndex(0)
                combo.blockSignals(False)
        finally:
            self._updating_camera_layout_controls = False

    def _reset_camera_slot_order_for_reduced_roles(self) -> None:
        """Keep one/two-channel views in the first canvases while layout is fixed."""
        display_roles = [role for role in CAMERA_ROLES if role in self._display_role_set()]
        if len(display_roles) >= len(CAMERA_ROLES):
            return
        self._camera_slot_roles = self._normalize_camera_slot_roles(display_roles)

    @staticmethod
    def _set_combo_current_data(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _on_camera_layout_changed(self) -> None:
        if self._updating_camera_layout_controls:
            return
        layout_id = self._normalize_camera_layout_id(self.cmb_camera_layout.currentData())
        if layout_id == self._camera_layout_id:
            return
        self._camera_layout_id = layout_id
        self._apply_camera_layout(refresh_previews=True)
        self._emit_camera_layout_settings_changed()

    def _on_camera_slot_changed(self, slot_index: int) -> None:
        if self._updating_camera_layout_controls:
            return
        if slot_index < 0 or slot_index >= len(self._camera_slot_roles):
            return
        slot = self._camera_slots[slot_index]
        new_role = normalize_camera_role(slot.cmb_camera.currentData())
        if not new_role:
            return
        old_role = self._camera_slot_roles[slot_index]
        if new_role == old_role:
            return
        for other_index, other_role in enumerate(self._camera_slot_roles):
            if other_index != slot_index and other_role == new_role:
                self._camera_slot_roles[other_index] = old_role
                break
        self._camera_slot_roles[slot_index] = new_role
        self._camera_slot_roles = self._normalize_camera_slot_roles(self._camera_slot_roles)
        self._apply_camera_layout(refresh_previews=True)
        self._emit_camera_layout_settings_changed()

    def _emit_camera_layout_settings_changed(self) -> None:
        if not self._updating_camera_layout_controls:
            self.cameraLayoutSettingsChanged.emit(self.camera_layout_settings())

    def _apply_camera_layout(self, *, refresh_previews: bool = False) -> None:
        if not hasattr(self, "_camera_grid"):
            return
        self._camera_layout_id = self._normalize_camera_layout_id(self._camera_layout_id)
        self._camera_slot_roles = self._normalize_camera_slot_roles(self._camera_slot_roles)
        self._populate_camera_layout_combo()
        self._populate_camera_slot_combos()

        for slot in self._camera_slots:
            slot.detach_view()
        for view in self._camera_views_by_role.values():
            view.hide()
            view.setParent(None)
        for slot_index, role in enumerate(self._camera_slot_roles):
            slot = self._camera_slots[slot_index]
            view = self._camera_views_by_role.get(role)
            slot.set_view(view)
            if view is not None:
                view.show()
        self._refresh_camera_role_layout()
        if refresh_previews:
            QtCore.QTimer.singleShot(0, self._refresh_camera_previews)

    def _rebuild_camera_grid(self) -> None:
        if not hasattr(self, "_camera_grid"):
            return
        while self._camera_grid.count():
            self._camera_grid.takeAt(0)
        for row in range(3):
            self._camera_grid.setRowStretch(row, 0)
        for column in range(3):
            self._camera_grid.setColumnStretch(column, 0)

        display_roles = self._display_role_set()
        visible_slot_indexes = [
            index
            for index, role in enumerate(self._camera_slot_roles)
            if role in display_roles
        ]
        specs, row_stretches, column_stretches = self._camera_grid_specs(visible_slot_indexes)
        for index, row, column, row_span, column_span in specs:
            self._camera_grid.addWidget(
                self._camera_slots[index],
                row,
                column,
                row_span,
                column_span,
            )
        for row, stretch in row_stretches.items():
            self._camera_grid.setRowStretch(row, stretch)
        for column, stretch in column_stretches.items():
            self._camera_grid.setColumnStretch(column, stretch)

    def _camera_grid_specs(
        self,
        visible_slot_indexes: list[int],
    ) -> tuple[list[tuple[int, int, int, int, int]], dict[int, int], dict[int, int]]:
        if not visible_slot_indexes:
            return [], {}, {}
        if len(visible_slot_indexes) == 1:
            return [(visible_slot_indexes[0], 0, 0, 1, 1)], {0: 1}, {0: 1}
        if len(visible_slot_indexes) == 2:
            return [
                (visible_slot_indexes[0], 0, 0, 1, 1),
                (visible_slot_indexes[1], 0, 1, 1, 1),
            ], {0: 1}, {0: 1, 1: 1}

        layout_id = self._normalize_camera_layout_id(self._camera_layout_id)
        if layout_id == "row":
            return [
                (0, 0, 0, 1, 1),
                (1, 0, 1, 1, 1),
                (2, 0, 2, 1, 1),
            ], {0: 1}, {0: 1, 1: 1, 2: 1}
        if layout_id == "one_top_two_bottom":
            return [
                (0, 0, 0, 1, 2),
                (1, 1, 0, 1, 1),
                (2, 1, 1, 1, 1),
            ], {0: 1, 1: 1}, {0: 1, 1: 1}
        if layout_id == "main_left":
            return [
                (0, 0, 0, 2, 1),
                (1, 0, 1, 1, 1),
                (2, 1, 1, 1, 1),
            ], {0: 1, 1: 1}, {0: 2, 1: 1}
        if layout_id == "main_right":
            return [
                (0, 0, 0, 1, 1),
                (1, 1, 0, 1, 1),
                (2, 0, 1, 2, 1),
            ], {0: 1, 1: 1}, {0: 1, 1: 2}
        return [
            (0, 0, 0, 1, 1),
            (1, 0, 1, 1, 1),
            (2, 1, 0, 1, 2),
        ], {0: 1, 1: 1}, {0: 1, 1: 1}

    def camera_bindings(self) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for role in CAMERA_ROLES:
            if role not in self._configured_role_set:
                continue
            serial = self.camera_serial(role)
            if serial:
                bindings[role] = serial
        return bindings

    def camera_serial(self, role: str) -> str:
        role_text = normalize_camera_role(role)
        editor = getattr(self, f"edit_{role_text}_serial", None)
        if editor is None:
            return ""
        return editor.text().strip()

    def set_camera_serial(self, role: str, serial: str) -> None:
        role_text = normalize_camera_role(role)
        editor = getattr(self, f"edit_{role_text}_serial", None)
        if editor is not None:
            editor.setText(str(serial or "").strip())

    def release_password(self) -> str:
        return self.edit_release_password.text()

    def set_available_cameras(self, descriptions: list[str]) -> None:
        n = len(descriptions)
        self._available_camera_count = n
        self.lbl_header_info.setText(
            tr("runtime.visible_cameras", count=n) if n else tr("runtime.no_cameras_found")
        )

    def set_current_product(self, product_name: str) -> None:
        product_text = str(product_name or "").strip()
        if product_text != self._current_product_name:
            self._current_product_name = product_text
            self._reload_daily_result_counters()
        self.lbl_current_product.setText(
            f"{tr('runtime.product')}: {product_name}" if product_name else "-"
        )

    def set_runtime_state(self, state_text: str) -> None:
        """顶栏：NG 锁或本轮 NG 结束时为「已锁定」红字，其余为「已解锁」绿字；底栏同步状态机中文。"""
        st = str(state_text or "").strip()
        self._last_runtime_state = st
        locked_states = frozenset({"LockedByNg", "CompletedNg"})
        if st == "Unavailable":
            text = tr("runtime.not_ready")
            color = _TEXT_DIM
        elif st in locked_states:
            text = tr("runtime.locked")
            color = _NG_RED
        else:
            text = tr("runtime.unlocked")
            color = _OK_GREEN
        self.lbl_run_indicator.setText(text)
        self.lbl_run_indicator.setStyleSheet(f"color:{color};font-size:15px;font-weight:bold;")
        footer_detail = tr_runtime_state(st) or st
        self.lbl_footer_state.setText(f"{tr('runtime.status')}: {footer_detail}")

    def set_permission_status(self, status_text: str) -> None:
        self._last_permission_status = str(status_text or "")
        display = tr_status_text(status_text)
        self.lbl_footer_permission.setText(f"{tr('runtime.release')}: {display}" if display else "")

    def set_connection_status(self, status_text: str) -> None:
        self._last_connection_status = str(status_text or "")
        self.lbl_footer_connection.setText(f"{tr('runtime.camera')}: {tr_status_text(status_text)}")

    def set_tower_light_status(self, status_text: str) -> None:
        pass

    def set_runtime_status(self, status_text: str) -> None:
        self._last_runtime_status = str(status_text or "")
        clean_text = self._sanitize_runtime_status_text_v3(status_text)
        display = tr_status_text(clean_text)
        self.lbl_footer_state.setText(
            f"{tr('runtime.status')}: {display}" if display else f"{tr('runtime.status')}: -"
        )

    def set_final_result(self, result_text: str, detail_text: str) -> None:
        result_upper = str(result_text).strip().upper()
        if result_upper == "OK":
            bg = _OK_GREEN
            display = "OK"
        elif result_upper == "NG":
            bg = _NG_RED
            display = "NG"
        elif result_upper in {"ERROR", "BLOCKED"}:
            bg = _NG_RED
            display = result_upper
        else:
            bg = _PENDING_GRAY
            display = result_text or "-"
        self.lbl_final_result.setText(display)
        self.lbl_final_result.setStyleSheet(
            f"background:{bg};color:white;font-size:16px;font-weight:bold;border-radius:4px;"
        )
        self.lbl_final_result.setToolTip(str(detail_text or ""))
        self.lbl_final_result.show()
        if result_upper in {"OK", "NG"}:
            self._increment_result_counter(result_upper)

    def set_record_path(self, record_path: str) -> None:
        self.lbl_footer_record.setText(record_path or "")
        normalized_path = self._normalize_record_path(record_path)
        if normalized_path != self._current_record_path:
            self._current_record_path = normalized_path
            self._reload_daily_result_counters()

    def set_configured_camera_roles(self, roles: list[str]) -> None:
        self._configured_role_set = set(configured_camera_roles(roles))
        self._reset_camera_slot_order_for_reduced_roles()
        self._populate_camera_slot_combos()
        self._refresh_camera_role_layout()
        self._refresh_trigger_buttons()

    def set_active_camera_roles(self, roles: list[str]) -> None:
        role_set = {str(role).strip() for role in roles if str(role).strip()}
        self._active_role_set = role_set
        if not role_set:
            for role in CAMERA_ROLES:
                view = self._camera_views_by_role.get(role)
                if view is not None:
                    placeholder = (
                        tr("runtime.no_camera_connected")
                        if role == DEFAULT_CAMERA_ROLE
                        else role.upper()
                    )
                    view.set_runtime_pixmap(None, placeholder=placeholder)
        self.lbl_footer_connection.setText(
            f"{tr('runtime.camera')}: "
            + (", ".join(sorted(role_set)) if role_set else tr("runtime.not_connected"))
        )
        self._reset_camera_slot_order_for_reduced_roles()
        self._populate_camera_slot_combos()
        self._refresh_camera_role_layout()
        self._refresh_camera_timing_visibility()
        self._refresh_trigger_buttons()

    def set_inspection_items(self, rows: list[dict]) -> None:
        self._inspection_rows = list(rows or [])

        grouped_rows: dict[str, list[dict]] = {role: [] for role in CAMERA_ROLES}
        for row in self._inspection_rows:
            camera_id = str(row.get("camera_id", "cam1")).strip() or "cam1"
            if camera_id not in grouped_rows:
                grouped_rows[camera_id] = []
            grouped_rows[camera_id].append(row)
        display_grouped_rows = {
            camera_id: self._runtime_display_rows_for_camera(camera_rows)
            for camera_id, camera_rows in grouped_rows.items()
        }
        display_entries = [
            (camera_id, row)
            for camera_id in CAMERA_ROLES
            for row in display_grouped_rows.get(camera_id, [])
        ]
        structure_signature = tuple(
            (
                camera_id,
                str(row.get("item_id", "") or "").strip(),
                self._runtime_item_display_name(row),
                str(row.get("algorithm_code", "") or "").strip(),
            )
            for camera_id, row in display_entries
        )

        if (
            self._inspection_structure_initialized
            and structure_signature == self._inspection_structure_signature
            and len(display_entries) == len(self._item_indicators)
        ):
            for indicator, (_camera_id, row) in zip(self._item_indicators, display_entries):
                indicator.set_result(
                    str(row.get("status_kind", "pending")),
                    str(row.get("status_text", "")),
                )
            self._refresh_ng_summary(display_grouped_rows)
            self._refresh_trigger_buttons()
            return

        self._inspection_structure_signature = structure_signature
        self._inspection_structure_initialized = True
        while self._items_vbox.count():
            item = self._items_vbox.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        self._item_indicators.clear()
        self._item_indicators_by_item_id.clear()
        self._camera_section_headers.clear()

        insert_index = 0
        display_index = 1
        for camera_id in CAMERA_ROLES:
            camera_rows = display_grouped_rows.get(camera_id, [])
            if not camera_rows:
                continue

            header = _CameraSectionHeader(camera_id)
            self._items_vbox.insertWidget(insert_index, header)
            self._camera_section_headers[camera_id] = header
            insert_index += 1

            for row in camera_rows:
                name = self._runtime_item_display_name(row)
                indicator = _ItemIndicator(display_index, name)
                kind = str(row.get("status_kind", "pending"))
                text = str(row.get("status_text", ""))
                indicator.set_result(kind, text)
                self._items_vbox.insertWidget(insert_index, indicator)
                self._item_indicators.append(indicator)
                item_id = str(row.get("item_id", "")).strip()
                if item_id:
                    self._item_indicators_by_item_id[item_id] = indicator
                insert_index += 1
                display_index += 1

        self._items_vbox.addStretch(1)
        self._items_container.update()
        self._items_scroll.viewport().update()
        self._refresh_ng_summary(display_grouped_rows)
        self._refresh_trigger_buttons()

    def _refresh_ng_summary(self, grouped_rows: dict[str, list[dict]]) -> None:
        summaries: list[str] = []
        for camera_id in CAMERA_ROLES:
            first_ng_row = next(
                (
                    row
                    for row in grouped_rows.get(camera_id, [])
                    if str(row.get("status_kind", "")).strip().lower() == "ng"
                    or str(row.get("result", "")).strip().upper() == "NG"
                ),
                None,
            )
            if first_ng_row is None:
                continue
            camera_index = camera_index_for_role(camera_id)
            camera_name = f"Cam{camera_index}" if camera_index else camera_id
            if self._is_template_match_failure_row(first_ng_row):
                summaries.append(
                    tr("runtime.ng_summary_match_failed", camera=camera_name)
                )
            else:
                item_name = self._runtime_item_display_name(first_ng_row)
                summaries.append(
                    tr("runtime.ng_summary_item", camera=camera_name, item=item_name)
                )

        summary_text = "，".join(summaries)
        self.lbl_ng_summary.setText(summary_text)
        self.lbl_ng_summary.setToolTip(summary_text)
        self.lbl_ng_summary.setVisible(bool(summary_text))
        self._refresh_ng_timing_summary()
        self._refresh_result_summary_bar_visibility()

    @staticmethod
    def _is_template_match_failure_row(row: dict) -> bool:
        detail_text = " ".join(
            str(row.get(key, "") or "")
            for key in ("status_text", "detail", "error_message")
        ).casefold()
        return any(marker.casefold() in detail_text for marker in _TEMPLATE_MATCH_FAILURE_MARKERS)

    @staticmethod
    def _runtime_item_display_name(row: dict) -> str:
        name = str(row.get("display_name", "") or "").strip()
        algorithm = str(row.get("algorithm_code", "") or "").strip()
        item_id = str(row.get("item_id", "") or "").strip()
        if algorithm in LINE_DISTANCE_ALGORITHMS:
            default_names = {"", "Line Distance", "line_distance", tr("debug.algorithm.line_distance")}
            display_key = "debug.algorithm.line_distance"
            if algorithm == "line_distance_ref_normal":
                default_names.update(
                    {
                        "Reference Normal Distance",
                        "line_distance_ref_normal",
                        tr("debug.algorithm.line_distance_ref_normal"),
                    }
                )
                display_key = "debug.algorithm.line_distance_ref_normal"
            if item_id.startswith("line_distance"):
                default_names.add(item_id)
            if name in default_names:
                return tr(display_key)
        if algorithm in CENTER_DISTANCE_ALGORITHMS:
            default_names = {"", "Center Distance", "center_distance", tr("debug.algorithm.center_distance")}
            if item_id.startswith("center_distance"):
                default_names.add(item_id)
            if name in default_names:
                return tr("debug.algorithm.center_distance")
        return name or str(row.get("roi_label", "") or item_id)

    @staticmethod
    def _runtime_display_rows_for_camera(rows: list[dict]) -> list[dict]:
        camera_rows = list(rows or [])
        line_helper_ids: set[str] = set()
        center_helper_ids: set[str] = set()
        for row in camera_rows:
            algorithm = str(row.get("algorithm_code", "") or "").strip()
            if algorithm not in LINE_DISTANCE_ALGORITHMS and algorithm not in CENTER_DISTANCE_ALGORITHMS:
                continue
            params = row.get("params", {})
            if not isinstance(params, dict):
                continue
            helper_ids = center_helper_ids if algorithm in CENTER_DISTANCE_ALGORITHMS else line_helper_ids
            keys = (
                ("center_a_item_id", "center_b_item_id")
                if algorithm in CENTER_DISTANCE_ALGORITHMS
                else ("line_a_item_id", "line_b_item_id")
            )
            for key in keys:
                item_id = str(params.get(key, "") or "").strip()
                if item_id:
                    helper_ids.add(item_id)
        if not line_helper_ids and not center_helper_ids:
            return camera_rows
        return [
            row
            for row in camera_rows
            if not (
                str(row.get("algorithm_code", "") or "").strip() in FIND_LINE_ALGORITHMS
                and str(row.get("item_id", "") or "").strip() in line_helper_ids
            )
            and not (
                str(row.get("algorithm_code", "") or "").strip() == BRIGHT_BLOCK_CENTER_ALGORITHM
                and str(row.get("item_id", "") or "").strip() in center_helper_ids
            )
        ]

    def set_camera_pixmap(self, role: str, pixmap: QtGui.QPixmap | None, *, placeholder: str | None = None) -> None:
        view = self._camera_views_by_role.get(normalize_camera_role(role))
        if view is not None:
            view.set_runtime_pixmap(pixmap, placeholder=placeholder)

    def set_camera_source_path(self, role: str, path: str) -> None:
        source = str(path or "").strip()
        role_text = normalize_camera_role(role, default=DEFAULT_CAMERA_ROLE)
        self._camera_preview_sources[role_text] = source if source else None

    def set_camera_preview_source(self, role: str, source: object) -> None:
        role_text = normalize_camera_role(role, default=DEFAULT_CAMERA_ROLE)
        self._camera_preview_sources[role_text] = source

    def roi_statuses_for_camera(self, camera_id: str) -> dict[str, str]:
        rows = [row for row in self._inspection_rows if bool(row.get("enabled", True))]
        return merge_roi_statuses(rows, camera_id=camera_id)

    def roi_labels_for_camera(self, camera_id: str) -> set[str]:
        wanted_camera = normalize_camera_role(camera_id, default=DEFAULT_CAMERA_ROLE)
        labels: set[str] = set()
        for row in self._inspection_rows:
            if not bool(row.get("enabled", True)):
                continue
            label = str(row.get("roi_label", "") or "").strip()
            if not is_roi_label(label):
                continue
            row_camera = normalize_camera_role(row.get("camera_id", ""), default=DEFAULT_CAMERA_ROLE)
            if row_camera == wanted_camera:
                labels.add(label)
        return labels

    def clear_camera_views(self) -> None:
        self._active_role_set = set()
        self._camera_preview_sources = {role: None for role in CAMERA_ROLES}
        for role in CAMERA_ROLES:
            view = self._camera_views_by_role.get(role)
            if view is not None:
                view.set_runtime_pixmap(None, placeholder=role.upper())
            label = getattr(self, f"lbl_{role}_timing", None)
            if label is not None:
                label.setText(f"{role.upper()}: -")
        self._refresh_camera_role_layout()
        self._refresh_camera_timing_visibility()
        self.set_camera_results({})
        self._refresh_trigger_buttons()

    def set_camera_results(self, result_map: dict[str, str]) -> None:
        normalized = {
            str(camera_id).strip(): str(result or "").strip()
            for camera_id, result in dict(result_map or {}).items()
            if str(camera_id).strip()
        }
        for camera_id, header in self._camera_section_headers.items():
            header.set_result(normalized.get(camera_id, ""))

    def _refresh_camera_previews(self) -> None:
        """Re-render cached previews after a camera-layout change."""
        from ui.window_common import update_runtime_preview

        for role in CAMERA_ROLES:
            source = self._camera_preview_sources.get(role)
            if source is not None:
                update_runtime_preview(self, role, source)

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._refresh_trigger_buttons()

    def set_duration_ms(self, ms: int) -> None:
        self._set_total_duration_labels(float(ms or 0.0))

    def set_timing_breakdown(self, timing_map: dict[str, object]) -> None:
        timing_map = dict(timing_map or {})
        self._last_timing_map = timing_map
        capture_ms = self._coerce_ms(timing_map.get("capture_ms"))
        match_ms = self._coerce_ms(timing_map.get("match_ms"))
        infer_ms = self._coerce_ms(timing_map.get("infer_ms"))
        duration_ms = self._coerce_ms(timing_map.get("duration_ms"))
        self.lbl_capture_time.setText(self._format_timing_label(tr("runtime.capture"), capture_ms))
        self.lbl_match_time.setText(self._format_timing_label(tr("runtime.match"), match_ms))
        self.lbl_infer_time.setText(self._format_timing_label(tr("runtime.infer"), infer_ms))
        self._set_total_duration_labels(duration_ms)
        for role in CAMERA_ROLES:
            label = getattr(self, f"lbl_{role}_timing", None)
            if label is None:
                continue
            label.setText(
                self._format_camera_timing_text(
                    role.upper(),
                    self._coerce_ms(timing_map.get(f"{role}_capture_ms")),
                    self._coerce_ms(timing_map.get(f"{role}_match_ms")),
                    self._coerce_ms(timing_map.get(f"{role}_infer_ms")),
                    self._coerce_ms(timing_map.get(f"{role}_total_ms")),
                )
            )
        self._refresh_camera_timing_visibility()
        self._refresh_ng_timing_summary()

    @staticmethod
    def _coerce_ms(value: object) -> float:
        try:
            return max(0.0, float(value or 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _format_timing_label(title: str, value: float) -> str:
        return f"{title}: {value:.1f} ms" if value > 0.0 else f"{title}: -"

    @staticmethod
    def _format_camera_timing_text(
        title: str,
        capture_ms: float,
        match_ms: float,
        infer_ms: float,
        total_ms: float,
    ) -> str:
        if capture_ms <= 0.0 and match_ms <= 0.0 and infer_ms <= 0.0 and total_ms <= 0.0:
            return f"{title}: -"
        return (
            f"{title}: {tr('runtime.capture')} {capture_ms:.1f}  "
            f"{tr('runtime.match')} {match_ms:.1f}\n"
            f"{tr('runtime.infer')} {infer_ms:.1f}  "
            f"{tr('runtime.total_flow')} {total_ms:.1f} ms"
        )

    def _refresh_ng_timing_summary(self) -> None:
        label = getattr(self, "lbl_ng_timing_summary", None)
        if label is None:
            return
        parts: list[str] = []
        timing_map = self._last_timing_map
        for role in CAMERA_ROLES:
            capture_ms = self._coerce_ms(timing_map.get(f"{role}_capture_ms"))
            match_ms = self._coerce_ms(timing_map.get(f"{role}_match_ms"))
            infer_ms = self._coerce_ms(timing_map.get(f"{role}_infer_ms"))
            total_ms = self._coerce_ms(timing_map.get(f"{role}_total_ms"))
            if capture_ms <= 0.0 and match_ms <= 0.0 and infer_ms <= 0.0 and total_ms <= 0.0:
                continue
            parts.append(
                f"{role.upper()} "
                f"{tr('runtime.capture')}{capture_ms:.1f} "
                f"{tr('runtime.match')}{match_ms:.1f} "
                f"{tr('runtime.infer')}{infer_ms:.1f} "
                f"{tr('runtime.total_flow')}{total_ms:.1f}ms"
            )
        text = "  |  ".join(parts)
        label.setText(text)
        label.setVisible(bool(text))
        self._refresh_result_summary_bar_visibility()

    def _refresh_result_summary_bar_visibility(self) -> None:
        bar = getattr(self, "ng_summary_bar", None)
        timing_label = getattr(self, "lbl_ng_timing_summary", None)
        summary_label = getattr(self, "lbl_ng_summary", None)
        if bar is None or timing_label is None or summary_label is None:
            return
        has_timing = bool(str(getattr(timing_label, "_full_text", "") or "").strip())
        has_ng_summary = bool(summary_label.text().strip())
        bar.setVisible(has_timing or has_ng_summary)

    @staticmethod
    def _sanitize_runtime_status_text(status_text: str) -> str:
        text = " ".join(str(status_text or "").split())
        patterns = [
            r"(?:^|[\s,;，；]+)capture\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)match\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)infer\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)total\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[;, ]+)耗时[:：]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[;, ]+)处理[:：]?\s*\d+(?:\.\d+)?\s*ms",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip(" ;,")
        return text

    def _set_total_duration_labels(self, duration_ms: float) -> None:
        self._last_duration_ms = float(duration_ms or 0.0)
        self.lbl_footer_time.setText(
            f"{tr('runtime.process')}: {duration_ms:.1f}ms"
            if duration_ms > 0.0
            else self._format_timing_label(tr("runtime.process"), 0.0)
        )
        self.lbl_duration.setText(self._format_timing_label(tr("runtime.total_flow"), duration_ms))

    @staticmethod
    def _sanitize_runtime_status_text_v2(status_text: str) -> str:
        text = " ".join(str(status_text or "").split())
        patterns = [
            r"(?:^|[\s,;，；]+)capture\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)match\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)infer\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)total\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)耗时\s*[:：=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;，；]+)处理\s*[:：=]?\s*\d+(?:\.\d+)?\s*ms",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*[;；,，]\s*[;；,，]+\s*", "; ", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ;,，；")
        return text

    def _refresh_camera_timing_visibility(self) -> None:
        for role in CAMERA_ROLES:
            label = getattr(self, f"lbl_{role}_timing", None)
            if label is not None:
                label.hide()

    def _display_role_set(self) -> set[str]:
        return set(self._active_role_set or self._configured_role_set)

    def _refresh_camera_role_layout(self) -> None:
        display_roles = self._display_role_set()
        self._refresh_camera_layout_selector_state(display_roles)
        for role in CAMERA_ROLES:
            view = self._camera_views_by_role.get(role)
            if view is not None:
                view.setVisible(role in display_roles)
        for slot_index, slot in enumerate(self._camera_slots):
            role = self._camera_slot_roles[slot_index]
            slot.setVisible(role in display_roles)
        self._rebuild_camera_grid()

    def _refresh_camera_layout_selector_state(self, display_roles: set[str]) -> None:
        """Only three active logical channels need a user-selectable layout."""
        selectable = len(display_roles) == len(CAMERA_ROLES)
        layout_combo = getattr(self, "cmb_camera_layout", None)
        if layout_combo is not None:
            layout_combo.setEnabled(selectable)
        caption = getattr(self, "lbl_camera_layout_caption", None)
        if caption is not None:
            caption.setEnabled(selectable)

    @staticmethod
    def _sanitize_runtime_status_text_v3(status_text: str) -> str:
        text = " ".join(str(status_text or "").split())
        patterns = [
            r"(?:^|[\s,;\uFF0C\uFF1B]+)capture\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;\uFF0C\uFF1B]+)match\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;\uFF0C\uFF1B]+)infer\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;\uFF0C\uFF1B]+)total\s*[:=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;\uFF0C\uFF1B]+)\u8017\u65F6\s*[:\uFF1A=]?\s*\d+(?:\.\d+)?\s*ms",
            r"(?:^|[\s,;\uFF0C\uFF1B]+)\u5904\u7406\s*[:\uFF1A=]?\s*\d+(?:\.\d+)?\s*ms",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*[;,\uFF0C\uFF1B]\s*[;,\uFF0C\uFF1B]+\s*", "; ", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ;,\uFF0C\uFF1B")
        return text

    def _refresh_trigger_buttons(self) -> None:
        allow_full_trigger = (not self._busy) and any(
            self._has_enabled_items(role) for role in self._active_role_set
        )
        self.btn_simulate_foot.setEnabled(allow_full_trigger)
        for role in CAMERA_ROLES:
            button = getattr(self, f"btn_trigger_{role}", None)
            if button is None:
                continue
            button.setEnabled(
                (not self._busy)
                and (role in self._configured_role_set)
                and (role in self._active_role_set)
                and self._has_enabled_items(role)
            )

    def _has_enabled_items(self, camera_id: str) -> bool:
        camera_text = str(camera_id or "").strip()
        return any(
            bool(row.get("enabled", True)) and str(row.get("camera_id", "")).strip() == camera_text
            for row in self._inspection_rows
        )

    def _emit_connect_requested(self) -> None:
        self.connectCamerasRequested.emit(self.camera_bindings())

    def _emit_release_requested(self) -> None:
        self.releaseRequested.emit(self.release_password())

    def _refresh_count_labels(self) -> None:
        self.lbl_ok_count.setText(f"OK: {int(self._ok_count_total)}")
        self.lbl_ng_count.setText(f"NG: {int(self._ng_count_total)}")

    def _reload_daily_result_counters(self) -> None:
        ok_count = 0
        ng_count = 0
        product_name = str(self._current_product_name or "").strip()
        record_path_text = str(self._current_record_path or "").strip()
        if product_name and record_path_text:
            record_path = Path(record_path_text)
            if record_path.exists() and record_path.is_file():
                try:
                    with record_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                        for row in csv.DictReader(csv_file):
                            if str(row.get("product_name", "")).strip() != product_name:
                                continue
                            final_result = _runtime_record_row_result(row)
                            if final_result == "OK":
                                ok_count += 1
                            elif final_result == "NG":
                                ng_count += 1
                except Exception:
                    ok_count = 0
                    ng_count = 0
        self._ok_count_total = ok_count
        self._ng_count_total = ng_count
        self._refresh_count_labels()

    @staticmethod
    def _normalize_record_path(record_path: str) -> str:
        path_text = str(record_path or "").strip()
        if not path_text or path_text == "-":
            return ""
        try:
            return str(Path(path_text))
        except Exception:
            return ""

    def _increment_result_counter(self, result_text: str) -> None:
        result_key = str(result_text or "").strip().upper()
        if result_key not in {"OK", "NG"}:
            return
        if result_key == "OK":
            self._ok_count_total += 1
        else:
            self._ng_count_total += 1
        self._refresh_count_labels()
