
from __future__ import annotations

import csv
import re
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    CENTER_DISTANCE_ALGORITHMS,
    FIND_LINE_ALGORITHMS,
    LINE_DISTANCE_ALGORITHMS,
)
from ui.i18n import tr, tr_runtime_state, tr_status_text
from ui.roi_overlay_colors import merge_roi_statuses


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


def _camera_title(camera_id: str) -> str:
    return tr("runtime.camera1") if str(camera_id).strip() == "cam1" else tr("runtime.camera2")


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
        bg = color_map.get(status_kind, _PENDING_GRAY)
        display = tr_status_text(status_text.split("(")[0].strip()) if status_text else ""
        self.lbl_result.setFixedWidth(_status_badge_width(self.lbl_result, display, maximum=180 if status_kind == "measured" else 140))
        self.lbl_result.setText(display)
        self.lbl_result.setToolTip(str(status_text or display or ""))
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


class RuntimeModePage(QtWidgets.QWidget):
    refreshCamerasRequested = QtCore.Signal()
    connectCamerasRequested = QtCore.Signal(object)
    disconnectCamerasRequested = QtCore.Signal()
    triggerRequested = QtCore.Signal()
    triggerCameraRequested = QtCore.Signal(int)
    releaseRequested = QtCore.Signal(str)


    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._item_indicators: list[_ItemIndicator] = []
        self._item_indicators_by_item_id: dict[str, _ItemIndicator] = {}
        self._camera_section_headers: dict[str, _CameraSectionHeader] = {}
        self._cam1_serial = ""
        self._cam2_serial = ""
        self._release_pwd = ""
        self._configured_role_set: set[str] = {"cam1", "cam2"}
        self._active_role_set: set[str] = set()
        self._busy = False
        self._inspection_rows: list[dict] = []
        self._camera_preview_sources: dict[str, object | None] = {"cam1": None, "cam2": None}
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
        self._build_ui()

    def retranslate_ui(self) -> None:
        self.btn_simulate_foot.setText(tr("runtime.simulate_foot"))
        self.btn_simulate_foot.setToolTip(tr("runtime.simulate_foot_tip"))
        self.btn_trigger_cam1.setText(tr("runtime.trigger_cam1"))
        self.btn_trigger_cam2.setText(tr("runtime.trigger_cam2"))
        self.lbl_panel_title.setText(tr("runtime.items"))
        self.lbl_total_label.setText(tr("runtime.stats"))

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

        camera_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        camera_splitter.setChildrenCollapsible(False)
        camera_splitter.setHandleWidth(2)
        camera_splitter.setStyleSheet("QSplitter::handle{background:#383838;}")

        self.view_cam1 = RuntimeImageView("Cam1")
        self.view_cam2 = RuntimeImageView("Cam2")
        camera_splitter.addWidget(self.view_cam1)
        camera_splitter.addWidget(self.view_cam2)
        camera_splitter.setStretchFactor(0, 1)
        camera_splitter.setStretchFactor(1, 1)
        camera_layout.addWidget(camera_splitter, 1)
        self._camera_splitter = camera_splitter

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
        self.lbl_ok_count.setStyleSheet(f"color:{_OK_GREEN};font-size:12px;font-weight:bold;")
        count_grid.addWidget(self.lbl_ok_count, 0, 0)

        self.lbl_ng_count = QtWidgets.QLabel("NG: 0")
        self.lbl_ng_count.setStyleSheet(f"color:{_NG_RED};font-size:12px;font-weight:bold;")
        count_grid.addWidget(self.lbl_ng_count, 0, 1)

        total_layout.addLayout(count_grid)

        self.lbl_cam1_timing = QtWidgets.QLabel("Cam1: -")
        self.lbl_cam1_timing.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_cam1_timing.setWordWrap(True)
        self.lbl_cam1_timing.setMinimumWidth(0)
        total_layout.addWidget(self.lbl_cam1_timing)

        self.lbl_cam2_timing = QtWidgets.QLabel("Cam2: -")
        self.lbl_cam2_timing.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_cam2_timing.setWordWrap(True)
        self.lbl_cam2_timing.setMinimumWidth(0)
        total_layout.addWidget(self.lbl_cam2_timing)
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
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 4, 12, 4)
        footer_layout.setSpacing(12)

        self.lbl_footer_time = QtWidgets.QLabel(self._format_timing_label(tr("runtime.process"), 0.0))
        self.lbl_footer_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_time)
        self.lbl_footer_time.hide()

        self.lbl_footer_state = QtWidgets.QLabel(
            f"{tr('runtime.status')}: {tr_runtime_state('WaitingTrigger')}"
        )
        self.lbl_footer_state.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_state)

        self.lbl_footer_connection = QtWidgets.QLabel(
            f"{tr('runtime.camera')}: {tr('runtime.not_connected')}"
        )
        self.lbl_footer_connection.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_connection)

        self.lbl_footer_permission = QtWidgets.QLabel("")
        self.lbl_footer_permission.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_permission)

        self.lbl_footer_record = _ElidedLabel("")
        self.lbl_footer_record.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        self.lbl_footer_record.setMinimumWidth(0)
        self.lbl_footer_record.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        footer_layout.addWidget(self.lbl_footer_record, 1)

        root.addWidget(footer)

        # ── 隐藏控件（保持接口兼容） ──
        self.edit_cam1_serial = QtWidgets.QLineEdit()
        self.edit_cam1_serial.hide()
        self.edit_cam2_serial = QtWidgets.QLineEdit()
        self.edit_cam2_serial.hide()
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

    def camera_bindings(self) -> dict[str, str]:
        bindings: dict[str, str] = {}
        cam1 = self.edit_cam1_serial.text().strip()
        cam2 = self.edit_cam2_serial.text().strip()
        if cam1:
            bindings["cam1"] = cam1
        if cam2:
            bindings["cam2"] = cam2
        return bindings

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
        if result_upper in {"OK", "NG"}:
            self._increment_result_counter(result_upper)

    def set_record_path(self, record_path: str) -> None:
        self.lbl_footer_record.setText(record_path or "")
        normalized_path = self._normalize_record_path(record_path)
        if normalized_path != self._current_record_path:
            self._current_record_path = normalized_path
            self._reload_daily_result_counters()

    def set_configured_camera_roles(self, roles: list[str]) -> None:
        configured = {
            str(role).strip()
            for role in roles
            if str(role).strip() in {"cam1", "cam2"}
        }
        if not configured:
            configured = {"cam1"}
        self._configured_role_set = configured
        self._refresh_camera_role_layout()
        self._refresh_trigger_buttons()

    def set_active_camera_roles(self, roles: list[str]) -> None:
        role_set = {str(role).strip() for role in roles if str(role).strip()}
        self._active_role_set = role_set
        if not role_set:
            self.view_cam1.set_runtime_pixmap(None, placeholder=tr("runtime.no_camera_connected"))
            self.view_cam2.set_runtime_pixmap(None, placeholder="Cam2")
        self.lbl_footer_connection.setText(
            f"{tr('runtime.camera')}: "
            + (", ".join(sorted(role_set)) if role_set else tr("runtime.not_connected"))
        )
        self._refresh_camera_role_layout()
        self._refresh_camera_timing_visibility()
        self._refresh_trigger_buttons()

    def set_inspection_items(self, rows: list[dict]) -> None:
        self._inspection_rows = list(rows or [])
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

        grouped_rows: dict[str, list[dict]] = {"cam1": [], "cam2": []}
        for row in rows:
            camera_id = str(row.get("camera_id", "cam1")).strip() or "cam1"
            if camera_id not in grouped_rows:
                grouped_rows[camera_id] = []
            grouped_rows[camera_id].append(row)
        display_grouped_rows = {
            camera_id: self._runtime_display_rows_for_camera(camera_rows)
            for camera_id, camera_rows in grouped_rows.items()
        }

        insert_index = 0
        display_index = 1
        for camera_id in ["cam1", "cam2"]:
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
        self._refresh_camera_previews()
        self._refresh_trigger_buttons()

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
        if role == "cam1":
            self.view_cam1.set_runtime_pixmap(pixmap, placeholder=placeholder)
        elif role == "cam2":
            self.view_cam2.set_runtime_pixmap(pixmap, placeholder=placeholder)

    def set_camera_source_path(self, role: str, path: str) -> None:
        source = str(path or "").strip()
        self._camera_preview_sources[str(role).strip() or "cam1"] = source if source else None

    def set_camera_preview_source(self, role: str, source: object) -> None:
        self._camera_preview_sources[str(role).strip() or "cam1"] = source

    def roi_statuses_for_camera(self, camera_id: str) -> dict[str, str]:
        return merge_roi_statuses(self._inspection_rows, camera_id=camera_id)

    def clear_camera_views(self) -> None:
        self._active_role_set = set()
        self._camera_preview_sources = {"cam1": None, "cam2": None}
        self.view_cam1.set_runtime_pixmap(None, placeholder="Cam1")
        self.view_cam2.set_runtime_pixmap(None, placeholder="Cam2")
        self.lbl_cam1_timing.setText("Cam1: -")
        self.lbl_cam2_timing.setText("Cam2: -")
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
        from ui.window_common import update_runtime_preview

        for role in ("cam1", "cam2"):
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
        cam1_capture_ms = self._coerce_ms(timing_map.get("cam1_capture_ms"))
        cam1_match_ms = self._coerce_ms(timing_map.get("cam1_match_ms"))
        cam1_infer_ms = self._coerce_ms(timing_map.get("cam1_infer_ms"))
        cam1_total_ms = self._coerce_ms(timing_map.get("cam1_total_ms"))
        cam2_capture_ms = self._coerce_ms(timing_map.get("cam2_capture_ms"))
        cam2_match_ms = self._coerce_ms(timing_map.get("cam2_match_ms"))
        cam2_infer_ms = self._coerce_ms(timing_map.get("cam2_infer_ms"))
        cam2_total_ms = self._coerce_ms(timing_map.get("cam2_total_ms"))

        self.lbl_capture_time.setText(self._format_timing_label(tr("runtime.capture"), capture_ms))
        self.lbl_match_time.setText(self._format_timing_label(tr("runtime.match"), match_ms))
        self.lbl_infer_time.setText(self._format_timing_label(tr("runtime.infer"), infer_ms))
        self._set_total_duration_labels(duration_ms)
        self.lbl_cam1_timing.setText(
            self._format_camera_timing_text("Cam1", cam1_capture_ms, cam1_match_ms, cam1_infer_ms, cam1_total_ms)
        )
        self.lbl_cam2_timing.setText(
            self._format_camera_timing_text("Cam2", cam2_capture_ms, cam2_match_ms, cam2_infer_ms, cam2_total_ms)
        )
        self._refresh_camera_timing_visibility()

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
        display_roles = self._display_role_set()
        self.lbl_cam1_timing.setVisible("cam1" in display_roles)
        self.lbl_cam2_timing.setVisible("cam2" in display_roles)

    def _display_role_set(self) -> set[str]:
        return set(self._active_role_set or self._configured_role_set)

    def _refresh_camera_role_layout(self) -> None:
        show_cam2 = "cam2" in self._display_role_set()
        self.view_cam2.setVisible(show_cam2)
        if hasattr(self, "_camera_splitter"):
            self._camera_splitter.setSizes([1, 1] if show_cam2 else [1, 0])

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
        allow_cam1 = (not self._busy) and ("cam1" in self._active_role_set) and self._has_enabled_items("cam1")
        allow_cam2 = (
            (not self._busy)
            and ("cam2" in self._configured_role_set)
            and ("cam2" in self._active_role_set)
            and self._has_enabled_items("cam2")
        )
        allow_full_trigger = (not self._busy) and any(
            self._has_enabled_items(role) for role in self._active_role_set
        )
        self.btn_simulate_foot.setEnabled(allow_full_trigger)
        self.btn_trigger_cam1.setEnabled(allow_cam1)
        self.btn_trigger_cam2.setEnabled(allow_cam2)

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
                            final_result = str(row.get("final_result", "")).strip().upper()
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
