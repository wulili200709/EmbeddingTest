"""
runtime_mode_pyside6.py

运行界面 — 参考基恩士 IV Smart Navigator 风格。

布局：
  ┌──────────────────────────────────────────────────────┐
  │ ▶ 运行中  ●已学习:32模板  ○外部触发   产品: xxx     │  顶栏
  ├──────────────────────────┬───────────────────────────┤
  │                          │  01  PusherL2      [OK]  │
  │   相机画面               │  02  PusherR1      [OK]  │
  │   (1台全屏/2台左右分屏)  │  03  SpringL2     [NG]  │
  │                          │  04  SpringR1      [OK]  │
  │                          │  ...                     │
  │                          ├───────────────────────────┤
  │                          │  总结果:  NG   69ms      │
  ├──────────────────────────┴───────────────────────────┤
  │  处理: 69ms  状态: 等待触发  相机: 已连接cam1,cam2   │  底栏
  └──────────────────────────────────────────────────────┘

所有配置操作（相机序列号、连接/断开、密码放行）移到菜单栏或对话框。
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.roi_overlay_colors import merge_roi_statuses


_DARK_BG = "#2d2d2d"
_PANEL_BG = "#363636"
_HEADER_BG = "#3a3a3a"
_TEXT_LIGHT = "#e0e0e0"
_TEXT_DIM = "#888888"
_OK_GREEN = "#379b37"
_NG_RED = "#dc1e1e"
_PENDING_GRAY = "#666666"
_RUNNING_YELLOW = "#eab308"

_RUN_STATE_FOOTER_ZH = {
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


class RuntimeImageView(QtWidgets.QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._pixmap: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_DIM};font-size:14px;")

    def set_runtime_pixmap(self, pixmap: QtGui.QPixmap | None, *, placeholder: str | None = None) -> None:
        self._pixmap = pixmap
        if pixmap is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText(placeholder or "等待画面")
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
        self.lbl_index.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;font-weight:bold;min-width:24px;")
        self.lbl_index.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_index)

        self.lbl_name = QtWidgets.QLabel(name)
        self.lbl_name.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:14px;")
        layout.addWidget(self.lbl_name, 1)

        self.lbl_result = QtWidgets.QLabel("")
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result.setFixedSize(64, 32)
        self.lbl_result.setStyleSheet(
            f"background:{_PENDING_GRAY};color:white;font-size:14px;font-weight:bold;"
            "border-radius:4px;"
        )
        layout.addWidget(self.lbl_result)

    def set_result(self, status_kind: str, status_text: str) -> None:
        color_map = {
            "ok": _OK_GREEN,
            "ng": _NG_RED,
            "pending": _PENDING_GRAY,
            "running": _RUNNING_YELLOW,
            "disabled": "#444444",
            "inactive": "#444444",
        }
        bg = color_map.get(status_kind, _PENDING_GRAY)
        display = status_text.split("(")[0].strip() if status_text else ""
        self.lbl_result.setText(display)
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

        camera_name = "相机1" if self._camera_id == "cam1" else "相机2"
        self.lbl_title = QtWidgets.QLabel(camera_name)
        self.lbl_title.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:13px;font-weight:bold;")
        layout.addWidget(self.lbl_title)

        layout.addStretch(1)

        self.lbl_result = QtWidgets.QLabel("未检测")
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result.setFixedSize(64, 28)
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
            display = "检测中"
        else:
            bg = _PENDING_GRAY
            display = "未检测"
        self.lbl_result.setText(display)
        self.lbl_result.setStyleSheet(
            f"background:{bg};color:white;font-size:12px;font-weight:bold;border-radius:4px;"
        )


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
        self._active_role_set: set[str] = set()
        self._busy = False
        self._inspection_rows: list[dict] = []
        self._camera_source_paths: dict[str, str] = {"cam1": "", "cam2": ""}
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background:{_DARK_BG};")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶栏 ──
        header = QtWidgets.QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(
            f"background:{_HEADER_BG};border-bottom:1px solid #505050;"
        )
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 16, 0)
        header_layout.setSpacing(8)

        self.lbl_run_indicator = QtWidgets.QLabel("已解锁")
        self.lbl_run_indicator.setStyleSheet(
            f"color:{_OK_GREEN};font-size:15px;font-weight:bold;"
        )
        header_layout.addWidget(self.lbl_run_indicator)

        self.lbl_header_info = QtWidgets.QLabel("○ 外部触发")
        self.lbl_header_info.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;")
        header_layout.addWidget(self.lbl_header_info)

        header_layout.addSpacing(20)

        _trigger_btn_css = (
            "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
            "padding:4px 12px;border-radius:3px;font-size:12px;}"
            "QPushButton:hover{background:#505050;}"
            "QPushButton:pressed{background:#3794ff;color:white;}"
        )
        self.btn_trigger_cam1 = QtWidgets.QPushButton("▶ 触发相机1")
        self.btn_trigger_cam1.setStyleSheet(_trigger_btn_css)
        self.btn_trigger_cam1.setAutoDefault(False)
        self.btn_trigger_cam1.setDefault(False)
        self.btn_trigger_cam1.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_trigger_cam1.setEnabled(False)
        self.btn_trigger_cam1.clicked.connect(lambda: self.triggerCameraRequested.emit(1))
        header_layout.addWidget(self.btn_trigger_cam1)

        self.btn_trigger_cam2 = QtWidgets.QPushButton("▶ 触发相机2")
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
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        camera_frame = QtWidgets.QFrame()
        camera_frame.setStyleSheet(f"background:{_DARK_BG};")
        camera_layout = QtWidgets.QHBoxLayout(camera_frame)
        camera_layout.setContentsMargins(2, 2, 2, 2)
        camera_layout.setSpacing(2)

        self.view_cam1 = RuntimeImageView("Cam1")
        self.view_cam2 = RuntimeImageView("Cam2")
        camera_layout.addWidget(self.view_cam1, 1)
        camera_layout.addWidget(self.view_cam2, 1)
        body.addWidget(camera_frame, 3)

        right_panel = QtWidgets.QFrame()
        right_panel.setFixedWidth(280)
        right_panel.setStyleSheet(
            f"background:{_PANEL_BG};border-left:1px solid #505050;"
        )
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        panel_title = QtWidgets.QLabel("  检测项")
        panel_title.setFixedHeight(32)
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
        total_frame.setStyleSheet(f"background:#404040;border-top:1px solid #5a5a5a;")
        total_layout = QtWidgets.QVBoxLayout(total_frame)
        total_layout.setContentsMargins(12, 8, 12, 8)
        total_layout.setSpacing(6)

        total_header = QtWidgets.QHBoxLayout()
        total_header.setContentsMargins(0, 0, 0, 0)
        total_header.setSpacing(8)

        total_label = QtWidgets.QLabel("总结果")
        total_label.setStyleSheet(f"color:{_TEXT_LIGHT};font-size:14px;font-weight:bold;")
        total_header.addWidget(total_label)

        total_layout.addLayout(total_header)

        timing_grid = QtWidgets.QGridLayout()
        timing_grid.setContentsMargins(0, 0, 0, 0)
        timing_grid.setHorizontalSpacing(10)
        timing_grid.setVerticalSpacing(4)

        self.lbl_capture_time = QtWidgets.QLabel("取图: -")
        self.lbl_capture_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_capture_time, 0, 0)

        self.lbl_match_time = QtWidgets.QLabel("匹配: -")
        self.lbl_match_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_match_time, 0, 1)

        self.lbl_infer_time = QtWidgets.QLabel("推理: -")
        self.lbl_infer_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_infer_time, 1, 0)

        self.lbl_duration = QtWidgets.QLabel("总流程: -")
        self.lbl_duration.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        timing_grid.addWidget(self.lbl_duration, 1, 1)
        total_layout.addLayout(timing_grid)

        self.lbl_final_result = QtWidgets.QLabel("-")
        self.lbl_final_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_final_result.setMinimumHeight(36)
        self.lbl_final_result.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.lbl_final_result.setStyleSheet(
            f"background:{_PENDING_GRAY};color:white;font-size:16px;font-weight:bold;border-radius:4px;"
        )
        total_layout.addWidget(self.lbl_final_result, 1)

        right_layout.addWidget(total_frame)
        body.addWidget(right_panel, 0)

        root.addLayout(body, 1)

        # ── 底栏 ──
        footer = QtWidgets.QFrame()
        footer.setFixedHeight(32)
        footer.setStyleSheet(
            f"background:{_HEADER_BG};border-top:1px solid #505050;"
        )
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 0, 16, 0)
        footer_layout.setSpacing(20)

        self.lbl_footer_time = QtWidgets.QLabel("处理: -")
        self.lbl_footer_time.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_time)

        self.lbl_footer_state = QtWidgets.QLabel("状态: 等待触发")
        self.lbl_footer_state.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_state)

        self.lbl_footer_connection = QtWidgets.QLabel("相机: 未连接")
        self.lbl_footer_connection.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_connection)

        self.lbl_footer_permission = QtWidgets.QLabel("")
        self.lbl_footer_permission.setStyleSheet(f"color:{_TEXT_DIM};font-size:12px;")
        footer_layout.addWidget(self.lbl_footer_permission)

        footer_layout.addStretch(1)

        self.lbl_footer_record = QtWidgets.QLabel("")
        self.lbl_footer_record.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_record)

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
        self.lbl_header_info.setText(f"可见相机: {n} 台" if n else "未发现相机")

    def set_current_product(self, product_name: str) -> None:
        self.lbl_current_product.setText(f"产品: {product_name}" if product_name else "-")

    def set_runtime_state(self, state_text: str) -> None:
        """顶栏：NG 锁或本轮 NG 结束时为「已锁定」红字，其余为「已解锁」绿字；底栏同步状态机中文。"""
        st = str(state_text or "").strip()
        locked_states = frozenset({"LockedByNg", "CompletedNg"})
        if st == "Unavailable":
            text = "未就绪"
            color = _TEXT_DIM
        elif st in locked_states:
            text = "已锁定"
            color = _NG_RED
        else:
            text = "已解锁"
            color = _OK_GREEN
        self.lbl_run_indicator.setText(text)
        self.lbl_run_indicator.setStyleSheet(f"color:{color};font-size:15px;font-weight:bold;")
        footer_detail = _RUN_STATE_FOOTER_ZH.get(st, st)
        self.lbl_footer_state.setText(f"状态: {footer_detail}")

    def set_permission_status(self, status_text: str) -> None:
        self.lbl_footer_permission.setText(f"放行: {status_text}" if status_text else "")

    def set_connection_status(self, status_text: str) -> None:
        self.lbl_footer_connection.setText(f"相机: {status_text}")

    def set_tower_light_status(self, status_text: str) -> None:
        pass

    def set_runtime_status(self, status_text: str) -> None:
        self.lbl_footer_state.setText(f"状态: {status_text}")

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

    def set_record_path(self, record_path: str) -> None:
        self.lbl_footer_record.setText(record_path or "")

    def set_active_camera_roles(self, roles: list[str]) -> None:
        role_set = {str(role).strip() for role in roles if str(role).strip()}
        self._active_role_set = role_set
        show_cam2 = "cam2" in role_set
        self.view_cam2.setVisible(show_cam2)
        if not role_set:
            self.view_cam1.set_runtime_pixmap(None, placeholder="未连接相机")
            self.view_cam2.set_runtime_pixmap(None, placeholder="Cam2")
        self.lbl_footer_connection.setText(
            "相机: " + (", ".join(sorted(role_set)) if role_set else "未连接")
        )
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

        insert_index = 0
        display_index = 1
        for camera_id in ["cam1", "cam2"]:
            camera_rows = grouped_rows.get(camera_id, [])
            if not camera_rows:
                continue

            header = _CameraSectionHeader(camera_id)
            self._items_vbox.insertWidget(insert_index, header)
            self._camera_section_headers[camera_id] = header
            insert_index += 1

            for row in camera_rows:
                name = str(row.get("display_name", ""))
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

    def set_camera_pixmap(self, role: str, pixmap: QtGui.QPixmap | None, *, placeholder: str | None = None) -> None:
        if role == "cam1":
            self.view_cam1.set_runtime_pixmap(pixmap, placeholder=placeholder)
        elif role == "cam2":
            self.view_cam2.set_runtime_pixmap(pixmap, placeholder=placeholder)

    def set_camera_source_path(self, role: str, path: str) -> None:
        self._camera_source_paths[str(role).strip() or "cam1"] = str(path or "").strip()

    def roi_statuses_for_camera(self, camera_id: str) -> dict[str, str]:
        return merge_roi_statuses(self._inspection_rows, camera_id=camera_id)

    def clear_camera_views(self) -> None:
        self._active_role_set = set()
        self._camera_source_paths = {"cam1": "", "cam2": ""}
        self.view_cam1.set_runtime_pixmap(None, placeholder="Cam1")
        self.view_cam2.set_runtime_pixmap(None, placeholder="Cam2")
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

        for role, path in self._camera_source_paths.items():
            if path:
                update_runtime_preview(self, role, path)

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._refresh_trigger_buttons()

    def set_duration_ms(self, ms: int) -> None:
        self._set_total_duration_labels(float(ms or 0.0))

    def set_timing_breakdown(self, timing_map: dict[str, object]) -> None:
        timing_map = dict(timing_map or {})
        capture_ms = self._coerce_ms(timing_map.get("capture_ms"))
        match_ms = self._coerce_ms(timing_map.get("match_ms"))
        infer_ms = self._coerce_ms(timing_map.get("infer_ms"))
        duration_ms = self._coerce_ms(timing_map.get("duration_ms"))

        self.lbl_capture_time.setText(self._format_timing_label("取图", capture_ms))
        self.lbl_match_time.setText(self._format_timing_label("匹配", match_ms))
        self.lbl_infer_time.setText(self._format_timing_label("推理", infer_ms))
        self._set_total_duration_labels(duration_ms)

    @staticmethod
    def _coerce_ms(value: object) -> float:
        try:
            return max(0.0, float(value or 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _format_timing_label(title: str, value: float) -> str:
        return f"{title}: {value:.1f} ms" if value > 0.0 else f"{title}: -"

    def _set_total_duration_labels(self, duration_ms: float) -> None:
        self.lbl_footer_time.setText(
            f"处理: {duration_ms:.1f}ms" if duration_ms > 0.0 else "处理: -"
        )
        self.lbl_duration.setText(self._format_timing_label("总流程", duration_ms))

    def _refresh_trigger_buttons(self) -> None:
        allow_cam1 = (not self._busy) and ("cam1" in self._active_role_set) and self._has_enabled_items("cam1")
        allow_cam2 = (not self._busy) and ("cam2" in self._active_role_set) and self._has_enabled_items("cam2")
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
