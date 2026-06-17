from __future__ import annotations

from PySide6 import QtWidgets

from ui.i18n import tr
from ui.runtime import RuntimeImageView


def build_camera_debug_page(
    owner,
    *,
    styles: dict[str, str],
    standard_icon,
    standard_pixmap,
) -> None:
    dark_bg = styles["dark_bg"]
    panel_bg = styles["panel_bg"]
    header_bg = styles["header_bg"]
    text_light = styles["text_light"]
    text_dim = styles["text_dim"]
    input_style = styles["input_style"]
    compact_btn = styles["compact_btn"]

    owner.camera_debug_page = QtWidgets.QWidget()
    owner.camera_debug_page.setStyleSheet(f"background:{dark_bg};color:{text_light};")
    cam_main = QtWidgets.QHBoxLayout(owner.camera_debug_page)
    cam_main.setContentsMargins(0, 0, 0, 0)
    cam_main.setSpacing(0)

    cam_left = QtWidgets.QFrame()
    cam_left.setFixedWidth(220)
    cam_left.setStyleSheet(f"QFrame{{background:{panel_bg};border-right:1px solid #505050;}}")
    cam_left_vbox = QtWidgets.QVBoxLayout(cam_left)
    cam_left_vbox.setContentsMargins(0, 0, 0, 0)
    cam_left_vbox.setSpacing(0)

    owner.lbl_cam_left_title = QtWidgets.QLabel(tr("debug.device_list"))
    owner.lbl_cam_left_title.setFixedHeight(28)
    owner.lbl_cam_left_title.setStyleSheet(
        f"background:#404040;color:{text_light};font-size:12px;font-weight:bold;"
        "border-bottom:1px solid #505050;padding-left:8px;"
    )
    cam_left_vbox.addWidget(owner.lbl_cam_left_title)

    role_row = QtWidgets.QWidget()
    role_layout = QtWidgets.QHBoxLayout(role_row)
    role_layout.setContentsMargins(8, 6, 8, 2)
    role_layout.setSpacing(6)
    owner.lbl_debug_role = QtWidgets.QLabel(tr("debug.debug_role"))
    owner.lbl_debug_role.setStyleSheet(f"color:{text_dim};font-size:12px;")
    role_layout.addWidget(owner.lbl_debug_role)
    owner.cmb_debug_camera_role = QtWidgets.QComboBox()
    owner.cmb_debug_camera_role.setStyleSheet(input_style)
    owner.cmb_debug_camera_role.addItem("cam1", "cam1")
    owner.cmb_debug_camera_role.addItem("cam2", "cam2")
    owner.cmb_debug_camera_role.currentIndexChanged.connect(owner._on_debug_camera_role_changed)
    role_layout.addWidget(owner.cmb_debug_camera_role, 1)
    cam_left_vbox.addWidget(role_row)

    owner.cmb_debug_camera = QtWidgets.QComboBox()
    owner.cmb_debug_camera.setStyleSheet(input_style)
    owner.cmb_debug_camera.currentIndexChanged.connect(owner._on_debug_camera_selected)
    cam_left_vbox.addWidget(owner.cmb_debug_camera)

    owner.btn_debug_refresh_camera = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_BrowserReload),
        tr("debug.scan_camera"),
    )
    owner.btn_debug_refresh_camera.setStyleSheet(compact_btn)
    owner.btn_debug_refresh_camera.clicked.connect(owner._refresh_debug_camera_list)
    cam_left_vbox.addWidget(owner.btn_debug_refresh_camera)

    cam_left_vbox.addSpacing(8)
    owner.lbl_cam_info_title = QtWidgets.QLabel(tr("debug.device_info"))
    owner.lbl_cam_info_title.setFixedHeight(28)
    owner.lbl_cam_info_title.setStyleSheet(
        f"background:#404040;color:{text_light};font-size:12px;font-weight:bold;"
        "border-bottom:1px solid #505050;border-top:1px solid #505050;padding-left:8px;"
    )
    cam_left_vbox.addWidget(owner.lbl_cam_info_title)

    owner.lbl_debug_camera_info = QtWidgets.QLabel(tr("debug.camera_info"))
    owner.lbl_debug_camera_info.setWordWrap(True)
    owner.lbl_debug_camera_info.setStyleSheet(f"color:{text_dim};font-size:11px;padding:8px;")
    cam_left_vbox.addWidget(owner.lbl_debug_camera_info)
    owner.lbl_debug_current_role = QtWidgets.QLabel(tr("debug.current_debug_role", role="cam1"))
    owner.lbl_debug_current_role.setStyleSheet(f"color:{text_dim};font-size:11px;padding:0 8px 8px 8px;")
    cam_left_vbox.addWidget(owner.lbl_debug_current_role)
    cam_left_vbox.addStretch(1)
    cam_main.addWidget(cam_left)

    cam_center = QtWidgets.QWidget()
    cam_center_vbox = QtWidgets.QVBoxLayout(cam_center)
    cam_center_vbox.setContentsMargins(0, 0, 0, 0)
    cam_center_vbox.setSpacing(0)

    cam_toolbar = QtWidgets.QFrame()
    cam_toolbar.setFixedHeight(36)
    cam_toolbar.setStyleSheet(f"QFrame{{background:{header_bg};border-bottom:1px solid #505050;}}")
    cam_tb_layout = QtWidgets.QHBoxLayout(cam_toolbar)
    cam_tb_layout.setContentsMargins(8, 2, 8, 2)
    cam_tb_layout.setSpacing(6)

    owner.btn_debug_connect_camera = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DriveNetIcon),
        tr("debug.connect"),
    )
    owner.btn_debug_connect_camera.setStyleSheet(compact_btn)
    owner.btn_debug_connect_camera.clicked.connect(owner._connect_debug_camera)
    cam_tb_layout.addWidget(owner.btn_debug_connect_camera)

    owner.btn_debug_disconnect_camera = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DialogDiscardButton),
        tr("debug.disconnect"),
    )
    owner.btn_debug_disconnect_camera.setStyleSheet(compact_btn)
    owner.btn_debug_disconnect_camera.clicked.connect(owner._disconnect_debug_camera)
    cam_tb_layout.addWidget(owner.btn_debug_disconnect_camera)

    cam_tb_layout.addSpacing(12)
    owner.btn_debug_live_preview = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_MediaPlay),
        tr("debug.live_preview"),
    )
    owner.btn_debug_live_preview.setCheckable(True)
    owner.btn_debug_live_preview.setStyleSheet(compact_btn)
    owner.btn_debug_live_preview.toggled.connect(owner._toggle_debug_camera_preview)
    cam_tb_layout.addWidget(owner.btn_debug_live_preview)

    owner.btn_debug_grab_once = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DesktopIcon),
        tr("debug.grab_to_test"),
    )
    owner.btn_debug_grab_once.setStyleSheet(compact_btn)
    owner.btn_debug_grab_once.clicked.connect(owner._grab_debug_camera_once)
    cam_tb_layout.addWidget(owner.btn_debug_grab_once)

    cam_tb_layout.addSpacing(12)
    owner.btn_debug_save_image = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DialogSaveButton),
        tr("debug.save_image"),
    )
    owner.btn_debug_save_image.setStyleSheet(compact_btn)
    owner.btn_debug_save_image.clicked.connect(owner._save_debug_camera_image)
    cam_tb_layout.addWidget(owner.btn_debug_save_image)

    cam_tb_layout.addStretch(1)
    cam_center_vbox.addWidget(cam_toolbar)

    owner.view_debug_camera = RuntimeImageView(tr("debug.preview_title"))
    owner.view_debug_camera.setMinimumSize(640, 400)
    owner.view_debug_camera.set_runtime_pixmap(None, placeholder=tr("debug.preview_closed"))
    cam_center_vbox.addWidget(owner.view_debug_camera, 1)

    cam_statusbar = QtWidgets.QFrame()
    cam_statusbar.setFixedHeight(26)
    cam_statusbar.setStyleSheet(f"QFrame{{background:{header_bg};border-top:1px solid #505050;}}")
    cam_sb_layout = QtWidgets.QHBoxLayout(cam_statusbar)
    cam_sb_layout.setContentsMargins(10, 0, 10, 0)
    owner.lbl_debug_camera_status = QtWidgets.QLabel(tr("debug.camera_status_unscanned"))
    owner.lbl_debug_camera_status.setWordWrap(False)
    owner.lbl_debug_camera_status.setStyleSheet(f"color:{text_dim};font-size:11px;")
    cam_sb_layout.addWidget(owner.lbl_debug_camera_status)
    cam_center_vbox.addWidget(cam_statusbar)
    cam_main.addWidget(cam_center, 1)

    cam_right = QtWidgets.QFrame()
    cam_right.setFixedWidth(240)
    cam_right.setStyleSheet(f"QFrame{{background:{panel_bg};border-left:1px solid #505050;}}")
    cam_right_vbox = QtWidgets.QVBoxLayout(cam_right)
    cam_right_vbox.setContentsMargins(0, 0, 0, 0)
    cam_right_vbox.setSpacing(0)

    owner.lbl_cam_right_title = QtWidgets.QLabel(tr("debug.param_settings"))
    owner.lbl_cam_right_title.setFixedHeight(28)
    owner.lbl_cam_right_title.setStyleSheet(
        f"background:#404040;color:{text_light};font-size:12px;font-weight:bold;"
        "border-bottom:1px solid #505050;padding-left:8px;"
    )
    cam_right_vbox.addWidget(owner.lbl_cam_right_title)

    cam_params = QtWidgets.QWidget()
    cam_params_form = QtWidgets.QFormLayout(cam_params)
    owner.cam_params_form = cam_params_form
    cam_params_form.setContentsMargins(12, 12, 12, 12)
    cam_params_form.setSpacing(10)
    owner.view_debug_camera.set_runtime_pixmap(None, placeholder=tr("debug.preview_closed"))

    owner.spin_debug_exposure = QtWidgets.QDoubleSpinBox()
    owner.spin_debug_exposure.setDecimals(1)
    owner.spin_debug_exposure.setRange(1.0, 1000000.0)
    owner.spin_debug_exposure.setValue(20000.0)
    owner.spin_debug_exposure.setStyleSheet(input_style)
    owner._debug_exposure_row_label = tr("debug.exposure")
    cam_params_form.addRow(owner._debug_exposure_row_label, owner.spin_debug_exposure)
    owner.lbl_debug_camera_status = QtWidgets.QLabel(tr("debug.camera_status_unscanned"))

    owner.spin_debug_gain = QtWidgets.QDoubleSpinBox()
    owner.spin_debug_gain.setDecimals(2)
    owner.spin_debug_gain.setRange(0.0, 48.0)
    owner.spin_debug_gain.setValue(0.0)
    owner.spin_debug_gain.setStyleSheet(input_style)
    owner._debug_gain_row_label = tr("debug.gain")
    cam_params_form.addRow(owner._debug_gain_row_label, owner.spin_debug_gain)

    owner.chk_debug_digital_shift_enable = QtWidgets.QCheckBox(tr("debug.enable"))
    owner.chk_debug_digital_shift_enable.setStyleSheet(f"color:{text_light};")
    owner._debug_shift_enable_row_label = tr("debug.digital_shift_enable")
    cam_params_form.addRow(owner._debug_shift_enable_row_label, owner.chk_debug_digital_shift_enable)

    owner.spin_debug_digital_shift = QtWidgets.QDoubleSpinBox()
    owner.spin_debug_digital_shift.setDecimals(4)
    owner.spin_debug_digital_shift.setRange(0.0, 16.0)
    owner.spin_debug_digital_shift.setValue(0.0)
    owner.spin_debug_digital_shift.setStyleSheet(input_style)
    owner.spin_debug_digital_shift.setEnabled(False)
    owner.spin_debug_digital_shift.setToolTip("Digital Shift")
    owner._debug_shift_row_label = tr("debug.digital_shift")
    cam_params_form.addRow(owner._debug_shift_row_label, owner.spin_debug_digital_shift)

    owner.spin_debug_exposure.setKeyboardTracking(False)
    owner.spin_debug_gain.setKeyboardTracking(False)
    owner.spin_debug_digital_shift.setKeyboardTracking(False)
    owner.spin_debug_exposure.editingFinished.connect(owner._on_debug_camera_param_editing_finished)
    owner.spin_debug_gain.editingFinished.connect(owner._on_debug_camera_param_editing_finished)
    owner.spin_debug_digital_shift.editingFinished.connect(owner._on_debug_camera_param_editing_finished)
    owner.chk_debug_digital_shift_enable.toggled.connect(owner.spin_debug_digital_shift.setEnabled)
    owner.chk_debug_digital_shift_enable.toggled.connect(
        lambda _checked: owner._on_debug_camera_param_editing_finished()
    )

    owner.cmb_debug_trigger_mode = QtWidgets.QComboBox()
    owner.cmb_debug_trigger_mode.addItems(["software", "continuous"])
    owner.cmb_debug_trigger_mode.setCurrentText("continuous")
    owner.cmb_debug_trigger_mode.setStyleSheet(input_style)
    owner._debug_trigger_row_label = tr("debug.trigger_mode")
    cam_params_form.addRow(owner._debug_trigger_row_label, owner.cmb_debug_trigger_mode)
    owner.cmb_debug_trigger_mode.activated.connect(owner._on_debug_camera_trigger_activated)

    owner.cmb_debug_light_source_mode = QtWidgets.QComboBox()
    owner.cmb_debug_light_source_mode.addItem(tr("debug.board_do_light"), "board_io")
    owner.cmb_debug_light_source_mode.addItem(tr("debug.camera_line1_strobe"), "camera_line1_strobe")
    owner.cmb_debug_light_source_mode.setCurrentIndex(0)
    owner.cmb_debug_light_source_mode.setStyleSheet(input_style)
    owner.cmb_debug_light_source_mode.setToolTip(tr("debug.camera_line1_tip"))
    owner._debug_light_row_label = tr("debug.light_source")
    cam_params_form.addRow(owner._debug_light_row_label, owner.cmb_debug_light_source_mode)
    owner.cmb_debug_light_source_mode.activated.connect(owner._on_debug_camera_trigger_activated)

    cam_right_vbox.addWidget(cam_params)
    cam_right_vbox.addSpacing(8)

    cam_btns_w = QtWidgets.QWidget()
    cam_btns_layout = QtWidgets.QVBoxLayout(cam_btns_w)
    cam_btns_layout.setContentsMargins(12, 0, 12, 12)
    cam_btns_layout.setSpacing(6)

    owner.btn_debug_read_camera_settings = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_FileDialogInfoView),
        tr("debug.read_camera_params"),
    )
    owner.btn_debug_read_camera_settings.setStyleSheet(compact_btn)
    owner.btn_debug_read_camera_settings.clicked.connect(owner._refresh_debug_camera_settings)
    cam_btns_layout.addWidget(owner.btn_debug_read_camera_settings)

    owner.btn_debug_apply_camera_settings = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DialogApplyButton),
        tr("debug.apply_camera_params"),
    )
    owner.btn_debug_apply_camera_settings.setStyleSheet(compact_btn)
    owner.btn_debug_apply_camera_settings.clicked.connect(owner._apply_debug_camera_settings)
    cam_btns_layout.addWidget(owner.btn_debug_apply_camera_settings)

    cam_right_vbox.addWidget(cam_btns_w)
    cam_right_vbox.addStretch(1)
    cam_main.addWidget(cam_right)
