from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr


def build_io_debug_page(
    owner,
    *,
    styles: dict[str, str],
    standard_icon,
    standard_pixmap,
) -> None:
    dark_bg = styles["dark_bg"]
    panel_bg = styles["panel_bg"]
    text_light = styles["text_light"]
    text_dim = styles["text_dim"]
    compact_btn = styles["compact_btn"]

    owner.io_debug_page = QtWidgets.QWidget()
    owner.io_debug_page.setStyleSheet(f"background:{dark_bg};color:{text_light};")
    io_main = QtWidgets.QHBoxLayout(owner.io_debug_page)
    io_main.setContentsMargins(0, 0, 0, 0)
    io_main.setSpacing(0)

    io_left = QtWidgets.QFrame()
    io_left.setFixedWidth(260)
    io_left.setStyleSheet(f"QFrame{{background:{panel_bg};border-right:1px solid #505050;}}")
    io_left_vbox = QtWidgets.QVBoxLayout(io_left)
    io_left_vbox.setContentsMargins(0, 0, 0, 0)
    io_left_vbox.setSpacing(0)

    owner.lbl_io_ctrl_title = QtWidgets.QLabel(tr("debug.connection_control"))
    owner.lbl_io_ctrl_title.setFixedHeight(28)
    owner.lbl_io_ctrl_title.setStyleSheet(
        f"background:#404040;color:{text_light};font-size:12px;font-weight:bold;"
        "border-bottom:1px solid #505050;padding-left:8px;"
    )
    io_left_vbox.addWidget(owner.lbl_io_ctrl_title)

    io_ctrl_w = QtWidgets.QWidget()
    io_ctrl_layout = QtWidgets.QVBoxLayout(io_ctrl_w)
    io_ctrl_layout.setContentsMargins(10, 10, 10, 10)
    io_ctrl_layout.setSpacing(6)

    owner.btn_debug_open_io = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DriveNetIcon),
        tr("debug.open_io_debug"),
    )
    owner.btn_debug_open_io.setStyleSheet(compact_btn)
    owner.btn_debug_open_io.clicked.connect(owner._open_debug_io)
    io_ctrl_layout.addWidget(owner.btn_debug_open_io)

    owner.btn_debug_close_io = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_DialogCloseButton),
        tr("debug.close_io_debug"),
    )
    owner.btn_debug_close_io.setStyleSheet(compact_btn)
    owner.btn_debug_close_io.clicked.connect(owner._close_debug_io)
    io_ctrl_layout.addWidget(owner.btn_debug_close_io)

    owner.btn_debug_refresh_io = QtWidgets.QPushButton(
        standard_icon(standard_pixmap.SP_BrowserReload),
        tr("debug.refresh_dido"),
    )
    owner.btn_debug_refresh_io.setStyleSheet(compact_btn)
    owner.btn_debug_refresh_io.clicked.connect(owner._refresh_debug_io_snapshot)
    io_ctrl_layout.addWidget(owner.btn_debug_refresh_io)

    io_left_vbox.addWidget(io_ctrl_w)
    io_left_vbox.addSpacing(4)

    owner.lbl_io_status_title = QtWidgets.QLabel(tr("debug.io_status"))
    owner.lbl_io_status_title.setFixedHeight(28)
    owner.lbl_io_status_title.setStyleSheet(
        f"background:#404040;color:{text_light};font-size:12px;font-weight:bold;"
        "border-bottom:1px solid #505050;border-top:1px solid #505050;padding-left:8px;"
    )
    io_left_vbox.addWidget(owner.lbl_io_status_title)

    io_status_w = QtWidgets.QWidget()
    io_status_layout = QtWidgets.QVBoxLayout(io_status_w)
    io_status_layout.setContentsMargins(10, 10, 10, 10)
    io_status_layout.setSpacing(6)

    owner.lbl_debug_di_snapshot = QtWidgets.QLabel(tr("debug.di_disconnected"))
    owner.lbl_debug_di_snapshot.setWordWrap(True)
    owner.lbl_debug_di_snapshot.setStyleSheet(f"color:{text_dim};font-size:11px;")
    io_status_layout.addWidget(owner.lbl_debug_di_snapshot)

    owner.lbl_debug_do_snapshot = QtWidgets.QLabel(tr("debug.do_disconnected"))
    owner.lbl_debug_do_snapshot.setWordWrap(True)
    owner.lbl_debug_do_snapshot.setStyleSheet(f"color:{text_dim};font-size:11px;")
    io_status_layout.addWidget(owner.lbl_debug_do_snapshot)

    owner.lbl_debug_io_mapping_summary = QtWidgets.QLabel(tr("debug.mapping_not_loaded"))
    owner.lbl_debug_io_mapping_summary.setWordWrap(True)
    owner.lbl_debug_io_mapping_summary.setStyleSheet(f"color:{text_dim};font-size:11px;")
    io_status_layout.addWidget(owner.lbl_debug_io_mapping_summary)

    io_left_vbox.addWidget(io_status_w)
    io_left_vbox.addStretch(1)
    io_main.addWidget(io_left)

    io_right = QtWidgets.QWidget()
    io_right_vbox = QtWidgets.QVBoxLayout(io_right)
    io_right_vbox.setContentsMargins(0, 0, 0, 0)
    io_right_vbox.setSpacing(12)

    owner.lbl_io_panel_title = QtWidgets.QLabel(tr("debug.dido_panel"))
    owner.lbl_io_panel_title.setFixedHeight(28)
    owner.lbl_io_panel_title.setStyleSheet(
        f"background:#404040;color:{text_light};font-size:12px;font-weight:bold;"
        "border-bottom:1px solid #505050;padding-left:8px;"
    )
    io_right_vbox.addWidget(owner.lbl_io_panel_title)

    io_panel_w = QtWidgets.QWidget()
    io_panel_w.setStyleSheet(f"background:{dark_bg};")
    io_panel_layout = QtWidgets.QVBoxLayout(io_panel_w)
    io_panel_layout.setContentsMargins(16, 16, 16, 16)
    io_panel_layout.setSpacing(16)

    channel_card_css = f"QFrame{{background:{panel_bg};border:1px solid #4f4f4f;border-radius:8px;}}"
    di_indicator_off = "background:#7a7a7a;border:2px solid #9a9a9a;border-radius:16px;"
    do_button_css = (
        "QPushButton{background:#4a4a4a;color:#d8d8d8;border:1px solid #666666;"
        "border-radius:6px;padding:8px 6px;font-size:12px;font-weight:bold;}"
        "QPushButton:hover:!disabled{background:#5b5b5b;}"
        "QPushButton:checked{background:#1f9d55;color:white;border:1px solid #1f9d55;}"
        "QPushButton:disabled{background:#363636;color:#737373;border:1px solid #474747;}"
    )

    owner.lbl_di_title = QtWidgets.QLabel(tr("debug.di_monitor"))
    owner.lbl_di_title.setStyleSheet(f"color:{text_light};font-size:13px;font-weight:bold;")
    io_panel_layout.addWidget(owner.lbl_di_title)

    di_grid = QtWidgets.QGridLayout()
    di_grid.setHorizontalSpacing(10)
    di_grid.setVerticalSpacing(10)
    for channel in range(16):
        card = QtWidgets.QFrame()
        card.setStyleSheet(channel_card_css)
        card.setVisible(False)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        title = QtWidgets.QLabel(f"DI_{channel}")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{text_light};font-size:11px;font-weight:bold;border:none;")
        card_layout.addWidget(title)

        indicator = QtWidgets.QLabel()
        indicator.setFixedSize(32, 32)
        indicator.setStyleSheet(di_indicator_off)
        indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(indicator, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        hint = QtWidgets.QLabel(tr("debug.unmapped"))
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{text_dim};font-size:9px;border:none;")
        card_layout.addWidget(hint)

        owner._debug_di_cards[channel] = card
        owner._debug_di_indicators[channel] = indicator
        owner._debug_di_hints[channel] = hint
        di_grid.addWidget(card, channel // 8, channel % 8)

    io_panel_layout.addLayout(di_grid)

    owner.lbl_do_title = QtWidgets.QLabel(tr("debug.do_control"))
    owner.lbl_do_title.setStyleSheet(f"color:{text_light};font-size:13px;font-weight:bold;")
    io_panel_layout.addWidget(owner.lbl_do_title)

    do_grid = QtWidgets.QGridLayout()
    do_grid.setHorizontalSpacing(10)
    do_grid.setVerticalSpacing(10)
    for channel in range(16):
        card = QtWidgets.QFrame()
        card.setStyleSheet(channel_card_css)
        card.setVisible(False)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        button = QtWidgets.QPushButton(f"DO_{channel}")
        button.setCheckable(True)
        button.setEnabled(False)
        button.setMinimumHeight(34)
        button.setStyleSheet(do_button_css)
        button.toggled.connect(
            lambda checked, do_channel=channel: owner._set_debug_output_channel(do_channel, checked)
        )
        card_layout.addWidget(button)

        hint = QtWidgets.QLabel(tr("debug.unmapped"))
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{text_dim};font-size:9px;border:none;")
        card_layout.addWidget(hint)

        owner._debug_do_cards[channel] = card
        owner._debug_do_channel_buttons[channel] = button
        owner._debug_do_hints[channel] = hint
        do_grid.addWidget(card, channel // 8, channel % 8)

    io_panel_layout.addLayout(do_grid)
    io_panel_layout.addStretch(1)

    io_right_vbox.addWidget(io_panel_w)
    io_right_vbox.addStretch(1)
    io_main.addWidget(io_right, 1)
