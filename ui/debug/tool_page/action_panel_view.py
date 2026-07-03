from __future__ import annotations

from PySide6 import QtWidgets

from ui.i18n import tr


def build_action_panel(
    owner,
    right_vbox: QtWidgets.QVBoxLayout,
    *,
    styles: dict[str, str],
    standard_icon,
    standard_pixmap,
) -> None:
    text_light = styles["text_light"]
    text_dim = styles["text_dim"]
    section_style = styles["section_style"]
    compact_btn = styles["compact_btn"]

    owner.lbl_action_section = QtWidgets.QLabel(tr("debug.actions"))
    owner.lbl_action_section.setFixedHeight(28)
    owner.lbl_action_section.setStyleSheet(section_style)
    right_vbox.addWidget(owner.lbl_action_section)

    action_frame = QtWidgets.QWidget()
    action_vbox = QtWidgets.QVBoxLayout(action_frame)
    action_vbox.setContentsMargins(8, 6, 8, 6)
    action_vbox.setSpacing(4)
    owner.lbl_training_validation = QtWidgets.QLabel(tr("debug.training_validation_select"))
    owner.lbl_training_validation.setWordWrap(True)
    owner.lbl_training_validation.setStyleSheet(f"color:{text_dim};font-size:11px;padding:0 2px 4px 2px;")
    action_vbox.addWidget(owner.lbl_training_validation)
    if getattr(owner, "lite_mode", False):
        owner.training_progress_bar = QtWidgets.QProgressBar()
        owner.training_progress_bar.setRange(0, 100)
        owner.training_progress_bar.setValue(0)
        owner.training_progress_bar.setTextVisible(True)
        owner.training_progress_bar.setFormat("")
        owner.training_progress_bar.setFixedHeight(16)
        owner.training_progress_bar.setVisible(False)
        owner.training_progress_bar.setStyleSheet(
            "QProgressBar{background:#2f2f2f;color:#e6e6e6;border:1px solid #555;"
            "border-radius:3px;text-align:center;font-size:10px;}"
            "QProgressBar::chunk{background:#2d8cff;border-radius:2px;}"
        )
        action_vbox.addWidget(owner.training_progress_bar)

    action_btn_style = (
        "QPushButton{background:#2d5aa0;color:white;border:none;"
        "padding:6px 12px;border-radius:3px;font-size:13px;font-weight:bold;}"
        "QPushButton:hover{background:#3a6abf;}"
        "QPushButton:pressed{background:#244a85;}"
    )
    confirm_action_btn_style = (
        "QPushButton{background:#b36a19;color:white;border:none;"
        "padding:6px 12px;border-radius:3px;font-size:13px;font-weight:bold;}"
        "QPushButton:hover{background:#ca7b22;}"
        "QPushButton:pressed{background:#985914;}"
    )
    cancel_action_btn_style = (
        "QPushButton{background:#4a4a4a;color:#e0e0e0;border:1px solid #666666;"
        "padding:0px;border-radius:3px;font-size:13px;font-weight:bold;min-width:24px;max-width:24px;min-height:34px;max-height:34px;}"
        "QPushButton:hover{background:#5a5a5a;color:white;}"
        "QPushButton:pressed{background:#3d3d3d;}"
    )

    train_row = QtWidgets.QHBoxLayout()
    train_row.setContentsMargins(0, 0, 0, 0)
    train_row.setSpacing(4)

    owner.btn_train = QtWidgets.QPushButton(standard_icon(standard_pixmap.SP_DialogApplyButton), tr("debug.train_all_tools"))
    owner._train_action_btn_style = action_btn_style
    owner._train_current_btn_style = compact_btn
    owner._train_confirm_btn_style = confirm_action_btn_style
    owner.btn_train.setStyleSheet(owner._train_action_btn_style)
    owner.btn_train.clicked.connect(owner._train_all_tools)
    train_row.addWidget(owner.btn_train, 1)

    owner.btn_train_cancel = QtWidgets.QPushButton("x")
    owner.btn_train_cancel.setToolTip(tr("debug.cancel_train_confirm"))
    owner.btn_train_cancel.setStyleSheet(cancel_action_btn_style)
    owner.btn_train_cancel.setVisible(False)
    owner.btn_train_cancel.clicked.connect(lambda: owner._cancel_training_pending_action("all"))
    train_row.addWidget(owner.btn_train_cancel, 0)
    action_vbox.addLayout(train_row)

    train_current_row = QtWidgets.QHBoxLayout()
    train_current_row.setContentsMargins(0, 0, 0, 0)
    train_current_row.setSpacing(4)

    owner.btn_train_current = QtWidgets.QPushButton(tr("debug.calibrate_current_tool"))
    owner.btn_train_current.setStyleSheet(owner._train_current_btn_style)
    owner.btn_train_current.clicked.connect(owner._train)
    train_current_row.addWidget(owner.btn_train_current, 1)

    owner.btn_train_current_cancel = QtWidgets.QPushButton("x")
    owner.btn_train_current_cancel.setToolTip(tr("debug.cancel_current_confirm"))
    owner.btn_train_current_cancel.setStyleSheet(cancel_action_btn_style)
    owner.btn_train_current_cancel.setVisible(False)
    owner.btn_train_current_cancel.clicked.connect(lambda: owner._cancel_training_pending_action("current"))
    train_current_row.addWidget(owner.btn_train_current_cancel, 0)
    action_vbox.addLayout(train_current_row)

    if getattr(owner, "lite_mode", False):
        owner.btn_export_onnx = QtWidgets.QPushButton(
            standard_icon(standard_pixmap.SP_DialogSaveButton),
            "导出ONNX",
        )
        owner.btn_export_onnx.setStyleSheet(compact_btn)
        owner.btn_export_onnx.clicked.connect(owner._export_current_backbone_onnx)
        action_vbox.addWidget(owner.btn_export_onnx)

    act_row = QtWidgets.QHBoxLayout()
    act_row.setSpacing(4)
    owner.btn_test = QtWidgets.QPushButton(standard_icon(standard_pixmap.SP_MediaPlay), tr("debug.test_current_image"))
    owner.btn_test.setStyleSheet(compact_btn)
    owner.btn_test.clicked.connect(owner._run_test)
    owner.btn_export_test = QtWidgets.QPushButton(standard_icon(standard_pixmap.SP_DialogSaveButton), tr("debug.export_report"))
    owner.btn_export_test.setStyleSheet(compact_btn)
    owner.btn_export_test.clicked.connect(owner._export_current_results_csv)
    owner.btn_clear_session = QtWidgets.QPushButton(standard_icon(standard_pixmap.SP_MediaPlay), tr("debug.test_all_test_samples"))
    owner.btn_clear_session.setStyleSheet(compact_btn)
    owner.btn_clear_session.clicked.connect(owner._run_all_test_samples)
    act_row.addWidget(owner.btn_test)
    act_row.addWidget(owner.btn_export_test)
    act_row.addWidget(owner.btn_clear_session)
    action_vbox.addLayout(act_row)
    right_vbox.addWidget(action_frame)
    owner._sync_training_action_buttons()
