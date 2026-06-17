from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr


def build_sample_panel(owner, right_vbox: QtWidgets.QVBoxLayout, *, styles: dict[str, str]) -> None:
    text_light = styles["text_light"]
    text_dim = styles["text_dim"]
    section_style = styles["section_style"]
    compact_btn = styles["compact_btn"]

    owner.lbl_images_section = QtWidgets.QLabel(tr("debug.image_list"))
    owner.lbl_images_section.setFixedHeight(28)
    owner.lbl_images_section.setStyleSheet(section_style)
    right_vbox.addWidget(owner.lbl_images_section)

    owner.tabs = QtWidgets.QTabWidget()
    owner.tabs.setStyleSheet(
        "QTabWidget::pane{border:none;}"
        f"QTabBar::tab{{background:#3a3a3a;color:{text_dim};padding:4px 14px;border:none;font-size:12px;}}"
        f"QTabBar::tab:selected{{background:#4a4a4a;color:{text_light};border-bottom:2px solid #3794ff;}}"
    )
    owner.tabs.currentChanged.connect(owner._on_tab_changed)

    list_style = (
        f"QListWidget{{background:#333333;color:{text_light};border:none;font-size:12px;outline:0;}}"
        "QListWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
        "QListWidget::item:hover:!selected{background:#4a4a4a;}"
    )

    owner.ok_list = QtWidgets.QListWidget()
    owner.ok_list.setStyleSheet(list_style)
    owner.ok_list.setFocusPolicy(QtCore.Qt.NoFocus)
    owner.ok_list.itemSelectionChanged.connect(owner._on_select_ok)

    train_tab = QtWidgets.QWidget()
    train_layout = QtWidgets.QVBoxLayout(train_tab)
    train_layout.setContentsMargins(4, 4, 4, 4)
    train_layout.setSpacing(4)
    train_layout.addWidget(owner.ok_list, 1)

    train_actions = QtWidgets.QGridLayout()
    train_actions.setHorizontalSpacing(4)
    train_actions.setVerticalSpacing(4)
    owner.btn_import_train = QtWidgets.QPushButton(tr("debug.add_external_images"))
    owner.btn_import_train.setStyleSheet(compact_btn)
    owner.btn_import_train.clicked.connect(lambda: owner._add_images_to("TRAIN"))
    owner.btn_train_to_test = QtWidgets.QPushButton(tr("debug.move_to_test"))
    owner.btn_train_to_test.setStyleSheet(compact_btn)
    owner.btn_train_to_test.clicked.connect(lambda: owner._move_selected_sample_to("TEST"))
    owner.btn_sample_annotation = QtWidgets.QPushButton(tr("debug.sample_annotation"))
    owner.btn_sample_annotation.setStyleSheet(compact_btn)
    owner.btn_sample_annotation.clicked.connect(owner._open_sample_annotation_dialog)
    owner.btn_del_ok = QtWidgets.QPushButton(tr("debug.remove"))
    owner.btn_del_ok.setStyleSheet(compact_btn)
    owner.btn_del_ok.clicked.connect(lambda: owner._remove_selected_from("TRAIN"))
    train_actions.addWidget(owner.btn_import_train, 0, 0)
    train_actions.addWidget(owner.btn_train_to_test, 0, 1)
    train_actions.addWidget(owner.btn_sample_annotation, 1, 0)
    train_actions.addWidget(owner.btn_del_ok, 1, 1)
    train_layout.addLayout(train_actions)
    owner.tabs.addTab(train_tab, tr("debug.train_samples"))

    owner.ng_list = QtWidgets.QListWidget(owner)
    owner.ng_list.setStyleSheet(list_style)
    owner.ng_list.hide()

    owner.test_list = QtWidgets.QListWidget()
    owner.test_list.setStyleSheet(list_style)
    owner.test_list.setFocusPolicy(QtCore.Qt.NoFocus)
    owner.test_list.itemSelectionChanged.connect(owner._on_select_test)

    test_tab = QtWidgets.QWidget()
    test_layout = QtWidgets.QVBoxLayout(test_tab)
    test_layout.setContentsMargins(4, 4, 4, 4)
    test_layout.setSpacing(4)
    test_layout.addWidget(owner.test_list, 1)

    test_actions = QtWidgets.QGridLayout()
    test_actions.setHorizontalSpacing(4)
    test_actions.setVerticalSpacing(4)
    owner.btn_test_to_train = QtWidgets.QPushButton(tr("debug.move_to_train"))
    owner.btn_test_to_train.setStyleSheet(compact_btn)
    owner.btn_test_to_train.clicked.connect(lambda: owner._move_selected_sample_to("TRAIN"))
    owner.btn_add_test = QtWidgets.QPushButton(tr("debug.add_external_images"))
    owner.btn_add_test.setStyleSheet(compact_btn)
    owner.btn_add_test.clicked.connect(lambda: owner._add_images_to("TEST"))
    owner.btn_del_test = QtWidgets.QPushButton(tr("debug.remove"))
    owner.btn_del_test.setStyleSheet(compact_btn)
    owner.btn_del_test.clicked.connect(lambda: owner._remove_selected_from("TEST"))
    owner.btn_sample_annotation_test = QtWidgets.QPushButton(tr("debug.clear_current_test_list"))
    owner.btn_sample_annotation_test.setStyleSheet(compact_btn)
    owner.btn_sample_annotation_test.clicked.connect(owner._clear_current_test_list)
    test_actions.addWidget(owner.btn_test_to_train, 0, 0)
    test_actions.addWidget(owner.btn_add_test, 0, 1)
    test_actions.addWidget(owner.btn_sample_annotation_test, 1, 0)
    test_actions.addWidget(owner.btn_del_test, 1, 1)
    test_layout.addLayout(test_actions)
    owner.tabs.addTab(test_tab, tr("debug.test_samples"))
    right_vbox.addWidget(owner.tabs, 1)

    owner.lbl_current_image_sample_state = QtWidgets.QLabel(f"  {tr('debug.current_image_state_none')}")
    owner.lbl_current_image_sample_state.setWordWrap(True)
    owner.lbl_current_image_sample_state.setStyleSheet(
        f"color:{text_dim};font-size:11px;padding:4px 10px 8px 10px;border-bottom:1px solid #505050;"
    )
    right_vbox.addWidget(owner.lbl_current_image_sample_state)
