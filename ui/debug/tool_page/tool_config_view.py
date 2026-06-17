from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr


def build_tool_config_panel(owner, right_vbox: QtWidgets.QVBoxLayout, *, styles: dict[str, str]) -> None:
    text_light = styles["text_light"]
    text_dim = styles["text_dim"]
    input_style = styles["input_style"]
    compact_btn = styles["compact_btn"]
    label_style = styles["label_style"]

    tool_gap = QtWidgets.QWidget()
    tool_gap.setFixedHeight(14)
    right_vbox.addWidget(tool_gap)

    owner.btn_toggle_tools = QtWidgets.QToolButton()
    owner.btn_toggle_tools.setText(tr("debug.inspection_tools"))
    owner.btn_toggle_tools.setCheckable(True)
    owner.btn_toggle_tools.setChecked(True)
    owner.btn_toggle_tools.setArrowType(QtCore.Qt.ArrowType.DownArrow)
    owner.btn_toggle_tools.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    owner.btn_toggle_tools.setStyleSheet(
        (
            f"QToolButton{{background:#404040;color:{text_light};font-size:12px;"
            "font-weight:bold;border:none;border-bottom:1px solid #505050;padding:6px 10px;}"
            "QToolButton:hover{background:#474747;}"
        )
    )
    owner.btn_toggle_tools.toggled.connect(owner._toggle_tool_config_section)
    right_vbox.addWidget(owner.btn_toggle_tools)

    tool_frame = QtWidgets.QWidget()
    tool_vbox = QtWidgets.QVBoxLayout(tool_frame)
    tool_vbox.setContentsMargins(0, 0, 0, 0)
    tool_vbox.setSpacing(4)

    owner.lbl_tool_config_hint = QtWidgets.QLabel("")
    owner.lbl_tool_config_hint.setStyleSheet(f"color:{text_dim};font-size:11px;")
    tool_vbox.addWidget(owner.lbl_tool_config_hint)

    owner.measurement_params_frame = QtWidgets.QWidget()
    measurement_form = QtWidgets.QFormLayout(owner.measurement_params_frame)
    owner.measurement_form = measurement_form
    measurement_form.setContentsMargins(8, 6, 8, 6)
    measurement_form.setSpacing(6)
    owner._measurement_params_loading = False

    owner.chk_measurement_lower = QtWidgets.QCheckBox(tr("debug.measurement.use_lower"))
    owner.chk_measurement_upper = QtWidgets.QCheckBox(tr("debug.measurement.use_upper"))
    owner.spin_measurement_lower = QtWidgets.QDoubleSpinBox()
    owner.spin_measurement_upper = QtWidgets.QDoubleSpinBox()
    owner.spin_measurement_pixel_size = QtWidgets.QDoubleSpinBox()
    owner.cmb_measurement_line_a_tool = QtWidgets.QComboBox()
    owner.cmb_measurement_line_b_tool = QtWidgets.QComboBox()
    for combo in (owner.cmb_measurement_line_a_tool, owner.cmb_measurement_line_b_tool):
        combo.setStyleSheet(input_style)
        combo.currentIndexChanged.connect(owner._on_measurement_params_changed)

    owner.cmb_measurement_distance_mode = QtWidgets.QComboBox()
    owner.cmb_measurement_distance_mode.addItem(tr("debug.measurement.distance_mode.vertical"), "vertical")
    owner.cmb_measurement_distance_mode.addItem(tr("debug.measurement.distance_mode.horizontal"), "horizontal")
    owner.cmb_measurement_distance_mode.addItem(tr("debug.measurement.distance_mode.euclidean"), "euclidean")
    owner.cmb_measurement_distance_mode.setStyleSheet(input_style)
    owner.cmb_measurement_distance_mode.currentIndexChanged.connect(owner._on_measurement_params_changed)

    owner.cmb_measurement_line_a_direction = QtWidgets.QComboBox()
    owner.cmb_measurement_line_b_direction = QtWidgets.QComboBox()
    for combo in (owner.cmb_measurement_line_a_direction, owner.cmb_measurement_line_b_direction):
        combo.addItem(tr("debug.measurement.direction.left_right"), "left_right")
        combo.addItem(tr("debug.measurement.direction.right_left"), "right_left")
        combo.addItem(tr("debug.measurement.direction.top_down"), "top_down")
        combo.addItem(tr("debug.measurement.direction.bottom_up"), "bottom_up")
        combo.setStyleSheet(input_style)
        combo.currentIndexChanged.connect(owner._on_measurement_params_changed)

    owner.cmb_measurement_polarity = QtWidgets.QComboBox()
    owner.cmb_measurement_polarity.addItem(tr("debug.measurement.polarity.any"), "any")
    owner.cmb_measurement_polarity.addItem(tr("debug.measurement.polarity.dark_to_bright"), "dark_to_bright")
    owner.cmb_measurement_polarity.addItem(tr("debug.measurement.polarity.bright_to_dark"), "bright_to_dark")
    owner.cmb_measurement_polarity.setStyleSheet(input_style)
    owner.cmb_measurement_polarity.currentIndexChanged.connect(owner._on_measurement_params_changed)

    owner.spin_measurement_edge_threshold = QtWidgets.QDoubleSpinBox()
    owner.spin_measurement_edge_threshold.setDecimals(3)
    owner.spin_measurement_edge_threshold.setRange(0.0, 255.0)
    owner.spin_measurement_edge_threshold.setSingleStep(1.0)
    owner.spin_measurement_edge_threshold.setValue(10.0)
    owner.spin_measurement_scan_step = QtWidgets.QSpinBox()
    owner.spin_measurement_scan_step.setRange(1, 1000)
    owner.spin_measurement_scan_step.setValue(2)
    owner.spin_measurement_min_points = QtWidgets.QSpinBox()
    owner.spin_measurement_min_points.setRange(2, 100000)
    owner.spin_measurement_min_points.setValue(10)
    for spin in (
        owner.spin_measurement_edge_threshold,
        owner.spin_measurement_scan_step,
        owner.spin_measurement_min_points,
    ):
        spin.setKeyboardTracking(False)
        spin.setStyleSheet(input_style)
        spin.valueChanged.connect(owner._on_measurement_params_changed)

    for spin in (owner.spin_measurement_lower, owner.spin_measurement_upper):
        spin.setDecimals(4)
        spin.setRange(-1000000.0, 1000000.0)
        spin.setSingleStep(0.01)
        spin.setKeyboardTracking(False)
        spin.setStyleSheet(input_style)
        spin.valueChanged.connect(owner._on_measurement_params_changed)

    for checkbox in (owner.chk_measurement_lower, owner.chk_measurement_upper):
        checkbox.setStyleSheet(f"color:{text_light};font-size:12px;")
        checkbox.toggled.connect(owner._on_measurement_params_changed)

    owner.spin_measurement_pixel_size.setDecimals(9)
    owner.spin_measurement_pixel_size.setRange(0.0, 1000000.0)
    owner.spin_measurement_pixel_size.setSingleStep(0.000001)
    owner.spin_measurement_pixel_size.setKeyboardTracking(False)
    owner.spin_measurement_pixel_size.setStyleSheet(input_style)
    owner.spin_measurement_pixel_size.valueChanged.connect(owner._on_measurement_params_changed)

    owner.cmb_measurement_unit = QtWidgets.QComboBox()
    owner.cmb_measurement_unit.addItems(["px", "mm"])
    owner.cmb_measurement_unit.setStyleSheet(input_style)
    owner.cmb_measurement_unit.currentTextChanged.connect(owner._on_measurement_params_changed)

    measurement_form.addRow(tr("debug.measurement.line_a_tool"), owner.cmb_measurement_line_a_tool)
    measurement_form.addRow(tr("debug.measurement.line_b_tool"), owner.cmb_measurement_line_b_tool)
    measurement_form.addRow(tr("debug.measurement.distance_mode"), owner.cmb_measurement_distance_mode)
    measurement_form.addRow(tr("debug.measurement.line_a_direction"), owner.cmb_measurement_line_a_direction)
    measurement_form.addRow(tr("debug.measurement.line_b_direction"), owner.cmb_measurement_line_b_direction)
    measurement_form.addRow(tr("debug.measurement.polarity"), owner.cmb_measurement_polarity)
    measurement_form.addRow(tr("debug.measurement.edge_threshold"), owner.spin_measurement_edge_threshold)
    measurement_form.addRow(tr("debug.measurement.scan_step"), owner.spin_measurement_scan_step)
    measurement_form.addRow(tr("debug.measurement.min_points"), owner.spin_measurement_min_points)
    measurement_form.addRow(owner.chk_measurement_lower, owner.spin_measurement_lower)
    measurement_form.addRow(owner.chk_measurement_upper, owner.spin_measurement_upper)
    measurement_form.addRow(tr("debug.measurement.unit"), owner.cmb_measurement_unit)
    measurement_form.addRow(tr("debug.measurement.pixel_size"), owner.spin_measurement_pixel_size)
    owner.measurement_params_frame.hide()
    tool_vbox.addWidget(owner.measurement_params_frame)

    measurement_action_row = QtWidgets.QHBoxLayout()
    measurement_action_row.setContentsMargins(0, 0, 0, 0)
    measurement_action_row.setSpacing(4)

    owner.btn_add_line_distance_tool = QtWidgets.QPushButton(tr("debug.measurement.add_line_distance_tool"))
    owner.btn_add_line_distance_tool.setStyleSheet(compact_btn)
    owner.btn_add_line_distance_tool.clicked.connect(owner._add_line_distance_tool)
    measurement_action_row.addWidget(owner.btn_add_line_distance_tool, 1)

    owner.btn_delete_line_distance_tool = QtWidgets.QPushButton(tr("debug.measurement.delete_line_distance_tool"))
    owner.btn_delete_line_distance_tool.setStyleSheet(compact_btn)
    owner.btn_delete_line_distance_tool.setEnabled(False)
    owner.btn_delete_line_distance_tool.hide()
    owner.btn_delete_line_distance_tool.clicked.connect(owner._delete_selected_line_distance_tool)
    measurement_action_row.addWidget(owner.btn_delete_line_distance_tool, 1)
    tool_vbox.addLayout(measurement_action_row)

    owner.tool_config_frame = tool_frame
    owner.inspection_items_table = QtWidgets.QTableWidget(0, 5)
    owner.inspection_items_table.setHorizontalHeaderLabels([
        tr("debug.tools_table.enabled"),
        tr("debug.tools_table.name"),
        tr("debug.tools_table.camera"),
        tr("debug.tools_table.algorithm"),
        tr("debug.tools_table.status"),
    ])
    owner.inspection_items_table.verticalHeader().setVisible(False)
    owner.inspection_items_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    owner.inspection_items_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    header = owner.inspection_items_table.horizontalHeader()
    header.setStretchLastSection(False)
    owner.btn_toggle_tools.setText(tr("debug.inspection_tools"))
    for column in (1, 2, 3, 4):
        header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Stretch)
    owner.inspection_items_table.setStyleSheet(
        "QTableWidget{background:#333333;color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;font-size:12px;}"
        "QTableWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
        "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:4px;}"
    )
    owner.inspection_items_table.setMinimumHeight(170)
    owner.inspection_items_table.setColumnWidth(0, 52)
    owner.inspection_items_table.itemChanged.connect(owner._on_inspection_items_table_item_changed)
    owner.inspection_items_table.itemSelectionChanged.connect(owner._on_inspection_items_selection_changed)
    tool_vbox.addWidget(owner.inspection_items_table)

    owner.tool_config_scroll = QtWidgets.QScrollArea()
    owner.tool_config_scroll.setWidgetResizable(True)
    owner.tool_config_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    owner.tool_config_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    owner.tool_config_scroll.setStyleSheet(
        "QScrollArea{background:#2f2f2f;border:none;}"
        "QScrollArea > QWidget > QWidget{background:#2f2f2f;}"
        "QScrollBar:vertical{background:#2f2f2f;width:10px;margin:0;}"
        "QScrollBar::handle:vertical{background:#5a5a5a;min-height:28px;border-radius:5px;}"
        "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
    )
    owner.tool_config_scroll.setWidget(tool_frame)
    owner.tool_config_scroll.setMinimumHeight(220)
    right_vbox.addWidget(owner.tool_config_scroll, 1)

    owner.lbl_current_tool_sample_stats = QtWidgets.QLabel(f"  {tr('debug.current_tool_stats_select')}")
    owner.lbl_current_tool_sample_stats.setWordWrap(True)
    owner.lbl_current_tool_sample_stats.setStyleSheet(
        f"color:{text_dim};font-size:11px;padding:6px 10px;border-top:1px solid #505050;border-bottom:1px solid #505050;"
    )
    right_vbox.addWidget(owner.lbl_current_tool_sample_stats)
    owner._update_learning_backbone_hint()
