from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.image_io import imread
from shape.core.bootstrap import ensure_repo_root_on_path
from shape.core.locator import (
    product_paths,
    resolved_model_path_for_product,
    resolved_recipe_path_for_product,
)
from shape.core.recipe import ShapeRecipe, load_recipe, save_recipe
from shape.core.recipe_store import ShapeRecipeStore
from shape.core.roi_follow import locate_and_follow
from shape.core.template_core import (
    BACKEND_ITEMS,
    BACKEND_KEY_TO_LABEL,
    BACKEND_LABEL_TO_KEY,
    MaskRect,
    RoiRect,
    angle_deg_to_label,
    build_mask_from_rects,
    clone_levels,
    label_to_angle_deg,
    load_class_source_assets,
    normalize_extracted_levels_to_roi,
    parse_levels,
    roi_level_shapes_from_image,
    sync_levels_from_level0,
)
from ui.debug import OverlayShape, RoiCanvas, pixmap_from_path
from ui.i18n import tr
from ui.shape_template.reference_roi_tab import ReferenceRoiTabMixin
from ui.shape_template.template_build_worker import _TemplateBuildWorker
from ui.shape_template.template_page_utils import (
    _arrow_endpoint,
    _button_left,
    _button_right,
    _clamp_rect_to_roi,
    _compact_path_text,
    _cv_to_qpixmap,
    _draw_match_overlay,
    _orientation_palette,
    _overlay_follow_result,
)

ensure_repo_root_on_path()

from shape.like_matcher import (  # noqa: E402
    Feature,
    ShapeLikeDetector,
    TemplateLevel,
    load_detector_model,
)


_DIALOG_STYLESHEET = """
QDialog {
    background: #2d2d2d;
    color: #e0e0e0;
}
QWidget {
    color: #e0e0e0;
}
QGroupBox {
    background: #363636;
    border: 1px solid #505050;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QTabWidget::pane {
    border: 1px solid #505050;
    background: #2d2d2d;
}
QTabBar::tab {
    background: #3a3a3a;
    color: #888888;
    padding: 6px 14px;
    border: 1px solid #505050;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #4a4a4a;
    color: #e0e0e0;
}
QPushButton {
    background: #444444;
    color: #d0d0d0;
    border: 1px solid #5a5a5a;
    padding: 4px 8px;
    border-radius: 3px;
}
QPushButton:hover {
    background: #505050;
}
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QListWidget,
QTableWidget {
    background: #333333;
    color: #e0e0e0;
    border: 1px solid #5a5a5a;
    selection-background-color: #6ec0ff;
    selection-color: #1a1a1a;
}
QHeaderView::section {
    background: #3a3a3a;
    color: #d0d0d0;
    border: 1px solid #404040;
    padding: 4px;
}
QAbstractItemView {
    background: #333333;
    color: #e0e0e0;
    selection-background-color: #6ec0ff;
    selection-color: #1a1a1a;
}
QLabel {
    color: #e0e0e0;
}
"""

_DEFAULT_FIND_BACKEND_LABEL = "Fusion V2"


class ShapeTemplateDialog(ReferenceRoiTabMixin, QtWidgets.QDialog):
    modelSaved = QtCore.Signal(str, str)
    referenceRegionsChanged = QtCore.Signal()

    def __init__(
        self,
        *,
        product_name: str,
        product_dir: str,
        camera_role: str = "cam1",
        initial_image_path: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("template.title", product=product_name))
        available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(
            min(1450, max(1180, available.width() - 40)),
            min(920, max(680, available.height() - 60)),
        )
        self.setMinimumSize(980, 620)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_DIALOG_STYLESHEET)

        self.product_name = product_name
        self.product_dir = product_dir
        self.camera_role = str(camera_role or "cam1").strip() or "cam1"
        self.paths = product_paths(product_dir, self.camera_role)
        self._resolved_recipe_path = resolved_recipe_path_for_product(product_dir, self.camera_role)
        self._resolved_model_path = resolved_model_path_for_product(product_dir, self.camera_role)

        self.image_path = initial_image_path
        self.image_bgr: Optional[np.ndarray] = None

        self.detector: Optional[ShapeLikeDetector] = None
        self.detector_path: str = ""
        self.find_detector: Optional[ShapeLikeDetector] = None
        self.find_detector_path: str = ""
        self.find_model_path = self._resolved_model_path

        self.template_roi: Optional[RoiRect] = None
        self.mask_rects: List[MaskRect] = []
        self.editor_levels: List[TemplateLevel] = []
        self.original_mode = "auto"
        self.points_dirty = False
        self._feature_count = 0
        self._hover_feature_index: Optional[int] = None
        self._point_drag_start: Optional[Tuple[int, int]] = None
        self._point_drag_end: Optional[Tuple[int, int]] = None
        self._syncing_recipe_controls = False
        self._recipe_reference_shape_type: str = ""
        self._recipe_reference_points: List[List[float]] = []
        self._reference_regions: List[Dict[str, object]] = []
        self._selected_reference_idx: Optional[int] = None
        self._syncing_reference_view = False
        self._reference_dirty = False
        self._reference_regions_explicit = False
        self._search_roi_shape_type: str = ""
        self._search_roi_points: List[List[float]] = []
        self._find_result_cache: Dict[str, Dict[str, object]] = {}
        self._suppress_create_auto_apply = False
        self._build_thread: Optional[QtCore.QThread] = None
        self._build_worker: Optional[_TemplateBuildWorker] = None
        self._build_progress: Optional[QtWidgets.QProgressDialog] = None

        self._build_ui()
        self._disable_enter_default_buttons()
        self._load_recipe()
        if self.image_path and os.path.exists(self.image_path):
            self._set_image(self.image_path, reset_state=False)
        if os.path.exists(self.find_model_path):
            self._load_existing_model(silent=True)
        self._update_find_model_label()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self.tab_create = QtWidgets.QWidget()
        self.tab_reference = QtWidgets.QWidget()
        self.tab_find = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_create, tr("template.tab.create"))
        self.tabs.addTab(self.tab_reference, tr("template.tab.reference"))
        self.tabs.addTab(self.tab_find, tr("template.tab.find"))

        self._build_create_tab()
        self._build_reference_tab()
        self._build_find_tab()

    def _disable_enter_default_buttons(self) -> None:
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def retranslate_ui(self) -> None:
        self.setWindowTitle(tr("template.title", product=self.product_name))
        self.tabs.setTabText(0, tr("template.tab.create"))
        self.tabs.setTabText(1, tr("template.tab.reference"))
        self.tabs.setTabText(2, tr("template.tab.find"))
        for attr, text in (
            ("file_box", tr("template.reference_image")),
            ("select_box", tr("template.template_edit")),
            ("point_box", tr("template.feature_point_edit")),
            ("param_box", tr("template.params")),
            ("recipe_box", tr("template.recipe")),
            ("info_box", tr("template.reference_attrs")),
            ("region_box", tr("template.reference_regions")),
            ("model_box", tr("template.model")),
            ("list_box", tr("template.test_images")),
            ("find_recipe_box", tr("template.find_params")),
            ("search_box", tr("template.search_roi")),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setTitle(text)
        for attr, text in (
            ("btn_open_image", tr("template.open_image")),
            ("btn_load_model", tr("template.load_existing_model")),
            ("btn_apply_selection", tr("template.apply_current_box")),
            ("btn_clear_roi", tr("template.clear_template_roi")),
            ("btn_clear_masks", tr("template.clear_mask")),
            ("btn_undo_mask", tr("template.undo_last_mask")),
            ("chk_edit_points", tr("template.edit_points")),
            ("btn_extract_points", tr("template.extract_points")),
            ("btn_reset_points", tr("template.reset_points")),
            ("btn_build", tr("template.save")),
            ("btn_build_pinned", tr("template.save")),
            ("btn_apply_region_name", tr("template.apply_name")),
            ("btn_add_reference_roi", tr("template.new_roi")),
            ("btn_remove_reference_roi", tr("template.delete_selected_roi")),
            ("btn_clear_reference_rois", tr("template.clear_all_roi")),
            ("btn_load_reference_roi", tr("template.load_reference_roi")),
            ("btn_save_reference_roi", tr("template.save_roi_config")),
            ("btn_find_open_model", tr("template.open_model")),
            ("btn_find_use_product", tr("template.use_product_model")),
            ("btn_add_find_images", tr("template.add_images")),
            ("btn_remove_find_image", tr("template.remove")),
            ("btn_clear_find_images", tr("template.clear")),
            ("chk_auto_threshold_sweep", tr("template.auto_threshold_sweep")),
            ("btn_apply_search_roi", tr("template.apply_search_roi")),
            ("btn_clear_search_roi", tr("template.clear_search_roi")),
            ("btn_run_find_selected", tr("template.run_selected")),
            ("btn_run_find_all", tr("template.run_all")),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(text)
        if hasattr(self, "lbl_current_usage"):
            self.lbl_current_usage.setText(tr("template.current_usage"))
        if hasattr(self, "lbl_mask_count"):
            self._refresh_mask_count()
        if hasattr(self, "lbl_default_direction"):
            self.lbl_default_direction.setText(tr("template.default_direction_label"))
        if hasattr(self, "lbl_point_help"):
            self.lbl_point_help.setText(tr("template.point_help"))
        if hasattr(self, "chk_edit_points"):
            self._sync_point_edit_detail_visibility()
        if hasattr(self, "help_text"):
            self.help_text.setText(tr("template.create_help"))
        if hasattr(self, "btn_build_pinned"):
            self.btn_build_pinned.setToolTip(tr("template.save_template"))
        if hasattr(self, "edit_output_label"):
            self.edit_output_label.setPlaceholderText(tr("template.select_roi_placeholder"))
        if hasattr(self, "edit_display_name"):
            self.edit_display_name.setPlaceholderText(tr("template.select_roi_placeholder"))
        if hasattr(self, "chk_auto_threshold_sweep"):
            self.chk_auto_threshold_sweep.setToolTip(tr("template.auto_threshold_tip"))
        if hasattr(self, "table_reference_regions"):
            self.table_reference_regions.setHorizontalHeaderLabels([
                tr("template.reference_table.index"),
                tr("template.reference_table.name"),
                tr("template.reference_table.roi_label"),
                tr("template.reference_table.info"),
            ])
        if hasattr(self, "lbl_find_model"):
            self._update_find_model_label()
        if hasattr(self, "lbl_search_roi"):
            self._refresh_search_roi_status()
        if (
            hasattr(self, "lbl_status")
            and not self.image_path
            and self.detector is None
            and self.template_roi is None
        ):
            self.lbl_status.setText(tr("template.create_status"))
        if hasattr(self, "lbl_reference_status") and not self._reference_regions:
            if self._reference_dirty:
                self.lbl_reference_status.setText(self._reference_dirty_status())
            else:
                self.lbl_reference_status.setText(tr("template.reference_status"))
        if hasattr(self, "lbl_find_status") and self.list_find_images.count() <= 0:
            self.lbl_find_status.setText(tr("template.find_status"))

    def _make_left_scroll_column(self) -> Tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setStyleSheet(
            "QScrollArea{background:#2f2f2f;border:none;}"
            "QScrollArea > QWidget > QWidget{background:#2f2f2f;}"
            "QScrollBar:vertical{background:#2f2f2f;width:10px;margin:0;}"
            "QScrollBar::handle:vertical{background:#5a5a5a;min-height:28px;border-radius:5px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        host = QtWidgets.QWidget()
        host.setObjectName("leftScrollHost")
        host.setStyleSheet("#leftScrollHost{background:#2f2f2f;}")
        host.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 18, 0)
        layout.setSpacing(8)
        scroll.setWidget(host)
        return scroll, layout

    def _make_horizontal_splitter(self) -> QtWidgets.QSplitter:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(
            "QSplitter::handle{background:#343434;}"
            "QSplitter::handle:hover{background:#4f4f4f;}"
        )
        return splitter

    def _build_create_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout(self.tab_create)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        splitter = self._make_horizontal_splitter()
        layout.addWidget(splitter, 1)

        left_container = QtWidgets.QWidget()
        left_container.setMinimumWidth(440)
        left_container_layout = QtWidgets.QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(8)
        left_scroll, left = self._make_left_scroll_column()
        left_container_layout.addWidget(left_scroll, 1)
        splitter.addWidget(left_container)

        self.file_box = QtWidgets.QGroupBox(tr("template.reference_image"))
        file_box = self.file_box
        file_l = QtWidgets.QGridLayout(file_box)
        self.lbl_image = QtWidgets.QLabel(tr("template.not_loaded"))
        self.lbl_image.setWordWrap(True)
        self.btn_open_image = QtWidgets.QPushButton(tr("template.open_image"))
        self.btn_open_image.clicked.connect(self._pick_image)
        self.btn_load_model = QtWidgets.QPushButton(tr("template.load_existing_model"))
        self.btn_load_model.clicked.connect(lambda: self._load_existing_model(silent=False))
        file_l.addWidget(self.lbl_image, 0, 0, 1, 2)
        file_l.addWidget(self.btn_open_image, 1, 0)
        file_l.addWidget(self.btn_load_model, 1, 1)
        file_box.setTitle(tr("template.reference_image"))
        self.lbl_image.setText(tr("template.not_loaded"))
        self.btn_open_image.setText(tr("template.open_image"))
        self.btn_load_model.setText(tr("template.load_existing_model"))
        left.addWidget(file_box)

        self.select_box = QtWidgets.QGroupBox(tr("template.template_edit"))
        select_box = self.select_box
        select_l = QtWidgets.QGridLayout(select_box)
        self.cmb_role = QtWidgets.QComboBox()
        self.cmb_role.addItems(["template_roi", "exclude_mask"])
        self.cmb_role.setMinimumWidth(200)
        self.cmb_role.currentTextChanged.connect(self._on_role_changed)
        self.btn_apply_selection = QtWidgets.QPushButton(tr("template.apply_current_box"))
        self.btn_apply_selection.clicked.connect(self._apply_current_selection)
        self.btn_apply_selection.setVisible(False)
        self.btn_clear_roi = QtWidgets.QPushButton(tr("template.clear_template_roi"))
        self.btn_clear_roi.clicked.connect(self._clear_template_roi)
        self.btn_clear_masks = QtWidgets.QPushButton(tr("template.clear_mask"))
        self.btn_clear_masks.clicked.connect(self._clear_masks)
        self.btn_undo_mask = QtWidgets.QPushButton(tr("template.undo_last_mask"))
        self.btn_undo_mask.clicked.connect(self._undo_last_mask)
        self.lbl_mask_count = QtWidgets.QLabel()
        select_l.setColumnStretch(0, 0)
        select_l.setColumnStretch(1, 1)
        self.lbl_current_usage = QtWidgets.QLabel(tr("template.current_usage"))
        select_l.addWidget(self.lbl_current_usage, 0, 0)
        select_l.addWidget(self.cmb_role, 0, 1)
        select_l.addWidget(self.btn_clear_roi, 1, 0)
        select_l.addWidget(self.btn_clear_masks, 1, 1)
        select_l.addWidget(self.btn_undo_mask, 2, 0)
        select_l.addWidget(self.lbl_mask_count, 2, 1)
        select_box.setTitle(tr("template.template_edit"))
        self.lbl_current_usage.setText(tr("template.current_usage"))
        self.btn_apply_selection.setText(tr("template.apply_current_box"))
        self.btn_clear_roi.setText(tr("template.clear_template_roi"))
        self.btn_clear_masks.setText(tr("template.clear_mask"))
        self.btn_undo_mask.setText(tr("template.undo_last_mask"))
        self._refresh_mask_count()
        left.addWidget(select_box)

        self.point_box = QtWidgets.QGroupBox(tr("template.feature_point_edit"))
        point_box = self.point_box
        point_l = QtWidgets.QGridLayout(point_box)
        self.chk_edit_points = QtWidgets.QCheckBox(tr("template.edit_points"))
        self.chk_edit_points.toggled.connect(self._on_point_edit_toggled)
        self.chk_edit_points.setText(tr("template.edit_points"))
        self.spin_point_label = QtWidgets.QSpinBox()
        self.spin_point_label.setRange(0, 7)
        self.spin_point_label.setMinimumWidth(160)
        self.btn_extract_points = QtWidgets.QPushButton(tr("template.extract_points"))
        self.btn_extract_points.clicked.connect(self._extract_points_from_roi)
        self.btn_extract_points.setEnabled(False)
        self.btn_reset_points = QtWidgets.QPushButton(tr("template.reset_points"))
        self.btn_reset_points.clicked.connect(self._reset_editor_levels_from_detector)
        self.lbl_point_help = QtWidgets.QLabel(tr("template.point_help"))
        self.lbl_point_help.setWordWrap(True)
        point_l.setColumnStretch(0, 0)
        point_l.setColumnStretch(1, 1)
        self.lbl_point_help.setText(tr("template.point_help"))
        point_l.addWidget(self.chk_edit_points, 0, 0, 1, 2)
        self.lbl_default_direction = QtWidgets.QLabel(tr("template.default_direction_label"))
        point_l.addWidget(self.lbl_default_direction, 1, 0)
        point_l.addWidget(self.spin_point_label, 1, 1)
        point_l.addWidget(self.btn_reset_points, 2, 0, 1, 2)
        point_l.addWidget(self.lbl_point_help, 3, 0, 1, 2)
        point_l.addWidget(self.btn_extract_points, 4, 0, 1, 2)
        self.lbl_point_help.setText(tr("template.point_help"))
        point_box.setTitle(tr("template.feature_point_edit"))
        self.chk_edit_points.setText(tr("template.edit_points"))
        self.lbl_default_direction.setText(tr("template.default_direction_label"))
        self.btn_reset_points.setText(tr("template.reset_points"))
        self.lbl_point_help.setText(tr("template.point_help"))
        self._sync_point_edit_detail_visibility()
        left.addWidget(point_box)

        self.param_box = QtWidgets.QGroupBox(tr("template.params"))
        param_box = self.param_box
        form = QtWidgets.QFormLayout(param_box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.edit_class_id = QtWidgets.QLineEdit(self.product_name or "object")
        self.edit_class_id.setMinimumWidth(220)
        self.edit_levels = QtWidgets.QLineEdit("4,8")
        self.edit_levels.setMinimumWidth(220)
        self.spin_num_features = QtWidgets.QSpinBox()
        self.spin_num_features.setRange(16, 4096)
        self.spin_num_features.setValue(128)
        self.spin_weak = QtWidgets.QDoubleSpinBox()
        self.spin_weak.setRange(0.0, 255.0)
        self.spin_weak.setValue(30.0)
        self.spin_strong = QtWidgets.QDoubleSpinBox()
        self.spin_strong.setRange(0.0, 255.0)
        self.spin_strong.setValue(60.0)
        self.spin_angle_start = QtWidgets.QDoubleSpinBox()
        self.spin_angle_start.setRange(-360.0, 360.0)
        self.spin_angle_start.setValue(-5.0)
        self.spin_angle_end = QtWidgets.QDoubleSpinBox()
        self.spin_angle_end.setRange(-360.0, 360.0)
        self.spin_angle_end.setValue(10.0)
        self.spin_angle_step = QtWidgets.QDoubleSpinBox()
        self.spin_angle_step.setRange(0.1, 360.0)
        self.spin_angle_step.setValue(1.0)
        self.spin_scale_start = QtWidgets.QDoubleSpinBox()
        self.spin_scale_start.setRange(0.05, 10.0)
        self.spin_scale_start.setValue(1.0)
        self.spin_scale_end = QtWidgets.QDoubleSpinBox()
        self.spin_scale_end.setRange(0.05, 10.0)
        self.spin_scale_end.setValue(1.0)
        self.spin_scale_step = QtWidgets.QDoubleSpinBox()
        self.spin_scale_step.setRange(0.001, 5.0)
        self.spin_scale_step.setDecimals(3)
        self.spin_scale_step.setValue(0.05)
        form.addRow("class_id", self.edit_class_id)
        form.addRow("levels", self.edit_levels)
        form.addRow("num_features", self.spin_num_features)
        form.addRow("weak_threshold", self.spin_weak)
        form.addRow("strong_threshold", self.spin_strong)
        form.addRow("angle_start", self.spin_angle_start)
        form.addRow("angle_end", self.spin_angle_end)
        form.addRow("angle_step", self.spin_angle_step)
        form.addRow("scale_start", self.spin_scale_start)
        form.addRow("scale_end", self.spin_scale_end)
        form.addRow("scale_step", self.spin_scale_step)
        param_box.setTitle(tr("template.params"))
        left.addWidget(param_box)

        self.recipe_box = QtWidgets.QGroupBox(tr("template.recipe"))
        recipe_box = self.recipe_box
        recipe_form = QtWidgets.QFormLayout(recipe_box)
        self.cmb_backend_create = QtWidgets.QComboBox()
        self.cmb_backend_create.addItems([label for label, _ in BACKEND_ITEMS])
        self.cmb_backend_create.setCurrentText(_DEFAULT_FIND_BACKEND_LABEL)
        self.spin_threshold_create = QtWidgets.QDoubleSpinBox()
        self.spin_threshold_create.setRange(0.0, 100.0)
        self.spin_threshold_create.setValue(50.0)
        self.spin_nms_create = QtWidgets.QDoubleSpinBox()
        self.spin_nms_create.setRange(0.0, 1.0)
        self.spin_nms_create.setDecimals(2)
        self.spin_nms_create.setValue(0.3)
        self.cmb_follow_create = QtWidgets.QComboBox()
        self.cmb_follow_create.addItems(["affine_roi", "match_bbox"])
        recipe_form.addRow("backend", self.cmb_backend_create)
        recipe_form.addRow("threshold", self.spin_threshold_create)
        recipe_form.addRow("nms_iou", self.spin_nms_create)
        recipe_form.addRow("follow_mode", self.cmb_follow_create)
        recipe_box.setTitle(tr("template.recipe"))
        left.addWidget(recipe_box)
        recipe_box.setVisible(False)

        for widget, source in [
            (self.cmb_backend_create, "create"),
            (self.spin_threshold_create, "create"),
            (self.spin_nms_create, "create"),
            (self.cmb_follow_create, "create"),
        ]:
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(lambda _text, s=source: self._sync_recipe_controls(s))
            else:
                widget.valueChanged.connect(lambda _value, s=source: self._sync_recipe_controls(s))

        action_row = QtWidgets.QHBoxLayout()
        self.btn_build = QtWidgets.QPushButton(tr("template.save"))
        self.btn_build.clicked.connect(self._build_and_save)
        action_row.addWidget(self.btn_build)
        self.btn_build.setText(tr("template.save"))
        self.btn_build.hide()
        left.addLayout(action_row)

        self.lbl_status = QtWidgets.QLabel(tr("template.create_status"))
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setText(tr("template.create_status"))
        left.addWidget(self.lbl_status)
        left.addStretch(1)
        footer_box = QtWidgets.QFrame()
        footer_box.setStyleSheet("QFrame{background:#2f2f2f;border-top:1px solid #505050;}")
        footer_layout = QtWidgets.QVBoxLayout(footer_box)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(0)
        self.btn_build_pinned = QtWidgets.QPushButton(tr("template.save"))
        self.btn_build_pinned.clicked.connect(self._build_and_save)
        self.btn_build_pinned.setText(tr("template.save"))
        self.btn_build_pinned.setToolTip(tr("template.save_template"))
        self.btn_build_pinned.setText(tr("template.save"))
        self.btn_build_pinned.setToolTip(tr("template.save_template"))
        footer_layout.addWidget(self.btn_build_pinned)
        left_container_layout.addWidget(footer_box, 0)

        right_container = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1000])
        self.create_canvas = RoiCanvas()
        self.canvas = self.create_canvas
        self.create_canvas.setMinimumSize(640, 480)
        self.create_canvas.draw_shape = "rect"
        self.create_canvas.shapesChanged.connect(self._on_create_canvas_shape_changed)
        self.create_canvas.imagePressed.connect(self._on_create_canvas_pressed)
        self.create_canvas.imageMoved.connect(self._on_create_canvas_moved)
        self.create_canvas.imageReleased.connect(self._on_create_canvas_released)
        right.addWidget(self.create_canvas, 1)

        self.help_text = QtWidgets.QLabel(tr("template.create_help"))
        help_text = self.help_text
        help_text.setWordWrap(True)
        help_text.setText(tr("template.create_help"))
        help_text.setText(tr("template.create_help"))
        right.addWidget(help_text)

        self._on_role_changed(self.cmb_role.currentText())
        self._on_point_edit_toggled(False)
        self.lbl_status.setText(tr("template.create_status"))

    def _build_reference_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout(self.tab_reference)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        left_scroll, left = self._make_left_scroll_column()
        layout.addWidget(left_scroll, 0)

        self.info_box = QtWidgets.QGroupBox(tr("template.reference_attrs"))
        info_box = self.info_box
        info_form = QtWidgets.QFormLayout(info_box)
        self.edit_reference_label = QtWidgets.QLineEdit()
        self.edit_output_label = QtWidgets.QLineEdit()
        self.edit_output_label.setPlaceholderText(tr("template.select_roi_placeholder"))
        self.edit_display_name = QtWidgets.QLineEdit()
        self.edit_display_name.setPlaceholderText(tr("template.select_roi_placeholder"))
        self.cmb_reference_shape = QtWidgets.QComboBox()
        self.cmb_reference_shape.addItems(["rectangle", "polygon"])
        self.cmb_reference_shape.currentTextChanged.connect(self._on_reference_shape_changed)
        self.edit_output_label.textChanged.connect(self._on_region_field_edited)
        self.edit_display_name.textChanged.connect(self._on_region_field_edited)
        self.edit_output_label.returnPressed.connect(self._apply_reference_region_fields)
        self.edit_display_name.returnPressed.connect(self._apply_reference_region_fields)
        self.btn_apply_region_name = QtWidgets.QPushButton(tr("template.apply_name"))
        self.btn_apply_region_name.clicked.connect(self._apply_reference_region_fields)
        info_form.addRow(tr("template.roi_label"), self.edit_output_label)
        info_form.addRow(tr("template.display_name"), self.edit_display_name)
        name_btn_row = QtWidgets.QHBoxLayout()
        name_btn_row.addWidget(self.btn_apply_region_name)
        name_btn_row.addWidget(self.cmb_reference_shape)
        info_form.addRow(tr("template.shape_action"), name_btn_row)
        self.edit_output_label.setEnabled(False)
        self.edit_display_name.setEnabled(False)
        self.btn_apply_region_name.setEnabled(False)
        left.addWidget(info_box)

        self.region_box = QtWidgets.QGroupBox(tr("template.reference_regions"))
        region_box = self.region_box
        region_l = QtWidgets.QVBoxLayout(region_box)
        region_body = QtWidgets.QHBoxLayout()
        self.table_reference_regions = QtWidgets.QTableWidget(0, 4)
        self.table_reference_regions.setHorizontalHeaderLabels([
            tr("template.reference_table.index"),
            tr("template.reference_table.name"),
            tr("template.reference_table.roi_label"),
            tr("template.reference_table.info"),
        ])
        self.table_reference_regions.verticalHeader().setVisible(False)
        self.table_reference_regions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_reference_regions.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_reference_regions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_reference_regions.setAlternatingRowColors(True)
        self.table_reference_regions.currentCellChanged.connect(self._on_reference_region_selected)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        region_body.addWidget(self.table_reference_regions, 1)

        region_btn_col = QtWidgets.QVBoxLayout()
        self.btn_add_reference_roi = QtWidgets.QPushButton(tr("template.new_roi"))
        self.btn_add_reference_roi.clicked.connect(self._prepare_new_reference_roi)
        self.btn_remove_reference_roi = QtWidgets.QPushButton(tr("template.delete_selected_roi"))
        self.btn_remove_reference_roi.clicked.connect(self._remove_selected_reference_roi)
        self.btn_clear_reference_rois = QtWidgets.QPushButton(tr("template.clear_all_roi"))
        self.btn_clear_reference_rois.clicked.connect(self._clear_reference_roi)
        self.btn_load_reference_roi = QtWidgets.QPushButton(tr("template.load_reference_roi"))
        self.btn_load_reference_roi.clicked.connect(lambda: self._load_reference_roi_from_json(silent=False))
        self.btn_save_reference_roi = QtWidgets.QPushButton(tr("template.save_roi_config"))
        self.btn_save_reference_roi.clicked.connect(self._save_reference_roi_config)
        for button in [
            self.btn_add_reference_roi,
            self.btn_load_reference_roi,
            self.btn_remove_reference_roi,
            self.btn_clear_reference_rois,
            self.btn_save_reference_roi,
        ]:
            button.setMinimumWidth(120)
            region_btn_col.addWidget(button)
        region_btn_col.addStretch(1)
        region_body.addLayout(region_btn_col)
        region_l.addLayout(region_body, 1)
        left.addWidget(region_box)

        self.lbl_reference_status = QtWidgets.QLabel(tr("template.reference_status"))
        self.lbl_reference_status.setWordWrap(True)
        left.addWidget(self.lbl_reference_status)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        self.ref_canvas = RoiCanvas()
        self.ref_canvas.setMinimumSize(640, 480)
        self.ref_canvas.set_roi_style(
            roi_color=QtGui.QColor(0, 0, 255),
            roi_width=3.5,
            preview_color=QtGui.QColor(0, 0, 255),
            preview_dash=False,
            preview_width=2.0,
        )
        self.ref_canvas.shapesChanged.connect(self._on_reference_canvas_shape_changed)
        right.addWidget(self.ref_canvas, 1)

    def _build_find_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout(self.tab_find)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        left_container = QtWidgets.QWidget()
        left_container.setFixedWidth(430)
        left_container_layout = QtWidgets.QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(8)
        left_scroll, left = self._make_left_scroll_column()
        left_container_layout.addWidget(left_scroll, 1)
        layout.addWidget(left_container, 0)

        self.model_box = QtWidgets.QGroupBox(tr("template.model"))
        model_box = self.model_box
        model_l = QtWidgets.QGridLayout(model_box)
        self.lbl_find_model = QtWidgets.QLabel(tr("template.not_set"))
        self.lbl_find_model.setWordWrap(True)
        self.btn_find_open_model = QtWidgets.QPushButton(tr("template.open_model"))
        self.btn_find_open_model.clicked.connect(self._pick_find_model)
        self.btn_find_use_product = QtWidgets.QPushButton(tr("template.use_product_model"))
        self.btn_find_use_product.clicked.connect(self._use_product_model)
        model_l.addWidget(self.lbl_find_model, 0, 0, 1, 2)
        model_l.addWidget(self.btn_find_open_model, 1, 0)
        model_l.addWidget(self.btn_find_use_product, 1, 1)
        left.addWidget(model_box)

        self.list_box = QtWidgets.QGroupBox(tr("template.test_images"))
        list_box = self.list_box
        list_l = QtWidgets.QVBoxLayout(list_box)
        self.list_find_images = QtWidgets.QListWidget()
        self.list_find_images.itemDoubleClicked.connect(self._run_find_for_item)
        self.list_find_images.currentItemChanged.connect(self._on_find_item_selected)
        list_l.addWidget(self.list_find_images, 1)
        list_btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_find_images = QtWidgets.QPushButton(tr("template.add_images"))
        self.btn_add_find_images.clicked.connect(self._add_find_images)
        self.btn_remove_find_image = QtWidgets.QPushButton(tr("template.remove"))
        self.btn_remove_find_image.clicked.connect(self._remove_selected_find_images)
        self.btn_clear_find_images = QtWidgets.QPushButton(tr("template.clear"))
        self.btn_clear_find_images.clicked.connect(self._clear_find_images)
        for button in [self.btn_add_find_images, self.btn_remove_find_image, self.btn_clear_find_images]:
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        list_btn_row.addWidget(self.btn_add_find_images)
        list_btn_row.addWidget(self.btn_remove_find_image)
        list_btn_row.addWidget(self.btn_clear_find_images)
        list_l.addLayout(list_btn_row)
        left.addWidget(list_box, 1)

        self.find_recipe_box = QtWidgets.QGroupBox(tr("template.find_params"))
        recipe_box = self.find_recipe_box
        recipe_form = QtWidgets.QFormLayout(recipe_box)
        self.cmb_backend_find = QtWidgets.QComboBox()
        self.cmb_backend_find.addItems([label for label, _ in BACKEND_ITEMS])
        self.cmb_backend_find.setCurrentText(_DEFAULT_FIND_BACKEND_LABEL)
        self.spin_threshold_find = QtWidgets.QDoubleSpinBox()
        self.spin_threshold_find.setRange(0.0, 100.0)
        self.spin_threshold_find.setValue(50.0)
        self.spin_nms_find = QtWidgets.QDoubleSpinBox()
        self.spin_nms_find.setRange(0.0, 1.0)
        self.spin_nms_find.setDecimals(2)
        self.spin_nms_find.setValue(0.3)
        self.cmb_follow_find = QtWidgets.QComboBox()
        self.cmb_follow_find.addItems(["affine_roi", "match_bbox"])
        self.chk_auto_threshold_sweep = QtWidgets.QCheckBox(tr("template.auto_threshold_sweep"))
        self.chk_auto_threshold_sweep.setToolTip(tr("template.auto_threshold_tip"))
        self.chk_auto_threshold_sweep.toggled.connect(lambda _checked: self._save_recipe())
        recipe_form.addRow("backend", self.cmb_backend_find)
        recipe_form.addRow("threshold", self.spin_threshold_find)
        recipe_form.addRow("nms_iou", self.spin_nms_find)
        recipe_form.addRow("follow_mode", self.cmb_follow_find)
        recipe_form.addRow("", self.chk_auto_threshold_sweep)
        left.addWidget(recipe_box)

        self.search_box = QtWidgets.QGroupBox(tr("template.search_roi"))
        search_box = self.search_box
        search_l = QtWidgets.QVBoxLayout(search_box)
        self.btn_apply_search_roi = QtWidgets.QPushButton(tr("template.apply_search_roi"))
        self.btn_apply_search_roi.clicked.connect(self._apply_find_search_roi)
        self.btn_clear_search_roi = QtWidgets.QPushButton(tr("template.clear_search_roi"))
        self.btn_clear_search_roi.clicked.connect(self._clear_find_search_roi)
        self.lbl_search_roi = QtWidgets.QLabel(tr("template.search_roi_unset"))
        self.lbl_search_roi.setWordWrap(True)
        search_l.addWidget(self.btn_apply_search_roi)
        search_l.addWidget(self.btn_clear_search_roi)
        search_l.addWidget(self.lbl_search_roi)
        left.addWidget(search_box)

        for widget, source in [
            (self.cmb_backend_find, "find"),
            (self.spin_threshold_find, "find"),
            (self.spin_nms_find, "find"),
            (self.cmb_follow_find, "find"),
        ]:
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(lambda _text, s=source: self._sync_recipe_controls(s))
            else:
                widget.valueChanged.connect(lambda _value, s=source: self._sync_recipe_controls(s))

        run_row = QtWidgets.QHBoxLayout()
        self.btn_run_find_selected = QtWidgets.QPushButton(tr("template.run_selected"))
        self.btn_run_find_selected.clicked.connect(self._run_selected_find)
        self.btn_run_find_all = QtWidgets.QPushButton(tr("template.run_all"))
        self.btn_run_find_all.clicked.connect(self._run_all_find)
        for button in [self.btn_run_find_selected, self.btn_run_find_all]:
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        run_row.addWidget(self.btn_run_find_selected)
        run_row.addWidget(self.btn_run_find_all)
        left.addLayout(run_row)

        self.lbl_find_status = QtWidgets.QLabel(tr("template.find_status"))
        self.lbl_find_status.setWordWrap(True)
        left.addWidget(self.lbl_find_status)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        self.find_canvas = RoiCanvas()
        self.find_canvas.setMinimumSize(640, 480)
        right.addWidget(self.find_canvas, 1)

    def _pick_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            tr("auto.pick_reference_title"),
            self.image_path or "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self._set_image(path, reset_state=True)

    def _set_image(self, path: str, *, reset_state: bool) -> None:
        image_bgr = imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            QtWidgets.QMessageBox.critical(self, tr("common.read_failed"), path)
            return
        self.image_path = path
        self.image_bgr = image_bgr
        self.lbl_image.setText(_compact_path_text(path))
        self.lbl_image.setToolTip(path)
        self.create_canvas.set_image(path, pixmap=pixmap_from_path(path))
        self.ref_canvas.set_image(path, pixmap=pixmap_from_path(path))
        if reset_state:
            self.template_roi = None
            self.mask_rects = []
            self.editor_levels = []
            self.original_mode = "auto"
            self.points_dirty = False
            self._hover_feature_index = None
            self._point_drag_start = None
            self._point_drag_end = None
            self.detector = None
            self.detector_path = ""
            self._refresh_mask_count()
        self._refresh_create_overlays()
        self.lbl_status.setText(tr("template.status_extracted", count=self._feature_count))
        loaded_json_regions = False
        if not self._apply_reference_roi_from_recipe():
            loaded_json_regions = self._load_reference_roi_from_json(silent=True)
        if loaded_json_regions:
            self._set_reference_dirty(True, tr("template.status_reference_loaded_unsaved", count=len(self._reference_regions)))
        else:
            self._set_reference_dirty(False)
            self._save_recipe()

        self.lbl_status.setText(tr("template.status_loaded_reference"))

    def _sync_recipe_controls(self, source: str) -> None:
        if self._syncing_recipe_controls:
            return
        self._syncing_recipe_controls = True
        try:
            if source == "create":
                self.cmb_backend_find.setCurrentText(self.cmb_backend_create.currentText())
                self.spin_threshold_find.setValue(float(self.spin_threshold_create.value()))
                self.spin_nms_find.setValue(float(self.spin_nms_create.value()))
                self.cmb_follow_find.setCurrentText(self.cmb_follow_create.currentText())
            else:
                self.cmb_backend_create.setCurrentText(self.cmb_backend_find.currentText())
                self.spin_threshold_create.setValue(float(self.spin_threshold_find.value()))
                self.spin_nms_create.setValue(float(self.spin_nms_find.value()))
                self.cmb_follow_create.setCurrentText(self.cmb_follow_find.currentText())
        finally:
            self._syncing_recipe_controls = False
        self._save_recipe()

    def _apply_find_backend_default(self) -> None:
        if not hasattr(self, "cmb_backend_find"):
            return
        idx = self.cmb_backend_find.findText(_DEFAULT_FIND_BACKEND_LABEL)
        for combo in (self.cmb_backend_create, self.cmb_backend_find):
            with QtCore.QSignalBlocker(combo):
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentText(_DEFAULT_FIND_BACKEND_LABEL)

    def _recipe_from_controls(self, *, use_find_values: bool = False) -> ShapeRecipe:
        if use_find_values:
            backend = BACKEND_LABEL_TO_KEY.get(self.cmb_backend_find.currentText(), "original")
            threshold = float(self.spin_threshold_find.value())
            nms_iou = float(self.spin_nms_find.value())
            follow_mode = self.cmb_follow_find.currentText().strip() or "affine_roi"
        else:
            backend = BACKEND_LABEL_TO_KEY.get(self.cmb_backend_create.currentText(), "original")
            threshold = float(self.spin_threshold_create.value())
            nms_iou = float(self.spin_nms_create.value())
            follow_mode = self.cmb_follow_create.currentText().strip() or "affine_roi"

        reference_regions = [
            {
                "reference_label": str(region.get("reference_label") or region.get("label") or ""),
                "output_label": str(region.get("output_label") or region.get("reference_label") or region.get("label") or ""),
                "display_name": str(
                    region.get("display_name")
                    or region.get("name")
                    or region.get("output_label")
                    or region.get("reference_label")
                    or region.get("label")
                    or ""
                ),
                "shape_type": str(region.get("shape_type", "rectangle")),
                "points": [
                    [float(pt[0]), float(pt[1])]
                    for pt in region.get("points", []) or []
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ],
            }
            for region in self._reference_regions
            if isinstance(region, dict)
        ]
        first_region = reference_regions[0] if reference_regions else None
        saved_reference_regions = reference_regions if (reference_regions or self._reference_regions_explicit) else None
        reference_shape_type = str((first_region or {}).get("shape_type", self._recipe_reference_shape_type))
        reference_points = list((first_region or {}).get("points", self._recipe_reference_points))
        reference_label = str((first_region or {}).get("reference_label", "roi") or "roi")
        output_label = str((first_region or {}).get("output_label", reference_label) or reference_label)
        return ShapeRecipe(
            model_path=self.paths.model_path,
            reference_image=self.image_path,
            class_id=self.edit_class_id.text().strip() or self.product_name or "object",
            backend=backend,
            threshold=threshold,
            auto_threshold_sweep=bool(self.chk_auto_threshold_sweep.isChecked()),
            threshold_sweep_step=10,
            threshold_sweep_min=20,
            nms_iou=nms_iou,
            topk=1,
            crop_stride=0,
            use_scene_mask=False,
            follow_mode=follow_mode,
            output_label=output_label,
            reference_label=reference_label,
            reference_shape_type=reference_shape_type,
            reference_points=reference_points or None,
            reference_regions=saved_reference_regions,
            search_shape_type=str(self._search_roi_shape_type or ""),
            search_points=[list(pt) for pt in (self._search_roi_points or [])] or None,
        )

    def _save_recipe(self) -> None:
        recipe = self._recipe_from_controls(use_find_values=False)
        recipe.model_path = self.paths.model_path
        os.makedirs(self.paths.role_dir, exist_ok=True)
        save_recipe(recipe, self.paths.recipe_path)

    def _load_recipe(self) -> None:
        recipe_path = self._resolved_recipe_path
        if not os.path.exists(recipe_path):
            self._refresh_reference_region_fields()
            self._apply_find_backend_default()
            self._set_reference_dirty(False)
            return
        recipe = load_recipe(recipe_path)
        if recipe.reference_image and os.path.exists(recipe.reference_image) and not self.image_path:
            self.image_path = recipe.reference_image
        self.edit_class_id.setText(recipe.class_id or self.product_name or "object")
        self._recipe_reference_shape_type = str(recipe.reference_shape_type or "")
        self._recipe_reference_points = [list(pt) for pt in (recipe.reference_points or [])]
        self._search_roi_shape_type = str(recipe.search_shape_type or "")
        self._search_roi_points = [list(pt) for pt in (recipe.search_points or [])]
        self._reference_regions = [
            {
                "reference_label": str(region.get("reference_label") or region.get("label") or ""),
                "output_label": str(region.get("output_label") or region.get("reference_label") or region.get("label") or ""),
                "display_name": str(
                    region.get("display_name")
                    or region.get("name")
                    or region.get("output_label")
                    or region.get("reference_label")
                    or region.get("label")
                    or ""
                ),
                "shape_type": str(region.get("shape_type", "rectangle")),
                "points": [
                    [float(pt[0]), float(pt[1])]
                    for pt in region.get("points", []) or []
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ],
            }
            for region in (recipe.reference_regions or [])
            if isinstance(region, dict)
        ]
        self._reference_regions_explicit = ShapeRecipeStore.has_explicit_reference_regions(recipe)
        self._selected_reference_idx = None
        self._refresh_search_roi_status()
        self.cmb_backend_create.setCurrentText(BACKEND_KEY_TO_LABEL.get(recipe.backend, "Original"))
        self.spin_threshold_create.setValue(float(recipe.threshold))
        self.spin_nms_create.setValue(float(recipe.nms_iou))
        self.chk_auto_threshold_sweep.setChecked(bool(recipe.auto_threshold_sweep))
        idx = self.cmb_follow_create.findText(recipe.follow_mode)
        if idx >= 0:
            self.cmb_follow_create.setCurrentIndex(idx)
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self._set_reference_dirty(False)
        self._sync_recipe_controls("create")

    def _apply_reference_roi_from_recipe(self) -> bool:
        if not self._reference_regions:
            if self._reference_regions_explicit:
                self._selected_reference_idx = None
                self._refresh_reference_region_list()
                self._refresh_reference_canvas()
                self._refresh_reference_region_fields()
                self._set_reference_dirty(False, tr("template.status_loaded_recipe_roi"))
                return True
            points = self._recipe_reference_points or []
            if not points:
                return False
            shape_type = str(self._recipe_reference_shape_type or "rectangle")
            self._reference_regions = [
                {
                    "reference_label": "roi1",
                    "output_label": "roi1",
                    "display_name": "roi1",
                    "shape_type": shape_type,
                    "points": [list(pt) for pt in points],
                }
            ]
        self._selected_reference_idx = None
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self._set_reference_dirty(False, tr("template.status_loaded_recipe_roi"))
        return bool(self._reference_regions)

    def _load_existing_model(self, *, silent: bool) -> None:
        if not os.path.exists(self.paths.model_path):
            if not silent:
                QtWidgets.QMessageBox.information(self, tr("common.info"), tr("template.no_model"))
            return
        try:
            detector = self._get_detector_for_model(self.paths.model_path, reuse_shared=False)
            class_ids = detector.class_ids()
            if not class_ids:
                raise RuntimeError(tr("template.model_no_class"))
            class_id = class_ids[0]
            source_info, roi_img, _roi_mask, roi_rect, mask_rects = load_class_source_assets(detector, class_id)
            editor_levels = detector.get_original_editor_levels(class_id)
            if not editor_levels:
                editor_levels = normalize_extracted_levels_to_roi(
                    detector.get_templates(class_id, 0, backend="original"),
                    roi_img,
                )
        except Exception as exc:
            if not silent:
                QtWidgets.QMessageBox.critical(self, tr("common.load_failed"), str(exc))
            return

        source_block = source_info.get("source", {}) if isinstance(source_info, dict) else {}
        source_image_path = str(source_block.get("image_path", "")).strip() if isinstance(source_block, dict) else ""
        if not source_image_path and os.path.exists(self.paths.recipe_path):
            try:
                recipe = load_recipe(self.paths.recipe_path)
                source_image_path = str(recipe.reference_image or "").strip()
            except Exception:
                source_image_path = ""
        if source_image_path and os.path.exists(source_image_path) and os.path.abspath(source_image_path) != os.path.abspath(self.image_path or ""):
            self._set_image(source_image_path, reset_state=False)

        self.detector = detector
        self.detector_path = self.paths.model_path
        self._apply_detector_template_params(detector, source_info)
        self.template_roi = roi_rect
        self.btn_extract_points.setEnabled(self.template_roi is not None)
        self.mask_rects = [MaskRect(x=int(r.x), y=int(r.y), w=int(r.w), h=int(r.h)) for r in mask_rects]
        self.editor_levels = clone_levels(editor_levels)
        self.original_mode = str(source_info.get("original_mode", "auto")) if isinstance(source_info, dict) else "auto"
        self.points_dirty = False
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self.btn_extract_points.setEnabled(False)
        self._clear_create_canvas_roi()
        self._refresh_mask_count()
        self.edit_class_id.setText(class_id)
        self.find_model_path = self.paths.model_path
        self._update_find_model_label()
        self._refresh_create_overlays()
        self.lbl_status.setText(
            tr("template.status_loaded_model", model=os.path.basename(self.paths.model_path), count=self._feature_count)
        )

    def _apply_detector_template_params(self, detector: ShapeLikeDetector, source_info: object) -> None:
        pose_ui = {}
        if isinstance(source_info, dict):
            pose_info_block = source_info.get("pose_infos", {})
            if isinstance(pose_info_block, dict):
                pose_ui = pose_info_block.get("ui", {})
                if not isinstance(pose_ui, dict):
                    pose_ui = {}

        levels_text = ",".join(str(int(level)) for level in list(getattr(detector, "T_at_level", []) or []))
        self.edit_levels.setText(levels_text or "4,8")
        self.spin_num_features.setValue(max(16, int(getattr(detector, "num_features", 128) or 128)))
        self.spin_weak.setValue(float(getattr(detector, "weak_threshold", 30.0) or 30.0))
        self.spin_strong.setValue(float(getattr(detector, "strong_threshold", 60.0) or 60.0))
        self.spin_angle_start.setValue(float(pose_ui.get("angle_start", self.spin_angle_start.value())))
        self.spin_angle_end.setValue(float(pose_ui.get("angle_end", self.spin_angle_end.value())))
        self.spin_angle_step.setValue(float(pose_ui.get("angle_step", self.spin_angle_step.value())))
        self.spin_scale_start.setValue(float(pose_ui.get("scale_start", self.spin_scale_start.value())))
        self.spin_scale_end.setValue(float(pose_ui.get("scale_end", self.spin_scale_end.value())))
        self.spin_scale_step.setValue(float(pose_ui.get("scale_step", self.spin_scale_step.value())))

    def _extract_points_from_roi(self, _checked: object = None, *, show_errors: bool = True) -> bool:
        if self.image_bgr is None or not self.image_path:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.load_reference_first"))
            return False
        if self.template_roi is None:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.set_template_roi_first"))
            return False

        x, y, w, h = self.template_roi.x, self.template_roi.y, self.template_roi.w, self.template_roi.h
        roi_img = self.image_bgr[y : y + h, x : x + w].copy()
        if roi_img.size == 0:
            if show_errors:
                QtWidgets.QMessageBox.critical(self, tr("common.invalid_template"), tr("template.roi_out_of_range"))
            return False

        try:
            levels = parse_levels(self.edit_levels.text().strip())
            class_id = self.edit_class_id.text().strip() or self.product_name or "object"
            num_features = int(self.spin_num_features.value())
            weak_threshold = float(self.spin_weak.value())
            strong_threshold = float(self.spin_strong.value())
            roi_mask = build_mask_from_rects(roi_img.shape[1], roi_img.shape[0], self.mask_rects)
            detector = ShapeLikeDetector(
                num_features=num_features,
                T_levels=levels,
                weak_threshold=weak_threshold,
                strong_threshold=strong_threshold,
            )
            template_id = detector.add_template(
                roi_img,
                class_id=class_id,
                object_mask=roi_mask,
                num_features=num_features,
                metadata={"angle": 0.0, "scale": 1.0},
                backend="original",
            )
            if template_id < 0:
                raise RuntimeError("No template extracted. Try lower strong/weak threshold or select a clearer ROI.")
            extracted_levels = clone_levels(detector.get_templates(class_id, template_id, backend="original"))
            if not extracted_levels:
                raise RuntimeError("No template extracted.")
        except Exception as exc:
            if show_errors:
                QtWidgets.QMessageBox.critical(self, tr("common.extract_failed"), str(exc))
            else:
                self.lbl_status.setText(tr("template.status_auto_extract_failed", error=exc))
            return False

        self.editor_levels = [extracted_levels[0]]
        self._sync_editor_levels()
        self.original_mode = "manual_points"
        self.points_dirty = False
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self.chk_edit_points.setChecked(False)
        self._refresh_create_overlays()
        self.lbl_status.setText(tr("template.status_extracted", count=self._feature_count))
        return True

    def _auto_extract_points_from_roi(self) -> None:
        if self._build_thread is not None:
            return
        if self.chk_edit_points.isChecked():
            return
        if self.points_dirty:
            self.lbl_status.setText(tr("template.status_auto_extract_skipped_dirty"))
            return
        self._extract_points_from_roi(show_errors=False)

    def _build_and_save(self) -> None:
        if self._build_thread is not None:
            return
        if self.image_bgr is None or not self.image_path:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.load_reference_first"))
            return
        if self.template_roi is None:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.set_template_roi_first"))
            return

        x, y, w, h = self.template_roi.x, self.template_roi.y, self.template_roi.w, self.template_roi.h
        roi_img = self.image_bgr[y : y + h, x : x + w].copy()
        if roi_img.size == 0:
            QtWidgets.QMessageBox.critical(self, tr("common.invalid_template"), tr("template.roi_out_of_range"))
            return

        original_mode = "manual_points" if (self.points_dirty or self.original_mode == "manual_points") and self.editor_levels else "auto"
        original_editor_levels = clone_levels(self.editor_levels) if original_mode == "manual_points" else None

        try:
            levels = parse_levels(self.edit_levels.text().strip())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, tr("common.create_failed"), str(exc))
            return

        payload: Dict[str, object] = {
            "class_id": self.edit_class_id.text().strip() or self.product_name or "object",
            "roi_img": roi_img,
            "roi_rect": self.template_roi,
            "mask_rects": list(self.mask_rects),
            "pose": self._pose_ui_values(),
            "pose_ui": self._pose_ui_values(),
            "levels": levels,
            "num_features": int(self.spin_num_features.value()),
            "weak_threshold": float(self.spin_weak.value()),
            "strong_threshold": float(self.spin_strong.value()),
            "original_mode": original_mode,
            "original_editor_levels": original_editor_levels,
            "source_image_path": self.image_path,
            "role_dir": self.paths.role_dir,
            "model_path": self.paths.model_path,
        }
        self._start_build_worker(payload)

    def _start_build_worker(self, payload: Dict[str, object]) -> None:
        progress = QtWidgets.QProgressDialog(
            tr("template.saving_model"),
            "",
            0,
            100,
            self,
        )
        progress.setWindowTitle(tr("template.saving_model_title"))
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._build_progress = progress

        thread = QtCore.QThread(self)
        worker = _TemplateBuildWorker(payload)
        worker.moveToThread(thread)
        self._build_thread = thread
        self._build_worker = worker

        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_build_progress)
        worker.finished.connect(self._on_build_finished)
        worker.failed.connect(self._on_build_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_build_worker_refs)

        self._set_build_controls_enabled(False)
        self.lbl_status.setText(tr("template.status_saving_model"))
        progress.show()
        thread.start()

    def _on_build_progress(self, value: int, message: str) -> None:
        if self._build_progress is not None:
            self._build_progress.setLabelText(str(message or tr("template.saving_model")))
            self._build_progress.setValue(max(0, min(100, int(value))))

    def _on_build_finished(self, payload: object) -> None:
        if self._build_progress is not None:
            self._build_progress.setValue(100)
            self._build_progress.close()
        self._set_build_controls_enabled(True)
        result = cast(Dict[str, object], payload)
        detector = cast(ShapeLikeDetector, result.get("detector"))
        kept = int(result.get("kept", 0) or 0)
        skipped = int(result.get("skipped", 0) or 0)
        self.detector = detector
        self.detector_path = self.paths.model_path
        self.find_model_path = self.paths.model_path
        self._update_find_model_label()
        self._save_recipe()
        class_id = self.edit_class_id.text().strip() or self.product_name or "object"
        saved_levels = detector.get_original_editor_levels(class_id)
        if saved_levels:
            self.editor_levels = clone_levels(saved_levels)
        self.points_dirty = False
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self.btn_extract_points.setEnabled(False)
        self._clear_create_canvas_roi()
        self._refresh_create_overlays()
        self.lbl_status.setText(
            tr("template.status_saved_model", kept=kept, skipped=skipped, count=self._feature_count)
        )
        self.modelSaved.emit(self.paths.model_path, self.paths.recipe_path)
        QtWidgets.QMessageBox.information(self, tr("common.done"), tr("template.saved_model", path=self.paths.model_path))

    def _on_build_failed(self, message: str) -> None:
        if self._build_progress is not None:
            self._build_progress.close()
        self._set_build_controls_enabled(True)
        QtWidgets.QMessageBox.critical(self, tr("common.create_failed"), str(message))

    def _clear_build_worker_refs(self) -> None:
        self._build_thread = None
        self._build_worker = None
        self._build_progress = None
        self._refresh_mask_count()

    def _set_build_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.btn_build,
            self.btn_build_pinned,
            self.btn_open_image,
            self.btn_load_model,
            self.btn_extract_points,
            self.btn_reset_points,
            self.cmb_role,
            self.btn_apply_selection,
            self.btn_clear_roi,
            self.btn_clear_masks,
            self.btn_undo_mask,
        ):
            widget.setEnabled(bool(enabled))
        if enabled:
            self.btn_extract_points.setEnabled(self.template_roi is not None)
            self.btn_undo_mask.setEnabled(bool(self.mask_rects))
            if self.chk_edit_points.isChecked():
                self.cmb_role.setEnabled(False)
                self.btn_apply_selection.setEnabled(False)
                self.btn_clear_roi.setEnabled(False)
                self.btn_clear_masks.setEnabled(False)
                self.btn_undo_mask.setEnabled(False)

    def _pose_ui_values(self) -> Dict[str, float]:
        return {
            "angle_start": float(self.spin_angle_start.value()),
            "angle_end": float(self.spin_angle_end.value()),
            "angle_step": float(self.spin_angle_step.value()),
            "scale_start": float(self.spin_scale_start.value()),
            "scale_end": float(self.spin_scale_end.value()),
            "scale_step": float(self.spin_scale_step.value()),
        }

    def _refresh_mask_count(self) -> None:
        if hasattr(self, "lbl_mask_count"):
            self.lbl_mask_count.setText(tr("template.mask_count", count=len(self.mask_rects)))
        if hasattr(self, "btn_undo_mask"):
            edit_enabled = bool(getattr(self, "chk_edit_points", None) and self.chk_edit_points.isChecked())
            self.btn_undo_mask.setEnabled(bool(self.mask_rects) and self._build_thread is None and not edit_enabled)

    def _clear_create_canvas_roi(self) -> None:
        self._suppress_create_auto_apply = True
        try:
            self.create_canvas.clear_roi()
        finally:
            self._suppress_create_auto_apply = False

    def _apply_current_selection(self, _checked: object = None, *, show_warnings: bool = True, auto: bool = False) -> bool:
        xywh = self.create_canvas.roi_xywh()
        if xywh is None:
            if show_warnings:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.drag_rect_first"))
            return False
        role = self.cmb_role.currentText()
        should_auto_extract = False
        if role == "template_roi":
            self.template_roi = RoiRect(*xywh)
            self.mask_rects = []
            self.btn_extract_points.setEnabled(True)
            self.editor_levels = []
            self.points_dirty = False
            self.original_mode = "auto"
            self._hover_feature_index = None
            self._point_drag_start = None
            self._point_drag_end = None
            self.lbl_status.setText(tr("template.status_template_roi_updated"))
            should_auto_extract = True
        else:
            if self.template_roi is None:
                if show_warnings:
                    QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.set_template_roi_first"))
                return False
            rel = _clamp_rect_to_roi(xywh, self.template_roi)
            if rel is None:
                if show_warnings:
                    QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.mask_intersect_required"))
                return False
            self.mask_rects.append(rel)
            self.btn_extract_points.setEnabled(self.template_roi is not None)
            self.lbl_status.setText(tr("template.status_mask_added", count=len(self.mask_rects)))
            should_auto_extract = True
        self._refresh_mask_count()
        self._clear_create_canvas_roi()
        self._refresh_create_overlays()
        if should_auto_extract:
            self._auto_extract_points_from_roi()
        return True

    def _clear_template_roi(self) -> None:
        self.template_roi = None
        self.mask_rects = []
        self.editor_levels = []
        self.points_dirty = False
        self.original_mode = "auto"
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self._clear_create_canvas_roi()
        self._refresh_mask_count()
        self._refresh_create_overlays()
        self.lbl_status.setText(tr("template.status_template_roi_cleared"))

    def _clear_masks(self) -> None:
        self.mask_rects = []
        self.btn_extract_points.setEnabled(self.template_roi is not None)
        self._clear_create_canvas_roi()
        self._refresh_mask_count()
        self._refresh_create_overlays()
        self.lbl_status.setText(tr("template.status_mask_cleared"))
        self._auto_extract_points_from_roi()

    def _undo_last_mask(self) -> None:
        if not self.mask_rects:
            return
        self.mask_rects.pop()
        self.btn_extract_points.setEnabled(self.template_roi is not None)
        self._clear_create_canvas_roi()
        self._refresh_mask_count()
        self._refresh_create_overlays()
        self.lbl_status.setText(tr("template.status_mask_undone", count=len(self.mask_rects)))
        self._auto_extract_points_from_roi()

    def _on_role_changed(self, role: str) -> None:
        if role == "exclude_mask":
            self.create_canvas.set_roi_style(
                roi_color=QtGui.QColor(255, 64, 64),
                roi_dash=True,
                roi_width=2,
                preview_color=QtGui.QColor(255, 64, 64),
                preview_dash=True,
                preview_width=2,
            )
        else:
            self.create_canvas.set_roi_style(
                roi_color=QtGui.QColor(0, 255, 0),
                roi_dash=False,
                roi_width=2,
                preview_color=QtGui.QColor(0, 255, 0),
                preview_dash=False,
                preview_width=2,
            )

    def _sync_point_edit_detail_visibility(self) -> None:
        visible = bool(self.chk_edit_points.isChecked())
        for widget in (
            self.lbl_default_direction,
            self.spin_point_label,
            self.btn_reset_points,
            self.lbl_point_help,
        ):
            widget.setVisible(visible)

    def _on_point_edit_toggled(self, enabled: bool) -> None:
        self.create_canvas.set_interaction_enabled(not enabled)
        self.cmb_role.setEnabled(not enabled)
        self.btn_apply_selection.setEnabled(not enabled)
        self.btn_clear_roi.setEnabled(not enabled)
        self.btn_clear_masks.setEnabled(not enabled)
        self.btn_undo_mask.setEnabled((not enabled) and bool(self.mask_rects))
        self.btn_extract_points.setEnabled(self.template_roi is not None)
        if enabled:
            self.lbl_status.setText(tr("template.status_point_edit_enabled"))
        else:
            self._point_drag_start = None
            self._point_drag_end = None
        self._sync_point_edit_detail_visibility()
        self._refresh_create_overlays()

    def _on_create_canvas_shape_changed(self) -> None:
        if self._suppress_create_auto_apply:
            return
        if self.chk_edit_points.isChecked():
            return
        if self._build_thread is not None:
            return
        self._apply_current_selection(show_warnings=False, auto=True)

    def _on_create_canvas_pressed(self, button: int, x: int, y: int) -> None:
        if not self.chk_edit_points.isChecked():
            return
        if self.template_roi is None or not self.editor_levels:
            self.lbl_status.setText(tr("template.status_edit_model_first"))
            return
        if not self._point_in_template_roi(x, y):
            return
        if button == _button_left():
            self._point_drag_start = (int(x), int(y))
            self._point_drag_end = (int(x), int(y))
            self._refresh_create_overlays()
        elif button == _button_right():
            idx = self._hover_feature_index
            if idx is not None and self._delete_feature_index(int(idx)):
                self.lbl_status.setText(tr("template.status_deleted_selected_point", count=self._feature_count))
                return
            if self._delete_feature_near(x, y):
                self.lbl_status.setText(tr("template.status_deleted_near_point", count=self._feature_count))
            else:
                self.lbl_status.setText(tr("template.status_no_point_to_delete"))

    def _on_create_canvas_moved(self, _buttons: int, x: int, y: int) -> None:
        if self.chk_edit_points.isChecked():
            hover = self._find_nearest_feature_index(x, y)
            if hover != self._hover_feature_index:
                self._hover_feature_index = hover
                self._refresh_create_overlays()
            if self._point_drag_start is not None:
                self._point_drag_end = (int(x), int(y))
                self._refresh_create_overlays()

    def _on_create_canvas_released(self, button: int, x: int, y: int) -> None:
        if not self.chk_edit_points.isChecked():
            return
        if button != _button_left() or self._point_drag_start is None:
            return
        self._point_drag_end = (int(x), int(y))
        sx, sy = self._point_drag_start
        ex, ey = self._point_drag_end
        dx = float(ex - sx)
        dy = float(ey - sy)
        if math.hypot(dx, dy) >= 2.0:
            theta = float(np.degrees(np.arctan2(dy, dx)))
            label = angle_deg_to_label(theta)
            self.spin_point_label.setValue(int(label))
            self._add_feature_at(sx, sy, int(label), theta_deg=theta)
        else:
            label = int(self.spin_point_label.value()) % 8
            self._add_feature_at(sx, sy, label, theta_deg=label_to_angle_deg(label))
        self._point_drag_start = None
        self._point_drag_end = None
        self.lbl_status.setText(tr("template.status_feature_points_updated", count=self._feature_count))

    def _point_in_template_roi(self, x: int, y: int) -> bool:
        if self.template_roi is None:
            return False
        return (
            self.template_roi.x <= int(x) <= self.template_roi.x + self.template_roi.w
            and self.template_roi.y <= int(y) <= self.template_roi.y + self.template_roi.h
        )

    def _get_roi_image(self) -> Optional[np.ndarray]:
        if self.image_bgr is None or self.template_roi is None:
            return None
        x, y, w, h = self.template_roi.x, self.template_roi.y, self.template_roi.w, self.template_roi.h
        roi_img = self.image_bgr[y : y + h, x : x + w].copy()
        return roi_img if roi_img.size > 0 else None

    def _sync_editor_levels(self) -> None:
        if not self.editor_levels:
            return
        roi_img = self._get_roi_image()
        if roi_img is None:
            return
        shapes = roi_level_shapes_from_image(roi_img, len(self.editor_levels))
        self.editor_levels = sync_levels_from_level0(self.editor_levels[0], shapes)

    def _editor_level0(self) -> Optional[TemplateLevel]:
        if not self.editor_levels:
            return None
        return self.editor_levels[0]

    def _editor_feature_abs_points(self) -> List[Tuple[float, float]]:
        level0 = self._editor_level0()
        if level0 is None or self.template_roi is None:
            return []
        base_x = float(self.template_roi.x + int(level0.tl_x))
        base_y = float(self.template_roi.y + int(level0.tl_y))
        return [(base_x + float(f.x), base_y + float(f.y)) for f in level0.features]

    def _find_nearest_feature_index(self, x: int, y: int, max_dist: float = 8.0) -> Optional[int]:
        pts = self._editor_feature_abs_points()
        if not pts:
            return None
        best_idx = -1
        best_d2 = float("inf")
        for idx, (px, py) in enumerate(pts):
            dx = float(px - x)
            dy = float(py - y)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_idx = idx
                best_d2 = d2
        if best_idx < 0 or best_d2 > float(max_dist * max_dist):
            return None
        return best_idx

    def _add_feature_at(self, x: int, y: int, label: int, *, theta_deg: float) -> None:
        level0 = self._editor_level0()
        if level0 is None or self.template_roi is None:
            return
        xr = int(round(float(x) - float(self.template_roi.x) - float(level0.tl_x)))
        yr = int(round(float(y) - float(self.template_roi.y) - float(level0.tl_y)))
        xr = max(0, min(xr, int(level0.width)))
        yr = max(0, min(yr, int(level0.height)))
        level0.features.append(Feature(x=int(xr), y=int(yr), label=int(label) & 7, theta=float(theta_deg)))
        self.points_dirty = True
        self.original_mode = "manual_points"
        self._sync_editor_levels()
        self._refresh_create_overlays()

    def _delete_feature_index(self, idx: int) -> bool:
        level0 = self._editor_level0()
        if level0 is None:
            return False
        if idx < 0 or idx >= len(level0.features):
            return False
        del level0.features[int(idx)]
        self.points_dirty = True
        self.original_mode = "manual_points"
        self._hover_feature_index = None
        self._sync_editor_levels()
        self._refresh_create_overlays()
        return True

    def _delete_feature_near(self, x: int, y: int) -> bool:
        level0 = self._editor_level0()
        if level0 is None:
            return False
        idx = self._find_nearest_feature_index(x, y)
        if idx is None:
            return False
        return self._delete_feature_index(int(idx))

    def _reset_editor_levels_from_detector(self) -> None:
        if self.detector is None:
            self.lbl_status.setText(tr("template.status_no_loaded_model"))
            return
        class_id = self.edit_class_id.text().strip() or (self.detector.class_ids()[0] if self.detector.class_ids() else "")
        if not class_id:
            self.lbl_status.setText(tr("template.status_model_no_restorable_class"))
            return
        levels = self.detector.get_original_editor_levels(class_id)
        if not levels:
            try:
                _source_info, roi_img, _roi_mask, _roi_rect, _mask_rects = load_class_source_assets(self.detector, class_id)
                levels = normalize_extracted_levels_to_roi(
                    self.detector.get_templates(class_id, 0, backend="original"),
                    roi_img,
                )
            except Exception as exc:
                self.lbl_status.setText(tr("template.status_restore_features_failed", error=exc))
                return
        self.editor_levels = clone_levels(levels)
        self.points_dirty = False
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self._refresh_create_overlays()
        self.lbl_status.setText(tr("template.status_restored_features", count=self._feature_count))

    def _feature_overlays(self) -> List[OverlayShape]:
        self._feature_count = 0
        level0 = self._editor_level0()
        if level0 is None or self.template_roi is None:
            return []

        palette = _orientation_palette()
        point_groups: List[List[Tuple[float, float]]] = [[] for _ in palette]
        segment_groups: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = [[] for _ in palette]

        base_x = float(self.template_roi.x + int(level0.tl_x))
        base_y = float(self.template_roi.y + int(level0.tl_y))
        for feature in level0.features:
            px = base_x + float(feature.x)
            py = base_y + float(feature.y)
            label = int(feature.label) % len(palette)
            theta = float(feature.theta)
            if not np.isfinite(theta):
                theta = label_to_angle_deg(int(feature.label))
            p2 = _arrow_endpoint(px, py, theta, 8.0)
            point_groups[label].append((px, py))
            segment_groups[label].append(((px, py), p2))
            self._feature_count += 1

        overlays: List[OverlayShape] = []
        for idx, color in enumerate(palette):
            if segment_groups[idx]:
                overlays.append(OverlayShape(shape_type="segments", segments=segment_groups[idx], color=color, width=1, dash=False))
            if point_groups[idx]:
                overlays.append(OverlayShape(shape_type="points", points=point_groups[idx], color=color, width=3, dash=False))

        if self._hover_feature_index is not None:
            pts = self._editor_feature_abs_points()
            if 0 <= self._hover_feature_index < len(pts):
                overlays.append(
                    OverlayShape(
                        shape_type="points",
                        points=[pts[self._hover_feature_index]],
                        color=QtGui.QColor(255, 255, 255),
                        width=8,
                        dash=False,
                    )
                )

        if self._point_drag_start is not None and self._point_drag_end is not None:
            sx, sy = self._point_drag_start
            ex, ey = self._point_drag_end
            dx = float(ex - sx)
            dy = float(ey - sy)
            label = int(self.spin_point_label.value()) % 8
            if abs(dx) + abs(dy) > 0:
                label = angle_deg_to_label(float(np.degrees(np.arctan2(dy, dx))))
            overlays.append(
                OverlayShape(
                    shape_type="segments",
                    segments=[((float(sx), float(sy)), (float(ex), float(ey)))],
                    color=palette[int(label) % len(palette)],
                    width=2,
                    dash=False,
                )
            )

        return overlays

    def _refresh_create_overlays(self) -> None:
        overlays: List[OverlayShape] = []
        if self.template_roi is not None:
            overlays.append(
                OverlayShape(
                    shape_type="rect",
                    xywh=(self.template_roi.x, self.template_roi.y, self.template_roi.w, self.template_roi.h),
                    color=QtGui.QColor(0, 255, 0),
                    width=2,
                    dash=False,
                )
            )
            for rect in self.mask_rects:
                overlays.append(
                    OverlayShape(
                        shape_type="rect",
                        xywh=(int(self.template_roi.x + rect.x), int(self.template_roi.y + rect.y), int(rect.w), int(rect.h)),
                        color=QtGui.QColor(255, 64, 64),
                        width=2,
                        dash=True,
                    )
                )
        overlays.extend(self._feature_overlays())
        self.create_canvas.set_overlays(overlays)

    def _search_roi_xywh(self) -> Optional[Tuple[int, int, int, int]]:
        points = [
            [float(pt[0]), float(pt[1])]
            for pt in (self._search_roi_points or [])
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if len(points) < 2:
            return None
        (x0, y0), (x1, y1) = points[:2]
        x = int(round(min(x0, x1)))
        y = int(round(min(y0, y1)))
        w = max(1, int(round(abs(x1 - x0))))
        h = max(1, int(round(abs(y1 - y0))))
        return x, y, w, h

    def _refresh_search_roi_status(self) -> None:
        xywh = self._search_roi_xywh()
        if xywh is None:
            self.lbl_search_roi.setText(tr("template.search_roi_unset"))
            return
        x, y, w, h = xywh
        self.lbl_search_roi.setText(tr("template.status_search_roi_set", x=x, y=y, w=w, h=h))

    def _apply_search_roi_to_find_canvas(self) -> None:
        xywh = self._search_roi_xywh()
        if xywh is None:
            self.find_canvas.clear_roi()
            self.find_canvas.set_roi_style(roi_color=QtGui.QColor(0, 0, 255), roi_dash=False, roi_width=0.5)
            self._refresh_search_roi_status()
            return
        self.find_canvas.set_roi_rect(xywh)
        self.find_canvas.set_roi_style(roi_color=QtGui.QColor(0, 0, 255), roi_dash=False, roi_width=0.5)
        self._refresh_search_roi_status()

    def _apply_find_search_roi(self) -> None:
        xywh = self.find_canvas.roi_xywh()
        if xywh is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("template.draw_search_roi_first"))
            return
        x, y, w, h = xywh
        self._search_roi_shape_type = "rectangle"
        self._search_roi_points = [[float(x), float(y)], [float(x + w), float(y + h)]]
        self._apply_search_roi_to_find_canvas()
        self._save_recipe()

    def _clear_find_search_roi(self) -> None:
        self._search_roi_shape_type = ""
        self._search_roi_points = []
        self.find_canvas.clear_roi()
        self.find_canvas.set_roi_style(roi_color=QtGui.QColor(0, 0, 255), roi_dash=False, roi_width=0.5)
        self._save_recipe()
        self._refresh_search_roi_status()

    def _pick_find_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            tr("template.pick_model_title"),
            self.find_model_path or self.paths.model_path,
            "template model (*.json)",
        )
        if path:
            self.find_model_path = path
            self._update_find_model_label()

    def _use_product_model(self) -> None:
        self.find_model_path = self.paths.model_path
        self._update_find_model_label()

    def _update_find_model_label(self) -> None:
        text = self.find_model_path or tr("template.not_set")
        self.lbl_find_model.setText(text)
        self.lbl_find_model.setToolTip(text)

    def _add_find_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            tr("template.pick_test_images_title"),
            self.image_path or "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        existing = {self.list_find_images.item(i).data(QtCore.Qt.UserRole) for i in range(self.list_find_images.count())}
        for path in paths:
            if not path or path in existing:
                continue
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setData(QtCore.Qt.UserRole, path)
            item.setToolTip(path)
            self.list_find_images.addItem(item)

    def _remove_selected_find_images(self) -> None:
        for item in self.list_find_images.selectedItems():
            scene_path = str(item.data(QtCore.Qt.UserRole) or "")
            if scene_path:
                self._find_result_cache.pop(scene_path, None)
            row = self.list_find_images.row(item)
            self.list_find_images.takeItem(row)
        if self.list_find_images.count() <= 0:
            self.find_canvas.clear_image()
            self.lbl_find_status.setText(tr("template.status_find_list_empty"))

    def _clear_find_images(self) -> None:
        self._find_result_cache.clear()
        self.list_find_images.clear()
        self.find_canvas.clear_image()
        self.lbl_find_status.setText(tr("template.status_find_list_cleared"))

    def _run_selected_find(self) -> None:
        item = self.list_find_images.currentItem()
        if item is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("template.select_test_image_first"))
            return
        self._run_find_for_item(item)

    def _run_all_find(self) -> None:
        for idx in range(self.list_find_images.count()):
            item = self.list_find_images.item(idx)
            self._run_find_for_item(item)
            QtWidgets.QApplication.processEvents()

    def _on_find_item_selected(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        if current is None:
            return
        scene_path = str(current.data(QtCore.Qt.UserRole) or "")
        if not scene_path:
            return
        cached = self._find_result_cache.get(scene_path)
        if cached:
            pixmap = cached.get("pixmap")
            message = str(cached.get("message") or "")
            if isinstance(pixmap, QtGui.QPixmap) and not pixmap.isNull():
                self.find_canvas.set_image(scene_path, pixmap=QtGui.QPixmap(pixmap))
            elif os.path.exists(scene_path):
                self.find_canvas.set_image(scene_path, pixmap=pixmap_from_path(scene_path))
            self._apply_search_roi_to_find_canvas()
            if message:
                self.lbl_find_status.setText(message)
            return
        if os.path.exists(scene_path):
            self.find_canvas.set_image(scene_path, pixmap=pixmap_from_path(scene_path))
            self._apply_search_roi_to_find_canvas()
        self.lbl_find_status.setText(tr("template.status_find_switched", image=os.path.basename(scene_path)))

    def _get_detector_for_model(self, model_path: str, *, reuse_shared: bool = True) -> ShapeLikeDetector:
        if reuse_shared and self.detector is not None and os.path.abspath(model_path) == os.path.abspath(self.detector_path or ""):
            return self.detector
        if self.find_detector is not None and os.path.abspath(model_path) == os.path.abspath(self.find_detector_path or ""):
            return self.find_detector
        detector = load_detector_model(model_path)
        if reuse_shared and os.path.abspath(model_path) == os.path.abspath(self.paths.model_path):
            self.detector = detector
            self.detector_path = model_path
        else:
            self.find_detector = detector
            self.find_detector_path = model_path
        return detector

    def _run_find_for_item(self, item: QtWidgets.QListWidgetItem) -> None:
        scene_path = str(item.data(QtCore.Qt.UserRole) or "")
        if not scene_path:
            return
        model_path = self.find_model_path or self.paths.model_path
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.load_find_model_first"))
            return
        recipe = self._recipe_from_controls(use_find_values=True)
        recipe.model_path = model_path
        if not recipe.reference_image and self.image_path:
            recipe.reference_image = self.image_path
        if not recipe.reference_image or not os.path.exists(recipe.reference_image):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("template.set_reference_first"))
            return

        scene_bgr = imread(scene_path, cv2.IMREAD_COLOR)
        if scene_bgr is None:
            self._set_find_item_error(item, tr("template.read_image_failed", path=scene_path))
            return

        used_threshold = float(recipe.threshold)
        elapsed_ms = -1.0
        try:
            detector = self._get_detector_for_model(model_path, reuse_shared=True)
            thresholds = [float(recipe.threshold)]
            if bool(recipe.auto_threshold_sweep):
                step = max(1.0, float(recipe.threshold_sweep_step))
                minimum = float(recipe.threshold_sweep_min)
                current = float(recipe.threshold) - step
                while current >= minimum:
                    if all(abs(current - existing) > 1e-6 for existing in thresholds):
                        thresholds.append(float(current))
                    current -= step

            result = None
            last_exc: Optional[Exception] = None
            started = time.perf_counter()
            for threshold in thresholds:
                recipe.threshold = float(threshold)
                try:
                    result = locate_and_follow(scene_bgr, recipe.reference_image, recipe, detector=detector, scene_mask=None)
                    used_threshold = float(threshold)
                    break
                except RuntimeError as exc:
                    exc_text = str(exc)
                    if (
                        "match failure" in exc_text
                        or "shape did not find any match" in exc_text
                    ):
                        last_exc = exc
                        continue
                    raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if result is None:
                raise last_exc or RuntimeError("match failure")
            overlay = _draw_match_overlay(detector, scene_bgr, result.match)
            overlay = _overlay_follow_result(overlay, result, recipe.output_label, elapsed_ms=elapsed_ms)
        except Exception as exc:
            self._set_find_item_error(item, str(exc))
            return

        overlay_pixmap = _cv_to_qpixmap(overlay)
        self.find_canvas.set_image(scene_path, pixmap=overlay_pixmap)
        self._apply_search_roi_to_find_canvas()
        msg = (
            f"sim={result.match.similarity:.2f} "
            f"bbox={result.bbox[0]},{result.bbox[1]},{result.bbox[2]},{result.bbox[3]} "
            f"time={elapsed_ms:.1f}ms"
        )
        if abs(float(used_threshold) - float(self.spin_threshold_find.value())) > 1e-6:
            msg += f" th={used_threshold:.0f}"
        item.setText(f"{os.path.basename(scene_path)} | {msg}")
        item.setToolTip(f"{scene_path}\n{msg}")
        item.setForeground(QtGui.QBrush(QtGui.QColor(20, 160, 20)))
        status_msg = tr("template.status_find_done", image=os.path.basename(scene_path), message=msg)
        self._find_result_cache[scene_path] = {
            "pixmap": QtGui.QPixmap(overlay_pixmap),
            "message": status_msg,
        }
        self.lbl_find_status.setText(status_msg)

    def _set_find_item_error(self, item: QtWidgets.QListWidgetItem, message: str) -> None:
        scene_path = str(item.data(QtCore.Qt.UserRole) or "")
        item.setText(f"{os.path.basename(scene_path)} | ERROR: {message}")
        item.setToolTip(f"{scene_path}\nERROR: {message}")
        item.setForeground(QtGui.QBrush(QtGui.QColor(200, 40, 40)))
        status_msg = tr("template.status_find_failed", message=message)
        pixmap: Optional[QtGui.QPixmap] = None
        if scene_path and os.path.exists(scene_path):
            pixmap = pixmap_from_path(scene_path)
            if self.list_find_images.currentItem() is item:
                self.find_canvas.set_image(scene_path, pixmap=pixmap)
                self._apply_search_roi_to_find_canvas()
        self._find_result_cache[scene_path] = {
            "pixmap": QtGui.QPixmap(pixmap) if isinstance(pixmap, QtGui.QPixmap) else None,
            "message": status_msg,
        }
        self.lbl_find_status.setText(status_msg)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._build_thread is not None:
            event.ignore()
            self.lbl_status.setText(tr("template.status_saving_model"))
            return
        if not self._confirm_reference_roi_discard():
            event.ignore()
            return
        self._clear_find_images()
        super().closeEvent(event)


__all__ = ["ShapeTemplateDialog"]


