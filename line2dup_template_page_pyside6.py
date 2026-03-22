from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import qr_core_proxy as qr_core
from line2dup_bootstrap import ensure_repo_root_on_path
from line2dup_locator import product_paths
from line2dup_recipe import Line2DupRecipe, load_recipe, save_recipe
from line2dup_roi_follow import FollowResult, locate_and_follow
from line2dup_template_core import (
    BACKEND_ITEMS,
    BACKEND_KEY_TO_LABEL,
    BACKEND_LABEL_TO_KEY,
    MaskRect,
    RoiRect,
    angle_deg_to_label,
    build_multi_backend_detector,
    clone_levels,
    label_to_angle_deg,
    load_class_source_assets,
    normalize_extracted_levels_to_roi,
    parse_levels,
    pose_infos_from_ui_values,
    roi_level_shapes_from_image,
    sync_levels_from_level0,
)
from ui.debug import OverlayShape, RoiCanvas, pixmap_from_path

ensure_repo_root_on_path()

from line2dup_like_matcher import (  # noqa: E402
    Feature,
    Line2DupLikeDetector,
    TemplateLevel,
    load_detector_model,
    match_quad,
    save_detector_model,
)


def _cv_to_qpixmap(image_bgr: np.ndarray) -> QtGui.QPixmap:
    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _orientation_palette() -> List[QtGui.QColor]:
    return [
        QtGui.QColor(255, 0, 0),
        QtGui.QColor(255, 128, 0),
        QtGui.QColor(255, 255, 0),
        QtGui.QColor(0, 255, 0),
        QtGui.QColor(0, 255, 255),
        QtGui.QColor(0, 128, 255),
        QtGui.QColor(0, 0, 255),
        QtGui.QColor(255, 0, 255),
    ]


def _arrow_endpoint(x: float, y: float, theta_deg: float, length: float) -> Tuple[float, float]:
    rad = math.radians(float(theta_deg))
    return float(x + length * math.cos(rad)), float(y + length * math.sin(rad))


def _apply_affine_point(transform: Optional[np.ndarray], x: float, y: float) -> Tuple[float, float]:
    if transform is None:
        return float(x), float(y)
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape == (3, 3):
        matrix = matrix[:2, :]
    px = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
    py = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
    return px, py


def _button_left() -> int:
    return int(QtCore.Qt.MouseButton.LeftButton.value)


def _button_right() -> int:
    return int(QtCore.Qt.MouseButton.RightButton.value)


def _shape_to_rect(shape: dict) -> Optional[Tuple[int, int, int, int]]:
    pts = shape.get("points", [])
    if not pts:
        return None
    if shape.get("shape_type") == "rectangle" and len(pts) >= 2:
        (x0, y0), (x1, y1) = pts[:2]
        x = int(round(min(float(x0), float(x1))))
        y = int(round(min(float(y0), float(y1))))
        w = int(round(abs(float(x1) - float(x0))))
        h = int(round(abs(float(y1) - float(y0))))
        return x, y, max(1, w), max(1, h)
    arr = np.asarray(pts, dtype=np.float32)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    return int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))


def _clamp_rect_to_roi(rect: Tuple[int, int, int, int], roi: RoiRect) -> Optional[MaskRect]:
    x, y, w, h = rect
    x1 = max(int(roi.x), int(x))
    y1 = max(int(roi.y), int(y))
    x2 = min(int(roi.x + roi.w), int(x + w))
    y2 = min(int(roi.y + roi.h), int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return MaskRect(x=x1 - int(roi.x), y=y1 - int(roi.y), w=x2 - x1, h=y2 - y1)


def _overlay_follow_result(
    image_bgr: np.ndarray,
    result: FollowResult,
    label_name: str,
    *,
    elapsed_ms: Optional[float] = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    text = f"{label_name} sim={result.match.similarity:.2f} class={result.match.class_id} tid={result.match.template_id}"
    if elapsed_ms is not None and elapsed_ms >= 0.0:
        text += f" time={elapsed_ms:.1f}ms"
    cv2.putText(canvas, text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 230, 255), 4, cv2.LINE_AA)
    region_palette = [(0, 255, 255), (255, 0, 255), (0, 220, 120), (255, 180, 0), (0, 180, 255)]
    for idx, region in enumerate(result.regions):
        color = region_palette[idx % len(region_palette)]
        pts = np.round(np.asarray(region.points, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        tx = int(round(region.points[0][0]))
        ty = int(round(region.points[0][1]))
        cv2.putText(canvas, region.label_name, (tx, max(20, ty - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return canvas


def _draw_match_overlay(detector: Line2DupLikeDetector, image_bgr: np.ndarray, match) -> np.ndarray:
    out = image_bgr.copy()
    color = (0, 255, 0)
    t0 = detector.get_templates(match.class_id, match.template_id, backend=match.backend)[0]
    refined_transform = match.refined_transform
    for f in t0.features:
        px = float(f.x + match.x)
        py = float(f.y + match.y)
        if refined_transform is not None:
            px, py = _apply_affine_point(refined_transform, px, py)
        pxi = int(round(px))
        pyi = int(round(py))
        theta = float(f.theta)
        if not np.isfinite(theta):
            theta = label_to_angle_deg(int(f.label))
        rad = np.deg2rad(theta)
        p2x = float(f.x + match.x + 7.0 * float(np.cos(rad)))
        p2y = float(f.y + match.y + 7.0 * float(np.sin(rad)))
        if refined_transform is not None:
            p2x, p2y = _apply_affine_point(refined_transform, p2x, p2y)
        cv2.arrowedLine(out, (pxi, pyi), (int(round(p2x)), int(round(p2y))), (0, 0, 0), 3, cv2.LINE_AA, 0, 0.35)
        cv2.arrowedLine(out, (pxi, pyi), (int(round(p2x)), int(round(p2y))), color, 1, cv2.LINE_AA, 0, 0.35)
    corners = match_quad(detector, match)
    pts_i = np.array([[int(round(x)), int(round(y))] for x, y in corners], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    return out


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


class Line2DupTemplateDialog(QtWidgets.QDialog):
    modelSaved = QtCore.Signal(str, str)
    referenceRegionsChanged = QtCore.Signal()

    def __init__(
        self,
        *,
        product_name: str,
        product_dir: str,
        initial_image_path: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"模板页 - {product_name}")
        self.resize(1450, 920)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_DIALOG_STYLESHEET)

        self.product_name = product_name
        self.product_dir = product_dir
        self.paths = product_paths(product_dir)

        self.image_path = initial_image_path
        self.image_bgr: Optional[np.ndarray] = None

        self.detector: Optional[Line2DupLikeDetector] = None
        self.detector_path: str = ""
        self.find_detector: Optional[Line2DupLikeDetector] = None
        self.find_detector_path: str = ""
        self.find_model_path = self.paths.model_path

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
        self._search_roi_shape_type: str = ""
        self._search_roi_points: List[List[float]] = []
        self._find_result_cache: Dict[str, Dict[str, object]] = {}

        self._build_ui()
        self._load_recipe()
        if self.image_path and os.path.exists(self.image_path):
            self._set_image(self.image_path, reset_state=False)
        if os.path.exists(self.paths.model_path):
            self._load_existing_model(silent=True)
        self._update_find_model_label()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self.tab_create = QtWidgets.QWidget()
        self.tab_reference = QtWidgets.QWidget()
        self.tab_find = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_create, "Create")
        self.tabs.addTab(self.tab_reference, "Reference ROI")
        self.tabs.addTab(self.tab_find, "Find")

        self._build_create_tab()
        self._build_reference_tab()
        self._build_find_tab()

    def _build_create_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout(self.tab_create)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)

        file_box = QtWidgets.QGroupBox("参考图")
        file_l = QtWidgets.QGridLayout(file_box)
        self.lbl_image = QtWidgets.QLabel("(未加载)")
        self.lbl_image.setWordWrap(True)
        self.btn_open_image = QtWidgets.QPushButton("打开图片...")
        self.btn_open_image.clicked.connect(self._pick_image)
        self.btn_load_model = QtWidgets.QPushButton("加载已有模型")
        self.btn_load_model.clicked.connect(lambda: self._load_existing_model(silent=False))
        file_l.addWidget(self.lbl_image, 0, 0, 1, 2)
        file_l.addWidget(self.btn_open_image, 1, 0)
        file_l.addWidget(self.btn_load_model, 1, 1)
        left.addWidget(file_box)

        select_box = QtWidgets.QGroupBox("模板编辑")
        select_l = QtWidgets.QGridLayout(select_box)
        self.cmb_role = QtWidgets.QComboBox()
        self.cmb_role.addItems(["template_roi", "exclude_mask"])
        self.cmb_role.currentTextChanged.connect(self._on_role_changed)
        self.btn_apply_selection = QtWidgets.QPushButton("应用当前框")
        self.btn_apply_selection.clicked.connect(self._apply_current_selection)
        self.btn_clear_roi = QtWidgets.QPushButton("清空模板ROI")
        self.btn_clear_roi.clicked.connect(self._clear_template_roi)
        self.btn_clear_masks = QtWidgets.QPushButton("清空Mask")
        self.btn_clear_masks.clicked.connect(self._clear_masks)
        select_l.addWidget(QtWidgets.QLabel("当前用途"), 0, 0)
        select_l.addWidget(self.cmb_role, 0, 1)
        select_l.addWidget(self.btn_apply_selection, 1, 0, 1, 2)
        select_l.addWidget(self.btn_clear_roi, 2, 0)
        select_l.addWidget(self.btn_clear_masks, 2, 1)
        left.addWidget(select_box)

        point_box = QtWidgets.QGroupBox("特征点编辑")
        point_l = QtWidgets.QGridLayout(point_box)
        self.chk_edit_points = QtWidgets.QCheckBox("启用点编辑")
        self.chk_edit_points.toggled.connect(self._on_point_edit_toggled)
        self.spin_point_label = QtWidgets.QSpinBox()
        self.spin_point_label.setRange(0, 7)
        self.btn_reset_points = QtWidgets.QPushButton("恢复模型特征点")
        self.btn_reset_points.clicked.connect(self._reset_editor_levels_from_detector)
        self.lbl_point_help = QtWidgets.QLabel("左击添加点，短拖可设置方向；右击删除最近特征点。")
        self.lbl_point_help.setWordWrap(True)
        point_l.addWidget(self.chk_edit_points, 0, 0, 1, 2)
        point_l.addWidget(QtWidgets.QLabel("默认方向label"), 1, 0)
        point_l.addWidget(self.spin_point_label, 1, 1)
        point_l.addWidget(self.btn_reset_points, 2, 0, 1, 2)
        point_l.addWidget(self.lbl_point_help, 3, 0, 1, 2)
        left.addWidget(point_box)

        param_box = QtWidgets.QGroupBox("模板参数")
        form = QtWidgets.QFormLayout(param_box)
        self.edit_class_id = QtWidgets.QLineEdit(self.product_name or "object")
        self.edit_levels = QtWidgets.QLineEdit("4,8")
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
        self.spin_angle_start.setValue(0.0)
        self.spin_angle_end = QtWidgets.QDoubleSpinBox()
        self.spin_angle_end.setRange(-360.0, 360.0)
        self.spin_angle_end.setValue(360.0)
        self.spin_angle_step = QtWidgets.QDoubleSpinBox()
        self.spin_angle_step.setRange(0.1, 360.0)
        self.spin_angle_step.setValue(10.0)
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
        left.addWidget(param_box)

        recipe_box = QtWidgets.QGroupBox("定位Recipe")
        recipe_form = QtWidgets.QFormLayout(recipe_box)
        self.cmb_backend_create = QtWidgets.QComboBox()
        self.cmb_backend_create.addItems([label for label, _ in BACKEND_ITEMS])
        self.cmb_backend_create.setCurrentText("Original")
        self.spin_threshold_create = QtWidgets.QDoubleSpinBox()
        self.spin_threshold_create.setRange(0.0, 100.0)
        self.spin_threshold_create.setValue(70.0)
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
        self.btn_build = QtWidgets.QPushButton("创建并保存模型")
        self.btn_build.clicked.connect(self._build_and_save)
        action_row.addWidget(self.btn_build)
        left.addLayout(action_row)

        self.lbl_status = QtWidgets.QLabel("状态：先选参考图并设置 template_roi。")
        self.lbl_status.setWordWrap(True)
        left.addWidget(self.lbl_status)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        self.create_canvas = RoiCanvas()
        self.canvas = self.create_canvas
        self.create_canvas.setMinimumSize(840, 720)
        self.create_canvas.draw_shape = "rect"
        self.create_canvas.shapesChanged.connect(self._on_create_canvas_shape_changed)
        self.create_canvas.imagePressed.connect(self._on_create_canvas_pressed)
        self.create_canvas.imageMoved.connect(self._on_create_canvas_moved)
        self.create_canvas.imageReleased.connect(self._on_create_canvas_released)
        right.addWidget(self.create_canvas, 1)

        help_text = QtWidgets.QLabel(
            "Create：先画 template_roi，再按需画 exclude_mask。"
            "\n启用点编辑后：左击添加点，短拖设置方向，右击删除最近特征点。"
        )
        help_text.setWordWrap(True)
        right.addWidget(help_text)

        self._on_role_changed(self.cmb_role.currentText())
        self._on_point_edit_toggled(False)

    def _build_reference_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout(self.tab_reference)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)

        info_box = QtWidgets.QGroupBox("选中 ROI 属性")
        info_form = QtWidgets.QFormLayout(info_box)
        self.edit_reference_label = QtWidgets.QLineEdit()
        self.edit_output_label = QtWidgets.QLineEdit()
        self.edit_output_label.setPlaceholderText("先在列表中选择一个 ROI")
        self.edit_display_name = QtWidgets.QLineEdit()
        self.edit_display_name.setPlaceholderText("先在列表中选择一个 ROI")
        self.cmb_reference_shape = QtWidgets.QComboBox()
        self.cmb_reference_shape.addItems(["rectangle", "polygon"])
        self.cmb_reference_shape.currentTextChanged.connect(self._on_reference_shape_changed)
        self.edit_output_label.textChanged.connect(self._on_region_field_edited)
        self.edit_display_name.textChanged.connect(self._on_region_field_edited)
        self.edit_output_label.returnPressed.connect(self._apply_reference_region_fields)
        self.edit_display_name.returnPressed.connect(self._apply_reference_region_fields)
        self.btn_apply_region_name = QtWidgets.QPushButton("应用名称")
        self.btn_apply_region_name.clicked.connect(self._apply_reference_region_fields)
        info_form.addRow("ROI 标签", self.edit_output_label)
        info_form.addRow("显示名称", self.edit_display_name)
        name_btn_row = QtWidgets.QHBoxLayout()
        name_btn_row.addWidget(self.btn_apply_region_name)
        name_btn_row.addWidget(self.cmb_reference_shape)
        info_form.addRow("形状 / 操作", name_btn_row)
        self.edit_output_label.setEnabled(False)
        self.edit_display_name.setEnabled(False)
        self.btn_apply_region_name.setEnabled(False)
        left.addWidget(info_box)

        region_box = QtWidgets.QGroupBox("Reference Regions")
        region_l = QtWidgets.QVBoxLayout(region_box)
        region_body = QtWidgets.QHBoxLayout()
        self.table_reference_regions = QtWidgets.QTableWidget(0, 4)
        self.table_reference_regions.setHorizontalHeaderLabels(["#", "Name", "ROI Label", "Info"])
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
        self.btn_add_reference_roi = QtWidgets.QPushButton("新建ROI")
        self.btn_add_reference_roi.clicked.connect(self._prepare_new_reference_roi)
        self.btn_remove_reference_roi = QtWidgets.QPushButton("删除选中ROI")
        self.btn_remove_reference_roi.clicked.connect(self._remove_selected_reference_roi)
        self.btn_clear_reference_rois = QtWidgets.QPushButton("清空全部ROI")
        self.btn_clear_reference_rois.clicked.connect(self._clear_reference_roi)
        self.btn_load_reference_roi = QtWidgets.QPushButton("加载已有参考ROI")
        self.btn_load_reference_roi.clicked.connect(lambda: self._load_reference_roi_from_json(silent=False))
        self.btn_save_reference_roi = QtWidgets.QPushButton("保存当前ROI")
        self.btn_save_reference_roi.clicked.connect(self._save_reference_roi_to_json)
        for button in [
            self.btn_add_reference_roi,
            self.btn_load_reference_roi,
            self.btn_remove_reference_roi,
            self.btn_clear_reference_rois,
            self.btn_save_reference_roi,
        ]:
            button.setMinimumWidth(150)
            region_btn_col.addWidget(button)
        region_btn_col.addStretch(1)
        region_body.addLayout(region_btn_col)
        region_l.addLayout(region_body, 1)
        left.addWidget(region_box)

        self.lbl_reference_status = QtWidgets.QLabel("状态：这里设置的是最终输出给 embedding 的参考ROI。")
        self.lbl_reference_status.setWordWrap(True)
        left.addWidget(self.lbl_reference_status)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        self.ref_canvas = RoiCanvas()
        self.ref_canvas.setMinimumSize(840, 720)
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

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)

        model_box = QtWidgets.QGroupBox("模型")
        model_l = QtWidgets.QGridLayout(model_box)
        self.lbl_find_model = QtWidgets.QLabel("(未设置)")
        self.lbl_find_model.setWordWrap(True)
        self.btn_find_open_model = QtWidgets.QPushButton("打开模型...")
        self.btn_find_open_model.clicked.connect(self._pick_find_model)
        self.btn_find_use_product = QtWidgets.QPushButton("使用产品模型")
        self.btn_find_use_product.clicked.connect(self._use_product_model)
        model_l.addWidget(self.lbl_find_model, 0, 0, 1, 2)
        model_l.addWidget(self.btn_find_open_model, 1, 0)
        model_l.addWidget(self.btn_find_use_product, 1, 1)
        left.addWidget(model_box)

        list_box = QtWidgets.QGroupBox("测试图片")
        list_l = QtWidgets.QVBoxLayout(list_box)
        self.list_find_images = QtWidgets.QListWidget()
        self.list_find_images.itemDoubleClicked.connect(self._run_find_for_item)
        self.list_find_images.currentItemChanged.connect(self._on_find_item_selected)
        list_l.addWidget(self.list_find_images, 1)
        list_btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_find_images = QtWidgets.QPushButton("Add Images")
        self.btn_add_find_images.clicked.connect(self._add_find_images)
        self.btn_remove_find_image = QtWidgets.QPushButton("Remove")
        self.btn_remove_find_image.clicked.connect(self._remove_selected_find_images)
        self.btn_clear_find_images = QtWidgets.QPushButton("Clear")
        self.btn_clear_find_images.clicked.connect(self._clear_find_images)
        list_btn_row.addWidget(self.btn_add_find_images)
        list_btn_row.addWidget(self.btn_remove_find_image)
        list_btn_row.addWidget(self.btn_clear_find_images)
        list_l.addLayout(list_btn_row)
        left.addWidget(list_box, 1)

        recipe_box = QtWidgets.QGroupBox("Find 参数")
        recipe_form = QtWidgets.QFormLayout(recipe_box)
        self.cmb_backend_find = QtWidgets.QComboBox()
        self.cmb_backend_find.addItems([label for label, _ in BACKEND_ITEMS])
        self.cmb_backend_find.setCurrentText("Original")
        self.spin_threshold_find = QtWidgets.QDoubleSpinBox()
        self.spin_threshold_find.setRange(0.0, 100.0)
        self.spin_threshold_find.setValue(70.0)
        self.spin_nms_find = QtWidgets.QDoubleSpinBox()
        self.spin_nms_find.setRange(0.0, 1.0)
        self.spin_nms_find.setDecimals(2)
        self.spin_nms_find.setValue(0.3)
        self.cmb_follow_find = QtWidgets.QComboBox()
        self.cmb_follow_find.addItems(["affine_roi", "match_bbox"])
        self.chk_auto_threshold_sweep = QtWidgets.QCheckBox("Auto Threshold Sweep")
        self.chk_auto_threshold_sweep.setToolTip("未命中时按当前阈值每次减 10，最低扫到 20。")
        self.chk_auto_threshold_sweep.toggled.connect(lambda _checked: self._save_recipe())
        recipe_form.addRow("backend", self.cmb_backend_find)
        recipe_form.addRow("threshold", self.spin_threshold_find)
        recipe_form.addRow("nms_iou", self.spin_nms_find)
        recipe_form.addRow("follow_mode", self.cmb_follow_find)
        recipe_form.addRow("", self.chk_auto_threshold_sweep)
        left.addWidget(recipe_box)

        search_box = QtWidgets.QGroupBox("搜索ROI")
        search_l = QtWidgets.QVBoxLayout(search_box)
        self.btn_apply_search_roi = QtWidgets.QPushButton("应用当前框为搜索ROI")
        self.btn_apply_search_roi.clicked.connect(self._apply_find_search_roi)
        self.btn_clear_search_roi = QtWidgets.QPushButton("清空搜索ROI")
        self.btn_clear_search_roi.clicked.connect(self._clear_find_search_roi)
        self.lbl_search_roi = QtWidgets.QLabel("状态：未设置搜索ROI，默认全图搜索。")
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
        self.btn_run_find_selected = QtWidgets.QPushButton("Run Selected")
        self.btn_run_find_selected.clicked.connect(self._run_selected_find)
        self.btn_run_find_all = QtWidgets.QPushButton("Run All")
        self.btn_run_find_all.clicked.connect(self._run_all_find)
        run_row.addWidget(self.btn_run_find_selected)
        run_row.addWidget(self.btn_run_find_all)
        left.addLayout(run_row)

        self.lbl_find_status = QtWidgets.QLabel("状态：双击某张图片可直接测试。")
        self.lbl_find_status.setWordWrap(True)
        left.addWidget(self.lbl_find_status)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        self.find_canvas = RoiCanvas()
        self.find_canvas.setMinimumSize(840, 720)
        right.addWidget(self.find_canvas, 1)

    def _pick_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择参考图",
            self.image_path or "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self._set_image(path, reset_state=True)

    def _set_image(self, path: str, *, reset_state: bool) -> None:
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            QtWidgets.QMessageBox.critical(self, "读取失败", path)
            return
        self.image_path = path
        self.image_bgr = image_bgr
        self.lbl_image.setText(path)
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
        self._refresh_create_overlays()
        if not self._apply_reference_roi_from_recipe():
            self._load_reference_roi_from_json(silent=True)
        self._save_recipe()

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

    def _recipe_from_controls(self, *, use_find_values: bool = False) -> Line2DupRecipe:
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
        reference_shape_type = str((first_region or {}).get("shape_type", self._recipe_reference_shape_type))
        reference_points = list((first_region or {}).get("points", self._recipe_reference_points))
        reference_label = str((first_region or {}).get("reference_label", "roi") or "roi")
        output_label = str((first_region or {}).get("output_label", reference_label) or reference_label)
        return Line2DupRecipe(
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
            reference_regions=reference_regions or None,
            search_shape_type=str(self._search_roi_shape_type or ""),
            search_points=[list(pt) for pt in (self._search_roi_points or [])] or None,
        )

    def _save_recipe(self) -> None:
        recipe = self._recipe_from_controls(use_find_values=False)
        recipe.model_path = self.paths.model_path
        save_recipe(recipe, self.paths.recipe_path)

    def _load_recipe(self) -> None:
        if not os.path.exists(self.paths.recipe_path):
            self._refresh_reference_region_fields()
            self._sync_recipe_controls("create")
            return
        recipe = load_recipe(self.paths.recipe_path)
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
        self._sync_recipe_controls("create")

    def _apply_reference_roi_from_recipe(self) -> bool:
        if not self._reference_regions:
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
        self.lbl_reference_status.setText("状态：已从 recipe 恢复参考ROI。")
        return bool(self._reference_regions)

    def _load_existing_model(self, *, silent: bool) -> None:
        if not os.path.exists(self.paths.model_path):
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "当前产品目录还没有模板模型。")
            return
        try:
            detector = self._get_detector_for_model(self.paths.model_path, reuse_shared=False)
            class_ids = detector.class_ids()
            if not class_ids:
                raise RuntimeError("模型中没有 class")
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
                QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
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
        self.template_roi = roi_rect
        self.mask_rects = [MaskRect(x=int(r.x), y=int(r.y), w=int(r.w), h=int(r.h)) for r in mask_rects]
        self.editor_levels = clone_levels(editor_levels)
        self.original_mode = str(source_info.get("original_mode", "auto")) if isinstance(source_info, dict) else "auto"
        self.points_dirty = False
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self.create_canvas.clear_roi()
        self.edit_class_id.setText(class_id)
        self.find_model_path = self.paths.model_path
        self._update_find_model_label()
        self._refresh_create_overlays()
        self.lbl_status.setText(
            f"状态：已加载模型 {os.path.basename(self.paths.model_path)}，特征点={self._feature_count}"
        )

    def _build_and_save(self) -> None:
        if self.image_bgr is None or not self.image_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载参考图。")
            return
        if self.template_roi is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先设置 template_roi。")
            return

        x, y, w, h = self.template_roi.x, self.template_roi.y, self.template_roi.w, self.template_roi.h
        roi_img = self.image_bgr[y : y + h, x : x + w].copy()
        if roi_img.size == 0:
            QtWidgets.QMessageBox.critical(self, "模板无效", "模板ROI超出图像范围。")
            return

        original_mode = "manual_points" if (self.points_dirty or self.original_mode == "manual_points") and self.editor_levels else "auto"
        original_editor_levels = self.editor_levels if original_mode == "manual_points" else None

        try:
            pose_infos = pose_infos_from_ui_values(
                float(self.spin_angle_start.value()),
                float(self.spin_angle_end.value()),
                float(self.spin_angle_step.value()),
                float(self.spin_scale_start.value()),
                float(self.spin_scale_end.value()),
                float(self.spin_scale_step.value()),
            )
            detector, kept, skipped = build_multi_backend_detector(
                class_id=self.edit_class_id.text().strip() or self.product_name or "object",
                roi_img=roi_img,
                roi_rect=self.template_roi,
                mask_rects=self.mask_rects,
                pose_infos=pose_infos,
                pose_ui=self._pose_ui_values(),
                levels=parse_levels(self.edit_levels.text().strip()),
                num_features=int(self.spin_num_features.value()),
                weak_threshold=float(self.spin_weak.value()),
                strong_threshold=float(self.spin_strong.value()),
                original_mode=original_mode,
                original_editor_levels=original_editor_levels,
                source_image_path=self.image_path,
            )
            save_detector_model(detector, self.paths.model_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "创建失败", str(exc))
            return

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
        self.create_canvas.clear_roi()
        self._refresh_create_overlays()
        self.lbl_status.setText(
            f"状态：模型已保存，kept={kept} skipped={skipped} 特征点={self._feature_count}"
        )
        self.modelSaved.emit(self.paths.model_path, self.paths.recipe_path)
        QtWidgets.QMessageBox.information(self, "完成", f"模板模型已保存：\n{self.paths.model_path}")

    def _pose_ui_values(self) -> Dict[str, float]:
        return {
            "angle_start": float(self.spin_angle_start.value()),
            "angle_end": float(self.spin_angle_end.value()),
            "angle_step": float(self.spin_angle_step.value()),
            "scale_start": float(self.spin_scale_start.value()),
            "scale_end": float(self.spin_scale_end.value()),
            "scale_step": float(self.spin_scale_step.value()),
        }

    def _apply_current_selection(self) -> None:
        xywh = self.create_canvas.roi_xywh()
        if xywh is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在图上拉一个矩形。")
            return
        role = self.cmb_role.currentText()
        if role == "template_roi":
            self.template_roi = RoiRect(*xywh)
            self.mask_rects = []
            self.editor_levels = []
            self.points_dirty = False
            self.original_mode = "auto"
            self._hover_feature_index = None
            self._point_drag_start = None
            self._point_drag_end = None
            self.lbl_status.setText("状态：已更新模板ROI。重新建模后会提取新的特征点。")
        else:
            if self.template_roi is None:
                QtWidgets.QMessageBox.warning(self, "提示", "请先设置 template_roi。")
                return
            rel = _clamp_rect_to_roi(xywh, self.template_roi)
            if rel is None:
                QtWidgets.QMessageBox.warning(self, "提示", "Mask 矩形需要和模板ROI相交。")
                return
            self.mask_rects.append(rel)
            self.lbl_status.setText(f"状态：已添加 {len(self.mask_rects)} 个 mask 矩形。")
        self.create_canvas.clear_roi()
        self._refresh_create_overlays()

    def _clear_template_roi(self) -> None:
        self.template_roi = None
        self.mask_rects = []
        self.editor_levels = []
        self.points_dirty = False
        self.original_mode = "auto"
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self.create_canvas.clear_roi()
        self._refresh_create_overlays()
        self.lbl_status.setText("状态：已清空模板ROI。")

    def _clear_masks(self) -> None:
        self.mask_rects = []
        self.create_canvas.clear_roi()
        self._refresh_create_overlays()
        self.lbl_status.setText("状态：已清空 mask。")

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

    def _on_point_edit_toggled(self, enabled: bool) -> None:
        self.create_canvas.set_interaction_enabled(not enabled)
        self.cmb_role.setEnabled(not enabled)
        self.btn_apply_selection.setEnabled(not enabled)
        self.btn_clear_roi.setEnabled(not enabled)
        self.btn_clear_masks.setEnabled(not enabled)
        if enabled:
            self.lbl_status.setText("状态：点编辑已开启。左击添加，短拖设方向，右击删最近点。")
        else:
            self._point_drag_start = None
            self._point_drag_end = None
        self._refresh_create_overlays()

    def _on_create_canvas_shape_changed(self) -> None:
        if self.chk_edit_points.isChecked():
            return

    def _on_create_canvas_pressed(self, button: int, x: int, y: int) -> None:
        if not self.chk_edit_points.isChecked():
            return
        if self.template_roi is None or not self.editor_levels:
            self.lbl_status.setText("状态：请先创建或加载模型后再编辑特征点。")
            return
        if not self._point_in_template_roi(x, y):
            return
        if button == _button_left():
            self._point_drag_start = (int(x), int(y))
            self._point_drag_end = (int(x), int(y))
            self._refresh_create_overlays()
        elif button == _button_right():
            if self._delete_feature_near(x, y):
                self.lbl_status.setText(f"状态：已删除特征点，剩余={self._feature_count}")
            else:
                self.lbl_status.setText("状态：右击位置附近没有可删除的特征点。")

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
        self.lbl_status.setText(f"状态：已更新特征点，当前总数={self._feature_count}")

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
        self._sync_editor_levels()
        self._refresh_create_overlays()

    def _delete_feature_near(self, x: int, y: int) -> bool:
        level0 = self._editor_level0()
        if level0 is None:
            return False
        idx = self._find_nearest_feature_index(x, y)
        if idx is None:
            return False
        del level0.features[int(idx)]
        self.points_dirty = True
        self._hover_feature_index = None
        self._sync_editor_levels()
        self._refresh_create_overlays()
        return True

    def _reset_editor_levels_from_detector(self) -> None:
        if self.detector is None:
            self.lbl_status.setText("状态：当前没有已加载模型。")
            return
        class_id = self.edit_class_id.text().strip() or (self.detector.class_ids()[0] if self.detector.class_ids() else "")
        if not class_id:
            self.lbl_status.setText("状态：模型里没有可恢复的 class。")
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
                self.lbl_status.setText(f"状态：恢复模型特征失败：{exc}")
                return
        self.editor_levels = clone_levels(levels)
        self.points_dirty = False
        self._hover_feature_index = None
        self._point_drag_start = None
        self._point_drag_end = None
        self._refresh_create_overlays()
        self.lbl_status.setText(f"状态：已恢复模型特征点，当前总数={self._feature_count}")

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

    def _next_reference_label(self) -> str:
        used = {
            str(region.get("reference_label") or region.get("output_label") or "")
            for region in self._reference_regions
            if isinstance(region, dict)
        }
        idx = 1
        while f"roi{idx}" in used:
            idx += 1
        return f"roi{idx}"

    def _region_points_from_canvas(self) -> Tuple[str, List[List[float]]]:
        if self.ref_canvas.roi.shape_type == "polygon" and self.ref_canvas.roi.points:
            return "polygon", [[float(x), float(y)] for x, y in self.ref_canvas.roi.points]
        xywh = self.ref_canvas.roi_xywh()
        if xywh is None:
            return "", []
        x, y, w, h = xywh
        return "rectangle", [[float(x), float(y)], [float(x + w), float(y + h)]]

    def _region_overlay_shape(self, region: Dict[str, object], color: QtGui.QColor, width: int, dash: bool) -> OverlayShape:
        shape_type = str(region.get("shape_type", "rectangle"))
        points = [
            (float(pt[0]), float(pt[1]))
            for pt in region.get("points", []) or []
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if shape_type == "polygon" and len(points) >= 3:
            return OverlayShape(shape_type="polygon", points=points, color=color, width=width, dash=dash)
        if len(points) >= 2:
            x0, y0 = float(points[0][0]), float(points[0][1])
            x1, y1 = float(points[1][0]), float(points[1][1])
            x = int(round(min(x0, x1)))
            y = int(round(min(y0, y1)))
            w = max(1, int(round(abs(x1 - x0))))
            h = max(1, int(round(abs(y1 - y0))))
            return OverlayShape(shape_type="rect", xywh=(x, y, w, h), color=color, width=width, dash=dash)
        return OverlayShape(shape_type="rect", xywh=(0, 0, 1, 1), color=color, width=width, dash=dash)

    def _refresh_reference_region_list(self) -> None:
        if not hasattr(self, "table_reference_regions"):
            return
        self.table_reference_regions.blockSignals(True)
        self.table_reference_regions.setRowCount(0)
        for idx, region in enumerate(self._reference_regions):
            label = str(region.get("output_label") or region.get("reference_label") or f"roi{idx + 1}")
            display_name = str(region.get("display_name") or region.get("name") or label).strip() or label
            shape_type = str(region.get("shape_type", "rectangle"))
            points = [
                [float(pt[0]), float(pt[1])]
                for pt in region.get("points", []) or []
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if shape_type == "polygon":
                info_text = f"Polygon · {len(points)} pts"
            elif len(points) >= 2:
                x0, y0 = float(points[0][0]), float(points[0][1])
                x1, y1 = float(points[1][0]), float(points[1][1])
                w = max(1, int(round(abs(x1 - x0))))
                h = max(1, int(round(abs(y1 - y0))))
                info_text = f"Rect · {w}x{h}"
            else:
                info_text = "Rect"
            row = self.table_reference_regions.rowCount()
            self.table_reference_regions.insertRow(row)
            values = [
                str(idx),
                display_name,
                label,
                info_text,
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, idx)
                if col == 0:
                    item.setTextAlignment(
                        int(
                            QtCore.Qt.AlignmentFlag.AlignHCenter
                            | QtCore.Qt.AlignmentFlag.AlignVCenter
                        )
                    )
                self.table_reference_regions.setItem(row, col, item)
        if self._selected_reference_idx is not None and 0 <= self._selected_reference_idx < self.table_reference_regions.rowCount():
            self.table_reference_regions.setCurrentCell(self._selected_reference_idx, 0)
        self.table_reference_regions.blockSignals(False)

    def _refresh_reference_region_fields(self) -> None:
        self.edit_output_label.blockSignals(True)
        self.edit_display_name.blockSignals(True)
        try:
            has_selection = (
                self._selected_reference_idx is not None
                and 0 <= self._selected_reference_idx < len(self._reference_regions)
            )
            self.edit_output_label.setEnabled(has_selection)
            self.edit_display_name.setEnabled(has_selection)
            self.btn_apply_region_name.setEnabled(has_selection)
            if not has_selection:
                self.edit_output_label.setText("")
                self.edit_display_name.setText("")
                return
            region = self._reference_regions[self._selected_reference_idx]
            label = str(region.get("output_label") or region.get("reference_label") or "").strip()
            display_name = str(region.get("display_name") or region.get("name") or label).strip() or label
            self.edit_output_label.setText(label)
            self.edit_display_name.setText(display_name)
        finally:
            self.edit_output_label.blockSignals(False)
            self.edit_display_name.blockSignals(False)

    def _on_region_field_edited(self) -> None:
        pass

    def _apply_reference_region_fields(self) -> None:
        if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
            return
        region = self._reference_regions[self._selected_reference_idx]
        label = self.edit_output_label.text().strip()
        if not label:
            label = str(region.get("output_label") or region.get("reference_label") or "").strip()
        if not label:
            label = self._next_reference_label()
        display_name = self.edit_display_name.text().strip() or label
        region["reference_label"] = label
        region["output_label"] = label
        region["display_name"] = display_name
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._save_recipe()
        self.referenceRegionsChanged.emit()
        self.lbl_reference_status.setText(f"状态：已更新参考ROI名称：{display_name}")

    def _refresh_reference_canvas(self) -> None:
        overlays: List[OverlayShape] = []
        inactive_color = QtGui.QColor(255, 0, 255)
        for idx, region in enumerate(self._reference_regions):
            if idx == self._selected_reference_idx:
                continue
            overlays.append(self._region_overlay_shape(region, inactive_color, 1.8, False))
        self.ref_canvas.set_overlays(overlays)
        self._syncing_reference_view = True
        try:
            if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
                self.ref_canvas.clear_roi()
            else:
                region = self._reference_regions[self._selected_reference_idx]
                shape_type = str(region.get("shape_type", "rectangle"))
                points = [
                    (float(pt[0]), float(pt[1]))
                    for pt in region.get("points", []) or []
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]
                self.cmb_reference_shape.setCurrentText("polygon" if shape_type == "polygon" else "rectangle")
                if shape_type == "polygon" and len(points) >= 3:
                    self.ref_canvas.set_roi_polygon(points)
                elif len(points) >= 2:
                    x0, y0 = float(points[0][0]), float(points[0][1])
                    x1, y1 = float(points[1][0]), float(points[1][1])
                    self.ref_canvas.set_roi_rect(
                        (
                            int(round(min(x0, x1))),
                            int(round(min(y0, y1))),
                            max(1, int(round(abs(x1 - x0)))),
                            max(1, int(round(abs(y1 - y0)))),
                        )
                    )
                else:
                    self.ref_canvas.clear_roi()
        finally:
            self._syncing_reference_view = False

    def _on_reference_region_selected(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if current_row < 0 or current_row >= len(self._reference_regions):
            self._selected_reference_idx = None
        else:
            self._selected_reference_idx = current_row
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()

    def _prepare_new_reference_roi(self) -> None:
        self._selected_reference_idx = None
        if hasattr(self, "table_reference_regions"):
            self.table_reference_regions.blockSignals(True)
            self.table_reference_regions.clearSelection()
            self.table_reference_regions.setCurrentIndex(QtCore.QModelIndex())
            self.table_reference_regions.blockSignals(False)
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self.lbl_reference_status.setText("状态：已切换到新增ROI模式，请直接在右侧画布上继续画框。")

    def _remove_selected_reference_roi(self) -> None:
        if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
            return
        del self._reference_regions[self._selected_reference_idx]
        self._selected_reference_idx = None
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self._save_recipe()
        self.referenceRegionsChanged.emit()
        self.lbl_reference_status.setText("状态：已删除选中的参考ROI。")

    def _on_reference_canvas_shape_changed(self) -> None:
        if self._syncing_reference_view:
            return
        shape_type, points = self._region_points_from_canvas()
        if not shape_type or not points:
            return
        if self._selected_reference_idx is None:
            label = self._next_reference_label()
            self._reference_regions.append(
                {
                    "reference_label": label,
                    "output_label": label,
                    "display_name": label,
                    "shape_type": shape_type,
                    "points": points,
                }
            )
            self._selected_reference_idx = len(self._reference_regions) - 1
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            self._save_recipe()
            self.lbl_reference_status.setText(f"状态：已新增参考ROI：{label}")
            return
        if 0 <= self._selected_reference_idx < len(self._reference_regions):
            region = self._reference_regions[self._selected_reference_idx]
            region["shape_type"] = shape_type
            region["points"] = points
            label = str(region.get("output_label") or region.get("reference_label") or f"roi{self._selected_reference_idx + 1}")
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._save_recipe()
            self.lbl_reference_status.setText(f"状态：已更新参考ROI：{label}")

    def _on_reference_shape_changed(self, shape_name: str) -> None:
        self.ref_canvas.draw_shape = "polygon" if shape_name == "polygon" else "rect"

    def _load_reference_roi_from_json(self, *, silent: bool) -> None:
        if not self.image_path or not os.path.exists(self.image_path):
            if not silent:
                QtWidgets.QMessageBox.warning(self, "提示", "请先加载参考图。")
            return
        jpath = qr_core.labelme_json_of_image(self.image_path)
        if not os.path.exists(jpath):
            self._reference_regions = []
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            if not silent:
                QtWidgets.QMessageBox.information(self, "提示", "当前参考图还没有 labelme json。")
            return
        try:
            regions: List[Dict[str, object]] = []
            for shape in qr_core.list_shapes_from_labelme(jpath, label_prefix="roi"):
                label_name = str(shape.get("label", "")).strip()
                if not label_name:
                    continue
                shape_type = str(shape.get("shape_type", "rectangle"))
                if shape_type == "polygon":
                    points = [[float(x), float(y)] for x, y in shape.get("points", [])]
                    if len(points) < 3:
                        continue
                else:
                    xywh = _shape_to_rect(shape)
                    if xywh is None:
                        continue
                    x, y, w, h = xywh
                    points = [[float(x), float(y)], [float(x + w), float(y + h)]]
                    shape_type = "rectangle"
                regions.append(
                    {
                        "reference_label": label_name,
                        "output_label": label_name,
                        "display_name": label_name,
                        "shape_type": shape_type,
                        "points": points,
                    }
                )
            if not regions:
                raise RuntimeError("json 中没有 roi1/roi2/... 参考ROI")
            self._reference_regions = regions
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            self.lbl_reference_status.setText(f"状态：已加载 {len(regions)} 个参考ROI")
            self._save_recipe()
            self.referenceRegionsChanged.emit()
        except Exception as exc:
            self._reference_regions = []
            self._selected_reference_idx = None
            self._refresh_reference_region_list()
            self._refresh_reference_canvas()
            self._refresh_reference_region_fields()
            if not silent:
                QtWidgets.QMessageBox.warning(self, "加载失败", str(exc))

    def _save_reference_roi_to_json(self) -> None:
        if not self.image_path or not os.path.exists(self.image_path):
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载参考图。")
            return
        try:
            if not self._reference_regions:
                raise RuntimeError("当前没有可保存的参考ROI。")
            old_recipe = load_recipe(self.paths.recipe_path) if os.path.exists(self.paths.recipe_path) else Line2DupRecipe()
            old_labels = {
                str(region.get("output_label") or region.get("reference_label") or "")
                for region in (old_recipe.reference_regions or [])
                if isinstance(region, dict)
            }
            new_labels = {
                str(region.get("output_label") or region.get("reference_label") or "")
                for region in self._reference_regions
                if isinstance(region, dict)
            }
            for label_name in old_labels - new_labels:
                if label_name:
                    qr_core.delete_labelme_shape(self.image_path, label_name=label_name)
            for region in self._reference_regions:
                label_name = str(region.get("output_label") or region.get("reference_label") or "").strip()
                shape_type = str(region.get("shape_type", "rectangle"))
                points = [
                    (float(pt[0]), float(pt[1]))
                    for pt in region.get("points", []) or []
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]
                if not label_name or len(points) < 2:
                    continue
                if shape_type == "polygon" and len(points) >= 3:
                    qr_core.upsert_labelme_polygon(self.image_path, points, label_name=label_name)
                else:
                    x0, y0 = points[0]
                    x1, y1 = points[1]
                    qr_core.upsert_labelme_rect(
                        self.image_path,
                        (
                            int(round(min(x0, x1))),
                            int(round(min(y0, y1))),
                            max(1, int(round(abs(x1 - x0)))),
                            max(1, int(round(abs(y1 - y0)))),
                        ),
                        label_name=label_name,
                    )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._save_recipe()
        self.referenceRegionsChanged.emit()
        self.lbl_reference_status.setText(f"状态：参考ROI已保存，共 {len(self._reference_regions)} 个")

    def _clear_reference_roi(self) -> None:
        self._reference_regions = []
        self._selected_reference_idx = None
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._refresh_reference_region_fields()
        self._recipe_reference_shape_type = ""
        self._recipe_reference_points = []
        self._save_recipe()
        self.referenceRegionsChanged.emit()
        self.lbl_reference_status.setText("状态：已清空全部参考ROI。")

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
            self.lbl_search_roi.setText("状态：未设置搜索ROI，默认全图搜索。")
            return
        x, y, w, h = xywh
        self.lbl_search_roi.setText(f"状态：搜索ROI=({x},{y},{w},{h})，Find 与主界面模板定位都会限制在此区域搜索。")

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
            QtWidgets.QMessageBox.information(self, "提示", "请先在右侧图片上拖一个矩形搜索区域。")
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
            "选择模型",
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
        text = self.find_model_path or "(未设置)"
        self.lbl_find_model.setText(text)
        self.lbl_find_model.setToolTip(text)

    def _add_find_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择测试图片",
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
            self.lbl_find_status.setText("状态：Find 列表为空。")

    def _clear_find_images(self) -> None:
        self._find_result_cache.clear()
        self.list_find_images.clear()
        self.find_canvas.clear_image()
        self.lbl_find_status.setText("状态：已清空 Find 列表。")

    def _run_selected_find(self) -> None:
        item = self.list_find_images.currentItem()
        if item is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先选择一张测试图片。")
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
        self.lbl_find_status.setText(f"状态：已切换到 {os.path.basename(scene_path)}。双击或点 Run Selected 开始测试。")

    def _get_detector_for_model(self, model_path: str, *, reuse_shared: bool = True) -> Line2DupLikeDetector:
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
            QtWidgets.QMessageBox.warning(self, "提示", "请先在 Find 页加载可用模型。")
            return
        recipe = self._recipe_from_controls(use_find_values=True)
        recipe.model_path = model_path
        if not recipe.reference_image and self.image_path:
            recipe.reference_image = self.image_path
        if not recipe.reference_image or not os.path.exists(recipe.reference_image):
            QtWidgets.QMessageBox.warning(self, "提示", "请先设置参考图。")
            return

        scene_bgr = cv2.imread(scene_path, cv2.IMREAD_COLOR)
        if scene_bgr is None:
            self._set_find_item_error(item, f"读图失败: {scene_path}")
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
                    if "line2dup did not find any match" in str(exc):
                        last_exc = exc
                        continue
                    raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if result is None:
                raise last_exc or RuntimeError("line2dup did not find any match")
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
        status_msg = f"状态：{os.path.basename(scene_path)} 测试完成，{msg}"
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
        status_msg = f"状态：测试失败：{message}"
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
        self._clear_find_images()
        super().closeEvent(event)


__all__ = ["Line2DupTemplateDialog"]
