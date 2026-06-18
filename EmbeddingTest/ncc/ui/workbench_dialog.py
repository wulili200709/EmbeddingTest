from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.image_io import imread
from algorithms.labelme import (
    labelme_json_of_image,
    list_shapes_from_labelme,
    upsert_labelme_polygon,
    upsert_labelme_rect,
)
from ncc.authoring import (
    ensure_default_assets,
    mask_image_path,
    preview_image_path,
    set_source_from_image_file,
    set_template_from_roi,
    source_image_path,
    template_image_path,
)
from ncc.locator import resolved_model_path_for_product
from ncc.model import (
    NccAngleRange,
    NccAngleSearch,
    NccMatchModel,
    NccMatchOptions,
    NccMatchRect,
    NccMatchResult,
    NccReferenceRegion,
    create_default_model,
    load_model,
    model_summary,
    save_model,
)
from ncc.runtime_service import NccCompiledModel, NccMatchResponse
from ui.debug.roi_canvas_pyside6 import OverlayShape, RoiCanvas


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
QListWidget,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QPlainTextEdit,
QTableWidget {
    background: #333333;
    color: #e0e0e0;
    border: 1px solid #5a5a5a;
    selection-background-color: #6ec0ff;
    selection-color: #1a1a1a;
}
QAbstractItemView {
    background: #333333;
    color: #e0e0e0;
    alternate-background-color: #383838;
    selection-background-color: #6ec0ff;
    selection-color: #1a1a1a;
}
QListWidget::item {
    padding: 6px 8px;
    min-height: 26px;
}
QListWidget::item:selected {
    background: #6ec0ff;
    color: #1a1a1a;
}
QHeaderView::section {
    background: #3a3a3a;
    color: #d0d0d0;
    border: 1px solid #404040;
    padding: 4px;
}
QLabel {
    color: #e0e0e0;
}
QWidget#nccTabPage {
    background: #2d2d2d;
}
QWidget#nccLeftScrollHost {
    background: #2f2f2f;
}
QSplitter::handle {
    background: #343434;
}
QSplitter::handle:hover {
    background: #4f4f4f;
}
"""


_LEFT_SCROLL_STYLESHEET = (
    "QScrollArea{background:#2f2f2f;border:none;}"
    "QScrollArea > QWidget > QWidget{background:#2f2f2f;}"
    "QScrollBar:vertical{background:#2f2f2f;width:10px;margin:0;}"
    "QScrollBar::handle:vertical{background:#5a5a5a;min-height:28px;border-radius:5px;}"
    "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
)


def _image_file_filter() -> str:
    return "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"


def _make_tab_page() -> QtWidgets.QWidget:
    page = QtWidgets.QWidget()
    page.setObjectName("nccTabPage")
    page.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    return page


def _make_horizontal_splitter() -> QtWidgets.QSplitter:
    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(10)
    return splitter


def _make_scrollable_side_panel(
    panel: QtWidgets.QWidget,
    *,
    min_width: int,
    max_width: int,
) -> QtWidgets.QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setStyleSheet(_LEFT_SCROLL_STYLESHEET)
    scroll.setMinimumWidth(min_width)
    scroll.setMaximumWidth(max_width)
    scroll.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred,
        QtWidgets.QSizePolicy.Policy.Expanding,
    )
    panel.setObjectName("nccLeftScrollHost")
    panel.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    panel.setMinimumWidth(max(0, min_width - 20))
    panel.setMaximumWidth(max_width)
    scroll.setWidget(panel)
    return scroll


def _same_xywh(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return tuple(int(v) for v in a) == tuple(int(v) for v in b)


def _shape_to_rect(shape: dict) -> Optional[Tuple[int, int, int, int]]:
    pts = list(shape.get("points", []) or [])
    if not pts:
        return None
    if str(shape.get("shape_type", "rectangle")) == "rectangle" and len(pts) >= 2:
        (x0, y0), (x1, y1) = pts[:2]
        x = int(round(min(float(x0), float(x1))))
        y = int(round(min(float(y0), float(y1))))
        w = max(1, int(round(abs(float(x1) - float(x0)))))
        h = max(1, int(round(abs(float(y1) - float(y0)))))
        return x, y, w, h
    arr = np.asarray(pts, dtype=np.float32)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    return int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))


def _rect_to_polygon(xywh: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
    x, y, w, h = [int(v) for v in xywh]
    return [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]


def _region_polygon_points(region: NccReferenceRegion) -> List[Tuple[float, float]]:
    if region.shape_type == "polygon" and len(region.points) >= 3:
        return [(float(x), float(y)) for x, y in region.points]
    if len(region.points) >= 2:
        (x0, y0), (x1, y1) = region.points[:2]
        x = int(round(min(float(x0), float(x1))))
        y = int(round(min(float(y0), float(y1))))
        w = max(1, int(round(abs(float(x1) - float(x0)))))
        h = max(1, int(round(abs(float(y1) - float(y0)))))
        return _rect_to_polygon((x, y, w, h))
    return []


def _point_hits_polygon(points: Sequence[Tuple[float, float]], x: float, y: float, *, tolerance: float = 4.0) -> bool:
    if len(points) < 3:
        return False
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, (float(x), float(y)), True)) >= -float(tolerance)


def _region_info_text(region: NccReferenceRegion) -> str:
    if region.shape_type == "polygon":
        return f"Polygon · {len(region.points)} pts"
    if len(region.points) >= 2:
        (x0, y0), (x1, y1) = region.points[:2]
        w = max(1, int(round(abs(float(x1) - float(x0)))))
        h = max(1, int(round(abs(float(y1) - float(y0)))))
        return f"Rect · {w}x{h}"
    return region.shape_type


class _NccFindWorker(QtCore.QObject):
    itemFinished = QtCore.Signal(int, str, object, str)
    progressChanged = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        model_path: str,
        model: NccMatchModel,
        scene_paths: Sequence[str],
        options: NccMatchOptions,
        search_roi: Optional[Tuple[int, int, int, int]],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model_path = str(model_path)
        self._model = model.normalized()
        self._scene_paths = [str(path or "").strip() for path in scene_paths if str(path or "").strip()]
        self._options = options.normalized()
        self._search_roi = search_roi

    @QtCore.Slot()
    def run(self) -> None:
        compiled: Optional[NccCompiledModel] = None
        try:
            compiled = NccCompiledModel(self._model_path, self._model)
            total = len(self._scene_paths)
            for index, scene_path in enumerate(self._scene_paths):
                self.progressChanged.emit(f"matching {index + 1}/{total}: {Path(scene_path).name}")
                try:
                    scene = imread(scene_path, cv2.IMREAD_COLOR)
                    if scene is None:
                        raise RuntimeError(f"failed to read scene image: {scene_path}")
                    response = compiled.match(
                        scene,
                        options=self._options,
                        search_roi=self._search_roi,
                    )
                    self.itemFinished.emit(index, scene_path, response, "")
                except Exception as exc:
                    self.itemFinished.emit(index, scene_path, None, str(exc))
        finally:
            if compiled is not None:
                try:
                    compiled.close()
                except Exception:
                    pass
            self.finished.emit()


class NccMatchWorkbenchDialog(QtWidgets.QDialog):
    modelSaved = QtCore.Signal(str)

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
        available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(
            min(1450, max(1100, available.width() - 40)),
            min(920, max(620, available.height() - 60)),
        )
        self.setMinimumSize(900, 560)
        self.setWindowTitle(f"NCC位置修正工具 - {product_name or '未命名产品'}")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_DIALOG_STYLESHEET)

        self._product_name = str(product_name or "").strip()
        self._product_dir = str(product_dir or "").strip()
        self._camera_role = str(camera_role or "cam1").strip().lower() or "cam1"
        self._initial_image_path = str(initial_image_path or "").strip()
        self._model_path = resolved_model_path_for_product(self._product_dir, self._camera_role)
        self._model: NccMatchModel = create_default_model()
        self._latest_response: Optional[NccMatchResponse] = None
        self._reference_regions: List[NccReferenceRegion] = []
        self._find_result_cache: dict[str, dict[str, object]] = {}
        self._selected_reference_idx: Optional[int] = None
        self._selected_reference_indices: Set[int] = set()
        self._syncing_roi = False
        self._syncing_mask_view = False
        self._syncing_reference_view = False
        self._syncing_reference_table = False
        self._moving_reference_regions = False
        self._reference_move_start: Optional[Tuple[float, float]] = None
        self._reference_move_original: Dict[int, List[Tuple[float, float]]] = {}
        self._loading_model = False
        self._suppress_source_roi_auto_apply = False
        self._find_running = False
        self._find_thread: Optional[QtCore.QThread] = None
        self._find_worker: Optional[_NccFindWorker] = None
        self._find_options_save_timer = QtCore.QTimer(self)
        self._find_options_save_timer.setSingleShot(True)
        self._find_options_save_timer.setInterval(250)
        self._find_options_save_timer.timeout.connect(self._save_find_options_to_model)
        self._find_elapsed_timer = QtCore.QElapsedTimer()
        self._find_progress_timer = QtCore.QTimer(self)
        self._find_progress_timer.setInterval(100)
        self._find_progress_timer.timeout.connect(self._update_find_progress_elapsed)
        self._find_active_paths: List[str] = []

        self._build_ui()
        self._load_model()
        self._load_initial_image()
        self._finalize_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        header_box = QtWidgets.QGroupBox("产品信息")
        header_form = QtWidgets.QFormLayout(header_box)
        self.edt_product = QtWidgets.QLineEdit(self._product_name)
        self.edt_product.setReadOnly(True)
        self.edt_camera = QtWidgets.QLineEdit(self._camera_role)
        self.edt_camera.setReadOnly(True)
        self.edt_model_path = QtWidgets.QLineEdit(self._model_path)
        self.edt_model_path.setReadOnly(True)
        header_form.addRow("产品", self.edt_product)
        header_form.addRow("相机", self.edt_camera)
        header_form.addRow("模型", self.edt_model_path)
        root.addWidget(header_box)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_create_tab(), "Create")
        self._mask_tab_page = self._build_mask_tab()
        self._mask_tab_index = self.tabs.addTab(self._mask_tab_page, "Template Mask")
        self.tabs.addTab(self._build_reference_tab(), "Reference ROI")
        self.tabs.addTab(self._build_find_tab(), "Find")
        root.addWidget(self.tabs, 1)

        self.lbl_status = QtWidgets.QLabel("状态：先加载参考图并设置 template ROI。")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

    def _build_create_tab(self) -> QtWidgets.QWidget:
        page = _make_tab_page()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        splitter = _make_horizontal_splitter()
        layout.addWidget(splitter, 1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        file_box = QtWidgets.QGroupBox("参考图")
        file_layout = QtWidgets.QVBoxLayout(file_box)
        self.edt_source_path = QtWidgets.QLineEdit()
        self.edt_source_path.setReadOnly(True)
        self.btn_pick_source = QtWidgets.QPushButton("打开图片...")
        file_layout.addWidget(self.edt_source_path)
        file_layout.addWidget(self.btn_pick_source)
        left_layout.addWidget(file_box)

        select_box = QtWidgets.QGroupBox("模板编辑")
        select_layout = QtWidgets.QVBoxLayout(select_box)
        form = QtWidgets.QFormLayout()
        self.edt_display_name = QtWidgets.QLineEdit()
        self.edt_template_path = QtWidgets.QLineEdit()
        self.edt_template_path.setReadOnly(True)
        self.edt_preview_path = QtWidgets.QLineEdit()
        self.edt_preview_path.setReadOnly(True)
        self.edt_roi_usage = QtWidgets.QLineEdit("template_roi")
        self.edt_roi_usage.setReadOnly(True)
        form.addRow("显示名称", self.edt_display_name)
        form.addRow("当前用途", self.edt_roi_usage)
        form.addRow("模板", self.edt_template_path)
        form.addRow("预览", self.edt_preview_path)
        select_layout.addLayout(form)
        self.chk_enable_template_mask = QtWidgets.QCheckBox("启用 Template Mask")
        self.chk_enable_template_mask.setToolTip("打开后显示 Template Mask 页签，并在保存模板和匹配时启用 mask。")
        select_layout.addWidget(self.chk_enable_template_mask)

        roi_row = QtWidgets.QHBoxLayout()
        self.spn_roi_x = QtWidgets.QSpinBox()
        self.spn_roi_y = QtWidgets.QSpinBox()
        self.spn_roi_w = QtWidgets.QSpinBox()
        self.spn_roi_h = QtWidgets.QSpinBox()
        for spin in (self.spn_roi_x, self.spn_roi_y, self.spn_roi_w, self.spn_roi_h):
            spin.setRange(0, 100000)
            spin.setAccelerated(True)
        roi_row.addWidget(QtWidgets.QLabel("X"))
        roi_row.addWidget(self.spn_roi_x)
        roi_row.addWidget(QtWidgets.QLabel("Y"))
        roi_row.addWidget(self.spn_roi_y)
        roi_row.addWidget(QtWidgets.QLabel("W"))
        roi_row.addWidget(self.spn_roi_w)
        roi_row.addWidget(QtWidgets.QLabel("H"))
        roi_row.addWidget(self.spn_roi_h)
        select_layout.addLayout(roi_row)

        self.btn_apply_template_roi = QtWidgets.QPushButton("应用当前框")
        self.btn_clear_source_roi = QtWidgets.QPushButton("清空模板ROI")
        self.btn_save_template = QtWidgets.QPushButton("保存模板")
        select_layout.addWidget(self.btn_apply_template_roi)
        select_layout.addWidget(self.btn_clear_source_roi)
        select_layout.addWidget(self.btn_save_template)
        left_layout.addWidget(select_box)

        summary_box = QtWidgets.QGroupBox("模型摘要")
        summary_layout = QtWidgets.QVBoxLayout(summary_box)
        self.txt_model_summary = QtWidgets.QPlainTextEdit()
        self.txt_model_summary.setReadOnly(True)
        self.txt_model_summary.setMinimumHeight(200)
        summary_layout.addWidget(self.txt_model_summary)
        left_layout.addWidget(summary_box, 1)

        splitter.addWidget(_make_scrollable_side_panel(left_panel, min_width=360, max_width=450))

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(6)

        source_box = QtWidgets.QGroupBox("参考图")
        source_layout = QtWidgets.QVBoxLayout(source_box)
        self.source_canvas = RoiCanvas()
        self.source_canvas.setMinimumSize(520, 320)
        source_layout.addWidget(self.source_canvas, 1)
        right_splitter.addWidget(source_box)

        preview_splitter = _make_horizontal_splitter()

        template_box = QtWidgets.QGroupBox("模板预览")
        template_layout = QtWidgets.QVBoxLayout(template_box)
        self.template_canvas = RoiCanvas()
        self.template_canvas.set_interaction_enabled(False)
        self.template_canvas.setMinimumSize(260, 160)
        template_layout.addWidget(self.template_canvas, 1)
        preview_splitter.addWidget(template_box)

        marked_box = QtWidgets.QGroupBox("模板框预览")
        marked_layout = QtWidgets.QVBoxLayout(marked_box)
        self.preview_canvas = RoiCanvas()
        self.preview_canvas.set_interaction_enabled(False)
        self.preview_canvas.setMinimumSize(260, 160)
        marked_layout.addWidget(self.preview_canvas, 1)
        preview_splitter.addWidget(marked_box)

        preview_splitter.setSizes([460, 460])
        right_splitter.addWidget(preview_splitter)
        right_splitter.setSizes([620, 260])

        splitter.addWidget(right_splitter)
        splitter.setSizes([390, 980])

        self.btn_pick_source.clicked.connect(self._pick_source_image)
        self.btn_apply_template_roi.clicked.connect(self._apply_current_template_roi)
        self.btn_clear_source_roi.clicked.connect(self._clear_source_roi)
        self.btn_save_template.clicked.connect(self._save_template)
        self.edt_display_name.editingFinished.connect(self._save_model_metadata)
        self.chk_enable_template_mask.toggled.connect(self._on_template_mask_enabled_toggled)
        self.source_canvas.shapesChanged.connect(self._sync_roi_from_canvas)
        for spin in (self.spn_roi_x, self.spn_roi_y, self.spn_roi_w, self.spn_roi_h):
            spin.valueChanged.connect(self._sync_roi_to_canvas)
        return page

    def _build_mask_tab(self) -> QtWidgets.QWidget:
        page = _make_tab_page()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        mask_box = QtWidgets.QGroupBox("Template Mask")
        mask_form = QtWidgets.QFormLayout(mask_box)
        self.edt_mask_path = QtWidgets.QLineEdit()
        self.edt_mask_path.setReadOnly(True)
        self.cmb_mask_shape = QtWidgets.QComboBox()
        self.cmb_mask_shape.addItem("rectangle", "rectangle")
        self.cmb_mask_shape.addItem("polygon", "polygon")
        self.btn_apply_template_mask = QtWidgets.QPushButton("保存当前 Mask")
        self.btn_clear_template_mask = QtWidgets.QPushButton("清空 Mask")
        self.lbl_mask_status = QtWidgets.QLabel("状态：未启用 Template Mask。")
        self.lbl_mask_status.setWordWrap(True)
        self.lbl_mask_hint = QtWidgets.QLabel(
            "说明：矩形直接拖框；多边形模式下左键逐点，右键闭合。"
            "Mask 只保留选中区域参与 NCC 匹配。"
        )
        self.lbl_mask_hint.setWordWrap(True)
        mask_form.addRow("Mask 文件", self.edt_mask_path)
        mask_form.addRow("Mask 形状", self.cmb_mask_shape)
        mask_form.addRow("", self.btn_apply_template_mask)
        mask_form.addRow("", self.btn_clear_template_mask)
        mask_form.addRow("", self.lbl_mask_status)
        mask_form.addRow("", self.lbl_mask_hint)
        left_layout.addWidget(mask_box)
        left_layout.addStretch(1)

        layout.addWidget(_make_scrollable_side_panel(left_panel, min_width=340, max_width=420))

        canvas_box = QtWidgets.QGroupBox("参考图 Mask 编辑")
        canvas_layout = QtWidgets.QVBoxLayout(canvas_box)
        self.mask_canvas = RoiCanvas()
        self.mask_canvas.setMinimumSize(520, 360)
        self.mask_canvas.set_roi_style(
            roi_color=QtGui.QColor(255, 165, 0),
            roi_dash=False,
            roi_width=2.0,
            preview_color=QtGui.QColor(255, 220, 120),
            preview_dash=True,
            preview_width=1.4,
        )
        canvas_layout.addWidget(self.mask_canvas, 1)
        layout.addWidget(canvas_box, 1)

        self.cmb_mask_shape.currentIndexChanged.connect(self._on_mask_shape_changed)
        self.btn_apply_template_mask.clicked.connect(self._save_template_mask)
        self.btn_clear_template_mask.clicked.connect(self._clear_template_mask)
        self.mask_canvas.shapesChanged.connect(self._on_mask_canvas_shape_changed)
        return page

    def _build_reference_tab(self) -> QtWidgets.QWidget:
        page = _make_tab_page()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        info_box = QtWidgets.QGroupBox("选中 ROI 属性")
        info_form = QtWidgets.QFormLayout(info_box)
        self.edit_output_label = QtWidgets.QLineEdit()
        self.edit_output_label.setPlaceholderText("先在列表中选择一个 ROI")
        self.edit_display_name = QtWidgets.QLineEdit()
        self.edit_display_name.setPlaceholderText("先在列表中选择一个 ROI")
        shape_row = QtWidgets.QHBoxLayout()
        self.btn_apply_region_name = QtWidgets.QPushButton("应用名称")
        self.cmb_reference_shape = QtWidgets.QComboBox()
        self.cmb_reference_shape.addItems(["rectangle", "polygon"])
        shape_row.addWidget(self.btn_apply_region_name)
        shape_row.addWidget(self.cmb_reference_shape, 1)
        shape_widget = QtWidgets.QWidget()
        shape_widget.setLayout(shape_row)
        info_form.addRow("ROI 标签", self.edit_output_label)
        info_form.addRow("显示名称", self.edit_display_name)
        info_form.addRow("形状 / 操作", shape_widget)
        left_layout.addWidget(info_box)

        region_box = QtWidgets.QGroupBox("Reference Regions")
        region_layout = QtWidgets.QVBoxLayout(region_box)
        self.table_reference_regions = QtWidgets.QTableWidget(0, 4)
        self.table_reference_regions.setHorizontalHeaderLabels(["#", "Name", "ROI Label", "Info"])
        self.table_reference_regions.verticalHeader().setVisible(False)
        self.table_reference_regions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_reference_regions.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_reference_regions.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_reference_regions.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_reference_regions.setMinimumHeight(180)
        self.table_reference_regions.currentCellChanged.connect(self._on_reference_region_selected)
        self.table_reference_regions.itemSelectionChanged.connect(self._on_reference_region_selection_changed)
        region_layout.addWidget(self.table_reference_regions, 1)

        button_col = QtWidgets.QVBoxLayout()
        self.btn_add_reference_roi = QtWidgets.QPushButton("新建ROI")
        self.btn_load_reference_roi = QtWidgets.QPushButton("加载已有参考ROI")
        self.btn_remove_reference_roi = QtWidgets.QPushButton("删除选中ROI")
        self.btn_clear_reference_rois = QtWidgets.QPushButton("清空全部ROI")
        self.btn_save_reference_roi = QtWidgets.QPushButton("保存当前ROI")
        for button in (
            self.btn_add_reference_roi,
            self.btn_load_reference_roi,
            self.btn_remove_reference_roi,
            self.btn_clear_reference_rois,
            self.btn_save_reference_roi,
        ):
            button_col.addWidget(button)
        button_col.addStretch(1)
        button_widget = QtWidgets.QWidget()
        button_widget.setLayout(button_col)

        region_bottom = QtWidgets.QHBoxLayout()
        region_bottom.addStretch(1)
        region_bottom.addWidget(button_widget)
        region_layout.addLayout(region_bottom)
        left_layout.addWidget(region_box, 1)

        self.lbl_reference_status = QtWidgets.QLabel("状态：这里设置的是标准片上的基准 ROI。")
        self.lbl_reference_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_reference_status)

        layout.addWidget(_make_scrollable_side_panel(left_panel, min_width=400, max_width=480))

        ref_box = QtWidgets.QGroupBox("参考区域编辑")
        ref_layout = QtWidgets.QVBoxLayout(ref_box)
        self.ref_canvas = RoiCanvas()
        self.ref_canvas.setMinimumSize(520, 360)
        self.ref_canvas.draw_shape = "rect"
        self.ref_canvas.set_outside_image_events_enabled(True)
        self.ref_canvas.set_roi_style(roi_color=QtGui.QColor(0, 140, 255), roi_dash=False, roi_width=2.0)
        ref_layout.addWidget(self.ref_canvas, 1)
        layout.addWidget(ref_box, 1)

        self.btn_apply_region_name.clicked.connect(self._apply_reference_region_fields)
        self.cmb_reference_shape.currentTextChanged.connect(self._on_reference_shape_changed)
        self.btn_add_reference_roi.clicked.connect(self._prepare_new_reference_roi)
        self.btn_remove_reference_roi.clicked.connect(self._remove_selected_reference_roi)
        self.btn_clear_reference_rois.clicked.connect(self._clear_reference_roi)
        self.btn_load_reference_roi.clicked.connect(lambda: self._load_reference_roi_from_json(silent=False))
        self.btn_save_reference_roi.clicked.connect(self._save_reference_roi_to_json)
        self.ref_canvas.imagePressed.connect(self._on_reference_canvas_pressed)
        self.ref_canvas.imageMoved.connect(self._on_reference_canvas_moved)
        self.ref_canvas.imageReleased.connect(self._on_reference_canvas_released)
        self.ref_canvas.shapesChanged.connect(self._on_reference_canvas_shape_changed)
        return page

    def _build_find_tab(self) -> QtWidgets.QWidget:
        page = _make_tab_page()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        scene_box = QtWidgets.QGroupBox("场景图")
        scene_form = QtWidgets.QFormLayout(scene_box)
        self.scene_form = scene_form
        self.edt_scene_path = QtWidgets.QLineEdit()
        self.edt_scene_path.setReadOnly(True)
        self.edt_backend = QtWidgets.QLineEdit("python-ncc")
        self.edt_backend.setReadOnly(True)
        self.edt_writeback_label = QtWidgets.QLineEdit("ncc_roi")
        self.lbl_writeback_hint = QtWidgets.QLabel()
        self.lbl_writeback_hint.setWordWrap(True)
        scene_form.addRow("路径", self.edt_scene_path)
        scene_form.addRow("后端", self.edt_backend)
        scene_form.addRow("写回标签名", self.edt_writeback_label)
        scene_form.addRow("", self.lbl_writeback_hint)
        left_layout.addWidget(scene_box)

        list_box = QtWidgets.QGroupBox("测试图片")
        list_layout = QtWidgets.QVBoxLayout(list_box)
        self.list_find_images = QtWidgets.QListWidget()
        self.list_find_images.setAlternatingRowColors(True)
        self.list_find_images.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_find_images.setMinimumHeight(130)
        self.list_find_images.itemDoubleClicked.connect(self._run_find_for_item)
        self.list_find_images.currentItemChanged.connect(self._on_find_item_selected)
        list_layout.addWidget(self.list_find_images, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        self.btn_pick_scene = QtWidgets.QPushButton("Add Images")
        self.btn_remove_find_image = QtWidgets.QPushButton("Remove")
        self.btn_clear_find_images = QtWidgets.QPushButton("Clear")
        for button in (self.btn_pick_scene, self.btn_remove_find_image, self.btn_clear_find_images):
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        list_buttons.addWidget(self.btn_pick_scene)
        list_buttons.addWidget(self.btn_remove_find_image)
        list_buttons.addWidget(self.btn_clear_find_images)
        list_layout.addLayout(list_buttons)
        left_layout.addWidget(list_box, 2)

        params_box = QtWidgets.QGroupBox("Find 参数")
        params_grid = QtWidgets.QGridLayout(params_box)
        self.spn_target_num = QtWidgets.QSpinBox()
        self.spn_target_num.setRange(1, 200)
        self.spn_score = QtWidgets.QDoubleSpinBox()
        self.spn_score.setRange(0.0, 1.0)
        self.spn_score.setDecimals(3)
        self.spn_score.setSingleStep(0.01)
        self.spn_overlap = QtWidgets.QDoubleSpinBox()
        self.spn_overlap.setRange(0.0, 1.0)
        self.spn_overlap.setDecimals(2)
        self.spn_overlap.setSingleStep(0.05)
        self.spn_min_area = QtWidgets.QSpinBox()
        self.spn_min_area.setRange(1, 100000)
        self.spn_angle_start = QtWidgets.QDoubleSpinBox()
        self.spn_angle_start.setRange(-180.0, 180.0)
        self.spn_angle_start.setDecimals(1)
        self.spn_angle_end = QtWidgets.QDoubleSpinBox()
        self.spn_angle_end.setRange(-180.0, 180.0)
        self.spn_angle_end.setDecimals(1)
        self.chk_use_simd = QtWidgets.QCheckBox("SIMD")
        self.chk_use_subpixel = QtWidgets.QCheckBox("Subpixel")
        self.chk_bitwise_not = QtWidgets.QCheckBox("Bitwise Not")
        self.chk_stop_layer1 = QtWidgets.QCheckBox("快速模式(Layer1)")
        params_grid.addWidget(QtWidgets.QLabel("Target Num"), 0, 0)
        params_grid.addWidget(self.spn_target_num, 0, 1)
        params_grid.addWidget(QtWidgets.QLabel("Score"), 0, 2)
        params_grid.addWidget(self.spn_score, 0, 3)
        params_grid.addWidget(QtWidgets.QLabel("Max Overlap"), 1, 0)
        params_grid.addWidget(self.spn_overlap, 1, 1)
        params_grid.addWidget(QtWidgets.QLabel("Min Area"), 1, 2)
        params_grid.addWidget(self.spn_min_area, 1, 3)
        params_grid.addWidget(QtWidgets.QLabel("Angle Start"), 2, 0)
        params_grid.addWidget(self.spn_angle_start, 2, 1)
        params_grid.addWidget(QtWidgets.QLabel("Angle End"), 2, 2)
        params_grid.addWidget(self.spn_angle_end, 2, 3)
        params_grid.addWidget(self.chk_use_simd, 3, 0)
        params_grid.addWidget(self.chk_use_subpixel, 3, 1)
        params_grid.addWidget(self.chk_bitwise_not, 3, 2)
        params_grid.addWidget(self.chk_stop_layer1, 3, 3)
        left_layout.addWidget(params_box)

        search_box = QtWidgets.QGroupBox("搜索ROI")
        search_layout = QtWidgets.QVBoxLayout(search_box)
        self.btn_apply_search_roi = QtWidgets.QPushButton("应用当前框为搜索ROI")
        self.btn_clear_search_roi = QtWidgets.QPushButton("清空搜索ROI")
        self.lbl_search_roi = QtWidgets.QLabel("状态：未设置搜索ROI，默认全图搜索。")
        self.lbl_search_roi.setWordWrap(True)
        search_layout.addWidget(self.btn_apply_search_roi)
        search_layout.addWidget(self.btn_clear_search_roi)
        search_layout.addWidget(self.lbl_search_roi)
        left_layout.addWidget(search_box)

        run_box = QtWidgets.QGroupBox("执行")
        run_layout = QtWidgets.QVBoxLayout(run_box)
        self.btn_run_match = QtWidgets.QPushButton("Run Selected")
        self.btn_run_all = QtWidgets.QPushButton("Run All")
        self.btn_writeback = QtWidgets.QPushButton("写回 Top1 ROI")
        self.btn_writeback_regions = QtWidgets.QPushButton("写回投影参考ROI")
        run_layout.addWidget(self.btn_run_match)
        run_layout.addWidget(self.btn_run_all)
        run_layout.addWidget(self.btn_writeback)
        run_layout.addWidget(self.btn_writeback_regions)
        left_layout.addWidget(run_box)

        result_box = QtWidgets.QGroupBox("结果")
        result_layout = QtWidgets.QVBoxLayout(result_box)
        self.txt_find_summary = QtWidgets.QPlainTextEdit()
        self.txt_find_summary.setReadOnly(True)
        self.txt_find_summary.setMinimumHeight(90)
        self.txt_find_summary.setMaximumHeight(140)
        result_layout.addWidget(self.txt_find_summary)
        left_layout.addWidget(result_box, 0)

        layout.addWidget(_make_scrollable_side_panel(left_panel, min_width=340, max_width=450))

        canvas_box = QtWidgets.QGroupBox("场景图预览")
        canvas_layout = QtWidgets.QVBoxLayout(canvas_box)
        self.find_canvas = RoiCanvas()
        self.find_canvas.setMinimumSize(520, 360)
        self.find_canvas.set_roi_style(roi_color=QtGui.QColor(0, 140, 255), roi_dash=False, roi_width=1.0)
        canvas_layout.addWidget(self.find_canvas, 1)
        layout.addWidget(canvas_box, 1)

        self.btn_pick_scene.clicked.connect(self._pick_scene_image)
        self.btn_remove_find_image.clicked.connect(self._remove_selected_find_images)
        self.btn_clear_find_images.clicked.connect(self._clear_find_images)
        self.btn_apply_search_roi.clicked.connect(self._apply_find_search_roi)
        self.btn_clear_search_roi.clicked.connect(self._clear_find_search_roi)
        self.btn_run_match.clicked.connect(self._run_match)
        self.btn_run_all.clicked.connect(self._run_all_find)
        self.btn_writeback.clicked.connect(self._writeback_top1)
        self.btn_writeback_regions.clicked.connect(self._writeback_reference_regions)
        self.find_canvas.shapesChanged.connect(self._refresh_search_roi_status)
        for spin in (
            self.spn_target_num,
            self.spn_score,
            self.spn_overlap,
            self.spn_min_area,
            self.spn_angle_start,
            self.spn_angle_end,
        ):
            spin.valueChanged.connect(self._schedule_find_options_save)
        for checkbox in (
            self.chk_use_simd,
            self.chk_use_subpixel,
            self.chk_bitwise_not,
            self.chk_stop_layer1,
        ):
            checkbox.toggled.connect(self._schedule_find_options_save)
        return page

    def _finalize_ui(self) -> None:
        root_layout = self.layout()
        if isinstance(root_layout, QtWidgets.QVBoxLayout) and root_layout.count() >= 3:
            header_widget = root_layout.itemAt(0).widget()
            if isinstance(header_widget, QtWidgets.QWidget):
                header_widget.hide()

        self.edt_writeback_label.setPlaceholderText("例如：ncc_roi")
        self.edt_writeback_label.setToolTip("写入到当前场景图对应的 LabelMe JSON 里的标签名。")
        self.btn_writeback.setText("写回当前 Top1 外框")
        self.btn_writeback.setToolTip("按上面的标签名，把当前图片的 Top1 匹配外框写回 LabelMe JSON。")
        self.btn_writeback_regions.setToolTip("把当前 Top1 匹配下投影得到的 roi1/roi2/... 参考区域批量写回 LabelMe JSON。")
        self.chk_stop_layer1.setToolTip("只做到金字塔第1层就停止细化，通常更快，但角度和位置精度可能略降。")
        if hasattr(self, "lbl_writeback_hint") and isinstance(self.lbl_writeback_hint, QtWidgets.QLabel):
            self.lbl_writeback_hint.setText(
                "说明：上面填写的是写回到 LabelMe 的标签名；下面按钮会把当前 Top1 匹配结果按这个标签写回。"
            )
        if hasattr(self, "scene_form") and isinstance(self.scene_form, QtWidgets.QFormLayout):
            if hasattr(self.scene_form, "setRowVisible"):
                self.scene_form.setRowVisible(self.edt_backend, False)
                self.scene_form.setRowVisible(self.lbl_writeback_hint, False)
            else:
                label = self.scene_form.labelForField(self.edt_backend)
                if label is not None:
                    label.hide()
                self.edt_backend.hide()
                self.lbl_writeback_hint.hide()

    def _load_model(self) -> None:
        self._loading_model = True
        try:
            self._model = load_model(self._model_path).normalized()
            self._reference_regions = [region.normalized() for region in list(self._model.reference_regions or [])]
            self._selected_reference_idx = None
            self._selected_reference_indices = set()
            ensure_default_assets(self._model_path, self._model)
            self.edt_display_name.setText(self._model.display_name)
            self.edt_source_path.setText(source_image_path(self._model_path, self._model))
            self.edt_template_path.setText(template_image_path(self._model_path, self._model))
            self.edt_preview_path.setText(preview_image_path(self._model_path, self._model))
            if hasattr(self, "edt_mask_path"):
                self.edt_mask_path.setText(mask_image_path(self._model_path, self._model))
            if hasattr(self, "chk_enable_template_mask"):
                self.chk_enable_template_mask.setChecked(bool(getattr(self._model, "template_mask_enabled", False)))
            self._set_roi_spin_values(self._model.template_roi.to_xywh())
            self._apply_options_to_form(self._model.options)
            self._reload_authoring_canvases(force_reference=True)
            self._refresh_reference_region_list()
            self._refresh_reference_region_fields()
            self._refresh_reference_canvas()
            self._refresh_model_summary()
            self._refresh_search_roi_status()
            self._refresh_template_mask_visibility()
        finally:
            self._loading_model = False

    def _load_initial_image(self) -> None:
        if not self._initial_image_path or not Path(self._initial_image_path).exists():
            return
        if not Path(self.edt_source_path.text()).exists():
            self.source_canvas.set_image(self._initial_image_path)
            self._refresh_reference_image()
        if not self.edt_scene_path.text():
            self._set_scene_path(self._initial_image_path)

    def _reference_image_path(self) -> str:
        model_source = self.edt_source_path.text().strip()
        if model_source and Path(model_source).exists():
            return model_source
        current_source = str(self.source_canvas.image_path() or "").strip()
        if current_source and Path(current_source).exists():
            return current_source
        if self._initial_image_path and Path(self._initial_image_path).exists():
            return self._initial_image_path
        return ""

    def _reload_authoring_canvases(self, *, force_reference: bool = False) -> None:
        source_path = self.edt_source_path.text().strip()
        template_path = self.edt_template_path.text().strip()
        preview_path = self.edt_preview_path.text().strip()

        if source_path and Path(source_path).exists():
            self.source_canvas.set_image(source_path)
            self.source_canvas.set_roi_rect(self._model.template_roi.to_xywh(), emit_signal=False)
            self._refresh_source_canvas_overlays()
        elif self.source_canvas.image_path() is None:
            self.source_canvas.clear_image()

        if template_path and Path(template_path).exists():
            self.template_canvas.set_image(template_path)
        else:
            self.template_canvas.clear_image()

        if preview_path and Path(preview_path).exists():
            self.preview_canvas.set_image(preview_path)
        else:
            self.preview_canvas.clear_image()

        self._refresh_mask_canvas()
        self._refresh_reference_image(force=force_reference)

    def _refresh_reference_image(self, *, force: bool = False) -> None:
        path = self._reference_image_path()
        current_path = str(self.ref_canvas.image_path() or "").strip()
        if path and Path(path).exists():
            if force or current_path != path:
                self.ref_canvas.set_image(path)
                self.ref_canvas.set_roi_style(roi_color=QtGui.QColor(0, 140, 255), roi_dash=False, roi_width=2.0)
            self._refresh_reference_canvas()
            return
        self.ref_canvas.clear_image()

    def _template_mask_enabled(self) -> bool:
        if hasattr(self, "chk_enable_template_mask"):
            return bool(self.chk_enable_template_mask.isChecked())
        return bool(getattr(self._model, "template_mask_enabled", False))

    def _set_mask_tab_visible(self, visible: bool) -> None:
        if not hasattr(self, "tabs"):
            return
        if (not visible) and self.tabs.currentWidget() is getattr(self, "_mask_tab_page", None):
            self.tabs.setCurrentIndex(0)
        if hasattr(self.tabs, "setTabVisible"):
            self.tabs.setTabVisible(int(self._mask_tab_index), bool(visible))
            return
        current_index = self.tabs.indexOf(self._mask_tab_page)
        if visible and current_index < 0:
            self.tabs.insertTab(int(self._mask_tab_index), self._mask_tab_page, "Template Mask")
            return
        if (not visible) and current_index >= 0:
            if self.tabs.currentWidget() is self._mask_tab_page:
                self.tabs.setCurrentIndex(0)
            self.tabs.removeTab(current_index)

    def _refresh_template_mask_visibility(self) -> None:
        enabled = self._template_mask_enabled()
        self._set_mask_tab_visible(enabled)
        self._refresh_source_canvas_overlays()
        if enabled:
            self._refresh_mask_canvas()
        elif hasattr(self, "mask_canvas"):
            self.mask_canvas.clear_roi(emit_signal=False)
            self.mask_canvas.set_overlays([])

    def _on_template_mask_enabled_toggled(self, checked: bool) -> None:
        self._model.template_mask_enabled = bool(checked)
        self._refresh_template_mask_visibility()
        if getattr(self, "_loading_model", False):
            return
        save_model(self._model_path, self._model)
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)
        if checked:
            self._set_status("已启用 Template Mask。")
        else:
            self._set_status("已关闭 Template Mask。")

    def _mask_shape_name(self) -> str:
        if not self._template_mask_enabled() or not hasattr(self, "cmb_mask_shape"):
            return "disabled"
        value = str(self.cmb_mask_shape.currentData() or "").strip().lower()
        if value in {"rectangle", "polygon"}:
            return value
        return "rectangle"

    def _mask_overlay_shape(self, region: NccReferenceRegion | None) -> Optional[OverlayShape]:
        if not isinstance(region, NccReferenceRegion):
            return None
        points = _region_polygon_points(region)
        if len(points) < 3:
            return None
        return OverlayShape(
            shape_type="polygon",
            points=points,
            color=QtGui.QColor(255, 165, 0),
            width=1.8,
            dash=False,
        )

    def _refresh_source_canvas_overlays(self) -> None:
        if not hasattr(self, "source_canvas") or not self.source_canvas.has_image():
            return
        overlays: List[OverlayShape] = []
        overlay = None
        if self._template_mask_enabled():
            overlay = self._mask_overlay_shape(getattr(self._model, "template_mask", None))
        if overlay is not None:
            overlays.append(overlay)
        self.source_canvas.set_overlays(overlays)

    def _apply_mask_edit_mode(self) -> None:
        if not hasattr(self, "mask_canvas"):
            return
        shape_name = self._mask_shape_name()
        enabled = shape_name != "disabled"
        self.mask_canvas.set_interaction_enabled(enabled)
        self.mask_canvas.draw_shape = "polygon" if shape_name == "polygon" else "rect"

    def _template_mask_from_canvas(self) -> NccReferenceRegion | None:
        if not hasattr(self, "mask_canvas") or not self.mask_canvas.has_image():
            return self._model.template_mask
        shape_name = self._mask_shape_name()
        if shape_name == "disabled":
            return None
        if shape_name == "polygon" and self.mask_canvas.roi.points and len(self.mask_canvas.roi.points) >= 3:
            return NccReferenceRegion(
                label_name="template_mask",
                display_name="template_mask",
                shape_type="polygon",
                points=[(float(x), float(y)) for x, y in self.mask_canvas.roi.points],
            ).normalized()
        xywh = self.mask_canvas.roi_xywh()
        if xywh is None:
            return None
        x, y, w, h = [int(v) for v in xywh]
        return NccReferenceRegion(
            label_name="template_mask",
            display_name="template_mask",
            shape_type="rectangle",
            points=[(float(x), float(y)), (float(x + w), float(y + h))],
        ).normalized()

    def _refresh_mask_canvas(self) -> None:
        if not hasattr(self, "mask_canvas"):
            return
        if not self._template_mask_enabled():
            self.mask_canvas.clear_roi(emit_signal=False)
            self.mask_canvas.set_overlays([])
            self.lbl_mask_status.setText("状态：Template Mask 已关闭。")
            self._apply_mask_edit_mode()
            return
        source_path = self._reference_image_path()
        if source_path and Path(source_path).exists():
            current_path = str(self.mask_canvas.image_path() or "").strip()
            if current_path != source_path:
                self.mask_canvas.set_image(source_path)
                self.mask_canvas.set_roi_style(
                    roi_color=QtGui.QColor(255, 165, 0),
                    roi_dash=False,
                    roi_width=2.0,
                    preview_color=QtGui.QColor(255, 220, 120),
                    preview_dash=True,
                    preview_width=1.4,
                )
        else:
            self.mask_canvas.clear_image()
            self.lbl_mask_status.setText("状态：请先加载参考图。")
            return

        overlay_items: List[OverlayShape] = []
        roi = self._model.template_roi.normalized()
        if roi.width > 0 and roi.height > 0:
            overlay_items.append(
                OverlayShape(
                    shape_type="rect",
                    xywh=roi.to_xywh(),
                    color=QtGui.QColor(0, 255, 0),
                    width=1.4,
                    dash=True,
                )
            )
        self.mask_canvas.set_overlays(overlay_items)

        region = getattr(self._model, "template_mask", None)
        self._syncing_mask_view = True
        try:
            if isinstance(region, NccReferenceRegion):
                shape_name = "polygon" if region.shape_type == "polygon" else "rectangle"
                index = self.cmb_mask_shape.findData(shape_name)
                if index >= 0:
                    self.cmb_mask_shape.setCurrentIndex(index)
                if region.shape_type == "polygon" and len(region.points) >= 3:
                    self.mask_canvas.set_roi_polygon([(float(x), float(y)) for x, y in region.points], emit_signal=False)
                elif len(region.points) >= 2:
                    (x0, y0), (x1, y1) = region.points[:2]
                    self.mask_canvas.set_roi_rect(
                        (
                            int(round(min(float(x0), float(x1)))),
                            int(round(min(float(y0), float(y1)))),
                            max(1, int(round(abs(float(x1) - float(x0))))),
                            max(1, int(round(abs(float(y1) - float(y0))))),
                        ),
                        emit_signal=False,
                    )
                self.lbl_mask_status.setText("状态：当前 Template Mask 已加载。")
            else:
                index = self.cmb_mask_shape.findData("disabled")
                if index >= 0:
                    self.cmb_mask_shape.setCurrentIndex(index)
                self.mask_canvas.clear_roi(emit_signal=False)
                self.lbl_mask_status.setText("状态：未启用 Template Mask。")
        finally:
            self._syncing_mask_view = False
        self._apply_mask_edit_mode()

    def _apply_options_to_form(self, options: NccMatchOptions) -> None:
        normalized = options.normalized()
        self.spn_target_num.setValue(normalized.target_num)
        self.spn_score.setValue(normalized.score_threshold)
        self.spn_overlap.setValue(normalized.max_overlap)
        self.spn_min_area.setValue(normalized.min_reduced_area)
        angle_range = normalized.angle_search.ranges[0] if normalized.angle_search.ranges else NccAngleRange(-180.0, 180.0)
        self.spn_angle_start.setValue(float(angle_range.start))
        self.spn_angle_end.setValue(float(angle_range.end))
        self.chk_use_simd.setChecked(normalized.use_simd)
        self.chk_use_subpixel.setChecked(normalized.use_subpixel)
        self.chk_bitwise_not.setChecked(normalized.bitwise_not)
        self.chk_stop_layer1.setChecked(normalized.stop_layer1)

    def _current_find_options(self) -> NccMatchOptions:
        return NccMatchOptions(
            target_num=self.spn_target_num.value(),
            max_overlap=self.spn_overlap.value(),
            score_threshold=self.spn_score.value(),
            angle_search=NccAngleSearch(
                mode="ranges",
                tolerance_angle=0.0,
                ranges=[
                    NccAngleRange(
                        start=self.spn_angle_start.value(),
                        end=self.spn_angle_end.value(),
                    )
                ],
            ),
            min_reduced_area=self.spn_min_area.value(),
            use_simd=self.chk_use_simd.isChecked(),
            use_subpixel=self.chk_use_subpixel.isChecked(),
            bitwise_not=self.chk_bitwise_not.isChecked(),
            stop_layer1=self.chk_stop_layer1.isChecked(),
        ).normalized()

    def _refresh_model_summary(self) -> None:
        self.txt_model_summary.setPlainText(model_summary(self._model))

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(f"状态：{text}")

    def _set_reference_status(self, text: str) -> None:
        self.lbl_reference_status.setText(f"状态：{text}")

    def _sync_model_from_ui(self) -> None:
        self._model.display_name = self.edt_display_name.text().strip() or self._model.display_name
        self._model.template_mask_enabled = self._template_mask_enabled()
        self._model.template_roi = NccMatchRect(
            x=self.spn_roi_x.value(),
            y=self.spn_roi_y.value(),
            width=max(1, self.spn_roi_w.value()),
            height=max(1, self.spn_roi_h.value()),
        ).normalized()
        self._model.template_mask = self._template_mask_from_canvas()
        self._model.options = self._current_find_options()
        self._model.reference_regions = [region.normalized() for region in self._reference_regions]

    def _sync_model_without_template_roi_from_ui(self) -> None:
        self._model.display_name = self.edt_display_name.text().strip() or self._model.display_name
        self._model.template_mask_enabled = self._template_mask_enabled()
        self._model.template_mask = self._template_mask_from_canvas()
        self._model.options = self._current_find_options()
        self._model.reference_regions = [region.normalized() for region in self._reference_regions]

    def _save_find_options_to_model(self, *_args) -> None:
        if getattr(self, "_loading_model", False):
            return
        self._sync_model_without_template_roi_from_ui()
        save_model(self._model_path, self._model)
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)

    def _schedule_find_options_save(self, *_args) -> None:
        if getattr(self, "_loading_model", False):
            return
        self._sync_model_without_template_roi_from_ui()
        self._find_options_save_timer.start()

    def _flush_find_options_save(self) -> None:
        timer = getattr(self, "_find_options_save_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._save_find_options_to_model()

    def _stored_search_roi_xywh(self) -> Optional[Tuple[int, int, int, int]]:
        if self._model.search_roi is None:
            return None
        return self._model.search_roi.normalized().to_xywh()

    def _update_search_roi(self, xywh: Optional[Tuple[int, int, int, int]], *, persist: bool) -> None:
        self._model.search_roi = None if xywh is None else NccMatchRect(*[int(v) for v in xywh]).normalized()
        if persist:
            self._sync_model_without_template_roi_from_ui()
            save_model(self._model_path, self._model)
            self._refresh_model_summary()
            self.modelSaved.emit(self._model_path)

    def _save_reference_regions_to_model(self) -> None:
        self._sync_model_without_template_roi_from_ui()
        save_model(self._model_path, self._model)
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)

    def _set_roi_spin_values(self, xywh: Tuple[int, int, int, int]) -> None:
        self._syncing_roi = True
        try:
            self.spn_roi_x.setValue(int(xywh[0]))
            self.spn_roi_y.setValue(int(xywh[1]))
            self.spn_roi_w.setValue(int(xywh[2]))
            self.spn_roi_h.setValue(int(xywh[3]))
        finally:
            self._syncing_roi = False

    def _apply_current_template_roi(self) -> None:
        roi = self.source_canvas.roi_xywh()
        if roi is None:
            QtWidgets.QMessageBox.information(self, "NCC", "请先在右侧参考图上框出 template ROI。")
            return
        self._save_template()

    def _sync_roi_from_canvas(self) -> None:
        if self._syncing_roi:
            return
        roi = self.source_canvas.roi_xywh()
        if roi is None:
            if getattr(self, "_loading_model", False):
                return
            roi = (0, 0, 1, 1)
            self._set_roi_spin_values(roi)
            return
        self._set_roi_spin_values(roi)
        self._auto_apply_template_roi_from_canvas()

    def _auto_apply_template_roi_from_canvas(self) -> None:
        if getattr(self, "_loading_model", False) or getattr(self, "_suppress_source_roi_auto_apply", False):
            return
        if self.source_canvas.roi_xywh() is None:
            return
        self._save_template()

    def _sync_roi_to_canvas(self) -> None:
        if self._syncing_roi:
            return
        self.source_canvas.set_roi_rect(
            (
                self.spn_roi_x.value(),
                self.spn_roi_y.value(),
                max(1, self.spn_roi_w.value()),
                max(1, self.spn_roi_h.value()),
            ),
            emit_signal=False,
        )

    def _clear_source_roi(self) -> None:
        self.source_canvas.clear_roi(emit_signal=False)
        self._set_roi_spin_values((0, 0, 1, 1))
        self._set_status("已清空模板 ROI。")

    def _pick_source_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择参考图",
            self.source_canvas.image_path() or self._initial_image_path or self._product_dir,
            _image_file_filter(),
        )
        if not path:
            return
        self.source_canvas.set_image(path)
        self._refresh_reference_image(force=True)
        self._set_status(f"已加载参考图：{path}")

    def _save_model_metadata(self) -> None:
        self._sync_model_without_template_roi_from_ui()
        save_model(self._model_path, self._model)
        self._refresh_model_summary()

    def _save_template(self) -> None:
        source_path = str(self.source_canvas.image_path() or "").strip()
        if not source_path:
            QtWidgets.QMessageBox.warning(self, "NCC", "请先加载参考图。")
            return
        roi = self.source_canvas.roi_xywh()
        if roi is None:
            QtWidgets.QMessageBox.warning(self, "NCC", "请先在右侧参考图上框选模板 ROI。")
            return
        self._sync_model_from_ui()
        set_source_from_image_file(self._model_path, self._model, source_path)
        self._model = set_template_from_roi(self._model_path, self._model, roi)
        self._model.reference_regions = [region.normalized() for region in self._reference_regions]
        save_model(self._model_path, self._model)
        self._load_model()
        self._set_status("模板已保存。")
        self.modelSaved.emit(self._model_path)

    def _reference_region_row_values(self, index: int, region: NccReferenceRegion) -> List[str]:
        label = region.label_name or f"roi{index + 1}"
        display_name = region.display_name or label
        return [str(index), display_name, label, _region_info_text(region)]

    def _refresh_reference_region_list(self) -> None:
        table = self.table_reference_regions
        self._syncing_reference_table = True
        blocker = QtCore.QSignalBlocker(table)
        try:
            table.setRowCount(len(self._reference_regions))
            for index, region in enumerate(self._reference_regions):
                values = self._reference_region_row_values(index, region)
                for column, value in enumerate(values):
                    item = table.item(index, column)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        if column == 0:
                            item.setTextAlignment(
                                QtCore.Qt.AlignmentFlag.AlignHCenter
                                | QtCore.Qt.AlignmentFlag.AlignVCenter
                            )
                        table.setItem(index, column, item)
                    item.setText(value)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            selected_rows = {
                index
                for index in getattr(self, "_selected_reference_indices", set())
                if 0 <= index < len(self._reference_regions)
            }
            self._selected_reference_indices = selected_rows
            if selected_rows and self._selected_reference_idx not in selected_rows:
                self._selected_reference_idx = min(selected_rows) if selected_rows else None
            if not selected_rows:
                self._selected_reference_idx = None
            table.clearSelection()
            selection_model = table.selectionModel()
            if selection_model is not None:
                model = table.model()
                for row in sorted(selected_rows):
                    selection = QtCore.QItemSelection(
                        model.index(row, 0),
                        model.index(row, max(0, table.columnCount() - 1)),
                    )
                    selection_model.select(
                        selection,
                        QtCore.QItemSelectionModel.SelectionFlag.Select
                        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                    )
            if self._selected_reference_idx is not None and 0 <= self._selected_reference_idx < len(self._reference_regions):
                table.setCurrentCell(self._selected_reference_idx, 0)
            else:
                table.clearSelection()
                table.setCurrentIndex(QtCore.QModelIndex())
        finally:
            del blocker
            self._syncing_reference_table = False

    def _refresh_reference_region_fields(self) -> None:
        selected = self._selected_reference_idx
        has_selection = selected is not None and 0 <= selected < len(self._reference_regions)
        self.edit_output_label.setEnabled(has_selection)
        self.edit_display_name.setEnabled(has_selection)
        self.btn_apply_region_name.setEnabled(has_selection)
        if not has_selection:
            self.edit_output_label.setText("")
            self.edit_display_name.setText("")
            return
        region = self._reference_regions[selected]
        self.edit_output_label.setText(region.label_name)
        self.edit_display_name.setText(region.display_name or region.label_name)

    def _reference_region_overlay_shapes(self) -> List[OverlayShape]:
        overlays: List[OverlayShape] = []
        selected_indices = getattr(self, "_selected_reference_indices", set())
        for index, region in enumerate(self._reference_regions):
            points = _region_polygon_points(region)
            if len(points) < 3:
                continue
            selected = index in selected_indices
            overlays.append(
                OverlayShape(
                    shape_type="polygon",
                    points=points,
                    color=QtGui.QColor(0, 140, 255) if selected else QtGui.QColor(255, 0, 255),
                    width=2.4 if selected else 1.2,
                    dash=not selected,
                )
            )
        return overlays

    def _refresh_reference_canvas(self) -> None:
        if not self.ref_canvas.has_image():
            return
        self.ref_canvas.set_overlays(self._reference_region_overlay_shapes())
        self._syncing_reference_view = True
        try:
            if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
                self.ref_canvas.clear_roi(emit_signal=False)
                self.cmb_reference_shape.setCurrentText("rectangle")
                return
            region = self._reference_regions[self._selected_reference_idx]
            self.cmb_reference_shape.setCurrentText("polygon" if region.shape_type == "polygon" else "rectangle")
            if region.shape_type == "polygon" and len(region.points) >= 3:
                self.ref_canvas.set_roi_polygon([(float(x), float(y)) for x, y in region.points], emit_signal=False)
                return
            if len(region.points) >= 2:
                (x0, y0), (x1, y1) = region.points[:2]
                self.ref_canvas.set_roi_rect(
                    (
                        int(round(min(float(x0), float(x1)))),
                        int(round(min(float(y0), float(y1)))),
                        max(1, int(round(abs(float(x1) - float(x0))))),
                        max(1, int(round(abs(float(y1) - float(y0))))),
                    ),
                    emit_signal=False,
                )
                return
            self.ref_canvas.clear_roi(emit_signal=False)
        finally:
            self._syncing_reference_view = False

    def _reference_region_at_point(self, x: float, y: float) -> Optional[int]:
        for index in range(len(self._reference_regions) - 1, -1, -1):
            points = _region_polygon_points(self._reference_regions[index])
            if _point_hits_polygon(points, x, y):
                return index
        return None

    def _set_reference_selection(self, indices: Sequence[int], *, primary: Optional[int] = None) -> None:
        valid = {
            int(index)
            for index in indices
            if 0 <= int(index) < len(self._reference_regions)
        }
        if primary is not None and primary not in valid:
            primary = None
        self._selected_reference_indices = valid
        if primary is not None:
            self._selected_reference_idx = int(primary)
        elif valid:
            self._selected_reference_idx = min(valid)
        else:
            self._selected_reference_idx = None
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._refresh_reference_canvas()

    def _begin_reference_region_move(self, x: float, y: float) -> None:
        selected = {
            index
            for index in getattr(self, "_selected_reference_indices", set())
            if 0 <= index < len(self._reference_regions)
        }
        if not selected:
            return
        self._moving_reference_regions = True
        self._reference_move_start = (float(x), float(y))
        self._reference_move_original = {
            index: [(float(px), float(py)) for px, py in self._reference_regions[index].points]
            for index in selected
        }
        self.ref_canvas.set_interaction_enabled(False)
        self.ref_canvas.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def _translate_reference_move_selection(self, dx: float, dy: float) -> None:
        for index, original_points in self._reference_move_original.items():
            if not (0 <= index < len(self._reference_regions)):
                continue
            self._reference_regions[index].points = [
                (float(x) + dx, float(y) + dy)
                for x, y in original_points
            ]
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._refresh_reference_canvas()

    def _on_reference_canvas_pressed(self, button: int, x: int, y: int) -> None:
        if button != int(QtCore.Qt.MouseButton.LeftButton.value):
            return
        selected = self._reference_region_at_point(float(x), float(y))
        if selected is None:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if not (
                modifiers
                & (
                    QtCore.Qt.KeyboardModifier.ControlModifier
                    | QtCore.Qt.KeyboardModifier.ShiftModifier
                )
            ):
                self._set_reference_selection([])
            return
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        additive = bool(
            modifiers
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.ShiftModifier
            )
        )
        selected_indices = set(getattr(self, "_selected_reference_indices", set()))
        if additive:
            selected_indices.add(selected)
        else:
            selected_indices = {selected}
        self._set_reference_selection(selected_indices, primary=selected)
        self._begin_reference_region_move(float(x), float(y))

    def _on_reference_canvas_moved(self, buttons: int, x: int, y: int) -> None:
        if not getattr(self, "_moving_reference_regions", False):
            return
        if not (int(buttons) & int(QtCore.Qt.MouseButton.LeftButton.value)):
            return
        if self._reference_move_start is None:
            return
        x0, y0 = self._reference_move_start
        self._translate_reference_move_selection(float(x) - x0, float(y) - y0)

    def _on_reference_canvas_released(self, button: int, _x: int, _y: int) -> None:
        if button != int(QtCore.Qt.MouseButton.LeftButton.value):
            return
        if not getattr(self, "_moving_reference_regions", False):
            return
        self._moving_reference_regions = False
        self._reference_move_start = None
        self._reference_move_original = {}
        self.ref_canvas.set_interaction_enabled(True)
        self.ref_canvas.unsetCursor()
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._refresh_reference_canvas()
        self._save_reference_regions_to_model()
        count = len(getattr(self, "_selected_reference_indices", set()))
        self._set_reference_status(f"已平移 {count} 个参考 ROI。")

    def _next_reference_label(self) -> str:
        existing = {
            str(region.label_name or "").strip().lower()
            for region in self._reference_regions
            if str(region.label_name or "").strip()
        }
        index = 1
        while True:
            label = f"roi{index}"
            if label.lower() not in existing:
                return label
            index += 1

    def _on_reference_region_selected(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if getattr(self, "_syncing_reference_table", False):
            return
        if current_row < 0 or current_row >= len(self._reference_regions):
            self._set_reference_selection([])
        else:
            selected_rows = self._selected_rows_from_reference_table() or {current_row}
            self._set_reference_selection(selected_rows, primary=current_row)

    def _selected_rows_from_reference_table(self) -> Set[int]:
        rows: Set[int] = set()
        selection_model = self.table_reference_regions.selectionModel()
        if selection_model is None:
            return rows
        for index in selection_model.selectedRows():
            row = int(index.row())
            if 0 <= row < len(self._reference_regions):
                rows.add(row)
        return rows

    def _on_reference_region_selection_changed(self) -> None:
        if getattr(self, "_syncing_reference_table", False):
            return
        rows = self._selected_rows_from_reference_table()
        current_row = int(self.table_reference_regions.currentRow())
        primary = current_row if current_row in rows else (min(rows) if rows else None)
        self._set_reference_selection(rows, primary=primary)

    def _prepare_new_reference_roi(self) -> None:
        self._selected_reference_idx = None
        self._selected_reference_indices = set()
        blocker = QtCore.QSignalBlocker(self.table_reference_regions)
        try:
            self.table_reference_regions.clearSelection()
            self.table_reference_regions.setCurrentIndex(QtCore.QModelIndex())
        finally:
            del blocker
        self._refresh_reference_region_fields()
        self._refresh_reference_canvas()
        self._set_reference_status("已切换到新增 ROI 模式，请直接在右侧画布上继续画框。")

    def _remove_selected_reference_roi(self) -> None:
        selected = {
            index
            for index in getattr(self, "_selected_reference_indices", set())
            if 0 <= index < len(self._reference_regions)
        }
        if not selected and self._selected_reference_idx is not None and 0 <= self._selected_reference_idx < len(self._reference_regions):
            selected = {self._selected_reference_idx}
        if not selected:
            return
        for index in sorted(selected, reverse=True):
            del self._reference_regions[index]
        self._selected_reference_idx = None
        self._selected_reference_indices = set()
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._refresh_reference_canvas()
        self._save_reference_regions_to_model()
        self._set_reference_status("已删除选中的参考 ROI。")

    def _clear_reference_roi(self) -> None:
        self._reference_regions = []
        self._selected_reference_idx = None
        self._selected_reference_indices = set()
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._refresh_reference_canvas()
        self._save_reference_regions_to_model()
        self._set_reference_status("已清空全部参考 ROI。")

    def _apply_reference_region_fields(self) -> None:
        if self._selected_reference_idx is None or not (0 <= self._selected_reference_idx < len(self._reference_regions)):
            return
        region = self._reference_regions[self._selected_reference_idx]
        label = self.edit_output_label.text().strip() or region.label_name or self._next_reference_label()
        display_name = self.edit_display_name.text().strip() or label
        region.label_name = label
        region.display_name = display_name
        self._refresh_reference_region_list()
        self._refresh_reference_region_fields()
        self._save_reference_regions_to_model()
        self._set_reference_status(f"已更新参考 ROI 名称：{display_name}")

    def _on_reference_shape_changed(self, shape_name: str) -> None:
        self.ref_canvas.draw_shape = "polygon" if shape_name == "polygon" else "rect"

    def _region_points_from_canvas(self) -> Tuple[str, List[Tuple[float, float]]]:
        if self.ref_canvas.roi.shape_type == "polygon" and self.ref_canvas.roi.points and len(self.ref_canvas.roi.points) >= 3:
            return "polygon", [(float(x), float(y)) for x, y in self.ref_canvas.roi.points]
        xywh = self.ref_canvas.roi_xywh()
        if xywh is None:
            return "", []
        x, y, w, h = [int(v) for v in xywh]
        return "rectangle", [(float(x), float(y)), (float(x + w), float(y + h))]

    def _on_reference_canvas_shape_changed(self) -> None:
        if self._syncing_reference_view:
            return
        shape_type, points = self._region_points_from_canvas()
        if not shape_type or not points:
            return
        if self._selected_reference_idx is None:
            label = self._next_reference_label()
            region = NccReferenceRegion(
                label_name=label,
                display_name=label,
                shape_type=shape_type,
                points=points,
            ).normalized()
            self._reference_regions.append(region)
            self._selected_reference_idx = len(self._reference_regions) - 1
            self._selected_reference_indices = {self._selected_reference_idx}
            self._refresh_reference_region_list()
            self._refresh_reference_region_fields()
            self._refresh_reference_canvas()
            self._save_reference_regions_to_model()
            self._set_reference_status(f"已新增参考 ROI：{label}")
            return

        region = self._reference_regions[self._selected_reference_idx]
        region.shape_type = shape_type
        region.points = points
        self._refresh_reference_region_list()
        self._refresh_reference_canvas()
        self._save_reference_regions_to_model()
        self._set_reference_status(f"已更新参考 ROI：{region.display_name or region.label_name}")

    def _load_reference_roi_from_json(self, *, silent: bool) -> None:
        image_path = self._reference_image_path()
        if not image_path or not Path(image_path).exists():
            if not silent:
                QtWidgets.QMessageBox.warning(self, "NCC", "请先加载参考图。")
            return
        jpath = Path(labelme_json_of_image(image_path))
        if not jpath.exists():
            if not silent:
                QtWidgets.QMessageBox.information(self, "NCC", "当前参考图还没有 labelme json。")
            return
        try:
            regions: List[NccReferenceRegion] = []
            for shape in list_shapes_from_labelme(str(jpath), label_prefix="roi"):
                label_name = str(shape.get("label", "")).strip()
                if not label_name:
                    continue
                shape_type = str(shape.get("shape_type", "rectangle"))
                if shape_type == "polygon":
                    points = [
                        (float(item[0]), float(item[1]))
                        for item in list(shape.get("points", []) or [])
                        if isinstance(item, Sequence) and len(item) >= 2
                    ]
                    if len(points) < 3:
                        continue
                else:
                    xywh = _shape_to_rect(shape)
                    if xywh is None:
                        continue
                    x, y, w, h = xywh
                    points = [(float(x), float(y)), (float(x + w), float(y + h))]
                    shape_type = "rectangle"
                regions.append(
                    NccReferenceRegion(
                        label_name=label_name,
                        display_name=label_name,
                        shape_type=shape_type,
                        points=points,
                    ).normalized()
                )
            if not regions:
                raise RuntimeError("json 中没有 roi1/roi2/... 参考 ROI。")
            self._reference_regions = regions
            self._selected_reference_idx = None
            self._selected_reference_indices = set()
            self._refresh_reference_region_list()
            self._refresh_reference_region_fields()
            self._refresh_reference_canvas()
            self._save_reference_regions_to_model()
            self._set_reference_status(f"已加载 {len(regions)} 个参考 ROI。")
        except Exception as exc:
            if not silent:
                QtWidgets.QMessageBox.warning(self, "NCC", str(exc))

    def _save_reference_roi_to_json(self) -> None:
        image_path = self._reference_image_path()
        if not image_path or not Path(image_path).exists():
            QtWidgets.QMessageBox.warning(self, "NCC", "请先加载参考图。")
            return
        if not self._reference_regions:
            QtWidgets.QMessageBox.warning(self, "NCC", "当前没有可保存的参考 ROI。")
            return
        for region in self._reference_regions:
            label_name = str(region.label_name or "").strip()
            if not label_name:
                continue
            if region.shape_type == "polygon":
                points = [(float(x), float(y)) for x, y in region.points]
                if len(points) >= 3:
                    upsert_labelme_polygon(image_path, points, label_name=label_name)
                continue
            if len(region.points) < 2:
                continue
            (x0, y0), (x1, y1) = region.points[:2]
            upsert_labelme_rect(
                image_path,
                (
                    int(round(min(float(x0), float(x1)))),
                    int(round(min(float(y0), float(y1)))),
                    max(1, int(round(abs(float(x1) - float(x0))))),
                    max(1, int(round(abs(float(y1) - float(y0))))),
                ),
                label_name=label_name,
            )
        self._set_reference_status(f"参考 ROI 已保存，共 {len(self._reference_regions)} 个。")

    def _add_find_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择测试图片",
            self.edt_scene_path.text() or self._initial_image_path or self._product_dir,
            _image_file_filter(),
        )
        existing = {
            str(self.list_find_images.item(index).data(QtCore.Qt.ItemDataRole.UserRole) or "")
            for index in range(self.list_find_images.count())
        }
        first_added: Optional[QtWidgets.QListWidgetItem] = None
        for path in paths:
            scene_path = str(path or "").strip()
            if not scene_path or scene_path in existing:
                continue
            item = QtWidgets.QListWidgetItem(Path(scene_path).name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, scene_path)
            item.setToolTip(scene_path)
            self.list_find_images.addItem(item)
            if first_added is None:
                first_added = item
        if first_added is not None:
            self.list_find_images.setCurrentItem(first_added)

    def _pick_scene_image(self) -> None:
        self._add_find_images()

    def _remove_selected_find_images(self) -> None:
        selected_items = list(self.list_find_images.selectedItems())
        for item in selected_items:
            scene_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            if scene_path:
                self._find_result_cache.pop(scene_path, None)
            row = self.list_find_images.row(item)
            self.list_find_images.takeItem(row)
        if self.list_find_images.count() <= 0:
            self._latest_response = None
            self.edt_scene_path.clear()
            self.txt_find_summary.clear()
            self.find_canvas.clear_image()
            self.find_canvas.set_overlays([])

    def _clear_find_images(self) -> None:
        self._find_result_cache.clear()
        self.list_find_images.clear()
        self._latest_response = None
        self.edt_scene_path.clear()
        self.txt_find_summary.clear()
        self.find_canvas.clear_image()
        self.find_canvas.set_overlays([])

    def _set_scene_path(self, path: str) -> None:
        scene_path = str(path or "").strip()
        self.edt_scene_path.setText(scene_path)
        self.find_canvas.set_overlays([])
        if scene_path and Path(scene_path).exists():
            self.find_canvas.set_image(scene_path)
        else:
            self.find_canvas.clear_image()
        self.find_canvas.set_roi_style(roi_color=QtGui.QColor(0, 140, 255), roi_dash=False, roi_width=1.0)
        self._apply_saved_search_roi_to_find_canvas()

    def _show_find_scene(
        self,
        scene_path: str,
        *,
        response: Optional[NccMatchResponse] = None,
        summary_text: str = "",
    ) -> None:
        self._set_scene_path(scene_path)
        if response is not None:
            self.find_canvas.set_overlays(self._match_overlays(response))
            self.edt_backend.setText(response.backend_name)
        else:
            self.edt_backend.clear()
        if not summary_text:
            self.txt_find_summary.clear()
        if summary_text:
            self.txt_find_summary.setPlainText(summary_text)

    def _on_find_item_selected(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        if current is None:
            return
        scene_path = str(current.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not scene_path:
            return
        cached = self._find_result_cache.get(scene_path) or {}
        response = cached.get("response")
        summary_text = str(cached.get("summary_text") or "")
        if isinstance(response, NccMatchResponse):
            self._latest_response = response
            self._show_find_scene(scene_path, response=response, summary_text=summary_text)
            return
        self._latest_response = None
        self._show_find_scene(scene_path, summary_text=summary_text)

    def _apply_saved_search_roi_to_find_canvas(self) -> None:
        if not self.find_canvas.has_image():
            self._refresh_search_roi_status()
            return
        search_roi = self._stored_search_roi_xywh()
        if search_roi is None:
            self.find_canvas.clear_roi()
        else:
            self.find_canvas.set_roi_rect(search_roi)
        self._refresh_search_roi_status()

    def _refresh_search_roi_status(self) -> None:
        current_roi = self.find_canvas.roi_xywh() if self.find_canvas.has_image() else None
        stored_roi = self._stored_search_roi_xywh()
        if current_roi is not None and not _same_xywh(current_roi, stored_roi):
            x, y, w, h = current_roi
            self.lbl_search_roi.setText(
                f"状态：当前框选 ROI=({x},{y},{w},{h})。本次执行会使用该区域；点“应用当前框为搜索ROI”后会写入模型。"
            )
            return
        if stored_roi is not None:
            x, y, w, h = stored_roi
            self.lbl_search_roi.setText(f"状态：搜索ROI=({x},{y},{w},{h})，Find 默认限制在此区域搜索。")
            return
        self.lbl_search_roi.setText("状态：未设置搜索ROI，默认全图搜索。")

    def _apply_find_search_roi(self) -> None:
        xywh = self.find_canvas.roi_xywh()
        if xywh is None:
            QtWidgets.QMessageBox.information(self, "NCC", "请先在右侧图片上拖一个矩形搜索区域。")
            return
        self._update_search_roi(xywh, persist=True)
        self._refresh_search_roi_status()
        self._set_status("已保存搜索 ROI。")

    def _clear_find_search_roi(self) -> None:
        self.find_canvas.clear_roi()
        self._update_search_roi(None, persist=True)
        self._refresh_search_roi_status()
        self._set_status("已清空搜索 ROI。")

    def _effective_search_roi(self) -> Optional[Tuple[int, int, int, int]]:
        current_roi = self.find_canvas.roi_xywh() if self.find_canvas.has_image() else None
        return current_roi or self._stored_search_roi_xywh()

    def _template_roi_quad(self) -> Optional[np.ndarray]:
        rect = self._model.template_roi.normalized()
        if rect.width <= 0 or rect.height <= 0:
            return None
        return np.asarray(
            [
                [float(rect.x), float(rect.y)],
                [float(rect.x + rect.width), float(rect.y)],
                [float(rect.x + rect.width), float(rect.y + rect.height)],
                [float(rect.x), float(rect.y + rect.height)],
            ],
            dtype=np.float32,
        )

    def _projected_reference_regions(
        self,
        match: NccMatchResult,
    ) -> List[Tuple[str, List[Tuple[float, float]]]]:
        if not self._reference_regions:
            return []
        src_quad = self._template_roi_quad()
        dst_quad = np.asarray(match.quad, dtype=np.float32)
        if src_quad is None or dst_quad.shape != (4, 2):
            return []
        matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
        projected: List[Tuple[str, List[Tuple[float, float]]]] = []
        for index, region in enumerate(self._reference_regions, start=1):
            points = _region_polygon_points(region)
            if len(points) < 3:
                continue
            src = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
            dst = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
            label_name = str(region.label_name or "").strip() or f"roi{index}"
            projected.append((label_name, [(float(x), float(y)) for x, y in dst]))
        return projected

    def _reference_region_match_overlays(self, match: NccMatchResult) -> List[OverlayShape]:
        overlays: List[OverlayShape] = []
        for _label_name, points in self._projected_reference_regions(match):
            overlays.append(
                OverlayShape(
                    shape_type="polygon",
                    points=points,
                    color=QtGui.QColor(255, 0, 255),
                    width=1.4,
                    dash=False,
                )
            )
        return overlays

    def _match_overlays(self, response: NccMatchResponse) -> List[OverlayShape]:
        palette = [
            QtGui.QColor(0, 220, 0),
            QtGui.QColor(255, 165, 0),
            QtGui.QColor(0, 180, 255),
            QtGui.QColor(255, 60, 60),
        ]
        overlays: List[OverlayShape] = []
        for index, item in enumerate(response.matches):
            overlays.append(
                OverlayShape(
                    shape_type="polygon",
                    points=[(float(x), float(y)) for x, y in item.quad],
                    color=palette[index % len(palette)],
                    width=2.0,
                    dash=False,
                )
            )
            if index == 0:
                overlays.extend(self._reference_region_match_overlays(item))
        return overlays

    def _summary_text_for_response(self, response: NccMatchResponse) -> str:
        lines = [
            f"backend={response.backend_name}",
            f"elapsed_ms={response.elapsed_ms:.2f}",
            f"matches={len(response.matches)}",
            f"reference_roi_count={len(self._reference_regions)}",
        ]
        for index, item in enumerate(response.matches, start=1):
            lines.append(
                f"#{index}: score={item.score:.4f}, angle={item.angle:.2f}, "
                f"bbox=({item.bbox.x:.1f}, {item.bbox.y:.1f}, {item.bbox.width:.1f}, {item.bbox.height:.1f})"
            )
        return "\n".join(lines)

    def _run_match_for_scene_path(self, scene_path: str) -> NccMatchResponse:
        scene = imread(scene_path, cv2.IMREAD_COLOR)
        if scene is None:
            raise RuntimeError(f"无法读取场景图：{scene_path}")
        self._flush_find_options_save()
        compiled: Optional[NccCompiledModel] = None
        try:
            compiled = NccCompiledModel(self._model_path, self._model)
            return compiled.match(
                scene,
                options=self._current_find_options(),
                search_roi=self._effective_search_roi(),
            )
        finally:
            if compiled is not None:
                try:
                    compiled.close()
                except Exception:
                    pass

    def _set_find_item_success(self, item: QtWidgets.QListWidgetItem, response: NccMatchResponse) -> None:
        scene_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        basename = Path(scene_path).name
        if response.matches:
            top1 = response.matches[0]
            summary = (
                f"score={top1.score:.3f} "
                f"angle={top1.angle:.1f} "
                f"time={response.elapsed_ms:.1f}ms"
            )
        else:
            summary = f"matches=0 time={response.elapsed_ms:.1f}ms"
        item.setText(f"{basename} | {summary}")
        item.setToolTip(f"{scene_path}\n{summary}")
        item.setForeground(QtGui.QBrush(QtGui.QColor(20, 160, 20)))

    def _set_find_item_error(self, item: QtWidgets.QListWidgetItem, message: str) -> None:
        scene_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        basename = Path(scene_path).name
        item.setText(f"{basename} | ERROR: {message}")
        item.setToolTip(f"{scene_path}\nERROR: {message}")
        item.setForeground(QtGui.QBrush(QtGui.QColor(200, 40, 40)))

    def _find_item_for_scene_path(self, scene_path: str) -> Optional[QtWidgets.QListWidgetItem]:
        target = str(scene_path or "").strip()
        for index in range(self.list_find_images.count()):
            item = self.list_find_images.item(index)
            if item is None:
                continue
            if str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip() == target:
                return item
        return None

    def _set_find_running(self, running: bool) -> None:
        self._find_running = bool(running)
        for button in (
            self.btn_pick_scene,
            self.btn_remove_find_image,
            self.btn_clear_find_images,
            self.btn_run_match,
            self.btn_run_all,
            self.btn_writeback,
            self.btn_writeback_regions,
        ):
            button.setEnabled(not running)

    def _running_find_summary_text(self) -> str:
        elapsed_ms = float(self._find_elapsed_timer.elapsed()) if self._find_elapsed_timer.isValid() else 0.0
        count = len(getattr(self, "_find_active_paths", []) or [])
        current_path = ""
        current_item = self.list_find_images.currentItem()
        if current_item is not None:
            current_path = str(current_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not current_path and count == 1:
            current_path = self._find_active_paths[0]
        lines = [
            "backend=native-ncc",
            "status=running",
            f"elapsed_ms={elapsed_ms:.0f}",
            f"queued={count}",
        ]
        if current_path:
            lines.append(f"image={Path(current_path).name}")
        return "\n".join(lines)

    def _update_find_progress_elapsed(self) -> None:
        if not self._find_running:
            return
        self.txt_find_summary.setPlainText(self._running_find_summary_text())

    def _start_find_worker(self, scene_paths: Sequence[str]) -> None:
        if self._find_running:
            self._set_status("NCC find is already running")
            return
        paths = [str(path or "").strip() for path in scene_paths if str(path or "").strip()]
        if not paths:
            return
        self._flush_find_options_save()
        self._latest_response = None
        self._find_active_paths = list(paths)
        self._find_elapsed_timer.restart()
        self._set_find_running(True)
        self._update_find_progress_elapsed()
        self._find_progress_timer.start()
        for scene_path in paths:
            item = self._find_item_for_scene_path(scene_path)
            if item is None:
                continue
            item.setText(f"{Path(scene_path).name} | RUNNING")
            item.setForeground(QtGui.QBrush(QtGui.QColor(220, 180, 60)))
        self._set_status(f"NCC find running: {len(paths)} image(s)")

        thread = QtCore.QThread(self)
        worker = _NccFindWorker(
            model_path=self._model_path,
            model=self._model,
            scene_paths=paths,
            options=self._current_find_options(),
            search_roi=self._effective_search_roi(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._set_status)
        worker.itemFinished.connect(self._on_find_worker_item_finished)
        worker.finished.connect(self._on_find_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._find_thread = thread
        self._find_worker = worker
        thread.start()

    @QtCore.Slot(int, str, object, str)
    def _on_find_worker_item_finished(
        self,
        _index: int,
        scene_path: str,
        response_obj: object,
        error_message: str,
    ) -> None:
        item = self._find_item_for_scene_path(scene_path)
        if error_message or not isinstance(response_obj, NccMatchResponse):
            message = str(error_message or "NCC find failed")
            if item is not None:
                self._set_find_item_error(item, message)
            self._find_result_cache[scene_path] = {
                "response": None,
                "summary_text": f"ERROR: {message}",
            }
            if self.list_find_images.currentItem() is item:
                self._latest_response = None
                self._show_find_scene(scene_path, summary_text=f"ERROR: {message}")
            self._set_status(f"NCC find failed: {message}")
            return

        response = response_obj
        summary_text = self._summary_text_for_response(response)
        if item is not None:
            self._set_find_item_success(item, response)
        self._find_result_cache[scene_path] = {
            "response": response,
            "summary_text": summary_text,
        }
        if self.list_find_images.currentItem() is item:
            self._latest_response = response
            self._show_find_scene(scene_path, response=response, summary_text=summary_text)
        self._set_status(f"{Path(scene_path).name} NCC find done")

    @QtCore.Slot()
    def _on_find_worker_finished(self) -> None:
        self._find_thread = None
        self._find_worker = None
        self._find_progress_timer.stop()
        self._find_active_paths = []
        self._set_find_running(False)
        self._set_status("NCC find finished")

    def _run_find_for_item(self, item: QtWidgets.QListWidgetItem) -> None:
        scene_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not scene_path:
            return
        try:
            response = self._run_match_for_scene_path(scene_path)
        except Exception as exc:
            message = str(exc)
            self._set_find_item_error(item, message)
            self._find_result_cache[scene_path] = {
                "response": None,
                "summary_text": f"ERROR: {message}",
            }
            if self.list_find_images.currentItem() is item:
                self._show_find_scene(scene_path, summary_text=f"ERROR: {message}")
            self._set_status(f"测试失败：{message}")
            return

        self._latest_response = response
        summary_text = self._summary_text_for_response(response)
        self._set_find_item_success(item, response)
        self._find_result_cache[scene_path] = {
            "response": response,
            "summary_text": summary_text,
        }
        if self.list_find_images.currentItem() is item:
            self._show_find_scene(scene_path, response=response, summary_text=summary_text)
        self._set_status(f"{Path(scene_path).name} 测试完成。")

    def _run_match(self) -> None:
        current_item = self.list_find_images.currentItem()
        if current_item is not None:
            self._run_find_for_item(current_item)
            return

        scene_path = self.edt_scene_path.text().strip()
        if not scene_path:
            QtWidgets.QMessageBox.warning(self, "NCC", "请先加载场景图。")
            return
        try:
            response = self._run_match_for_scene_path(scene_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "NCC", str(exc))
            self._set_status("匹配失败。")
            return

        self._latest_response = response
        summary_text = self._summary_text_for_response(response)
        self._show_find_scene(scene_path, response=response, summary_text=summary_text)
        self._set_status(f"匹配完成，命中 {len(response.matches)} 个结果。")

    def _run_all_find(self) -> None:
        if self.list_find_images.count() <= 0:
            QtWidgets.QMessageBox.information(self, "NCC", "请先添加测试图片。")
            return
        for index in range(self.list_find_images.count()):
            item = self.list_find_images.item(index)
            if item is None:
                continue
            self.list_find_images.setCurrentItem(item)
            self._run_find_for_item(item)
            QtWidgets.QApplication.processEvents()

    def _run_find_for_item(self, item: QtWidgets.QListWidgetItem) -> None:
        scene_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not scene_path:
            return
        self._start_find_worker([scene_path])

    def _run_match(self) -> None:
        current_item = self.list_find_images.currentItem()
        if current_item is not None:
            self._run_find_for_item(current_item)
            return

        scene_path = self.edt_scene_path.text().strip()
        if not scene_path:
            QtWidgets.QMessageBox.warning(self, "NCC", "Please load a scene image first.")
            return
        self._start_find_worker([scene_path])

    def _run_all_find(self) -> None:
        if self.list_find_images.count() <= 0:
            QtWidgets.QMessageBox.information(self, "NCC", "Please add test images first.")
            return
        paths: List[str] = []
        for index in range(self.list_find_images.count()):
            item = self.list_find_images.item(index)
            if item is None:
                continue
            scene_path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
            if scene_path:
                paths.append(scene_path)
        self._start_find_worker(paths)

    def _writeback_top1(self) -> None:
        scene_path = self.edt_scene_path.text().strip()
        if not scene_path or self._latest_response is None or not self._latest_response.matches:
            QtWidgets.QMessageBox.warning(self, "NCC", "当前没有可写回的匹配结果。")
            return
        label_name = self.edt_writeback_label.text().strip() or "ncc_roi"
        top1 = self._latest_response.matches[0]
        upsert_labelme_polygon(scene_path, list(top1.quad), label_name=label_name)
        self._set_status(f"已写回 LabelMe 标签：{label_name}")

    def _writeback_reference_regions(self) -> None:
        scene_path = self.edt_scene_path.text().strip()
        if not scene_path or self._latest_response is None or not self._latest_response.matches:
            QtWidgets.QMessageBox.warning(self, "NCC", "当前没有可写回的匹配结果。")
            return
        if not self._reference_regions:
            QtWidgets.QMessageBox.information(self, "NCC", "当前还没有配置 Reference ROI。")
            return
        projected = self._projected_reference_regions(self._latest_response.matches[0])
        if not projected:
            QtWidgets.QMessageBox.warning(self, "NCC", "当前没有可写回的投影参考 ROI。")
            return

        written_labels: List[str] = []
        for label_name, points in projected:
            upsert_labelme_polygon(scene_path, points, label_name=label_name)
            written_labels.append(label_name)

        preview = ", ".join(written_labels[:6])
        suffix = " ..." if len(written_labels) > 6 else ""
        self._set_status(f"已写回 {len(written_labels)} 个参考ROI：{preview}{suffix}")

    def _sync_roi_to_canvas(self) -> None:
        if self._syncing_roi:
            return
        self.source_canvas.set_roi_rect(
            (
                self.spn_roi_x.value(),
                self.spn_roi_y.value(),
                max(1, self.spn_roi_w.value()),
                max(1, self.spn_roi_h.value()),
            ),
            emit_signal=False,
        )
        self._refresh_mask_canvas()

    def _clear_source_roi(self) -> None:
        self.source_canvas.clear_roi(emit_signal=False)
        self._set_roi_spin_values((0, 0, 1, 1))
        self._refresh_mask_canvas()
        self._set_status("已清空模板 ROI。")

    def _pick_source_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择参考图",
            self.source_canvas.image_path() or self._initial_image_path or self._product_dir,
            _image_file_filter(),
        )
        if not path:
            return
        self.source_canvas.set_image(path)
        self._refresh_source_canvas_overlays()
        self._refresh_mask_canvas()
        self._refresh_reference_image(force=True)
        self._set_status(f"已加载参考图：{path}")

    def _on_mask_shape_changed(self, _index: int) -> None:
        self._apply_mask_edit_mode()
        if getattr(self, "_syncing_mask_view", False):
            return
        if self._mask_shape_name() == "disabled":
            self.mask_canvas.clear_roi(emit_signal=False)
            self._model.template_mask = None
            self._refresh_source_canvas_overlays()
            self.lbl_mask_status.setText("状态：未启用 Template Mask。")
        else:
            self.lbl_mask_status.setText("状态：请在右侧参考图上绘制 Template Mask。")

    def _on_mask_canvas_shape_changed(self) -> None:
        if getattr(self, "_syncing_mask_view", False) or self._mask_shape_name() == "disabled":
            return
        region = self._template_mask_from_canvas()
        if region is None:
            self.lbl_mask_status.setText("状态：请先绘制一个有效的 Mask。")
            return
        self._model.template_mask = region
        self._refresh_source_canvas_overlays()
        self.lbl_mask_status.setText("状态：Mask 已更新，点击“保存当前 Mask”或直接“保存模板”即可生效。")

    def _save_template_mask(self) -> None:
        source_path = str(self.mask_canvas.image_path() or "").strip()
        if not source_path:
            QtWidgets.QMessageBox.warning(self, "NCC", "请先加载参考图。")
            return
        self._sync_model_without_template_roi_from_ui()
        save_model(self._model_path, self._model)
        self._refresh_source_canvas_overlays()
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)
        if self._model.template_mask is None:
            self.lbl_mask_status.setText("状态：Template Mask 已关闭。")
            self._set_status("已关闭 Template Mask。")
        else:
            self.lbl_mask_status.setText("状态：Template Mask 已保存到模型，保存模板后会生成 mask 图。")
            self._set_status("Template Mask 已保存，保存模板后生效。")

    def _clear_template_mask(self) -> None:
        self.mask_canvas.clear_roi(emit_signal=False)
        index = self.cmb_mask_shape.findData("disabled")
        if index >= 0:
            self.cmb_mask_shape.setCurrentIndex(index)
        self._model.template_mask = None
        save_model(self._model_path, self._model)
        self._refresh_source_canvas_overlays()
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)
        self.lbl_mask_status.setText("状态：Template Mask 已清空。")
        self._set_status("已清空 Template Mask。")

    def _refresh_source_canvas_overlays(self) -> None:
        if not hasattr(self, "source_canvas") or not self.source_canvas.has_image():
            return
        overlays: List[OverlayShape] = []
        if self._template_mask_enabled():
            overlay = self._mask_overlay_shape(getattr(self._model, "template_mask", None))
            if overlay is not None:
                overlays.append(overlay)
        self.source_canvas.set_overlays(overlays)

    def _apply_mask_edit_mode(self) -> None:
        if not hasattr(self, "mask_canvas"):
            return
        enabled = self._template_mask_enabled()
        self.mask_canvas.set_interaction_enabled(enabled)
        shape_name = self._mask_shape_name()
        self.mask_canvas.draw_shape = "polygon" if shape_name == "polygon" else "rect"

    def _template_mask_from_canvas(self) -> NccReferenceRegion | None:
        if not self._template_mask_enabled():
            return getattr(self._model, "template_mask", None)
        if not hasattr(self, "mask_canvas") or not self.mask_canvas.has_image():
            return getattr(self._model, "template_mask", None)
        shape_name = self._mask_shape_name()
        if shape_name == "polygon" and self.mask_canvas.roi.points and len(self.mask_canvas.roi.points) >= 3:
            return NccReferenceRegion(
                label_name="template_mask",
                display_name="template_mask",
                shape_type="polygon",
                points=[(float(x), float(y)) for x, y in self.mask_canvas.roi.points],
            ).normalized()
        xywh = self.mask_canvas.roi_xywh()
        if xywh is None:
            return None
        x, y, w, h = [int(v) for v in xywh]
        return NccReferenceRegion(
            label_name="template_mask",
            display_name="template_mask",
            shape_type="rectangle",
            points=[(float(x), float(y)), (float(x + w), float(y + h))],
        ).normalized()

    def _refresh_mask_canvas(self) -> None:
        if not hasattr(self, "mask_canvas"):
            return
        if not self._template_mask_enabled():
            self.mask_canvas.clear_roi(emit_signal=False)
            self.mask_canvas.set_overlays([])
            self.lbl_mask_status.setText("状态：Template Mask 已关闭。")
            self._apply_mask_edit_mode()
            return
        source_path = self._reference_image_path()
        if source_path and Path(source_path).exists():
            current_path = str(self.mask_canvas.image_path() or "").strip()
            if current_path != source_path:
                self.mask_canvas.set_image(source_path)
                self.mask_canvas.set_roi_style(
                    roi_color=QtGui.QColor(255, 165, 0),
                    roi_dash=False,
                    roi_width=2.0,
                    preview_color=QtGui.QColor(255, 220, 120),
                    preview_dash=True,
                    preview_width=1.4,
                )
        else:
            self.mask_canvas.clear_image()
            self.lbl_mask_status.setText("状态：请先加载参考图。")
            return

        overlay_items: List[OverlayShape] = []
        roi = self._model.template_roi.normalized()
        if roi.width > 0 and roi.height > 0:
            overlay_items.append(
                OverlayShape(
                    shape_type="rect",
                    xywh=roi.to_xywh(),
                    color=QtGui.QColor(0, 255, 0),
                    width=1.4,
                    dash=True,
                )
            )
        self.mask_canvas.set_overlays(overlay_items)

        region = getattr(self._model, "template_mask", None)
        self._syncing_mask_view = True
        try:
            if isinstance(region, NccReferenceRegion):
                shape_name = "polygon" if region.shape_type == "polygon" else "rectangle"
                index = self.cmb_mask_shape.findData(shape_name)
                if index >= 0:
                    self.cmb_mask_shape.setCurrentIndex(index)
                if region.shape_type == "polygon" and len(region.points) >= 3:
                    self.mask_canvas.set_roi_polygon([(float(x), float(y)) for x, y in region.points], emit_signal=False)
                elif len(region.points) >= 2:
                    (x0, y0), (x1, y1) = region.points[:2]
                    self.mask_canvas.set_roi_rect(
                        (
                            int(round(min(float(x0), float(x1)))),
                            int(round(min(float(y0), float(y1)))),
                            max(1, int(round(abs(float(x1) - float(x0))))),
                            max(1, int(round(abs(float(y1) - float(y0))))),
                        ),
                        emit_signal=False,
                    )
                self.lbl_mask_status.setText("状态：当前 Template Mask 已加载。")
            else:
                if self.cmb_mask_shape.count() > 0:
                    self.cmb_mask_shape.setCurrentIndex(0)
                self.mask_canvas.clear_roi(emit_signal=False)
                self.lbl_mask_status.setText("状态：请在右侧参考图上绘制 Template Mask。")
        finally:
            self._syncing_mask_view = False
        self._apply_mask_edit_mode()

    def _on_mask_shape_changed(self, _index: int) -> None:
        self._apply_mask_edit_mode()
        if getattr(self, "_syncing_mask_view", False):
            return
        if not self._template_mask_enabled():
            self.lbl_mask_status.setText("状态：Template Mask 已关闭。")
            return
        self.lbl_mask_status.setText("状态：请在右侧参考图上绘制 Template Mask。")

    def _on_mask_canvas_shape_changed(self) -> None:
        if getattr(self, "_syncing_mask_view", False) or not self._template_mask_enabled():
            return
        region = self._template_mask_from_canvas()
        if region is None:
            self.lbl_mask_status.setText("状态：请先绘制一个有效的 Mask。")
            return
        self._model.template_mask = region
        self._refresh_source_canvas_overlays()
        self.lbl_mask_status.setText("状态：Mask 已更新，点击“保存当前 Mask”或直接“保存模板”即可生效。")

    def _save_template_mask(self) -> None:
        if not self._template_mask_enabled():
            QtWidgets.QMessageBox.information(self, "NCC", "请先打开“启用 Template Mask”。")
            return
        source_path = str(self.mask_canvas.image_path() or "").strip()
        if not source_path:
            QtWidgets.QMessageBox.warning(self, "NCC", "请先加载参考图。")
            return
        self._sync_model_without_template_roi_from_ui()
        save_model(self._model_path, self._model)
        self._refresh_source_canvas_overlays()
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)
        self.lbl_mask_status.setText("状态：Template Mask 已保存到模型，保存模板后会生成 mask 图。")
        self._set_status("Template Mask 已保存，保存模板后生效。")

    def _clear_template_mask(self) -> None:
        self.mask_canvas.clear_roi(emit_signal=False)
        self._model.template_mask = None
        save_model(self._model_path, self._model)
        self._refresh_source_canvas_overlays()
        self._refresh_model_summary()
        self.modelSaved.emit(self._model_path)
        self.lbl_mask_status.setText("状态：Template Mask 已清空。")
        self._set_status("已清空 Template Mask。")


__all__ = ["NccMatchWorkbenchDialog"]
