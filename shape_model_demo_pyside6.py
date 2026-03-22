from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.debug import OverlayShape, RoiCanvas
from shape_model_like import ScaledShapeModel


Point = Tuple[float, float]


@dataclass
class StoredShape:
    shape_type: str
    xywh: Optional[Tuple[int, int, int, int]] = None
    points: Optional[List[Point]] = None


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _rect_xywh_to_points(xywh: Tuple[int, int, int, int]) -> List[Point]:
    x, y, w, h = xywh
    return [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]


def _shape_to_points(shape: StoredShape) -> Optional[List[Point]]:
    if shape.shape_type == "rect" and shape.xywh is not None:
        return _rect_xywh_to_points(shape.xywh)
    if shape.shape_type == "polygon" and shape.points:
        return list(shape.points)
    return None


def _mask_from_points(h: int, w: int, points: List[Point], fill_value: int = 255) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    if not points:
        return mask
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], int(fill_value))
    return mask


def _transform_points(
    points: List[Point],
    origin_rc: Tuple[float, float],
    row: float,
    col: float,
    angle: float,
    scale: float,
) -> List[Point]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    orow, ocol = origin_rc
    out: List[Point] = []
    for x, y in points:
        dcol = float(x) - float(ocol)
        drow = float(y) - float(orow)
        col_f = float(col) + scale * (cos_a * dcol - sin_a * drow)
        row_f = float(row) + scale * (sin_a * dcol + cos_a * drow)
        out.append((col_f, row_f))
    return out


class FindWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)

    def __init__(
        self,
        *,
        model: ScaledShapeModel,
        test_path: str,
        anchor_pts: List[Point],
        roi_pts: List[Point],
        search_region_xywh: Optional[Tuple[int, int, int, int]],
        search_resize: float,
        params: Dict[str, float],
        force_best: bool,
        num_levels: int,
    ) -> None:
        super().__init__()
        self._model = model
        self._test_path = test_path
        self._anchor_pts = anchor_pts
        self._roi_pts = roi_pts
        self._search_region_xywh = search_region_xywh
        self._search_resize = float(search_resize)
        self._params = params
        self._force_best = force_best
        self._num_levels = num_levels

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if not self._anchor_pts or not self._roi_pts:
                raise RuntimeError("Missing anchor/roi points")
            t0 = time.perf_counter()
            self.progress.emit("Loading test image ...")
            tgt = _read_gray(self._test_path)
            row_offset = 0
            col_offset = 0
            if self._search_region_xywh is not None:
                self.progress.emit("Applying search region ...")
                x, y, w, h = self._search_region_xywh
                H, W = tgt.shape[:2]
                x = max(0, min(int(x), W - 1))
                y = max(0, min(int(y), H - 1))
                w = max(1, min(int(w), W - x))
                h = max(1, min(int(h), H - y))
                tgt = tgt[y : y + h, x : x + w]
                row_offset = y
                col_offset = x
            resize = float(self._search_resize)
            if resize <= 0.0:
                resize = 1.0
            if resize != 1.0:
                self.progress.emit(f"Resizing search image ({resize:.2f}x) ...")
                h, w = tgt.shape[:2]
                new_w = max(1, int(round(w * resize)))
                new_h = max(1, int(round(h * resize)))
                interp = cv2.INTER_AREA if resize < 1.0 else cv2.INTER_LINEAR
                tgt = cv2.resize(tgt, (new_w, new_h), interpolation=interp)

            params = dict(self._params)
            params["max_dist"] = params["max_dist"] * resize
            # Matching is executed on resized search image.
            # Scale search range must follow the same resize factor.
            params["scale_min"] = max(1e-4, params["scale_min"] * resize)
            params["scale_max"] = max(params["scale_min"], params["scale_max"] * resize)
            if "scale_step" in params:
                params["scale_step"] = max(1e-4, params["scale_step"] * resize)
            self.progress.emit("Finding candidates ...")
            rows, cols, angs, scs, scores = self._model.find(tgt, num_levels=self._num_levels, **params)
            below_threshold = False
            if rows.size == 0 and self._force_best:
                self.progress.emit("No match above threshold, trying best candidate ...")
                params = dict(self._params)
                params["min_score"] = 0.0
                params["max_overlap"] = 1.0
                params["max_dist"] = params["max_dist"] * resize
                params["scale_min"] = max(1e-4, params["scale_min"] * resize)
                params["scale_max"] = max(params["scale_min"], params["scale_max"] * resize)
                if "scale_step" in params:
                    params["scale_step"] = max(1e-4, params["scale_step"] * resize)
                rows, cols, angs, scs, scores = self._model.find(tgt, num_levels=self._num_levels, **params)
                below_threshold = rows.size > 0
            elapsed = time.perf_counter() - t0

            if rows.size == 0:
                self.finished.emit({"matched": False, "elapsed": elapsed})
                return

            inv_resize = 1.0 / resize
            row = float(rows[0]) * inv_resize + float(row_offset)
            col = float(cols[0]) * inv_resize + float(col_offset)
            angle = float(angs[0])
            scale = float(scs[0]) * inv_resize
            score = float(scores[0])
            tgt_roi_pts = _transform_points(self._roi_pts, self._model.origin_rc, row, col, angle, scale)
            tgt_anchor_pts = _transform_points(self._anchor_pts, self._model.origin_rc, row, col, angle, scale)
            self.finished.emit(
                {
                    "matched": True,
                    "below_threshold": below_threshold,
                    "row": row,
                    "col": col,
                    "angle": angle,
                    "scale": scale,
                    "score": score,
                    "tgt_roi_pts": tgt_roi_pts,
                    "tgt_anchor_pts": tgt_anchor_pts,
                    "elapsed": elapsed,
                }
            )
        except Exception as e:
            self.failed.emit(str(e))


class ShapeModelDemo(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Shape Model Demo")

        self.ref_path: Optional[str] = None
        self.test_path: Optional[str] = None
        self.model: Optional[ScaledShapeModel] = None
        self.ref_shapes: Dict[str, StoredShape] = {}
        self._find_thread: Optional[QtCore.QThread] = None
        self._find_worker: Optional[FindWorker] = None
        self._find_start_ts: Optional[float] = None
        self._find_timer: Optional[QtCore.QTimer] = None
        self._model_points: Optional[List[Point]] = None
        self.search_region_xywh: Optional[Tuple[int, int, int, int]] = None
        self._last_anchor_pts: Optional[List[Point]] = None
        self._last_anchor_color = QtGui.QColor(0, 128, 255)

        self._build_ui()

    def _build_ui(self) -> None:
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QHBoxLayout(cw)

        left_wrap = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_wrap)
        left.setContentsMargins(8, 8, 8, 8)
        left.setSpacing(8)
        left.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_wrap)
        left_scroll.setMinimumWidth(360)
        root.addWidget(left_scroll, 0)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        ref_box = QtWidgets.QGroupBox("Reference")
        ref_l = QtWidgets.QVBoxLayout(ref_box)
        self.btn_load_ref = QtWidgets.QPushButton("Load Reference")
        self.btn_load_ref.clicked.connect(self._load_reference)
        self.lbl_ref = QtWidgets.QLabel("No reference loaded")
        self.cmb_label = QtWidgets.QComboBox()
        self.cmb_label.addItems(["anchor", "roi", "anchor_mask"])
        self.cmb_label.currentTextChanged.connect(self._on_label_changed)
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(["rect", "polygon"])
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        self.btn_save_shape = QtWidgets.QPushButton("Save Shape")
        self.btn_save_shape.clicked.connect(self._save_shape)
        self.btn_clear_shape = QtWidgets.QPushButton("Clear Shape")
        self.btn_clear_shape.clicked.connect(self._clear_shape)
        ref_l.addWidget(self.btn_load_ref)
        ref_l.addWidget(self.lbl_ref)
        ref_l.addWidget(QtWidgets.QLabel("Label"))
        ref_l.addWidget(self.cmb_label)
        ref_l.addWidget(QtWidgets.QLabel("Shape"))
        ref_l.addWidget(self.cmb_shape)
        ref_l.addWidget(self.btn_save_shape)
        ref_l.addWidget(self.btn_clear_shape)
        ref_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        left.addWidget(ref_box)

        model_box = QtWidgets.QGroupBox("Model Params")
        model_l = QtWidgets.QFormLayout(model_box)
        self.spin_angle_start = QtWidgets.QDoubleSpinBox()
        self.spin_angle_start.setRange(-180.0, 180.0)
        self.spin_angle_start.setSingleStep(5.0)
        self.spin_angle_start.setValue(-20.0)
        self.spin_angle_extent = QtWidgets.QDoubleSpinBox()
        self.spin_angle_extent.setRange(0.0, 360.0)
        self.spin_angle_extent.setSingleStep(5.0)
        self.spin_angle_extent.setValue(40.0)
        self.spin_scale_min = QtWidgets.QDoubleSpinBox()
        self.spin_scale_min.setRange(0.1, 3.0)
        self.spin_scale_min.setSingleStep(0.02)
        self.spin_scale_min.setValue(0.95)
        self.spin_scale_max = QtWidgets.QDoubleSpinBox()
        self.spin_scale_max.setRange(0.1, 3.0)
        self.spin_scale_max.setSingleStep(0.02)
        self.spin_scale_max.setValue(1.05)
        self.spin_min_score = QtWidgets.QDoubleSpinBox()
        self.spin_min_score.setRange(0.0, 1.0)
        self.spin_min_score.setSingleStep(0.01)
        self.spin_min_score.setValue(0.15)
        self.spin_max_overlap = QtWidgets.QDoubleSpinBox()
        self.spin_max_overlap.setRange(0.0, 1.0)
        self.spin_max_overlap.setSingleStep(0.05)
        self.spin_max_overlap.setValue(0.3)
        self.spin_num_levels = QtWidgets.QSpinBox()
        self.spin_num_levels.setRange(0, 6)
        self.spin_num_levels.setValue(0)
        self.spin_max_dist = QtWidgets.QDoubleSpinBox()
        self.spin_max_dist.setRange(0.5, 10.0)
        self.spin_max_dist.setSingleStep(0.5)
        self.spin_max_dist.setValue(2.0)
        self.spin_max_ori = QtWidgets.QDoubleSpinBox()
        self.spin_max_ori.setRange(5.0, 90.0)
        self.spin_max_ori.setSingleStep(5.0)
        self.spin_max_ori.setValue(25.0)
        self.spin_angle_step = QtWidgets.QDoubleSpinBox()
        self.spin_angle_step.setRange(0.5, 45.0)
        self.spin_angle_step.setSingleStep(0.5)
        self.spin_angle_step.setValue(5.0)
        self.spin_scale_step = QtWidgets.QDoubleSpinBox()
        self.spin_scale_step.setRange(0.001, 0.2)
        self.spin_scale_step.setSingleStep(0.005)
        self.spin_scale_step.setValue(0.04)
        self.spin_greediness = QtWidgets.QDoubleSpinBox()
        self.spin_greediness.setRange(0.0, 1.0)
        self.spin_greediness.setSingleStep(0.05)
        self.spin_greediness.setValue(0.9)
        self.spin_search_resize = QtWidgets.QDoubleSpinBox()
        self.spin_search_resize.setRange(0.25, 1.0)
        self.spin_search_resize.setSingleStep(0.25)
        self.spin_search_resize.setValue(0.5)
        self.spin_topk_pose = QtWidgets.QSpinBox()
        self.spin_topk_pose.setRange(1, 50)
        self.spin_topk_pose.setValue(3)
        self.spin_max_candidates = QtWidgets.QSpinBox()
        self.spin_max_candidates.setRange(50, 1000)
        self.spin_max_candidates.setValue(60)
        self.chk_force_best = QtWidgets.QCheckBox("Force best (debug)")
        self.chk_force_best.setChecked(False)
        self.lbl_model = QtWidgets.QLabel("Model: none")
        self.btn_create_model = QtWidgets.QPushButton("Create Model")
        self.btn_create_model.clicked.connect(self._create_model)
        model_l.addRow("Angle start (deg)", self.spin_angle_start)
        model_l.addRow("Angle extent (deg)", self.spin_angle_extent)
        model_l.addRow("Scale min", self.spin_scale_min)
        model_l.addRow("Scale max", self.spin_scale_max)
        model_l.addRow("Min score", self.spin_min_score)
        model_l.addRow("Max overlap", self.spin_max_overlap)
        model_l.addRow("Pyramid levels (0=auto)", self.spin_num_levels)
        model_l.addRow("Max dist (px)", self.spin_max_dist)
        model_l.addRow("Max ori diff (deg)", self.spin_max_ori)
        model_l.addRow("Angle step (deg)", self.spin_angle_step)
        model_l.addRow("Scale step", self.spin_scale_step)
        model_l.addRow("Greediness", self.spin_greediness)
        model_l.addRow("Search resize", self.spin_search_resize)
        model_l.addRow("Top K per pose", self.spin_topk_pose)
        model_l.addRow("Max candidates", self.spin_max_candidates)
        model_l.addRow(self.chk_force_best)
        model_l.addRow(self.lbl_model)
        model_l.addRow(self.btn_create_model)
        model_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        left.addWidget(model_box)

        test_box = QtWidgets.QGroupBox("Test")
        test_l = QtWidgets.QVBoxLayout(test_box)
        self.btn_load_test = QtWidgets.QPushButton("Load Test")
        self.btn_load_test.clicked.connect(self._load_test)
        self.lbl_test = QtWidgets.QLabel("No test loaded")
        self.btn_find = QtWidgets.QPushButton("Find")
        self.btn_find.clicked.connect(self._run_find)
        self.lbl_result = QtWidgets.QLabel("")
        self.lbl_elapsed = QtWidgets.QLabel("Elapsed: --")
        self.btn_edit_search = QtWidgets.QPushButton("Draw Search Region")
        self.btn_edit_search.clicked.connect(self._edit_search_region)
        self.btn_save_search = QtWidgets.QPushButton("Save Search Region")
        self.btn_save_search.clicked.connect(self._save_search_region)
        self.btn_clear_search = QtWidgets.QPushButton("Clear Search Region")
        self.btn_clear_search.clicked.connect(self._clear_search_region)
        self.lbl_search = QtWidgets.QLabel("Search region: none")
        test_l.addWidget(self.btn_load_test)
        test_l.addWidget(self.lbl_test)
        test_l.addWidget(self.btn_find)
        test_l.addWidget(self.lbl_result)
        test_l.addWidget(self.lbl_elapsed)
        test_l.addWidget(self.btn_edit_search)
        test_l.addWidget(self.btn_save_search)
        test_l.addWidget(self.btn_clear_search)
        test_l.addWidget(self.lbl_search)
        test_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        left.addWidget(test_box)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        ref_view = QtWidgets.QWidget()
        ref_view_l = QtWidgets.QVBoxLayout(ref_view)
        ref_view_l.addWidget(QtWidgets.QLabel("Reference View"))
        self.ref_canvas = RoiCanvas()
        self.ref_canvas.setMinimumSize(640, 360)
        self.ref_canvas.shapesChanged.connect(self._on_ref_shapes_changed)
        ref_view_l.addWidget(self.ref_canvas, 1)
        splitter.addWidget(ref_view)

        test_view = QtWidgets.QWidget()
        test_view_l = QtWidgets.QVBoxLayout(test_view)
        test_view_l.addWidget(QtWidgets.QLabel("Test View"))
        self.test_canvas = RoiCanvas()
        self.test_canvas.setMinimumSize(640, 360)
        test_view_l.addWidget(self.test_canvas, 1)
        splitter.addWidget(test_view)

        right.addWidget(splitter, 1)

        self._update_save_enabled()

    def _update_save_enabled(self) -> None:
        st = self.ref_canvas.roi
        ok = (st.shape_type == "rect" and st.xywh is not None) or (st.shape_type == "polygon" and st.points)
        self.btn_save_shape.setEnabled(bool(ok))

    def _on_ref_shapes_changed(self) -> None:
        self._update_save_enabled()

    def _on_shape_changed(self) -> None:
        self.ref_canvas.draw_shape = self.cmb_shape.currentText()
        self.ref_canvas._poly_pts = []
        self.ref_canvas.update()

    def _on_label_changed(self) -> None:
        label = self.cmb_label.currentText()
        shape = self.ref_shapes.get(label)
        self.ref_canvas.clear_roi()
        if shape:
            if shape.shape_type == "rect" and shape.xywh is not None:
                self.ref_canvas.set_roi_rect(shape.xywh)
                self.cmb_shape.setCurrentText("rect")
            elif shape.shape_type == "polygon" and shape.points:
                self.ref_canvas.set_roi_polygon(shape.points)
                self.cmb_shape.setCurrentText("polygon")
        self._update_ref_overlays()

    def _load_reference(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        self.ref_path = path
        self.model = None
        self.ref_shapes = {}
        self.lbl_ref.setText(os.path.basename(path))
        self.ref_canvas.set_image(path)
        self._load_shapes_for_reference()
        self._load_model_for_reference()
        self._update_ref_overlays()
        self._update_save_enabled()

    def _load_test(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Test Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        self.test_path = path
        self.lbl_test.setText(os.path.basename(path))
        self.test_canvas.set_image(path)
        self.search_region_xywh = None
        self._last_anchor_pts = None
        self._refresh_test_overlays()
        self.test_canvas.clear_roi()
        self.lbl_result.setText("")
        self.lbl_elapsed.setText("Elapsed: --")
        self.lbl_search.setText("Search region: none")

    def _save_shape(self) -> None:
        label = self.cmb_label.currentText()
        st = self.ref_canvas.roi
        if label == "anchor_mask" and st.shape_type != "polygon":
            QtWidgets.QMessageBox.warning(self, "Warning", "anchor_mask must be polygon")
            return
        if st.shape_type == "rect" and st.xywh is not None:
            self.ref_shapes[label] = StoredShape(shape_type="rect", xywh=st.xywh, points=None)
        elif st.shape_type == "polygon" and st.points:
            self.ref_shapes[label] = StoredShape(shape_type="polygon", xywh=None, points=list(st.points))
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No valid shape")
            return
        self._update_ref_overlays()
        self._persist_shapes()

    def _clear_shape(self) -> None:
        label = self.cmb_label.currentText()
        if label in self.ref_shapes:
            del self.ref_shapes[label]
        self.ref_canvas.clear_roi()
        self._update_ref_overlays()
        self._persist_shapes()

    def _update_ref_overlays(self) -> None:
        label = self.cmb_label.currentText()
        overlays: List[OverlayShape] = []
        for name, color in (
            ("anchor", QtGui.QColor(0, 128, 255)),
            ("roi", QtGui.QColor(255, 165, 0)),
            ("anchor_mask", QtGui.QColor(255, 0, 0)),
        ):
            if name == label:
                continue
            shape = self.ref_shapes.get(name)
            if not shape:
                continue
            if shape.shape_type == "rect" and shape.xywh is not None:
                overlays.append(OverlayShape(shape_type="rect", xywh=shape.xywh, color=color))
            elif shape.shape_type == "polygon" and shape.points:
                overlays.append(OverlayShape(shape_type="polygon", points=shape.points, color=color))
        if self._model_points:
            overlays.append(
                OverlayShape(
                    shape_type="points",
                    points=self._model_points,
                    color=QtGui.QColor(0, 255, 0),
                    width=2,
                    dash=False,
                )
            )
        self.ref_canvas.set_overlays(overlays)

    def _refresh_test_overlays(self) -> None:
        overlays: List[OverlayShape] = []
        if self.search_region_xywh is not None:
            overlays.append(
                OverlayShape(
                    shape_type="rect",
                    xywh=self.search_region_xywh,
                    color=QtGui.QColor(160, 160, 160),
                    width=1,
                    dash=True,
                )
            )
        if self._last_anchor_pts:
            overlays.append(
                OverlayShape(
                    shape_type="polygon",
                    points=self._last_anchor_pts,
                    color=self._last_anchor_color,
                    width=2,
                    dash=False,
                )
            )
        self.test_canvas.set_overlays(overlays)

    def _edit_search_region(self) -> None:
        if not self.test_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load test image first")
            return
        self.test_canvas.draw_shape = "rect"
        if self.search_region_xywh is not None:
            self.test_canvas.set_roi_rect(self.search_region_xywh)
        else:
            self.test_canvas.clear_roi()
        self.lbl_result.setText("Draw search region on test view, then click Save Search Region")

    def _save_search_region(self) -> None:
        if not self.test_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load test image first")
            return
        roi = self.test_canvas.roi_xywh()
        if roi is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No valid search region")
            return
        self.search_region_xywh = roi
        self.test_canvas.clear_roi()
        self.lbl_search.setText(f"Search region: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
        self._refresh_test_overlays()

    def _clear_search_region(self) -> None:
        self.search_region_xywh = None
        self.test_canvas.clear_roi()
        self.lbl_search.setText("Search region: none")
        self._refresh_test_overlays()

    def _create_model(self) -> None:
        if not self.ref_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load reference image first")
            return
        anchor_shape = self.ref_shapes.get("anchor")
        if not anchor_shape:
            QtWidgets.QMessageBox.warning(self, "Warning", "Need anchor shape")
            return

        try:
            ref = _read_gray(self.ref_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        anchor_pts = _shape_to_points(anchor_shape)
        if not anchor_pts:
            QtWidgets.QMessageBox.warning(self, "Warning", "Anchor shape invalid")
            return
        mask = _mask_from_points(ref.shape[0], ref.shape[1], anchor_pts, fill_value=255)
        mask_shape = self.ref_shapes.get("anchor_mask")
        if mask_shape:
            mask_pts = _shape_to_points(mask_shape) or []
            if mask_pts:
                m = _mask_from_points(ref.shape[0], ref.shape[1], mask_pts, fill_value=255)
                mask[m > 0] = 0

        t0 = time.perf_counter()
        try:
            self.model = ScaledShapeModel.create(
                ref,
                mask=mask,
                max_model_points=800,
                max_r_vectors_per_bin=40,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Create Model Failed", str(e))
            return
        elapsed = time.perf_counter() - t0
        self._update_model_points()
        self._save_model_for_reference()
        self._update_ref_overlays()
        QtWidgets.QMessageBox.information(self, "Model Ready", f"Shape model created ({elapsed:.3f}s)")

    def _set_find_running(self, running: bool) -> None:
        self.btn_find.setEnabled(not running)
        self.btn_create_model.setEnabled(not running)
        self.btn_load_test.setEnabled(not running)
        self.btn_load_ref.setEnabled(not running)
        self.cmb_label.setEnabled(not running)
        self.cmb_shape.setEnabled(not running)
        self.spin_search_resize.setEnabled(not running)
        self.spin_topk_pose.setEnabled(not running)
        self.spin_max_candidates.setEnabled(not running)
        self.btn_edit_search.setEnabled(not running)
        self.btn_save_search.setEnabled(not running)
        self.btn_clear_search.setEnabled(not running)
        if running:
            self.lbl_result.setText("Finding...")
            self._find_start_ts = time.perf_counter()
            self._start_find_timer()
        else:
            self._stop_find_timer()
            self._find_start_ts = None

    def _run_find(self) -> None:
        if self._find_thread is not None:
            return
        if not self.model:
            QtWidgets.QMessageBox.warning(self, "Warning", "Create model first")
            return
        if not self.test_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load test image first")
            return
        roi_shape = self.ref_shapes.get("roi")
        anchor_shape = self.ref_shapes.get("anchor")
        if not roi_shape or not anchor_shape:
            QtWidgets.QMessageBox.warning(self, "Warning", "Need anchor and roi shapes")
            return

        roi_pts = _shape_to_points(roi_shape) or []
        anchor_pts = _shape_to_points(anchor_shape) or []
        if not roi_pts or not anchor_pts:
            QtWidgets.QMessageBox.warning(self, "Warning", "ROI/Anchor shape invalid")
            return

        angle_start = math.radians(float(self.spin_angle_start.value()))
        angle_extent = math.radians(float(self.spin_angle_extent.value()))
        scale_min = float(self.spin_scale_min.value())
        scale_max = float(self.spin_scale_max.value())
        min_score = float(self.spin_min_score.value())
        max_overlap = float(self.spin_max_overlap.value())
        num_levels = int(self.spin_num_levels.value())
        max_dist = float(self.spin_max_dist.value())
        max_ori = math.radians(float(self.spin_max_ori.value()))
        angle_step = math.radians(float(self.spin_angle_step.value()))
        scale_step = float(self.spin_scale_step.value())
        greediness = float(self.spin_greediness.value())
        search_resize = float(self.spin_search_resize.value())
        top_k_per_pose = int(self.spin_topk_pose.value())
        max_candidates = int(self.spin_max_candidates.value())
        if scale_min > scale_max:
            QtWidgets.QMessageBox.warning(self, "Warning", "scale_min must be <= scale_max")
            return

        params = {
            "angle_start": angle_start,
            "angle_extent": angle_extent,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "min_score": min_score,
            "num_matches": 1,
            "max_overlap": max_overlap,
            "subpixel": "none",
            "max_dist": max_dist,
            "max_ori_diff": max_ori,
            "angle_step": angle_step,
            "scale_step": scale_step,
            "greediness": greediness,
            "top_k_per_pose": top_k_per_pose,
            "max_candidates": max_candidates,
        }

        worker = FindWorker(
            model=self.model,
            test_path=self.test_path,
            anchor_pts=anchor_pts,
            roi_pts=roi_pts,
            search_region_xywh=self.search_region_xywh,
            search_resize=search_resize,
            params=params,
            force_best=self.chk_force_best.isChecked(),
            num_levels=num_levels,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_find_progress)
        worker.finished.connect(self._on_find_finished)
        worker.failed.connect(self._on_find_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup_find_thread)

        self._find_worker = worker
        self._find_thread = thread
        self._set_find_running(True)
        thread.start()

    def _on_find_progress(self, message: str) -> None:
        if self._find_thread is None:
            return
        elapsed = 0.0
        if self._find_start_ts is not None:
            elapsed = time.perf_counter() - self._find_start_ts
        self.lbl_result.setText(f"{message} (t={elapsed:.2f}s)")

    def _on_find_finished(self, result: dict) -> None:
        elapsed = float(result.get("elapsed", 0.0))
        self.lbl_elapsed.setText(f"Elapsed: {elapsed:.2f}s")
        if not result.get("matched"):
            self.lbl_result.setText(f"No match (t={elapsed:.2f}s)")
            self._last_anchor_pts = None
            self._refresh_test_overlays()
            self.test_canvas.clear_roi()
            return

        row = float(result["row"])
        col = float(result["col"])
        angle = float(result["angle"])
        scale = float(result["scale"])
        score = float(result["score"])
        below_threshold = bool(result.get("below_threshold", False))
        tgt_anchor_pts = result["tgt_anchor_pts"]
        tgt_roi_pts = result["tgt_roi_pts"]

        self.test_canvas.set_image(self.test_path)
        self._last_anchor_color = QtGui.QColor(255, 165, 0) if below_threshold else QtGui.QColor(0, 128, 255)
        self._last_anchor_pts = tgt_anchor_pts
        self.test_canvas.set_roi_polygon(tgt_roi_pts)
        self._refresh_test_overlays()

        status = "BEST (below threshold) " if below_threshold else ""
        self.lbl_result.setText(
            f"{status}row={row:.2f} col={col:.2f} angle={math.degrees(angle):.2f}deg "
            f"scale={scale:.3f} score={score:.3f} t={elapsed:.2f}s"
        )

    def _on_find_failed(self, message: str) -> None:
        if self._find_start_ts is not None:
            elapsed = time.perf_counter() - self._find_start_ts
            self.lbl_elapsed.setText(f"Elapsed: {elapsed:.2f}s")
        else:
            self.lbl_elapsed.setText("Elapsed: --")
        self.lbl_result.setText("Find failed")
        QtWidgets.QMessageBox.critical(self, "Find Failed", message)

    def _cleanup_find_thread(self) -> None:
        if self._find_worker is not None:
            self._find_worker.deleteLater()
        if self._find_thread is not None:
            self._find_thread.deleteLater()
        self._find_worker = None
        self._find_thread = None
        self._set_find_running(False)

    def _start_find_timer(self) -> None:
        if self._find_timer is None:
            self._find_timer = QtCore.QTimer(self)
            self._find_timer.timeout.connect(self._on_find_timer_tick)
        self._find_timer.start(100)

    def _stop_find_timer(self) -> None:
        if self._find_timer is not None:
            self._find_timer.stop()

    def _on_find_timer_tick(self) -> None:
        if self._find_start_ts is None:
            return
        elapsed = time.perf_counter() - self._find_start_ts
        self.lbl_elapsed.setText(f"Elapsed: {elapsed:.2f}s")

    def _default_model_path(self) -> Optional[str]:
        if not self.ref_path:
            return None
        base, _ext = os.path.splitext(self.ref_path)
        return base + ".shape_model.npz"

    def _default_shapes_path(self) -> Optional[str]:
        if not self.ref_path:
            return None
        base, _ext = os.path.splitext(self.ref_path)
        return base + ".shape_model.json"

    def _persist_shapes(self) -> None:
        path = self._default_shapes_path()
        if not path:
            return
        data = {
            "version": 1,
            "image": os.path.basename(self.ref_path) if self.ref_path else "",
            "shapes": {},
        }
        for label, shape in self.ref_shapes.items():
            data["shapes"][label] = {
                "shape_type": shape.shape_type,
                "xywh": shape.xywh,
                "points": shape.points,
            }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_shapes_for_reference(self) -> None:
        path = self._default_shapes_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        shapes = data.get("shapes", {})
        loaded: Dict[str, StoredShape] = {}
        if isinstance(shapes, dict):
            for label, s in shapes.items():
                if not isinstance(s, dict):
                    continue
                loaded[label] = StoredShape(
                    shape_type=str(s.get("shape_type", "rect")),
                    xywh=tuple(s["xywh"]) if s.get("xywh") else None,
                    points=[tuple(p) for p in s.get("points", [])] if s.get("points") else None,
                )
        self.ref_shapes = loaded

    def _save_model_for_reference(self) -> None:
        if not self.model:
            return
        path = self._default_model_path()
        if not path:
            return
        try:
            self.model.save(path)
            self.lbl_model.setText(f"Model: {os.path.basename(path)}")
        except Exception:
            self.lbl_model.setText("Model: save failed")

    def _load_model_for_reference(self) -> None:
        path = self._default_model_path()
        if not path or not os.path.exists(path):
            self.lbl_model.setText("Model: none")
            self._model_points = None
            return
        try:
            self.model = ScaledShapeModel.load(path)
            self.lbl_model.setText(f"Model: {os.path.basename(path)}")
            self._update_model_points()
        except Exception:
            self.model = None
            self._model_points = None
            self.lbl_model.setText("Model: load failed")

    def _update_model_points(self) -> None:
        if not self.model:
            self._model_points = None
            return
        orow, ocol = self.model.origin_rc
        pts: List[Point] = []
        for dcol, drow in self.model.model_rel_xy.tolist():
            pts.append((float(ocol + dcol), float(orow + drow)))
        self._model_points = pts


def main() -> None:
    app = QtWidgets.QApplication([])
    w = ShapeModelDemo()
    w.resize(1300, 800)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
