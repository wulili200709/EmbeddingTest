from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.anomaly_heatmap import AnomalyHeatmapResult, generate_anomaly_heatmap
from algorithms.registry import algorithm_display_name


class AnomalyHeatmapDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        algo_controller,
        product_dir: str,
        image_path: str,
        algorithm: str,
        model_key: str,
        tool_name: str,
        roi_label: str,
        ok_files: list[str],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("异常热力图")
        self.resize(1380, 900)

        self.algo = algo_controller
        self.product_dir = str(product_dir or "").strip()
        self.image_path = str(image_path or "").strip()
        self.algorithm = str(algorithm or "").strip()
        self.model_key = str(model_key or "").strip()
        self.roi_label = str(roi_label or "").strip() or "roi"
        self.tool_name = str(tool_name or "").strip() or self.model_key or self.roi_label
        self.ok_files = [str(path or "").strip() for path in list(ok_files or []) if str(path or "").strip()]
        self._result: Optional[AnomalyHeatmapResult] = None

        self._build_ui()
        QtCore.QTimer.singleShot(0, self._refresh_heatmap)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QGridLayout()
        self.lbl_tool = QtWidgets.QLabel(self.tool_name)
        self.lbl_tool.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_roi = QtWidgets.QLabel(self.roi_label)
        self.lbl_image = QtWidgets.QLabel(self.image_path)
        self.lbl_image.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_image.setToolTip(self.image_path)
        self.lbl_model = QtWidgets.QLabel("-")
        self.lbl_model.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.btn_refresh = QtWidgets.QPushButton("刷新热力图")
        self.btn_refresh.clicked.connect(self._refresh_heatmap)

        controls.addWidget(QtWidgets.QLabel("工具"), 0, 0)
        controls.addWidget(self.lbl_tool, 0, 1)
        controls.addWidget(QtWidgets.QLabel("ROI"), 0, 2)
        controls.addWidget(self.lbl_roi, 0, 3)
        controls.addWidget(self.btn_refresh, 0, 4)
        controls.addWidget(QtWidgets.QLabel("图片"), 1, 0)
        controls.addWidget(self.lbl_image, 1, 1, 1, 4)
        controls.addWidget(QtWidgets.QLabel("模型"), 2, 0)
        controls.addWidget(self.lbl_model, 2, 1, 1, 4)
        root.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        summary_panel = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_panel)
        summary_layout.addWidget(QtWidgets.QLabel("判定说明"))
        self.txt_summary = QtWidgets.QPlainTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumBlockCount(400)
        summary_layout.addWidget(self.txt_summary, 1)
        splitter.addWidget(summary_panel)

        image_panel = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_panel)
        self.tabs = QtWidgets.QTabWidget()
        self.lbl_full_overlay = self._build_image_tab("整图叠加", self.tabs)
        self.lbl_roi_raw = self._build_image_tab("ROI原图", self.tabs)
        self.lbl_roi_overlay = self._build_image_tab("ROI叠加", self.tabs)
        self.lbl_roi_heatmap = self._build_image_tab("ROI热力图", self.tabs)
        image_layout.addWidget(self.tabs, 1)
        splitter.addWidget(image_panel)
        splitter.setSizes([350, 1030])

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def _build_image_tab(self, title: str, tabs: QtWidgets.QTabWidget) -> QtWidgets.QLabel:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        label = QtWidgets.QLabel("等待生成...")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setBackgroundRole(QtGui.QPalette.Base)
        label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        tabs.addTab(page, title)
        return label

    def _refresh_heatmap(self) -> None:
        self.btn_refresh.setEnabled(False)
        self.txt_summary.setPlainText("正在生成异常热力图...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            model_path = self.algo.anomaly_model_path(
                self.algorithm,
                self.product_dir,
                model_key=self.model_key,
            )
            self.lbl_model.setText(model_path)
            self.lbl_model.setToolTip(model_path)
            self.algo.load_model_for_algorithm(
                self.algorithm,
                self.product_dir,
                model_key=self.model_key,
            )
            model = self.algo.model
            if model is None:
                raise RuntimeError("当前异常模型未加载")
            feat_net = self.algo.get_feat_net(model.backbone, getattr(model, "device", None))
            result = generate_anomaly_heatmap(
                self.image_path,
                ok_files=self.ok_files,
                model=model,
                label_name=self.roi_label,
                feat_net=feat_net,
            )
        except Exception as exc:
            self._result = None
            self._clear_image_tabs(str(exc))
            self.txt_summary.setPlainText(f"生成异常热力图失败：\n{exc}")
        else:
            self._result = result
            self._render_result(result)
        finally:
            self.btn_refresh.setEnabled(True)
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()

    def _clear_image_tabs(self, message: str) -> None:
        for label in (
            self.lbl_full_overlay,
            self.lbl_roi_raw,
            self.lbl_roi_overlay,
            self.lbl_roi_heatmap,
        ):
            label.clear()
            label.setText(message)

    def _render_result(self, result: AnomalyHeatmapResult) -> None:
        self.txt_summary.setPlainText(self._format_summary(result))
        self._set_bgr_pixmap(self.lbl_full_overlay, result.overlay_bgr, max_width=980, max_height=760)
        self._set_bgr_pixmap(self.lbl_roi_raw, result.roi_bgr, max_width=760, max_height=760, min_width=420)
        self._set_bgr_pixmap(self.lbl_roi_overlay, result.roi_overlay_bgr, max_width=760, max_height=760, min_width=420)
        self._set_bgr_pixmap(self.lbl_roi_heatmap, result.roi_heatmap_bgr, max_width=760, max_height=760, min_width=420)

    def _format_summary(self, result: AnomalyHeatmapResult) -> str:
        algorithm_name = algorithm_display_name(self.algorithm) or self.algorithm
        x, y, w, h = result.roi_xywh
        lines = [
            f"工具: {self.tool_name}",
            f"算法: {algorithm_name}",
            f"图片: {os.path.basename(result.image_path)}",
            f"ROI: {result.roi_label} @ ({x}, {y}, {w}, {h})",
            "",
            f"判定结果: {result.pred}",
            f"整图 anomaly score: {result.score:.4f}",
            f"threshold: {result.threshold:.4f}",
            f"threshold-score: {result.diff:.4f}",
            "",
            f"patch max: {result.patch_max:.4f}",
            f"patch mean: {result.patch_mean:.4f}",
            f"coarse patch grid: {result.coarse_patch_scores.shape[1]} x {result.coarse_patch_scores.shape[0]}",
            f"OK 样本数: {result.ok_image_count}",
            f"patch bank 数量: {result.ok_patch_count}",
            f"topk: {result.topk}",
            "",
            "说明:",
            "1. 最终 OK/NG 仍按整图 anomaly score 与 threshold 判定。",
            "2. 热力图颜色显示的是当前 ROI 内相对更异常的位置，不直接等于判定阈值。",
            "3. 如果局部很红但整图仍判 OK，通常说明缺陷区域太小，被整图特征平均掉了。",
        ]
        return "\n".join(lines)

    def _set_bgr_pixmap(
        self,
        label: QtWidgets.QLabel,
        image_bgr: np.ndarray,
        *,
        max_width: int,
        max_height: int,
        min_width: int = 0,
    ) -> None:
        pixmap = self._pixmap_from_bgr(image_bgr)
        width = pixmap.width()
        height = pixmap.height()
        scale = 1.0
        if width > max_width or height > max_height:
            scale = min(float(max_width) / float(max(1, width)), float(max_height) / float(max(1, height)))
        elif min_width > 0 and width < min_width:
            scale = min(float(min_width) / float(max(1, width)), 4.0)
        if abs(scale - 1.0) > 1e-6:
            pixmap = pixmap.scaled(
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        label.setPixmap(pixmap)
        label.setText("")
        label.setMinimumSize(pixmap.size())
        label.setToolTip(f"原始尺寸: {width} x {height}")

    @staticmethod
    def _pixmap_from_bgr(image_bgr: np.ndarray) -> QtGui.QPixmap:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"unsupported image shape: {image.shape!r}")
        rgb = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
        qimage = QtGui.QImage(
            rgb.data,
            int(rgb.shape[1]),
            int(rgb.shape[0]),
            int(rgb.strides[0]),
            QtGui.QImage.Format_RGB888,
        )
        return QtGui.QPixmap.fromImage(qimage.copy())
