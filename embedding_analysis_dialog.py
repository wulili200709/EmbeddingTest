from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - handled in UI
    FigureCanvas = None
    Figure = None

from visualize_embeddings import (
    EmbeddingAnalysisResult,
    list_available_backbones,
    list_product_names,
    load_product_analysis,
)


class EmbeddingAnalysisDialog(QtWidgets.QDialog):
    def __init__(
        self,
        session_root: str,
        initial_product: str = "",
        initial_backbone: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("特征分析")
        self.resize(1200, 760)
        self.session_root = session_root
        self._result: Optional[EmbeddingAnalysisResult] = None

        self._build_ui()
        self._load_products(initial_product, initial_backbone)
        self._refresh_analysis()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QGridLayout()
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.currentTextChanged.connect(self._on_product_changed)
        self.cmb_backbone = QtWidgets.QComboBox()
        self.cmb_projection = QtWidgets.QComboBox()
        self.cmb_projection.addItems(["tsne", "pca"])
        self.btn_refresh = QtWidgets.QPushButton("刷新分析")
        self.btn_refresh.clicked.connect(self._refresh_analysis)

        self.lbl_model_path = QtWidgets.QLabel("-")
        self.lbl_session_path = QtWidgets.QLabel("-")
        self.lbl_model_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_session_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        controls.addWidget(QtWidgets.QLabel("产品"), 0, 0)
        controls.addWidget(self.cmb_product, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Backbone"), 0, 2)
        controls.addWidget(self.cmb_backbone, 0, 3)
        controls.addWidget(QtWidgets.QLabel("投影"), 0, 4)
        controls.addWidget(self.cmb_projection, 0, 5)
        controls.addWidget(self.btn_refresh, 0, 6)
        controls.addWidget(QtWidgets.QLabel("模型"), 1, 0)
        controls.addWidget(self.lbl_model_path, 1, 1, 1, 6)
        controls.addWidget(QtWidgets.QLabel("Session"), 2, 0)
        controls.addWidget(self.lbl_session_path, 2, 1, 1, 6)
        root.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left)
        self.txt_summary = QtWidgets.QPlainTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumBlockCount(200)
        left_l.addWidget(QtWidgets.QLabel("统计"))
        left_l.addWidget(self.txt_summary, 0)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["GT", "Pred", "diff", "sim_ok", "sim_ng", "文件"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        left_l.addWidget(QtWidgets.QLabel("样本"))
        left_l.addWidget(self.table, 1)

        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        if FigureCanvas is None or Figure is None:
            self.figure = None
            self.figure_canvas = None
            self.lbl_plot_error = QtWidgets.QLabel("matplotlib 不可用，无法显示散点图。")
            self.lbl_plot_error.setAlignment(QtCore.Qt.AlignCenter)
            right_l.addWidget(self.lbl_plot_error, 1)
        else:
            self.figure = Figure(figsize=(6.0, 5.0))
            self.figure_canvas = FigureCanvas(self.figure)
            right_l.addWidget(self.figure_canvas, 1)
        splitter.addWidget(right)
        splitter.setSizes([430, 770])

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def _load_products(self, initial_product: str, initial_backbone: str):
        names = list_product_names(self.session_root)
        self.cmb_product.blockSignals(True)
        self.cmb_product.clear()
        self.cmb_product.addItems(names)
        self.cmb_product.blockSignals(False)
        if initial_product and initial_product in names:
            self.cmb_product.setCurrentText(initial_product)
        elif names:
            self.cmb_product.setCurrentIndex(0)
        self._load_backbones(initial_backbone)

    def _load_backbones(self, initial_backbone: str = ""):
        product_name = self.cmb_product.currentText().strip()
        product_dir = os.path.join(self.session_root, product_name)
        backbones = list_available_backbones(product_dir)
        self.cmb_backbone.blockSignals(True)
        self.cmb_backbone.clear()
        self.cmb_backbone.addItems(backbones)
        self.cmb_backbone.blockSignals(False)
        if initial_backbone and initial_backbone in backbones:
            self.cmb_backbone.setCurrentText(initial_backbone)
        elif backbones:
            self.cmb_backbone.setCurrentIndex(0)

    def _on_product_changed(self, _product_name: str):
        self._load_backbones()

    def _refresh_analysis(self):
        product_name = self.cmb_product.currentText().strip()
        backbone = self.cmb_backbone.currentText().strip()
        projection = self.cmb_projection.currentText().strip()
        if not product_name:
            self._show_error("没有可用产品。")
            return
        if not backbone:
            self._show_error("当前产品下没有可用 register_model_*.npz。")
            return

        try:
            result = load_product_analysis(
                session_root=self.session_root,
                product_name=product_name,
                backbone=backbone,
                projection_method=projection,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._result = result
        self._render_result(result)

    def _show_error(self, message: str):
        self._result = None
        self.lbl_model_path.setText("-")
        self.lbl_session_path.setText("-")
        self.txt_summary.setPlainText(message)
        self.table.setRowCount(0)
        if self.figure is not None and self.figure_canvas is not None:
            self.figure.clear()
            self.figure_canvas.draw_idle()

    def _render_result(self, result: EmbeddingAnalysisResult):
        self.lbl_model_path.setText(result.model_path)
        self.lbl_session_path.setText(result.session_file)
        self.txt_summary.setPlainText(self._format_summary(result))
        self._fill_table(result)
        self._draw_plot(result)

    def _format_summary(self, result: EmbeddingAnalysisResult) -> str:
        m = result.metrics
        lines = [
            f"产品: {result.product_name}",
            f"Backbone: {result.backbone}",
            f"投影: {result.projection_method.upper()}",
            f"特征维度: {result.feature_dim}",
            f"OK 样本数: {int(m['ok_count'])}",
            f"NG 样本数: {int(m['ng_count'])}",
            f"训练集判定准确率: {m['train_accuracy']:.4f}",
            f"OK 类内平均相似度: {m['ok_intra_mean']:.4f}",
            f"NG 类内平均相似度: {m['ng_intra_mean']:.4f}",
            f"OK-NG 类间平均相似度: {m['ok_ng_cross_mean']:.4f}",
            f"OK -> OK proto: {m['ok_to_ok_proto']:.4f}",
            f"NG -> NG proto: {m['ng_to_ng_proto']:.4f}",
            f"OK -> NG proto: {m['ok_to_ng_proto']:.4f}",
            f"NG -> OK proto: {m['ng_to_ok_proto']:.4f}",
            f"OK/NG proto 相似度: {m['proto_similarity']:.4f}",
        ]
        if result.notes:
            lines.append("")
            lines.append("备注:")
            lines.extend(f"- {note}" for note in result.notes)
        return "\n".join(lines)

    def _fill_table(self, result: EmbeddingAnalysisResult):
        self.table.setRowCount(0)
        for row_idx, row in enumerate(result.rows):
            self.table.insertRow(row_idx)
            values = [
                row.gt_label,
                row.pred_label,
                f"{row.diff:.4f}",
                f"{row.sim_ok:.4f}",
                f"{row.sim_ng:.4f}",
                row.file_name,
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 5 and row.file_path:
                    item.setToolTip(row.file_path)
                if row.gt_label != row.pred_label:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(192, 32, 32)))
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()

    def _draw_plot(self, result: EmbeddingAnalysisResult):
        if self.figure is None or self.figure_canvas is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        coords = result.point_coords
        if coords.size == 0:
            ax.set_title("无特征可显示")
            self.figure_canvas.draw_idle()
            return

        ok_idx = [idx for idx, label in enumerate(result.point_labels) if label == "OK"]
        ng_idx = [idx for idx, label in enumerate(result.point_labels) if label == "NG"]
        if ok_idx:
            ok_pts = coords[ok_idx]
            ax.scatter(ok_pts[:, 0], ok_pts[:, 1], c="tab:blue", s=55, alpha=0.85, label="OK")
            center = ok_pts.mean(axis=0)
            ax.scatter([center[0]], [center[1]], c="navy", marker="*", s=260, label="OK center")
        if ng_idx:
            ng_pts = coords[ng_idx]
            ax.scatter(ng_pts[:, 0], ng_pts[:, 1], c="tab:red", s=55, alpha=0.85, label="NG")
            center = ng_pts.mean(axis=0)
            ax.scatter([center[0]], [center[1]], c="darkred", marker="*", s=260, label="NG center")

        for idx, name in enumerate(result.point_names):
            ax.annotate(name, (coords[idx, 0], coords[idx, 1]), fontsize=8, alpha=0.75)

        ax.set_title(f"{result.product_name} / {result.backbone} / {result.projection_method.upper()}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        self.figure.tight_layout()
        self.figure_canvas.draw_idle()
