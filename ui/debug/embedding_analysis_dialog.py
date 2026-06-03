from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.registry import algorithm_display_name
from ui.i18n import tr

try:
    from matplotlib import font_manager as mpl_font_manager
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - handled in UI
    FigureCanvas = None
    Figure = None
    mpl_font_manager = None

from .embedding_analysis import (
    EmbeddingAnalysisResult,
    EmbeddingModelEntry,
    list_available_embedding_models,
    list_product_names,
    load_product_analysis,
)


class EmbeddingAnalysisDialog(QtWidgets.QDialog):
    _analysis_cache: dict[tuple[str, str, str, str, int, int, int], EmbeddingAnalysisResult] = {}
    _plot_font_props = None

    def __init__(
        self,
        session_root: str,
        initial_product: str = "",
        initial_backbone: str = "",
        initial_model_key: str = "",
        allowed_model_keys: Optional[list[str]] = None,
        allowed_backbones: Optional[list[str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.resize(1200, 760)
        self.session_root = session_root
        self._allowed_model_keys = {
            str(model_key or "").strip()
            for model_key in list(allowed_model_keys or [])
            if str(model_key or "").strip()
        }
        self._allowed_backbones = {
            str(backbone or "").strip()
            for backbone in list(allowed_backbones or [])
            if str(backbone or "").strip()
        }
        self._result: Optional[EmbeddingAnalysisResult] = None
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_analysis)

        self._build_ui()
        self.retranslate_ui()
        self._load_products(initial_product, initial_backbone, initial_model_key)
        QtCore.QTimer.singleShot(0, self._refresh_analysis)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QGridLayout()
        self.lbl_product = QtWidgets.QLabel()
        self.lbl_learning_tool = QtWidgets.QLabel()
        self.lbl_projection = QtWidgets.QLabel()
        self.lbl_model = QtWidgets.QLabel()
        self.lbl_session = QtWidgets.QLabel()

        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.currentTextChanged.connect(self._on_product_changed)
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.currentIndexChanged.connect(self._schedule_refresh_analysis)
        self.cmb_projection = QtWidgets.QComboBox()
        self.cmb_projection.addItem("TSNE", "tsne")
        self.cmb_projection.addItem("PCA", "pca")
        self.cmb_projection.currentIndexChanged.connect(self._schedule_refresh_analysis)
        self.btn_refresh = QtWidgets.QPushButton()
        self.btn_refresh.clicked.connect(self._refresh_analysis)

        self.lbl_model_path = QtWidgets.QLabel("-")
        self.lbl_session_path = QtWidgets.QLabel("-")
        self.lbl_model_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_session_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        controls.addWidget(self.lbl_product, 0, 0)
        controls.addWidget(self.cmb_product, 0, 1)
        controls.addWidget(self.lbl_learning_tool, 0, 2)
        controls.addWidget(self.cmb_model, 0, 3)
        controls.addWidget(self.lbl_projection, 0, 4)
        controls.addWidget(self.cmb_projection, 0, 5)
        controls.addWidget(self.btn_refresh, 0, 6)
        controls.addWidget(self.lbl_model, 1, 0)
        controls.addWidget(self.lbl_model_path, 1, 1, 1, 6)
        controls.addWidget(self.lbl_session, 2, 0)
        controls.addWidget(self.lbl_session_path, 2, 1, 1, 6)
        root.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left)
        self.lbl_stats = QtWidgets.QLabel()
        self.txt_summary = QtWidgets.QPlainTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumBlockCount(200)
        left_l.addWidget(self.lbl_stats)
        left_l.addWidget(self.txt_summary, 0)

        self.lbl_samples = QtWidgets.QLabel()
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        left_l.addWidget(self.lbl_samples)
        left_l.addWidget(self.table, 1)

        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        if FigureCanvas is None or Figure is None:
            self.figure = None
            self.figure_canvas = None
            self.lbl_plot_error = QtWidgets.QLabel()
            self.lbl_plot_error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            right_l.addWidget(self.lbl_plot_error, 1)
        else:
            self.figure = Figure(figsize=(6.0, 5.0))
            self.figure_canvas = FigureCanvas(self.figure)
            self.lbl_plot_error = None
            right_l.addWidget(self.figure_canvas, 1)
        splitter.addWidget(right)
        splitter.setSizes([430, 770])

        self.btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.btn_box.rejected.connect(self.reject)
        root.addWidget(self.btn_box)

    def retranslate_ui(self) -> None:
        self.setWindowTitle(tr("debug.embedding.title"))
        self.lbl_product.setText(tr("debug.embedding.product"))
        self.lbl_learning_tool.setText(tr("debug.embedding.learning_tool"))
        self.lbl_projection.setText(tr("debug.embedding.projection"))
        self.lbl_model.setText(tr("debug.embedding.model"))
        self.lbl_session.setText(tr("debug.embedding.session"))
        self.lbl_stats.setText(tr("debug.embedding.stats"))
        self.lbl_samples.setText(tr("debug.embedding.samples"))
        self.btn_refresh.setText(tr("debug.embedding.refresh"))
        self.table.setHorizontalHeaderLabels([
            "GT",
            "Pred",
            "diff",
            "sim_ok",
            "sim_ng",
            tr("debug.embedding.table.file"),
        ])
        close_button = self.btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText(tr("sample.close"))
        if self.lbl_plot_error is not None:
            self.lbl_plot_error.setText(tr("debug.embedding.matplotlib_unavailable"))
        if self._result is not None:
            self._render_result(self._result)

    def _shared_model_name(self) -> str:
        return tr("debug.embedding.shared_model")

    def _load_products(self, initial_product: str, initial_backbone: str, initial_model_key: str) -> None:
        names = list_product_names(self.session_root)
        self.cmb_product.blockSignals(True)
        self.cmb_product.clear()
        self.cmb_product.addItems(names)
        self.cmb_product.blockSignals(False)
        if initial_product and initial_product in names:
            self.cmb_product.setCurrentText(initial_product)
        elif names:
            self.cmb_product.setCurrentIndex(0)
        self._load_models(initial_backbone, initial_model_key)

    def _load_models(self, initial_backbone: str = "", initial_model_key: str = "") -> None:
        product_name = self.cmb_product.currentText().strip()
        product_dir = os.path.join(self.session_root, product_name)
        entries = list_available_embedding_models(product_dir)
        if self._allowed_model_keys:
            entries = [entry for entry in entries if entry.model_key in self._allowed_model_keys]
        if self._allowed_backbones:
            entries = [entry for entry in entries if entry.backbone in self._allowed_backbones]
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        for entry in entries:
            self.cmb_model.addItem(entry.display_name, entry)
        self.cmb_model.blockSignals(False)
        initial_backbone = str(initial_backbone or "").strip()
        initial_model_key = str(initial_model_key or "").strip()
        if entries:
            selected_index = 0
            for index, entry in enumerate(entries):
                if initial_model_key and entry.model_key != initial_model_key:
                    continue
                if initial_backbone and entry.backbone != initial_backbone:
                    continue
                selected_index = index
                break
            self.cmb_model.setCurrentIndex(selected_index)

    def _on_product_changed(self, _product_name: str) -> None:
        self._load_models()
        self._schedule_refresh_analysis()

    def _schedule_refresh_analysis(self) -> None:
        self._refresh_timer.start(50)

    def _current_model_entry(self) -> Optional[EmbeddingModelEntry]:
        entry = self.cmb_model.currentData()
        if isinstance(entry, EmbeddingModelEntry):
            return entry
        return None

    def _analysis_cache_key(
        self,
        product_name: str,
        entry: EmbeddingModelEntry,
        projection: str,
    ) -> tuple[str, str, str, str, int, int, int]:
        product_dir = os.path.join(self.session_root, product_name)
        session_file = os.path.join(product_dir, "session.json")
        params_file = os.path.join(product_dir, "product_params.json")
        model_mtime = int(os.path.getmtime(entry.model_path) * 1000) if os.path.exists(entry.model_path) else -1
        session_mtime = int(os.path.getmtime(session_file) * 1000) if os.path.exists(session_file) else -1
        params_mtime = int(os.path.getmtime(params_file) * 1000) if os.path.exists(params_file) else -1
        return (
            product_name,
            entry.backbone,
            entry.model_key,
            str(projection or "tsne").strip().lower(),
            model_mtime,
            session_mtime,
            params_mtime,
        )

    def _refresh_analysis(self) -> None:
        product_name = self.cmb_product.currentText().strip()
        projection = str(self.cmb_projection.currentData() or self.cmb_projection.currentText() or "tsne").strip().lower()
        entry = self._current_model_entry()
        if not product_name:
            self._show_error(tr("debug.embedding.no_product"))
            return
        if entry is None:
            self._show_error(tr("debug.embedding.no_model"))
            return

        cache_key = self._analysis_cache_key(product_name, entry, projection)
        self.btn_refresh.setEnabled(False)
        self.txt_summary.setPlainText(tr("debug.embedding.loading"))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            result = self._analysis_cache.get(cache_key)
            if result is None:
                result = load_product_analysis(
                    session_root=self.session_root,
                    product_name=product_name,
                    backbone=entry.backbone,
                    model_key=entry.model_key,
                    projection_method=projection,
                )
                self._analysis_cache[cache_key] = result
        except Exception as exc:
            self._show_error(str(exc))
        else:
            self._result = result
            self._render_result(result)
        finally:
            self.btn_refresh.setEnabled(True)
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()

    def _show_error(self, message: str) -> None:
        self._result = None
        self.lbl_model_path.setText("-")
        self.lbl_session_path.setText("-")
        self.txt_summary.setPlainText(message)
        self.table.setRowCount(0)
        if self.figure is not None and self.figure_canvas is not None:
            self.figure.clear()
            self.figure_canvas.draw_idle()

    @classmethod
    def _resolve_plot_font_props(cls):
        if cls._plot_font_props is not None or mpl_font_manager is None:
            return cls._plot_font_props
        font_families = [
            "Microsoft YaHei",
            "Microsoft JhengHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "WenQuanYi Micro Hei",
            "Arial Unicode MS",
        ]
        for family in font_families:
            try:
                font_path = mpl_font_manager.findfont(
                    mpl_font_manager.FontProperties(family=family),
                    fallback_to_default=False,
                )
            except Exception:
                continue
            if font_path and os.path.exists(font_path):
                cls._plot_font_props = mpl_font_manager.FontProperties(fname=font_path)
                break
        return cls._plot_font_props

    @classmethod
    def _plot_text(cls, text: str) -> str:
        if cls._resolve_plot_font_props() is not None:
            return text
        return text.encode("ascii", "replace").decode("ascii")

    def _render_result(self, result: EmbeddingAnalysisResult) -> None:
        self.lbl_model_path.setText(result.model_path)
        self.lbl_session_path.setText(result.session_file)
        self.txt_summary.setPlainText(self._format_summary(result))
        self._fill_table(result)
        self._draw_plot(result)

    def _format_summary_legacy(self, result: EmbeddingAnalysisResult) -> str:
        return self._format_summary(result)

    def _format_summary(self, result: EmbeddingAnalysisResult) -> str:
        m = result.metrics
        safe_low = float(m.get("safe_margin_low", float("nan")))
        safe_high = float(m.get("safe_margin_high", float("nan")))
        diff_gap = float(m.get("diff_gap", float("nan")))

        def _fmt_metric(name: str) -> str:
            value = float(m.get(name, float("nan")))
            return f"{value:.4f}" if value == value else "-"

        if safe_low == safe_low and safe_high == safe_high and diff_gap > 0:
            safe_range_text = f"{safe_low:.4f} ~ {safe_high:.4f}"
        elif diff_gap == diff_gap:
            safe_range_text = tr("debug.embedding.summary.safe_margin_none")
        else:
            safe_range_text = "-"

        lines = [
            f"{tr('debug.embedding.summary.product')}: {result.product_name}",
            f"{tr('debug.embedding.summary.tool')}: {result.tool_name or self._shared_model_name()}",
            f"{tr('debug.embedding.summary.backbone')}: {algorithm_display_name(result.backbone) or result.backbone}",
            f"{tr('debug.embedding.summary.roi')}: {', '.join(result.label_names) if result.label_names else '-'}",
            f"{tr('debug.embedding.summary.projection')}: {result.projection_method.upper()}",
            f"{tr('debug.embedding.summary.feature_dim')}: {result.feature_dim}",
            f"{tr('debug.embedding.summary.rule')}: {tr('debug.embedding.summary.rule_margin') if result.score_mode else '-'}",
            f"{tr('debug.embedding.summary.score_mode')}: {result.score_mode or '-'}",
            f"{tr('debug.embedding.summary.current_margin')}: {result.margin:.4f}",
            f"{tr('debug.embedding.summary.topk')}: {result.topk if result.topk else '-'}",
            f"{tr('debug.embedding.summary.ok_count')}: {int(m['ok_count'])}",
            f"{tr('debug.embedding.summary.ng_count')}: {int(m['ng_count'])}",
            f"{tr('debug.embedding.summary.train_accuracy')}: {m['train_accuracy']:.4f}",
            f"{tr('debug.embedding.summary.ok_diff_range')}: {_fmt_metric('ok_diff_min')} ~ {_fmt_metric('ok_diff_max')}",
            f"{tr('debug.embedding.summary.ng_diff_range')}: {_fmt_metric('ng_diff_min')} ~ {_fmt_metric('ng_diff_max')}",
            f"{tr('debug.embedding.summary.diff_gap')}: {_fmt_metric('diff_gap')}",
            f"{tr('debug.embedding.summary.safe_margin_range')}: {safe_range_text}",
            f"{tr('debug.embedding.summary.suggested_margin')}: {_fmt_metric('suggested_margin')}",
            f"{tr('debug.embedding.summary.suggested_accuracy')}: {_fmt_metric('suggested_accuracy')}",
            f"{tr('debug.embedding.summary.ok_intra_mean')}: {m['ok_intra_mean']:.4f}",
            f"{tr('debug.embedding.summary.ng_intra_mean')}: {m['ng_intra_mean']:.4f}",
            f"{tr('debug.embedding.summary.ok_ng_cross_mean')}: {m['ok_ng_cross_mean']:.4f}",
            f"{tr('debug.embedding.summary.ok_to_ok_proto')}: {m['ok_to_ok_proto']:.4f}",
            f"{tr('debug.embedding.summary.ng_to_ng_proto')}: {m['ng_to_ng_proto']:.4f}",
            f"{tr('debug.embedding.summary.ok_to_ng_proto')}: {m['ok_to_ng_proto']:.4f}",
            f"{tr('debug.embedding.summary.ng_to_ok_proto')}: {m['ng_to_ok_proto']:.4f}",
            f"{tr('debug.embedding.summary.proto_similarity')}: {m['proto_similarity']:.4f}",
        ]
        if result.notes:
            lines.append("")
            lines.append(f"{tr('debug.embedding.summary.notes')}:")
            lines.extend(f"- {note}" for note in result.notes)
        return "\n".join(lines)

    def _fill_table(self, result: EmbeddingAnalysisResult) -> None:
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

    def _draw_plot(self, result: EmbeddingAnalysisResult) -> None:
        if self.figure is None or self.figure_canvas is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        font_props = self._resolve_plot_font_props()

        coords = result.point_coords
        if coords.size == 0:
            ax.set_title(self._plot_text(tr("debug.embedding.plot.no_features")), fontproperties=font_props)
            self.figure_canvas.draw_idle()
            return

        ok_idx = [idx for idx, label in enumerate(result.point_labels) if label == "OK"]
        ng_idx = [idx for idx, label in enumerate(result.point_labels) if label == "NG"]
        if ok_idx:
            ok_pts = coords[ok_idx]
            ax.scatter(ok_pts[:, 0], ok_pts[:, 1], c="tab:blue", s=55, alpha=0.85, label="OK")
            center = ok_pts.mean(axis=0)
            ax.scatter([center[0]], [center[1]], c="navy", marker="*", s=260, label=tr("debug.embedding.plot.ok_center"))
        if ng_idx:
            ng_pts = coords[ng_idx]
            ax.scatter(ng_pts[:, 0], ng_pts[:, 1], c="tab:red", s=55, alpha=0.85, label="NG")
            center = ng_pts.mean(axis=0)
            ax.scatter([center[0]], [center[1]], c="darkred", marker="*", s=260, label=tr("debug.embedding.plot.ng_center"))

        for idx, name in enumerate(result.point_names):
            ax.annotate(
                self._plot_text(name),
                (coords[idx, 0], coords[idx, 1]),
                fontsize=8,
                alpha=0.75,
                fontproperties=font_props,
            )

        ax.set_title(
            self._plot_text(
                f"{result.product_name} / {result.tool_name or self._shared_model_name()} / {result.projection_method.upper()}"
            ),
            fontproperties=font_props,
        )
        ax.set_xlabel(tr("debug.embedding.plot.dimension1"))
        ax.set_ylabel(tr("debug.embedding.plot.dimension2"))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        self.figure.tight_layout()
        self.figure_canvas.draw_idle()
