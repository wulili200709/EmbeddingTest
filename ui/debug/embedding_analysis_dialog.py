from __future__ import annotations

import os
from importlib import import_module
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.registry import algorithm_display_name
from ui.i18n import tr

FigureCanvas = None
Figure = None
mpl_font_manager = None
_MATPLOTLIB_IMPORT_ATTEMPTED = False


def _analysis_module():
    return import_module("ui.debug.embedding_analysis")


def _ensure_matplotlib() -> bool:
    global FigureCanvas, Figure, mpl_font_manager, _MATPLOTLIB_IMPORT_ATTEMPTED
    if _MATPLOTLIB_IMPORT_ATTEMPTED:
        return FigureCanvas is not None and Figure is not None
    _MATPLOTLIB_IMPORT_ATTEMPTED = True
    try:
        from matplotlib import font_manager as _font_manager
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FigureCanvas
        from matplotlib.figure import Figure as _Figure
    except Exception:
        FigureCanvas = None
        Figure = None
        mpl_font_manager = None
        return False
    FigureCanvas = _FigureCanvas
    Figure = _Figure
    mpl_font_manager = _font_manager
    return True


class _BootstrapWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        session_root: str,
        initial_product: str,
        initial_backbone: str,
        initial_model_key: str,
        allowed_model_keys: set[str],
        allowed_backbones: set[str],
    ) -> None:
        super().__init__()
        self.session_root = session_root
        self.initial_product = initial_product
        self.initial_backbone = initial_backbone
        self.initial_model_key = initial_model_key
        self.allowed_model_keys = set(allowed_model_keys)
        self.allowed_backbones = set(allowed_backbones)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            module = _analysis_module()
            names = module.list_product_names(self.session_root)
            selected_product = self.initial_product if self.initial_product in names else (names[0] if names else "")
            entries = self._entries_for_product(module, selected_product)
            selected_model_index = self._selected_model_index(entries)
            self.finished.emit({
                "names": names,
                "selected_product": selected_product,
                "entries": entries,
                "selected_model_index": selected_model_index,
            })
        except Exception as exc:
            self.failed.emit(str(exc))

    def _entries_for_product(self, module, product_name: str) -> list[object]:
        if not product_name:
            return []
        product_dir = os.path.join(self.session_root, product_name)
        entries = module.list_available_embedding_models(product_dir)
        if self.allowed_model_keys:
            entries = [entry for entry in entries if getattr(entry, "model_key", "") in self.allowed_model_keys]
        if self.allowed_backbones:
            entries = [entry for entry in entries if getattr(entry, "backbone", "") in self.allowed_backbones]
        return entries

    def _selected_model_index(self, entries: list[object]) -> int:
        initial_backbone = str(self.initial_backbone or "").strip()
        initial_model_key = str(self.initial_model_key or "").strip()
        for index, entry in enumerate(entries):
            if initial_model_key and getattr(entry, "model_key", "") != initial_model_key:
                continue
            if initial_backbone and getattr(entry, "backbone", "") != initial_backbone:
                continue
            return index
        return 0 if entries else -1


class _AnalysisWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)

    def __init__(
        self,
        *,
        request_id: int,
        session_root: str,
        product_name: str,
        backbone: str,
        model_key: str,
        projection_method: str,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.session_root = session_root
        self.product_name = product_name
        self.backbone = backbone
        self.model_key = model_key
        self.projection_method = projection_method

    @QtCore.Slot()
    def run(self) -> None:
        try:
            module = _analysis_module()
            result = module.load_product_analysis(
                session_root=self.session_root,
                product_name=self.product_name,
                backbone=self.backbone,
                model_key=self.model_key,
                projection_method=self.projection_method,
            )
            self.finished.emit(self.request_id, result)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


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
        self._initial_product = str(initial_product or "").strip()
        self._initial_backbone = str(initial_backbone or "").strip()
        self._initial_model_key = str(initial_model_key or "").strip()
        self._bootstrap_thread: Optional[QtCore.QThread] = None
        self._bootstrap_worker: Optional[_BootstrapWorker] = None
        self._analysis_thread: Optional[QtCore.QThread] = None
        self._analysis_worker: Optional[_AnalysisWorker] = None
        self._analysis_request_id = 0
        self._analysis_cache_keys: dict[int, tuple[str, str, str, str, int, int, int]] = {}
        self._pending_refresh = False
        self._bootstrapped = False
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_analysis)

        self._build_ui()
        self.retranslate_ui()
        self._set_controls_enabled(False)
        self.txt_summary.setPlainText(tr("debug.embedding.loading"))
        QtCore.QTimer.singleShot(0, self._start_bootstrap_load)

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
        self._plot_layout = right_l
        self.figure = None
        self.figure_canvas = None
        self.lbl_plot_error = QtWidgets.QLabel()
        self.lbl_plot_error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right_l.addWidget(self.lbl_plot_error, 1)
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

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.cmb_product.setEnabled(enabled)
        self.cmb_model.setEnabled(enabled)
        self.cmb_projection.setEnabled(enabled)
        self.btn_refresh.setEnabled(enabled)

    def _start_bootstrap_load(self) -> None:
        self.txt_summary.setPlainText(tr("debug.embedding.loading"))
        if self.lbl_plot_error is not None:
            self.lbl_plot_error.setText(tr("debug.embedding.loading"))
        thread = QtCore.QThread(self)
        worker = _BootstrapWorker(
            session_root=self.session_root,
            initial_product=self._initial_product,
            initial_backbone=self._initial_backbone,
            initial_model_key=self._initial_model_key,
            allowed_model_keys=self._allowed_model_keys,
            allowed_backbones=self._allowed_backbones,
        )
        worker.moveToThread(thread)
        self._bootstrap_thread = thread
        self._bootstrap_worker = worker
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_bootstrap_finished)
        worker.failed.connect(self._on_bootstrap_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_bootstrap_worker)
        thread.start()

    def _clear_bootstrap_worker(self) -> None:
        self._bootstrap_thread = None
        self._bootstrap_worker = None

    def _on_bootstrap_failed(self, message: str) -> None:
        self._bootstrapped = True
        self._set_controls_enabled(True)
        self._show_error(message)

    def _on_bootstrap_finished(self, payload: object) -> None:
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        names = list(data.get("names", []) or [])
        selected_product = str(data.get("selected_product", "") or "")
        entries = list(data.get("entries", []) or [])
        selected_model_index = int(data.get("selected_model_index", -1))
        self.cmb_product.blockSignals(True)
        self.cmb_product.clear()
        self.cmb_product.addItems(names)
        if selected_product and selected_product in names:
            self.cmb_product.setCurrentText(selected_product)
        elif names:
            self.cmb_product.setCurrentIndex(0)
        self.cmb_product.blockSignals(False)
        self._populate_models(entries, selected_model_index)
        self._bootstrapped = True
        self._set_controls_enabled(True)
        QtCore.QTimer.singleShot(0, self._refresh_analysis)

    def _load_models(self, initial_backbone: str = "", initial_model_key: str = "") -> None:
        product_name = self.cmb_product.currentText().strip()
        product_dir = os.path.join(self.session_root, product_name)
        entries = _analysis_module().list_available_embedding_models(product_dir)
        if self._allowed_model_keys:
            entries = [entry for entry in entries if getattr(entry, "model_key", "") in self._allowed_model_keys]
        if self._allowed_backbones:
            entries = [entry for entry in entries if getattr(entry, "backbone", "") in self._allowed_backbones]
        initial_backbone = str(initial_backbone or "").strip()
        initial_model_key = str(initial_model_key or "").strip()
        selected_index = 0 if entries else -1
        if entries:
            for index, entry in enumerate(entries):
                if initial_model_key and getattr(entry, "model_key", "") != initial_model_key:
                    continue
                if initial_backbone and getattr(entry, "backbone", "") != initial_backbone:
                    continue
                selected_index = index
                break
        self._populate_models(entries, selected_index)

    def _populate_models(self, entries: list[object], selected_index: int) -> None:
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        for entry in entries:
            self.cmb_model.addItem(str(getattr(entry, "display_name", "")), entry)
        if entries and selected_index >= 0:
            self.cmb_model.setCurrentIndex(min(selected_index, len(entries) - 1))
        self.cmb_model.blockSignals(False)

    def _on_product_changed(self, _product_name: str) -> None:
        if not self._bootstrapped:
            return
        self._load_models()
        self._schedule_refresh_analysis()

    def _schedule_refresh_analysis(self) -> None:
        self._refresh_timer.start(50)

    def _current_model_entry(self) -> Optional[object]:
        entry = self.cmb_model.currentData()
        if entry is not None and hasattr(entry, "backbone") and hasattr(entry, "model_path"):
            return entry
        return None

    def _analysis_cache_key(
        self,
        product_name: str,
        entry: object,
        projection: str,
    ) -> tuple[str, str, str, str, int, int, int]:
        product_dir = os.path.join(self.session_root, product_name)
        session_file = os.path.join(product_dir, "session.json")
        params_file = os.path.join(product_dir, "product_params.json")
        model_path = str(getattr(entry, "model_path", "") or "")
        model_mtime = int(os.path.getmtime(model_path) * 1000) if os.path.exists(model_path) else -1
        session_mtime = int(os.path.getmtime(session_file) * 1000) if os.path.exists(session_file) else -1
        params_mtime = int(os.path.getmtime(params_file) * 1000) if os.path.exists(params_file) else -1
        return (
            product_name,
            str(getattr(entry, "backbone", "")),
            str(getattr(entry, "model_key", "")),
            str(projection or "tsne").strip().lower(),
            model_mtime,
            session_mtime,
            params_mtime,
        )

    def _refresh_analysis(self) -> None:
        if not self._bootstrapped:
            self._pending_refresh = True
            return
        if self._analysis_thread is not None:
            self._pending_refresh = True
            return
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
        result = self._analysis_cache.get(cache_key)
        if result is not None:
            self._result = result
            self._render_result(result)
            return

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._analysis_cache_keys[request_id] = cache_key
        self._set_controls_enabled(False)
        self.txt_summary.setPlainText(tr("debug.embedding.loading"))
        if self.lbl_plot_error is not None and self.figure_canvas is None:
            self.lbl_plot_error.setText(tr("debug.embedding.loading"))

        thread = QtCore.QThread(self)
        worker = _AnalysisWorker(
            request_id=request_id,
            session_root=self.session_root,
            product_name=product_name,
            backbone=str(getattr(entry, "backbone", "")),
            model_key=str(getattr(entry, "model_key", "")),
            projection_method=projection,
        )
        worker.moveToThread(thread)
        self._analysis_thread = thread
        self._analysis_worker = worker
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_analysis_finished)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_analysis_worker)
        thread.start()

    def _clear_analysis_worker(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None
        self._set_controls_enabled(True)
        if self._pending_refresh:
            self._pending_refresh = False
            QtCore.QTimer.singleShot(0, self._refresh_analysis)

    def _on_analysis_finished(self, request_id: object, result: object) -> None:
        request_id_int = int(request_id)
        if request_id_int != self._analysis_request_id:
            return
        cache_key = self._analysis_cache_keys.pop(request_id_int, None)
        if cache_key is not None:
            self._analysis_cache[cache_key] = result
        self._result = result
        self._render_result(result)

    def _on_analysis_failed(self, request_id: object, message: str) -> None:
        if int(request_id) != self._analysis_request_id:
            return
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        self._result = None
        self.lbl_model_path.setText("-")
        self.lbl_session_path.setText("-")
        self.txt_summary.setPlainText(message)
        self.table.setRowCount(0)
        if self.figure is not None and self.figure_canvas is not None:
            self.figure.clear()
            self.figure_canvas.draw_idle()
        elif self.lbl_plot_error is not None:
            self.lbl_plot_error.setText(message)

    def _ensure_plot_canvas(self) -> bool:
        if self.figure is not None and self.figure_canvas is not None:
            return True
        if not _ensure_matplotlib():
            if self.lbl_plot_error is not None:
                self.lbl_plot_error.setText(tr("debug.embedding.matplotlib_unavailable"))
            return False
        self.figure = Figure(figsize=(6.0, 5.0))
        self.figure_canvas = FigureCanvas(self.figure)
        if self.lbl_plot_error is not None:
            self._plot_layout.removeWidget(self.lbl_plot_error)
            self.lbl_plot_error.deleteLater()
            self.lbl_plot_error = None
        self._plot_layout.addWidget(self.figure_canvas, 1)
        return True

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
        if not self._ensure_plot_canvas():
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

    def closeEvent(self, event) -> None:
        if (
            (self._bootstrap_thread is not None and self._bootstrap_thread.isRunning())
            or (self._analysis_thread is not None and self._analysis_thread.isRunning())
        ):
            self.txt_summary.setPlainText(tr("debug.embedding.loading"))
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if (
            (self._bootstrap_thread is not None and self._bootstrap_thread.isRunning())
            or (self._analysis_thread is not None and self._analysis_thread.isRunning())
        ):
            self.txt_summary.setPlainText(tr("debug.embedding.loading"))
            return
        super().reject()
