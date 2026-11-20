"""Explicit model sharing controls for the selected inspection ROI."""

from __future__ import annotations

from copy import copy

from PySide6 import QtCore, QtWidgets

from common.algorithm_codes import normalize_tool_algorithm_code
from common.model_groups import OrderModelGroups
from ui.i18n import tr


def model_source(item) -> str:
    return str(getattr(item, "model_source", "independent") or "independent")


def binding_description(owner, item) -> dict:
    try:
        return dict(owner.algo.describe_model_binding(item, owner.session.product_dir))
    except Exception as exc:
        return {"ready": False, "message": str(exc), "source": model_source(item)}


def binding_status(owner, item) -> tuple[str, str, str]:
    description = binding_description(owner, item)
    source = model_source(item)
    ready = bool(description.get("ready"))
    if source == "library":
        name = str(description.get("name") or getattr(item, "library_model_id", ""))
        version = str(description.get("version") or getattr(item, "library_version", ""))
        text = tr("debug.model_source.library_status", name=name, version=version)
    elif source == "group":
        key = "debug.model_source.group_status" if ready else "debug.model_source.group_untrained_status"
        text = tr(key, group=getattr(item, "task_group", ""))
    else:
        text = tr("debug.status.trained") if ready else tr("debug.status.untrained")
    if not ready and source == "library":
        text = tr("debug.model_source.unavailable_status", source=text)
    detail = str(description.get("message") or "")
    path = str(description.get("path") or "")
    return text, "\n".join(part for part in (text, detail, path) if part), "#79d279" if ready else "#d98c8c"


def training_sample_summary(owner, item) -> tuple[str, list, list, list, int, int, int]:
    """Use the same ROI group as the training task builder."""
    builder = owner.training_controller.task_builder
    ok_files, ng_files, candidates = builder.sample_groups_for_item(item)
    peers = builder.group_items_for_item(item)
    role = str(getattr(item, "camera_id", "") or owner.current_camera_role())
    ok_count = ng_count = unset_count = 0
    for peer in peers:
        label = str(getattr(peer, "roi_label", "") or "roi")
        ok, ng, unset = owner._sample_annotation_counts_for_roi(label, role, paths=candidates)
        ok_count += ok
        ng_count += ng
        unset_count += unset
    label = str(getattr(item, "task_group", "")) if model_source(item) == "group" else str(item.roi_label or "roi")
    return label, ok_files, ng_files, candidates, ok_count, ng_count, unset_count


def open_model_source_dialog(owner) -> None:
    """Open the selected ROI settings without moving the tools viewport."""
    item = owner._selected_inspection_item()
    if item is None:
        QtWidgets.QMessageBox.information(owner, tr("debug.model_source.title"), tr("debug.model_source.select_item"))
        return
    if owner.algo.is_measurement_tool(item.algorithm_code):
        QtWidgets.QMessageBox.information(owner, tr("debug.model_source.title"), tr("debug.measurement.no_training"))
        return
    dialog = QtWidgets.QDialog(owner)
    dialog.setObjectName("model_source_dialog")
    dialog.setWindowTitle(tr("debug.model_source.title"))
    layout = QtWidgets.QVBoxLayout(dialog)
    selected_label = QtWidgets.QLabel(tr(
        "debug.model_source.current_item",
        name=str(item.display_name or item.roi_label or item.item_id),
        roi=str(item.roi_label or item.item_id),
        camera=str(item.camera_id),
    ))
    selected_label.setObjectName("model_source_current_item")
    selected_label.setWordWrap(True)
    layout.addWidget(selected_label)
    inline_panel = getattr(owner, "model_source_panel", None)
    panel = ModelSourcePanel(
        owner, layout,
        input_style=getattr(inline_panel, "_input_style", ""),
        button_style=getattr(inline_panel, "_button_style", ""),
    )
    panel.prompt_parent = dialog
    dialog.model_source_panel = panel
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setText(tr("common.close"))
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(460)
    dialog.resize(520, dialog.sizeHint().height())
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


class ModelSourcePanel:
    def __init__(self, owner, layout, *, input_style: str = "", button_style: str = "", section_button=None) -> None:
        self.owner = owner
        self.section_button = section_button
        self.prompt_parent = owner
        self._input_style = input_style
        self._button_style = button_style
        self._loading = False
        self._models: list[dict] = []
        self.frame = QtWidgets.QGroupBox()
        if section_button is not None:
            self.frame.setStyleSheet("QGroupBox{border:none;}")
        form = QtWidgets.QFormLayout(self.frame)
        form.setContentsMargins(8, 6, 8, 6)
        self.mode = QtWidgets.QComboBox()
        self.group = QtWidgets.QComboBox()
        self.group.setEditable(True)
        self.group.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.library = QtWidgets.QComboBox()
        self.mode_label = QtWidgets.QLabel()
        self.group_label = QtWidgets.QLabel()
        self.library_label = QtWidgets.QLabel()
        form.addRow(self.mode_label, self.mode)
        form.addRow(self.group_label, self.group)
        form.addRow(self.library_label, self.library)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        row = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton()
        self.publish_button = QtWidgets.QPushButton()
        self.refresh_button = QtWidgets.QPushButton()
        for widget in (self.mode, self.group, self.library):
            widget.setStyleSheet(input_style)
            widget.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(12)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        for button in (self.apply_button, self.publish_button, self.refresh_button):
            button.setStyleSheet(button_style)
            row.addWidget(button)
        form.addRow(row)
        layout.addWidget(self.frame)
        self.mode.currentIndexChanged.connect(self._sync_mode)
        self.apply_button.clicked.connect(self.apply)
        self.publish_button.clicked.connect(self.publish)
        self.refresh_button.clicked.connect(self.reload_library)
        if section_button is not None:
            section_button.toggled.connect(self.toggle_section)
        self.retranslate()
        self.reload_library()

    def _selected(self):
        return self.owner._selected_inspection_item()

    def _algorithm(self, item) -> str:
        resolver = getattr(self.owner.algo, "resolve_item_algorithm", None)
        if callable(resolver):
            try:
                return str(resolver(item) or "")
            except Exception:
                return ""
        if self.owner.algo.is_learning_tool(item.algorithm_code):
            return str(self.owner.algo.current_learning_backbone(item.camera_id) or "")
        return str(self.owner.algo.resolve_tool_algorithm(item.algorithm_code, item.camera_id) or "")

    def _has_permission(self, key: str) -> bool:
        checker = getattr(self.owner.window(), "_has_permission", None)
        return bool(checker(key)) if callable(checker) else True

    def retranslate(self) -> None:
        current = self.mode.currentData() or "independent"
        with QtCore.QSignalBlocker(self.mode):
            self.mode.clear()
            for source in ("independent", "group", "library"):
                self.mode.addItem(tr(f"debug.model_source.{source}"), source)
            self.mode.setCurrentIndex(max(0, self.mode.findData(current)))
        self.frame.setTitle(tr("debug.model_source.title") if self.section_button is None else "")
        if self.section_button is not None:
            self.section_button.setText(tr("debug.model_source.title"))
        entry_button = getattr(self.owner, "btn_model_source", None)
        if entry_button is not None:
            entry_button.setText(tr("debug.model_source.open"))
            entry_button.setToolTip(tr("debug.model_source.open_tip"))
        self.mode_label.setText(tr("debug.model_source.mode"))
        self.group_label.setText(tr("debug.model_source.group_label"))
        self.library_label.setText(tr("debug.model_source.library_label"))
        self.group.setToolTip(tr("debug.model_source.group_tip"))
        self.library.setToolTip(tr("debug.model_source.library_tip"))
        self.apply_button.setText(tr("debug.model_source.apply"))
        self.publish_button.setText(tr("debug.model_source.publish"))
        self.refresh_button.setText(tr("debug.model_source.refresh"))
        self.refresh()

    def reload_library(self) -> None:
        try:
            self._models = list(self.owner.algo.model_library.list_models())
        except Exception as exc:
            self._models = []
            self.status.setText(str(exc))
        self.refresh()

    def refresh(self) -> None:
        item = self._selected()
        measurement = item is not None and self.owner.algo.is_measurement_tool(item.algorithm_code)
        self._sync_section_visibility(measurement)
        entry_button = getattr(self.owner, "btn_model_source", None)
        if entry_button is not None:
            entry_button.setEnabled(not measurement and not getattr(self.owner, "_training_in_progress", False))
            entry_button.setToolTip(tr("debug.measurement.no_training") if measurement else tr("debug.model_source.open_tip"))
        self.frame.setEnabled(item is not None and not getattr(self.owner, "_training_in_progress", False))
        if item is None or measurement:
            self.status.setText(tr("debug.model_source.select_item"))
            return
        self._loading = True
        try:
            source = model_source(item)
            self.mode.setCurrentIndex(max(0, self.mode.findData(source)))
            self.group.clear()
            groups = {
                str(peer.task_group) for peer in self.owner.inspection_items
                if model_source(peer) == "group" and peer.camera_id == item.camera_id
                and self._algorithm(peer) == self._algorithm(item)
            }
            groups.update(record["task_group"] for record in OrderModelGroups(
                self.owner.session.product_dir
            ).list_groups(item.camera_id, public_models=self._models))
            self.group.addItems(sorted(groups))
            self.group.setCurrentText(str(getattr(item, "task_group", "") or ""))
            self.library.clear()
            self.library.addItem(tr("debug.model_source.choose_library"), None)
            for model in self._models:
                if (model.get("kind") == "embedding") != self.owner.algo.is_learning_tool(item.algorithm_code):
                    continue
                key = (str(model["model_id"]), str(model["version"]))
                self.library.addItem(f"{model['name']} / {model['version']}", key)
                self.library.setItemData(self.library.count() - 1, str(model["model_id"]), QtCore.Qt.ItemDataRole.ToolTipRole)
                if key == (getattr(item, "library_model_id", ""), str(getattr(item, "library_version", ""))):
                    self.library.setCurrentIndex(self.library.count() - 1)
            if source == "library" and self.library.currentData() is None:
                key = (item.library_model_id, str(item.library_version))
                self.library.addItem(tr("debug.model_source.missing_library", model=key[0], version=key[1]), key)
                self.library.setCurrentIndex(self.library.count() - 1)
            text, detail, color = binding_status(self.owner, item)
            self.status.setText(text)
            self.status.setToolTip(detail)
            self.status.setStyleSheet(f"color:{color};")
            self.apply_button.setEnabled(self._has_permission("inspection.edit_items"))
            self.mode.setEnabled(self._has_permission("inspection.edit_items"))
            description = binding_description(self.owner, item)
            self.publish_button.setEnabled(source != "library" and bool(description.get("ready")) and self._has_permission("model.train"))
        finally:
            self._loading = False
        self._sync_mode()

    def toggle_section(self, _checked: bool) -> None:
        item = self._selected()
        measurement = item is not None and self.owner.algo.is_measurement_tool(item.algorithm_code)
        self._sync_section_visibility(measurement)

    def _sync_section_visibility(self, measurement: bool) -> None:
        expanded = True
        if self.section_button is not None:
            expanded = self.section_button.isChecked()
            self.section_button.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
            self.section_button.setEnabled(not measurement)
            self.section_button.setToolTip(tr("debug.measurement.no_training") if measurement else "")
        self.frame.setVisible(expanded and not measurement)

    def _sync_mode(self, *_args) -> None:
        source = self.mode.currentData()
        if source == "group" and not self.group.currentText().strip() and self.group.count() == 1:
            self.group.setCurrentIndex(0)
        for widget in (self.group, self.group_label):
            widget.setVisible(source == "group")
        for widget in (self.library, self.library_label):
            widget.setVisible(source == "library")
        self.group.setEnabled(self._has_permission("inspection.edit_items"))
        self.library.setEnabled(self._has_permission("inspection.edit_items"))

    def has_pending_changes(self) -> bool:
        item = self._selected()
        if item is None or self._loading or self.owner.algo.is_measurement_tool(item.algorithm_code):
            return False
        source = str(self.mode.currentData() or "independent")
        if source != model_source(item):
            return True
        if source == "group":
            return self.group.currentText().strip() != str(item.task_group or "").strip()
        if source == "library":
            return self.library.currentData() != (item.library_model_id, item.library_version)
        return False

    def apply_pending(self) -> bool:
        """Training must use the visible choice, never an older saved source."""
        return self.apply() if self.has_pending_changes() else True

    def apply(self) -> bool:
        item = self._selected()
        if item is None or self._loading or getattr(self.owner, "_training_in_progress", False):
            return False
        require = getattr(self.owner.window(), "_require_permission", None)
        if callable(require) and not require("inspection.edit_items", tr("debug.model_source.apply")):
            return False
        candidate = copy(item)
        candidate.model_source = str(self.mode.currentData() or "independent")
        candidate.task_group = self.group.currentText().strip() if candidate.model_source == "group" else ""
        candidate.library_model_id = candidate.library_version = ""
        try:
            if candidate.model_source == "group":
                if not candidate.task_group:
                    raise ValueError(tr("debug.model_source.group_required"))
                for peer in self.owner.inspection_items:
                    if peer is not item and model_source(peer) == "group" and peer.camera_id == item.camera_id and peer.effective_model_key == candidate.effective_model_key and self._algorithm(peer) != self._algorithm(item):
                        raise ValueError(tr("debug.model_source.group_incompatible"))
            elif candidate.model_source == "library":
                selected = self.library.currentData()
                if not selected:
                    raise ValueError(tr("debug.model_source.library_required"))
                candidate.library_model_id, candidate.library_version = selected
                manifest = self.owner.algo.model_library.resolve(*selected)
                candidate.algorithm_code = normalize_tool_algorithm_code(manifest["algorithm"])
            candidate.validate_model_binding()
            before = {key: getattr(item, key) for key in ("model_source", "task_group", "library_model_id", "library_version", "algorithm_code")}
            for key in before:
                setattr(item, key, getattr(candidate, key))
            self.owner._persist_inspection_items()
        except Exception as exc:
            if "before" in locals():
                for key, value in before.items():
                    setattr(item, key, value)
            QtWidgets.QMessageBox.warning(self.prompt_parent, tr("debug.model_source.title"), str(exc))
            return False
        self.owner.algo.model = None
        self.owner.algo._loaded_embedding_model_key = ("", "")
        self.owner.algo.clear_embedding_model_cache()
        self.owner._clear_training_roi_review_state(item.camera_id)
        self.owner._refresh_inspection_items_table()
        self.owner._refresh_lists()
        self.owner._update_runtime_widgets()
        audit = getattr(self.owner.window(), "_audit_event", None)
        if callable(audit):
            audit(module="检测项", action="修改模型来源", target=item.item_id, before_value=str(before), after_value=str({key: getattr(item, key) for key in before}))
        self.refresh()
        return True

    def publish(self) -> None:
        item = self._selected()
        if item is None or model_source(item) == "library" or getattr(self.owner, "_training_in_progress", False):
            return
        require = getattr(self.owner.window(), "_require_permission", None)
        if callable(require) and not require("model.train", tr("debug.model_source.publish")):
            return
        latest: dict[str, dict] = {}
        for model in self._models:
            if model.get("algorithm") == self._algorithm(item):
                latest.setdefault(str(model["model_id"]), model)
        options = [tr("debug.model_source.publish_new")]
        existing = list(latest.values())
        options.extend(f"{model['name']} / {str(model['model_id'])[:8]}" for model in existing)
        target, accepted = QtWidgets.QInputDialog.getItem(self.prompt_parent, tr("debug.model_source.publish"), tr("debug.model_source.publish_target"), options, 0, False)
        if not accepted:
            return
        index = options.index(target)
        model_id = str(existing[index - 1]["model_id"]) if index else None
        default_name = str(existing[index - 1]["name"]) if index else str(item.task_group or item.display_name or item.roi_label)
        name, accepted = QtWidgets.QInputDialog.getText(self.prompt_parent, tr("debug.model_source.publish"), tr("debug.model_source.publish_name"), text=default_name)
        if not accepted or not name.strip():
            return
        try:
            manifest = self.owner.algo.publish_model_to_library(item, self.owner.session.product_dir, name.strip(), model_id=model_id)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.prompt_parent, tr("debug.model_source.publish"), str(exc))
            return
        self.reload_library()
        QtWidgets.QMessageBox.information(self.prompt_parent, tr("debug.model_source.publish"), tr("debug.model_source.published", name=manifest["name"], version=manifest["version"]))
