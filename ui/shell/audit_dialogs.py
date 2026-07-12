from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from infrastructure.audit_store import (
    PERMISSION_LABELS,
    PERMISSION_MODULES,
    AuditStore,
    RuntimeRecordStore,
)
from ui.i18n import tr


ROLE_LABEL_KEYS = {
    "operator": "role.operator",
    "admin": "role.admin",
    "super_admin": "role.super_admin",
}

PERMISSION_MODULE_KEYS = {
    "product.select": "permission.module.product",
    "product.create": "permission.module.product",
    "product.delete": "permission.module.product",
    "runtime.run": "permission.module.runtime",
    "runtime.connect_camera": "permission.module.runtime",
    "runtime.release_ng": "permission.module.runtime",
    "io.debug": "permission.module.hardware",
    "camera.edit_params": "permission.module.camera",
    "template.edit_roi": "permission.module.template",
    "template.edit_params": "permission.module.template",
    "sample.manage": "permission.module.sample",
    "inspection.edit_items": "permission.module.inspection",
    "inspection.edit_limits": "permission.module.inspection",
    "model.train": "permission.module.model",
    "settings.tower_light": "permission.module.settings",
    "settings.record_path": "permission.module.settings",
    "settings.passwords": "permission.module.settings",
    "audit.view": "permission.module.audit",
    "audit.export": "permission.module.audit",
    "runtime_records.view": "permission.module.runtime_records",
    "runtime_records.export": "permission.module.runtime_records",
    "user.manage": "permission.module.user",
    "software.version_log": "permission.module.software",
}


def _role_label(role_key: object, fallback: object = "") -> str:
    key = str(role_key or "").strip()
    if key in ROLE_LABEL_KEYS:
        return tr(ROLE_LABEL_KEYS[key])
    return str(fallback or key)


def _permission_module_label(permission_key: str) -> str:
    i18n_key = PERMISSION_MODULE_KEYS.get(str(permission_key or "").strip())
    if i18n_key:
        return tr(i18n_key)
    return str(PERMISSION_MODULES.get(permission_key, ""))


def _permission_label(permission_key: str) -> str:
    key = str(permission_key or "").strip()
    text = tr(f"permission.{key}")
    return text if text != f"permission.{key}" else str(PERMISSION_LABELS.get(key, key))


def _translate_dialog_buttons(buttons: QtWidgets.QDialogButtonBox) -> None:
    mapping = {
        QtWidgets.QDialogButtonBox.StandardButton.Ok: "common.ok",
        QtWidgets.QDialogButtonBox.StandardButton.Cancel: "common.cancel",
        QtWidgets.QDialogButtonBox.StandardButton.Close: "common.close",
    }
    for standard_button, text_key in mapping.items():
        button = buttons.button(standard_button)
        if button is not None:
            button.setText(tr(text_key))


def _new_filter_combo(placeholder: str) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
    combo.setMinimumWidth(180)
    combo.addItem("")
    line_edit = combo.lineEdit()
    if line_edit is not None:
        line_edit.setPlaceholderText(placeholder)
    return combo


def _combo_filter_text(combo: QtWidgets.QComboBox) -> str:
    return str(combo.currentText() or "").strip()


def _new_audit_time_editor(date_time: QtCore.QDateTime) -> QtWidgets.QDateTimeEdit:
    editor = QtWidgets.QDateTimeEdit(date_time)
    editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
    editor.setCalendarPopup(True)
    editor.setEnabled(False)
    editor.setMinimumWidth(200)
    return editor


def _apply_dark_dialog_style(dialog: QtWidgets.QDialog) -> None:
    dialog.setStyleSheet(
        "QDialog{background:#2d2d2d;color:#e0e0e0;}"
        "QWidget{color:#e0e0e0;}"
        "QLabel{color:#d0d0d0;font-size:12px;}"
        "QTabWidget::pane{background:#2d2d2d;border:1px solid #505050;top:-1px;}"
        "QTabBar::tab{background:#3a3a3a;color:#d0d0d0;border:1px solid #505050;"
        "border-bottom:none;padding:6px 14px;min-width:72px;}"
        "QTabBar::tab:selected{background:#404040;color:#ffffff;border-top:2px solid #3794ff;}"
        "QTabBar::tab:hover{background:#454545;}"
        "QLineEdit,QComboBox,QDateTimeEdit,QPlainTextEdit{background:#404040;color:#e0e0e0;"
        "border:1px solid #5a5a5a;padding:5px 6px;border-radius:3px;"
        "selection-background-color:#3794ff;}"
        "QLineEdit:focus,QComboBox:focus,QDateTimeEdit:focus,QPlainTextEdit:focus{border:1px solid #3794ff;}"
        "QDateTimeEdit:disabled{background:#353535;color:#7a7a7a;border:1px solid #484848;}"
        "QComboBox::drop-down{border-left:1px solid #5a5a5a;width:22px;}"
        "QComboBox QAbstractItemView{background:#3a3a3a;color:#e0e0e0;"
        "selection-background-color:#3794ff;selection-color:#ffffff;border:1px solid #505050;}"
        "QTableWidget,QListWidget{background:#333333;alternate-background-color:#383838;"
        "color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;"
        "selection-background-color:#6ec0ff;selection-color:#1a1a1a;}"
        "QTableWidget::item,QListWidget::item{padding:4px;border:none;}"
        "QTableWidget::item:selected,QListWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
        "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:5px;}"
        "QTableCornerButton::section{background:#3a3a3a;border:1px solid #404040;}"
        "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
        "padding:5px 14px;border-radius:3px;min-height:22px;}"
        "QPushButton:hover{background:#505050;}"
        "QPushButton:pressed{background:#3d3d40;}"
        "QPushButton:disabled{background:#353535;color:#7a7a7a;border:1px solid #484848;}"
        "QCheckBox{color:#d0d0d0;spacing:6px;}"
        "QCheckBox::indicator{width:14px;height:14px;}"
        "QScrollBar:vertical{background:#2d2d2d;width:12px;margin:0;}"
        "QScrollBar::handle:vertical{background:#5a5a5a;border-radius:5px;min-height:24px;}"
        "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        "QScrollBar:horizontal{background:#2d2d2d;height:12px;margin:0;}"
        "QScrollBar::handle:horizontal{background:#5a5a5a;border-radius:5px;min-width:24px;}"
        "QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0;}"
    )


def _configure_table(table: QtWidgets.QTableWidget) -> None:
    table.setAlternatingRowColors(True)
    table.setShowGrid(True)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(28)
    table.horizontalHeader().setDefaultAlignment(
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
    )


class LoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("auth.login.title"))
        self.setMinimumWidth(320)
        _apply_dark_dialog_style(self)
        layout = QtWidgets.QFormLayout(self)
        self.edit_user = QtWidgets.QLineEdit()
        self.edit_user.setPlaceholderText(tr("auth.user_name"))
        self.edit_password = QtWidgets.QLineEdit()
        self.edit_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_password.setPlaceholderText(tr("auth.password"))
        layout.addRow(tr("auth.user_name"), self.edit_user)
        layout.addRow(tr("auth.password"), self.edit_password)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        _translate_dialog_buttons(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.edit_user.setFocus()

    def credentials(self) -> tuple[str, str]:
        return self.edit_user.text().strip(), self.edit_password.text()


class ChangePasswordDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or tr("auth.change_password.title"))
        self.setMinimumWidth(340)
        _apply_dark_dialog_style(self)
        layout = QtWidgets.QFormLayout(self)
        self.edit_password = QtWidgets.QLineEdit()
        self.edit_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_confirm = QtWidgets.QLineEdit()
        self.edit_confirm.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        layout.addRow(tr("auth.new_password"), self.edit_password)
        layout.addRow(tr("auth.confirm_password"), self.edit_confirm)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        _translate_dialog_buttons(buttons)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _accept_if_valid(self) -> None:
        password = self.edit_password.text()
        if len(password.strip()) < 4:
            QtWidgets.QMessageBox.warning(self, tr("auth.change_password.title"), tr("auth.password_min_length"))
            return
        if password != self.edit_confirm.text():
            QtWidgets.QMessageBox.warning(self, tr("auth.change_password.title"), tr("auth.password_mismatch"))
            return
        self.accept()

    def password(self) -> str:
        return self.edit_password.text()


class UserPermissionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        store: AuditStore,
        *,
        can_manage_runtime_records: bool = True,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.can_manage_runtime_records = bool(can_manage_runtime_records)
        self.setWindowTitle(tr("auth.user_permissions.title"))
        self.resize(760, 520)
        _apply_dark_dialog_style(self)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        root.addWidget(tabs, 1)

        self.tab_users = QtWidgets.QWidget()
        self.tab_roles = QtWidgets.QWidget()
        tabs.addTab(self.tab_users, tr("auth.tab.users"))
        tabs.addTab(self.tab_roles, tr("auth.tab.role_permissions"))
        self._build_users_tab()
        self._build_roles_tab()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        _translate_dialog_buttons(buttons)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._reload_roles()
        self._reload_users()

    def _build_users_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_users)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.table_users = QtWidgets.QTableWidget(0, 6)
        self.table_users.setHorizontalHeaderLabels([
            "ID",
            tr("auth.user_name"),
            tr("auth.role"),
            tr("auth.enabled"),
            tr("auth.must_change_password"),
            tr("auth.last_login"),
        ])
        _configure_table(self.table_users)
        self.table_users.setColumnHidden(0, True)
        self.table_users.horizontalHeader().setStretchLastSection(True)
        self.table_users.itemSelectionChanged.connect(self._sync_selected_user_to_form)
        layout.addWidget(self.table_users, 1)

        form = QtWidgets.QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.edit_new_user = QtWidgets.QLineEdit()
        self.edit_new_user.setPlaceholderText(tr("auth.new_user_placeholder"))
        self.edit_new_password = QtWidgets.QLineEdit()
        self.edit_new_password.setPlaceholderText(tr("auth.initial_password_placeholder"))
        self.edit_new_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.cmb_user_role = QtWidgets.QComboBox()
        self.chk_user_enabled = QtWidgets.QCheckBox(tr("auth.enabled"))
        self.chk_user_enabled.setChecked(True)
        form.addWidget(QtWidgets.QLabel(tr("auth.user_name")), 0, 0)
        form.addWidget(self.edit_new_user, 0, 1)
        form.addWidget(QtWidgets.QLabel(tr("auth.password")), 0, 2)
        form.addWidget(self.edit_new_password, 0, 3)
        form.addWidget(QtWidgets.QLabel(tr("auth.role")), 1, 0)
        form.addWidget(self.cmb_user_role, 1, 1)
        form.addWidget(self.chk_user_enabled, 1, 2)
        btn_add = QtWidgets.QPushButton(tr("auth.add_user"))
        btn_save = QtWidgets.QPushButton(tr("auth.save_selected_user"))
        btn_reset = QtWidgets.QPushButton(tr("auth.reset_selected_password"))
        btn_add.clicked.connect(self._add_user)
        btn_save.clicked.connect(self._save_selected_user)
        btn_reset.clicked.connect(self._reset_selected_password)
        form.addWidget(btn_add, 2, 1)
        form.addWidget(btn_save, 2, 2)
        form.addWidget(btn_reset, 2, 3)
        layout.addLayout(form)

    def _build_roles_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_roles)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel(tr("auth.role")))
        self.cmb_role = QtWidgets.QComboBox()
        self.cmb_role.currentIndexChanged.connect(self._load_role_permissions)
        row.addWidget(self.cmb_role)
        row.addStretch(1)
        layout.addLayout(row)
        self.list_permissions = QtWidgets.QListWidget()
        self.list_permissions.setAlternatingRowColors(True)
        layout.addWidget(self.list_permissions, 1)
        btn_save = QtWidgets.QPushButton(tr("auth.save_role_permissions"))
        btn_save.clicked.connect(self._save_role_permissions)
        layout.addWidget(btn_save, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    def _reload_roles(self) -> None:
        roles = self.store.roles()
        for combo in (self.cmb_user_role, self.cmb_role):
            combo.blockSignals(True)
            combo.clear()
            for role in roles:
                combo.addItem(
                    _role_label(role.get("role_key"), role.get("role_name", "")),
                    str(role["role_key"]),
                )
            combo.blockSignals(False)
        self._load_role_permissions()

    def _reload_users(self) -> None:
        users = self.store.users()
        self.table_users.setRowCount(len(users))
        for row, user in enumerate(users):
            values = [
                user.get("id", ""),
                user.get("user_name", ""),
                _role_label(user.get("role_key"), user.get("role_name", "")),
                tr("common.yes") if user.get("enabled") else tr("common.no"),
                tr("common.yes") if user.get("must_change_password") else tr("common.no"),
                user.get("last_login_at", "") or "",
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if col == 2:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(user.get("role_key", "") or ""))
                self.table_users.setItem(row, col, item)
        self.table_users.resizeColumnsToContents()

    def _selected_user_id(self) -> int | None:
        row = self.table_users.currentRow()
        if row < 0:
            return None
        item = self.table_users.item(row, 0)
        try:
            return int(item.text()) if item is not None else None
        except Exception:
            return None

    def _sync_selected_user_to_form(self) -> None:
        row = self.table_users.currentRow()
        if row < 0:
            return
        name_item = self.table_users.item(row, 1)
        role_item = self.table_users.item(row, 2)
        enabled_item = self.table_users.item(row, 3)
        if name_item is not None:
            self.edit_new_user.setText(name_item.text())
        if role_item is not None:
            role_key = str(role_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            idx = self.cmb_user_role.findData(role_key) if role_key else self.cmb_user_role.findText(role_item.text())
            if idx >= 0:
                self.cmb_user_role.setCurrentIndex(idx)
        if enabled_item is not None:
            self.chk_user_enabled.setChecked(enabled_item.text() == tr("common.yes"))

    def _add_user(self) -> None:
        try:
            self.store.create_user(
                self.edit_new_user.text(),
                self.edit_new_password.text(),
                str(self.cmb_user_role.currentData() or "admin"),
                enabled=self.chk_user_enabled.isChecked(),
                must_change_password=False,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("auth.add_user"), str(exc))
            return
        self.edit_new_user.clear()
        self.edit_new_password.clear()
        self._reload_users()

    def _save_selected_user(self) -> None:
        user_id = self._selected_user_id()
        if user_id is None:
            return
        try:
            self.store.update_user(
                user_id,
                role_key=str(self.cmb_user_role.currentData() or "admin"),
                enabled=self.chk_user_enabled.isChecked(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("auth.save_selected_user"), str(exc))
            return
        self._reload_users()

    def _reset_selected_password(self) -> None:
        user_id = self._selected_user_id()
        if user_id is None:
            return
        dialog = ChangePasswordDialog(self, title=tr("auth.reset_password"))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            self.store.set_user_password(user_id, dialog.password(), must_change_password=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("auth.reset_password"), str(exc))
            return
        self._reload_users()

    def _load_role_permissions(self) -> None:
        role_key = str(self.cmb_role.currentData() or "")
        selected = self.store.role_permissions(role_key) if role_key else set()
        self.list_permissions.clear()
        for key in sorted(
            PERMISSION_LABELS,
            key=lambda item: (_permission_module_label(item), _permission_label(item)),
        ):
            label = f"{_permission_module_label(key)} - {_permission_label(key)} ({key})"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if key in selected else QtCore.Qt.CheckState.Unchecked
            )
            if role_key == "super_admin":
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
            elif (
                key in {"runtime_records.view", "runtime_records.export"}
                and not self.can_manage_runtime_records
            ):
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.list_permissions.addItem(item)

    def _save_role_permissions(self) -> None:
        role_key = str(self.cmb_role.currentData() or "")
        if not role_key or role_key == "super_admin":
            return
        permissions: list[str] = []
        for index in range(self.list_permissions.count()):
            item = self.list_permissions.item(index)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                permissions.append(str(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        self.store.set_role_permissions(role_key, permissions)
        QtWidgets.QMessageBox.information(
            self,
            tr("auth.tab.role_permissions"),
            tr("auth.role_permissions_saved"),
        )


class AuditLogDialog(QtWidgets.QDialog):
    def __init__(self, parent, store: AuditStore, *, can_export: bool) -> None:
        super().__init__(parent)
        self.store = store
        self.can_export = can_export
        self._rows: list[dict] = []
        self._loading_filter_values = False
        self._query_timer = QtCore.QTimer(self)
        self._query_timer.setSingleShot(True)
        self._query_timer.setInterval(200)
        self._query_timer.timeout.connect(self._query)
        self.setWindowTitle(tr("audit.title"))
        self.resize(980, 560)
        _apply_dark_dialog_style(self)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        filters = QtWidgets.QGridLayout()
        filters.setHorizontalSpacing(8)
        filters.setVerticalSpacing(8)
        now = QtCore.QDateTime.currentDateTime()
        start_of_day = QtCore.QDateTime(now.date(), QtCore.QTime(0, 0, 0))
        self.chk_start_time = QtWidgets.QCheckBox(tr("audit.start_time"))
        self.chk_end_time = QtWidgets.QCheckBox(tr("audit.end_time"))
        self.edit_start = _new_audit_time_editor(start_of_day)
        self.edit_end = _new_audit_time_editor(now)
        self.chk_start_time.toggled.connect(self.edit_start.setEnabled)
        self.chk_end_time.toggled.connect(self.edit_end.setEnabled)
        self.cmb_product_filter = _new_filter_combo(tr("audit.all_products"))
        self.cmb_module_filter = _new_filter_combo(tr("audit.all_modules"))
        self.cmb_user_filter = _new_filter_combo(tr("audit.all_users"))
        filters.addWidget(self.chk_start_time, 0, 0)
        filters.addWidget(self.edit_start, 0, 1)
        filters.addWidget(self.chk_end_time, 0, 2)
        filters.addWidget(self.edit_end, 0, 3)
        filters.addWidget(QtWidgets.QLabel(tr("audit.product")), 1, 0)
        filters.addWidget(self.cmb_product_filter, 1, 1)
        filters.addWidget(QtWidgets.QLabel(tr("audit.module")), 1, 2)
        filters.addWidget(self.cmb_module_filter, 1, 3)
        filters.addWidget(QtWidgets.QLabel(tr("audit.user")), 1, 4)
        filters.addWidget(self.cmb_user_filter, 1, 5)
        btn_query = QtWidgets.QPushButton(tr("audit.query"))
        btn_export = QtWidgets.QPushButton(tr("audit.export_csv"))
        btn_export.setEnabled(can_export)
        btn_query.clicked.connect(self._query)
        btn_export.clicked.connect(self._export)
        filters.addWidget(btn_query, 0, 5)
        filters.addWidget(btn_export, 0, 6)
        root.addLayout(filters)

        self.table = QtWidgets.QTableWidget(0, 12)
        _configure_table(self.table)
        self.table.setHorizontalHeaderLabels([
            tr("audit.time"),
            tr("audit.user"),
            tr("auth.role"),
            tr("audit.product"),
            tr("audit.module"),
            tr("audit.action"),
            tr("audit.target"),
            tr("audit.before"),
            tr("audit.after"),
            tr("audit.result"),
            tr("audit.remark"),
            tr("audit.software_version"),
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)
        self._connect_filter_auto_query()
        self._reload_filter_values()
        self._query()

    def _connect_filter_auto_query(self) -> None:
        self.chk_start_time.toggled.connect(self._schedule_query)
        self.chk_end_time.toggled.connect(self._schedule_query)
        self.edit_start.dateTimeChanged.connect(
            lambda _value: self._schedule_query() if self.chk_start_time.isChecked() else None
        )
        self.edit_end.dateTimeChanged.connect(
            lambda _value: self._schedule_query() if self.chk_end_time.isChecked() else None
        )
        for combo in (self.cmb_product_filter, self.cmb_module_filter, self.cmb_user_filter):
            combo.currentTextChanged.connect(self._schedule_query)

    def _schedule_query(self, *_args: object) -> None:
        if self._loading_filter_values:
            return
        self._query_timer.start()

    def _query(self) -> None:
        self._rows = self.store.query_events(
            start_at=self._time_filter_text(self.chk_start_time, self.edit_start),
            end_at=self._time_filter_text(self.chk_end_time, self.edit_end),
            product_name=_combo_filter_text(self.cmb_product_filter),
            module=_combo_filter_text(self.cmb_module_filter),
            user_name=_combo_filter_text(self.cmb_user_filter),
        )
        self.table.setRowCount(len(self._rows))
        keys = [
            "created_at",
            "user_name",
            "role_name",
            "product_name",
            "module",
            "action",
            "target",
            "before_value",
            "after_value",
            "result",
            "remark",
            "software_version",
        ]
        for row, data in enumerate(self._rows):
            for col, key in enumerate(keys):
                item = QtWidgets.QTableWidgetItem(str(data.get(key, "") or ""))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _time_filter_text(
        self,
        checkbox: QtWidgets.QCheckBox,
        editor: QtWidgets.QDateTimeEdit,
    ) -> str:
        if not checkbox.isChecked():
            return ""
        return editor.dateTime().toString("yyyy-MM-dd HH:mm:ss")

    def _reload_filter_values(self) -> None:
        self._loading_filter_values = True
        try:
            self._populate_filter_combo(
                self.cmb_product_filter,
                self.store.audit_filter_values("product_name"),
            )
            self._populate_filter_combo(
                self.cmb_module_filter,
                self.store.audit_filter_values("module"),
            )
            self._populate_filter_combo(
                self.cmb_user_filter,
                self.store.audit_filter_values("user_name"),
            )
        finally:
            self._loading_filter_values = False

    def _populate_filter_combo(self, combo: QtWidgets.QComboBox, values: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        for value in values:
            combo.addItem(value)
        index = combo.findText(current)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(current)
        combo.blockSignals(False)

    def _export(self) -> None:
        if not self.can_export:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            tr("audit.export_title"),
            "audit_events.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        self.store.export_events_csv(self._rows, Path(path))
        audit_event = getattr(self.parent(), "_audit_event", None)
        if callable(audit_event):
            audit_event(
                module="履历",
                action="导出履历",
                after_value=str(path),
                product_name="",
            )
        QtWidgets.QMessageBox.information(self, tr("audit.export_title"), tr("audit.export_done"))


class RuntimeRecordsDialog(QtWidgets.QDialog):
    def __init__(self, parent, store: RuntimeRecordStore, *, can_export: bool) -> None:
        super().__init__(parent)
        self.store = store
        self.can_export = can_export
        self._rows: list[dict] = []
        self._loading_filter_values = False
        self._query_timer = QtCore.QTimer(self)
        self._query_timer.setSingleShot(True)
        self._query_timer.setInterval(200)
        self._query_timer.timeout.connect(self._query)
        self.setWindowTitle(tr("runtime_records.title"))
        self.resize(1080, 720)
        _apply_dark_dialog_style(self)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        filters = QtWidgets.QGridLayout()
        filters.setHorizontalSpacing(8)
        filters.setVerticalSpacing(8)
        now = QtCore.QDateTime.currentDateTime()
        start_of_day = QtCore.QDateTime(now.date(), QtCore.QTime(0, 0, 0))
        self.chk_start_time = QtWidgets.QCheckBox(tr("audit.start_time"))
        self.chk_end_time = QtWidgets.QCheckBox(tr("audit.end_time"))
        self.edit_start = _new_audit_time_editor(start_of_day)
        self.edit_end = _new_audit_time_editor(now)
        self.chk_start_time.toggled.connect(self.edit_start.setEnabled)
        self.chk_end_time.toggled.connect(self.edit_end.setEnabled)
        self.cmb_product_filter = _new_filter_combo(tr("audit.all_products"))
        self.cmb_result_filter = _new_filter_combo(tr("runtime_records.all_results"))
        btn_query = QtWidgets.QPushButton(tr("audit.query"))
        btn_export = QtWidgets.QPushButton(tr("audit.export_csv"))
        btn_export.setEnabled(can_export)
        btn_query.clicked.connect(self._query)
        btn_export.clicked.connect(self._export)
        filters.addWidget(self.chk_start_time, 0, 0)
        filters.addWidget(self.edit_start, 0, 1)
        filters.addWidget(self.chk_end_time, 0, 2)
        filters.addWidget(self.edit_end, 0, 3)
        filters.addWidget(btn_query, 0, 5)
        filters.addWidget(btn_export, 0, 6)
        filters.addWidget(QtWidgets.QLabel(tr("audit.product")), 1, 0)
        filters.addWidget(self.cmb_product_filter, 1, 1)
        filters.addWidget(QtWidgets.QLabel(tr("runtime_records.total_result")), 1, 2)
        filters.addWidget(self.cmb_result_filter, 1, 3)
        root.addLayout(filters)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.table_runs = QtWidgets.QTableWidget(0, 7)
        _configure_table(self.table_runs)
        self.table_runs.setHorizontalHeaderLabels([
            tr("audit.time"),
            tr("audit.product"),
            tr("runtime_records.total_result"),
            tr("runtime_records.recipe"),
            tr("runtime_records.duration_ms"),
            tr("runtime_records.error_message"),
            tr("runtime_records.image_paths"),
        ])
        self.table_runs.horizontalHeader().setStretchLastSection(True)
        self.table_runs.itemSelectionChanged.connect(self._load_selected_roi_results)
        splitter.addWidget(self.table_runs)

        details_widget = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)
        details_layout.addWidget(QtWidgets.QLabel(tr("runtime_records.roi_details")))
        self.table_roi_results = QtWidgets.QTableWidget(0, 7)
        _configure_table(self.table_roi_results)
        self.table_roi_results.setHorizontalHeaderLabels([
            tr("runtime_records.camera"),
            tr("runtime_records.display_name"),
            tr("runtime_records.roi_label"),
            tr("audit.result"),
            tr("runtime_records.value"),
            tr("runtime_records.unit"),
            tr("runtime_records.detail"),
        ])
        self.table_roi_results.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(self.table_roi_results, 1)
        splitter.addWidget(details_widget)
        splitter.setSizes([400, 260])
        root.addWidget(splitter, 1)

        self._connect_filter_auto_query()
        self._reload_filter_values()
        self._query()

    def _connect_filter_auto_query(self) -> None:
        self.chk_start_time.toggled.connect(self._schedule_query)
        self.chk_end_time.toggled.connect(self._schedule_query)
        self.edit_start.dateTimeChanged.connect(
            lambda _value: self._schedule_query() if self.chk_start_time.isChecked() else None
        )
        self.edit_end.dateTimeChanged.connect(
            lambda _value: self._schedule_query() if self.chk_end_time.isChecked() else None
        )
        self.cmb_product_filter.currentTextChanged.connect(self._schedule_query)
        self.cmb_result_filter.currentTextChanged.connect(self._schedule_query)

    def _schedule_query(self, *_args: object) -> None:
        if not self._loading_filter_values:
            self._query_timer.start()

    def _query(self) -> None:
        self._rows = self.store.query_runs(
            start_at=self._time_filter_text(self.chk_start_time, self.edit_start),
            end_at=self._time_filter_text(self.chk_end_time, self.edit_end),
            product_name=_combo_filter_text(self.cmb_product_filter),
            final_result=_combo_filter_text(self.cmb_result_filter),
        )
        keys = [
            "record_time",
            "product_name",
            "final_result",
            "recipe_name",
            "duration_ms",
            "error_message",
            "image_paths_json",
        ]
        self.table_runs.blockSignals(True)
        self.table_runs.setRowCount(len(self._rows))
        for row_index, data in enumerate(self._rows):
            for column_index, key in enumerate(keys):
                value = data.get(key, "")
                if key == "image_paths_json":
                    value = RuntimeRecordStore.image_paths_text(value)
                item = QtWidgets.QTableWidgetItem(str(value or ""))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if column_index == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, int(data.get("id", 0) or 0))
                self.table_runs.setItem(row_index, column_index, item)
        self.table_runs.blockSignals(False)
        self.table_runs.resizeColumnsToContents()
        if self._rows:
            self.table_runs.selectRow(0)
            self._load_selected_roi_results()
        else:
            self.table_roi_results.setRowCount(0)

    @staticmethod
    def _time_filter_text(
        checkbox: QtWidgets.QCheckBox,
        editor: QtWidgets.QDateTimeEdit,
    ) -> str:
        if not checkbox.isChecked():
            return ""
        return editor.dateTime().toString("yyyy-MM-dd HH:mm:ss")

    def _reload_filter_values(self) -> None:
        self._loading_filter_values = True
        try:
            self._populate_filter_combo(
                self.cmb_product_filter,
                self.store.runtime_filter_values("product_name"),
            )
            self._populate_filter_combo(
                self.cmb_result_filter,
                self.store.runtime_filter_values("final_result"),
            )
        finally:
            self._loading_filter_values = False

    @staticmethod
    def _populate_filter_combo(combo: QtWidgets.QComboBox, values: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        combo.addItems(values)
        index = combo.findText(current)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(current)
        combo.blockSignals(False)

    def _selected_run_id(self) -> int | None:
        row = self.table_runs.currentRow()
        if row < 0:
            return None
        item = self.table_runs.item(row, 0)
        try:
            return int(item.data(QtCore.Qt.ItemDataRole.UserRole)) if item is not None else None
        except (TypeError, ValueError):
            return None

    def _load_selected_roi_results(self) -> None:
        run_id = self._selected_run_id()
        rows = self.store.roi_results_for_run(run_id) if run_id is not None else []
        self.table_roi_results.setRowCount(len(rows))
        keys = ["camera_id", "display_name", "roi_label", "result", "value", "unit", "detail"]
        for row_index, data in enumerate(rows):
            for column_index, key in enumerate(keys):
                value = data.get(key, "")
                text = "" if value is None else str(value)
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table_roi_results.setItem(row_index, column_index, item)
        self.table_roi_results.resizeColumnsToContents()

    def _export(self) -> None:
        if not self.can_export:
            return
        if not self._rows:
            QtWidgets.QMessageBox.information(
                self,
                tr("runtime_records.title"),
                tr("runtime_records.no_results"),
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            tr("runtime_records.export_title"),
            "runtime_records.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        details_by_run = self.store.roi_results_for_run_ids(
            [row.get("id", 0) for row in self._rows]
        )
        self.store.export_runs_csv(self._rows, details_by_run, Path(path))
        audit_event = getattr(self.parent(), "_audit_event", None)
        if callable(audit_event):
            audit_event(
                module="运行记录",
                action="导出运行记录",
                after_value=str(path),
                product_name="",
            )
        QtWidgets.QMessageBox.information(
            self,
            tr("runtime_records.export_title"),
            tr("runtime_records.export_done"),
        )


class SoftwareVersionDialog(QtWidgets.QDialog):
    def __init__(self, parent, store: AuditStore, *, can_edit: bool, current_user: str, software_version: str) -> None:
        super().__init__(parent)
        self.store = store
        self.can_edit = can_edit
        self.current_user = current_user
        self.software_version = software_version
        self.setWindowTitle(tr("software.title"))
        self.resize(820, 520)
        _apply_dark_dialog_style(self)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        self.table = QtWidgets.QTableWidget(0, 6)
        _configure_table(self.table)
        self.table.setHorizontalHeaderLabels([
            tr("audit.time"),
            tr("software.version"),
            tr("software.summary"),
            tr("software.details"),
            tr("software.operator"),
            tr("software.current_version"),
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.edit_version = QtWidgets.QLineEdit(software_version)
        self.edit_summary = QtWidgets.QLineEdit()
        self.edit_details = QtWidgets.QPlainTextEdit()
        self.edit_details.setFixedHeight(70)
        form.addWidget(QtWidgets.QLabel(tr("software.version")), 0, 0)
        form.addWidget(self.edit_version, 0, 1)
        form.addWidget(QtWidgets.QLabel(tr("software.summary")), 0, 2)
        form.addWidget(self.edit_summary, 0, 3)
        form.addWidget(QtWidgets.QLabel(tr("software.details")), 1, 0)
        form.addWidget(self.edit_details, 1, 1, 1, 3)
        btn_add = QtWidgets.QPushButton(tr("software.add_record"))
        btn_add.setEnabled(can_edit)
        btn_add.clicked.connect(self._add_version)
        form.addWidget(btn_add, 2, 3)
        root.addLayout(form)
        self._reload()

    def _reload(self) -> None:
        rows = self.store.software_versions()
        self.table.setRowCount(len(rows))
        keys = ["created_at", "version", "summary", "details", "operator", "software_version"]
        for row, data in enumerate(rows):
            for col, key in enumerate(keys):
                item = QtWidgets.QTableWidgetItem(str(data.get(key, "") or ""))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _add_version(self) -> None:
        if not self.can_edit:
            return
        version = self.edit_version.text().strip()
        summary = self.edit_summary.text().strip()
        if not version or not summary:
            QtWidgets.QMessageBox.warning(self, tr("software.title"), tr("software.version_summary_required"))
            return
        self.store.add_software_version(
            version=version,
            summary=summary,
            details=self.edit_details.toPlainText().strip(),
            operator=self.current_user,
            software_version=self.software_version,
        )
        audit_event = getattr(self.parent(), "_audit_event", None)
        if callable(audit_event):
            audit_event(
                module="软件版本",
                action="新增版本记录",
                target=version,
                after_value=summary,
                product_name="",
            )
        self.edit_summary.clear()
        self.edit_details.clear()
        self._reload()
