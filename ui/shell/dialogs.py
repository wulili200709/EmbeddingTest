from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtWidgets

from ui.window_common import embedding_test_root


DEFAULT_ADMIN_PASSWORD = "admin123"
SYSTEM_PASSWORDS_FILENAME = "system_passwords.json"


def _dialog_style_sheet() -> str:
    return (
        "QDialog{background:#3a3a3a;color:#e0e0e0;}"
        "QLabel{color:#e0e0e0;}"
        "QLineEdit{background:#404040;color:#e0e0e0;border:1px solid #5a5a5a;"
        "padding:5px 6px;border-radius:3px;selection-background-color:#3794ff;}"
        "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
        "padding:5px 18px;border-radius:4px;min-width:72px;}"
        "QPushButton:hover{background:#505050;}"
    )


def _apply_dialog_theme(dialog: QtWidgets.QDialog) -> None:
    dialog.setStyleSheet(_dialog_style_sheet())


class PasswordSettingsStore:
    def __init__(
        self,
        *,
        default_release_password: str,
        default_admin_password: str = DEFAULT_ADMIN_PASSWORD,
    ) -> None:
        self._default_release_password = str(default_release_password).strip() or "1234"
        self._default_admin_password = str(default_admin_password).strip() or DEFAULT_ADMIN_PASSWORD

    def path(self) -> Path:
        config_dir = embedding_test_root(__file__) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / SYSTEM_PASSWORDS_FILENAME

    def default_settings(self) -> dict[str, str]:
        return {
            "run_password": self._default_release_password,
            "engineer_password": self._default_admin_password,
        }

    def load(self) -> dict[str, str]:
        settings = self.default_settings()
        path = self.path()
        raw: dict = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                raw = {}

        if isinstance(raw, dict):
            run_password = str(raw.get("run_password", "")).strip()
            engineer_password = str(raw.get("engineer_password", "")).strip()
            if run_password:
                settings["run_password"] = run_password
            if engineer_password:
                settings["engineer_password"] = engineer_password

        try:
            self.save(settings)
        except Exception:
            pass
        return settings

    def save(self, settings: dict[str, str]) -> None:
        payload = {
            "run_password": str(settings.get("run_password", self._default_release_password)).strip()
            or self._default_release_password,
            "engineer_password": str(settings.get("engineer_password", self._default_admin_password)).strip()
            or self._default_admin_password,
        }
        self.path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def prompt_password_dialog(
    parent: QtWidgets.QWidget,
    *,
    title: str,
    label: str,
) -> tuple[str, bool]:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(320)
    _apply_dialog_theme(dialog)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)

    prompt_label = QtWidgets.QLabel(label)
    layout.addWidget(prompt_label)

    edit_password = QtWidgets.QLineEdit()
    edit_password.setEchoMode(QtWidgets.QLineEdit.Password)
    layout.addWidget(edit_password)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)

    edit_password.returnPressed.connect(dialog.accept)
    edit_password.setFocus()

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return "", False
    return edit_password.text(), True


def confirm_admin_password(
    parent: QtWidgets.QWidget,
    *,
    admin_password: str,
) -> bool:
    entered_password, ok = prompt_password_dialog(
        parent,
        title="管理员验证",
        label="输入管理员密码：",
    )
    if not ok:
        return False
    if str(entered_password) != str(admin_password):
        QtWidgets.QMessageBox.warning(
            parent,
            "管理员验证",
            "管理员密码错误。",
        )
        return False
    return True


def prompt_change_release_password(parent: QtWidgets.QWidget) -> str | None:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("修改放行密码")
    dialog.setMinimumWidth(360)
    _apply_dialog_theme(dialog)
    layout = QtWidgets.QFormLayout(dialog)

    label_tip = QtWidgets.QLabel("仅管理员可修改 NG 放行密码。")
    label_tip.setStyleSheet("color:#b8b8b8;")
    layout.addRow(label_tip)

    edit_new_password = QtWidgets.QLineEdit()
    edit_new_password.setEchoMode(QtWidgets.QLineEdit.Password)
    edit_new_password.setPlaceholderText("输入新的放行密码")
    edit_confirm_password = QtWidgets.QLineEdit()
    edit_confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
    edit_confirm_password.setPlaceholderText("再次输入新密码")
    layout.addRow("新密码", edit_new_password)
    layout.addRow("确认密码", edit_confirm_password)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addRow(button_box)

    while True:
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        new_password = edit_new_password.text().strip()
        confirm_password = edit_confirm_password.text().strip()
        if not new_password:
            QtWidgets.QMessageBox.warning(parent, "修改放行密码", "新密码不能为空。")
            continue
        if new_password != confirm_password:
            QtWidgets.QMessageBox.warning(parent, "修改放行密码", "两次输入的新密码不一致。")
            continue
        return new_password


def prompt_connect_camera_bindings(
    parent: QtWidgets.QWidget,
    *,
    cam1_serial: str,
    cam2_serial: str,
) -> tuple[str, str] | None:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("连接相机")
    dialog.setMinimumWidth(360)
    _apply_dialog_theme(dialog)
    layout = QtWidgets.QFormLayout(dialog)

    edit_cam1 = QtWidgets.QLineEdit(cam1_serial)
    edit_cam1.setPlaceholderText("Cam1 序列号")
    edit_cam2 = QtWidgets.QLineEdit(cam2_serial)
    edit_cam2.setPlaceholderText("Cam2 序列号（可选）")
    layout.addRow("Cam1 序列号", edit_cam1)
    layout.addRow("Cam2 序列号", edit_cam2)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addRow(button_box)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None
    return edit_cam1.text().strip(), edit_cam2.text().strip()
