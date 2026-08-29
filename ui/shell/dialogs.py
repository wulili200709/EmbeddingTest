from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from common.app_paths import writable_embedding_test_root
from common.camera_roles import CAMERA_ROLES, normalize_camera_role
from common.safe_io import atomic_write_json, load_json_with_backup
from ui.i18n import tr


DEFAULT_ADMIN_PASSWORD = "admin123"
SYSTEM_PASSWORDS_FILENAME = "system_passwords.json"
TOWER_LIGHT_SETTINGS_FILENAME = "tower_light_settings.json"
RUNTIME_RECORD_SETTINGS_FILENAME = "runtime_record_settings.json"
RUNTIME_MODE_SETTINGS_FILENAME = "runtime_mode_settings.json"


def _dialog_style_sheet() -> str:
    return (
        "QDialog{background:#3a3a3a;color:#e0e0e0;}"
        "QLabel{color:#e0e0e0;}"
        "QLineEdit{background:#404040;color:#e0e0e0;border:1px solid #5a5a5a;"
        "padding:5px 6px;border-radius:3px;selection-background-color:#3794ff;}"
        "QSpinBox,QDoubleSpinBox{background:#404040;color:#e0e0e0;border:1px solid #5a5a5a;"
        "padding:4px 6px;border-radius:3px;selection-background-color:#3794ff;}"
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
        config_dir = writable_embedding_test_root(__file__) / "config"
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
        raw = load_json_with_backup(path, default={})

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
        atomic_write_json(self.path(), payload, ensure_ascii=False, indent=2)


class TowerLightSettingsStore:
    def path(self) -> Path:
        config_dir = writable_embedding_test_root(__file__) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / TOWER_LIGHT_SETTINGS_FILENAME

    def default_settings(self) -> dict[str, int]:
        return {
            "ok_flash_ms": 200,
            "ng_flash_ms": 200,
            "ng_buzzer_ms": 500,
            "idle_blue_delay_ms": 30000,
        }

    def load(self) -> dict[str, int]:
        settings = self.default_settings()
        path = self.path()
        raw = load_json_with_backup(path, default={})

        if isinstance(raw, dict):
            for key in tuple(settings.keys()):
                try:
                    value = int(raw.get(key, settings[key]))
                except Exception:
                    value = settings[key]
                settings[key] = max(0, value)

        try:
            self.save(settings)
        except Exception:
            pass
        return settings

    def save(self, settings: dict[str, int]) -> None:
        defaults = self.default_settings()
        payload = {}
        for key, default_value in defaults.items():
            try:
                payload[key] = max(0, int(settings.get(key, default_value)))
            except Exception:
                payload[key] = default_value
        atomic_write_json(self.path(), payload, ensure_ascii=False, indent=2)


class RuntimeRecordSettingsStore:
    def path(self) -> Path:
        config_dir = writable_embedding_test_root(__file__) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / RUNTIME_RECORD_SETTINGS_FILENAME

    def default_settings(self) -> dict[str, str]:
        return {
            "runtime_records_dir": "",
            "runtime_images_dir": "",
        }

    def load(self) -> dict[str, str]:
        settings = self.default_settings()
        path = self.path()
        raw = load_json_with_backup(path, default={})

        if isinstance(raw, dict):
            settings["runtime_records_dir"] = str(raw.get("runtime_records_dir", "")).strip()
            settings["runtime_images_dir"] = str(raw.get("runtime_images_dir", "")).strip()

        try:
            self.save(settings)
        except Exception:
            pass
        return settings

    def save(self, settings: dict[str, str]) -> None:
        payload = {
            "runtime_records_dir": str(settings.get("runtime_records_dir", "")).strip(),
            "runtime_images_dir": str(settings.get("runtime_images_dir", "")).strip(),
        }
        atomic_write_json(self.path(), payload, ensure_ascii=False, indent=2)


class RuntimeModeSettingsStore:
    def path(self) -> Path:
        config_dir = writable_embedding_test_root(__file__) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / RUNTIME_MODE_SETTINGS_FILENAME

    def default_settings(self) -> dict[str, object]:
        return {
            "auto_show_release_dialog_on_ng": True,
            "camera_layout": "two_top_one_bottom",
            "camera_slots": list(CAMERA_ROLES),
            "cpu_inference_chunk_size": 2,
        }

    def load(self) -> dict[str, object]:
        settings = self.default_settings()
        path = self.path()
        raw = load_json_with_backup(path, default={})

        if isinstance(raw, dict):
            raw_value = settings["auto_show_release_dialog_on_ng"]
            for key in (
                "auto_show_release_dialog_on_ng",
                "AUTO_SHOW_RELEASE_DIALOG_ON_NG",
                "lock_on_ng",
            ):
                if key in raw:
                    raw_value = raw.get(key)
                    break
            settings["auto_show_release_dialog_on_ng"] = self._as_bool(
                raw_value,
                default=settings["auto_show_release_dialog_on_ng"],
            )
            layout = str(raw.get("camera_layout", settings["camera_layout"]) or "").strip()
            if layout in {
                "row",
                "two_top_one_bottom",
                "one_top_two_bottom",
                "main_left",
                "main_right",
            }:
                settings["camera_layout"] = layout
            settings["camera_slots"] = self._normalize_camera_slots(
                raw.get("camera_slots", settings["camera_slots"])
            )
            settings["cpu_inference_chunk_size"] = self._normalize_cpu_inference_chunk_size(
                raw.get("cpu_inference_chunk_size", settings["cpu_inference_chunk_size"])
            )

        try:
            self.save(settings)
        except Exception:
            pass
        return settings

    def save(self, settings: dict[str, object]) -> None:
        payload = {
            "auto_show_release_dialog_on_ng": self._as_bool(
                settings.get("auto_show_release_dialog_on_ng", True),
                default=True,
            ),
            "camera_layout": str(settings.get("camera_layout", "two_top_one_bottom") or "two_top_one_bottom").strip(),
            "camera_slots": self._normalize_camera_slots(settings.get("camera_slots", list(CAMERA_ROLES))),
            "cpu_inference_chunk_size": self._normalize_cpu_inference_chunk_size(
                settings.get("cpu_inference_chunk_size", 2)
            ),
        }
        if payload["camera_layout"] not in {
            "row",
            "two_top_one_bottom",
            "one_top_two_bottom",
            "main_left",
            "main_right",
        }:
            payload["camera_layout"] = "two_top_one_bottom"
        atomic_write_json(self.path(), payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _as_bool(value: object, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _normalize_camera_slots(value: object) -> list[str]:
        normalized: list[str] = []
        raw_slots = list(value) if isinstance(value, (list, tuple)) else []
        for role in raw_slots:
            role_text = normalize_camera_role(role)
            if role_text and role_text not in normalized:
                normalized.append(role_text)
        for role in CAMERA_ROLES:
            if role not in normalized:
                normalized.append(role)
        return normalized[: len(CAMERA_ROLES)]

    @staticmethod
    def _normalize_cpu_inference_chunk_size(value: object) -> int:
        try:
            chunk_size = int(value)
        except (TypeError, ValueError):
            chunk_size = 2
        return max(1, min(256, chunk_size))


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
        title=tr("shell.admin_verify"),
        label=tr("shell.admin_password_prompt"),
    )
    if not ok:
        return False
    if str(entered_password) != str(admin_password):
        QtWidgets.QMessageBox.warning(
            parent,
            tr("shell.admin_verify"),
            tr("shell.admin_password_wrong"),
        )
        return False
    return True


def prompt_change_release_password(parent: QtWidgets.QWidget) -> str | None:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(tr("shell.release_password_title"))
    dialog.setMinimumWidth(360)
    _apply_dialog_theme(dialog)
    layout = QtWidgets.QFormLayout(dialog)

    label_tip = QtWidgets.QLabel(tr("shell.release_password_admin_only"))
    label_tip.setStyleSheet("color:#b8b8b8;")
    layout.addRow(label_tip)

    edit_new_password = QtWidgets.QLineEdit()
    edit_new_password.setEchoMode(QtWidgets.QLineEdit.Password)
    edit_new_password.setPlaceholderText(tr("shell.release_password_new_placeholder"))
    edit_confirm_password = QtWidgets.QLineEdit()
    edit_confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
    edit_confirm_password.setPlaceholderText(tr("shell.release_password_confirm_placeholder"))
    layout.addRow(tr("shell.release_password_new"), edit_new_password)
    layout.addRow(tr("shell.release_password_confirm"), edit_confirm_password)

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
            QtWidgets.QMessageBox.warning(
                parent, tr("shell.release_password_title"), tr("shell.release_password_empty")
            )
            continue
        if new_password != confirm_password:
            QtWidgets.QMessageBox.warning(
                parent, tr("shell.release_password_title"), tr("shell.release_password_mismatch")
            )
            continue
        return new_password


def prompt_connect_camera_bindings(
    parent: QtWidgets.QWidget,
    *,
    cam1_serial: str,
    cam2_serial: str,
    cam3_serial: str = "",
    enabled_roles: list[str] | None = None,
    visible_roles: list[str] | None = None,
    physical_roles_only: bool = False,
) -> tuple[dict[str, str], list[str]] | None:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(tr("shell.connect_camera"))
    dialog.setMinimumWidth(420)
    _apply_dialog_theme(dialog)
    layout = QtWidgets.QGridLayout(dialog)
    layout.setColumnStretch(1, 1)
    layout.addWidget(
        QtWidgets.QLabel(
            tr("shell.camera_scope_physical")
            if physical_roles_only
            else tr("shell.camera_scope_product")
        ),
        0,
        0,
    )
    layout.addWidget(QtWidgets.QLabel(tr("shell.camera_serial")), 0, 1)

    enabled = {str(role).strip() for role in (enabled_roles or []) if str(role).strip()}
    if not enabled:
        enabled = {"cam1"}
    visible = {str(role).strip() for role in (visible_roles or []) if str(role).strip()}

    chk_cam1 = QtWidgets.QCheckBox("Cam1")
    chk_cam1.setChecked("cam1" in enabled)
    edit_cam1 = QtWidgets.QLineEdit(cam1_serial)
    edit_cam1.setPlaceholderText(tr("shell.camera_serial_placeholder", camera="Cam1"))

    chk_cam2 = QtWidgets.QCheckBox("Cam2")
    chk_cam2.setChecked("cam2" in enabled)
    edit_cam2 = QtWidgets.QLineEdit(cam2_serial)

    chk_cam3 = QtWidgets.QCheckBox("Cam3")
    chk_cam3.setChecked("cam3" in enabled)
    edit_cam3 = QtWidgets.QLineEdit(cam3_serial)
    edit_cam3.setPlaceholderText(tr("shell.camera_serial_optional", camera="Cam3"))
    edit_cam2.setPlaceholderText(tr("shell.camera_serial_optional", camera="Cam2"))

    layout.addWidget(chk_cam1, 1, 0)
    layout.addWidget(edit_cam1, 1, 1)
    layout.addWidget(chk_cam2, 2, 0)
    layout.addWidget(edit_cam2, 2, 1)
    layout.addWidget(chk_cam3, 3, 0)
    layout.addWidget(edit_cam3, 3, 1)
    if visible:
        for role, checkbox, editor in (
            ("cam1", chk_cam1, edit_cam1),
            ("cam2", chk_cam2, edit_cam2),
            ("cam3", chk_cam3, edit_cam3),
        ):
            row_visible = role in visible
            checkbox.setVisible(row_visible)
            editor.setVisible(row_visible)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    role_checkboxes = (
        ("cam1", chk_cam1),
        ("cam2", chk_cam2),
        ("cam3", chk_cam3),
    )

    def _accept_selected_roles() -> None:
        if not any(checkbox.isChecked() for _role, checkbox in role_checkboxes):
            QtWidgets.QMessageBox.warning(
                dialog, tr("shell.connect_camera"), tr("shell.camera_select_one")
            )
            return
        dialog.accept()

    button_box.accepted.connect(_accept_selected_roles)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box, 4, 0, 1, 2)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None
    serials = {
        "cam1": edit_cam1.text().strip(),
        "cam2": edit_cam2.text().strip(),
        "cam3": edit_cam3.text().strip(),
    }
    roles = [role for role, checkbox in role_checkboxes if checkbox.isChecked()]
    return serials, roles


def prompt_tower_light_settings(
    parent: QtWidgets.QWidget,
    *,
    current_settings: dict[str, int],
) -> dict[str, int] | None:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(tr("shell.tower_settings"))
    dialog.setMinimumWidth(380)
    _apply_dialog_theme(dialog)

    layout = QtWidgets.QFormLayout(dialog)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)

    ok_spin = QtWidgets.QSpinBox()
    ok_spin.setRange(10, 10000)
    ok_spin.setSingleStep(10)
    ok_spin.setSuffix(" ms")
    ok_spin.setValue(max(10, int(current_settings.get("ok_flash_ms", 200))))
    layout.addRow(tr("shell.tower_ok_ms"), ok_spin)

    ng_spin = QtWidgets.QSpinBox()
    ng_spin.setRange(10, 10000)
    ng_spin.setSingleStep(10)
    ng_spin.setSuffix(" ms")
    ng_spin.setValue(max(10, int(current_settings.get("ng_flash_ms", 200))))
    layout.addRow(tr("shell.tower_ng_ms"), ng_spin)

    buzzer_spin = QtWidgets.QSpinBox()
    buzzer_spin.setRange(0, 10000)
    buzzer_spin.setSingleStep(10)
    buzzer_spin.setSuffix(" ms")
    buzzer_spin.setValue(max(0, int(current_settings.get("ng_buzzer_ms", 500))))
    layout.addRow(tr("shell.tower_buzzer_ms"), buzzer_spin)

    idle_spin = QtWidgets.QSpinBox()
    idle_spin.setRange(0, 600000)
    idle_spin.setSingleStep(100)
    idle_spin.setSuffix(" ms")
    idle_spin.setValue(max(0, int(current_settings.get("idle_blue_delay_ms", 30000))))
    layout.addRow(tr("shell.tower_idle_ms"), idle_spin)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addRow(button_box)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None

    return {
        "ok_flash_ms": int(ok_spin.value()),
        "ng_flash_ms": int(ng_spin.value()),
        "ng_buzzer_ms": int(buzzer_spin.value()),
        "idle_blue_delay_ms": int(idle_spin.value()),
    }
