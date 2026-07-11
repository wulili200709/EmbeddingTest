from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from application import (
    DEFAULT_RELEASE_PASSWORD,
    ProductRuntimeContext,
    RuntimeController,
)
from .runtime_mode_pyside6 import RuntimeModePage
from ui.shell.dialogs import RuntimeModeSettingsStore
from ui.window_common import (
    build_default_session_and_algo,
    connect_runtime_dialogs,
    connect_runtime_page,
    detect_runtime_import_error,
    update_runtime_preview,
)


_RUNTIME_IMPORT_ERROR = detect_runtime_import_error()


class RunMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quick Register Runtime")

        self.session, self.algo = build_default_session_and_algo(__file__)
        self.runtime_context = ProductRuntimeContext(self.session, self.algo)
        self.runtime_mode_store = RuntimeModeSettingsStore()
        self.runtime_mode_settings = self.runtime_mode_store.load()

        self.runtime_page = RuntimeModePage()
        self.runtime_page.edit_release_password.setText(DEFAULT_RELEASE_PASSWORD)
        self.runtime_page.set_camera_layout_settings(self.runtime_mode_settings)
        self.setCentralWidget(self.runtime_page)

        self.runtime_ctrl = RuntimeController(
            session=self.session,
            algo=self.algo,
            runtime_context=self.runtime_context,
            import_error=_RUNTIME_IMPORT_ERROR,
            release_password=DEFAULT_RELEASE_PASSWORD,
            parent=self,
        )

        self._connect_signals()
        self.runtime_ctrl.refresh_all_status()

    def _connect_signals(self) -> None:
        connect_runtime_page(self.runtime_page, self.runtime_ctrl)
        self.runtime_ctrl.previewUpdated.connect(self._on_runtime_preview_updated)
        self.runtime_page.cameraLayoutSettingsChanged.connect(
            self._on_runtime_camera_layout_settings_changed
        )

        connect_runtime_dialogs(self, self.runtime_ctrl)

    def _on_runtime_preview_updated(self, role: str, source: object) -> None:
        update_runtime_preview(self.runtime_page, role, source)

    def _on_runtime_camera_layout_settings_changed(self, settings: dict) -> None:
        merged = dict(self.runtime_mode_settings)
        merged.update(dict(settings or {}))
        self.runtime_mode_store.save(merged)
        self.runtime_mode_settings = merged

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.runtime_ctrl.disconnect(silent=True)
        super().closeEvent(event)


__all__ = ["RunMainWindow"]
