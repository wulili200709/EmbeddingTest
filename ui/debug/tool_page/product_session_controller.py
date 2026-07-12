from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from application import SessionData
from ui.i18n import tr


class ProductSessionController:
    def __init__(self, owner) -> None:
        self.owner = owner

    def current_product_name(self) -> str:
        return self.owner.session.current_product

    def refresh_product_selector(self) -> None:
        if not hasattr(self.owner, "cmb_product"):
            return
        blocker = QtCore.QSignalBlocker(self.owner.cmb_product)
        try:
            self.owner.cmb_product.clear()
            self.owner.cmb_product.addItems(self.owner.session.product_names)
            self.owner.cmb_product.setCurrentText(self.owner.session.current_product)
        finally:
            del blocker

    def sync_camera_settings_store_path(self) -> None:
        self.owner._camera_settings_store.set_path(self.owner.session.camera_settings_path)

    def load_session(self) -> None:
        if self.owner._defer_initial_session_load:
            top_level = self.owner.window()
            load_allowed = True
            if isinstance(top_level, QtWidgets.QWidget):
                load_allowed = bool(
                    getattr(top_level, "_allow_initial_tool_session_load", top_level.isVisible())
                )
            if not load_allowed:
                if not self.owner._deferred_session_load_scheduled:
                    self.owner._deferred_session_load_scheduled = True
                    QtCore.QTimer.singleShot(80, self.owner._run_deferred_initial_session_load)
                return
            self.owner._defer_initial_session_load = False
            self.owner._deferred_session_load_scheduled = False

        self.sync_camera_settings_store_path()
        load_capture_config = getattr(self.owner, "_load_capture_config_to_ui", None)
        if callable(load_capture_config):
            load_capture_config()

        self.owner.algo.load_params(self.owner.session.product_params_path)
        self.owner.algo.model = None
        self.owner._roi_results_by_image = {}
        self.owner._apply_runtime_params_to_ui()

        session_data = self.owner.session.load_session()
        self.owner.train_files = list(dict.fromkeys(session_data.train_files or (session_data.ok_files + session_data.ng_files)))
        self.owner.ok_files = []
        self.owner.ng_files = []
        self.owner.test_files = session_data.test_files
        self.owner._load_sample_roi_annotations()
        if hasattr(self.owner, "_set_session_loc_methods"):
            self.owner._set_session_loc_methods(session_data.loc_method, session_data.loc_methods)
        else:
            self.owner.loc_method = session_data.loc_method
        self.owner._shape_recipes_by_role = {}
        self.owner._clear_training_roi_review_state()
        sync_loc_combo = getattr(self.owner, "_sync_loc_combo", None)
        if callable(sync_loc_combo):
            sync_loc_combo()
        else:
            self.owner.cmb_loc.setCurrentText(self.owner.loc_method)
        self.owner._apply_current_role_recipe_state()
        current_method = (
            self.owner.loc_method_for_role(self.owner.current_camera_role())
            if hasattr(self.owner, "loc_method_for_role")
            else self.owner.loc_method
        )
        if (
            current_method == "shape"
            and not self.owner.ref_image
            and session_data.ref_image
            and os.path.exists(session_data.ref_image)
        ):
            self.owner.ref_image = session_data.ref_image
            self.owner.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(self.owner.ref_image)}")
            self.owner.lbl_ref.setToolTip(self.owner.ref_image)
        self.owner._refresh_lists()

        self.owner._reload_inspection_items()
        self.owner._sync_footer()

        self.owner.sessionLoaded.emit()

    def run_deferred_initial_session_load(self) -> None:
        self.owner._deferred_session_load_scheduled = False
        if self.owner._defer_initial_session_load:
            self.owner.load_session()

    def _clear_transient_product_state(self) -> None:
        self.owner.algo.model = None
        self.owner.shape_recipe = None
        self.owner._shape_recipes_by_role = {}
        if hasattr(self.owner, "_reset_loc_methods"):
            self.owner._reset_loc_methods()
        self.owner._clear_training_roi_review_state()
        self.owner.ref_image = None
        self.owner._shape_match_ms_by_image = {}
        self.owner._shape_autogen_ms_by_image = {}
        self.owner.train_files = []
        self.owner.ok_files = []
        self.owner.ng_files = []
        self.owner.test_files = []
        self.owner._sample_roi_annotations_by_path = {}
        self.owner._current_result_rows = []
        self.owner._roi_results_by_image = {}

    def apply_product_switch(self, name: str) -> None:
        self.owner.session.switch_product(name)
        self.owner.session.save_products()
        self.sync_camera_settings_store_path()

        self._clear_transient_product_state()
        self.owner.inspection_items = []

        self.owner.table.setRowCount(0)
        self.owner.canvas.clear_image()
        self.owner.lbl_ref.setText(tr("debug.reference_image_not_set"))
        self.owner.lbl_ref.setToolTip("")
        self.owner.lbl_status.setText(tr("debug.status_switched_product"))

        self.owner.load_session()
        self.owner._refresh_lists()

    def reset_for_clear(self) -> None:
        self._clear_transient_product_state()
        self.owner.lbl_ref.setText(tr("debug.reference_image_not_set"))
        self.owner.lbl_status.setText(tr("debug.status_untrained"))
        self.owner.table.setRowCount(0)
        self.owner._refresh_lists()
        self.owner.session.delete_session_file()
        self.owner._delete_sample_annotation_file()
        self.owner._reload_inspection_items()
        self.owner._sync_footer()

    def save_session(self) -> None:
        self.owner.session.save_session(SessionData(
            train_files=list(self.owner.train_files),
            ok_files=list(self.owner.ok_files),
            ng_files=list(self.owner.ng_files),
            test_files=list(self.owner.test_files),
            ref_image=self.owner.ref_image,
            loc_method=(
                self.owner.loc_method_for_role("cam1")
                if hasattr(self.owner, "loc_method_for_role")
                else self.owner.loc_method
            ),
            loc_methods=(
                dict(getattr(self.owner, "_loc_methods_by_role", {}) or {})
                if hasattr(self.owner, "_loc_methods_by_role")
                else {}
            ),
        ))

    def on_product_changed(self, product_name: str) -> None:
        if not product_name or product_name == self.owner.session.current_product:
            return
        self.save_session()
        self.owner.productChangeRequested.emit(product_name)

    def new_product(self) -> None:
        top_level = self.owner.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("product.create", "新增产品"):
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self.owner,
            tr("debug.new_product_title"),
            tr("debug.new_product_prompt"),
        )
        if not ok or not name.strip():
            return
        product_name = name.strip()
        error = self.owner.session.create_product(product_name)
        if error:
            QtWidgets.QMessageBox.warning(self.owner, tr("common.error"), error)
            return
        audit_event = getattr(top_level, "_audit_event", None)
        if callable(audit_event):
            audit_event(module="产品", action="新增产品", after_value=product_name, product_name=product_name)
        self.owner.cmb_product.addItem(product_name)
        self.owner.cmb_product.setCurrentText(product_name)

    def request_delete_product(self) -> None:
        top_level = self.owner.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("product.delete", "删除产品"):
            return
        name = str(self.owner.cmb_product.currentText() or self.owner.session.current_product or "").strip()
        if not name:
            return
        self.owner.productDeleteRequested.emit(name)

    def clear_session(self) -> None:
        result = QtWidgets.QMessageBox.question(
            self.owner,
            tr("debug.clear_session_title"),
            tr("debug.clear_session_confirm"),
        )
        if result != QtWidgets.QMessageBox.Yes:
            return
        self.owner.sessionClearRequested.emit()
