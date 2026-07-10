"""Reference image and shape template workflow helpers for ToolPage."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from common.app_logging import get_app_logger
from shape.core import locator as shape_locator
from ui.i18n import tr


LOGGER = get_app_logger(__name__)

def _set_reference(self, path: str) -> None:
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("template.edit_roi", "设置参考图"):
        return
    camera_role = self.current_camera_role()
    before = str(self.ref_image or "")
    self._clear_training_roi_review_state(camera_role)
    self.ref_image = path
    if self.lbl_ref is not None:
        self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(path)}")
        self.lbl_ref.setToolTip(path)
    try:
        recipe = self.shape_recipe_for_role(camera_role) or shape_locator.load_recipe_for_product(
            self.session.product_dir,
            camera_role,
        )
        recipe.reference_image = path
        recipe.model_path = self.shape_model_path_for_role(camera_role)
        shape_locator.save_recipe_for_product(self.session.product_dir, recipe, camera_role)
        self._shape_recipes_by_role[camera_role] = recipe
        self.shape_recipe = recipe
    except Exception as exc:
        LOGGER.exception("Failed to persist reference image %s: %s", path, exc)
    self._save_session()
    audit_event = getattr(self.window(), "_audit_event", None)
    if callable(audit_event):
        audit_event(
            module="模板ROI",
            action="设置参考图",
            target=str(camera_role),
            before_value=before,
            after_value=str(path or ""),
        )

def _set_ref_from_current(self) -> None:
    p = self.canvas.image_path()
    if not p:
        QtWidgets.QMessageBox.warning(self, "Info", "Open an image from the right-side list first")
        return
    self._set_reference(p)

def _pick_ref_image(self) -> None:
    p, _ = QtWidgets.QFileDialog.getOpenFileName(
        self, tr("auto.pick_reference_title"), "",
        "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
    )
    if not p:
        return
    self._set_reference(p)

def _open_shape_template_page(self) -> None:
    from ui.debug import ShapeTemplateDialog

    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("template.edit_roi", "模板区域设置"):
        return
    if self._template_editor_dialog is not None and self._template_editor_dialog.isVisible():
        self._template_editor_dialog.raise_()
        self._template_editor_dialog.activateWindow()
        return
    initial = self.ref_image or self.canvas.image_path() or ""
    dlg = ShapeTemplateDialog(
        product_name=self.session.current_product,
        product_dir=self.session.product_dir,
        camera_role=self.current_camera_role(),
        initial_image_path=initial,
        parent=self.window(),
    )
    dlg.setModal(False)
    dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.modelSaved.connect(self._on_shape_model_saved)
    dlg.referenceRegionsChanged.connect(self._on_shape_reference_regions_changed)
    dlg.destroyed.connect(self._on_template_editor_dialog_destroyed)
    self._template_editor_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def _on_template_editor_dialog_destroyed(self, *_args) -> None:
    self._template_editor_dialog = None

def _on_shape_model_saved(self, model_path: str, recipe_path: str) -> None:
    camera_role = self.current_camera_role()
    self._clear_training_roi_review_state(camera_role)
    try:
        self.shape_recipe = self.shape_recipe_for_role(camera_role, force_reload=True)
    except Exception:
        self.shape_recipe = None
    self._reload_inspection_items()
    self._apply_current_role_recipe_state()
    self.lbl_status.setText(f"Status: template model saved {os.path.basename(model_path)}")
    audit_event = getattr(self.window(), "_audit_event", None)
    if callable(audit_event):
        audit_event(
            module="模板ROI",
            action="保存模板模型",
            target=str(camera_role),
            after_value=f"model={os.path.basename(model_path)}, recipe={os.path.basename(recipe_path)}",
        )

def _on_shape_reference_regions_changed(self) -> None:
    self._clear_training_roi_review_state(self.current_camera_role())
    self._sync_shape_recipe_and_items()
    current_path = self.canvas.image_path()
    if current_path and os.path.exists(current_path):
        self._load_canvas_image(current_path)
    self.lbl_status.setText("Status: reference ROI synchronized to runtime")
    audit_event = getattr(self.window(), "_audit_event", None)
    if callable(audit_event):
        audit_event(
            module="模板ROI",
            action="修改模板区域",
            target=str(self.current_camera_role()),
        )

def _sync_shape_recipe_and_items(self) -> None:
    try:
        self.shape_recipe = self.shape_recipe_for_role(self.current_camera_role(), force_reload=True)
    except Exception:
        self.shape_recipe = None
    self._reload_inspection_items()


def _update_loc_ui(self) -> None:
    return None

def _on_loc_method_changed(self, method: str) -> None:
    if str(method or "") == str(self.loc_method or ""):
        self._update_loc_ui()
        return
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("template.edit_params", "修改定位方式"):
        blocker = QtCore.QSignalBlocker(self.cmb_loc)
        self.cmb_loc.setCurrentText(self.loc_method)
        del blocker
        return
    before = str(self.loc_method or "")
    self.loc_method = method
    self._clear_training_roi_review_state()
    self._update_loc_ui()
    self._save_session()
    audit_event = getattr(self.window(), "_audit_event", None)
    if callable(audit_event) and before != str(method or ""):
        audit_event(
            module="模板参数",
            action="修改定位方式",
            before_value=before,
            after_value=str(method or ""),
        )

