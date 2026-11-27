"""Reference image and shape template workflow helpers for ToolPage."""

from __future__ import annotations

import copy
import os

from PySide6 import QtCore, QtWidgets

from common.app_logging import get_app_logger
from shape.core import locator as shape_locator
from ui.i18n import tr


LOGGER = get_app_logger(__name__)


def _same_product_path(first: str, second: str) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))

def _set_reference(self, path: str) -> None:
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("template.edit_roi", "设置参考图"):
        return
    camera_role = self.current_camera_role()
    from ui.debug.tool_page.lens_correction_flow import prepare_sample
    try:
        path = prepare_sample(self, path, camera_role=camera_role)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(self, tr("common.error"), str(exc))
        return
    before = str(self.ref_image or "")
    reference_changed = False
    try:
        source_recipe = self.shape_recipe_for_role(camera_role) or shape_locator.load_recipe_for_product(
            self.session.product_dir,
            camera_role,
        )
        recipe = copy.deepcopy(source_recipe)
        previous_reference = str(recipe.reference_image or "")
        reference_changed = (
            (os.path.normcase(os.path.abspath(previous_reference)) if previous_reference else "")
            != (os.path.normcase(os.path.abspath(path)) if path else "")
        )
        if reference_changed:
            for group in recipe.localization_groups or []:
                group["needs_rebuild"] = True
        recipe.reference_image = path
        recipe.model_path = self.shape_model_path_for_role(camera_role)
        shape_locator.save_recipe_for_product(self.session.product_dir, recipe, camera_role)
    except Exception as exc:
        LOGGER.exception("Failed to persist reference image %s: %s", path, exc)
        status = getattr(self, "lbl_status", None)
        if status is not None:
            status.setText(f"Status: reference image was not saved: {exc}")
        return
    self._clear_training_roi_review_state(camera_role)
    self.ref_image = path
    if self.lbl_ref is not None:
        self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(path)}")
        self.lbl_ref.setToolTip(path)
    self._shape_recipes_by_role[camera_role] = recipe
    self.shape_recipe = recipe
    self._save_session()
    if reference_changed:
        changed_signal = getattr(self, "inspectionItemsChanged", None)
        if changed_signal is not None:
            changed_signal.emit()
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
    camera_role = self.current_camera_role()
    product_dir = str(self.session.product_dir)
    existing = self._template_editor_dialog
    if existing is not None:
        same_context = (
            _same_product_path(str(existing.product_dir), product_dir)
            and str(existing.camera_role) == str(camera_role)
        )
        if same_context:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        if not existing.close():
            return
        self._template_editor_dialog = None
    initial = self.ref_image or self.canvas.image_path() or ""
    dlg = ShapeTemplateDialog(
        product_name=self.session.current_product,
        product_dir=product_dir,
        camera_role=camera_role,
        initial_image_path=initial,
        parent=self.window(),
    )
    dlg.setModal(False)
    dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.modelSaved.connect(
        lambda model_path, recipe_path, role=camera_role, product=product_dir:
        self._on_shape_model_saved(model_path, recipe_path, camera_role=role, product_dir=product)
    )
    dlg.referenceRegionsChanged.connect(
        lambda role=camera_role, product=product_dir:
        self._on_shape_reference_regions_changed(camera_role=role, product_dir=product)
    )
    dlg.destroyed.connect(lambda *_args, dialog=dlg: self._on_template_editor_dialog_destroyed(dialog))
    self._template_editor_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def _on_template_editor_dialog_destroyed(self, dialog=None, *_args) -> None:
    if dialog is None or self._template_editor_dialog is dialog:
        self._template_editor_dialog = None

def _on_shape_model_saved(self, model_path: str, recipe_path: str, *, camera_role=None, product_dir: str = "") -> None:
    shape_locator.invalidate_shape_model_cache(model_path)
    if product_dir and not _same_product_path(product_dir, str(self.session.product_dir)):
        return
    camera_role = str(camera_role or self.current_camera_role())
    if hasattr(self, "_set_loc_method_for_role"):
        self._set_loc_method_for_role(camera_role, "shape")
    self._save_session()
    self._clear_training_roi_review_state(camera_role)
    try:
        recipe = self.shape_recipe_for_role(camera_role, force_reload=True)
        if camera_role == self.current_camera_role():
            self.shape_recipe = recipe
    except Exception:
        if camera_role == self.current_camera_role():
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

def _on_shape_reference_regions_changed(self, *, camera_role=None, product_dir: str = "") -> None:
    if product_dir and not _same_product_path(product_dir, str(self.session.product_dir)):
        return
    camera_role = str(camera_role or self.current_camera_role())
    self._clear_training_roi_review_state(camera_role)
    self._sync_shape_recipe_and_items(camera_role)
    if camera_role == self.current_camera_role():
        current_path = self.canvas.image_path()
        if current_path and os.path.exists(current_path):
            self._load_canvas_image(current_path)
    self.lbl_status.setText("Status: reference ROI synchronized to runtime")
    audit_event = getattr(self.window(), "_audit_event", None)
    if callable(audit_event):
        audit_event(
            module="模板ROI",
            action="修改模板区域",
            target=str(camera_role),
        )

def _sync_shape_recipe_and_items(self, camera_role=None) -> None:
    camera_role = str(camera_role or self.current_camera_role())
    try:
        recipe = self.shape_recipe_for_role(camera_role, force_reload=True)
        if camera_role == self.current_camera_role():
            self.shape_recipe = recipe
    except Exception:
        if camera_role == self.current_camera_role():
            self.shape_recipe = None
    self._reload_inspection_items()


def _update_loc_ui(self) -> None:
    method_getter = getattr(self, "loc_method_for_role", None)
    method = method_getter(self.current_camera_role()) if callable(method_getter) else self.loc_method
    shape_controls_enabled = method == "shape"
    for widget_name in ("btn_set_ref", "btn_pick_ref"):
        widget = getattr(self, widget_name, None)
        if widget is not None:
            widget.setEnabled(shape_controls_enabled)

def _on_loc_method_changed(self, method: str) -> None:
    method_text = str(method or "").strip().lower()
    if method_text == "line2dup":
        method_text = "shape"
    if method_text not in {"shape", "ncc"}:
        method_text = "shape"
    role = self.current_camera_role()
    current_getter = getattr(self, "loc_method_for_role", None)
    current_method = current_getter(role) if callable(current_getter) else self.loc_method
    if method_text == str(current_method or ""):
        self._update_loc_ui()
        return
    require_permission = getattr(self.window(), "_require_permission", None)
    if callable(require_permission) and not require_permission("template.edit_params", "修改定位方式"):
        sync_combo = getattr(self, "_sync_loc_combo", None)
        if callable(sync_combo):
            sync_combo()
        else:
            blocker = QtCore.QSignalBlocker(self.cmb_loc)
            self.cmb_loc.setCurrentText(self.loc_method)
            del blocker
        return
    before = str(current_method or "")
    setter = getattr(self, "_set_loc_method_for_role", None)
    if callable(setter):
        setter(role, method_text)
    else:
        self.loc_method = method_text
    self._clear_training_roi_review_state(role)
    self._update_loc_ui()
    self._apply_current_role_recipe_state()
    self._save_session()
    self._reload_inspection_items()
    audit_event = getattr(self.window(), "_audit_event", None)
    if callable(audit_event) and before != method_text:
        audit_event(
            module="模板参数",
            action="修改定位方式",
            target=str(role),
            before_value=before,
            after_value=method_text,
        )

