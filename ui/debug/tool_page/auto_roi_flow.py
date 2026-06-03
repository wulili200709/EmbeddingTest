"""Auto ROI workflow helpers for ToolPage."""

from __future__ import annotations

from common import labelme_io

from . import page as page_module
from ui.i18n import tr

List = page_module.List
Optional = page_module.Optional
os = page_module.os
QtCore = page_module.QtCore
QtWidgets = page_module.QtWidgets
qr_core = page_module.qr_core
shape_locator = page_module.shape_locator
_filter_paths_for_camera = page_module._filter_paths_for_camera


def _set_reference(self, path: str) -> None:
    camera_role = self.current_camera_role()
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
    except Exception:
        pass
    self._save_session()

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

def _on_shape_reference_regions_changed(self) -> None:
    self._clear_training_roi_review_state(self.current_camera_role())
    self._sync_shape_recipe_and_items()
    current_path = self.canvas.image_path()
    if current_path and os.path.exists(current_path):
        self._load_canvas_image(current_path)
    self.lbl_status.setText("Status: reference ROI synchronized to runtime")

def _sync_shape_recipe_and_items(self) -> None:
    try:
        self.shape_recipe = self.shape_recipe_for_role(self.current_camera_role(), force_reload=True)
    except Exception:
        self.shape_recipe = None
    self._reload_inspection_items()


def _update_loc_ui(self) -> None:
    return None

def _on_loc_method_changed(self, method: str) -> None:
    self.loc_method = method
    self._clear_training_roi_review_state()
    self._update_loc_ui()
    self._save_session()

def _resolve_autogen_targets(
    self,
    paths: List[str],
    *,
    only_missing: bool,
    silent: bool,
    camera_role=None,
) -> List[str]:
    self._skip_empty_autogen_message = False
    if not paths:
        return []
    missing = self._missing_roi_files(paths, camera_role=camera_role)
    if not missing:
        if not silent:
            QtWidgets.QMessageBox.information(self, "Info", "These images already have ROI.")
            self._skip_empty_autogen_message = True
        return []

    missing_set = set(missing)
    existing = [p for p in paths if p not in missing_set]
    if not existing or silent:
        return list(missing) if only_missing else list(paths)

    default_button = (
        QtWidgets.QMessageBox.StandardButton.No
        if only_missing
        else QtWidgets.QMessageBox.StandardButton.Yes
    )
    reply = QtWidgets.QMessageBox.question(
        self,
        tr("auto.overwrite_title"),
        tr("auto.overwrite_message", count=len(existing)),
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        default_button,
    )
    if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
        self._skip_empty_autogen_message = True
        return []
    if reply == QtWidgets.QMessageBox.StandardButton.No:
        return list(missing)
    return list(paths)

def _autogen_roi_for_images(
    self,
    paths: List[str],
    only_missing: bool,
    silent: bool = False,
    *,
    camera_role=None,
) -> None:
    if not paths:
        if not silent:
            QtWidgets.QMessageBox.information(self, "Info", "No image to process")
        return
    ref_image = self.ref_image
    method = self.loc_method
    role = self.current_camera_role() if camera_role is None else str(camera_role)
    if method == "shape":
        try:
            recipe = self.shape_recipe_for_role(role, force_reload=True)
            if role == self.current_camera_role():
                self.shape_recipe = recipe
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Info", f"Failed to load template recipe: {exc}")
            return
        if recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
            if self.ref_image != ref_image:
                if role == self.current_camera_role():
                    self._set_reference(ref_image)
                else:
                    self.ref_image = ref_image
                    if getattr(self, "lbl_ref", None) is not None:
                        self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(ref_image)}")
                        self.lbl_ref.setToolTip(ref_image)
        if not os.path.exists(self.shape_model_path_for_role(role)):
            QtWidgets.QMessageBox.warning(self, "Info", "Current product has no template model. Create a template first.")
            return
        labels = self._shape_output_labels(role)
        recipe_region_labels = {
            str(region.get("output_label") or region.get("reference_label") or "").strip()
            for region in (recipe.reference_regions or [])
            if isinstance(region, dict)
        }
        recipe_region_labels.discard("")
        if (not ref_image or not os.path.exists(ref_image)) and not recipe_region_labels:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.need_reference_or_saved"))
            return
        if labels:
            missing_labels = [label for label in labels if label not in recipe_region_labels]
            if missing_labels:
                ref_json = labelme_io.labelme_json_of_image(ref_image) if ref_image else ""
                if not ref_json or not os.path.exists(ref_json):
                    QtWidgets.QMessageBox.warning(
                        self,
                        tr("common.info"),
                        tr("auto.missing_reference_json", labels=", ".join(missing_labels)),
                    )
                    return
                missing_labels = [
                    label for label in missing_labels
                    if labelme_io.read_shape_from_labelme(ref_json, label) is None
                ]
                if missing_labels:
                    QtWidgets.QMessageBox.warning(
                        self,
                        tr("common.info"),
                        tr("auto.missing_reference_roi", labels=", ".join(missing_labels)),
                    )
                    return
    else:
        if not ref_image or not os.path.exists(ref_image):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.set_reference_first"))
            return
        ref_json = labelme_io.labelme_json_of_image(ref_image)
        if not os.path.exists(ref_json):
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.reference_missing_json"))
            return
        if labelme_io.try_read_xywh_from_labelme(ref_json, "anchor") is None:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.reference_missing_anchor"))
            return
        if labelme_io.try_read_xywh_from_labelme(ref_json, "roi") is None:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("auto.reference_missing_roi"))
            return

    todo = self._resolve_autogen_targets(paths, only_missing=only_missing, silent=silent, camera_role=role)
    if not todo:
        if getattr(self, "_skip_empty_autogen_message", False):
            self._skip_empty_autogen_message = False
            return
        if not silent:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("auto.images_already_have_roi"))
        return

    ok = 0
    errs: List[str] = []
    for p in todo:
        try:
            if method == "shape":
                run = shape_locator.autogen_roi_json_from_shape_timed(
                    tgt_img_path=p, ref_img_path=ref_image,
                    product_dir=self.session.product_dir,
                    camera_role=role,
                )
                self._shape_match_ms_by_image[p] = float(run.total_ms)
                self._shape_autogen_ms_by_image[p] = float(run.total_ms)
            else:
                qr_core.autogen_roi_json_from_reference(
                    tgt_img_path=p, ref_img_path=ref_image,
                    method=method, anchor_label="anchor", roi_label="roi",
                )
            ok += 1
        except Exception as e:
            errs.append(f"{os.path.basename(p)}: {e}")

    if not silent:
        msg = tr("auto.finished", ok=ok, failed=len(errs))
        if errs:
            msg += "\n\n" + tr("auto.failed_examples") + "\n" + "\n".join(errs[:10])
        QtWidgets.QMessageBox.information(self, tr("common.done"), msg)
        if ok:
            self.lbl_status.setText(tr("auto.status_generated", ok=ok, failed=len(errs)))

    if ok:
        self._reload_inspection_items()
        self.roiGeometryChanged.emit()

    cur = self.canvas.image_path()
    if cur and cur in todo:
        self._load_canvas_image(cur)
        self._set_status_for_current_image(cur)

def _autogen_roi_current_tab(self) -> None:
    tab = self.tabs.currentIndex()
    if tab == 0:
        paths = self._sample_paths_for_kind("train", self.current_camera_role())
    else:
        paths = _filter_paths_for_camera(self, self.test_files, self.current_camera_role())
    self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

def _autogen_roi_all(self) -> None:
    train_files = list(getattr(self, "train_files", []) or [])
    if not train_files:
        train_files = list(getattr(self, "ok_files", []) or []) + list(getattr(self, "ng_files", []) or [])
    paths = list(dict.fromkeys(train_files + list(self.test_files)))
    self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

def _clear_roi_for_images(
    self,
    paths: List[str],
    *,
    labels: Optional[List[str]] = None,
    silent: bool = False,
    camera_role=None,
) -> None:
    if not paths:
        if not silent:
            QtWidgets.QMessageBox.information(self, tr("common.info"), "No image to process")
        return
    role = self.current_camera_role() if camera_role is None else str(camera_role)
    self._clear_training_roi_review_state(role)
    if labels is None:
        labels, _clear_mode = self._clear_roi_labels_for_paths(paths, camera_role=role)
    labels = [str(label).strip() for label in (labels or []) if str(label).strip()] or ["roi"]

    removed = 0
    touched = 0
    for path in paths:
        any_removed = False
        for label in labels:
            try:
                if labelme_io.delete_labelme_shape(path, label):
                    removed += 1
                    any_removed = True
            except Exception:
                pass
        if any_removed:
            touched += 1
            self._shape_match_ms_by_image.pop(path, None)
            self._shape_autogen_ms_by_image.pop(path, None)

    cur = self.canvas.image_path()
    if cur and cur in paths:
        self._load_canvas_image(cur)
        self._set_status_for_current_image(cur)

    if touched:
        self.roiGeometryChanged.emit()

    if not silent:
        QtWidgets.QMessageBox.information(
            self,
            tr("common.done"),
            tr("auto.clear_done", images=touched, labels_count=removed, labels=", ".join(labels)),
        )
        self.lbl_status.setText(tr("auto.status_cleared", images=touched, labels_count=removed))

def _clear_roi_current_tab(self) -> None:
    tab = self.tabs.currentIndex()
    if tab == 0:
        paths = self._sample_paths_for_kind("train", self.current_camera_role())
        tab_name = tr("debug.train_samples")
    else:
        paths = _filter_paths_for_camera(self, self.test_files, self.current_camera_role())
        tab_name = tr("debug.test_samples")

    if not paths:
        QtWidgets.QMessageBox.information(self, tr("common.info"), tr("auto.current_list_empty"))
        return

    labels, clear_mode = self._clear_roi_labels_for_paths(paths)
    if clear_mode == "stale_only":
        action_text = tr("auto.clear_invalid_action")
    elif clear_mode == "all_existing":
        action_text = tr("auto.clear_all_action")
    else:
        action_text = tr("auto.clear_labels_action")
    reply = QtWidgets.QMessageBox.question(
        self,
        tr("auto.clear_title"),
        tr("auto.clear_confirm", tab=tab_name, action=action_text, labels=", ".join(labels)),
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
        return
    self._clear_roi_for_images(paths, labels=labels, silent=False)



