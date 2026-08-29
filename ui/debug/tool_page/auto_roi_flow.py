"""Auto ROI workflow helpers for ToolPage."""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6 import QtWidgets

from common import labelme_io
from common.app_logging import get_app_logger

from application.auto_roi_service import (
    AutoRoiIssue,
    missing_roi_files as service_missing_roi_files,
    run_auto_roi_batch,
    validate_autogen_reference,
)
from ui.debug.tool_page.camera_roles import filter_paths_for_camera as _filter_paths_for_camera
from ui.i18n import tr

LOGGER = get_app_logger(__name__)


def _auto_roi_issue_text(issue: AutoRoiIssue) -> str:
    if issue.message_key:
        try:
            return tr(issue.message_key, **dict(issue.message_args or {}))
        except Exception:
            pass
    return issue.fallback or issue.message_key or ""


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
    labels = self._loc_output_labels(camera_role)
    missing = service_missing_roi_files(paths, labels)
    if not missing:
        if not silent:
            QtWidgets.QMessageBox.information(
                self, tr("common.info"), tr("auto.images_already_have_roi")
            )
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
    if not silent:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_roi", "自动生成ROI"):
            return
    if not paths:
        if not silent:
            QtWidgets.QMessageBox.information(
                self, tr("common.info"), tr("debug.no_image_to_process")
            )
        return
    ref_image = self.ref_image
    labels: List[str] = ["roi"]
    role = self.current_camera_role() if camera_role is None else str(camera_role)
    method_getter = getattr(self, "loc_method_for_role", None)
    method = method_getter(role) if callable(method_getter) else self.loc_method
    if method == "shape":
        try:
            recipe = self.shape_recipe_for_role(role, force_reload=True)
            if role == self.current_camera_role():
                self.shape_recipe = recipe
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                tr("common.info"),
                tr("debug.template_recipe_failed", error=exc),
            )
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
        labels = self._shape_output_labels(role)
        validation = validate_autogen_reference(
            method=method,
            ref_image=ref_image,
            shape_model_path=self.shape_model_path_for_role(role),
            shape_labels=labels,
            reference_regions=recipe.reference_regions,
        )
    elif method == "ncc":
        labels = self._ncc_output_labels(role)
        validation = validate_autogen_reference(
            method=method,
            ref_image="",
            ncc_model_path=self.ncc_model_path_for_role(role),
            ncc_labels=labels,
        )
    else:
        validation = validate_autogen_reference(method=method, ref_image=ref_image)

    if not validation.ok:
        issue = validation.issue
        QtWidgets.QMessageBox.warning(
            self,
            tr("common.info"),
            _auto_roi_issue_text(issue) if issue is not None else tr("common.error"),
        )
        return
    labels = validation.labels or labels

    todo = self._resolve_autogen_targets(paths, only_missing=only_missing, silent=silent, camera_role=role)
    if not todo:
        if getattr(self, "_skip_empty_autogen_message", False):
            self._skip_empty_autogen_message = False
            return
        if not silent:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("auto.images_already_have_roi"))
        return

    result = run_auto_roi_batch(
        paths=todo,
        labels=labels,
        method=method,
        ref_image=ref_image,
        product_dir=self.session.product_dir,
        camera_role=role,
        only_missing=only_missing,
        pre_resolved=True,
    )
    ok = int(result.get("ok", 0) or 0)
    errs = [str(err) for err in result.get("errs", []) or [] if str(err)]
    for path, elapsed_ms in dict(result.get("timings", {}) or {}).items():
        try:
            value = float(elapsed_ms)
        except Exception:
            continue
        self._shape_match_ms_by_image[str(path)] = value
        self._shape_autogen_ms_by_image[str(path)] = value

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
        if not silent:
            audit_event = getattr(self.window(), "_audit_event", None)
            if callable(audit_event):
                audit_event(
                    module="模板ROI",
                    action="自动生成ROI",
                    target=str(role),
                    after_value=f"images={ok}, labels={','.join(labels)}",
                    result="成功" if not errs else "部分成功",
                    remark="\n".join(errs[:10]),
                )

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
    if not paths:
        self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())
        return
    current_role = self.current_camera_role()
    try:
        processed = False
        for role in self.configured_camera_roles():
            role_paths = _filter_paths_for_camera(self, paths, role)
            if not role_paths:
                continue
            processed = True
            self._autogen_roi_for_images(
                role_paths,
                only_missing=self.chk_only_missing.isChecked(),
                camera_role=role,
            )
        if not processed:
            self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())
    finally:
        self._set_current_camera_role(current_role, sync_debug_role=False)

def _clear_roi_for_images(
    self,
    paths: List[str],
    *,
    labels: Optional[List[str]] = None,
    silent: bool = False,
    camera_role=None,
) -> None:
    if not silent:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_roi", "清除ROI"):
            return
    if not paths:
        if not silent:
            QtWidgets.QMessageBox.information(
                self, tr("common.info"), tr("debug.no_image_to_process")
            )
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
            except Exception as exc:
                LOGGER.exception("Failed to delete ROI label %s from %s: %s", label, path, exc)
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
            audit_event = getattr(self.window(), "_audit_event", None)
            if callable(audit_event):
                audit_event(
                    module="模板ROI",
                    action="批量清除ROI",
                    target=str(role),
                    before_value=f"images={touched}, labels={','.join(labels)}",
                )

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



