from __future__ import annotations

import os
from typing import List

from PySide6 import QtWidgets

from ncc import locator as ncc_locator
from shape.core import locator as shape_locator
from ui.debug.tool_page.camera_roles import normalize_camera_role
from ui.i18n import tr


class TrainingRoiReview:
    def __init__(self, owner, task_builder) -> None:
        self.owner = owner
        self.task_builder = task_builder

    def refresh_current_image_after_roi_update(self, candidate_paths: List[str]) -> None:
        current_path = self.owner.canvas.image_path()
        if current_path and current_path in set(candidate_paths):
            self.owner._load_canvas_image(current_path)
            self.owner._set_status_for_current_image(current_path)

    def clear_review_state(self, camera_role: object = None) -> None:
        if camera_role is None:
            self.owner._training_roi_ready_signatures = {}
            self.owner._training_roi_pending_actions = {}
            self.owner._training_roi_confirmed_signatures = {}
        else:
            role = normalize_camera_role(camera_role) or "cam1"
            self.owner._training_roi_ready_signatures.pop(role, None)
            self.owner._training_roi_pending_actions.pop(role, None)
            self.owner._training_roi_confirmed_signatures.pop(role, None)
        self.owner._update_runtime_widgets()

    def sync_action_buttons(self) -> None:
        train_button = getattr(self.owner, "btn_train", None)
        train_current_button = getattr(self.owner, "btn_train_current", None)
        cancel_train_button = getattr(self.owner, "btn_train_cancel", None)
        cancel_current_button = getattr(self.owner, "btn_train_current_cancel", None)
        if train_button is None or train_current_button is None:
            return

        current_role = self.owner.current_camera_role()
        pending_action = getattr(self.owner, "_training_roi_pending_actions", {}).get(current_role, "")
        default_train_text = tr("debug.train_all_tools")
        default_current_text = tr("debug.calibrate_current_tool")
        default_train_style = getattr(self.owner, "_train_action_btn_style", "")
        default_current_style = getattr(self.owner, "_train_current_btn_style", "")
        confirm_style = getattr(self.owner, "_train_confirm_btn_style", default_train_style)

        if pending_action == "all":
            train_button.setText(tr("debug.confirm_train_all"))
            train_button.setStyleSheet(confirm_style)
            if cancel_train_button is not None:
                cancel_train_button.setVisible(True)
        else:
            train_button.setText(default_train_text)
            train_button.setStyleSheet(default_train_style)
            if cancel_train_button is not None:
                cancel_train_button.setVisible(False)

        if pending_action == "current":
            train_current_button.setText(tr("debug.confirm_current_tool"))
            train_current_button.setStyleSheet(confirm_style)
            if cancel_current_button is not None:
                cancel_current_button.setVisible(True)
            return

        train_current_button.setText(default_current_text)
        train_current_button.setStyleSheet(default_current_style)
        if cancel_current_button is not None:
            cancel_current_button.setVisible(False)

    def cancel_pending_action(self, action_key: str | None = None) -> None:
        role = self.owner.current_camera_role()
        pending_action = self.owner._training_roi_pending_actions.get(role, "")
        if action_key and pending_action != action_key:
            return
        if not pending_action:
            return
        self.owner._training_roi_pending_actions.pop(role, None)
        self.owner._update_runtime_widgets()
        action_text = tr("debug.train_all_tools") if pending_action == "all" else tr("debug.calibrate_current_tool")
        self.owner.lbl_status.setText(tr("debug.cancelled_confirm", action=action_text))

    def ensure_roi_reviewed(self, camera_role: object, *, action_name: str, action_key: str) -> bool:
        role = normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        method_getter = getattr(self.owner, "loc_method_for_role", None)
        method = method_getter(role) if callable(method_getter) else self.owner.loc_method
        if method not in {"shape", "ncc"}:
            return True
        candidate_paths = self.task_builder.train_sample_paths_for_role(role)
        if not candidate_paths:
            return True

        signature = self.task_builder.ready_signature(role)
        if self.owner._training_roi_confirmed_signatures.get(role) == signature:
            return True
        if (
            self.owner._training_roi_ready_signatures.get(role) == signature
            and self.owner._training_roi_pending_actions.get(role) == action_key
        ):
            self.owner._training_roi_ready_signatures.pop(role, None)
            self.owner._training_roi_pending_actions.pop(role, None)
            self.owner._training_roi_confirmed_signatures[role] = signature
            self.owner._update_runtime_widgets()
            return True

        ok_count = 0
        errors: List[str] = []
        if method == "shape":
            recipe = self.owner.shape_recipe_for_role(role, force_reload=True)
            if recipe is None:
                QtWidgets.QMessageBox.warning(self.owner, tr("common.info"), tr("debug.recipe_missing", role=role))
                return False
            ref_image = self.owner.ref_image
            if recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
            if not ref_image or not os.path.exists(ref_image):
                QtWidgets.QMessageBox.warning(self.owner, tr("common.info"), tr("debug.reference_missing", role=role))
                return False
        else:
            model_path = self.owner.ncc_model_path_for_role(role)
            if not model_path or not os.path.exists(model_path):
                QtWidgets.QMessageBox.warning(self.owner, tr("common.info"), f"NCC model missing for {role}")
                return False
            ref_image = ""
            recipe = None

        for path in candidate_paths:
            try:
                if method == "shape":
                    run = shape_locator.autogen_roi_json_from_shape_timed(
                        tgt_img_path=path,
                        ref_img_path=ref_image,
                        product_dir=self.owner.session.product_dir,
                        camera_role=role,
                    )
                else:
                    run = ncc_locator.autogen_roi_json_from_ncc_timed(
                        tgt_img_path=path,
                        product_dir=self.owner.session.product_dir,
                        camera_role=role,
                    )
                self.owner._shape_match_ms_by_image[path] = float(run.total_ms)
                self.owner._shape_autogen_ms_by_image[path] = float(run.total_ms)
                ok_count += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        self.refresh_current_image_after_roi_update(candidate_paths)

        if errors:
            self.clear_review_state(role)
            QtWidgets.QMessageBox.warning(
                self.owner,
                tr("debug.roi_generate_failed_title"),
                tr("debug.roi_generate_failed_message", errors="\n".join(errors[:20])),
            )
            return False

        self.owner._training_roi_ready_signatures[role] = signature
        self.owner._training_roi_pending_actions[role] = action_key
        self.owner._update_runtime_widgets()
        self.owner.lbl_status.setText(tr("debug.roi_updated_status", role=role, action=action_name))
        QtWidgets.QMessageBox.information(
            self.owner,
            tr("debug.roi_updated_title"),
            tr("debug.roi_updated_message", role=role, count=ok_count, action=action_name),
        )
        return False
