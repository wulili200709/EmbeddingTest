from __future__ import annotations

from typing import List

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
        self.owner._training_roi_ready_signatures.pop(role, None)
        self.owner._training_roi_confirmed_signatures.pop(role, None)
        self.owner._update_runtime_widgets()
        action_text = tr("debug.train_all_tools") if pending_action == "all" else tr("debug.calibrate_current_tool")
        self.owner.lbl_status.setText(tr("debug.cancelled_confirm", action=action_text))

    def ensure_roi_reviewed(
        self,
        camera_role: object,
        *,
        action_name: str,
        action_key: str,
        confirmation_token: str = "",
    ) -> bool:
        """Require a second click without running localization or changing ROI.

        Training is read-only with respect to sample ROI data. Template/NCC
        localization belongs to the explicit Auto ROI workflow; generated ROI
        geometry cannot supply the required human OK/NG annotation.
        """
        role = normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        token = str(confirmation_token or action_key or action_name).strip()
        pending_action = self.owner._training_roi_pending_actions.get(role, "")
        pending_token = self.owner._training_roi_ready_signatures.get(role, "")
        if pending_action == action_key and pending_token == token:
            self.owner._training_roi_ready_signatures.pop(role, None)
            self.owner._training_roi_pending_actions.pop(role, None)
            self.owner._training_roi_confirmed_signatures.pop(role, None)
            self.owner._update_runtime_widgets()
            return True

        self.owner._training_roi_confirmed_signatures.pop(role, None)
        self.owner._training_roi_ready_signatures[role] = token
        self.owner._training_roi_pending_actions[role] = action_key
        self.owner._update_runtime_widgets()
        self.owner.lbl_status.setText(
            tr("debug.training_confirmation_requested", action=action_name)
        )
        return False
