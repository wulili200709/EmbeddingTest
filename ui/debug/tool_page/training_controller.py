from __future__ import annotations

import re
from typing import List

from PySide6 import QtCore, QtWidgets

from domain import InspectionItem
from ui.debug.tool_page.camera_roles import normalize_camera_role
from ui.debug.tool_page.training_roi_review import TrainingRoiReview
from ui.debug.tool_page.training_task_builder import TrainingTaskBuilder
from ui.i18n import tr


class TrainingController:
    def __init__(self, owner, worker_cls) -> None:
        self.owner = owner
        self.worker_cls = worker_cls
        self.task_builder = TrainingTaskBuilder(owner)
        self.roi_review = TrainingRoiReview(owner, self.task_builder)

    @staticmethod
    def item_display_name(inspection_item: InspectionItem) -> str:
        return TrainingTaskBuilder.item_display_name(inspection_item)

    def resolve_algorithm(self, inspection_item: InspectionItem) -> str:
        return self.task_builder.resolve_algorithm(inspection_item)

    def train_sample_paths_for_role(self, camera_role: object = None) -> List[str]:
        return self.task_builder.train_sample_paths_for_role(camera_role)

    def sample_groups_for_role(self, camera_role: object = None, *, roi_label: object = None) -> tuple[List[str], List[str], List[str]]:
        return self.task_builder.sample_groups_for_role(camera_role, roi_label=roi_label)

    def ready_signature(self, camera_role: object = None) -> str:
        return self.task_builder.ready_signature(camera_role)

    def refresh_current_image_after_roi_update(self, candidate_paths: List[str]) -> None:
        self.roi_review.refresh_current_image_after_roi_update(candidate_paths)

    def clear_review_state(self, camera_role: object = None) -> None:
        self.roi_review.clear_review_state(camera_role)

    def sync_action_buttons(self) -> None:
        self.roi_review.sync_action_buttons()

    def cancel_pending_action(self, action_key: str | None = None) -> None:
        self.roi_review.cancel_pending_action(action_key)

    def ensure_roi_reviewed(self, camera_role: object, *, action_name: str, action_key: str) -> bool:
        return self.roi_review.ensure_roi_reviewed(camera_role, action_name=action_name, action_key=action_key)

    def missing_training_roi_paths(self, roi_label: str, candidate_paths: List[str]) -> List[str]:
        return self.task_builder.missing_training_roi_paths(roi_label, candidate_paths)

    def build_task_for_item(self, inspection_item: InspectionItem) -> dict:
        return self.task_builder.build_task_for_item(inspection_item)

    def payload(self, mode: str, tasks: List[dict], *, selected_item_id: str = "", failures: List[str] | None = None) -> dict:
        return self.task_builder.payload(mode, tasks, selected_item_id=selected_item_id, failures=failures)

    def set_running(self, running: bool) -> None:
        self.owner._training_in_progress = bool(running)
        for attr in ("btn_train", "btn_train_current", "btn_test", "btn_export_test", "btn_clear_session", "btn_export_onnx"):
            button = getattr(self.owner, attr, None)
            if button is not None:
                button.setEnabled(not running)
        progress_bar = getattr(self.owner, "training_progress_bar", None)
        if progress_bar is not None:
            progress_bar.setVisible(bool(running))
            if running:
                progress_bar.setRange(0, 0)
                progress_bar.setFormat("training...")
            else:
                progress_bar.setRange(0, 100)
                progress_bar.setValue(0)
                progress_bar.setFormat("")

    def on_progress(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        status_text = f"Status: {text}"
        status_label = getattr(self.owner, "lbl_status", None)
        if status_label is not None:
            status_label.setText(status_text)
        validation_label = getattr(self.owner, "lbl_training_validation", None)
        if validation_label is not None:
            validation_label.setText(status_text)
        progress_bar = getattr(self.owner, "training_progress_bar", None)
        if progress_bar is not None:
            progress_bar.setVisible(True)
            match = re.search(r"\((\d+)/(\d+)\)", text)
            if match is None:
                match = re.search(r"\b(?:OK|NG)\s+(\d+)/(\d+)\b", text, flags=re.IGNORECASE)
            if match is not None:
                current = max(0, int(match.group(1)))
                total = max(1, int(match.group(2)))
                percent = max(0, min(100, int(round(current * 100.0 / total))))
                progress_bar.setRange(0, 100)
                progress_bar.setValue(percent)
                progress_bar.setFormat(f"{current}/{total}  {percent}%")
            else:
                progress_bar.setRange(0, 0)
                progress_bar.setFormat(text[:48])

    def start_worker(self, payload: dict) -> None:
        if getattr(self.owner, "_training_in_progress", False):
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), "Training is already running.")
            return

        self.set_running(True)
        self.on_progress("training queued")

        if not isinstance(self.owner, QtCore.QObject):
            worker = self.worker_cls(self.owner.algo, payload)
            result = worker._run_training()
            self.on_finished(result)
            return

        thread = QtCore.QThread(self.owner)
        worker = self.worker_cls(self.owner.algo, payload)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(self.owner._on_training_progress)
        worker.finished.connect(self.owner._on_training_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self.owner._forget_training_job(thread, worker))
        self.owner._training_thread = thread
        self.owner._training_worker = worker
        thread.start()

    def forget_job(self, thread: QtCore.QThread, worker) -> None:
        if self.owner._training_thread is thread:
            self.owner._training_thread = None
        if self.owner._training_worker is worker:
            self.owner._training_worker = None

    def on_finished(self, payload: object) -> None:
        result = dict(payload or {}) if isinstance(payload, dict) else {}
        self.set_running(False)

        last_status_message = str(result.get("last_status_message", "") or "")
        if last_status_message:
            self.owner.lbl_status.setText(last_status_message)
            if hasattr(self.owner, "lbl_training_validation"):
                self.owner.lbl_training_validation.setText(last_status_message)

        success_names = [str(name) for name in result.get("success_names", []) if str(name)]
        failure_messages = [str(message) for message in result.get("failure_messages", []) if str(message)]
        mode = str(result.get("mode", "") or "")
        audit_event = getattr(self.owner.window(), "_audit_event", None)
        if callable(audit_event) and (success_names or failure_messages):
            if failure_messages and success_names:
                audit_result = "部分成功"
            elif failure_messages:
                audit_result = "失败"
            else:
                audit_result = "成功"
            audit_event(
                module="模型训练",
                action="重新训练",
                target=str(self.owner.current_camera_role()),
                after_value=f"mode={mode}, success={len(success_names)}, failed={len(failure_messages)}",
                result=audit_result,
                remark="\n".join(failure_messages[:10]),
            )

        display_rows = list(result.get("display_rows", []) or [])
        if display_rows:
            self.owner._populate_results_table(display_rows)

        if success_names:
            try:
                self.owner._save_runtime_params()
            except Exception as exc:
                self.owner.lbl_status.setText(f"Status: training done, but saving runtime params failed: {exc}")
            self.owner._save_session()
        self.owner._refresh_inspection_items_table()
        self.owner._update_runtime_widgets()

        if failure_messages:
            summary_lines: List[str] = []
            if success_names:
                summary_lines.append(f"Succeeded: {len(success_names)} tool(s) - " + ", ".join(success_names))
            summary_lines.append(f"Failed: {len(failure_messages)} tool(s)")
            summary_lines.extend(failure_messages[:20])
            self.owner.lbl_status.setText(
                f"Status: partial train done, success={len(success_names)}, failed={len(failure_messages)}"
            )
            QtWidgets.QMessageBox.warning(self.owner, tr("debug.train_result_title"), "\n".join(summary_lines))
            return

        if mode == "current":
            dialog_message = str(result.get("last_dialog_message", "") or "").strip()
            QtWidgets.QMessageBox.information(
                self.owner,
                tr("debug.train_done_title"),
                dialog_message or "OK/NG training finished. You can start testing.",
            )
            return

        QtWidgets.QMessageBox.information(
            self.owner,
            tr("debug.train_result_title"),
            f"Finished training/calibrating {len(success_names)} enabled tool(s).",
        )

    def train_inspection_item(self, inspection_item: InspectionItem):
        task = self.build_task_for_item(inspection_item)
        algorithm = str(task.get("algorithm", "") or "")
        roi_label = str((task.get("label_names") or ["roi"])[0] or "roi")
        self.owner.algo.product_params.algorithm = algorithm
        self.owner.algo.product_params.score_mode = self.owner.cmb_mode.currentText()
        self.owner.algo.product_params.margin = float(self.owner.spin_margin.value())
        self.owner.algo.product_params.topk = int(self.owner.spin_topk.value())
        return self.owner.algo.train(
            list(task.get("ok_files", []) or []),
            list(task.get("ng_files", []) or []),
            algorithm=algorithm,
            product_dir=self.owner.session.product_dir,
            label_names=[roi_label],
            model_key=str(task.get("model_key", "") or ""),
        )

    def train_all_tools(self) -> None:
        self.owner.algo.model = None
        self.owner.table.setRowCount(0)
        self.owner._current_result_rows = []
        if getattr(self.owner, "_training_in_progress", False):
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), "Training is already running.")
            return
        if self.owner._warn_mixed_training_camera_samples(self.owner.current_camera_role()):
            return
        if not self.owner._ensure_training_roi_reviewed(
            self.owner.current_camera_role(),
            action_name=tr("debug.train_all_tools"),
            action_key="all",
        ):
            return

        current_role = self.owner.current_camera_role()
        enabled_items = [
            item
            for item in self.owner.inspection_items
            if item.enabled and normalize_camera_role(getattr(item, "camera_id", "")) == current_role
        ]
        if not enabled_items:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.enable_one_tool", role=current_role))
            return

        selected_item = self.owner._selected_inspection_item()
        selected_item_id = str(selected_item.item_id or "") if selected_item is not None else ""
        tasks: List[dict] = []
        failure_messages: List[str] = []
        for inspection_item in enabled_items:
            display_name = self.item_display_name(inspection_item)
            try:
                tasks.append(self.build_task_for_item(inspection_item))
            except Exception as exc:
                failure_messages.append(f"{display_name}: {exc}")

        if not tasks:
            if failure_messages:
                audit_event = getattr(self.owner.window(), "_audit_event", None)
                if callable(audit_event):
                    audit_event(
                        module="模型训练",
                        action="重新训练",
                        target=str(current_role),
                        after_value=f"mode=all, success=0, failed={len(failure_messages)}",
                        result="失败",
                        remark="\n".join(failure_messages[:10]),
                    )
                self.owner.lbl_status.setText("Status: train failed")
                QtWidgets.QMessageBox.warning(
                    self.owner,
                    tr("debug.train_result_title"),
                    "\n".join([f"Failed: {len(failure_messages)} tool(s)", *failure_messages[:20]]),
                )
            else:
                QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.enable_one_tool", role=current_role))
            return

        self.start_worker(self.payload("all", tasks, selected_item_id=selected_item_id, failures=failure_messages))

    def train_current(self) -> None:
        self.owner.algo.model = None
        self.owner.table.setRowCount(0)
        self.owner._current_result_rows = []
        if getattr(self.owner, "_training_in_progress", False):
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), "Training is already running.")
            return
        inspection_item = self.owner._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.select_inspection_tool_in_table"))
            return
        if self.owner._warn_mixed_training_camera_samples(inspection_item.camera_id):
            return
        if not self.owner._ensure_training_roi_reviewed(
            inspection_item.camera_id,
            action_name=tr("debug.calibrate_current_tool"),
            action_key="current",
        ):
            return
        if not inspection_item.enabled:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.tool_disabled"))
            return
        try:
            task = self.build_task_for_item(inspection_item)
        except Exception as exc:
            audit_event = getattr(self.owner.window(), "_audit_event", None)
            if callable(audit_event):
                audit_event(
                    module="模型训练",
                    action="重新训练",
                    target=str(inspection_item.camera_id),
                    after_value="mode=current, success=0, failed=1",
                    result="失败",
                    remark=str(exc),
                )
            QtWidgets.QMessageBox.warning(self.owner, tr("debug.train_failed_title"), str(exc))
            return

        selected_item_id = str(inspection_item.item_id or "")
        self.start_worker(self.payload("current", [task], selected_item_id=selected_item_id))
