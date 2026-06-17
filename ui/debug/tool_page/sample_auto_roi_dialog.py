from __future__ import annotations

import os
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from application.auto_roi_service import (
    AutoRoiIssue,
    missing_roi_files as service_missing_roi_files,
    run_auto_roi_batch,
    validate_autogen_reference,
)
from ui.i18n import tr


def _auto_roi_issue_text(issue: AutoRoiIssue) -> str:
    if issue.message_key:
        try:
            return tr(issue.message_key, **dict(issue.message_args or {}))
        except Exception:
            pass
    return issue.fallback or issue.message_key or ""


class _SampleAutoRoiWorker(QtCore.QObject):
    progressChanged = QtCore.Signal(str)
    finished = QtCore.Signal(dict)

    def __init__(self, payload: dict, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._payload = dict(payload or {})

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._run_autogen()
        except Exception as exc:
            result = {
                "ok": 0,
                "errs": [str(exc)],
                "ok_paths": [],
                "todo_paths": [],
                "timings": {},
                "fatal": str(exc),
            }
        self.finished.emit(result)

    def _run_autogen(self) -> dict:
        return run_auto_roi_batch(
            paths=[str(path) for path in self._payload.get("paths", []) if str(path)],
            labels=[str(label).strip() for label in self._payload.get("labels", []) if str(label).strip()],
            method=str(self._payload.get("method", "") or "").strip(),
            ref_image=str(self._payload.get("ref_image", "") or ""),
            product_dir=str(self._payload.get("product_dir", "") or ""),
            camera_role=str(self._payload.get("camera_role", "cam1") or "cam1"),
            only_missing=bool(self._payload.get("only_missing", True)),
            pre_resolved=bool(self._payload.get("pre_resolved", False)),
            progress=self.progressChanged.emit,
        )

    @staticmethod
    def _missing_roi_files(paths: list[str], labels: list[str]) -> list[str]:
        return service_missing_roi_files(paths, labels)

class _SampleAnnotationAutoRoiDialog(QtWidgets.QDialog):
    def __init__(self, preview_dialog: "_SampleAnnotationPreviewDialog") -> None:
        super().__init__(preview_dialog)
        self._preview_dialog = preview_dialog
        self._tool_page = preview_dialog._tool_page
        self.setWindowTitle(tr("sample.auto_roi_tool"))
        self.setModal(False)
        self.resize(760, 180)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.lbl_scope = QtWidgets.QLabel("")
        self.lbl_scope.setStyleSheet("color:#d0d0d0;font-size:12px;")
        root.addWidget(self.lbl_scope)

        self.lbl_ref = QtWidgets.QLabel("")
        self.lbl_ref.setStyleSheet("color:#d0d0d0;font-size:12px;")
        root.addWidget(self.lbl_ref)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color:#86efac;font-size:12px;")
        root.addWidget(self.lbl_status)

        self.chk_only_missing = QtWidgets.QCheckBox(tr("debug.only_missing_roi"))
        self.chk_only_missing.setChecked(True)
        root.addWidget(self.chk_only_missing, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.btn_autogen_current = QtWidgets.QPushButton(tr("debug.batch_roi_current"))
        self.btn_autogen_current.clicked.connect(self._run_autogen_current_list)
        row.addWidget(self.btn_autogen_current)
        self.btn_autogen_current_image = QtWidgets.QPushButton(tr("debug.batch_roi_current_image_missing"))
        self.btn_autogen_current_image.clicked.connect(self._run_autogen_current_image)
        row.addWidget(self.btn_autogen_current_image)
        self.btn_clear_current = QtWidgets.QPushButton(tr("debug.clear_roi_current"))
        self.btn_clear_current.clicked.connect(self._run_clear_current_list)
        row.addWidget(self.btn_clear_current)
        root.addLayout(row)

        self._tool_page.roiGeometryChanged.connect(self._refresh_scope)
        self._tool_page.inspectionItemsChanged.connect(self._refresh_scope)
        self._autogen_thread: QtCore.QThread | None = None
        self._autogen_worker: _SampleAutoRoiWorker | None = None
        self._autogen_running = False
        self._refresh_scope()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_scope()

    def _camera_role(self) -> str:
        return str(self._preview_dialog.cmb_camera.currentData() or "cam1")

    def _sample_kind(self) -> str:
        return str(self._preview_dialog.cmb_sample_kind.currentData() or "train")

    def _scope_paths(self) -> List[str]:
        return self._tool_page._sample_paths_for_kind(self._sample_kind(), self._camera_role())

    def _current_path(self) -> str:
        path, _role = self._preview_dialog._current_path_and_role()
        return str(path or "").strip()

    def _refresh_scope(self) -> None:
        camera_role = self._camera_role()
        sample_kind = self._sample_kind()
        paths = self._scope_paths()
        current_path = self._current_path()
        sample_text = tr("debug.train_samples") if sample_kind == "train" else tr("debug.test_samples")
        self.lbl_scope.setText(
            tr("sample.current_scope", role=camera_role, sample=sample_text, count=len(paths))
            + (tr("sample.current_image_suffix", image=os.path.basename(current_path)) if current_path else "")
        )
        recipe = self._tool_page.shape_recipe_for_role(camera_role, force_reload=False)
        ref_image = ""
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = str(recipe.reference_image)
        self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(ref_image) if ref_image else '-'}")
        self.lbl_ref.setToolTip(ref_image)
        has_scope = bool(paths)
        if self._autogen_running:
            self.btn_autogen_current.setEnabled(False)
            self.btn_clear_current.setEnabled(False)
            self.btn_autogen_current_image.setEnabled(False)
            self.chk_only_missing.setEnabled(False)
        else:
            self.btn_autogen_current.setEnabled(has_scope)
            self.btn_clear_current.setEnabled(has_scope)
            self.btn_autogen_current_image.setEnabled(bool(current_path))
            self.chk_only_missing.setEnabled(True)

    def _sync_tool_page_role(self) -> None:
        self._tool_page._set_current_camera_role(self._camera_role(), sync_debug_role=True)

    def _run_autogen_current_list(self) -> None:
        paths = self._scope_paths()
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("sample.no_processable_images"))
            return
        self._sync_tool_page_role()
        self._start_autogen_job(paths, only_missing=self.chk_only_missing.isChecked())

    def _run_autogen_current_image(self) -> None:
        path = self._current_path()
        if not path:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("sample.no_selected_image"))
            return
        self._sync_tool_page_role()
        self._start_autogen_job([path], only_missing=True)

    def _set_autogen_running(self, running: bool, message: str = "") -> None:
        self._autogen_running = bool(running)
        self.lbl_status.setText(message)
        self._refresh_scope()

    def _prepare_autogen_payload(
        self,
        paths: List[str],
        *,
        only_missing: bool,
    ) -> dict | None:
        tool_page = self._tool_page
        ref_image = tool_page.ref_image
        method = tool_page.loc_method
        role = self._camera_role()
        labels: List[str]
        pre_resolved = False

        if method == "shape":
            try:
                recipe = tool_page.shape_recipe_for_role(role, force_reload=True)
                if role == tool_page.current_camera_role():
                    tool_page.shape_recipe = recipe
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Info", f"Failed to load template recipe: {exc}")
                return None
            if recipe.reference_image and os.path.exists(recipe.reference_image):
                ref_image = recipe.reference_image
                if tool_page.ref_image != ref_image:
                    if role == tool_page.current_camera_role():
                        tool_page._set_reference(ref_image)
                    else:
                        tool_page.ref_image = ref_image
                        if getattr(tool_page, "lbl_ref", None) is not None:
                            tool_page.lbl_ref.setText(
                                f"{tr('debug.reference_image')}: {os.path.basename(ref_image)}"
                            )
                            tool_page.lbl_ref.setToolTip(ref_image)
            labels = tool_page._shape_output_labels(role)
            validation = validate_autogen_reference(
                method=method,
                ref_image=ref_image,
                shape_model_path=tool_page.shape_model_path_for_role(role),
                shape_labels=labels,
                reference_regions=recipe.reference_regions,
            )
        else:
            validation = validate_autogen_reference(method=method, ref_image=ref_image)
            labels = validation.labels

        if not validation.ok:
            issue = validation.issue
            QtWidgets.QMessageBox.warning(
                self,
                tr("common.info"),
                _auto_roi_issue_text(issue) if issue is not None else tr("common.error"),
            )
            return None
        labels = validation.labels

        resolved_paths = list(paths)
        if not only_missing:
            resolved_paths = tool_page._resolve_autogen_targets(
                paths,
                only_missing=False,
                silent=False,
                camera_role=role,
            )
            pre_resolved = True
            if not resolved_paths:
                return None

        return {
            "paths": resolved_paths,
            "labels": labels,
            "method": method,
            "ref_image": ref_image,
            "product_dir": tool_page.session.product_dir,
            "camera_role": role,
            "only_missing": bool(only_missing),
            "pre_resolved": pre_resolved,
        }

    def _start_autogen_job(self, paths: List[str], *, only_missing: bool) -> None:
        if self._autogen_running:
            return
        if getattr(self._tool_page, "_sample_auto_roi_jobs", []):
            QtWidgets.QMessageBox.information(self, tr("common.info"), "自动 ROI 正在处理，请等待")
            return
        payload = self._prepare_autogen_payload(paths, only_missing=only_missing)
        if payload is None:
            return

        thread = QtCore.QThread(self._tool_page)
        worker = _SampleAutoRoiWorker(payload)
        worker.moveToThread(thread)
        self._autogen_thread = thread
        self._autogen_worker = worker

        jobs = getattr(self._tool_page, "_sample_auto_roi_jobs", None)
        if jobs is None:
            jobs = []
            setattr(self._tool_page, "_sample_auto_roi_jobs", jobs)
        jobs.append((thread, worker))

        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_autogen_progress)
        worker.finished.connect(self._on_autogen_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._forget_autogen_job(thread, worker))
        self._set_autogen_running(True, "处理中...")
        thread.start()

    def _forget_autogen_job(self, thread: QtCore.QThread, worker: _SampleAutoRoiWorker) -> None:
        jobs = getattr(self._tool_page, "_sample_auto_roi_jobs", [])
        try:
            jobs.remove((thread, worker))
        except ValueError:
            pass
        if self._autogen_thread is thread:
            self._autogen_thread = None
            self._autogen_worker = None

    def _on_autogen_progress(self, message: str) -> None:
        if self._autogen_running:
            self.lbl_status.setText(f"处理中... {message}")

    def _on_autogen_finished(self, result: dict) -> None:
        tool_page = self._tool_page
        result = dict(result or {})
        timings = dict(result.get("timings", {}) or {})
        for path, elapsed_ms in timings.items():
            try:
                value = float(elapsed_ms)
            except Exception:
                continue
            tool_page._shape_match_ms_by_image[str(path)] = value
            tool_page._shape_autogen_ms_by_image[str(path)] = value

        ok = int(result.get("ok", 0) or 0)
        errs = [str(err) for err in result.get("errs", []) or [] if str(err)]
        changed_paths = {str(path) for path in result.get("ok_paths", []) or [] if str(path)}
        todo_paths = {str(path) for path in result.get("todo_paths", []) or [] if str(path)}

        if ok:
            previous = bool(getattr(tool_page, "_suppress_sample_preview_reload", False))
            tool_page._suppress_sample_preview_reload = True
            try:
                tool_page._reload_inspection_items()
                tool_page.roiGeometryChanged.emit()
            finally:
                tool_page._suppress_sample_preview_reload = previous

            cur = tool_page.canvas.image_path()
            if cur and cur in todo_paths:
                tool_page._load_canvas_image(cur)
                tool_page._set_status_for_current_image(cur)
            self._preview_dialog._refresh_after_auto_roi_change(changed_paths, self._camera_role())

        if result.get("fatal"):
            message = str(result.get("fatal") or "")
            if self.isVisible():
                QtWidgets.QMessageBox.warning(self, tr("common.error"), message)
            self.lbl_status.setText(message)
        elif result.get("no_work"):
            if self.isVisible():
                QtWidgets.QMessageBox.information(self, tr("common.info"), tr("auto.images_already_have_roi"))
            self.lbl_status.setText(tr("auto.images_already_have_roi"))
        else:
            msg = tr("auto.finished", ok=ok, failed=len(errs))
            if errs:
                msg += "\n\n" + tr("auto.failed_examples") + "\n" + "\n".join(errs[:10])
            if self.isVisible():
                QtWidgets.QMessageBox.information(self, tr("common.done"), msg)
            tool_page.lbl_status.setText(tr("auto.status_generated", ok=ok, failed=len(errs)))
            self.lbl_status.setText(tr("auto.status_generated", ok=ok, failed=len(errs)))

        self._set_autogen_running(False, self.lbl_status.text())

    def _run_clear_current_list(self) -> None:
        paths = self._scope_paths()
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("sample.no_processable_images"))
            return
        sample_text = tr("debug.train_samples") if self._sample_kind() == "train" else tr("debug.test_samples")
        reply = QtWidgets.QMessageBox.question(
            self,
            tr("auto.clear_title"),
            tr("sample.clear_roi_confirm", role=self._camera_role(), sample=sample_text),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._sync_tool_page_role()
        self._tool_page._clear_roi_for_images(paths, silent=False, camera_role=self._camera_role())

