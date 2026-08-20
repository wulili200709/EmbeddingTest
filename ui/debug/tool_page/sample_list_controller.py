from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr
from ui.debug.tool_page.camera_roles import camera_role_from_path
from ui.debug.tool_page.training_assets import ensure_training_image_local


def _split_files_by_camera_role(files: list[str]) -> tuple[list[str], list[str]]:
    valid_files: list[str] = []
    invalid_files: list[str] = []
    for path in files:
        target = valid_files if camera_role_from_path(path) else invalid_files
        target.append(path)
    return valid_files, invalid_files


def _invalid_camera_name_details(files: list[str], *, limit: int = 10) -> str:
    visible_names = [QtCore.QFileInfo(path).fileName() for path in files[:limit]]
    details = "\n".join(f"- {name}" for name in visible_names)
    omitted = max(0, len(files) - len(visible_names))
    if omitted:
        details += f"\n{tr('debug.invalid_camera_name_more', count=omitted)}"
    return details


class SampleListController:
    def __init__(self, owner) -> None:
        self.owner = owner

    def _require_manage_permission(self, action_name: str) -> bool:
        require_permission = getattr(self.owner.window(), "_require_permission", None)
        if callable(require_permission):
            return bool(require_permission("sample.manage", action_name))
        return True

    def _audit(self, action: str, *, target: str = "", before_value: str = "", after_value: str = "") -> None:
        audit_event = getattr(self.owner.window(), "_audit_event", None)
        if callable(audit_event):
            audit_event(
                module="样本",
                action=action,
                target=target,
                before_value=before_value,
                after_value=after_value,
            )

    def current_selected_path(self) -> Optional[str]:
        if self.owner.tabs.currentIndex() == 0:
            items = self.owner.ok_list.selectedItems()
            if not items:
                return None
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if path:
                return str(path)
            visible = self.owner._sample_paths_for_kind("train", self.owner._selected_image_list_camera_role())
            row = self.owner.ok_list.row(items[0])
            return visible[row] if row < len(visible) else None

        items = self.owner.test_list.selectedItems()
        if not items:
            return None
        path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
        if path:
            return str(path)
        visible = self.owner._filter_paths_for_camera(self.owner.test_files, self.owner._selected_image_list_camera_role())
        row = self.owner.test_list.row(items[0])
        return visible[row] if row < len(visible) else None

    def select_path_in_current_tab(self, path: str) -> None:
        if not path:
            return
        list_widget = self.owner.ok_list if self.owner._current_sample_tab_kind() == "train" else self.owner.test_list
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item is None:
                continue
            item_path = item.data(QtCore.Qt.UserRole) or item.toolTip()
            if str(item_path or "") == str(path):
                blocker = QtCore.QSignalBlocker(list_widget)
                list_widget.setCurrentRow(row)
                del blocker
                self.show_selected_image_path(path)
                return

    def show_selected_image_path(self, path: Optional[str]) -> None:
        if not path:
            self.owner._update_sample_panel_widgets()
            return
        self.owner._clear_selected_inspection_item()
        if self.owner.canvas.image_path() != path:
            self.owner._load_canvas_image(path)
        self.owner._set_status_for_current_image(path)
        self.owner._update_sample_panel_widgets()

    def _localize_training_files(
        self,
        files: list[str],
    ) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, Exception]]]:
        product_dir = str(getattr(getattr(self.owner, "session", None), "product_dir", "") or "")
        localized: list[str] = []
        path_changes: list[tuple[str, str]] = []
        failures: list[tuple[str, Exception]] = []
        for path in files:
            try:
                local_path = ensure_training_image_local(path, product_dir=product_dir)
            except Exception as exc:
                failures.append((str(path), exc))
                continue
            localized.append(local_path)
            if os.path.normcase(os.path.normpath(local_path)) != os.path.normcase(os.path.normpath(path)):
                path_changes.append((str(path), local_path))
        return localized, path_changes, failures

    def _show_training_copy_failures(self, failures: list[tuple[str, Exception]]) -> None:
        if not failures:
            return
        details = "\n".join(
            f"- {QtCore.QFileInfo(path).fileName()}: {error}"
            for path, error in failures[:10]
        )
        if len(failures) > 10:
            details += f"\n... ({len(failures) - 10} more)"
        QtWidgets.QMessageBox.warning(
            self.owner,
            tr("common.error"),
            tr("debug.training_image_copy_failed", files=details),
        )

    def _transfer_annotation_path(self, old_path: str, new_path: str, *, remove_source: bool) -> bool:
        annotations_by_path = getattr(self.owner, "_sample_roi_annotations_by_path", None)
        if not isinstance(annotations_by_path, dict):
            return False
        old_key_normalized = os.path.normcase(os.path.normpath(str(old_path or "")))
        source_key = next(
            (
                key
                for key in list(annotations_by_path)
                if os.path.normcase(os.path.normpath(str(key or ""))) == old_key_normalized
            ),
            None,
        )
        if source_key is None:
            return False
        moved_annotations = dict(annotations_by_path.get(source_key, {}) or {})
        target_key = os.path.normpath(str(new_path or ""))
        merged = dict(annotations_by_path.get(target_key, {}) or {})
        merged.update(moved_annotations)
        if merged:
            annotations_by_path[target_key] = merged
        if remove_source and source_key != target_key:
            annotations_by_path.pop(source_key, None)
        return True

    def _save_annotations_if_changed(self, changed: bool) -> None:
        if not changed:
            return
        save_annotations = getattr(self.owner, "_save_sample_roi_annotations", None)
        if callable(save_annotations):
            save_annotations()

    def move_selected_sample_to(self, target_kind: str) -> None:
        if not self._require_manage_permission("移动样本"):
            return
        path = self.current_selected_path()
        if not path:
            return
        normalized_target = str(target_kind or "").strip().upper()
        target_path = path
        annotations_changed = False
        if normalized_target == "TRAIN":
            localized, path_changes, failures = self._localize_training_files([path])
            if failures or not localized:
                self._show_training_copy_failures(failures)
                return
            target_path = localized[0]
            if path_changes:
                annotations_changed = self._transfer_annotation_path(
                    path,
                    target_path,
                    remove_source=True,
                )
        for collection in (self.owner.train_files, self.owner.test_files, self.owner.ok_files, self.owner.ng_files):
            while path in collection:
                collection.remove(path)
        if normalized_target == "TRAIN":
            self.owner.train_files.append(target_path)
            self.owner.train_files = sorted(list(dict.fromkeys(self.owner.train_files)))
            self.owner.tabs.setCurrentIndex(0)
        else:
            self.owner.test_files.append(path)
            self.owner.test_files = sorted(list(dict.fromkeys(self.owner.test_files)))
            self.owner.tabs.setCurrentIndex(1)
        self.owner._refresh_lists()
        self.owner._clear_training_roi_review_state()
        self.owner._save_session()
        self._save_annotations_if_changed(annotations_changed)
        self._audit(
            "移动样本",
            target=str(target_path),
            before_value=str(path) if target_path != path else "",
            after_value=normalized_target,
        )
        self.select_path_in_current_tab(target_path)

    def add_images_to(self, kind: str) -> None:
        if not self._require_manage_permission("添加样本"):
            return
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.owner,
            tr("debug.add_images_title", kind=kind),
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return
        files, invalid_files = _split_files_by_camera_role(files)
        if invalid_files:
            QtWidgets.QMessageBox.warning(
                self.owner,
                tr("debug.invalid_camera_name_title"),
                tr(
                    "debug.invalid_camera_name_message",
                    count=len(invalid_files),
                    files=_invalid_camera_name_details(invalid_files),
                ),
            )
        if not files:
            return
        normalized_kind = str(kind or "").strip().upper()
        if normalized_kind in {"TRAIN", "OK", "NG", "TRAIN_OK", "TRAIN_NG"}:
            files, path_changes, failures = self._localize_training_files(files)
            self._show_training_copy_failures(failures)
            if not files:
                return
            annotations_changed = False
            for old_path, new_path in path_changes:
                annotations_changed = (
                    self._transfer_annotation_path(old_path, new_path, remove_source=False)
                    or annotations_changed
                )
            self.owner.train_files.extend(files)
            self.owner.train_files = sorted(list(dict.fromkeys(self.owner.train_files)))
            self._save_annotations_if_changed(annotations_changed)
        else:
            self.owner.test_files.extend(files)
            self.owner.test_files = sorted(list(dict.fromkeys(self.owner.test_files)))
        self.owner._refresh_lists()
        self.owner._clear_training_roi_review_state()
        self.owner._save_session()
        self._audit("添加样本", target=normalized_kind, after_value=f"count={len(files)}")

    def remove_selected_from(self, kind: str) -> None:
        if not self._require_manage_permission("删除样本"):
            return
        normalized_kind = str(kind or "").strip().upper()
        if normalized_kind == "TRAIN":
            path = self.current_selected_path()
            if not path:
                return
            if path in self.owner.train_files:
                self.owner.train_files.remove(str(path))
        else:
            items = self.owner.test_list.selectedItems()
            if not items:
                return
            path = items[0].data(QtCore.Qt.UserRole) or items[0].toolTip()
            if not path:
                visible = self.owner._filter_paths_for_camera(self.owner.test_files, self.owner._selected_image_list_camera_role())
                idx = self.owner.test_list.row(items[0])
                path = visible[idx] if idx < len(visible) else None
            if not path or path not in self.owner.test_files:
                return
            self.owner.test_files.remove(str(path))
        self.owner._refresh_lists()
        self.owner._clear_training_roi_review_state()
        self.owner._save_session()
        self._audit("删除样本", target=normalized_kind, before_value=str(path or ""))

    def clear_current_test_list(self) -> None:
        if not self._require_manage_permission("清空测试样本"):
            return
        current_role = self.owner._selected_image_list_camera_role()
        visible = self.owner._sample_paths_for_kind("test", current_role)
        if not visible:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.clear_current_test_list_empty"))
            return
        if current_role == "cam1":
            role_text = tr("runtime.camera1")
        elif current_role == "cam2":
            role_text = tr("runtime.camera2")
        elif current_role == "cam3":
            role_text = tr("runtime.camera3")
        else:
            role_text = current_role
        confirm = QtWidgets.QMessageBox.question(
            self.owner,
            tr("debug.clear_current_test_list"),
            tr("debug.clear_current_test_list_confirm", role=role_text),
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        remove_set = {str(path) for path in visible if str(path)}
        self.owner.test_files = [path for path in self.owner.test_files if str(path) not in remove_set]
        self.owner._refresh_lists()
        self.owner._clear_training_roi_review_state()
        self.owner._save_session()
        self._audit("清空测试样本", target=str(current_role), before_value=f"count={len(remove_set)}")
