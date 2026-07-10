from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr


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

    def move_selected_sample_to(self, target_kind: str) -> None:
        if not self._require_manage_permission("移动样本"):
            return
        path = self.current_selected_path()
        if not path:
            return
        normalized_target = str(target_kind or "").strip().upper()
        for collection in (self.owner.train_files, self.owner.test_files, self.owner.ok_files, self.owner.ng_files):
            while path in collection:
                collection.remove(path)
        if normalized_target == "TRAIN":
            self.owner.train_files.append(path)
            self.owner.train_files = sorted(list(dict.fromkeys(self.owner.train_files)))
            self.owner.tabs.setCurrentIndex(0)
        else:
            self.owner.test_files.append(path)
            self.owner.test_files = sorted(list(dict.fromkeys(self.owner.test_files)))
            self.owner.tabs.setCurrentIndex(1)
        self.owner._refresh_lists()
        self.owner._clear_training_roi_review_state()
        self.owner._save_session()
        self._audit("移动样本", target=str(path), after_value=normalized_target)
        self.select_path_in_current_tab(path)

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
        normalized_kind = str(kind or "").strip().upper()
        if normalized_kind in {"TRAIN", "OK", "NG", "TRAIN_OK", "TRAIN_NG"}:
            self.owner.train_files.extend(files)
            self.owner.train_files = sorted(list(dict.fromkeys(self.owner.train_files)))
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
