from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr


class SampleListController:
    def __init__(self, owner) -> None:
        self.owner = owner

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
        self.select_path_in_current_tab(path)

    def add_images_to(self, kind: str) -> None:
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

    def remove_selected_from(self, kind: str) -> None:
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

    def clear_current_test_list(self) -> None:
        current_role = self.owner._selected_image_list_camera_role()
        visible = self.owner._sample_paths_for_kind("test", current_role)
        if not visible:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.clear_current_test_list_empty"))
            return
        role_text = tr("runtime.camera1") if current_role == "cam1" else tr("runtime.camera2")
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
