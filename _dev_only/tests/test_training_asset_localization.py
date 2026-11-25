from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from application.product_session import ProductSession, SessionData
from ui.debug.tool_page.sample_list_controller import SampleListController
from ui.debug.tool_page.training_assets import ensure_training_image_local, is_path_in_product


class _SampleOwner(QtWidgets.QWidget):
    def __init__(self, product_dir: str) -> None:
        super().__init__()
        self.session = SimpleNamespace(product_dir=product_dir)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(QtWidgets.QWidget(), "Training")
        self.tabs.addTab(QtWidgets.QWidget(), "Test")
        self.ok_list = QtWidgets.QListWidget()
        self.test_list = QtWidgets.QListWidget()
        self.train_files: list[str] = []
        self.test_files: list[str] = []
        self.ok_files: list[str] = []
        self.ng_files: list[str] = []
        self._sample_roi_annotations_by_path: dict[str, dict[str, str]] = {}
        self.refresh_count = 0
        self.session_save_count = 0
        self.annotation_save_count = 0

    def _current_sample_tab_kind(self) -> str:
        return "train" if self.tabs.currentIndex() == 0 else "test"

    def _sample_paths_for_kind(self, kind: str, _role=None) -> list[str]:
        return list(self.train_files if kind == "train" else self.test_files)

    def _filter_paths_for_camera(self, paths, _role=None):
        return list(paths)

    def _selected_image_list_camera_role(self) -> str:
        return "cam1"

    def _refresh_lists(self) -> None:
        self.refresh_count += 1

    def _clear_training_roi_review_state(self) -> None:
        return None

    def _save_session(self) -> None:
        self.session_save_count += 1

    def _save_sample_roi_annotations(self) -> None:
        self.annotation_save_count += 1

    def _clear_selected_inspection_item(self) -> None:
        return None

    def _load_canvas_image(self, _path: str) -> None:
        return None

    def _set_status_for_current_image(self, _path: str) -> None:
        return None

    def _update_sample_panel_widgets(self) -> None:
        return None


class TrainingAssetLocalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_external_training_image_and_labelme_sidecar_are_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            product_dir = root / "products" / "P001"
            external = root / "desktop" / "part_cam1.bmp"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"image")
            external.with_suffix(".json").write_text('{"shapes": []}', encoding="utf-8")

            localized = Path(ensure_training_image_local(external, product_dir=product_dir))

            self.assertTrue(localized.is_file())
            self.assertEqual(localized.read_bytes(), b"image")
            self.assertEqual(localized.parent, product_dir / "samples" / "train")
            self.assertTrue(localized.with_suffix(".json").is_file())
            self.assertTrue(is_path_in_product(localized, product_dir=product_dir))

    def test_same_content_reuses_file_and_different_content_gets_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            product_dir = root / "product"
            first = root / "a" / "part_cam1.png"
            same = root / "b" / "part_cam1.png"
            different = root / "c" / "part_cam1.png"
            for path, content in ((first, b"same"), (same, b"same"), (different, b"different")):
                path.parent.mkdir(parents=True)
                path.write_bytes(content)

            first_local = ensure_training_image_local(first, product_dir=product_dir)
            same_local = ensure_training_image_local(same, product_dir=product_dir)
            different_local = ensure_training_image_local(different, product_dir=product_dir)

            self.assertEqual(first_local, same_local)
            self.assertNotEqual(first_local, different_local)
            self.assertTrue(Path(different_local).is_file())

    def test_adding_external_test_image_does_not_copy_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            product_dir = root / "product"
            external = root / "desktop" / "part_cam1.png"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"test")
            owner = _SampleOwner(str(product_dir))
            controller = SampleListController(owner)

            with mock.patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileNames",
                return_value=([str(external)], ""),
            ):
                controller.add_images_to("TEST")

            self.assertEqual(owner.test_files, [str(external)])
            self.assertFalse((product_dir / "samples" / "train").exists())

    def test_adding_external_training_image_records_product_local_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            product_dir = root / "product"
            external = root / "desktop" / "part_cam1.png"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"train")
            owner = _SampleOwner(str(product_dir))
            controller = SampleListController(owner)

            with mock.patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileNames",
                return_value=([str(external)], ""),
            ):
                controller.add_images_to("TRAIN")

            self.assertEqual(len(owner.train_files), 1)
            localized = owner.train_files[0]
            self.assertNotEqual(localized, str(external))
            self.assertTrue(is_path_in_product(localized, product_dir=product_dir))
            self.assertEqual(Path(localized).read_bytes(), b"train")
            self.assertEqual(owner.test_files, [])
            self.assertEqual(owner.session_save_count, 1)

    def test_adding_training_image_already_in_product_does_not_duplicate_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            product_dir = Path(temporary_dir) / "product"
            internal = product_dir / "manual" / "part_cam1.png"
            internal.parent.mkdir(parents=True)
            internal.write_bytes(b"train")
            owner = _SampleOwner(str(product_dir))
            controller = SampleListController(owner)

            with mock.patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileNames",
                return_value=([str(internal)], ""),
            ):
                controller.add_images_to("TRAIN")

            self.assertEqual(owner.train_files, [str(internal)])
            self.assertFalse((product_dir / "samples" / "train").exists())

    def test_moving_external_test_image_to_training_copies_and_rekeys_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            product_dir = root / "product"
            external = root / "desktop" / "part_cam1.png"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"test")
            owner = _SampleOwner(str(product_dir))
            owner.test_files = [str(external)]
            owner._sample_roi_annotations_by_path[os.path.normpath(str(external))] = {
                "cam1::roi1": "OK"
            }
            item = QtWidgets.QListWidgetItem(external.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(external))
            owner.test_list.addItem(item)
            owner.test_list.setCurrentRow(0)
            owner.tabs.setCurrentIndex(1)
            controller = SampleListController(owner)

            controller.move_selected_sample_to("TRAIN")

            self.assertEqual(owner.test_files, [])
            self.assertEqual(len(owner.train_files), 1)
            localized = owner.train_files[0]
            self.assertTrue(is_path_in_product(localized, product_dir=product_dir))
            self.assertTrue(Path(localized).is_file())
            self.assertNotIn(os.path.normpath(str(external)), owner._sample_roi_annotations_by_path)
            self.assertEqual(
                owner._sample_roi_annotations_by_path[os.path.normpath(localized)],
                {"cam1::roi1": "OK"},
            )
            self.assertEqual(owner.session_save_count, 1)
            self.assertEqual(owner.annotation_save_count, 1)

    def test_copy_failure_keeps_external_image_in_test_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            product_dir = root / "product"
            external = root / "desktop" / "part_cam1.png"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"test")
            owner = _SampleOwner(str(product_dir))
            owner.test_files = [str(external)]
            item = QtWidgets.QListWidgetItem(external.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(external))
            owner.test_list.addItem(item)
            owner.test_list.setCurrentRow(0)
            owner.tabs.setCurrentIndex(1)
            controller = SampleListController(owner)

            with mock.patch(
                "ui.debug.tool_page.sample_list_controller.ensure_training_image_local",
                side_effect=OSError("disk full"),
            ), mock.patch.object(QtWidgets.QMessageBox, "warning"):
                controller.move_selected_sample_to("TRAIN")

            self.assertEqual(owner.test_files, [str(external)])
            self.assertEqual(owner.train_files, [])
            self.assertEqual(owner.session_save_count, 0)

    def test_copied_product_keeps_training_image_and_drops_external_test_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            session = ProductSession(str(root / "products"))
            session.load()
            session.switch_product("Default")
            external_training = root / "desktop" / "part_cam1.png"
            external_test = root / "desktop" / "check_cam1.png"
            external_training.parent.mkdir(parents=True)
            external_training.write_bytes(b"train")
            external_test.write_bytes(b"test")
            localized = ensure_training_image_local(
                external_training,
                product_dir=session.product_dir,
            )
            session.save_session(
                SessionData(
                    train_files=[localized],
                    test_files=[str(external_test)],
                )
            )

            error = session.copy_product("Default", "Copied")

            self.assertEqual(error, "")
            session.switch_product("Copied")
            copied_data = session.load_session()
            self.assertEqual(len(copied_data.train_files), 1)
            self.assertTrue(Path(copied_data.train_files[0]).is_file())
            self.assertTrue(
                is_path_in_product(copied_data.train_files[0], product_dir=session.product_dir)
            )
            self.assertEqual(copied_data.test_files, [])


if __name__ == "__main__":
    unittest.main()
