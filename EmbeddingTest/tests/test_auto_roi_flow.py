from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from PySide6 import QtCore, QtTest, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ui.debug.tool_page import auto_roi_flow


class _DummyCanvas:
    def __init__(self, image_path: str) -> None:
        self._image_path = image_path

    def image_path(self) -> str:
        return self._image_path


class _DummyToolPage(QtCore.QObject):
    def __init__(self, image_path: str) -> None:
        super().__init__()
        self.canvas = _DummyCanvas(image_path)
        self.lbl_status = QtWidgets.QLabel()
        self._template_editor_dialog = object()
        self._cleared_roles = []
        self._sync_count = 0
        self._loaded_paths = []

    def current_camera_role(self) -> str:
        return "cam1"

    def _clear_training_roi_review_state(self, role: str) -> None:
        self._cleared_roles.append(role)

    def _sync_line2dup_recipe_and_items(self) -> None:
        self._sync_count += 1

    def _load_canvas_image(self, path: str) -> None:
        self._loaded_paths.append(path)

    def _schedule_line2dup_reference_regions_sync(self) -> None:
        auto_roi_flow._schedule_line2dup_reference_regions_sync(self)

    def _flush_line2dup_reference_regions_sync(self) -> None:
        auto_roi_flow._flush_line2dup_reference_regions_sync(self)


class _ResolveTargetsHarness:
    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        self._skip_empty_autogen_message = False

    def _missing_roi_files(self, paths, camera_role=None):
        return list(self.missing)


class AutoRoiFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_reference_region_sync_is_debounced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "scene.png"
            image_path.write_bytes(b"stub")
            page = _DummyToolPage(str(image_path))

            auto_roi_flow._on_line2dup_reference_regions_changed(page)
            auto_roi_flow._on_line2dup_reference_regions_changed(page)

            self.assertEqual(page._sync_count, 0)
            timer = getattr(page, "_line2dup_reference_regions_sync_timer", None)
            self.assertIsNotNone(timer)
            self.assertTrue(timer.isActive())

            QtTest.QTest.qWait(220)
            self.app.processEvents()

            self.assertEqual(page._sync_count, 1)
            self.assertEqual(page._cleared_roles, ["cam1"])
            self.assertEqual(page._loaded_paths, [str(image_path)])
            self.assertEqual(page.lbl_status.text(), "状态：参考ROI已同步到运行界面")

    def test_destroy_flushes_pending_reference_region_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "scene.png"
            image_path.write_bytes(b"stub")
            page = _DummyToolPage(str(image_path))

            auto_roi_flow._on_line2dup_reference_regions_changed(page)
            auto_roi_flow._on_template_editor_dialog_destroyed(page)

            self.assertEqual(page._sync_count, 1)
            self.assertEqual(page._template_editor_dialog, None)
            timer = getattr(page, "_line2dup_reference_regions_sync_timer", None)
            self.assertIsNotNone(timer)
            self.assertFalse(timer.isActive())

    def test_resolve_autogen_targets_allows_silent_overwrite_when_roi_exists(self) -> None:
        harness = _ResolveTargetsHarness(missing=[])
        paths = ["a.png", "b.png"]

        targets = auto_roi_flow._resolve_autogen_targets(
            harness,
            paths,
            only_missing=False,
            silent=True,
            camera_role="cam1",
        )

        self.assertEqual(targets, paths)

    def test_resolve_autogen_targets_skips_existing_roi_in_only_missing_mode(self) -> None:
        harness = _ResolveTargetsHarness(missing=[])

        targets = auto_roi_flow._resolve_autogen_targets(
            harness,
            ["a.png"],
            only_missing=True,
            silent=True,
            camera_role="cam1",
        )

        self.assertEqual(targets, [])


if __name__ == "__main__":
    unittest.main()
