from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ui.debug.tool_page import auto_roi_flow
from ui.debug.tool_page.page import (
    _BatchAutoRoiWorker,
    _SampleAnnotationAutoRoiDialog,
    _SampleAnnotationPreviewDialog,
)


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


class _TargetedRefreshHarness:
    _refresh_after_roi_batch = _SampleAnnotationPreviewDialog._refresh_after_roi_batch

    def __init__(self) -> None:
        self.sample_list = QtWidgets.QListWidget()
        self.cmb_camera = QtWidgets.QComboBox()
        self.cmb_camera.addItem("cam1", "cam1")
        self.cmb_sample_kind = QtWidgets.QComboBox()
        self.cmb_sample_kind.addItem("训练样本", "train")
        self.selected_count = 0
        self._tool_page = SimpleNamespace(
            _sample_item_display_text=lambda path, kind, role: f"updated:{Path(path).name}"
        )

    def _current_dialog_selected_path(self) -> str:
        item = self.sample_list.currentItem()
        return str(item.data(QtCore.Qt.UserRole) or "") if item is not None else ""

    def _on_sample_selected(self) -> None:
        self.selected_count += 1


class _BatchFinishHarness:
    _finish_autogen_progress = _SampleAnnotationAutoRoiDialog._finish_autogen_progress
    _on_autogen_worker_finished = _SampleAnnotationAutoRoiDialog._on_autogen_worker_finished

    def __init__(self, events: list[str]) -> None:
        self._autogen_thread = object()
        self._autogen_worker = object()
        self._autogen_preferred_path = "a.png"
        self._autogen_progress_total = 2
        self._autogen_progress_ok = 0
        self._autogen_progress_errors = 0
        self.progress_autogen = QtWidgets.QProgressBar()
        self.lbl_status = QtWidgets.QLabel()
        self._preview_dialog = SimpleNamespace(
            _refresh_after_roi_batch=lambda paths, preferred_path="": events.append("refresh")
        )
        self._tool_page = SimpleNamespace(
            _line2dup_match_ms_by_image={},
            _line2dup_autogen_ms_by_image={},
            _invalidate_shape_lookup_cache=lambda path: events.append(f"invalidate:{path}"),
            canvas=SimpleNamespace(image_path=lambda: ""),
            _load_canvas_image=lambda path: events.append(f"canvas:{path}"),
            _set_status_for_current_image=lambda path: None,
        )

    def _set_autogen_running(self, running: bool) -> None:
        pass


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

    def test_batch_worker_reuses_one_line2dup_detector(self) -> None:
        paths = ["a.png", "b.png", "c.png"]
        detector = object()
        worker = _BatchAutoRoiWorker(
            {
                "method": "line2dup",
                "camera_role": "cam1",
                "product_dir": "product",
                "ref_image": "ref.png",
                "recipe": SimpleNamespace(),
                "model_path": "model.json",
                "todo": paths,
            }
        )
        with (
            mock.patch(
                "line2dup.like_matcher.load_detector_model",
                return_value=detector,
            ) as load_detector,
            mock.patch(
                "ui.debug.tool_page.page.line2dup_locator.autogen_roi_json_from_line2dup_timed",
                side_effect=[
                    SimpleNamespace(locate_ms=1.0, total_ms=2.0),
                    SimpleNamespace(locate_ms=3.0, total_ms=4.0),
                    SimpleNamespace(locate_ms=5.0, total_ms=6.0),
                ],
            ) as autogen,
        ):
            result = worker._run_batch()

        self.assertEqual(load_detector.call_count, 1)
        self.assertEqual(autogen.call_count, 3)
        self.assertEqual(result["ok_paths"], paths)
        self.assertEqual(result["errors"], [])
        self.assertTrue(
            all(call.kwargs["detector"] is detector for call in autogen.call_args_list)
        )

    def test_batch_refresh_updates_changed_rows_without_reloading_list(self) -> None:
        harness = _TargetedRefreshHarness()
        paths = ["a.png", "b.png", "c.png"]
        for path in paths:
            item = QtWidgets.QListWidgetItem(f"old:{path}")
            item.setData(QtCore.Qt.UserRole, path)
            item.setToolTip(path)
            harness.sample_list.addItem(item)
        harness.sample_list.setCurrentRow(1)

        harness._refresh_after_roi_batch(["b.png", "c.png"], preferred_path="b.png")

        self.assertEqual(harness.sample_list.count(), 3)
        self.assertEqual(harness.sample_list.item(0).text(), "old:a.png")
        self.assertEqual(harness.sample_list.item(1).text(), "updated:b.png")
        self.assertEqual(harness.sample_list.item(2).text(), "updated:c.png")
        self.assertEqual(harness.selected_count, 1)

    def test_batch_completion_dialog_is_after_targeted_refresh_and_100_percent(self) -> None:
        events: list[str] = []
        harness = _BatchFinishHarness(events)
        with mock.patch(
            "PySide6.QtWidgets.QMessageBox.information",
            side_effect=lambda *args, **kwargs: events.append("dialog"),
        ):
            harness._on_autogen_worker_finished(
                {
                    "ok_paths": ["a.png", "b.png"],
                    "errors": [],
                    "timings": {},
                    "fatal": "",
                }
            )

        self.assertLess(events.index("refresh"), events.index("dialog"))
        self.assertEqual(harness.progress_autogen.value(), harness.progress_autogen.maximum())
        self.assertIn("100%", harness.progress_autogen.format())


if __name__ == "__main__":
    unittest.main()
