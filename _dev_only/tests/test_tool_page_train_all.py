from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PySide6 import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from application.algorithm_controller import TrainResult
from domain.inspection_items import InspectionItem
from ui.debug.tool_page.page import ToolPage


class _FakeAlgo:
    def __init__(self) -> None:
        self.product_params = type(
            "Params",
            (),
            {
                "algorithm": "",
                "score_mode": "proto",
                "margin": 0.02,
                "topk": 3,
            },
        )()
        self.model = None
        self.train_calls: list[dict] = []

    def is_learning_tool(self, code) -> bool:
        return str(code or "").strip() == "shared_backbone_register"

    def current_learning_backbone(self, camera_role=None) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, code, camera_role=None) -> str:
        return str(code or "").strip()

    def is_embedding_algorithm(self, algorithm) -> bool:
        return str(algorithm or "").strip() == "efficientnet_b0"

    def embedding_cache_dir(self, algorithm, product_dir) -> str:
        return os.path.join(str(product_dir), "embedding_cache", str(algorithm))

    def train(
        self,
        ok_files,
        ng_files,
        *,
        algorithm,
        product_dir,
        label_names,
        model_key,
        progress_callback=None,
        embedding_cache_dir=None,
    ):
        self.train_calls.append(
            {
                "algorithm": algorithm,
                "label_names": list(label_names),
                "model_key": str(model_key),
                "product_dir": str(product_dir),
            }
        )
        return TrainResult(
            algorithm=algorithm,
            is_embedding=(algorithm == "efficientnet_b0"),
            status_message=f"trained {algorithm}",
            dialog_message=f"done {algorithm}",
            result_rows=[],
        )


class _TrainAllHarness:
    _resolve_training_algorithm = ToolPage._resolve_training_algorithm
    _train_sample_paths_for_role = ToolPage._train_sample_paths_for_role
    _training_camera_roles_in_lists = ToolPage._training_camera_roles_in_lists
    _warn_mixed_training_camera_samples = ToolPage._warn_mixed_training_camera_samples
    _training_roi_ready_signature = ToolPage._training_roi_ready_signature
    _missing_training_roi_paths = ToolPage._missing_training_roi_paths
    _training_item_display_name = staticmethod(ToolPage._training_item_display_name)
    _build_training_task_for_item = ToolPage._build_training_task_for_item
    _training_payload = ToolPage._training_payload
    _set_training_running = ToolPage._set_training_running
    _on_training_progress = ToolPage._on_training_progress
    _start_training_worker = ToolPage._start_training_worker
    _on_training_finished = ToolPage._on_training_finished
    _train_inspection_item = ToolPage._train_inspection_item
    _train_all_tools = ToolPage._train_all_tools
    _train = ToolPage._train

    def __init__(self, product_dir: str, ok_files: list[str], ng_files: list[str]) -> None:
        self.algo = _FakeAlgo()
        self.train_files = list(ok_files) + list(ng_files)
        self.ok_files: list[str] = []
        self.ng_files: list[str] = []
        self.loc_method = "line2dup"
        self.session = type("Session", (), {"product_dir": product_dir})()
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code="shared_backbone_register",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam1",
                roi_label="roi2",
                algorithm_code="meanintensity",
            ),
        ]
        self.lbl_status = QtWidgets.QLabel("")
        self.table = QtWidgets.QTableWidget()
        self._current_result_rows = []
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["proto", "topk"])
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setValue(0.02)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setValue(3)
        self._selected_index = 0
        self.current_camera = "cam1"
        self.saved_runtime_params = 0
        self.saved_session = 0
        self.refresh_count = 0
        self.update_count = 0
        self._training_in_progress = False
        self._training_thread = None
        self._training_worker = None
        self._training_roi_ready_signatures = {}
        self._training_roi_pending_actions = {}
        self._training_roi_confirmed_signatures = {}
        self._line2dup_match_ms_by_image = {}
        self._line2dup_autogen_ms_by_image = {}
        self.ref_image = None

    def _selected_inspection_item(self):
        if 0 <= self._selected_index < len(self.inspection_items):
            return self.inspection_items[self._selected_index]
        return None

    def current_camera_role(self) -> str:
        return self.current_camera

    def _training_sample_groups_for_role(self, camera_role=None, *, roi_label=None):
        role = str(camera_role or self.current_camera)
        train_files = [path for path in self.train_files if role in os.path.basename(path) or "cam" not in os.path.basename(path)]
        ok_files = [path for path in train_files if "ok" in os.path.basename(path).lower()]
        ng_files = [path for path in train_files if "ng" in os.path.basename(path).lower()]
        return ok_files, ng_files, train_files

    def _autogen_roi_for_images(self, paths, only_missing=False, silent=False):
        return None

    def _populate_results_table(self, rows):
        self._current_result_rows = list(rows)

    def _save_runtime_params(self):
        self.saved_runtime_params += 1

    def _save_session(self):
        self.saved_session += 1

    def _refresh_inspection_items_table(self):
        self.refresh_count += 1

    def _update_runtime_widgets(self):
        self.update_count += 1

    def _ensure_training_roi_reviewed(self, _camera_role, *, action_name: str, action_key: str = ""):
        return True

    def _refresh_current_image_after_roi_update(self, candidate_paths):
        return None


class ToolPageTrainAllTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_train_all_enabled_items_trains_each_tool_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ok_path = Path(tmpdir) / "ok.png"
            ng_path = Path(tmpdir) / "ng.png"
            ok_path.write_bytes(b"ok")
            ng_path.write_bytes(b"ng")
            ok_path.with_suffix(".json").write_text(
                '{"shapes":[{"label":"roi1"},{"label":"roi2"}]}',
                encoding="utf-8",
            )
            ng_path.with_suffix(".json").write_text(
                '{"shapes":[{"label":"roi1"},{"label":"roi2"}]}',
                encoding="utf-8",
            )

            harness = _TrainAllHarness(tmpdir, [str(ok_path)], [str(ng_path)])
            harness.loc_method = "manual"

            with (
                mock.patch("PySide6.QtWidgets.QMessageBox.information") as info,
                mock.patch("PySide6.QtWidgets.QMessageBox.warning") as warning,
            ):
                harness._train_all_tools()

            self.assertEqual(
                harness.algo.train_calls,
                [
                    {
                        "algorithm": "efficientnet_b0",
                        "label_names": ["roi1"],
                        "model_key": "cam1__roi1",
                        "product_dir": tmpdir,
                    },
                    {
                        "algorithm": "meanintensity",
                        "label_names": ["roi2"],
                        "model_key": "cam1__roi2",
                        "product_dir": tmpdir,
                    },
                ],
            )
            self.assertEqual(harness.saved_runtime_params, 1)
            self.assertEqual(harness.saved_session, 1)
            self.assertEqual(harness.refresh_count, 1)
            self.assertEqual(harness.update_count, 1)
            self.assertTrue(info.called)
            self.assertFalse(warning.called)

    def test_train_all_uses_current_role_samples_when_global_lists_mix_cam1_cam2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ok_cam1 = Path(tmpdir) / "cam1_ok.png"
            ok_cam2 = Path(tmpdir) / "cam2_ok.png"
            ng_cam1 = Path(tmpdir) / "cam1_ng.png"
            ng_cam2 = Path(tmpdir) / "cam2_ng.png"
            for path in (ok_cam1, ok_cam2, ng_cam1, ng_cam2):
                path.write_bytes(b"x")
                path.with_suffix(".json").write_text(
                    '{"shapes":[{"label":"roi1"},{"label":"roi2"}]}',
                    encoding="utf-8",
                )

            harness = _TrainAllHarness(
                tmpdir,
                [str(ok_cam1), str(ok_cam2)],
                [str(ng_cam1), str(ng_cam2)],
            )
            harness.loc_method = "manual"

            with (
                mock.patch("PySide6.QtWidgets.QMessageBox.information") as info,
                mock.patch("PySide6.QtWidgets.QMessageBox.warning") as warning,
            ):
                harness._train_all_tools()

            self.assertEqual(
                harness.algo.train_calls,
                [
                    {
                        "algorithm": "efficientnet_b0",
                        "label_names": ["roi1"],
                        "model_key": "cam1__roi1",
                        "product_dir": tmpdir,
                    },
                    {
                        "algorithm": "meanintensity",
                        "label_names": ["roi2"],
                        "model_key": "cam1__roi2",
                        "product_dir": tmpdir,
                    },
                ],
            )
            self.assertEqual(harness.saved_runtime_params, 1)
            self.assertEqual(harness.saved_session, 1)
            self.assertEqual(harness.refresh_count, 1)
            self.assertEqual(harness.update_count, 1)
            self.assertTrue(info.called)
            self.assertFalse(warning.called)

    def test_train_all_line2dup_requires_second_click_after_roi_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ok_path = Path(tmpdir) / "cam1_ok.png"
            ng_path = Path(tmpdir) / "cam1_ng.png"
            for path in (ok_path, ng_path):
                path.write_bytes(b"x")
                path.with_suffix(".json").write_text(
                    '{"shapes":[{"label":"roi1"},{"label":"roi2"}]}',
                    encoding="utf-8",
                )

            harness = _TrainAllHarness(tmpdir, [str(ok_path)], [str(ng_path)])

            with (
                mock.patch.object(harness, "_ensure_training_roi_reviewed", side_effect=[False, True], create=True) as ensure_roi,
                mock.patch("PySide6.QtWidgets.QMessageBox.information") as info,
                mock.patch("PySide6.QtWidgets.QMessageBox.warning") as warning,
            ):
                harness._train_all_tools()
                self.assertEqual(harness.algo.train_calls, [])
                harness._train_all_tools()

            self.assertEqual(ensure_roi.call_count, 2)
            self.assertEqual(len(harness.algo.train_calls), 2)
            self.assertTrue(info.called)
            self.assertFalse(warning.called)

    def test_confirmed_line2dup_review_is_reused_for_same_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ok_path = Path(tmpdir) / "cam1_ok.png"
            ng_path = Path(tmpdir) / "cam1_ng.png"
            for path in (ok_path, ng_path):
                path.write_bytes(b"x")
                path.with_suffix(".json").write_text(
                    '{"shapes":[{"label":"roi1"},{"label":"roi2"}]}',
                    encoding="utf-8",
                )

            harness = _TrainAllHarness(tmpdir, [str(ok_path)], [str(ng_path)])
            harness._ensure_training_roi_reviewed = ToolPage._ensure_training_roi_reviewed.__get__(harness, _TrainAllHarness)
            harness.line2dup_recipe_path_for_role = lambda _role=None: ""
            harness.line2dup_model_path_for_role = lambda _role=None: ""
            harness.line2dup_recipe_for_role = lambda _role=None, force_reload=False: type(
                "Recipe",
                (),
                {"reference_image": str(ok_path)},
            )()
            harness.ref_image = str(ok_path)
            autogen_calls: list[str] = []

            def _fake_autogen(**kwargs):
                autogen_calls.append(os.path.basename(str(kwargs.get("tgt_img_path", ""))))
                return type("Run", (), {"total_ms": 1.0})()

            with (
                mock.patch("ui.debug.tool_page.page.line2dup_locator.autogen_roi_json_from_line2dup_timed", side_effect=_fake_autogen),
                mock.patch("PySide6.QtWidgets.QMessageBox.information"),
                mock.patch("PySide6.QtWidgets.QMessageBox.warning"),
            ):
                harness._train_all_tools()
                self.assertEqual(len(harness.algo.train_calls), 0)
                self.assertEqual(len(autogen_calls), 2)

                harness._train_all_tools()
                self.assertEqual(len(harness.algo.train_calls), 2)

                harness._train_all_tools()
                self.assertEqual(len(harness.algo.train_calls), 4)
                self.assertEqual(len(autogen_calls), 2)
