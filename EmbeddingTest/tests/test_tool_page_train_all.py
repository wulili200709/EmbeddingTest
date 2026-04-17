from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
                "traditional_models": {},
            },
        )()
        self.model = None
        self.train_calls: list[dict] = []
        self.clear_calls: list[dict] = []
        self.prune_calls: list[dict] = []
        self.force_legacy_traditional_key = False

    def is_learning_tool(self, code) -> bool:
        return str(code or "").strip() == "shared_backbone_register"

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, code) -> str:
        return str(code or "").strip()

    def is_embedding_algorithm(self, algorithm) -> bool:
        return str(algorithm or "").strip() == "efficientnet_b0"

    def traditional_model_storage_key(self, algorithm: str, *, model_key: object = "") -> str:
        normalized = str(model_key or "").strip()
        if normalized:
            return f"{algorithm}::{normalized}"
        return str(algorithm or "").strip()

    def get_traditional_model_dict(self, algorithm: str, *, model_key: object = ""):
        return self.product_params.traditional_models.get(
            self.traditional_model_storage_key(algorithm, model_key=model_key)
        )

    def train(
        self,
        ok_files,
        ng_files,
        *,
        algorithm,
        product_dir,
        label_names,
        model_key,
        ok_samples=None,
        ng_samples=None,
    ):
        self.train_calls.append(
            {
                "algorithm": algorithm,
                "label_names": list(label_names),
                "model_key": str(model_key),
                "product_dir": str(product_dir),
                "ok_samples": list(ok_samples or []),
                "ng_samples": list(ng_samples or []),
            }
        )
        if algorithm != "efficientnet_b0":
            stored_model_key = str(model_key)
            if self.force_legacy_traditional_key and str(model_key).startswith("cam1__hole"):
                stored_model_key = "cam1__roi1"
            storage_key = self.traditional_model_storage_key(algorithm, model_key=stored_model_key)
            traditional_model_dict = {
                "algorithm": str(algorithm),
                "threshold": 0.5,
                "ok_when": "greater_equal",
                "accuracy": 1.0,
            }
            self.product_params.traditional_models[storage_key] = traditional_model_dict
        else:
            traditional_model_dict = None
        return TrainResult(
            algorithm=algorithm,
            is_embedding=(algorithm == "efficientnet_b0"),
            status_message=f"trained {algorithm}",
            dialog_message=f"done {algorithm}",
            traditional_model_dict=traditional_model_dict,
            result_rows=[],
        )

    def clear_training_output(self, algorithm, product_dir, *, model_key=""):
        self.clear_calls.append(
            {
                "algorithm": str(algorithm),
                "product_dir": str(product_dir),
                "model_key": str(model_key or ""),
            }
        )

    def clear_obsolete_traditional_models(self, *, camera_role, valid_model_keys_by_algorithm):
        self.prune_calls.append(
            {
                "camera_role": str(camera_role),
                "valid_model_keys_by_algorithm": {
                    str(key): set(value)
                    for key, value in dict(valid_model_keys_by_algorithm or {}).items()
                },
            }
        )


class _TrainAllHarness:
    _effective_model_key_for_item = ToolPage._effective_model_key_for_item
    _group_items_for_inspection_item = ToolPage._group_items_for_inspection_item
    _training_samples_for_inspection_item = ToolPage._training_samples_for_inspection_item
    _store_runtime_params_for_group = ToolPage._store_runtime_params_for_group
    _clear_previous_training_output = ToolPage._clear_previous_training_output
    _prune_stale_traditional_models_for_role = ToolPage._prune_stale_traditional_models_for_role
    _reload_runtime_params_from_disk = ToolPage._reload_runtime_params_from_disk
    _resolve_training_algorithm = ToolPage._resolve_training_algorithm
    _train_sample_paths_for_role = ToolPage._train_sample_paths_for_role
    _training_camera_roles_in_lists = ToolPage._training_camera_roles_in_lists
    _warn_mixed_training_camera_samples = ToolPage._warn_mixed_training_camera_samples
    _missing_training_roi_paths = ToolPage._missing_training_roi_paths
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
        self.reload_count = 0

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

    def _path_has_roi_geometry(self, path: str, roi_label: str) -> bool:
        json_path = Path(path).with_suffix(".json")
        if not json_path.exists():
            return False
        text = json_path.read_text(encoding="utf-8")
        return f'"label":"{roi_label}"' in text or f'"label": "{roi_label}"' in text

    def _sample_roi_status_for_path(self, path: str, camera_role: object, roi_label: str) -> str:
        lower_name = os.path.basename(path).lower()
        if "ok" in lower_name:
            return "OK"
        if "ng" in lower_name:
            return "NG"
        return ""

    def _autogen_roi_for_images(self, paths, only_missing=False, silent=False):
        return None

    def _populate_results_table(self, rows):
        self._current_result_rows = list(rows)

    def _save_runtime_params(self):
        self.saved_runtime_params += 1

    def _save_session(self):
        self.saved_session += 1

    def _reload_runtime_params_from_disk(self):
        self.reload_count += 1

    def _refresh_inspection_items_table(self):
        self.refresh_count += 1

    def _update_runtime_widgets(self):
        self.update_count += 1

    def _ensure_training_roi_reviewed(self, _camera_role, *, action_name: str, action_key: str = ""):
        return True


class _TrainingConfirmHarness:
    _ensure_training_roi_reviewed = ToolPage._ensure_training_roi_reviewed
    _training_roi_ready_signature = ToolPage._training_roi_ready_signature
    _sync_training_action_buttons = ToolPage._sync_training_action_buttons

    def __init__(self, product_dir: str) -> None:
        self.loc_method = "ncc"
        self.current_camera = "cam1"
        self.ref_image = ""
        self.train_files = [str(Path(product_dir) / "cam1_ok.png")]
        self.session = SimpleNamespace(product_dir=product_dir)
        self.lbl_status = QtWidgets.QLabel("")
        self._training_roi_ready_signatures = {}
        self._training_roi_pending_actions = {}
        self._train_action_btn_style = "default-all"
        self._train_current_btn_style = "default-current"
        self._train_confirm_btn_style = "confirm"
        self.btn_train = QtWidgets.QPushButton("训练 / 标定全部启用工具")
        self.btn_train_current = QtWidgets.QPushButton("标定当前工具")
        self.btn_train_cancel = QtWidgets.QPushButton("×")
        self.btn_train_current_cancel = QtWidgets.QPushButton("×")
        self.update_count = 0

    def current_camera_role(self) -> str:
        return self.current_camera

    def _train_sample_paths_for_role(self, camera_role=None):
        return list(self.train_files)

    def _update_runtime_widgets(self):
        self.update_count += 1
        self._sync_training_action_buttons()


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
                        "ok_samples": [(str(ok_path), "roi1")],
                        "ng_samples": [(str(ng_path), "roi1")],
                    },
                    {
                        "algorithm": "meanintensity",
                        "label_names": ["roi2"],
                        "model_key": "cam1__roi2",
                        "product_dir": tmpdir,
                        "ok_samples": [(str(ok_path), "roi2")],
                        "ng_samples": [(str(ng_path), "roi2")],
                    },
                ],
            )
            self.assertEqual(
                harness.algo.clear_calls,
                [
                    {
                        "algorithm": "efficientnet_b0",
                        "product_dir": tmpdir,
                        "model_key": "cam1__roi1",
                    },
                    {
                        "algorithm": "meanintensity",
                        "product_dir": tmpdir,
                        "model_key": "cam1__roi2",
                    },
                ],
            )
            self.assertEqual(
                harness.algo.prune_calls,
                [
                    {
                        "camera_role": "cam1",
                        "valid_model_keys_by_algorithm": {"meanintensity": {"cam1__roi2"}},
                    }
                ],
            )
            self.assertEqual(harness.saved_runtime_params, 1)
            self.assertEqual(harness.saved_session, 1)
            self.assertEqual(harness.refresh_count, 1)
            self.assertEqual(harness.update_count, 1)
            self.assertEqual(harness.reload_count, 1)
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
                        "ok_samples": [(str(ok_cam1), "roi1")],
                        "ng_samples": [(str(ng_cam1), "roi1")],
                    },
                    {
                        "algorithm": "meanintensity",
                        "label_names": ["roi2"],
                        "model_key": "cam1__roi2",
                        "product_dir": tmpdir,
                        "ok_samples": [(str(ok_cam1), "roi2")],
                        "ng_samples": [(str(ng_cam1), "roi2")],
                    },
                ],
            )
            self.assertEqual(
                harness.algo.clear_calls,
                [
                    {
                        "algorithm": "efficientnet_b0",
                        "product_dir": tmpdir,
                        "model_key": "cam1__roi1",
                    },
                    {
                        "algorithm": "meanintensity",
                        "product_dir": tmpdir,
                        "model_key": "cam1__roi2",
                    },
                ],
            )
            self.assertEqual(
                harness.algo.prune_calls,
                [
                    {
                        "camera_role": "cam1",
                        "valid_model_keys_by_algorithm": {"meanintensity": {"cam1__roi2"}},
                    }
                ],
            )
            self.assertEqual(harness.saved_runtime_params, 1)
            self.assertEqual(harness.saved_session, 1)
            self.assertEqual(harness.refresh_count, 1)
            self.assertEqual(harness.update_count, 1)
            self.assertEqual(harness.reload_count, 1)
            self.assertTrue(info.called)
            self.assertFalse(warning.called)

    def test_train_all_deduplicates_grouped_traditional_tools_to_one_group_key(self) -> None:
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
            harness.loc_method = "manual"
            harness.inspection_items = [
                InspectionItem(
                    item_id="roi1",
                    display_name="hole",
                    camera_id="cam1",
                    roi_label="roi1",
                    task_group="hole",
                    algorithm_code="meanstd",
                ),
                InspectionItem(
                    item_id="roi2",
                    display_name="hole",
                    camera_id="cam1",
                    roi_label="roi2",
                    task_group="hole",
                    algorithm_code="meanstd",
                ),
            ]

            with (
                mock.patch("PySide6.QtWidgets.QMessageBox.information"),
                mock.patch("PySide6.QtWidgets.QMessageBox.warning"),
            ):
                harness._train_all_tools()

            self.assertEqual(
                harness.algo.train_calls,
                [
                    {
                        "algorithm": "meanstd",
                        "label_names": ["roi1", "roi2"],
                        "model_key": "cam1__hole",
                        "product_dir": tmpdir,
                        "ok_samples": [(str(ok_path), "roi1"), (str(ok_path), "roi2")],
                        "ng_samples": [(str(ng_path), "roi1"), (str(ng_path), "roi2")],
                    },
                ],
            )
            self.assertCountEqual(
                harness.algo.clear_calls,
                [
                    {
                        "algorithm": "meanstd",
                        "product_dir": tmpdir,
                        "model_key": "cam1__hole",
                    },
                    {
                        "algorithm": "meanstd",
                        "product_dir": tmpdir,
                        "model_key": "cam1__roi1",
                    },
                    {
                        "algorithm": "meanstd",
                        "product_dir": tmpdir,
                        "model_key": "cam1__roi2",
                    },
                ],
            )
            self.assertEqual(
                harness.algo.prune_calls,
                [
                    {
                        "camera_role": "cam1",
                        "valid_model_keys_by_algorithm": {"meanstd": {"cam1__hole"}},
                    }
                ],
            )
            self.assertIn("meanstd::cam1__hole", harness.algo.product_params.traditional_models)
            self.assertNotIn("meanstd::cam1__roi1", harness.algo.product_params.traditional_models)
            self.assertNotIn("meanstd::cam1__roi2", harness.algo.product_params.traditional_models)

    def test_train_all_repairs_grouped_traditional_storage_key(self) -> None:
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
            harness.loc_method = "manual"
            harness.algo.force_legacy_traditional_key = True
            harness.inspection_items = [
                InspectionItem(
                    item_id="roi1",
                    display_name="hole",
                    camera_id="cam1",
                    roi_label="roi1",
                    task_group="hole",
                    algorithm_code="meanstd",
                ),
                InspectionItem(
                    item_id="roi2",
                    display_name="hole",
                    camera_id="cam1",
                    roi_label="roi2",
                    task_group="hole",
                    algorithm_code="meanstd",
                ),
            ]

            with (
                mock.patch("PySide6.QtWidgets.QMessageBox.information"),
                mock.patch("PySide6.QtWidgets.QMessageBox.warning"),
            ):
                harness._train_all_tools()

            self.assertIn("meanstd::cam1__hole", harness.algo.product_params.traditional_models)
            self.assertNotIn("meanstd::cam1__roi1", harness.algo.product_params.traditional_models)

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

    def test_ncc_training_requires_second_click_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = _TrainingConfirmHarness(tmpdir)

            with mock.patch("PySide6.QtWidgets.QMessageBox.information") as info:
                first = harness._ensure_training_roi_reviewed(
                    "cam1",
                    action_name="训练 / 标定全部启用工具",
                    action_key="all",
                )
                second = harness._ensure_training_roi_reviewed(
                    "cam1",
                    action_name="训练 / 标定全部启用工具",
                    action_key="all",
                )

            self.assertFalse(first)
            self.assertTrue(second)
            self.assertEqual(harness._training_roi_pending_actions, {})
            self.assertEqual(harness.btn_train.text(), "训练 / 标定全部启用工具")
            self.assertTrue(info.called)
