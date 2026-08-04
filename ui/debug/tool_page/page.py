"""
Tool page UI for template setup, ROI review, training, and testing.

Main workflow groups:
  1. ROI editing: _load_canvas_image / _save_current_rect / _on_select_ok
  2. Shape localization: _autogen_roi_for_images / _open_shape_template_page
  3. Testing: _predict_image / _run_test / _populate_results_table / _append_test_log
  4. Validation: _suggest_margin_from_rows / _run_margin_validation / _run_traditional_baseline_debug

Signals consumed by MainWindow:
  productChangeRequested(str): ask MainWindow to switch products through apply_product_switch()
  sessionClearRequested(): ask MainWindow to clear state through reset_for_clear()
  sessionLoaded(): notify MainWindow that session state was loaded
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PySide6 import QtCore, QtGui, QtWidgets
import algorithms.lazy_api as qr_core
from common.camera_roles import CAMERA_ROLES, DEFAULT_CAMERA_ROLE, configured_camera_roles
from common.algorithm_codes import learning_backbone_storage_code
from common import labelme_io

from infrastructure.camera_settings_store import (
    CameraSettingsStore,
    hik_settings_kwargs_from_mapping,
)
from application import (
    AlgorithmController,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_EMBEDDING_ALGORITHMS,
    SUPPORTED_SCORE_MODES,
    ProductSession,
    TrainResult,
)
from domain import (
    InspectionItem,
    load_inspection_items,
    save_inspection_items,
    sync_items_with_labels,
)
from shape.core import locator as shape_locator
from shape.core.recipe import ShapeRecipe
from shape.core.recipe_labels import (
    clearable_roi_labels,
    inspection_item_specs_from_shape_recipe,
    output_labels_from_shape_recipe,
)
from ncc import locator as ncc_locator
from ui.debug import RoiCanvas
from ui.debug.tool_page.algorithm_catalog import (
    ALGORITHM_DISPLAY_NAMES,
    ALGORITHM_GROUPS,
    algorithm_display_name,
    algorithm_group_display_name,
)
from ui.debug.tool_page.camera_roles import (
    camera_role_from_path as _camera_role_from_path,
    filter_paths_for_camera as _filter_paths_for_camera,
    normalize_camera_role as _normalize_camera_role,
    selected_image_list_camera_role as _selected_image_list_camera_role,
)
from ui.debug.tool_page.debug_camera_runtime import DebugCameraPreviewThread
from ui.debug.tool_page.sample_annotation_dialog import _SampleAnnotationPreviewDialog
from ui.debug.tool_page.roi_annotation_controller import RoiAnnotationController
from ui.debug.tool_page.product_session_controller import ProductSessionController
from ui.debug.tool_page.sample_list_controller import SampleListController
from ui.debug.tool_page.test_execution_controller import TestExecutionController
from ui.debug.tool_page.training_controller import TrainingController
from ui.debug.tool_page.training_worker import TrainingJobWorker
from ui.debug.tool_page.view_builders import (
    build_action_panel,
    build_camera_debug_page,
    build_io_debug_page,
    build_sample_panel,
    build_tool_config_panel,
)
from ui.i18n import language_code, tr
from ui.roi_overlay_colors import is_roi_label


SUPPORTED_LOC_MODES = ["shape", "ncc"]
LOC_METHOD_TRANSLATION_KEYS = {
    "shape": "action.template_editor",
    "ncc": "action.ncc_tool",
}
SUPPORTED_SHAPES = ["rect", "polygon"]
ROI_OVERLAY_PALETTE = [
    QtGui.QColor(255, 215, 0),
    QtGui.QColor(255, 64, 128),
    QtGui.QColor(0, 0, 255),
    QtGui.QColor(0, 255, 128),
    QtGui.QColor(255, 128, 0),
    QtGui.QColor(128, 255, 0),
]

def _pixmap_from_path(path: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(path)


def _normalize_loc_method(method: object, *, default: str = "shape") -> str:
    value = str(method or "").strip().lower()
    if value == "line2dup":
        value = "shape"
    if value in SUPPORTED_LOC_MODES:
        return value
    return default if default in SUPPORTED_LOC_MODES else "shape"


def _loc_method_display_name(method: object) -> str:
    value = _normalize_loc_method(method)
    key = LOC_METHOD_TRANSLATION_KEYS.get(value)
    return tr(key) if key else value


# ---------------------------------------------------------------------------
# ToolPage
# ---------------------------------------------------------------------------

class ToolPage(QtWidgets.QWidget):
    """
    Main debug tool page widget.

    Typical MainWindow integration:

        self.tool_page = ToolPage(self.session, self.algo)
        self.tool_page.productChangeRequested.connect(self._on_product_change_request)
        self.tool_page.sessionClearRequested.connect(self._on_session_clear_request)
        self.tool_page.sessionLoaded.connect(lambda: self._refresh_runtime_status_ui("session loaded"))
        self.main_pages.addTab(self.tool_page, "Debug")
        self.tool_page.load_session()
    """

    productChangeRequested = QtCore.Signal(str)   # new product name
    productDeleteRequested = QtCore.Signal(str)
    sessionClearRequested = QtCore.Signal()
    sessionLoaded = QtCore.Signal()
    inspectionItemsChanged = QtCore.Signal()
    roiGeometryChanged = QtCore.Signal()
    debugCameraConnectRequested = QtCore.Signal(str)
    debugCameraConnected = QtCore.Signal(str, str)
    cameraSettingsApplied = QtCore.Signal(str, object)


    def __init__(
        self,
        session: ProductSession,
        algo: AlgorithmController,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        lite_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.algo = algo
        self.lite_mode = bool(lite_mode)
        self._configured_camera_roles: List[str] = list(CAMERA_ROLES)

        self.train_files: List[str] = []
        self.ok_files: List[str] = []
        self.ng_files: List[str] = []
        self.test_files: List[str] = []
        self.ref_image: Optional[str] = None
        self.loc_method: str = "shape"
        self._loc_methods_by_role: Dict[str, str] = {role: "shape" for role in CAMERA_ROLES}
        self.shape_recipe: Optional[ShapeRecipe] = None
        self._shape_recipes_by_role: Dict[str, Optional[ShapeRecipe]] = {}
        self._training_roi_ready_signatures: Dict[str, str] = {}
        self._training_roi_pending_actions: Dict[str, str] = {}
        self.inspection_items: List[InspectionItem] = []
        self._visible_inspection_item_indexes: List[int] = []
        self._inspection_items_table_loading = False
        self._shape_match_ms_by_image: Dict[str, float] = {}
        self._shape_autogen_ms_by_image: Dict[str, float] = {}
        self._current_result_rows: List[Dict[str, object]] = []
        self._roi_results_by_image: Dict[str, Dict[str, str]] = {}
        self._sample_roi_annotations_by_path: Dict[str, Dict[str, str]] = {}
        self._updating_runtime_params = False
        self._skip_empty_autogen_message = False
        self._tool_dialogs: Dict[str, QtWidgets.QDialog] = {}
        self._template_editor_dialog: Optional[QtWidgets.QDialog] = None
        self._ncc_workbench_dialog: Optional[QtWidgets.QDialog] = None
        self._sample_annotation_preview_dialog: Optional[QtWidgets.QDialog] = None
        self._debug_camera_manager = None
        self._debug_frame_grab_service = None
        self._debug_camera_infos: List[object] = []
        self._debug_preview_thread: Optional[DebugCameraPreviewThread] = None
        self._debug_io_controller = None
        self._runtime_io_controller = None
        self._runtime_io_ready = False
        self._runtime_io_status_detail = "IO not initialized"
        self._debug_io_uses_runtime_controller = False
        self._debug_output_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._debug_di_cards: Dict[int, QtWidgets.QFrame] = {}
        self._debug_di_indicators: Dict[int, QtWidgets.QLabel] = {}
        self._debug_di_hints: Dict[int, QtWidgets.QLabel] = {}
        self._debug_do_cards: Dict[int, QtWidgets.QFrame] = {}
        self._debug_do_channel_buttons: Dict[int, QtWidgets.QPushButton] = {}
        self._debug_do_hints: Dict[int, QtWidgets.QLabel] = {}
        self._debug_io_timer = QtCore.QTimer(self)
        self._debug_io_timer.setInterval(500)
        self._debug_io_timer.timeout.connect(self._refresh_debug_io_snapshot)
        self._camera_settings_store = CameraSettingsStore(self.session.camera_settings_path)
        self._current_camera_role = "cam1"
        # Guard against recursive setValue/apply callbacks in camera controls.
        self._debug_camera_block_spin_apply = False
        self._capture_config_loading = False
        self._main_right_panel: Optional[QtWidgets.QFrame] = None
        self._algorithm_picker_style_default = ""
        self._algorithm_picker_style_compact = ""
        self._defer_initial_session_load = True
        self._deferred_session_load_scheduled = False
        self._training_thread: Optional[QtCore.QThread] = None
        self._training_worker: Optional[TrainingJobWorker] = None
        self._training_in_progress = False
        self._training_roi_confirmed_signatures: Dict[str, str] = {}
        self.product_session_controller = ProductSessionController(self)
        self.roi_annotations = RoiAnnotationController(self)
        self.sample_list_controller = SampleListController(self)
        self.test_execution_controller = TestExecutionController(self)
        self.training_controller = TrainingController(self, TrainingJobWorker)

        self._build_ui()
        self._set_current_camera_role(self._current_camera_role)
        self._apply_configured_camera_roles_to_ui()
        QtCore.QTimer.singleShot(0, self._update_responsive_layout)
        if not self.lite_mode:
            self.destroyed.connect(lambda *_: self._cleanup_debug_hardware())

    # ------------------------------------------------------------------
    # Public API used by MainWindow
    # ------------------------------------------------------------------

    def retranslate_ui(self) -> None:
        for attr, text in (
            ("lbl_product_caption", tr("debug.product")),
            ("lbl_current_camera_caption", tr("debug.current_camera")),
            ("lbl_images_section", tr("debug.image_list")),
            ("lbl_algo_tool", tr("debug.tool")),
            ("lbl_algo_decision", tr("debug.decision")),
            ("lbl_algo_threshold", tr("debug.threshold")),
            ("lbl_action_section", tr("debug.actions")),
            ("lbl_roi_shape_caption", tr("debug.shape")),
            ("lbl_roi_label_caption", tr("debug.label")),
            ("lbl_loc_method", tr("debug.location_method")),
            ("lbl_cam_left_title", tr("debug.device_list")),
            ("lbl_debug_role", tr("debug.debug_role")),
            ("lbl_cam_info_title", tr("debug.device_info")),
            ("lbl_cam_right_title", tr("debug.param_settings")),
            ("lbl_io_ctrl_title", tr("debug.connection_control")),
            ("lbl_io_status_title", tr("debug.io_status")),
            ("lbl_io_panel_title", tr("debug.dido_panel")),
            ("lbl_di_title", tr("debug.di_monitor")),
            ("lbl_do_title", tr("debug.do_control")),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(text)

        for attr, text in (
            ("btn_new_product", tr("debug.new")),
            ("btn_delete_product", tr("debug.delete_product")),
            ("btn_import_train", tr("debug.add_external_images")),
            ("btn_train_to_test", tr("debug.move_to_test")),
            ("btn_sample_annotation", tr("debug.sample_annotation")),
            ("btn_del_ok", tr("debug.remove")),
            ("btn_test_to_train", tr("debug.move_to_train")),
            ("btn_add_test", tr("debug.add_external_images")),
            ("btn_del_test", tr("debug.remove")),
            ("btn_sample_annotation_test", tr("debug.clear_current_test_list")),
            ("btn_toggle_algo", tr("debug.algorithm_params")),
            ("btn_toggle_tools", tr("debug.inspection_tools")),
            ("btn_train", tr("debug.train_all_tools")),
            ("btn_train_current", tr("debug.calibrate_current_tool")),
            ("btn_test", tr("debug.test_current_image")),
            ("btn_export_test", tr("debug.export_report")),
            ("btn_clear_session", tr("debug.test_all_test_samples")),
            ("btn_save", tr("debug.save_annotation")),
            ("btn_clear", tr("debug.clear_annotation")),
            ("btn_set_ref", tr("debug.set_as_reference")),
            ("btn_pick_ref", tr("debug.pick_reference")),
            ("chk_only_missing", tr("debug.only_missing_roi")),
            ("btn_autogen", tr("debug.batch_roi_current")),
            ("btn_autogen_all", tr("debug.batch_roi_all")),
            ("btn_clear_roi_batch", tr("debug.clear_roi_current")),
            ("btn_debug_refresh_camera", tr("debug.scan_camera")),
            ("btn_debug_connect_camera", tr("debug.connect")),
            ("btn_debug_disconnect_camera", tr("debug.disconnect")),
            ("btn_debug_live_preview", tr("debug.live_preview")),
            ("btn_debug_grab_once", tr("debug.grab_to_test")),
            ("btn_debug_save_image", tr("debug.save_image")),
            ("chk_debug_digital_shift_enable", tr("debug.enable")),
            ("btn_debug_read_camera_settings", tr("debug.read_camera_params")),
            ("btn_debug_apply_camera_settings", tr("debug.apply_camera_params")),
            ("btn_debug_open_io", tr("debug.open_io_debug")),
            ("btn_debug_close_io", tr("debug.close_io_debug")),
            ("btn_debug_refresh_io", tr("debug.refresh_dido")),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(text)

        if hasattr(self, "cmb_algorithm"):
            current_algorithm = self.current_algorithm()
            blocker = QtCore.QSignalBlocker(self.cmb_algorithm)
            self._populate_algorithm_combo()
            index = self._find_algorithm_combo_index(current_algorithm)
            self.cmb_algorithm.setCurrentIndex(index if index >= 0 else -1)
            del blocker
        if hasattr(self, "btn_algorithm_picker"):
            self.btn_algorithm_picker.setMenu(self._build_algorithm_picker_menu())
            self._sync_algorithm_picker()
        if hasattr(self, "tabs"):
            self._update_sample_panel_widgets()
        if hasattr(self, "inspection_items_table"):
            self.inspection_items_table.setHorizontalHeaderLabels([
                tr("debug.tools_table.enabled"),
                tr("debug.tools_table.name"),
                tr("debug.tools_table.camera"),
                tr("debug.tools_table.algorithm"),
                tr("debug.tools_table.status"),
            ])
            self._refresh_inspection_items_table()
        if hasattr(self, "template_match_box"):
            self.template_match_box.setTitle(tr("debug.auto_roi"))
        if hasattr(self, "cmb_loc"):
            self._populate_loc_combo()
        if hasattr(self, "lbl_ref"):
            self._apply_current_role_recipe_state()
        if hasattr(self, "btn_train_cancel"):
            self.btn_train_cancel.setToolTip(tr("debug.cancel_train_confirm"))
        if hasattr(self, "btn_train_current_cancel"):
            self.btn_train_current_cancel.setToolTip(tr("debug.cancel_current_confirm"))
        if hasattr(self, "cmb_debug_light_source_mode"):
            current_data = self.cmb_debug_light_source_mode.currentData()
            blocker = QtCore.QSignalBlocker(self.cmb_debug_light_source_mode)
            self.cmb_debug_light_source_mode.clear()
            self.cmb_debug_light_source_mode.addItem(tr("debug.board_do_light"), "board_io")
            self.cmb_debug_light_source_mode.addItem(tr("debug.camera_line1_strobe"), "camera_line1_strobe")
            index = self.cmb_debug_light_source_mode.findData(current_data)
            self.cmb_debug_light_source_mode.setCurrentIndex(max(0, index))
            self.cmb_debug_light_source_mode.setToolTip(tr("debug.camera_line1_tip"))
            del blocker
        if hasattr(self, "cmb_capture_mode"):
            current_data = self.cmb_capture_mode.currentData()
            blocker = QtCore.QSignalBlocker(self.cmb_capture_mode)
            self.cmb_capture_mode.clear()
            self.cmb_capture_mode.addItem(tr("debug.capture_mode_independent"), "independent")
            self.cmb_capture_mode.addItem(tr("debug.capture_mode_single_multi_light"), "single_multi_light")
            self.cmb_capture_mode.addItem(tr("debug.capture_mode_flexible"), "flexible")
            index = self.cmb_capture_mode.findData(current_data)
            self.cmb_capture_mode.setCurrentIndex(max(0, index))
            del blocker
        if hasattr(self, "lbl_capture_mode_title"):
            self.lbl_capture_mode_title.setText(tr("debug.capture_mode"))
        if hasattr(self, "lbl_capture_channel_title"):
            self.lbl_capture_channel_title.setText(tr("debug.capture_channels"))
        if hasattr(self, "capture_channel_table"):
            self.capture_channel_table.setHorizontalHeaderLabels([
                tr("debug.capture_table.enabled"),
                tr("debug.capture_table.channel"),
                tr("debug.capture_table.camera"),
                tr("debug.capture_table.light"),
                tr("debug.capture_table.exposure"),
                tr("debug.capture_table.gain"),
            ])
            light_options = [
                (tr("debug.io_name.light_cam1"), "DO_LIGHT_CAM1"),
                (tr("debug.io_name.light_cam2"), "DO_LIGHT_CAM2"),
                (tr("debug.io_name.light_cam3"), "DO_LIGHT_CAM3"),
            ]
            for row in range(self.capture_channel_table.rowCount()):
                combo = self.capture_channel_table.cellWidget(row, 3)
                if not isinstance(combo, QtWidgets.QComboBox):
                    continue
                current_data = combo.currentData()
                blocker = QtCore.QSignalBlocker(combo)
                combo.clear()
                for label, data in light_options:
                    combo.addItem(label, data)
                index = combo.findData(current_data)
                combo.setCurrentIndex(index if index >= 0 else min(row, combo.count() - 1))
                del blocker
            self._update_capture_channel_visibility()
        form = getattr(self, "cam_params_form", None)
        if form is not None:
            row_labels = {
                "spin_debug_exposure": tr("debug.exposure"),
                "spin_debug_gain": tr("debug.gain"),
                "chk_debug_digital_shift_enable": tr("debug.digital_shift_enable"),
                "spin_debug_digital_shift": tr("debug.digital_shift"),
                "cmb_debug_trigger_mode": tr("debug.trigger_mode"),
                "cmb_debug_light_source_mode": tr("debug.light_source"),
            }
            for field_attr, label_text in row_labels.items():
                field = getattr(self, field_attr, None)
                label = form.labelForField(field) if field is not None else None
                if label is not None:
                    label.setText(label_text)
        measurement_form = getattr(self, "measurement_form", None)
        if measurement_form is not None:
            row_labels = {
                "cmb_measurement_line_a_tool": tr("debug.measurement.line_a_tool"),
                "cmb_measurement_line_b_tool": tr("debug.measurement.line_b_tool"),
                "cmb_measurement_distance_mode": tr("debug.measurement.distance_mode"),
                "cmb_measurement_line_a_direction": tr("debug.measurement.line_a_direction"),
                "cmb_measurement_line_b_direction": tr("debug.measurement.line_b_direction"),
                "cmb_measurement_polarity": tr("debug.measurement.polarity"),
                "spin_measurement_edge_threshold": tr("debug.measurement.edge_threshold"),
                "spin_measurement_scan_step": tr("debug.measurement.scan_step"),
                "spin_measurement_min_points": tr("debug.measurement.min_points"),
                "cmb_measurement_unit": tr("debug.measurement.unit"),
                "spin_measurement_pixel_size": tr("debug.measurement.pixel_size"),
            }
            for field_attr, label_text in row_labels.items():
                field = getattr(self, field_attr, None)
                label = measurement_form.labelForField(field) if field is not None else None
                if label is not None:
                    label.setText(label_text)
        if hasattr(self, "chk_measurement_lower"):
            self.chk_measurement_lower.setText(tr("debug.measurement.use_lower"))
        if hasattr(self, "chk_measurement_upper"):
            self.chk_measurement_upper.setText(tr("debug.measurement.use_upper"))
        if hasattr(self, "btn_add_line_distance_tool"):
            self.btn_add_line_distance_tool.setText(tr("debug.measurement.add_line_distance_tool"))
        if hasattr(self, "btn_delete_line_distance_tool"):
            self.btn_delete_line_distance_tool.setText(tr("debug.measurement.delete_line_distance_tool"))
        if hasattr(self, "cmb_measurement_line_a_direction"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_line_a_direction)
            for index, key in enumerate((
                "debug.measurement.direction.left_right",
                "debug.measurement.direction.right_left",
                "debug.measurement.direction.top_down",
                "debug.measurement.direction.bottom_up",
            )):
                if index < self.cmb_measurement_line_a_direction.count():
                    self.cmb_measurement_line_a_direction.setItemText(index, tr(key))
            del blocker
        if hasattr(self, "cmb_measurement_line_b_direction"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_line_b_direction)
            for index, key in enumerate((
                "debug.measurement.direction.left_right",
                "debug.measurement.direction.right_left",
                "debug.measurement.direction.top_down",
                "debug.measurement.direction.bottom_up",
            )):
                if index < self.cmb_measurement_line_b_direction.count():
                    self.cmb_measurement_line_b_direction.setItemText(index, tr(key))
            del blocker
        if hasattr(self, "cmb_measurement_distance_mode"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_distance_mode)
            for index, key in enumerate((
                "debug.measurement.distance_mode.vertical",
                "debug.measurement.distance_mode.horizontal",
                "debug.measurement.distance_mode.euclidean",
            )):
                if index < self.cmb_measurement_distance_mode.count():
                    self.cmb_measurement_distance_mode.setItemText(index, tr(key))
            del blocker
        if hasattr(self, "cmb_measurement_polarity"):
            blocker = QtCore.QSignalBlocker(self.cmb_measurement_polarity)
            for index, key in enumerate((
                "debug.measurement.polarity.any",
                "debug.measurement.polarity.dark_to_bright",
                "debug.measurement.polarity.bright_to_dark",
            )):
                if index < self.cmb_measurement_polarity.count():
                    self.cmb_measurement_polarity.setItemText(index, tr(key))
            del blocker
        dialog = getattr(self, "_template_editor_dialog", None)
        if dialog is not None and hasattr(dialog, "retranslate_ui"):
            dialog.retranslate_ui()
        self._retranslate_tool_dialogs()
        if hasattr(self, "lbl_debug_di_snapshot") and getattr(self, "_debug_io_controller", None) is None:
            self.lbl_debug_di_snapshot.setText(tr("debug.di_disconnected"))
            self.lbl_debug_do_snapshot.setText(tr("debug.do_disconnected"))
            self.lbl_debug_io_mapping_summary.setText(tr("debug.mapping_not_loaded"))
        self._sync_footer()

    def _tool_dialog_title(self, key: str) -> str:
        title_keys = {
            "camera_debug": "action.camera_tool",
            "io_debug": "action.io_tool",
            "template_match": "action.auto_region",
        }
        return tr(title_keys.get(str(key), str(key)))

    def _retranslate_tool_dialogs(self) -> None:
        for key, dialog in getattr(self, "_tool_dialogs", {}).items():
            if dialog is not None:
                dialog.setWindowTitle(self._tool_dialog_title(key))

    def current_algorithm(self) -> str:
        value = self.cmb_algorithm.currentData() if hasattr(self, "cmb_algorithm") else None
        if value is None:
            return ""
        return str(value).strip()

    def configured_camera_roles(self) -> List[str]:
        roles: List[str] = []
        for role in list(getattr(self, "_configured_camera_roles", []) or []):
            normalized = _normalize_camera_role(role)
            if normalized and normalized not in roles:
                roles.append(normalized)
        if not roles:
            roles = [DEFAULT_CAMERA_ROLE]
        if DEFAULT_CAMERA_ROLE not in roles:
            roles.insert(0, DEFAULT_CAMERA_ROLE)
        return roles

    def set_configured_camera_roles(self, roles: List[str]) -> None:
        normalized: List[str] = []
        for role in roles:
            role_text = _normalize_camera_role(role)
            if role_text and role_text not in normalized:
                normalized.append(role_text)
        if not normalized:
            normalized = [DEFAULT_CAMERA_ROLE]
        if DEFAULT_CAMERA_ROLE not in normalized:
            normalized.insert(0, DEFAULT_CAMERA_ROLE)
        self._configured_camera_roles = normalized
        self._apply_configured_camera_roles_to_ui()

    def current_camera_role(self) -> str:
        combo = getattr(self, "cmb_current_camera_role", None)
        if combo is None:
            return _normalize_camera_role(getattr(self, "_current_camera_role", "cam1")) or "cam1"
        return _normalize_camera_role(combo.currentData() or combo.currentText() or self._current_camera_role) or "cam1"

    def _reset_loc_methods(self) -> None:
        self._loc_methods_by_role = {role: "shape" for role in CAMERA_ROLES}
        self.loc_method = self.loc_method_for_role(self.current_camera_role())
        self._sync_loc_combo()

    def _set_session_loc_methods(self, default_method: object, loc_methods: Dict[str, str] | None) -> None:
        default = _normalize_loc_method(default_method)
        raw_methods = dict(loc_methods or {})
        self._loc_methods_by_role = {
            role: _normalize_loc_method(raw_methods.get(role, default), default=default)
            for role in CAMERA_ROLES
        }
        self.loc_method = self.loc_method_for_role(self.current_camera_role())

    def loc_method_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return _normalize_loc_method(
            self._loc_methods_by_role.get(role, self.loc_method),
            default=_normalize_loc_method(self.loc_method),
        )

    def _set_loc_method_for_role(self, camera_role: object, method: object, *, sync_combo: bool = True) -> None:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        value = _normalize_loc_method(method)
        self._loc_methods_by_role[role] = value
        if role == self.current_camera_role():
            self.loc_method = value
            if sync_combo:
                self._sync_loc_combo()

    def _populate_loc_combo(self) -> None:
        combo = getattr(self, "cmb_loc", None)
        if combo is None:
            return
        blocker = QtCore.QSignalBlocker(combo)
        combo.clear()
        for method in SUPPORTED_LOC_MODES:
            combo.addItem(_loc_method_display_name(method), method)
        del blocker
        self._sync_loc_combo()

    def _sync_loc_combo(self) -> None:
        combo = getattr(self, "cmb_loc", None)
        if combo is None:
            return
        method = self.loc_method_for_role(self.current_camera_role())
        blocker = QtCore.QSignalBlocker(combo)
        index = combo.findData(method)
        if index < 0:
            index = combo.findText(method)
        combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker

    def ncc_model_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return ncc_locator.resolved_model_path_for_product(self.session.product_dir, role)

    def ncc_model_ready_for_role(self, camera_role: object = None) -> bool:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return ncc_locator.model_is_ready(self.session.product_dir, role)

    def _apply_camera_role_options_to_combo(self, combo: object) -> None:
        if combo is None:
            return
        allowed_roles = self.configured_camera_roles()
        current_role = _normalize_camera_role(
            combo.currentData() if hasattr(combo, "currentData") else ""
        ) or _normalize_camera_role(getattr(self, "_current_camera_role", DEFAULT_CAMERA_ROLE)) or DEFAULT_CAMERA_ROLE
        blocker = QtCore.QSignalBlocker(combo) if hasattr(combo, "blockSignals") else None
        try:
            if hasattr(combo, "clear"):
                combo.clear()
            if hasattr(combo, "addItem"):
                for role in allowed_roles:
                    combo.addItem(role, role)
            if hasattr(combo, "findData") and hasattr(combo, "setCurrentIndex"):
                index = combo.findData(current_role)
                if index < 0:
                    index = combo.findData(allowed_roles[0])
                combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            if blocker is not None:
                del blocker
        if hasattr(combo, "setEnabled"):
            combo.setEnabled(len(allowed_roles) > 1)

    def _apply_configured_camera_roles_to_ui(self) -> None:
        allowed_roles = self.configured_camera_roles()
        self._apply_camera_role_options_to_combo(getattr(self, "cmb_current_camera_role", None))
        self._apply_camera_role_options_to_combo(getattr(self, "cmb_debug_camera_role", None))
        if _normalize_camera_role(getattr(self, "_current_camera_role", "")) not in set(allowed_roles):
            self._set_current_camera_role(allowed_roles[0], sync_debug_role=True)
        else:
            refresh_role_status = getattr(self, "_refresh_debug_role_status", None)
            if callable(refresh_role_status):
                refresh_role_status()
        dialog = getattr(self, "_sample_annotation_preview_dialog", None)
        if dialog is not None and hasattr(dialog, "sync_camera_roles"):
            dialog.sync_camera_roles(allowed_roles)

    def shape_paths_for_role(self, camera_role: object = None):
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return self.session.shape_paths_for_role(role)

    def load_shape_recipe_for_role(
        self,
        camera_role: object = None,
        *,
        force_reload: bool = False,
    ) -> Optional[ShapeRecipe]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        if not force_reload and role in self._shape_recipes_by_role:
            recipe = self._shape_recipes_by_role.get(role)
        else:
            paths = self.shape_paths_for_role(role)
            if not (os.path.exists(paths.recipe_path) or os.path.exists(paths.legacy_recipe_path)):
                recipe = None
            else:
                try:
                    recipe = shape_locator.load_recipe_for_product(self.session.product_dir, role)
                except Exception:
                    recipe = None
            self._shape_recipes_by_role[role] = recipe
        if role == self.current_camera_role():
            self.shape_recipe = recipe
        return recipe

    def shape_recipe_for_role(
        self,
        camera_role: object = None,
        *,
        force_reload: bool = False,
    ) -> Optional[ShapeRecipe]:
        return self.load_shape_recipe_for_role(camera_role, force_reload=force_reload)

    def shape_model_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        paths = self.shape_paths_for_role(role)
        return shape_locator.resolved_model_path_for_product(self.session.product_dir, role)

    def shape_recipe_path_for_role(self, camera_role: object = None) -> str:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        return shape_locator.resolved_recipe_path_for_product(self.session.product_dir, role)

    def _selected_image_list_camera_role(self) -> str:
        return _selected_image_list_camera_role(self)

    def _filter_paths_for_camera(self, paths: List[str], camera_id: object) -> List[str]:
        return _filter_paths_for_camera(self, paths, camera_id)

    def _apply_current_role_recipe_state(self) -> None:
        role = self.current_camera_role()
        method = self.loc_method_for_role(role)
        if method == "ncc":
            self.shape_recipe = None
            self.ref_image = None
            model_path = self.ncc_model_path_for_role(role)
            if os.path.exists(model_path):
                labels: List[str] = []
                try:
                    labels = ncc_locator.output_labels_for_product(self.session.product_dir, role)
                except Exception:
                    labels = []
                suffix = f" ({', '.join(labels)})" if labels else ""
                self.lbl_ref.setText(
                    tr(
                        "debug.ncc_model",
                        model=os.path.basename(model_path),
                        labels=suffix,
                    )
                )
                self.lbl_ref.setToolTip(model_path)
            else:
                self.lbl_ref.setText(tr("debug.ncc_model_not_set", role=role))
                self.lbl_ref.setToolTip(model_path)
            return

        recipe = self.load_shape_recipe_for_role(role)
        self.shape_recipe = recipe
        ref_image = ""
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = str(recipe.reference_image)
        self.ref_image = ref_image or None
        if self.ref_image:
            self.lbl_ref.setText(f"{tr('debug.reference_image')}: {os.path.basename(self.ref_image)}")
            self.lbl_ref.setToolTip(self.ref_image)
        else:
            self.lbl_ref.setText(tr("debug.reference_image_not_set"))
            self.lbl_ref.setToolTip("")

    def _train_sample_paths_for_role(self, camera_role: object = None) -> List[str]:
        return self.training_controller.train_sample_paths_for_role(camera_role)

    def _training_sample_groups_for_role(
        self,
        camera_role: object = None,
        *,
        roi_label: object = None,
    ) -> tuple[List[str], List[str], List[str]]:
        return self.training_controller.sample_groups_for_role(camera_role, roi_label=roi_label)

    def _training_roi_ready_signature(self, camera_role: object = None) -> str:
        return self.training_controller.ready_signature(camera_role)

    def _refresh_current_image_after_roi_update(self, candidate_paths: List[str]) -> None:
        self.training_controller.refresh_current_image_after_roi_update(candidate_paths)

    def _clear_training_roi_review_state(self, camera_role: object = None) -> None:
        self.training_controller.clear_review_state(camera_role)

    def _sync_training_action_buttons(self) -> None:
        self.training_controller.sync_action_buttons()

    def _cancel_training_pending_action(self, action_key: str | None = None) -> None:
        self.training_controller.cancel_pending_action(action_key)

    def _ensure_training_roi_reviewed(self, camera_role: object, *, action_name: str, action_key: str) -> bool:
        return self.training_controller.ensure_roi_reviewed(camera_role, action_name=action_name, action_key=action_key)

    def _set_current_camera_role(self, role: object, *, sync_debug_role: bool = True) -> None:
        normalized = _normalize_camera_role(role) or "cam1"
        previous = getattr(self, "_current_camera_role", "cam1")
        self._current_camera_role = normalized

        combo = getattr(self, "cmb_current_camera_role", None)
        if combo is not None:
            index = combo.findData(normalized)
            if index >= 0 and combo.currentIndex() != index:
                blocker = QtCore.QSignalBlocker(combo)
                combo.setCurrentIndex(index)
                del blocker

        if sync_debug_role:
            debug_combo = getattr(self, "cmb_debug_camera_role", None)
            if debug_combo is not None:
                index = debug_combo.findData(normalized)
                if index >= 0 and debug_combo.currentIndex() != index:
                    blocker = QtCore.QSignalBlocker(debug_combo)
                    debug_combo.setCurrentIndex(index)
                    del blocker

        if previous != normalized:
            self._clear_image_view_for_role_switch()
        self.loc_method = self.loc_method_for_role(normalized)
        self._sync_loc_combo()
        self._apply_current_role_recipe_state()
        self._refresh_lists()
        self._refresh_inspection_items_table()
        self._update_runtime_widgets()
        refresh_role_status = getattr(self, "_refresh_debug_role_status", None)
        if callable(refresh_role_status):
            refresh_role_status()

    def _on_current_camera_role_changed(self, value: str) -> None:
        self._set_current_camera_role(value, sync_debug_role=True)
        connected_serial = str(getattr(self._debug_camera_device(), "serial_number", "") or "").strip()
        self._apply_debug_role_binding_to_camera_combo()
        self._refresh_debug_camera_info()
        self._load_saved_debug_camera_settings_to_ui(self._selected_debug_camera_serial())
        selected_serial = self._selected_debug_camera_serial()
        if connected_serial and connected_serial != selected_serial:
            self._disconnect_debug_camera()
            self._set_debug_camera_status(tr("debug.status_switched_camera_reconnect", role=self.current_camera_role()))

    def current_algorithm_display_name(self) -> str:
        algorithm = self.current_algorithm()
        if not algorithm:
            return ""
        return algorithm_display_name(ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm), algorithm)

    def _populate_algorithm_combo(self) -> None:
        self.cmb_algorithm.clear()
        model = self.cmb_algorithm.model()
        for group_name, items in ALGORITHM_GROUPS:
            self.cmb_algorithm.addItem(algorithm_group_display_name(group_name), None)
            header_index = self.cmb_algorithm.count() - 1
            header_item = model.item(header_index) if hasattr(model, "item") else None
            if header_item is not None:
                header_item.setEnabled(False)
                header_font = QtGui.QFont(header_item.font())
                if header_font.pointSizeF() <= 0:
                    fallback_font = QtWidgets.QApplication.font(self.cmb_algorithm)
                    fallback_point_size = int(round(fallback_font.pointSizeF()))
                    if fallback_point_size > 0:
                        header_font.setPointSize(fallback_point_size)
                    elif fallback_font.pixelSize() > 0:
                        header_font.setPointSize(max(1, int(round(fallback_font.pixelSize() * 0.75))))
                    elif header_font.pixelSize() > 0:
                        header_font.setPointSize(max(1, int(round(header_font.pixelSize() * 0.75))))
                    else:
                        header_font.setPointSize(10)
                header_font.setBold(True)
                header_item.setFont(header_font)
                header_item.setForeground(QtGui.QColor("#9fd2ff"))

            for label, code, enabled in items:
                self.cmb_algorithm.addItem(algorithm_display_name(label, code), code if enabled else None)
                item_index = self.cmb_algorithm.count() - 1
                item = model.item(item_index) if hasattr(model, "item") else None
                if item is not None and not enabled:
                    item.setEnabled(False)
                    item.setForeground(QtGui.QColor("#707070"))
                    item.setToolTip(tr("debug.tool_unimplemented"))

        self.cmb_algorithm.setCurrentIndex(-1)

    def _find_algorithm_combo_index(self, algorithm: str) -> int:
        for index in range(self.cmb_algorithm.count()):
            value = self.cmb_algorithm.itemData(index)
            if value == algorithm:
                return index
        return -1

    def _set_current_algorithm(self, algorithm: str) -> None:
        algorithm = str(algorithm or "").strip()
        if not algorithm:
            self.cmb_algorithm.setCurrentIndex(-1)
            self._sync_algorithm_picker()
            return
        index = self._find_algorithm_combo_index(algorithm)
        if index >= 0:
            self.cmb_algorithm.setCurrentIndex(index)
        else:
            self.cmb_algorithm.setCurrentIndex(-1)
        self._sync_algorithm_picker()
        return

    def _build_algorithm_picker_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#3a3a3a;color:#e0e0e0;border:1px solid #505050;}"
            "QMenu::item{padding:6px 24px;}"
            "QMenu::item:selected{background:#3794ff;}"
        )
        self._algorithm_actions: Dict[str, QtGui.QAction] = {}
        self._algorithm_action_group = QtGui.QActionGroup(self)
        self._algorithm_action_group.setExclusive(True)
        for group_name, items in ALGORITHM_GROUPS:
            submenu = menu.addMenu(algorithm_group_display_name(group_name))
            for label, code, enabled in items:
                action = submenu.addAction(algorithm_display_name(label, code))
                if not enabled:
                    action.setEnabled(False)
                    action.setToolTip(tr("debug.tool_unimplemented"))
                    continue
                action.setCheckable(True)
                self._algorithm_action_group.addAction(action)
                action.triggered.connect(
                    lambda checked=False, algorithm=code: self._set_current_algorithm(algorithm)
                )
                self._algorithm_actions[code] = action
        return menu

    def _sync_algorithm_picker(self) -> None:
        button = getattr(self, "btn_algorithm_picker", None)
        if button is not None:
            text = self.current_algorithm_display_name() or tr("debug.select_tool")
            button.setText(text)
            button.setToolTip(text)
        actions = getattr(self, "_algorithm_actions", {})
        current_algorithm = self.current_algorithm()
        for code, action in actions.items():
            action.setChecked(code == current_algorithm)

    def open_camera_debug_dialog(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("camera.edit_params", "相机工具"):
            return
        if self.lite_mode or not hasattr(self, "camera_debug_page"):
            QtWidgets.QMessageBox.information(self, "LC System Lite", "轻量版未加载相机调试模块。")
            return
        self._show_tool_dialog(
            "camera_debug",
            self.camera_debug_page,
            size=(1100, 700),
        )
        self._refresh_debug_role_status()
        self._refresh_debug_camera_list()

    def open_io_debug_dialog(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("io.debug", "IO调试"):
            return
        if self.lite_mode or not hasattr(self, "io_debug_page"):
            QtWidgets.QMessageBox.information(self, "LC System Lite", "轻量版未加载 DI/DO 调试模块。")
            return
        self._show_tool_dialog(
            "io_debug",
            self.io_debug_page,
            size=(900, 480),
        )
        self._apply_runtime_io_debug_state()

    def open_template_editor_dialog(self) -> None:
        self._open_shape_template_page()

    def open_ncc_match_dialog(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission(
            "template.edit_roi", tr("action.ncc_tool")
        ):
            return
        try:
            dialog = self._ncc_workbench_dialog
            if dialog is not None and dialog.isVisible():
                dialog.raise_()
                dialog.activateWindow()
                return

            from ui.debug import NccMatchWorkbenchDialog

            dialog = NccMatchWorkbenchDialog(
                product_name=self.session.current_product,
                product_dir=self.session.product_dir,
                camera_role=self.current_camera_role(),
                initial_image_path=str(self.canvas.image_path() or ""),
                parent=self.window(),
            )
            dialog.setModal(False)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            if hasattr(dialog, "modelSaved"):
                dialog.modelSaved.connect(self._on_ncc_model_saved)
            dialog.destroyed.connect(lambda *_: setattr(self, "_ncc_workbench_dialog", None))
            self._ncc_workbench_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as exc:
            detail = traceback.format_exc()
            self.lbl_status.setText(f"Status: failed to open NCC tool: {exc}")
            QtWidgets.QMessageBox.critical(self, tr("action.ncc_tool"), f"{exc}\n\n{detail}")

    def _on_ncc_model_saved(self, model_path: str) -> None:
        role = self.current_camera_role()
        self._set_loc_method_for_role(role, "ncc")
        self._clear_training_roi_review_state(role)
        self._save_session()
        self._reload_inspection_items()
        self._apply_current_role_recipe_state()
        self.roiGeometryChanged.emit()
        self.lbl_status.setText(
            f"Status: NCC model saved for {role}: {os.path.basename(str(model_path or 'model.json'))}"
        )

    def runtime_controller(self):
        parent = self.parent()
        while parent is not None:
            runtime_ctrl = getattr(parent, "runtime_ctrl", None)
            if runtime_ctrl is not None:
                return runtime_ctrl
            parent = parent.parent() if hasattr(parent, "parent") else None
        return None

    def set_runtime_io_state(self, ready: bool, detail: str, controller: object = None) -> None:
        self._runtime_io_ready = bool(ready)
        self._runtime_io_status_detail = str(detail or "")
        self._runtime_io_controller = controller if bool(ready) else None
        self._apply_runtime_io_debug_state()

    def _apply_runtime_io_debug_state(self) -> None:
        if self.lite_mode or not hasattr(self, "btn_debug_open_io"):
            return
        has_permission = getattr(self.window(), "_has_permission", None)
        if callable(has_permission) and not has_permission("io.debug"):
            self.btn_debug_open_io.setEnabled(False)
            self.btn_debug_close_io.setEnabled(False)
            self.btn_debug_refresh_io.setEnabled(False)
            return
        if self._runtime_io_ready and self._runtime_io_controller is not None:
            if (
                self._debug_io_controller is not None
                and self._debug_io_controller is not self._runtime_io_controller
                and not self._debug_io_uses_runtime_controller
            ):
                self._close_debug_io(silent=True)
            self._debug_io_controller = self._runtime_io_controller
            self._debug_io_uses_runtime_controller = True
            self._debug_io_timer.start()
            self.btn_debug_open_io.setEnabled(False)
            self.btn_debug_close_io.setEnabled(False)
            self.btn_debug_refresh_io.setEnabled(True)
            self.lbl_debug_io_mapping_summary.setToolTip(self._runtime_io_status_detail)
            self._refresh_debug_io_snapshot()
            return

        if self._debug_io_uses_runtime_controller:
            self._debug_io_timer.stop()
            self._debug_io_controller = None
            self._debug_io_uses_runtime_controller = False
            self.lbl_debug_di_snapshot.setText(tr("debug.di_disconnected"))
            self.lbl_debug_do_snapshot.setText(tr("debug.do_disconnected"))
            self.lbl_debug_io_mapping_summary.setText(tr("debug.mapping_not_loaded"))

        self.btn_debug_open_io.setEnabled(True)
        self.btn_debug_close_io.setEnabled(self._debug_io_controller is not None and not self._debug_io_uses_runtime_controller)
        self.btn_debug_refresh_io.setEnabled(self._debug_io_controller is not None)
        self.lbl_debug_io_mapping_summary.setToolTip(self._runtime_io_status_detail)

    def open_template_match_dialog(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_roi", "模板区域设置"):
            return
        self._show_tool_dialog(
            "template_match",
            self.template_match_box,
            size=(880, 170),
        )

    def open_margin_validation_tool(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_params", "模板参数分析"):
            return
        self._run_margin_validation()

    def open_embedding_analysis_tool(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_params", "模板参数分析"):
            return
        self._open_embedding_analysis_dialog()

    def open_baseline_debug_tool(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_params", "模板参数分析"):
            return
        self._run_traditional_baseline_debug()

    def current_product_name(self) -> str:
        return self.product_session_controller.current_product_name()

    def refresh_product_selector(self) -> None:
        self.product_session_controller.refresh_product_selector()

    def _sync_camera_settings_store_path(self) -> None:
        self.product_session_controller.sync_camera_settings_store_path()

    def connected_debug_camera_serial(self) -> str:
        device = self._debug_camera_device()
        if device is not None:
            return str(getattr(device, "serial_number", "") or "").strip()
        return ""

    def release_debug_camera_for_runtime(self) -> str:
        serial = self.connected_debug_camera_serial()
        if serial:
            self._disconnect_debug_camera()
        return serial

    def inspection_item_rows(
        self,
        *,
        status_kind: str = "pending",
        status_text: str = "",
    ) -> List[Dict[str, object]]:
        effective_status_text = status_text or tr("runtime.untested")
        return [
            {
                "item_id": item.item_id,
                "display_name": item.display_name,
                "camera_id": item.camera_id,
                "roi_label": item.roi_label,
                "algorithm_code": item.algorithm_code,
                "algorithm_type": item.algorithm_type,
                "params": dict(item.params or {}),
                "enabled": bool(item.enabled),
                "status_kind": status_kind if item.enabled else "disabled",
                "status_text": effective_status_text if item.enabled else tr("debug.status.disabled"),
            }
            for item in self.inspection_items
        ]

    def _roi_status_for_path(self, path: str, label_name: str) -> str:
        if not path:
            return ""
        label = str(label_name or "").strip()
        if not is_roi_label(label):
            return ""
        return str(self._roi_results_by_image.get(path, {}).get(label, "") or "").strip().lower()

    def _record_roi_result(self, path: str, label_name: str, status: object) -> None:
        self.test_execution_controller.record_roi_result(path, label_name, status)

    def load_embedding_model(self, algorithm: str, model_key: Optional[str] = None) -> None:
        # Load the embedding model for the given algorithm and update lbl_status.
        _, msg = self.algo.load_model_for_algorithm(
            algorithm,
            self.session.product_dir,
            model_key=model_key or "",
        )
        self.lbl_status.setText(msg)

    def _export_current_backbone_onnx(self) -> None:
        if not self.lite_mode:
            return
        algorithm = self.current_algorithm()
        if not self._is_embedding_algorithm(algorithm):
            QtWidgets.QMessageBox.information(self, tr("common.info"), "请选择学习工具后再导出 ONNX。")
            return
        backbone = self.algo.resolve_learning_algorithm(algorithm)
        display_name = self.algo.algorithm_display_name(backbone) or backbone
        button = getattr(self, "btn_export_onnx", None)
        if button is not None:
            button.setEnabled(False)
        self.lbl_training_validation.setText(f"Status: exporting ONNX {display_name}...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            info = dict(qr_core.export_backbone_onnx(backbone, device="cpu") or {})
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导出ONNX失败", f"{exc}\n\n{traceback.format_exc()}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if button is not None:
                button.setEnabled(True)

        opsets = ", ".join(
            f"{domain}:{version}"
            for domain, version in list(info.get("opsets", []) or [])
        ) or "-"
        input_shape = info.get("input_shape", []) or []
        input_text = "x".join(str(value) for value in input_shape) if input_shape else "-"
        onnx_path = str(info.get("onnx_path", "") or "")
        runtime_path = str(info.get("runtime_path", "") or "")
        self.lbl_training_validation.setText(f"Status: ONNX exported {display_name} opset={opsets} input={input_text}")
        QtWidgets.QMessageBox.information(
            self,
            "导出ONNX完成",
            f"ONNX: {onnx_path}\n"
            f"ORT: {runtime_path}\n"
            f"opset: {opsets}\n"
            f"input: {input_text}",
        )

    def predict_image(
        self,
        path: str,
        *,
        feat_net=None,
        labels_override: Optional[List[str]] = None,
        algorithm_override: Optional[str] = None,
        model_key_override: Optional[str] = None,
        params_override: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        # Used by MainWindow runtime callback.
        return self._predict_image(
            path,
            feat_net=feat_net,
            labels_override=labels_override,
            algorithm_override=algorithm_override,
            model_key_override=model_key_override,
            params_override=params_override,
        )

    def load_session(self) -> None:
        self.product_session_controller.load_session()

    def _run_deferred_initial_session_load(self) -> None:
        self.product_session_controller.run_deferred_initial_session_load()

    def apply_product_switch(self, name: str) -> None:
        self.product_session_controller.apply_product_switch(name)

    def reset_for_clear(self) -> None:
        self.product_session_controller.reset_for_clear()

    # ------------------------------------------------------------------
    # UI 鏋勫缓
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        def _si(sp):
            return self.style().standardIcon(sp)
        SP = QtWidgets.QStyle.StandardPixmap
        _DARK_BG = "#2d2d2d"
        _PANEL_BG = "#363636"
        _HEADER_BG = "#3a3a3a"
        _TEXT_LIGHT = "#e0e0e0"
        _TEXT_DIM = "#888888"
        _compact_btn = (
            "QPushButton{background:#444444;color:#d0d0d0;border:1px solid #5a5a5a;"
            "padding:4px 8px;border-radius:3px;font-size:12px;}"
            "QPushButton:hover{background:#505050;}"
        )
        _input_style = (
            "QComboBox,QDoubleSpinBox,QSpinBox{"
            "background:#404040;color:#e0e0e0;border:1px solid #5a5a5a;padding:2px 4px;border-radius:3px;font-size:12px;}"
            "QComboBox:disabled,QDoubleSpinBox:disabled,QSpinBox:disabled{"
            "background:#353535;color:#8a8a8a;border:1px solid #4a4a4a;}"
            "QSpinBox::up-button:disabled,QSpinBox::down-button:disabled{"
            "background:#353535;border-left:1px solid #444444;}"
        )
        _section_style = (
            f"background:#404040;color:{_TEXT_LIGHT};font-size:12px;font-weight:bold;"
            "border-bottom:1px solid #505050;padding:6px 10px;"
        )

        self.setStyleSheet(f"background:{_DARK_BG};color:{_TEXT_LIGHT};")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QtWidgets.QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(f"background:{_HEADER_BG};border-bottom:1px solid #505050;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 16, 0)
        header_layout.setSpacing(4)

        self.lbl_product_caption = QtWidgets.QLabel(tr("debug.product"))
        header_layout.addWidget(self.lbl_product_caption)
        self.cmb_product = QtWidgets.QComboBox()
        self.cmb_product.setFixedWidth(180)
        self.cmb_product.addItems(self.session.product_names)
        self.cmb_product.setCurrentText(self.session.current_product)
        self.cmb_product.currentTextChanged.connect(self._on_product_changed)
        self.cmb_product.setStyleSheet(_input_style)
        header_layout.addWidget(self.cmb_product)

        self.btn_new_product = QtWidgets.QPushButton(_si(SP.SP_FileDialogNewFolder), tr("debug.new"))
        self.btn_new_product.setFixedWidth(60)
        self.btn_new_product.setStyleSheet(_compact_btn)
        self.btn_new_product.clicked.connect(self._new_product)
        header_layout.addWidget(self.btn_new_product)

        self.btn_copy_product = QtWidgets.QPushButton(_si(SP.SP_FileDialogDetailedView), tr("debug.copy_product"))
        self.btn_copy_product.setFixedWidth(76)
        self.btn_copy_product.setStyleSheet(_compact_btn)
        self.btn_copy_product.clicked.connect(self._copy_product)
        header_layout.addWidget(self.btn_copy_product)

        self.btn_delete_product = QtWidgets.QPushButton(_si(SP.SP_DialogDiscardButton), tr("debug.delete_product"))
        self.btn_delete_product.setFixedWidth(60)
        self.btn_delete_product.setStyleSheet(_compact_btn)
        self.btn_delete_product.clicked.connect(self._request_delete_product)
        header_layout.addWidget(self.btn_delete_product)

        header_layout.addSpacing(10)
        self.lbl_current_camera_caption = QtWidgets.QLabel(tr("debug.current_camera"))
        header_layout.addWidget(self.lbl_current_camera_caption)
        self.cmb_current_camera_role = QtWidgets.QComboBox()
        self.cmb_current_camera_role.setFixedWidth(84)
        for role in CAMERA_ROLES:
            self.cmb_current_camera_role.addItem(role, role)
        self.cmb_current_camera_role.setStyleSheet(_input_style)
        self.cmb_current_camera_role.currentTextChanged.connect(self._on_current_camera_role_changed)
        header_layout.addWidget(self.cmb_current_camera_role)

        header_layout.addStretch(1)

        self.lbl_status = QtWidgets.QLabel(tr("debug.status_untrained"))
        self.lbl_status.setStyleSheet(f"color:{_TEXT_DIM};font-size:13px;")
        self.lbl_status.hide()
        header_layout.addWidget(self.lbl_status)

        root.addWidget(header)

        # Main canvas and side panels
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left side: canvas and result table
        canvas_frame = QtWidgets.QWidget()
        canvas_frame.setStyleSheet(f"background:{_DARK_BG};")
        canvas_vbox = QtWidgets.QVBoxLayout(canvas_frame)
        canvas_vbox.setContentsMargins(2, 2, 2, 2)
        canvas_vbox.setSpacing(2)

        self.canvas = RoiCanvas()
        self.canvas.setMinimumSize(480, 360)
        self.canvas.shapesChanged.connect(self._on_shapes_changed)
        canvas_vbox.addWidget(self.canvas, 3)

        self.table = QtWidgets.QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(
            [
                "\u6587\u4ef6", "GT", "Pred", "diff", "sim_ok", "sim_ng", "value", "threshold",
                "match_ms", "infer_ms", "total_ms", "ROI\u6765\u6e90",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_table_click)
        self.table.setStyleSheet(
            "QTableWidget{background:#333333;color:#d0d0d0;gridline-color:#404040;border:1px solid #404040;}"
            "QTableWidget::item:selected{background:#6ec0ff;color:#1a1a1a;}"
            "QHeaderView::section{background:#3a3a3a;color:#d0d0d0;border:1px solid #404040;padding:4px;}"
        )
        self.table.setMaximumHeight(180)
        canvas_vbox.addWidget(self.table, 1)

        body.addWidget(canvas_frame, 3)

        # 鍙充晶闈㈡澘
        right_panel = QtWidgets.QFrame()
        self._main_right_panel = right_panel
        right_panel.setStyleSheet(f"background:{_PANEL_BG};border-left:1px solid #505050;")
        right_vbox = QtWidgets.QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        build_sample_panel(
            self,
            right_vbox,
            styles={
                "text_light": _TEXT_LIGHT,
                "text_dim": _TEXT_DIM,
                "section_style": _section_style,
                "compact_btn": _compact_btn,
            },
        )

        # --- 绠楁硶鍙傛暟 ---
        self.btn_toggle_algo = QtWidgets.QToolButton()
        self.btn_toggle_algo.setText(tr("debug.algorithm_params"))
        self.btn_toggle_algo.setCheckable(True)
        self.btn_toggle_algo.setChecked(True)
        self.btn_toggle_algo.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_toggle_algo.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_toggle_algo.setStyleSheet(
            (
                f"QToolButton{{background:#404040;color:{_TEXT_LIGHT};font-size:12px;"
                f"font-weight:bold;border:none;border-bottom:1px solid #505050;padding:6px 10px;}}"
                "QToolButton:hover{background:#474747;}"
            )
        )
        self.btn_toggle_algo.toggled.connect(self._toggle_algorithm_section)
        right_vbox.addWidget(self.btn_toggle_algo)

        algo_frame = QtWidgets.QWidget()
        algo_form = QtWidgets.QFormLayout(algo_frame)
        algo_form.setContentsMargins(10, 6, 10, 6)
        algo_form.setSpacing(4)
        algo_form.setHorizontalSpacing(6)
        algo_form.setLabelAlignment(QtCore.Qt.AlignRight)
        algo_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.cmb_algorithm = QtWidgets.QComboBox()
        self._populate_algorithm_combo()
        self.cmb_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        self.cmb_algorithm.hide()
        self.cmb_backbone = self.cmb_algorithm
        self.btn_algorithm_picker = QtWidgets.QToolButton()
        self.btn_algorithm_picker.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_algorithm_picker.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_algorithm_picker.setMenu(self._build_algorithm_picker_menu())
        self._algorithm_picker_style_default = (
            "QToolButton{background:#2f2f2f;color:#e0e0e0;border:1px solid #555;"
            "padding:5px 28px 5px 8px;border-radius:3px;font-size:12px;}"
            "QToolButton:hover{background:#3a3a3a;}"
        )
        self._algorithm_picker_style_compact = (
            "QToolButton{background:#2f2f2f;color:#e0e0e0;border:1px solid #555;"
            "padding:4px 22px 4px 6px;border-radius:3px;font-size:11px;}"
            "QToolButton:hover{background:#3a3a3a;}"
        )
        self.btn_algorithm_picker.setStyleSheet(self._algorithm_picker_style_default)
        self.btn_algorithm_picker.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_algorithm_picker.setMinimumWidth(240)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(SUPPORTED_SCORE_MODES)
        self.cmb_mode.currentTextChanged.connect(self._on_runtime_params_changed)
        self.cmb_mode.setStyleSheet(_input_style)
        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setDecimals(4)
        self.spin_margin.setSingleStep(0.005)
        self.spin_margin.setRange(-1.0, 1.0)
        self.spin_margin.setValue(0.02)
        self.spin_margin.valueChanged.connect(self._on_runtime_params_changed)
        self.spin_margin.setStyleSheet(_input_style)
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setRange(1, 50)
        self.spin_topk.setValue(3)
        self.spin_topk.valueChanged.connect(self._on_runtime_params_changed)
        self.spin_topk.setStyleSheet(_input_style)

        _lbl_s = f"color:{_TEXT_DIM};font-size:12px;"
        _lbl_disabled_s = "color:#7a7a7a;font-size:12px;"
        self._algo_param_label_style = _lbl_s
        self._algo_param_label_disabled_style = _lbl_disabled_s
        self.lbl_algo_tool = QtWidgets.QLabel(tr("debug.tool")); self.lbl_algo_tool.setStyleSheet(_lbl_s)
        self.lbl_algo_decision = QtWidgets.QLabel(tr("debug.decision")); self.lbl_algo_decision.setStyleSheet(_lbl_s)
        self.lbl_algo_threshold = QtWidgets.QLabel(tr("debug.threshold")); self.lbl_algo_threshold.setStyleSheet(_lbl_s)
        self.lbl_topk = QtWidgets.QLabel("TopK")
        self.lbl_topk.setStyleSheet(self._algo_param_label_style)
        algo_form.addRow(self.lbl_algo_tool, self.btn_algorithm_picker)
        algo_form.addRow(self.lbl_algo_decision, self.cmb_mode)
        algo_form.addRow(self.lbl_algo_threshold, self.spin_margin)
        algo_form.addRow(self.lbl_topk, self.spin_topk)
        self.algorithm_params_frame = algo_frame
        right_vbox.addWidget(algo_frame)
        self._sync_algorithm_picker()

        build_tool_config_panel(
            self,
            right_vbox,
            styles={
                "text_light": _TEXT_LIGHT,
                "text_dim": _TEXT_DIM,
                "input_style": _input_style,
                "compact_btn": _compact_btn,
                "label_style": _lbl_s,
            },
        )

        build_action_panel(
            self,
            right_vbox,
            styles={
                "text_light": _TEXT_LIGHT,
                "text_dim": _TEXT_DIM,
                "section_style": _section_style,
                "compact_btn": _compact_btn,
            },
            standard_icon=_si,
            standard_pixmap=SP,
        )

        body.addWidget(right_panel, 0)
        root.addLayout(body, 1)

        # Footer
        footer = QtWidgets.QFrame()
        footer.setFixedHeight(28)
        footer.setStyleSheet(f"background:{_HEADER_BG};border-top:1px solid #505050;")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 0, 16, 0)
        footer_layout.setSpacing(20)

        self.lbl_footer_ref = QtWidgets.QLabel(tr("debug.reference_image_not_set"))
        self.lbl_footer_ref.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_ref)
        self.lbl_footer_algo = QtWidgets.QLabel("")
        self.lbl_footer_algo.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_algo)
        footer_layout.addStretch(1)
        self.lbl_footer_product_dir = QtWidgets.QLabel("")
        self.lbl_footer_product_dir.setStyleSheet(f"color:{_TEXT_DIM};font-size:11px;")
        footer_layout.addWidget(self.lbl_footer_product_dir)
        root.addWidget(footer)

        # Hidden controls kept for dialog/API compatibility

        # Hidden ROI annotation toolbar
        roi_bar_w = QtWidgets.QWidget(self)
        roi_bar = QtWidgets.QHBoxLayout(roi_bar_w)
        self._manual_roi_bar = roi_bar
        self.lbl_roi_shape_caption = QtWidgets.QLabel(tr("debug.shape"))
        roi_bar.addWidget(self.lbl_roi_shape_caption)
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(SUPPORTED_SHAPES)
        self.cmb_shape.setCurrentText("rect")
        self.cmb_shape.currentTextChanged.connect(self._on_shape_changed)
        roi_bar.addWidget(self.cmb_shape)
        self.lbl_roi_label_caption = QtWidgets.QLabel(tr("debug.label"))
        roi_bar.addWidget(self.lbl_roi_label_caption)
        self.cmb_label = QtWidgets.QComboBox()
        self.cmb_label.addItems(["roi", "anchor", "anchor_mask"])
        self.cmb_label.setCurrentText("roi")
        self.cmb_label.currentTextChanged.connect(self._on_label_changed)
        roi_bar.addWidget(self.cmb_label)
        self.btn_save = QtWidgets.QPushButton(_si(SP.SP_DialogSaveButton), tr("debug.save_annotation"))
        self.btn_save.clicked.connect(self._save_current_rect)
        self.btn_clear = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), tr("debug.clear_annotation"))
        self.btn_clear.clicked.connect(self._clear_current_rect)
        roi_bar.addWidget(self.btn_save)
        roi_bar.addWidget(self.btn_clear)
        roi_bar_w.hide()

        # 自动 ROI 对话框
        auto_box = QtWidgets.QGroupBox(tr("debug.auto_roi"))
        auto_l = QtWidgets.QGridLayout(auto_box)
        self._auto_roi_layout = auto_l
        auto_l.setHorizontalSpacing(10)
        auto_l.setVerticalSpacing(10)
        auto_l.setColumnStretch(0, 1)
        auto_l.setColumnStretch(1, 1)
        auto_l.setColumnStretch(2, 1)
        self.lbl_ref = QtWidgets.QLabel(tr("debug.reference_not_set"))
        self.btn_set_ref = QtWidgets.QPushButton(_si(SP.SP_ArrowRight), tr("debug.set_as_reference"))
        self.btn_set_ref.clicked.connect(self._set_ref_from_current)
        self.btn_pick_ref = QtWidgets.QPushButton(_si(SP.SP_DirOpenIcon), tr("debug.pick_reference"))
        self.btn_pick_ref.clicked.connect(self._pick_ref_image)
        auto_l.addWidget(self.lbl_ref, 0, 0, 1, 3)
        auto_l.addWidget(self.btn_set_ref, 1, 0)
        auto_l.addWidget(self.btn_pick_ref, 1, 1)
        self.lbl_loc_method = QtWidgets.QLabel(tr("debug.location_method"))
        auto_l.addWidget(self.lbl_loc_method, 2, 0)
        self.cmb_loc = QtWidgets.QComboBox()
        self._populate_loc_combo()
        self.cmb_loc.currentIndexChanged.connect(lambda _idx: self._on_loc_method_changed(self.cmb_loc.currentData()))
        auto_l.addWidget(self.cmb_loc, 2, 1)
        self.chk_only_missing = QtWidgets.QCheckBox(tr("debug.only_missing_roi"))
        self.chk_only_missing.setChecked(True)
        auto_l.addWidget(self.chk_only_missing, 0, 0, 1, 3, QtCore.Qt.AlignmentFlag.AlignRight)
        self.btn_autogen = QtWidgets.QPushButton(_si(SP.SP_FileDialogListView), tr("debug.batch_roi_current"))
        self.btn_autogen.clicked.connect(self._autogen_roi_current_tab)
        self.btn_autogen_all = QtWidgets.QPushButton(_si(SP.SP_FileDialogListView), tr("debug.batch_roi_all"))
        self.btn_autogen_all.clicked.connect(self._autogen_roi_all)
        self.btn_clear_roi_batch = QtWidgets.QPushButton(_si(SP.SP_DialogResetButton), tr("debug.clear_roi_current"))
        self.btn_clear_roi_batch.clicked.connect(self._clear_roi_current_tab)
        auto_l.addWidget(self.btn_autogen, 1, 0)
        auto_l.addWidget(self.btn_autogen_all, 1, 1)
        auto_l.addWidget(self.btn_clear_roi_batch, 1, 2)
        self.template_match_box = auto_box
        self._update_loc_ui()

        if not self.lite_mode:
            build_camera_debug_page(
                self,
                styles={
                    "dark_bg": _DARK_BG,
                    "panel_bg": _PANEL_BG,
                    "header_bg": _HEADER_BG,
                    "text_light": _TEXT_LIGHT,
                    "text_dim": _TEXT_DIM,
                    "input_style": _input_style,
                    "compact_btn": _compact_btn,
                },
                standard_icon=_si,
                standard_pixmap=SP,
            )

            build_io_debug_page(
                self,
                styles={
                    "dark_bg": _DARK_BG,
                    "panel_bg": _PANEL_BG,
                    "text_light": _TEXT_LIGHT,
                    "text_dim": _TEXT_DIM,
                    "compact_btn": _compact_btn,
                },
                standard_icon=_si,
                standard_pixmap=SP,
            )

        self.lbl_template_tool_hint = QtWidgets.QLabel("")
        self.lbl_template_tool_hint.hide()
        self._normalize_stylesheet_font_units()
        self._update_responsive_layout()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._update_responsive_layout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _toggle_tool_config_section(self, checked: bool) -> None:
        frame = getattr(self, "tool_config_scroll", None) or getattr(self, "tool_config_frame", None)
        if frame is not None:
            frame.setVisible(bool(checked))
        toggle = getattr(self, "btn_toggle_tools", None)
        if toggle is not None:
            toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )

    def _toggle_algorithm_section(self, checked: bool) -> None:
        frame = getattr(self, "algorithm_params_frame", None)
        if frame is not None:
            frame.setVisible(bool(checked))
        toggle = getattr(self, "btn_toggle_algo", None)
        if toggle is not None:
            toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )

    @staticmethod
    def _normalize_font_size_units(style_sheet: str) -> str:
        def replace(match: re.Match[str]) -> str:
            px = int(match.group(1))
            pt = max(1, int(round(px * 0.75)))
            return f"font-size:{pt}pt"

        return re.sub(r"font-size\s*:\s*(\d+)px", replace, style_sheet)

    def _normalize_stylesheet_font_units(self) -> None:
        for widget in [self, *self.findChildren(QtWidgets.QWidget)]:
            style_sheet = widget.styleSheet()
            if not style_sheet or "font-size" not in style_sheet or "px" not in style_sheet:
                continue
            normalized = self._normalize_font_size_units(style_sheet)
            if normalized != style_sheet:
                widget.setStyleSheet(normalized)

    def _update_responsive_layout(self) -> None:
        width = max(self.width(), 1)
        compact = width <= 1366
        right_panel = getattr(self, "_main_right_panel", None)
        if right_panel is not None:
            panel_width = 660 if compact else 500
            if width >= 1700:
                panel_width = 540
            right_panel.setFixedWidth(panel_width)
        if hasattr(self, "cmb_product"):
            self.cmb_product.setFixedWidth(160 if compact else 180)
        if hasattr(self, "btn_new_product"):
            self.btn_new_product.setFixedWidth(56 if compact else 60)
        if hasattr(self, "btn_delete_product"):
            self.btn_delete_product.setFixedWidth(56 if compact else 60)
        if hasattr(self, "cmb_current_camera_role"):
            self.cmb_current_camera_role.setFixedWidth(72 if compact else 84)
        if hasattr(self, "btn_algorithm_picker"):
            style = self._algorithm_picker_style_compact if compact else self._algorithm_picker_style_default
            if style and self.btn_algorithm_picker.styleSheet() != style:
                self.btn_algorithm_picker.setStyleSheet(style)


    # ------------------------------------------------------------------
    # 搴曟爮鍚屾
    # ------------------------------------------------------------------

    def _sync_footer(self) -> None:
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "-"
        self.lbl_footer_ref.setText(f"{tr('debug.reference_image')}: {ref_name}")
        algo = self.current_algorithm_display_name() if hasattr(self, "cmb_algorithm") else ""
        self.lbl_footer_algo.setText(f"{tr('debug.algorithm')}: {algo}" if algo else "")
        self.lbl_footer_product_dir.setText(f"{tr('debug.product_dir')}: {self.session.product_dir}")

    # ------------------------------------------------------------------
    # 列表刷新
    # ------------------------------------------------------------------

    def _refresh_lists(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        if hasattr(self, "lbl_images_section"):
            title = tr("debug.image_list")
            if current_role:
                if language_code().lower().startswith("zh"):
                    title = f"{title}（{current_role}）"
                else:
                    title = f"{title} ({current_role})"
            self.lbl_images_section.setText(title)

        def fill(
            listw: QtWidgets.QListWidget,
            files: List[str],
            *,
            sample_kind: str,
        ) -> None:
            current_item = listw.currentItem()
            current_path = None
            if current_item is not None:
                current_path = current_item.data(QtCore.Qt.UserRole) or current_item.toolTip()
            blocker = QtCore.QSignalBlocker(listw)
            listw.clear()
            selected_row = -1
            for index, p in enumerate(files):
                it = QtWidgets.QListWidgetItem(self._sample_item_display_text(p, sample_kind, current_role))
                it.setToolTip(p)
                it.setData(QtCore.Qt.UserRole, p)
                listw.addItem(it)
                if current_path and p == current_path:
                    selected_row = index
            if selected_row >= 0:
                listw.setCurrentRow(selected_row)
            del blocker

        fill(self.ok_list, self._sample_paths_for_kind("train", current_role), sample_kind="train")
        fill(self.ng_list, [], sample_kind="ng")
        fill(self.test_list, self._sample_paths_for_kind("test", current_role), sample_kind="test")
        self._update_sample_panel_widgets()

    def _save_session(self) -> None:
        self.product_session_controller.save_session()

    # ------------------------------------------------------------------
        ref_name = os.path.basename(self.ref_image) if self.ref_image else "Not Set"
    # ------------------------------------------------------------------

    def _current_sample_tab_kind(self) -> str:
        return "train" if self.tabs.currentIndex() == 0 else "test"

    def _sample_paths_for_kind(
        self,
        kind: str,
        camera_role: object = None,
    ) -> List[str]:
        role = _normalize_camera_role(camera_role or _selected_image_list_camera_role(self)) or "cam1"
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind in {"train", "training"}:
            return self._train_sample_paths_for_role(role)
        return _filter_paths_for_camera(self, self.test_files, role)

    def _sample_usage_text(self, path: str) -> str:
        if path in self.train_files:
            return tr("debug.train_samples")
        if path in self.test_files:
            return tr("debug.test_samples")
        return "Uncategorized"

    def _inspection_label_names_for_role(self, camera_role: object = None) -> List[str]:
        role = _normalize_camera_role(camera_role or self.current_camera_role()) or "cam1"
        labels: List[str] = []
        seen: set[str] = set()
        for item in self.inspection_items:
            item_role = _normalize_camera_role(getattr(item, "camera_id", "")) or "cam1"
            if item_role != role:
                continue
            label = str(getattr(item, "roi_label", "") or "").strip()
            if not is_roi_label(label) or label in seen:
                continue
            labels.append(label)
            seen.add(label)
        if not labels:
            labels_getter = getattr(self, "_loc_output_labels", None)
            if callable(labels_getter):
                for label in labels_getter(role):
                    text = str(label or "").strip()
                    if not is_roi_label(text) or text in seen:
                        continue
                    labels.append(text)
                    seen.add(text)
        return labels

    def _sample_annotation_store_path(self) -> str:
        return self.roi_annotations.store_path()

    def _sample_annotation_path_key(self, path: object) -> str:
        return self.roi_annotations.path_key(path)

    def _sample_roi_annotation_key(self, camera_role: object, label_name: object) -> str:
        return self.roi_annotations.roi_key(camera_role, label_name)

    def _load_sample_roi_annotations(self) -> None:
        self.roi_annotations.load()

    def _save_sample_roi_annotations(self) -> None:
        self.roi_annotations.save()

    def _delete_sample_annotation_file(self) -> None:
        self.roi_annotations.delete_store()

    def _path_has_roi_geometry(self, path: str, label_name: str) -> bool:
        return self.roi_annotations.has_geometry(path, label_name)

    def _path_has_roi_label(self, path: str, label_name: str) -> bool:
        return self._path_has_roi_geometry(path, label_name)

    def _sample_roi_status_for_path(
        self,
        path: str,
        camera_role: object,
        label_name: str,
    ) -> str:
        return self.roi_annotations.status_for_path(path, camera_role, label_name)

    def _set_sample_roi_status_for_path(
        self,
        path: str,
        camera_role: object,
        label_name: str,
        status: object,
    ) -> None:
        self.roi_annotations.set_status_for_path(path, camera_role, label_name, status)

    def _mark_sample_path_all_ok(self, path: str, camera_role: object = None) -> None:
        self.roi_annotations.mark_all_ok(path, camera_role)

    def _mark_sample_path_all_ng(self, path: str, camera_role: object = None) -> None:
        self.roi_annotations.mark_all_ng(path, camera_role)

    def _clear_sample_path_annotations(self, path: str, camera_role: object = None) -> None:
        self.roi_annotations.clear_path(path, camera_role)

    def _sample_annotation_counts_for_roi(
        self,
        roi_label: str,
        camera_role: object = None,
        *,
        paths: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        return self.roi_annotations.counts_for_roi(roi_label, camera_role, paths=paths)

    def _sample_annotation_progress_for_path(
        self,
        path: str,
        camera_role: object = None,
    ) -> Tuple[int, int]:
        return self.roi_annotations.progress_for_path(path, camera_role)

    def _sample_annotation_state_for_path(
        self,
        path: str,
        camera_role: object = None,
    ) -> str:
        return self.roi_annotations.state_for_path(path, camera_role)

    def _sample_item_display_text(
        self,
        path: str,
        sample_kind: str,
        camera_role: object = None,
    ) -> str:
        status = self._sample_annotation_state_for_path(path, camera_role)
        name = os.path.basename(path)
        if str(sample_kind or "").strip().lower() in {"train", "training"}:
            return f"{name}    [{status}]"
        return f"{name}    [{status}]"

    def _current_image_sample_state_text(self) -> str:
        path = self.canvas.image_path()
        if not path:
            return tr("debug.current_image_state_none")
        return tr(
            "debug.current_image_state",
            usage=self._sample_usage_text(path),
            state=self._sample_annotation_state_for_path(path, self.current_camera_role()),
        )

    def _current_tool_sample_stats_text(self) -> str:
        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            return tr("debug.current_tool_stats_select")
        camera_role = _normalize_camera_role(getattr(inspection_item, "camera_id", "")) or self.current_camera_role()
        roi_label = str(getattr(inspection_item, "roi_label", "") or "").strip() or "roi"
        ok_count, ng_count, unset_count = self._sample_annotation_counts_for_roi(roi_label, camera_role)
        return tr("debug.current_tool_samples", roi=roi_label, ok=ok_count, ng=ng_count, unset=unset_count)

    def _training_validation_text(self) -> str:
        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            return tr("debug.training_validation_select")
        camera_role = _normalize_camera_role(getattr(inspection_item, "camera_id", "")) or self.current_camera_role()
        roi_label = str(getattr(inspection_item, "roi_label", "") or "").strip() or "roi"
        ok_files, ng_files, candidate_paths = self._training_sample_groups_for_role(camera_role, roi_label=roi_label)
        if not candidate_paths:
            return tr("debug.training_validation_no_samples", role=camera_role)
        _ok_count, _ng_count, missing_count = self._sample_annotation_counts_for_roi(roi_label, camera_role, paths=candidate_paths)
        if missing_count > 0:
            return tr("debug.training_validation_missing_annotations", roi=roi_label, count=missing_count)
        if not ok_files or not ng_files:
            missing_groups: List[str] = []
            if not ok_files:
                missing_groups.append("OK")
            if not ng_files:
                missing_groups.append("NG")
            return tr("debug.training_validation_missing_groups", roi=roi_label, groups="/".join(missing_groups))
        return tr("debug.training_validation_ready", roi=roi_label)

    def _update_sample_panel_widgets(self) -> None:
        current_role = _selected_image_list_camera_role(self)
        train_count = len(self._sample_paths_for_kind("train", current_role))
        test_count = len(self._sample_paths_for_kind("test", current_role))
        if hasattr(self, "tabs"):
            self.tabs.setTabText(0, f"{tr('debug.train_samples')} ({train_count})")
            self.tabs.setTabText(1, f"{tr('debug.test_samples')} ({test_count})")
        current_image_label = getattr(self, "lbl_current_image_sample_state", None)
        if current_image_label is not None:
            current_image_label.setText(f"  {self._current_image_sample_state_text()}")
        tool_stats_label = getattr(self, "lbl_current_tool_sample_stats", None)
        if tool_stats_label is not None:
            tool_stats_label.setText(f"  {self._current_tool_sample_stats_text()}")
        validation_label = getattr(self, "lbl_training_validation", None)
        if validation_label is not None and not getattr(self, "_training_in_progress", False):
            validation_label.setText(self._training_validation_text())

        selected_path = self._current_selected_path()
        current_tab_kind = self._current_sample_tab_kind()
        has_permission = getattr(self.window(), "_has_permission", None)
        can_manage_samples = True
        if callable(has_permission):
            can_manage_samples = bool(has_permission("sample.manage"))
        for attr_name in ("btn_import_train", "btn_sample_annotation", "btn_add_test"):
            button = getattr(self, attr_name, None)
            if button is not None:
                button.setEnabled(can_manage_samples)
        for attr_name, enabled in (
            ("btn_train_to_test", current_tab_kind == "train" and bool(selected_path)),
            ("btn_del_ok", current_tab_kind == "train" and bool(selected_path)),
            ("btn_test_to_train", current_tab_kind == "test" and bool(selected_path)),
            ("btn_del_test", current_tab_kind == "test" and bool(selected_path)),
            ("btn_sample_annotation_test", current_tab_kind == "test" and test_count > 0),
        ):
            button = getattr(self, attr_name, None)
            if button is not None:
                button.setEnabled(enabled and can_manage_samples)

    def _select_path_in_current_tab(self, path: str) -> None:
        self.sample_list_controller.select_path_in_current_tab(path)

    def _move_selected_sample_to(self, target_kind: str) -> None:
        self.sample_list_controller.move_selected_sample_to(target_kind)

    def _open_sample_annotation_dialog(self) -> None:
        require_permission = getattr(self.window(), "_require_permission", None)
        if callable(require_permission) and not require_permission("sample.manage", "样本标注"):
            return
        dialog = getattr(self, "_sample_annotation_preview_dialog", None)
        if dialog is None:
            dialog = _SampleAnnotationPreviewDialog(self, self)
            self._sample_annotation_preview_dialog = dialog
            dialog.finished.connect(lambda *_: setattr(self, "_sample_annotation_preview_dialog", None))
        dialog.sync_camera_roles(self.configured_camera_roles())
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _current_selected_path(self) -> Optional[str]:
        return self.sample_list_controller.current_selected_path()

    def _show_selected_image_path(self, path: Optional[str]) -> None:
        self.sample_list_controller.show_selected_image_path(path)

    def _clear_image_view_for_role_switch(self) -> None:
        for listw in (self.ok_list, self.ng_list, self.test_list):
            blocker = QtCore.QSignalBlocker(listw)
            listw.clearSelection()
            listw.setCurrentItem(None)
            del blocker
        self.canvas.clear_image()
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        self._current_result_rows = []
        self.lbl_status.setText(tr("debug.status_switched_camera_select", role=self.current_camera_role()))
        self._update_sample_panel_widgets()

    def _clear_selected_inspection_item(self) -> None:
        table = getattr(self, "inspection_items_table", None)
        if table is None:
            return
        table.clearSelection()
        table.setCurrentItem(None)

    def _on_select_ok(self) -> None:
        self._show_selected_image_path(self._current_selected_path())

    def _on_select_ng(self) -> None:
        self._show_selected_image_path(self._current_selected_path())

    def _on_select_test(self) -> None:
        self._show_selected_image_path(self._current_selected_path())

    def _add_images_to(self, kind: str) -> None:
        self.sample_list_controller.add_images_to(kind)

    def _remove_selected_from(self, kind: str) -> None:
        self.sample_list_controller.remove_selected_from(kind)

    def _clear_current_test_list(self) -> None:
        self.sample_list_controller.clear_current_test_list()

    def _on_tab_changed(self, index: int) -> None:
        if index == 0:
            listw = self.ok_list
            files = self._sample_paths_for_kind("train", _selected_image_list_camera_role(self))
        else:
            listw = self.test_list
            files = self._sample_paths_for_kind("test", _selected_image_list_camera_role(self))

        self._update_sample_panel_widgets()
        if not files:
            return

        row = listw.currentRow()
        if row < 0 or row >= len(files):
            blocker = QtCore.QSignalBlocker(listw)
            listw.setCurrentRow(0)
            del blocker
            row = 0
        self._show_selected_image_path(files[row])

    # ------------------------------------------------------------------
    # 产品管理
    # ------------------------------------------------------------------

    def _on_product_changed(self, product_name: str) -> None:
        self.product_session_controller.on_product_changed(product_name)

    def _new_product(self) -> None:
        self.product_session_controller.new_product()

    def _copy_product(self) -> None:
        self.product_session_controller.copy_product()

    def _request_delete_product(self) -> None:
        self.product_session_controller.request_delete_product()

    def _clear_session(self) -> None:
        self.product_session_controller.clear_session()

    # ------------------------------------------------------------------
    # Auto ROI
    # ------------------------------------------------------------------


    def _on_shape_changed(self) -> None:
        if self._current_label() == "anchor_mask" and self.cmb_shape.currentText() != "polygon":
            self.cmb_shape.setCurrentText("polygon")
            return
        self.canvas.draw_shape = self.cmb_shape.currentText()
        self.canvas._poly_pts = []
        self.canvas.update()
        self._on_shapes_changed()

    def _on_label_changed(self) -> None:
        label = self._current_label()
        if label == "anchor_mask":
            self.cmb_shape.setCurrentText("polygon")
            self.cmb_shape.setEnabled(False)
        else:
            self.cmb_shape.setEnabled(True)
        self._update_save_label_text()
        p = self.canvas.image_path()
        if p:
            self._load_shape_for_label(p, label)

    def _clear_current_rect(self) -> None:
        top_level = self.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_roi", "清空ROI"):
            return
        ret = QtWidgets.QMessageBox.question(
            self,
            tr("debug.clear_annotation_title"),
            tr("debug.clear_annotation_confirm"),
        )
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        p = self.canvas.image_path()
        label_name = self._current_label()
        if p is not None:
            try:
                deleted = labelme_io.delete_labelme_shape(p, label_name=label_name)
            except Exception:
                deleted = False
            if deleted:
                audit_event = getattr(top_level, "_audit_event", None)
                if callable(audit_event):
                    audit_event(
                        module="模板ROI",
                        action="清空ROI",
                        target=f"{os.path.basename(p)}:{label_name}",
                    )
                self._load_canvas_image(p)
                return
        self.canvas.update()
        self._on_shapes_changed()


    def _save_current_rect(self) -> None:
        p = self.canvas.image_path()
        if p is None:
            return
        top_level = self.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_roi", "保存ROI"):
            return
        st = self.canvas.roi
        label_name = self._current_label()

        if label_name == "anchor_mask" and st.shape_type != "polygon":
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.anchor_polygon_only"))
            return

        if st.shape_type == "rect":
            if st.xywh is None:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.draw_rect_first"))
                return
            jpath = labelme_io.upsert_labelme_rect(p, st.xywh, label_name=label_name)
        else:
            if not st.points or len(st.points) < 3:
                QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.polygon_min_points"))
                return
            jpath = labelme_io.upsert_labelme_polygon(p, st.points, label_name=label_name)

        QtWidgets.QMessageBox.information(
            self,
            tr("debug.annotation_saved_title"),
            tr("debug.annotation_saved_message", path=jpath, label=label_name, shape=st.shape_type),
        )
        audit_event = getattr(top_level, "_audit_event", None)
        if callable(audit_event):
            audit_event(
                module="模板ROI",
                action="保存ROI",
                target=f"{os.path.basename(p)}:{label_name}",
                after_value=str(st.xywh if st.shape_type == "rect" else st.points),
            )
        self._load_canvas_image(p)

    # ------------------------------------------------------------------
    # Training / calibration
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Auto ROI
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 绠楁硶鍙傛暟

    def _is_embedding_algorithm(self, algorithm: Optional[str] = None) -> bool:
        normalized = str(algorithm or self.current_algorithm() or "").strip()
        if not normalized:
            return False
        return self.algo.is_embedding_algorithm(normalized)

    def _embedding_model_path(self, algorithm: str) -> str:
        return self.algo.embedding_model_path(algorithm, self.session.product_dir)

    def _save_runtime_params(self) -> None:
        self.algo.save_params(self.session.product_params_path)

    def _apply_runtime_params_to_ui(self) -> None:
        self._updating_runtime_params = True
        try:
            algorithm = (
                self.algo.product_params.algorithm
                if self.algo.product_params.algorithm in SUPPORTED_ALGORITHMS
                else ""
            )
            score_mode = (
                self.algo.product_params.score_mode
                if self.algo.product_params.score_mode in SUPPORTED_SCORE_MODES
                else SUPPORTED_SCORE_MODES[0]
            )
            self._set_current_algorithm(algorithm)
            self.cmb_mode.setCurrentText(score_mode)
            self.spin_margin.setValue(float(self.algo.product_params.margin))
            self.spin_topk.setValue(max(1, int(self.algo.product_params.topk)))
        finally:
            self._updating_runtime_params = False
        self._update_runtime_widgets()
        self._update_learning_backbone_hint()

    def _update_runtime_widgets(self) -> None:
        algorithm_selected = bool(self.current_algorithm())
        embedding = algorithm_selected and self._is_embedding_algorithm()
        topk_enabled = embedding and self.cmb_mode.currentText() == "topk"
        inspection_items = list(getattr(self, "inspection_items", []) or [])
        current_role = self.current_camera_role()
        has_enabled_items = any(
            getattr(item, "enabled", False)
            and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
            for item in inspection_items
        )
        selected_item_fn = getattr(self, "_selected_inspection_item", None)
        selected_item = selected_item_fn() if callable(selected_item_fn) else None
        selected_tool_enabled = bool(
            algorithm_selected
            and selected_item is not None
            and selected_item.enabled
        )
        top_level = self.window()
        has_permission = getattr(top_level, "_has_permission", None)
        can_train = True
        can_edit_template_params = True
        if callable(has_permission):
            can_train = bool(has_permission("model.train"))
            can_edit_template_params = bool(has_permission("template.edit_params"))
        self.cmb_mode.setEnabled(embedding and can_edit_template_params)
        self.spin_margin.setEnabled(embedding and can_edit_template_params)
        self.spin_topk.setEnabled(topk_enabled and can_edit_template_params)
        topk_label = getattr(self, "lbl_topk", None)
        if topk_label is not None:
            enabled_style = getattr(self, "_algo_param_label_style", "")
            disabled_style = getattr(self, "_algo_param_label_disabled_style", enabled_style)
            topk_label.setEnabled(topk_enabled)
            topk_label.setStyleSheet(enabled_style if topk_enabled else disabled_style)
        self.btn_train.setEnabled(has_enabled_items and can_train)
        train_current_button = getattr(self, "btn_train_current", None)
        if train_current_button is not None:
            train_current_button.setEnabled(selected_tool_enabled and can_train)
        self.btn_test.setEnabled(algorithm_selected)
        margin_button = getattr(self, "btn_validate_margin", None)
        if margin_button is not None:
            margin_button.setEnabled(embedding and can_edit_template_params)
        embedding_button = getattr(self, "btn_embedding_analysis", None)
        if embedding_button is not None:
            embedding_button.setEnabled(embedding and can_edit_template_params)
        if getattr(self, "_training_in_progress", False):
            self._set_training_running(True)
        self._sync_training_action_buttons()
        self._update_sample_panel_widgets()

    def _on_runtime_params_changed(self, *args) -> None:
        if self._updating_runtime_params:
            return
        top_level = self.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_params", "修改模板参数"):
            self._apply_runtime_params_to_ui()
            return
        before = {
            "score_mode": self.algo.product_params.score_mode,
            "margin": self.algo.product_params.margin,
            "topk": self.algo.product_params.topk,
        }
        self.algo.product_params.algorithm = self.current_algorithm()
        self.algo.product_params.score_mode = self.cmb_mode.currentText()
        self.algo.product_params.margin = float(self.spin_margin.value())
        self.algo.product_params.topk = int(self.spin_topk.value())
        if self._is_embedding_algorithm():
            self.algo.apply_params_to_model()
        self._save_runtime_params()
        self._update_runtime_widgets()
        self._update_learning_backbone_hint()
        audit_event = getattr(top_level, "_audit_event", None)
        if callable(audit_event):
            audit_event(
                module="模板参数",
                action="修改判定参数",
                before_value=str(before),
                after_value=str({
                    "score_mode": self.algo.product_params.score_mode,
                    "margin": self.algo.product_params.margin,
                    "topk": self.algo.product_params.topk,
                }),
            )

    def _on_algorithm_changed(self, *args) -> None:
        algorithm = self.current_algorithm()
        if self._updating_runtime_params:
            return
        top_level = self.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("template.edit_params", "修改模板参数"):
            self._apply_runtime_params_to_ui()
            return
        selected_item = self._selected_inspection_item()
        before = str(getattr(selected_item, "algorithm_code", "") if selected_item is not None else self.algo.product_params.algorithm)
        if algorithm in SUPPORTED_EMBEDDING_ALGORITHMS:
            self.algo.set_learning_backbone(algorithm)
        if not algorithm:
            self.algo.product_params.algorithm = ""
            self.algo.model = None
            self._save_runtime_params()
            self._update_runtime_widgets()
            self.lbl_status.setText(tr("debug.training_validation_select"))
            return
        if selected_item is not None:
            selected_item.algorithm_code = (
                "shared_backbone_register"
                if algorithm in SUPPORTED_EMBEDDING_ALGORITHMS
                else algorithm
            )
            self._persist_inspection_items()
            self._refresh_inspection_items_table()
        self.algo.product_params.algorithm = algorithm
        self._save_runtime_params()
        self._update_runtime_widgets()
        self._update_learning_backbone_hint()
        audit_event = getattr(top_level, "_audit_event", None)
        if callable(audit_event) and before != str(algorithm or ""):
            audit_event(
                module="模板参数",
                action="修改算法",
                target=str(getattr(selected_item, "item_id", "") if selected_item is not None else ""),
                before_value=before,
                after_value=str(algorithm or ""),
            )
        try:
            _, msg = self.algo.load_model_for_algorithm(
                algorithm,
                self.session.product_dir,
                model_key=selected_item.model_key if selected_item is not None else "",
            )
            self.lbl_status.setText(msg)
        except Exception as exc:
            self.algo.model = None
            display_name = self.algo.algorithm_display_name(algorithm) or algorithm
            self.lbl_status.setText(f"Status: failed to load tool {display_name} - {exc}")

    # ------------------------------------------------------------------
    # 璁粌
    # ------------------------------------------------------------------

    def _resolve_training_algorithm(self, inspection_item: InspectionItem) -> str:
        return self.training_controller.resolve_algorithm(inspection_item)

    def _training_camera_roles_in_lists(self, camera_id: object | None = None) -> List[str]:
        if camera_id is None:
            candidate_paths = list(getattr(self, "train_files", []) or [])
            if not candidate_paths:
                candidate_paths = list(getattr(self, "ok_files", []) or []) + list(getattr(self, "ng_files", []) or [])
        else:
            candidate_paths = self._train_sample_paths_for_role(camera_id)
        roles = {
            _camera_role_from_path(path)
            for path in candidate_paths
        }
        roles.discard("")
        return sorted(roles)

    def _warn_mixed_training_camera_samples(self, camera_id: object | None = None) -> bool:
        roles = self._training_camera_roles_in_lists(camera_id)
        if len(roles) < 2:
            return False
        role_text = _normalize_camera_role(camera_id) if camera_id is not None else ""
        suffix = tr("debug.current_role_suffix", role=role_text) if role_text else ""
        QtWidgets.QMessageBox.warning(
            self,
            tr("debug.mixed_training_title"),
            tr("debug.mixed_training_message", suffix=suffix),
        )
        return True

    def _missing_training_roi_paths(self, roi_label: str, candidate_paths: List[str]) -> List[str]:
        return self.training_controller.missing_training_roi_paths(roi_label, candidate_paths)

    @staticmethod
    def _training_item_display_name(inspection_item: InspectionItem) -> str:
        return TrainingController.item_display_name(inspection_item)

    def _build_training_task_for_item(self, inspection_item: InspectionItem) -> dict:
        return self.training_controller.build_task_for_item(inspection_item)

    def _training_payload(self, mode: str, tasks: List[dict], *, selected_item_id: str = "", failures: Optional[List[str]] = None) -> dict:
        return self.training_controller.payload(mode, tasks, selected_item_id=selected_item_id, failures=list(failures or []))

    def _set_training_running(self, running: bool) -> None:
        self.training_controller.set_running(running)

    def _on_training_progress(self, message: str) -> None:
        self.training_controller.on_progress(message)

    def _start_training_worker(self, payload: dict) -> None:
        self.training_controller.start_worker(payload)

    def _forget_training_job(self, thread: QtCore.QThread, worker: TrainingJobWorker) -> None:
        self.training_controller.forget_job(thread, worker)

    def _on_training_finished(self, payload: object) -> None:
        self.training_controller.on_finished(payload)

    def _train_inspection_item(self, inspection_item: InspectionItem) -> TrainResult:
        return self.training_controller.train_inspection_item(inspection_item)

    def _train_all_tools(self) -> None:
        top_level = self.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("model.train", "重新训练"):
            return
        self.training_controller.train_all_tools()

    def _train(self) -> None:
        top_level = self.window()
        require_permission = getattr(top_level, "_require_permission", None)
        if callable(require_permission) and not require_permission("model.train", "重新训练"):
            return
        self.training_controller.train_current()

    # ------------------------------------------------------------------
    # 棰勬祴 / 娴嬭瘯
    # ------------------------------------------------------------------

    def _test_target_inspection_items(self) -> List[InspectionItem]:
        return self.test_execution_controller.target_inspection_items()

    def _execute_test_image(self, path: str) -> Dict[str, object]:
        return self.test_execution_controller.execute_image(path)

    def _run_test(self) -> None:
        self.test_execution_controller.run_current_image()

    def _run_all_test_samples(self) -> None:
        self.test_execution_controller.run_all_test_samples()


    def _show_tool_dialog(
        self,
        key: str,
        widget: QtWidgets.QWidget,
        *,
        size: Tuple[int, int],
    ) -> None:
        dialog = self._tool_dialogs.get(key)
        if dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setModal(False)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addWidget(widget)
            dialog.resize(*size)
            if key == "camera_debug":
                dialog.finished.connect(lambda *_: self._stop_debug_camera_preview())
            self._tool_dialogs[key] = dialog
        dialog.setWindowTitle(self._tool_dialog_title(key))
        if key == "camera_debug":
            # Prevent Enter in parameter editors from triggering dialog buttons.
            for _btn in dialog.findChildren(QtWidgets.QPushButton):
                _btn.setAutoDefault(False)
                _btn.setDefault(False)
        widget.show()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


    def _export_current_results_csv(self) -> None:
        self.test_execution_controller.export_current_results_csv()

    def _on_table_click(self, row: int, _col: int) -> None:
        self.test_execution_controller.on_table_click(row, _col)

    # ------------------------------------------------------------------
    # 分析 / 验证
    # ------------------------------------------------------------------


    def _save_margin_report(
        self, rows: List[Dict[str, object]], summary: Dict[str, object]
    ) -> Tuple[str, str]:
        report_dir = os.path.join(self.session.product_dir, "margin_reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm_code = learning_backbone_storage_code(self.current_algorithm())
        base = f"margin_report_{algorithm_code}_{stamp}"
        json_path = os.path.join(report_dir, base + ".json")
        csv_path = os.path.join(report_dir, base + ".csv")

        payload = {
            "product": self.session.current_product,
            "algorithm": algorithm_code,
            "score_mode": self.cmb_mode.currentText(),
            "topk": int(self.spin_topk.value()),
            "margin": float(self.spin_margin.value()),
            "loc_method": self.loc_method_for_role(self.current_camera_role()),
            "loc_methods": dict(getattr(self, "_loc_methods_by_role", {}) or {}),
            "summary": summary,
            "rows": rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self._write_test_rows_csv(csv_path, rows)
        return json_path, csv_path

    def _run_margin_validation(self) -> None:
        inspection_item = self._selected_inspection_item()
        if inspection_item is not None and inspection_item.enabled:
            algorithm = (
                self.algo.current_learning_backbone()
                if self.algo.is_learning_tool(inspection_item.algorithm_code)
                else self.algo.resolve_tool_algorithm(inspection_item.algorithm_code)
            )
            labels_override = [str(inspection_item.roi_label or "").strip() or "roi"]
            algorithm_override = inspection_item.algorithm_code
            model_key_override = inspection_item.model_key
            validation_ok_files, validation_ng_files, _candidate_paths = self._training_sample_groups_for_role(
                inspection_item.camera_id,
                roi_label=labels_override[0],
            )
        else:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_inspection_tool_first"))
            return
        if not self._is_embedding_algorithm(algorithm):
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.margin_traditional_unsupported"))
            return
        if not self.algo._loaded_embedding_matches(
            algorithm,
            labels=labels_override,
            model_key=model_key_override or "",
        ):
            try:
                self.load_embedding_model(algorithm, model_key=model_key_override)
            except Exception:
                pass
        if self.algo.model is None:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.train_register_first"))
            return
        if not validation_ok_files or not validation_ng_files:
            QtWidgets.QMessageBox.warning(self, tr("common.info"), tr("debug.margin_need_ok_ng"))
            return

        feat_net = self.algo.get_feat_net(
            self.algo.model.backbone,
            getattr(self.algo.model, "device", None),
        )
        rows: List[Dict[str, object]] = []
        try:
            for path in validation_ok_files:
                row = self._predict_image(
                    path,
                    feat_net=feat_net,
                    prefer_canvas_roi=False,
                    labels_override=labels_override,
                    algorithm_override=algorithm_override,
                    model_key_override=model_key_override,
                )
                row["gt"] = "OK"
                rows.append(row)
            for path in validation_ng_files:
                row = self._predict_image(
                    path,
                    feat_net=feat_net,
                    prefer_canvas_roi=False,
                    labels_override=labels_override,
                    algorithm_override=algorithm_override,
                    model_key_override=model_key_override,
                )
                row["gt"] = "NG"
                rows.append(row)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, tr("debug.margin_failed_title"), str(ex))
            return

        self._populate_results_table(rows)
        summary = self._suggest_margin_from_rows(rows)
        json_path, csv_path = self._save_margin_report(rows, summary)

        safe_range = summary.get("safe_range")
        safe_text = ""
        if isinstance(safe_range, tuple):
            safe_text = "\n" + tr("debug.safe_range", low=safe_range[0], high=safe_range[1])

        self.lbl_status.setText(
            "Status: "
            + f"current margin={summary['current_margin']:.4f} acc={summary['current_accuracy']:.4f}  "
            + f"suggested margin={summary['suggested_margin']:.4f} acc={summary['suggested_accuracy']:.4f}"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Margin suggestion",
            f"Current margin: {summary['current_margin']:.4f}\n"
            f"Current accuracy: {summary['current_accuracy']:.4f}\n"
            f"Suggested margin: {summary['suggested_margin']:.4f}\n"
            f"Suggested accuracy: {summary['suggested_accuracy']:.4f}"
            + safe_text
            + f"\n\nSaved reports:\n{json_path}\n{csv_path}",
        )

    def _open_embedding_analysis_dialog(self) -> None:
        try:
            from ui.debug import EmbeddingAnalysisDialog
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                tr("debug.embedding_analysis_failed_title"),
                tr("debug.embedding_analysis_import_failed", detail=f"{exc}\n\n{traceback.format_exc()}"),
            )
            return

        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_learning_tool_first"))
            return
        if not inspection_item.enabled:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.tool_disabled"))
            return
        if not self.algo.is_learning_tool(inspection_item.algorithm_code):
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.not_learning_tool"))
            return
        current_role = self.current_camera_role()
        allowed_learning_items = [
            item
            for item in list(getattr(self, "inspection_items", []) or [])
            if bool(getattr(item, "enabled", True))
            and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
            and self.algo.is_learning_tool(getattr(item, "algorithm_code", ""))
        ]
        allowed_model_keys = list(
            dict.fromkeys(
                str(getattr(item, "model_key", "") or "").strip()
                for item in allowed_learning_items
                if str(getattr(item, "model_key", "") or "").strip()
            )
        )
        allowed_backbones = []
        current_backbone = str(self.algo.current_learning_backbone() or "").strip()
        if current_backbone:
            allowed_backbones.append(current_backbone)
        try:
            dialog = EmbeddingAnalysisDialog(
                session_root=self.session.session_dir,
                initial_product=self.session.current_product,
                initial_backbone=current_backbone,
                initial_model_key=inspection_item.model_key,
                allowed_model_keys=allowed_model_keys,
                allowed_backbones=allowed_backbones,
                parent=self,
            )
            dialog.exec()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                tr("debug.embedding_analysis_failed_title"),
                tr("debug.embedding_analysis_init_failed", detail=f"{exc}\n\n{traceback.format_exc()}"),
            )

    # ------------------------------------------------------------------
    # 传统基线调试
    # ------------------------------------------------------------------


    def _run_traditional_baseline_debug(self) -> None:
        paths, tab_name = self._current_tab_paths_and_name()
        if not paths:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.current_list_empty"))
            return

        inspection_item = self._selected_inspection_item()
        if inspection_item is None:
            QtWidgets.QMessageBox.information(self, tr("common.info"), tr("debug.select_inspection_tool_first"))
            return
        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        display_name = str(
            inspection_item.display_name or inspection_item.roi_label or inspection_item.item_id or roi_label
        ).strip()

        rows: List[Dict[str, object]] = []
        ok = 0
        for path in paths:
            try:
                row = self._compute_traditional_baseline_metrics(path, preferred_label=roi_label)
                ok += 1
            except Exception as exc:
                row = {
                    "file_path": path, "file_name": os.path.basename(path),
                    "roi_label": "", "bbox_xywh": "", "mean_intensity": "", "mean_std": "",
                    "hsv_h_mean": "", "hsv_h_std": "", "hsv_s_mean": "", "hsv_s_std": "",
                    "hsv_v_mean": "", "hsv_v_std": "", "roi_area": "", "error": str(exc),
                }
            rows.append(row)

        json_path, csv_path = self._save_traditional_baseline_report(rows, tab_name=tab_name, roi_label=roi_label)
        self.lbl_status.setText(f"Status: baseline debug done for {display_name}, success {ok}/{len(paths)}")
        QtWidgets.QMessageBox.information(
            self,
            "Traditional Baseline Debug",
            f"Completed baseline metrics for {display_name} ({roi_label}) in {tab_name}.\n"
            f"Success: {ok}/{len(paths)}\n\nJSON:\n{json_path}\n\nCSV:\n{csv_path}",
        )





