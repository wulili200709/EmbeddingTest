"""Explicit ToolPage operation mixin.

This preserves the existing API while replacing import-time ``setattr`` class
mutation. Cohesive areas can now be migrated from this compatibility mixin to
the dedicated controllers already owned by ToolPage.
"""

from __future__ import annotations

from . import (
    analysis_tools,
    auto_roi,
    auto_roi_flow,
    auto_roi_reference_flow,
    camera_debug,
    debug_camera_flow,
    debug_io_flow,
    roi_ops,
    test_runner,
    tool_config,
)


class ToolPageOperationsMixin:
    _load_canvas_image = roi_ops._load_canvas_image
    _set_status_for_current_image = roi_ops._set_status_for_current_image
    _current_label = roi_ops._current_label
    _update_save_label_text = roi_ops._update_save_label_text
    _set_overlay_shapes = roi_ops._set_overlay_shapes
    _load_shape_for_label = roi_ops._load_shape_for_label
    _on_shapes_changed = roi_ops._on_shapes_changed
    _roi_xywh_from_canvas = roi_ops._roi_xywh_from_canvas

    _shape_output_labels = auto_roi._shape_output_labels
    _ncc_output_labels = auto_roi._ncc_output_labels
    _loc_output_labels = auto_roi._loc_output_labels
    _inspection_item_labels = auto_roi._inspection_item_labels
    _reload_inspection_items = auto_roi._reload_inspection_items
    _missing_roi_files = auto_roi._missing_roi_files
    _existing_roi_like_labels = auto_roi._existing_roi_like_labels
    _clear_roi_labels_for_paths = auto_roi._clear_roi_labels_for_paths

    _set_reference = auto_roi_reference_flow._set_reference
    _set_ref_from_current = auto_roi_reference_flow._set_ref_from_current
    _pick_ref_image = auto_roi_reference_flow._pick_ref_image
    _open_shape_template_page = auto_roi_reference_flow._open_shape_template_page
    _on_template_editor_dialog_destroyed = auto_roi_reference_flow._on_template_editor_dialog_destroyed
    _on_shape_model_saved = auto_roi_reference_flow._on_shape_model_saved
    _on_shape_reference_regions_changed = auto_roi_reference_flow._on_shape_reference_regions_changed
    _sync_shape_recipe_and_items = auto_roi_reference_flow._sync_shape_recipe_and_items
    _update_loc_ui = auto_roi_reference_flow._update_loc_ui
    _on_loc_method_changed = auto_roi_reference_flow._on_loc_method_changed

    _resolve_autogen_targets = auto_roi_flow._resolve_autogen_targets
    _autogen_roi_for_images = auto_roi_flow._autogen_roi_for_images
    _autogen_roi_current_tab = auto_roi_flow._autogen_roi_current_tab
    _autogen_roi_all = auto_roi_flow._autogen_roi_all
    _clear_roi_for_images = auto_roi_flow._clear_roi_for_images
    _clear_roi_current_tab = auto_roi_flow._clear_roi_current_tab

    _predict_image = test_runner._predict_image
    _populate_results_table = test_runner._populate_results_table
    _daily_test_log_path = test_runner._daily_test_log_path
    _append_test_log = test_runner._append_test_log

    _selected_inspection_item_row = tool_config._selected_inspection_item_row
    _selected_inspection_item = tool_config._selected_inspection_item
    _on_inspection_items_selection_changed = tool_config._on_inspection_items_selection_changed
    _persist_inspection_items = tool_config._persist_inspection_items
    _add_line_distance_tool = tool_config._add_line_distance_tool
    _delete_selected_line_distance_tool = tool_config._delete_selected_line_distance_tool
    _update_delete_line_distance_button = tool_config._update_delete_line_distance_button
    _refresh_inspection_items_table = tool_config._refresh_inspection_items_table
    _update_learning_backbone_hint = tool_config._update_learning_backbone_hint
    _update_measurement_params_panel = tool_config._update_measurement_params_panel
    _on_measurement_params_changed = tool_config._on_measurement_params_changed
    _on_inspection_items_table_item_changed = tool_config._on_inspection_items_table_item_changed
    _on_inspection_item_camera_changed = tool_config._on_inspection_item_camera_changed
    _on_inspection_item_algorithm_changed = tool_config._on_inspection_item_algorithm_changed

    _embedding_test_root = camera_debug._embedding_test_root
    _selected_debug_camera_serial = camera_debug._selected_debug_camera_serial
    _selected_debug_camera_role = camera_debug._selected_debug_camera_role
    _debug_capture_channel_for_role = camera_debug._debug_capture_channel_for_role
    _debug_physical_camera_role = camera_debug._debug_physical_camera_role
    _debug_capture_light_index = camera_debug._debug_capture_light_index
    _load_debug_role_binding = camera_debug._load_debug_role_binding
    _save_debug_role_binding = camera_debug._save_debug_role_binding
    _apply_debug_role_binding_to_camera_combo = camera_debug._apply_debug_role_binding_to_camera_combo
    _refresh_debug_role_status = camera_debug._refresh_debug_role_status
    _debug_camera_settings_payload_from_ui = camera_debug._debug_camera_settings_payload_from_ui
    _load_saved_debug_camera_settings_to_ui = camera_debug._load_saved_debug_camera_settings_to_ui
    _save_debug_camera_settings = camera_debug._save_debug_camera_settings
    _update_capture_channel_visibility = camera_debug._update_capture_channel_visibility
    _update_capture_channel_count = camera_debug._update_capture_channel_count
    _load_capture_config_to_ui = camera_debug._load_capture_config_to_ui
    _save_capture_config_from_ui = camera_debug._save_capture_config_from_ui
    _on_capture_mode_changed = camera_debug._on_capture_mode_changed
    _on_capture_channel_item_changed = camera_debug._on_capture_channel_item_changed
    _on_capture_channel_editor_changed = camera_debug._on_capture_channel_editor_changed
    _set_debug_preview_placeholder = camera_debug._set_debug_preview_placeholder
    _show_debug_preview_image = camera_debug._show_debug_preview_image
    _set_debug_preview_running = camera_debug._set_debug_preview_running
    _selected_debug_camera_info = camera_debug._selected_debug_camera_info
    _debug_camera_device = camera_debug._debug_camera_device
    _default_io_mapping_path = camera_debug._default_io_mapping_path
    _find_debug_nkio_dll_path = camera_debug._find_debug_nkio_dll_path
    _find_debug_nkio_config_path = camera_debug._find_debug_nkio_config_path
    _cleanup_debug_hardware = camera_debug._cleanup_debug_hardware

    _on_debug_camera_param_editing_finished = debug_camera_flow._on_debug_camera_param_editing_finished
    _on_debug_camera_trigger_activated = debug_camera_flow._on_debug_camera_trigger_activated
    _save_debug_camera_image = debug_camera_flow._save_debug_camera_image
    _toggle_debug_camera_preview = debug_camera_flow._toggle_debug_camera_preview
    _start_debug_camera_preview = debug_camera_flow._start_debug_camera_preview
    _stop_debug_camera_preview = debug_camera_flow._stop_debug_camera_preview
    _on_debug_preview_frame_ready = debug_camera_flow._on_debug_preview_frame_ready
    _on_debug_preview_error = debug_camera_flow._on_debug_preview_error
    _on_debug_preview_finished = debug_camera_flow._on_debug_preview_finished
    _set_debug_camera_status = debug_camera_flow._set_debug_camera_status
    _ensure_debug_camera_services = debug_camera_flow._ensure_debug_camera_services
    _refresh_debug_camera_list = debug_camera_flow._refresh_debug_camera_list
    _on_debug_camera_role_changed = debug_camera_flow._on_debug_camera_role_changed
    _on_debug_camera_selected = debug_camera_flow._on_debug_camera_selected
    _refresh_debug_camera_info = debug_camera_flow._refresh_debug_camera_info
    _connect_debug_camera = debug_camera_flow._connect_debug_camera
    _disconnect_debug_camera_requested = debug_camera_flow._disconnect_debug_camera_requested
    _disconnect_debug_camera = debug_camera_flow._disconnect_debug_camera
    _refresh_debug_camera_settings = debug_camera_flow._refresh_debug_camera_settings
    _apply_debug_camera_settings = debug_camera_flow._apply_debug_camera_settings
    _grab_debug_camera_once = debug_camera_flow._grab_debug_camera_once

    _open_debug_io = debug_io_flow._open_debug_io
    _close_debug_io = debug_io_flow._close_debug_io
    _refresh_debug_io_snapshot = debug_io_flow._refresh_debug_io_snapshot
    _set_debug_output_channel = debug_io_flow._set_debug_output_channel
    _set_debug_output = debug_io_flow._set_debug_output

    _summarize_test_rows = analysis_tools._summarize_test_rows
    _write_test_rows_csv = analysis_tools._write_test_rows_csv
    _save_test_result_report = analysis_tools._save_test_result_report
    _suggest_margin_from_rows = analysis_tools._suggest_margin_from_rows
    _current_tab_paths_and_name = analysis_tools._current_tab_paths_and_name
    _load_roi_mask_crop = analysis_tools._load_roi_mask_crop
    _compute_traditional_baseline_metrics = analysis_tools._compute_traditional_baseline_metrics
    _save_traditional_baseline_report = analysis_tools._save_traditional_baseline_report


__all__ = ["ToolPageOperationsMixin"]
