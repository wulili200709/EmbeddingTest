"""Inspection item status text and tooltip helpers."""

from __future__ import annotations

import os
from datetime import datetime

from ui.debug.tool_page.measurement_algorithms import (
    is_bright_block_center_algorithm,
    is_bright_block_y_distance_algorithm,
    is_center_distance_algorithm,
    is_line_distance_algorithm,
    is_single_roi_distance_algorithm,
    public_algorithm_code,
)
from ui.i18n import tr


def _format_timestamp(path: str) -> str:
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return "-"
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _inspection_item_status(tool_page, inspection_item):
    if not getattr(inspection_item, "enabled", True):
        return tr("debug.status.disabled"), "Current tool is disabled and will not join training or runtime.", "#8a8a8a"

    if tool_page.algo.is_learning_tool(inspection_item.algorithm_code):
        backbone = tool_page.algo.current_learning_backbone(inspection_item.camera_id)
        if not backbone:
            return tr("debug.status.not_selected"), "Select a subtype for the learning tool first.", "#d98c8c"
        model_key = inspection_item.effective_model_key
        storage_paths = tool_page.algo.embedding_model_storage_paths(
            backbone,
            tool_page.session.product_dir,
            model_key=model_key,
        )
        model_path = storage_paths[0]
        if os.path.exists(model_path):
            tooltip = tr(
                "debug.tooltip.learning_trained",
                model=os.path.basename(model_path),
                backbone=tool_page.algo.algorithm_display_name(backbone) or backbone,
                time=_format_timestamp(model_path),
            )
            return tr("debug.status.trained"), tooltip, "#79d279"
        legacy_path = next(
            (
                path
                for path in storage_paths
                if path != model_path and os.path.exists(path)
            ),
            "",
        )
        if legacy_path:
            tooltip = tr(
                "debug.tooltip.learning_trained_legacy",
                model=os.path.basename(legacy_path),
                backbone=tool_page.algo.algorithm_display_name(backbone) or backbone,
                time=_format_timestamp(legacy_path),
            )
            return tr("debug.status.trained_legacy"), tooltip, "#cfc76a"
        tooltip = tr(
            "debug.tooltip.learning_untrained",
            model=os.path.basename(model_path),
            backbone=tool_page.algo.algorithm_display_name(backbone) or backbone,
        )
        return tr("debug.status.untrained"), tooltip, "#d98c8c"

    if getattr(tool_page.algo, "is_measurement_tool", lambda _code: False)(inspection_item.algorithm_code):
        algorithm = tool_page.algo.resolve_tool_algorithm(
            inspection_item.algorithm_code,
            inspection_item.camera_id,
        )
        if is_line_distance_algorithm(algorithm):
            display_algorithm = "line_distance"
            params = dict(getattr(inspection_item, "params", {}) or {})
            line_a = str(params.get("line_a_item_id", "") or "").strip() or "-"
            line_b = str(params.get("line_b_item_id", "") or "").strip() or "-"
            ready = line_a != "-" and line_b != "-" and line_a != line_b
            tooltip = (
                f"Algorithm: {tool_page.algo.algorithm_display_name(display_algorithm) or display_algorithm}\n"
                f"Measures distance from {line_a} to {line_b} and judges OK/NG from lower/upper limits."
            )
            return (f"{line_a} -> {line_b}" if ready else "Select lines"), tooltip, "#79d279" if ready else "#d98c8c"
        if is_center_distance_algorithm(algorithm):
            display_algorithm = public_algorithm_code(algorithm)
            params = dict(getattr(inspection_item, "params", {}) or {})
            center_a = str(params.get("center_a_item_id", "") or "").strip() or "-"
            center_b = str(params.get("center_b_item_id", "") or "").strip() or "-"
            mode = str(params.get("distance_mode", "vertical") or "vertical").strip()
            tooltip = (
                f"Algorithm: {tool_page.algo.algorithm_display_name(display_algorithm) or display_algorithm}\n"
                f"Measures {mode} distance from {center_a} to {center_b} and judges OK/NG from lower/upper limits."
            )
            ready = center_a != "-" and center_b != "-" and center_a != center_b
            return (
                f"{center_a} -> {center_b}" if ready else "Select centers",
                tooltip,
                "#79d279" if ready else "#d98c8c",
            )
        if is_single_roi_distance_algorithm(algorithm):
            display_algorithm = public_algorithm_code(algorithm)
            description = (
                "Finds one vertical bright block and one horizontal bright block, then measures the Y distance."
                if is_bright_block_y_distance_algorithm(algorithm)
                else "Finds two bright pin-tip centers inside one ROI and measures the center distance."
            )
            tooltip = (
                f"Algorithm: {tool_page.algo.algorithm_display_name(display_algorithm) or display_algorithm}\n"
                f"{description}"
            )
            return "Ready", tooltip, "#79d279"
        if is_bright_block_center_algorithm(algorithm):
            display_algorithm = public_algorithm_code(algorithm)
            tooltip = (
                f"Algorithm: {tool_page.algo.algorithm_display_name(display_algorithm) or display_algorithm}\n"
                "Finds one bright block center inside the ROI for downstream center distance measurement."
            )
            return "Ready", tooltip, "#79d279"
        display_algorithm = public_algorithm_code(algorithm)
        tooltip = (
            f"Algorithm: {tool_page.algo.algorithm_display_name(display_algorithm) or display_algorithm}\n"
            "Finds one fitted line inside the ROI for downstream distance measurement."
        )
        return "Ready", tooltip, "#79d279"

    algorithm = tool_page.algo.resolve_tool_algorithm(
        inspection_item.algorithm_code,
        inspection_item.camera_id,
    )
    model_key = inspection_item.effective_model_key
    model_dict = tool_page.algo.get_traditional_model_dict(algorithm, model_key=model_key)
    if isinstance(model_dict, dict):
        storage_key = tool_page.algo.traditional_model_storage_key(algorithm, model_key=model_key)
        candidate_keys = [
            tool_page.algo.traditional_model_storage_key(algorithm, model_key=candidate_model_key)
            for candidate_model_key in tool_page.algo.tool_model_storage_keys(model_key)
        ]
        actual_key = next(
            (
                candidate_key
                for candidate_key in candidate_keys
                if candidate_key in tool_page.algo.product_params.traditional_models
            ),
            algorithm,
        )
        threshold = model_dict.get("threshold")
        ok_when = str(model_dict.get("ok_when", "")).strip() or "-"
        accuracy = model_dict.get("accuracy")
        detail_parts = [
            tr("debug.tooltip.algorithm", algorithm=tool_page.algo.algorithm_display_name(algorithm) or algorithm),
            tr("debug.tooltip.threshold", threshold=f"{float(threshold):.4f}" if threshold is not None else "-"),
            tr("debug.tooltip.rule", rule=ok_when),
        ]
        if accuracy is not None:
            detail_parts.append(tr("debug.tooltip.accuracy", accuracy=f"{float(accuracy):.4f}"))
        detail_parts.append(tr("debug.tooltip.storage_key", storage_key=actual_key))
        status_text = tr("debug.status.calibrated") if actual_key == storage_key else tr("debug.status.calibrated_legacy")
        color = "#79d279" if actual_key == storage_key else "#cfc76a"
        return status_text, "\n".join(detail_parts), color

    tooltip = tr(
        "debug.tooltip.traditional_untrained",
        algorithm=tool_page.algo.algorithm_display_name(algorithm) or algorithm,
        storage_key=tool_page.algo.traditional_model_storage_key(algorithm, model_key=model_key),
    )
    return tr("debug.status.uncalibrated"), tooltip, "#d98c8c"


