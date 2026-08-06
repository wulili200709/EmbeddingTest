"""Prediction and test-result helpers for ToolPage."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from common.algorithm_codes import learning_backbone_storage_code
from ncc import locator as ncc_locator
from shape.core import locator as shape_locator


def _predict_image(
    tool_page,
    path: str,
    *,
    feat_net=None,
    prefer_canvas_roi: bool = False,
    labels_override: Optional[List[str]] = None,
    algorithm_override: Optional[str] = None,
    model_key_override: Optional[str] = None,
    params_override: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    total_t0 = time.perf_counter()
    camera_role = tool_page.current_camera_role()
    override_text = str(algorithm_override or "").strip()
    algorithm = (
        tool_page.algo.resolve_tool_algorithm(override_text, camera_role)
        if override_text
        else tool_page.current_algorithm()
    )

    match_ms: Optional[float] = None
    method_getter = getattr(tool_page, "loc_method_for_role", None)
    method = method_getter(camera_role) if callable(method_getter) else tool_page.loc_method
    if method == "shape":
        recipe = tool_page.shape_recipe_for_role(camera_role)
        ref_image = tool_page.ref_image
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
        if ref_image and os.path.exists(ref_image):
            run = shape_locator.autogen_roi_json_from_shape_timed(
                tgt_img_path=path,
                ref_img_path=ref_image,
                product_dir=tool_page.session.product_dir,
                camera_role=camera_role,
            )
            match_ms = float(run.total_ms)
            tool_page._shape_match_ms_by_image[path] = match_ms
            tool_page._shape_autogen_ms_by_image[path] = float(run.total_ms)
    elif method == "ncc":
        run = ncc_locator.autogen_roi_json_from_ncc_timed(
            tgt_img_path=path,
            product_dir=tool_page.session.product_dir,
            camera_role=camera_role,
        )
        match_ms = float(run.total_ms)
        tool_page._shape_match_ms_by_image[path] = match_ms
        tool_page._shape_autogen_ms_by_image[path] = float(run.total_ms)
    elif tool_page.ref_image and os.path.exists(tool_page.ref_image):
        tool_page._autogen_roi_for_images([path], only_missing=True, silent=True)

    labels = [str(label).strip() for label in (labels_override or []) if str(label).strip()]
    if not labels:
        labels = tool_page._loc_output_labels(camera_role)
    roi = None
    if prefer_canvas_roi and len(labels) == 1 and tool_page.canvas.image_path() == path:
        roi = tool_page._roi_xywh_from_canvas()

    if tool_page.algo.is_embedding_algorithm(algorithm):
        if not tool_page.algo._loaded_embedding_matches(
            algorithm,
            labels=labels,
            model_key=model_key_override or "",
        ):
            tool_page.load_embedding_model(algorithm, model_key=model_key_override)

    result = tool_page.algo.predict_image(
        path,
        labels=labels,
        feat_net=feat_net,
        roi=roi,
        match_ms=match_ms,
        algorithm_override=algorithm,
        model_key_override=model_key_override,
        params_override=params_override,
    )
    payload = result.to_dict()
    payload["infer_ms"] = (
        float(payload.get("total_ms", 0.0))
        if payload.get("total_ms") is not None
        else None
    )
    payload["total_ms"] = float((time.perf_counter() - total_t0) * 1000.0)
    return payload


def _populate_results_table(tool_page, rows: List[Dict[str, object]]) -> None:
    tool_page._current_result_rows = list(rows)
    tool_page.table.setRowCount(0)
    for row_idx, row in enumerate(rows):
        tool_page.table.insertRow(row_idx)
        measurement = row.get("measurement")
        unit = ""
        if isinstance(measurement, dict):
            unit = str(measurement.get("unit", "") or "").strip()

        def _format_number(key: str, decimals: int = 4, *, with_unit: bool = False) -> str:
            if row.get(key) is None:
                return ""
            text = f"{float(row.get(key, 0.0)):.{decimals}f}"
            return f"{text}{unit}" if with_unit and unit else text

        values = [
            str(row.get("file_name", "")),
            str(row.get("gt", "")),
            str(row.get("pred", "")),
            _format_number("diff"),
            _format_number("sim_ok"),
            _format_number("sim_ng"),
            _format_number("value", with_unit=True),
            _format_number("threshold", with_unit=True),
            _format_number("match_ms", 1),
            _format_number("infer_ms", 1),
            _format_number("total_ms", 1),
            str(row.get("json_name", "")),
        ]
        for col_idx, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            if col_idx == 0:
                item.setData(QtCore.Qt.UserRole, str(row.get("file_path", "")))
            gt = str(row.get("gt", ""))
            pred = str(row.get("pred", ""))
            if gt and pred and gt != pred:
                item.setForeground(QtGui.QBrush(QtGui.QColor(220, 30, 30)))
            tool_page.table.setItem(row_idx, col_idx, item)


def _daily_test_log_path(tool_page) -> str:
    log_dir = os.path.join(tool_page.session.product_dir, "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, datetime.now().strftime("%Y%m%d") + ".csv")


def _append_test_log(tool_page, row: Dict[str, object]) -> str:
    csv_path = tool_page._daily_test_log_path()
    fields = [
        "timestamp", "product", "algorithm", "score_mode", "margin", "topk",
        "tool_name", "camera_id", "roi_label",
        "file_name", "gt", "pred", "diff", "sim_ok", "sim_ng",
        "value", "threshold", "match_ms", "infer_ms", "total_ms", "json_name",
    ]
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as existing_file:
                existing_fields = next(csv.reader(existing_file), [])
        except OSError:
            existing_fields = []
        if existing_fields and existing_fields != fields:
            stem, suffix = os.path.splitext(csv_path)
            csv_path = stem + "_timing_v2" + suffix
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "product": tool_page.session.current_product,
            "algorithm": learning_backbone_storage_code(row.get("algorithm", tool_page.current_algorithm())),
            "score_mode": tool_page.cmb_mode.currentText(),
            "margin": float(tool_page.spin_margin.value()),
            "topk": int(tool_page.spin_topk.value()),
            "tool_name": row.get("tool_name", ""),
            "camera_id": row.get("camera_id", ""),
            "roi_label": row.get("roi_label", ""),
            "file_name": row.get("file_name", ""),
            "gt": row.get("gt", ""),
            "pred": row.get("pred", ""),
            "diff": row.get("diff", ""),
            "sim_ok": row.get("sim_ok", ""),
            "sim_ng": row.get("sim_ng", ""),
            "value": row.get("value", ""),
            "threshold": row.get("threshold", ""),
            "match_ms": row.get("match_ms", ""),
            "infer_ms": row.get("infer_ms", ""),
            "total_ms": row.get("total_ms", ""),
            "json_name": row.get("json_name", ""),
        })
    return csv_path



