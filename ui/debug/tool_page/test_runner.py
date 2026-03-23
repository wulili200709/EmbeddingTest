"""Prediction and test-result helpers for ToolPage."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from line2dup.core import locator as line2dup_locator


def _predict_image(
    tool_page,
    path: str,
    *,
    feat_net=None,
    prefer_canvas_roi: bool = False,
    labels_override: Optional[List[str]] = None,
) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    total_t0 = time.perf_counter()
    algorithm = tool_page.current_algorithm()

    match_ms: Optional[float] = None
    if tool_page.loc_method == "line2dup":
        recipe = tool_page.line2dup_recipe
        if recipe is None and os.path.exists(tool_page.session.line2dup_recipe_path):
            recipe = line2dup_locator.load_recipe_for_product(tool_page.session.product_dir)
            tool_page.line2dup_recipe = recipe
        ref_image = tool_page.ref_image
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
        if ref_image and os.path.exists(ref_image):
            run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                tgt_img_path=path,
                ref_img_path=ref_image,
                product_dir=tool_page.session.product_dir,
            )
            match_ms = float(run.locate_ms)
            tool_page._line2dup_match_ms_by_image[path] = match_ms
            tool_page._line2dup_autogen_ms_by_image[path] = float(run.total_ms)
    elif tool_page.ref_image and os.path.exists(tool_page.ref_image):
        tool_page._autogen_roi_for_images([path], only_missing=True, silent=True)

    labels = [str(label).strip() for label in (labels_override or []) if str(label).strip()]
    if not labels:
        labels = tool_page._line2dup_output_labels() if tool_page.loc_method == "line2dup" else ["roi"]
    roi = None
    if prefer_canvas_roi and len(labels) == 1 and tool_page.canvas.image_path() == path:
        roi = tool_page._roi_xywh_from_canvas()

    if tool_page.algo.is_embedding_algorithm(algorithm):
        if tool_page.algo.model is None or tool_page.algo.model.backbone != algorithm:
            tool_page.load_embedding_model(algorithm)

    result = tool_page.algo.predict_image(
        path,
        labels=labels,
        feat_net=feat_net,
        roi=roi,
        match_ms=match_ms,
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
        values = [
            str(row.get("file_name", "")),
            str(row.get("gt", "")),
            str(row.get("pred", "")),
            f"{float(row.get('diff', 0.0)):.4f}" if row.get("diff") is not None else "",
            f"{float(row.get('sim_ok', 0.0)):.4f}" if row.get("sim_ok") is not None else "",
            f"{float(row.get('sim_ng', 0.0)):.4f}" if row.get("sim_ng") is not None else "",
            f"{float(row.get('value', 0.0)):.4f}" if row.get("value") is not None else "",
            f"{float(row.get('threshold', 0.0)):.4f}" if row.get("threshold") is not None else "",
            f"{float(row.get('match_ms', 0.0)):.1f}" if row.get("match_ms") is not None else "",
            f"{float(row.get('total_ms', 0.0)):.1f}" if row.get("total_ms") is not None else "",
            str(row.get("json_name", "")),
        ]
        for col_idx, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            if col_idx == 0:
                item.setData(QtCore.Qt.UserRole, str(row.get("file_path", "")))
            gt = str(row.get("gt", ""))
            pred = str(row.get("pred", ""))
            if gt and pred and gt != pred:
                item.setForeground(QtGui.QBrush(QtGui.QColor(192, 32, 32)))
            tool_page.table.setItem(row_idx, col_idx, item)


def _daily_test_log_path(tool_page) -> str:
    log_dir = os.path.join(tool_page.session.product_dir, "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, datetime.now().strftime("%Y%m%d") + ".csv")


def _append_test_log(tool_page, row: Dict[str, object]) -> str:
    csv_path = tool_page._daily_test_log_path()
    fields = [
        "timestamp", "product", "algorithm", "score_mode", "margin", "topk",
        "file_name", "gt", "pred", "diff", "sim_ok", "sim_ng",
        "value", "threshold", "match_ms", "total_ms", "json_name",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "product": tool_page.session.current_product,
            "algorithm": tool_page.current_algorithm(),
            "score_mode": tool_page.cmb_mode.currentText(),
            "margin": float(tool_page.spin_margin.value()),
            "topk": int(tool_page.spin_topk.value()),
            "file_name": row.get("file_name", ""),
            "gt": row.get("gt", ""),
            "pred": row.get("pred", ""),
            "diff": row.get("diff", ""),
            "sim_ok": row.get("sim_ok", ""),
            "sim_ng": row.get("sim_ng", ""),
            "value": row.get("value", ""),
            "threshold": row.get("threshold", ""),
            "match_ms": row.get("match_ms", ""),
            "total_ms": row.get("total_ms", ""),
            "json_name": row.get("json_name", ""),
        })
    return csv_path
