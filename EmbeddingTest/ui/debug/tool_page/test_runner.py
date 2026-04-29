"""Prediction and test-result helpers for ToolPage."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from algorithms.registry import normalize_learning_backbone
from line2dup.core import locator as line2dup_locator


def _format_result_reason(tool_page, row: Dict[str, object]) -> str:
    pred = str(row.get("pred", "") or "").strip()
    diff = row.get("diff")
    value = row.get("value")
    threshold = row.get("threshold")
    sim_ok = row.get("sim_ok")
    sim_ng = row.get("sim_ng")

    if value is not None and threshold is not None:
        score = float(value)
        threshold_value = float(threshold)
        gap = float(threshold_value - score)
        relation = "<=" if score <= threshold_value else ">"
        decision = pred or ("OK" if gap >= 0.0 else "NG")
        return (
            f"{decision}: anomaly score={score:.4f} {relation} threshold={threshold_value:.4f} "
            f"(threshold-score={gap:.4f})"
        )

    if diff is not None and sim_ok is not None and sim_ng is not None:
        diff_value = float(diff)
        margin_value = row.get("margin")
        if margin_value is not None:
            margin = float(margin_value)
        else:
            margin_widget = getattr(tool_page, "spin_margin", None)
            margin = float(margin_widget.value()) if margin_widget is not None else 0.0
        relation = ">=" if diff_value >= margin else "<"
        decision = pred or ("OK" if diff_value >= margin else "NG")
        return (
            f"{decision}: diff={diff_value:.4f} {relation} margin={margin:.4f} "
            f"(sim_ok={float(sim_ok):.4f}, sim_ng={float(sim_ng):.4f})"
        )

    if diff is not None and threshold is not None:
        diff_value = float(diff)
        threshold_value = float(threshold)
        decision = pred or ("OK" if diff_value >= 0.0 else "NG")
        return f"{decision}: value-threshold={diff_value:.4f} (threshold={threshold_value:.4f})"

    return pred


def _predict_image(
    tool_page,
    path: str,
    *,
    feat_net=None,
    prefer_canvas_roi: bool = False,
    labels_override: Optional[List[str]] = None,
    algorithm_override: Optional[str] = None,
    model_key_override: Optional[str] = None,
) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    total_t0 = time.perf_counter()
    override_text = str(algorithm_override or "").strip()
    algorithm = (
        tool_page.algo.resolve_tool_algorithm(override_text)
        if override_text
        else tool_page.current_algorithm()
    )

    match_ms: Optional[float] = None
    if tool_page.loc_method == "line2dup":
        camera_role = tool_page.current_camera_role()
        recipe = tool_page.line2dup_recipe_for_role(camera_role)
        ref_image = tool_page.ref_image
        if recipe is not None and recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
        if ref_image and os.path.exists(ref_image):
            run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                tgt_img_path=path,
                ref_img_path=ref_image,
                product_dir=tool_page.session.product_dir,
                camera_role=camera_role,
            )
            match_ms = float(run.total_ms)
            tool_page._line2dup_match_ms_by_image[path] = match_ms
            tool_page._line2dup_autogen_ms_by_image[path] = float(run.total_ms)
    elif tool_page.loc_method == "ncc":
        tool_page._autogen_roi_for_images([path], only_missing=False, silent=True)
        invalidate_shape_cache = getattr(tool_page, "_invalidate_shape_lookup_cache", None)
        if callable(invalidate_shape_cache):
            invalidate_shape_cache(path)
        match_ms = tool_page._line2dup_match_ms_by_image.get(path)

    labels = [str(label).strip() for label in (labels_override or []) if str(label).strip()]
    if not labels:
        labels_getter = getattr(tool_page, "_current_loc_output_labels", None)
        if callable(labels_getter):
            labels = list(labels_getter(tool_page.current_camera_role()))
        elif tool_page.loc_method == "line2dup":
            labels = tool_page._line2dup_output_labels()
        else:
            labels = ["roi"]
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
        algorithm_override=algorithm_override,
        model_key_override=model_key_override,
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
    tool_page.table.setHorizontalHeaderLabels(
        ["文件", "GT", "Pred", "diff", "sim_ok", "sim_ng", "score/value", "threshold", "match_ms", "total_ms", "json"]
    )
    tool_page.table.setToolTip(
        "学习工具: diff = sim_ok - sim_ng；PatchCore Lite: score/value 越大越异常，score 超过 threshold 才判 NG。"
    )
    for row_idx, row in enumerate(rows):
        tool_page.table.insertRow(row_idx)
        reason = _format_result_reason(tool_page, row)
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
            item.setToolTip(reason)
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
        "value", "threshold", "match_ms", "total_ms", "json_name",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        algorithm = row.get("algorithm", tool_page.current_algorithm())
        normalized_algorithm = normalize_learning_backbone(algorithm)
        if normalized_algorithm in {"b0", "b1", "b2"}:
            algorithm = normalized_algorithm
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "product": tool_page.session.current_product,
            "algorithm": algorithm,
            "score_mode": row.get("score_mode", tool_page.cmb_mode.currentText()),
            "margin": row.get("margin", float(tool_page.spin_margin.value())),
            "topk": row.get("topk", int(tool_page.spin_topk.value())),
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
            "total_ms": row.get("total_ms", ""),
            "json_name": row.get("json_name", ""),
        })
    return csv_path
