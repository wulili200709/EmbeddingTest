"""Analysis and reporting helpers for ToolPage."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np

import algorithms.proxy as qr_core


def _summarize_test_rows(tool_page, rows: List[Dict[str, object]]) -> Dict[str, object]:
    labeled_rows = [row for row in rows if str(row.get("gt", "")) in {"OK", "NG"}]
    matched_rows = [
        row for row in labeled_rows
        if str(row.get("gt", "")) == str(row.get("pred", ""))
    ]
    total_ms_values = [
        float(row["total_ms"])
        for row in rows
        if row.get("total_ms") is not None
    ]
    return {
        "row_count": len(rows),
        "labeled_count": len(labeled_rows),
        "matched_count": len(matched_rows),
        "pred_ok_count": sum(1 for row in rows if str(row.get("pred", "")) == "OK"),
        "pred_ng_count": sum(1 for row in rows if str(row.get("pred", "")) == "NG"),
        "accuracy": (
            float(len(matched_rows)) / float(len(labeled_rows))
            if labeled_rows else None
        ),
        "avg_total_ms": (
            sum(total_ms_values) / float(len(total_ms_values))
            if total_ms_values else None
        ),
    }


def _write_test_rows_csv(tool_page, csv_path: str, rows: List[Dict[str, object]]) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "file_path", "gt", "pred", "status", "diff", "sim_ok", "sim_ng",
            "value", "threshold", "match_ms", "total_ms", "json",
        ])
        for row in rows:
            gt = str(row.get("gt", ""))
            pred = str(row.get("pred", ""))
            status = ""
            if gt in {"OK", "NG"} and pred:
                status = "PASS" if gt == pred else "FAIL"
            writer.writerow([
                row.get("file_name", ""),
                row.get("file_path", ""),
                gt,
                pred,
                status,
                row.get("diff", ""),
                row.get("sim_ok", ""),
                row.get("sim_ng", ""),
                row.get("value", ""),
                row.get("threshold", ""),
                row.get("match_ms", ""),
                row.get("total_ms", ""),
                row.get("json_name", ""),
            ])


def _save_test_result_report(
    tool_page,
    rows: List[Dict[str, object]],
    *,
    report_prefix: str,
    summary: Dict[str, object] | None = None,
) -> Tuple[str, str]:
    report_dir = os.path.join(tool_page.session.product_dir, "test_exports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{report_prefix}_{tool_page.current_algorithm()}_{stamp}"
    json_path = os.path.join(report_dir, base + ".json")
    csv_path = os.path.join(report_dir, base + ".csv")

    payload = {
        "product": tool_page.session.current_product,
        "algorithm": tool_page.current_algorithm(),
        "score_mode": tool_page.cmb_mode.currentText(),
        "margin": float(tool_page.spin_margin.value()),
        "topk": int(tool_page.spin_topk.value()),
        "loc_method": tool_page.loc_method,
        "summary": summary or tool_page._summarize_test_rows(rows),
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tool_page._write_test_rows_csv(csv_path, rows)
    return json_path, csv_path


def _suggest_margin_from_rows(tool_page, rows: List[Dict[str, object]]) -> Dict[str, object]:
    labeled = [row for row in rows if str(row.get("gt", "")) in {"OK", "NG"} and row.get("diff") is not None]
    if not labeled:
        raise RuntimeError("no labeled rows for margin suggestion")

    current_margin = float(tool_page.spin_margin.value())
    diffs = sorted({float(row["diff"]) for row in labeled})
    candidates: List[float] = []
    if diffs:
        candidates.append(diffs[0] - 1e-6)
        candidates.extend(diffs)
        candidates.extend((a + b) * 0.5 for a, b in zip(diffs, diffs[1:]))
        candidates.append(diffs[-1] + 1e-6)

    def _accuracy(threshold: float):
        tp = tn = fp = fn = 0
        for row in labeled:
            gt = str(row["gt"])
            pred = "OK" if float(row["diff"]) >= threshold else "NG"
            if gt == "OK" and pred == "OK":
                tp += 1
            elif gt == "OK" and pred == "NG":
                fn += 1
            elif gt == "NG" and pred == "NG":
                tn += 1
            else:
                fp += 1
        acc = float(tp + tn) / float(len(labeled))
        return acc, tp, tn, fp, fn

    best_margin = current_margin
    best_acc = -1.0
    best_conf = (0, 0, 0, 0)
    for candidate in candidates:
        acc, tp, tn, fp, fn = _accuracy(candidate)
        if acc > best_acc + 1e-12 or (
            abs(acc - best_acc) <= 1e-12
            and abs(candidate - current_margin) < abs(best_margin - current_margin)
        ):
            best_acc = acc
            best_margin = float(candidate)
            best_conf = (tp, tn, fp, fn)

    current_acc, current_tp, current_tn, current_fp, current_fn = _accuracy(current_margin)
    ok_diffs = [float(row["diff"]) for row in labeled if str(row["gt"]) == "OK"]
    ng_diffs = [float(row["diff"]) for row in labeled if str(row["gt"]) == "NG"]
    safe_range = None
    if ok_diffs and ng_diffs:
        lower = max(ng_diffs)
        upper = min(ok_diffs)
        if lower < upper:
            safe_range = (float(lower), float(upper))
            best_margin = float((lower + upper) * 0.5)
            best_acc, *conf = _accuracy(best_margin)
            best_conf = tuple(conf)

    return {
        "current_margin": current_margin,
        "current_accuracy": float(current_acc),
        "current_confusion": {
            "tp_ok": current_tp,
            "tn_ng": current_tn,
            "fp_ok_as_ng": current_fn,
            "fp_ng_as_ok": current_fp,
        },
        "suggested_margin": float(best_margin),
        "suggested_accuracy": float(best_acc),
        "suggested_confusion": {
            "tp_ok": best_conf[0],
            "tn_ng": best_conf[1],
            "fp_ng_as_ok": best_conf[2],
            "fp_ok_as_ng": best_conf[3],
        },
        "ok_diff_min": float(min(ok_diffs)) if ok_diffs else None,
        "ok_diff_max": float(max(ok_diffs)) if ok_diffs else None,
        "ng_diff_min": float(min(ng_diffs)) if ng_diffs else None,
        "ng_diff_max": float(max(ng_diffs)) if ng_diffs else None,
        "safe_range": safe_range,
    }


def _current_tab_paths_and_name(tool_page) -> Tuple[List[str], str]:
    tab = tool_page.tabs.currentIndex()
    if tab == 0:
        return list(tool_page.ok_files), "OK"
    if tab == 1:
        return list(tool_page.ng_files), "NG"
    return list(tool_page.test_files), "TEST"


def _load_roi_mask_crop(tool_page, img_path: str, preferred_label: str = "roi1") -> Dict[str, object]:
    jpath = qr_core.labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"missing labelme json: {jpath}")

    label_name = preferred_label
    shape = qr_core.read_shape_from_labelme(jpath, preferred_label)
    if shape is None:
        label_name = "roi"
        shape = qr_core.read_shape_from_labelme(jpath, label_name)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi annotation")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    h_img, w_img = img_bgr.shape[:2]

    pts = np.asarray(shape.get("points", []), dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError(f"{os.path.basename(img_path)} ROI points empty")
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x = max(0, int(np.floor(float(x_min))))
    y = max(0, int(np.floor(float(y_min))))
    x2 = min(w_img, int(np.ceil(float(x_max))))
    y2 = min(h_img, int(np.ceil(float(y_max))))
    if x2 <= x or y2 <= y:
        raise RuntimeError(f"{os.path.basename(img_path)} ROI bbox invalid")

    crop_bgr = img_bgr[y:y2, x:x2].copy()
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    rel_pts = pts - np.array([[x, y]], dtype=np.float32)
    if str(shape.get("shape_type", "rectangle")) == "polygon" and len(rel_pts) >= 3:
        cv2.fillPoly(mask, [np.round(rel_pts).astype(np.int32)], 255)
    else:
        p0 = rel_pts.min(axis=0)
        p1 = rel_pts.max(axis=0)
        rx = max(0, int(np.floor(float(p0[0]))))
        ry = max(0, int(np.floor(float(p0[1]))))
        rx2 = min(mask.shape[1], int(np.ceil(float(p1[0]))))
        ry2 = min(mask.shape[0], int(np.ceil(float(p1[1]))))
        mask[ry:ry2, rx:rx2] = 255
    return {
        "label_name": label_name,
        "crop_bgr": crop_bgr,
        "crop_gray": crop_gray,
        "mask": mask,
        "bbox_xywh": (x, y, x2 - x, y2 - y),
    }


def _compute_traditional_baseline_metrics(tool_page, img_path: str, preferred_label: str = "roi1") -> Dict[str, object]:
    roi = tool_page._load_roi_mask_crop(img_path, preferred_label=preferred_label)
    crop_gray = np.asarray(roi["crop_gray"], dtype=np.float32)
    crop_bgr = np.asarray(roi["crop_bgr"], dtype=np.uint8)
    mask = np.asarray(roi["mask"], dtype=np.uint8)
    valid_gray = crop_gray[mask > 0]
    if valid_gray.size == 0:
        raise RuntimeError(f"{os.path.basename(img_path)} ROI valid pixels empty")
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    valid_hsv = hsv[mask > 0]
    if valid_hsv.size == 0:
        raise RuntimeError(f"{os.path.basename(img_path)} ROI HSV pixels empty")
    valid_hsv = np.asarray(valid_hsv, dtype=np.float32)
    h_vals = valid_hsv[:, 0]
    s_vals = valid_hsv[:, 1]
    v_vals = valid_hsv[:, 2]
    return {
        "file_path": img_path,
        "file_name": os.path.basename(img_path),
        "roi_label": str(roi["label_name"]),
        "bbox_xywh": list(roi["bbox_xywh"]),
        "mean_intensity": float(np.mean(valid_gray)),
        "mean_std": float(np.std(valid_gray)),
        "hsv_h_mean": float(np.mean(h_vals)),
        "hsv_h_std": float(np.std(h_vals)),
        "hsv_s_mean": float(np.mean(s_vals)),
        "hsv_s_std": float(np.std(s_vals)),
        "hsv_v_mean": float(np.mean(v_vals)),
        "hsv_v_std": float(np.std(v_vals)),
        "roi_area": int(valid_gray.size),
    }


def _save_traditional_baseline_report(
    tool_page,
    rows: List[Dict[str, object]],
    tab_name: str,
    roi_label: str = "roi1",
) -> Tuple[str, str]:
    report_dir = os.path.join(tool_page.session.product_dir, "traditional_baseline_reports")
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    roi_slug = str(roi_label or "roi").strip() or "roi"
    base = f"baseline_{roi_slug}_hsv_{tab_name.lower()}_{stamp}"
    json_path = os.path.join(report_dir, base + ".json")
    csv_path = os.path.join(report_dir, base + ".csv")

    payload = {
        "product": tool_page.session.current_product,
        "tab": tab_name,
        "roi_label": roi_slug,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "roi_label", "bbox_xywh", "mean_intensity", "mean_std",
            "hsv_h_mean", "hsv_h_std", "hsv_s_mean", "hsv_s_std",
            "hsv_v_mean", "hsv_v_std", "roi_area", "error",
        ])
        for row in rows:
            writer.writerow([
                row.get("file_name", ""),
                row.get("roi_label", ""),
                row.get("bbox_xywh", ""),
                row.get("mean_intensity", ""),
                row.get("mean_std", ""),
                row.get("hsv_h_mean", ""),
                row.get("hsv_h_std", ""),
                row.get("hsv_s_mean", ""),
                row.get("hsv_s_std", ""),
                row.get("hsv_v_mean", ""),
                row.get("hsv_v_std", ""),
                row.get("roi_area", ""),
                row.get("error", ""),
            ])
    return json_path, csv_path
