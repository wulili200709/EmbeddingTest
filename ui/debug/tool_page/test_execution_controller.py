from __future__ import annotations

import os
from typing import Dict, List

from PySide6 import QtCore, QtWidgets

from application import InspectionExecutionRequest, InspectionExecutor, ToolPageRuntimeContext
from domain import InspectionItem
from ui.i18n import tr
from ui.roi_overlay_colors import is_roi_label


def _normalize_camera_role(camera_id: object) -> str:
    text = str(camera_id or "").strip().lower()
    if text in {"cam1", "cam2"}:
        return text
    return ""


class TestExecutionController:
    def __init__(self, owner) -> None:
        self.owner = owner

    def target_inspection_items(self) -> List[InspectionItem]:
        current_role = self.owner.current_camera_role()
        return [
            item
            for item in self.owner.inspection_items
            if item.enabled and _normalize_camera_role(getattr(item, "camera_id", "")) == current_role
        ]

    def record_roi_result(self, path: str, label_name: str, status: object) -> None:
        if not path:
            return
        label = str(label_name or "").strip()
        if not is_roi_label(label):
            return
        status_text = str(status or "").strip().lower()
        if not status_text:
            return
        self.owner._roi_results_by_image.setdefault(path, {})[label] = status_text

    def execute_image(self, path: str) -> Dict[str, object]:
        p = str(path or "").strip()
        if p is None or not os.path.exists(p):
            raise FileNotFoundError(p or tr("debug.open_test_image_first"))
        self.owner.canvas.set_overlays([])

        target_items = self.target_inspection_items()
        camera_id = (
            str(target_items[0].camera_id or "").strip()
            if target_items
            else self.owner.current_camera_role()
        ) or "cam1"

        executor = InspectionExecutor(ToolPageRuntimeContext(self.owner))
        try:
            response = executor.execute(
                InspectionExecutionRequest(
                    camera_id=camera_id,
                    image_path=p,
                    items=target_items,
                )
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        rows: List[Dict[str, object]] = []
        log_names: List[str] = []
        raw_rows = []
        match_ms = float(response.match_ms or 0.0)
        infer_ms = float(response.infer_ms or 0.0)
        total_ms = float(response.total_ms or 0.0)
        if total_ms <= 0.0 and (match_ms > 0.0 or infer_ms > 0.0):
            total_ms = match_ms + infer_ms
        if isinstance(response.raw_row, dict):
            if isinstance(response.raw_row.get("item_rows"), list):
                raw_rows = [dict(row) for row in response.raw_row.get("item_rows", [])]
            elif response.raw_row:
                raw_rows = [dict(response.raw_row)]

        if target_items:
            for index, item_result in enumerate(response.item_results):
                row = dict(raw_rows[index]) if index < len(raw_rows) else {}
                display_name = str(item_result.display_name or item_result.item_id or "tool").strip()
                roi_label = str(item_result.roi_label or "").strip()
                algorithm = (
                    self.owner.algo.current_learning_backbone()
                    if self.owner.algo.is_learning_tool(item_result.algorithm_code)
                    else self.owner.algo.resolve_tool_algorithm(item_result.algorithm_code)
                )
                if row.get("pred") is not None and str(row.get("pred", "")).strip() != str(item_result.result or "").strip():
                    row.setdefault("raw_pred", row.get("pred"))
                row["pred"] = item_result.result
                row["match_ms"] = match_ms if match_ms > 0.0 else row.get("match_ms")
                row["total_ms"] = total_ms if total_ms > 0.0 else row.get("total_ms")
                row["tool_name"] = display_name
                row["camera_id"] = item_result.camera_id
                row["roi_label"] = roi_label
                row["algorithm"] = algorithm
                row["file_name"] = f"{os.path.basename(p)} [{display_name}]"
                if roi_label:
                    self.record_roi_result(p, roi_label, item_result.result)
                rows.append(row)
                log_names.append(os.path.basename(self.owner._append_test_log(row)))
        else:
            row = dict(raw_rows[0]) if raw_rows else {}
            row.setdefault("pred", response.result)
            row["match_ms"] = match_ms if match_ms > 0.0 else row.get("match_ms")
            row["total_ms"] = total_ms if total_ms > 0.0 else row.get("total_ms")
            labels_override = (
                self.owner._shape_output_labels()
                if self.owner.loc_method == "shape"
                else ["roi"]
            )
            for roi_label in labels_override:
                self.record_roi_result(p, roi_label, response.result)
            rows.append(row)
            log_names.append(os.path.basename(self.owner._append_test_log(row)))

        self.owner._populate_results_table(rows)

        overall_pred = str(response.result or "NG")
        ng_names = [
            str(row.get("tool_name", row.get("roi_label", "")) or "").strip()
            for row in rows
            if str(row.get("pred", "NG") or "NG").strip().upper() == "NG"
        ]
        status_text = (
            f"Status: TEST={os.path.basename(p)}  overall={overall_pred}"
            f"  tools={len(rows)}"
        )
        if ng_names:
            status_text += "  NG=" + ", ".join(ng_names[:5])
        if match_ms > 0.0:
            status_text += f"  match={match_ms:.1f}ms"
        status_text += f"  infer={infer_ms:.1f}ms"
        if log_names:
            status_text += f"  log={log_names[-1]}"
        self.owner.lbl_status.setText(status_text)
        self.owner._load_canvas_image(p)
        self.owner._update_sample_panel_widgets()
        return {
            "result": overall_pred,
            "rows": rows,
            "log_names": log_names,
            "status_text": status_text,
        }

    def run_current_image(self) -> None:
        path = self.owner.canvas.image_path()
        if path is None or not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self.owner, tr("common.info"), tr("debug.open_test_image_first"))
            return
        try:
            self.execute_image(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.owner, tr("debug.test_failed_title"), str(exc))

    def run_all_test_samples(self) -> None:
        current_role = self.owner._selected_image_list_camera_role()
        paths = list(self.owner._sample_paths_for_kind("test", current_role))
        if not paths:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.test_all_test_samples_empty"))
            return

        total = len(paths)
        ok_count = 0
        ng_count = 0
        failed_count = 0
        failures: List[str] = []
        buttons = [
            getattr(self.owner, "btn_clear_session", None),
            getattr(self.owner, "btn_test", None),
            getattr(self.owner, "btn_export_test", None),
        ]
        old_enabled = {
            button: bool(button.isEnabled())
            for button in buttons
            if button is not None
        }
        for button in old_enabled:
            button.setEnabled(False)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for index, path in enumerate(paths, start=1):
                self.owner.lbl_status.setText(
                    f"Status: batch test {index}/{total}  {os.path.basename(path)}"
                )
                QtWidgets.QApplication.processEvents()
                try:
                    result = self.execute_image(path)
                except Exception as exc:
                    failed_count += 1
                    if len(failures) < 5:
                        failures.append(f"{os.path.basename(path)}: {exc}")
                    continue
                if str(result.get("result", "") or "").strip().upper() == "OK":
                    ok_count += 1
                else:
                    ng_count += 1
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            for button, enabled in old_enabled.items():
                button.setEnabled(enabled)

        message = tr(
            "debug.test_all_test_samples_done_message",
            total=total,
            ok=ok_count,
            ng=ng_count,
            failed=failed_count,
        )
        if failures:
            message += "\n\n" + "\n".join(failures)
        QtWidgets.QMessageBox.information(
            self.owner,
            tr("debug.test_all_test_samples_done_title"),
            message,
        )
        self.owner.lbl_status.setText(
            f"Status: batch test done total={total} OK={ok_count} NG={ng_count} failed={failed_count}"
        )

    def export_current_results_csv(self) -> None:
        if not self.owner._current_result_rows:
            QtWidgets.QMessageBox.information(self.owner, tr("common.info"), tr("debug.no_export_results"))
            return
        json_path, csv_path = self.owner._save_test_result_report(
            self.owner._current_result_rows,
            report_prefix="test_result",
        )
        QtWidgets.QMessageBox.information(
            self.owner,
            tr("debug.export_done_title"),
            tr("debug.export_done_message", json_path=json_path, csv_path=csv_path),
        )

    def on_table_click(self, row: int, _col: int) -> None:
        item = self.owner.table.item(row, 0)
        if item is None:
            return
        path = item.data(QtCore.Qt.UserRole)
        if isinstance(path, str) and os.path.exists(path):
            self.owner._load_canvas_image(path)
            self.owner._set_status_for_current_image(path)
            self.owner._update_sample_panel_widgets()
