from __future__ import annotations

import traceback
from typing import Optional

from PySide6 import QtCore

from application import AlgorithmController


class TrainingJobWorker(QtCore.QObject):
    progressChanged = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(
        self,
        algo: AlgorithmController,
        payload: dict,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._algo = algo
        self._payload = dict(payload or {})

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._run_training()
        except Exception as exc:
            detail = traceback.format_exc()
            result = {
                "mode": str(self._payload.get("mode", "") or ""),
                "success_names": [],
                "failure_messages": [f"{exc}\n\n{detail}"],
                "display_rows": [],
                "last_status_message": "",
                "fatal": f"{exc}\n\n{detail}",
            }
        self.finished.emit(result)

    def _run_training(self) -> dict:
        tasks = [dict(task or {}) for task in self._payload.get("tasks", [])]
        mode = str(self._payload.get("mode", "") or "")
        product_dir = str(self._payload.get("product_dir", "") or "")
        score_mode = str(self._payload.get("score_mode", "proto") or "proto")
        margin = float(self._payload.get("margin", 0.02))
        topk = int(self._payload.get("topk", 3))
        selected_item_id = str(self._payload.get("selected_item_id", "") or "")

        success_names: list[str] = []
        failure_messages: list[str] = [
            str(message)
            for message in self._payload.get("failure_messages", [])
            if str(message)
        ]
        display_rows: list[dict[str, object]] = []
        last_status_message = ""
        last_dialog_message = ""
        total = len(tasks)

        for index, task in enumerate(tasks, start=1):
            display_name = str(task.get("display_name", "") or "tool").strip() or "tool"
            algorithm = str(task.get("algorithm", "") or "").strip()
            item_id = str(task.get("item_id", "") or "").strip()
            self.progressChanged.emit(f"training {index}/{total} {display_name}")

            def _progress(
                message: str,
                *,
                _index: int = index,
                _total: int = total,
                _name: str = display_name,
            ) -> None:
                self.progressChanged.emit(f"training {_index}/{_total} {_name}: {message}")

            try:
                self._algo.product_params.algorithm = algorithm
                self._algo.product_params.score_mode = score_mode
                self._algo.product_params.margin = margin
                self._algo.product_params.topk = topk
                result = self._algo.train(
                    list(task.get("ok_files", []) or []),
                    list(task.get("ng_files", []) or []),
                    algorithm=algorithm,
                    product_dir=product_dir,
                    label_names=list(task.get("label_names", []) or []),
                    model_key=task.get("model_key", ""),
                    ok_samples=list(task.get("ok_samples", []) or []),
                    ng_samples=list(task.get("ng_samples", []) or []),
                    progress_callback=_progress,
                    embedding_cache_dir=str(task.get("embedding_cache_dir", "") or ""),
                )
                success_names.append(display_name)
                last_status_message = str(result.status_message or "")
                last_dialog_message = str(result.dialog_message or "")
                if not result.is_embedding and result.result_rows:
                    if item_id == selected_item_id:
                        display_rows = list(result.result_rows)
                    elif not display_rows:
                        display_rows = list(result.result_rows)
            except Exception as exc:
                failure_messages.append(f"{display_name}: {exc}\n{traceback.format_exc()}")

        return {
            "mode": mode,
            "success_names": success_names,
            "failure_messages": failure_messages,
            "display_rows": display_rows,
            "last_status_message": last_status_message,
            "last_dialog_message": last_dialog_message,
            "fatal": "",
        }
