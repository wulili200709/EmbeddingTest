from __future__ import annotations

import ctypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from types import MethodType
from typing import Iterable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image

import algorithms.lazy_api as qr_core
from common import labelme_io
from common.app_paths import packaged_embedding_test_root
from common.camera_roles import camera_role_from_text, normalize_camera_role
from common.safe_io import atomic_write_json, load_json_with_backup
from domain import InspectionItem
from ui.i18n import language_code, tr
from application.runtime_context import RuntimeFrameBatchPrediction
from application.runtime.preview_frame import RuntimePreviewShape
from ui.debug.tool_page import test_execution_controller as test_execution_module

os.environ["LC_SYSTEM_LITE"] = "1"

from ui.debug_main_window import DebugMainWindow


APP_NAME = "LC System Lite"
WINDOWS_APP_ID = "LCSystem.Lite"
HALCON_CROP_SOURCE = "mainlite_halcon_crop"
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LITE_MODEL_NAMES_FILENAME = "lite_model_names.json"


class _LiteWarmupSignals(QtCore.QObject):
    finished = QtCore.Signal(str, str)


class _LiteUiSignals(QtCore.QObject):
    image_loaded = QtCore.Signal(int, str, QtGui.QImage, str)
    reconciliation_finished = QtCore.Signal(int, str, object)


class _LiteFileJobSignals(QtCore.QObject):
    finished = QtCore.Signal(object)


class _LiteImageLoadTask(QtCore.QRunnable):
    """Decode an image away from Qt's UI thread; QPixmap is created on delivery."""

    def __init__(self, signals: _LiteUiSignals, request_id: int, path: str) -> None:
        super().__init__()
        self._signals = signals
        self._request_id = int(request_id)
        self._path = str(path)

    def run(self) -> None:
        image = QtGui.QImage()
        error_text = ""
        try:
            reader = QtGui.QImageReader(self._path)
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                error_text = reader.errorString() or "unable to decode image"
        except Exception as exc:
            error_text = str(exc)
        try:
            self._signals.image_loaded.emit(
                self._request_id,
                self._path,
                image,
                error_text,
            )
        except RuntimeError:
            # The application may close while a slow decoder is still returning.
            pass


def _lite_text(zh_text: str, en_text: str) -> str:
    return zh_text if language_code().lower().startswith("zh") else en_text


def _safe_filename_token(value: object, fallback: str = "image") -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip()).strip("._-")
    return token or fallback


def _safe_export_stem(value: object, fallback: str = "model") -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "").strip())
    text = text.rstrip(" .")
    return text or fallback


def _halcon_crop_metadata(image_path: object) -> Optional[dict[str, str]]:
    path = str(image_path or "").strip()
    if not path:
        return None
    payload = load_json_with_backup(labelme_io.labelme_json_of_image(path), default=None)
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("lc_system_source")
    if not isinstance(metadata, dict):
        return None
    if str(metadata.get("source", "")) != HALCON_CROP_SOURCE:
        return None
    return {str(key): str(value or "") for key, value in metadata.items()}


def _halcon_crop_applies_to_label(
    image_path: object,
    *,
    camera_role: object,
    roi_label: object,
) -> bool:
    """Ordinary images remain shared; HALCON crops belong only to their selected tool."""
    metadata = _halcon_crop_metadata(image_path)
    if metadata is None:
        return True
    role = normalize_camera_role(camera_role, default="cam1")
    return (
        normalize_camera_role(metadata.get("camera_role"), default="cam1") == role
        and str(metadata.get("roi_label", "")).strip() == str(roi_label or "").strip()
    )


def _unique_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists() and not candidate.with_suffix(".json").exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 2
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
        index += 1


def _import_halcon_crop_files(
    source_paths: Iterable[object],
    *,
    product_dir: object,
    camera_role: object,
    item_id: object,
    roi_label: object,
    status: object,
) -> tuple[list[str], list[str]]:
    product_text = str(product_dir or "").strip()
    if not product_text:
        raise ValueError("product directory is not set")
    product_root = Path(product_text).resolve()
    role = normalize_camera_role(camera_role, default="cam1")
    item_token = _safe_filename_token(item_id or roi_label, "roi")
    label = str(roi_label or "").strip()
    if not label:
        raise ValueError("ROI label is empty")
    sample_status = str(status or "").strip().upper()
    if sample_status not in {"OK", "NG"}:
        raise ValueError("sample status must be OK or NG")

    output_dir = product_root / "halcon_crops" / role / item_token / sample_status.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    imported: list[str] = []
    errors: list[str] = []

    for raw_path in source_paths:
        source = Path(str(raw_path or "")).resolve()
        destination: Optional[Path] = None
        try:
            if not source.is_file():
                raise FileNotFoundError(str(source))
            suffix = source.suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                raise ValueError(f"unsupported image format: {suffix or '(none)'}")
            with Image.open(source) as image:
                width, height = image.size
                image.verify()
            if width <= 0 or height <= 0:
                raise ValueError("invalid image size")

            source_stem = _safe_filename_token(source.stem)
            filename = f"{role}__{item_token}__{sample_status.lower()}__{source_stem}{suffix}"
            destination = _unique_destination(output_dir, filename)
            shutil.copy2(source, destination)

            json_path = labelme_io.upsert_labelme_rect(
                str(destination),
                (0, 0, width, height),
                label_name=label,
            )
            payload = load_json_with_backup(json_path, default={})
            if not isinstance(payload, dict):
                payload = {}
            payload["lc_system_source"] = {
                "source": HALCON_CROP_SOURCE,
                "camera_role": role,
                "item_id": str(item_id or label),
                "roi_label": label,
                "status": sample_status,
                "source_name": source.name,
            }
            atomic_write_json(json_path, payload, ensure_ascii=False, indent=2)
            imported.append(str(destination))
        except Exception as exc:
            if destination is not None:
                try:
                    destination.unlink(missing_ok=True)
                    Path(labelme_io.labelme_json_of_image(str(destination))).unlink(missing_ok=True)
                except OSError:
                    pass
            errors.append(f"{source.name or source}: {exc}")

    return imported, errors


class LiteDebugMainWindow(DebugMainWindow):
    """MainLite-only additions; shared Main.py classes and modules stay untouched."""

    def __init__(self) -> None:
        super().__init__(lite_mode=True)
        self._lite_warmup_signals = _LiteWarmupSignals(self)
        self._lite_warmup_signals.finished.connect(self._on_lite_model_warmup_finished)
        self._lite_ui_signals = _LiteUiSignals(self)
        self._lite_ui_signals.image_loaded.connect(self._on_lite_image_loaded)
        self._lite_ui_signals.reconciliation_finished.connect(
            self._on_lite_reconciliation_finished
        )
        self._lite_annotation_cache: dict[
            str,
            tuple[Optional[tuple[int, int]], bool, frozenset[str], Optional[dict[str, str]]],
        ] = {}
        self._lite_reconcile_generation = 0
        self._lite_reconcile_running = False
        self._lite_reconcile_pending = False
        self._lite_image_request_id = 0
        self._lite_image_loading_path = ""
        self._lite_image_pool = QtCore.QThreadPool(self)
        self._lite_image_pool.setMaxThreadCount(1)
        self._lite_warmup_running = False
        self._lite_warmup_pending = False
        self._lite_warmup_ready_signature = ""
        self._lite_warmup_event = threading.Event()
        self._lite_warmup_event.set()
        self._install_lite_halcon_item_reconciliation()
        self._install_halcon_crop_import()
        self._install_lite_ui_performance()
        self._install_lite_calibration_model_name()
        self._install_lite_embedding_analysis()
        self._install_lite_onnx_export()
        self._install_lite_model_warmup()

    @QtCore.Slot(object)
    def _dispatch_lite_training_finished(self, payload: object) -> None:
        """Run MainLite's dynamically wrapped training callback on the UI thread."""
        self.tool_page._on_training_finished(payload)

    def _lite_model_warmup_targets(self) -> tuple[str, list[dict[str, str]]]:
        page = self.tool_page
        product_dir = str(page.session.product_dir or "").strip()
        targets: list[dict[str, str]] = []
        signature_parts = [os.path.normcase(os.path.abspath(product_dir))] if product_dir else []
        for item in list(getattr(page, "inspection_items", []) or []):
            if not bool(getattr(item, "enabled", True)):
                continue
            if not page.algo.is_learning_tool(getattr(item, "algorithm_code", "")):
                continue
            role = normalize_camera_role(getattr(item, "camera_id", ""), default="cam1")
            algorithm = page.algo.resolve_tool_algorithm(getattr(item, "algorithm_code", ""), role)
            model_key = str(getattr(item, "model_key", "") or "").strip()
            model_path = page.algo.embedding_model_path(
                algorithm,
                product_dir,
                model_key=model_key,
            )
            if not os.path.exists(model_path):
                continue
            stat = os.stat(model_path)
            targets.append({
                "algorithm": str(algorithm),
                "model_key": model_key,
                "model_path": model_path,
            })
            signature_parts.extend(
                [str(algorithm), model_key, str(int(stat.st_mtime_ns)), str(int(stat.st_size))]
            )
        return "|".join(signature_parts), targets

    def _install_lite_model_warmup(self) -> None:
        page = self.tool_page
        action_parent = page.lbl_training_validation.parentWidget()
        action_layout = action_parent.layout() if action_parent is not None else None
        if action_layout is not None:
            warmup_badge = QtWidgets.QLabel(action_parent)
            warmup_badge.setMinimumHeight(30)
            warmup_badge.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            warmup_badge.setWordWrap(True)
            action_layout.insertWidget(1, warmup_badge)
            self.lbl_lite_warmup_status = warmup_badge
        self._lite_warmup_visual_state = "checking"
        self._lite_warmup_visual_detail = ""
        self._set_lite_warmup_status("checking")

        original_switch = page.apply_product_switch

        def lite_product_switch(tool_page, name: str) -> None:
            try:
                self._wait_for_lite_model_warmup()
            except RuntimeError as exc:
                QtWidgets.QMessageBox.warning(page, tr("common.info"), str(exc))
                return
            original_switch(name)
            self._lite_warmup_ready_signature = ""
            self._set_lite_warmup_status("checking")
            QtCore.QTimer.singleShot(200, self._start_lite_model_warmup)

        page.apply_product_switch = MethodType(lite_product_switch, page)

        # This wraps the already Lite-specific completion handler, so a newly
        # calibrated model is warmed before the operator's first test.
        original_finished = page._on_training_finished

        def warm_after_training(tool_page, payload: object) -> None:
            original_finished(payload)
            result = dict(payload or {}) if isinstance(payload, dict) else {}
            if list(result.get("success_names", []) or []):
                self._lite_warmup_ready_signature = ""
                QtCore.QTimer.singleShot(200, self._start_lite_model_warmup)

        page._on_training_finished = MethodType(warm_after_training, page)
        QtCore.QTimer.singleShot(350, self._start_lite_model_warmup)

    def _set_lite_warmup_status(self, state: str, detail: str = "") -> None:
        self._lite_warmup_visual_state = str(state or "checking")
        self._lite_warmup_visual_detail = str(detail or "").strip()
        label = getattr(self, "lbl_lite_warmup_status", None)
        if label is None:
            return
        states = {
            "checking": (
                _lite_text("AI模型状态：● 正在检查", "AI model: ● Checking"),
                "#d6d6d6",
                "#3a3a3a",
                "#5c5c5c",
            ),
            "warming": (
                _lite_text("AI模型状态：● 后台预热中……", "AI model: ● Warming up..."),
                "#ffd166",
                "#443b24",
                "#8a7132",
            ),
            "ready": (
                _lite_text("AI模型状态：● 已就绪，可以测试", "AI model: ● Ready for testing"),
                "#72df8e",
                "#233d2b",
                "#3f8052",
            ),
            "untrained": (
                _lite_text("AI模型状态：● 尚未训练", "AI model: ● Not trained"),
                "#ffbd69",
                "#493525",
                "#865b31",
            ),
            "failed": (
                _lite_text("AI模型状态：● 预热失败", "AI model: ● Warmup failed"),
                "#ff7b7b",
                "#482727",
                "#8f4141",
            ),
        }
        text_value, foreground, background, border = states.get(state, states["checking"])
        if detail:
            text_value += f"  ({detail})"
        label.setText(text_value)
        label.setToolTip(detail)
        label.setStyleSheet(
            f"QLabel{{color:{foreground};background:{background};border:1px solid {border};"
            "border-radius:4px;padding:4px 8px;font-weight:bold;font-size:12px;}"
        )

    def _start_lite_model_warmup(self) -> None:
        signature, targets = self._lite_model_warmup_targets()
        if not targets:
            self._lite_warmup_event.set()
            self._set_lite_warmup_status("untrained")
            return
        if signature and signature == self._lite_warmup_ready_signature:
            self._lite_warmup_event.set()
            self._set_lite_warmup_status("ready")
            return
        if self._lite_warmup_running:
            self._lite_warmup_pending = True
            return

        self._lite_warmup_running = True
        self._lite_warmup_pending = False
        self._lite_warmup_event.clear()
        self._set_lite_warmup_status("warming")
        self.tool_page.lbl_status.setText(
            _lite_text(
                "Status: AI 模型后台预热中，完成后首次测试会明显加快……",
                "Status: warming AI model in the background for a faster first test...",
            )
        )

        product_dir = str(self.tool_page.session.product_dir)

        def warmup_worker() -> None:
            error_text = ""
            try:
                warmed_backbones: set[tuple[str, str]] = set()
                for target in targets:
                    model, _message = self.tool_page.algo.load_model_for_algorithm(
                        target["algorithm"],
                        product_dir,
                        model_key=target["model_key"],
                    )
                    if model is None:
                        raise RuntimeError(f"model not loaded: {target['model_path']}")
                    device = str(getattr(model, "device", "cpu") or "cpu")
                    backbone_key = (target["algorithm"], device)
                    if backbone_key in warmed_backbones:
                        continue
                    feat_net = self.tool_page.algo.get_feat_net(
                        target["algorithm"],
                        device,
                    )
                    runner = getattr(feat_net, "run", None)
                    if callable(runner):
                        runner(np.zeros((1, 3, 224, 224), dtype=np.float32))
                    warmed_backbones.add(backbone_key)
            except BaseException as exc:
                error_text = f"{exc}"
            self._lite_warmup_signals.finished.emit(signature, error_text)

        threading.Thread(
            target=warmup_worker,
            name="MainLiteModelWarmup",
            daemon=True,
        ).start()

    @QtCore.Slot(str, str)
    def _on_lite_model_warmup_finished(self, signature: str, error_text: str) -> None:
        self._lite_warmup_running = False
        self._lite_warmup_event.set()
        if error_text:
            self._set_lite_warmup_status("failed", error_text)
            self.tool_page.lbl_status.setText(
                _lite_text(
                    f"Status: AI 模型预热失败，首次测试时将重试：{error_text}",
                    f"Status: AI warmup failed; the first test will retry: {error_text}",
                )
            )
        else:
            self._lite_warmup_ready_signature = signature
            self._set_lite_warmup_status("ready")
            self.tool_page.lbl_status.setText(
                _lite_text("Status: AI 模型已就绪", "Status: AI model ready")
            )
        if self._lite_warmup_pending:
            self._lite_warmup_pending = False
            QtCore.QTimer.singleShot(100, self._start_lite_model_warmup)

    def _wait_for_lite_model_warmup(self) -> None:
        if not self._lite_warmup_running:
            return
        progress = QtWidgets.QProgressDialog(
            _lite_text(
                "AI 模型正在完成首次预热，请稍候……",
                "The AI model is completing its first warmup. Please wait...",
            ),
            "",
            0,
            0,
            self.tool_page,
        )
        progress.setWindowTitle(_lite_text("AI 模型准备中", "Preparing AI Model"))
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        wait_loop = QtCore.QEventLoop(self.tool_page)
        timer = QtCore.QTimer(self.tool_page)
        timer.setInterval(40)
        started_at = time.monotonic()
        timed_out = {"value": False}

        def poll_warmup() -> None:
            if self._lite_warmup_event.is_set():
                wait_loop.quit()
            elif time.monotonic() - started_at >= 120.0:
                timed_out["value"] = True
                wait_loop.quit()

        timer.timeout.connect(poll_warmup)
        timer.start()
        progress.setValue(0)
        wait_loop.exec()
        timer.stop()
        progress.close()
        progress.deleteLater()
        if timed_out["value"]:
            self._set_lite_warmup_status(
                "failed",
                _lite_text("超过120秒", "over 120 seconds"),
            )
            raise RuntimeError(
                _lite_text(
                    "AI 模型预热超过 120 秒仍未完成，请关闭程序后重新启动。",
                    "AI model warmup did not finish within 120 seconds. Close and restart the application.",
                )
            )

    def _lite_annotation_snapshot(
        self,
        image_path: object,
    ) -> tuple[bool, frozenset[str], Optional[dict[str, str]]]:
        """Read each LabelMe file once per file version for Lite list/status queries."""
        path = str(image_path or "").strip()
        if not path:
            return False, frozenset(), None
        json_path = labelme_io.labelme_json_of_image(path)
        cache_key = os.path.normcase(os.path.abspath(json_path))
        try:
            stat = os.stat(json_path)
            signature: Optional[tuple[int, int]] = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            signature = None

        cached = self._lite_annotation_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2], cached[3]

        payload = load_json_with_backup(json_path, default={}) if signature is not None else {}
        if not isinstance(payload, dict):
            payload = {}
        labels = frozenset(
            str(shape.get("label", "")).strip()
            for shape in list(payload.get("shapes", []) or [])
            if isinstance(shape, dict) and str(shape.get("label", "")).strip()
        )
        raw_metadata = payload.get("lc_system_source")
        metadata: Optional[dict[str, str]] = None
        if isinstance(raw_metadata, dict) and str(raw_metadata.get("source", "")) == HALCON_CROP_SOURCE:
            metadata = {str(key): str(value or "") for key, value in raw_metadata.items()}
        snapshot = (signature, signature is not None, labels, metadata)
        self._lite_annotation_cache[cache_key] = snapshot
        return snapshot[1], snapshot[2], snapshot[3]

    def _lite_explicit_crop_metadata(self, image_path: object) -> Optional[dict[str, str]]:
        return self._lite_annotation_snapshot(image_path)[2]

    @staticmethod
    def _scan_lite_halcon_item_specs(
        paths: Iterable[object],
        configured_names: dict[object, object],
    ) -> list[dict[str, str]]:
        specs: dict[tuple[str, str, str], dict[str, str]] = {}
        for path in paths:
            metadata = _halcon_crop_metadata(path)
            if metadata is None:
                continue
            role = normalize_camera_role(metadata.get("camera_role"), default="cam1")
            item_id = str(metadata.get("item_id", "") or metadata.get("roi_label", "")).strip()
            roi_label = str(metadata.get("roi_label", "") or item_id).strip()
            if not item_id or not roi_label:
                continue
            model_key = f"{role}__{item_id}"
            source_stem = Path(str(metadata.get("source_name", "") or item_id)).stem
            inferred_name = re.sub(r"_\d{8}_\d{6}$", "", source_stem).strip(" _-")
            specs[(role, item_id, roi_label)] = {
                "camera_role": role,
                "item_id": item_id,
                "roi_label": roi_label,
                "display_name": str(configured_names.get(model_key, "") or inferred_name or item_id),
            }
        return list(specs.values())

    def _halcon_item_specs(self) -> list[dict[str, str]]:
        page = self.tool_page
        configured_names = load_json_with_backup(
            os.path.join(page.session.product_dir, LITE_MODEL_NAMES_FILENAME),
            default={},
        )
        if not isinstance(configured_names, dict):
            configured_names = {}
        specs: dict[tuple[str, str, str], dict[str, str]] = {}
        for path in list(dict.fromkeys(list(page.train_files) + list(page.test_files))):
            metadata = self._lite_explicit_crop_metadata(path)
            if metadata is None:
                continue
            role = normalize_camera_role(metadata.get("camera_role"), default="cam1")
            item_id = str(metadata.get("item_id", "") or metadata.get("roi_label", "")).strip()
            roi_label = str(metadata.get("roi_label", "") or item_id).strip()
            if not item_id or not roi_label:
                continue
            model_key = f"{role}__{item_id}"
            source_stem = Path(str(metadata.get("source_name", "") or item_id)).stem
            inferred_name = re.sub(r"_\d{8}_\d{6}$", "", source_stem).strip(" _-")
            specs[(role, item_id, roi_label)] = {
                "camera_role": role,
                "item_id": item_id,
                "roi_label": roi_label,
                "display_name": str(configured_names.get(model_key, "") or inferred_name or item_id),
            }
        return list(specs.values())

    def _reconcile_lite_halcon_items(
        self,
        specs: Optional[list[dict[str, str]]] = None,
    ) -> bool:
        """Keep inspection items aligned with the ROI identities stored in HALCON crops."""
        page = self.tool_page
        specs = self._halcon_item_specs() if specs is None else list(specs)
        if not specs:
            return False
        desired_by_role = {
            role: {spec["roi_label"] for spec in specs if spec["camera_role"] == role}
            for role in {spec["camera_role"] for spec in specs}
        }
        existing = list(getattr(page, "inspection_items", []) or [])
        retained: list[InspectionItem] = []
        for item in existing:
            role = normalize_camera_role(getattr(item, "camera_id", ""), default="cam1")
            label = str(getattr(item, "roi_label", "") or "").strip()
            item_id = str(getattr(item, "item_id", "") or "").strip()
            if role in desired_by_role and label == "roi" and "roi" not in desired_by_role[role]:
                continue
            if role in desired_by_role and item_id == "roi" and "roi" not in desired_by_role[role]:
                continue
            retained.append(item)

        changed = len(retained) != len(existing)
        for spec in specs:
            match = next(
                (
                    item
                    for item in retained
                    if normalize_camera_role(getattr(item, "camera_id", ""), default="cam1") == spec["camera_role"]
                    and (
                        str(getattr(item, "item_id", "") or "").strip() == spec["item_id"]
                        or str(getattr(item, "roi_label", "") or "").strip() == spec["roi_label"]
                    )
                ),
                None,
            )
            if match is None:
                retained.append(
                    InspectionItem(
                        item_id=spec["item_id"],
                        display_name=spec["display_name"],
                        camera_id=spec["camera_role"],
                        roi_label=spec["roi_label"],
                        algorithm_code="shared_backbone_register",
                        enabled=True,
                        params={},
                    )
                )
                changed = True
        if changed:
            page.inspection_items = retained
            page._persist_inspection_items()
            page._refresh_inspection_items_table()
        return changed

    def _schedule_lite_halcon_reconciliation(self) -> None:
        self._lite_reconcile_generation += 1
        self._lite_reconcile_pending = True
        if not self._lite_reconcile_running:
            self._start_lite_halcon_reconciliation()

    def _start_lite_halcon_reconciliation(self) -> None:
        if self._lite_reconcile_running or not self._lite_reconcile_pending:
            return
        page = self.tool_page
        generation = self._lite_reconcile_generation
        product_dir = os.path.normcase(os.path.abspath(str(page.session.product_dir or "")))
        paths = list(dict.fromkeys(list(page.train_files) + list(page.test_files)))
        configured_names = load_json_with_backup(
            os.path.join(page.session.product_dir, LITE_MODEL_NAMES_FILENAME),
            default={},
        )
        if not isinstance(configured_names, dict):
            configured_names = {}
        self._lite_reconcile_running = True
        self._lite_reconcile_pending = False

        def scan_worker() -> None:
            try:
                specs = self._scan_lite_halcon_item_specs(paths, configured_names)
            except Exception:
                specs = []
            try:
                self._lite_ui_signals.reconciliation_finished.emit(
                    generation,
                    product_dir,
                    specs,
                )
            except RuntimeError:
                pass

        threading.Thread(
            target=scan_worker,
            name="MainLiteSampleMetadata",
            daemon=True,
        ).start()

    @QtCore.Slot(int, str, object)
    def _on_lite_reconciliation_finished(
        self,
        generation: int,
        product_dir: str,
        specs: object,
    ) -> None:
        self._lite_reconcile_running = False
        current_dir = os.path.normcase(
            os.path.abspath(str(self.tool_page.session.product_dir or ""))
        )
        if generation == self._lite_reconcile_generation and product_dir == current_dir:
            self._reconcile_lite_halcon_items(
                list(specs) if isinstance(specs, list) else []
            )
        if self._lite_reconcile_pending:
            self._start_lite_halcon_reconciliation()

    def _install_lite_halcon_item_reconciliation(self) -> None:
        page = self.tool_page
        original_reload = page._reload_inspection_items

        def lite_reload(tool_page) -> None:
            original_reload()
            self._schedule_lite_halcon_reconciliation()

        page._reload_inspection_items = MethodType(lite_reload, page)
        self._schedule_lite_halcon_reconciliation()

    def _lite_crop_context(self, image_path: object) -> Optional[dict[str, str]]:
        """Return explicit or safe inferred full-image ROI metadata for Lite testing."""
        metadata = self._lite_explicit_crop_metadata(image_path)
        if metadata is not None:
            return metadata
        path = str(image_path or "").strip()
        if not path:
            return None
        json_exists, labels, _metadata = self._lite_annotation_snapshot(path)
        if json_exists and labels:
            return None
        role = camera_role_from_text(os.path.basename(path), default="")
        if not role:
            return None
        known_roles = {
            normalize_camera_role(getattr(item, "camera_id", ""), default="cam1")
            for item in list(getattr(self.tool_page, "inspection_items", []) or [])
            if bool(getattr(item, "enabled", True))
        }
        if role not in known_roles:
            return None
        candidates = [
            item
            for item in list(getattr(self.tool_page, "inspection_items", []) or [])
            if bool(getattr(item, "enabled", True))
            and normalize_camera_role(getattr(item, "camera_id", ""), default="cam1") == role
            and self.tool_page.algo.is_learning_tool(getattr(item, "algorithm_code", ""))
        ]
        if len(candidates) != 1:
            return None
        item = candidates[0]
        return {
            "source": "mainlite_external_crop",
            "camera_role": role,
            "item_id": str(getattr(item, "item_id", "") or ""),
            "roi_label": str(getattr(item, "roi_label", "") or ""),
            "status": "",
            "source_name": os.path.basename(path),
        }

    def _install_halcon_crop_import(self) -> None:
        page = self.tool_page
        train_tab = page.tabs.widget(0)
        train_layout = train_tab.layout() if train_tab is not None else None
        if train_layout is None:
            return

        import_panel = QtWidgets.QFrame(train_tab)
        import_panel.setObjectName("liteImportPanel")
        import_panel.setStyleSheet(
            "QFrame#liteImportPanel{background:#303030;border:1px solid #4b4b4b;"
            "border-radius:4px;}"
            "QComboBox{background:#262626;color:#eeeeee;border:1px solid #5a5a5a;"
            "border-radius:3px;padding:4px 8px;min-height:22px;}"
            "QComboBox:hover{border-color:#777777;}"
            "QComboBox:focus{border-color:#2d8cff;}"
        )
        import_layout = QtWidgets.QHBoxLayout(import_panel)
        import_layout.setContentsMargins(6, 6, 6, 6)
        import_layout.setSpacing(6)
        import_label = QtWidgets.QLabel(import_panel)
        import_label.setStyleSheet("color:#d8d8d8;border:none;")
        import_mode = QtWidgets.QComboBox(import_panel)
        import_button = QtWidgets.QPushButton(import_panel)
        import_button.setStyleSheet(page.btn_import_train.styleSheet())
        import_button.setMinimumWidth(132)
        import_layout.addWidget(import_label)
        import_layout.addWidget(import_mode, 1)
        import_layout.addWidget(import_button)
        train_layout.insertWidget(1, import_panel)

        self.lite_import_panel = import_panel
        self.lbl_lite_import_mode = import_label
        self.cmb_lite_import_mode = import_mode
        self.btn_lite_import_execute = import_button
        self.btn_import_halcon_crop = import_button

        def run_selected_import() -> None:
            if import_mode.currentData() == "external":
                page.sample_list_controller.add_images_to("TRAIN")
            else:
                self._show_halcon_crop_import()

        import_button.clicked.connect(run_selected_import)

        # The original two-column action grid remains useful for sample actions,
        # but the old standalone import button is replaced by the selector above.
        page.btn_import_train.hide()
        action_grid = None
        for index in range(train_layout.count()):
            candidate = train_layout.itemAt(index).layout()
            if isinstance(candidate, QtWidgets.QGridLayout):
                action_grid = candidate
                break
        if action_grid is not None:
            for widget in (page.btn_import_train, page.btn_train_to_test, page.btn_sample_annotation, page.btn_del_ok):
                action_grid.removeWidget(widget)
            action_grid.setColumnStretch(0, 1)
            action_grid.setColumnStretch(1, 1)
            action_grid.setColumnStretch(2, 1)
            action_grid.addWidget(page.btn_train_to_test, 0, 0)
            action_grid.addWidget(page.btn_sample_annotation, 0, 1)
            action_grid.addWidget(page.btn_del_ok, 0, 2)
        self._refresh_lite_text()
        self._install_lite_training_filters()
        self._install_lite_test_execution()

    def _run_lite_file_job(
        self,
        title: str,
        message: str,
        job,
    ):
        """Run bulk file I/O in a worker while a modal progress UI pumps events."""
        progress = QtWidgets.QProgressDialog(message, "", 0, 0, self)
        progress.setWindowTitle(title)
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        signals = _LiteFileJobSignals(self)
        wait_loop = QtCore.QEventLoop(self)
        outcome: dict[str, object] = {}

        def finish(result: object) -> None:
            outcome["result"] = result
            wait_loop.quit()

        def worker() -> None:
            try:
                result = (True, job())
            except BaseException as exc:
                result = (False, exc)
            signals.finished.emit(result)

        signals.finished.connect(finish)
        progress.show()
        threading.Thread(target=worker, name="MainLiteFileImport", daemon=True).start()
        wait_loop.exec()
        progress.close()
        progress.deleteLater()
        signals.deleteLater()
        success, value = outcome.get("result", (False, RuntimeError("file job stopped")))
        if not success:
            if isinstance(value, BaseException):
                raise value
            raise RuntimeError(str(value))
        return value

    def _install_lite_ui_performance(self) -> None:
        """Keep large sample collections from monopolizing Qt's event loop."""
        page = self.tool_page
        controller = page.sample_list_controller
        for list_widget in (page.ok_list, page.ng_list, page.test_list):
            list_widget.setUniformItemSizes(True)
            list_widget.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
            list_widget.setBatchSize(128)

        self._lite_list_refresh_generation = 0
        self._lite_list_refresh_active = False
        self._lite_list_refresh_state: list[dict[str, object]] = []
        self._lite_pending_list_selection = ""
        self._lite_list_refresh_timer = QtCore.QTimer(self)
        self._lite_list_refresh_timer.setInterval(0)

        def item_path(item: Optional[QtWidgets.QListWidgetItem]) -> str:
            if item is None:
                return ""
            return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.toolTip() or "")

        def list_title(role: str, *, loading: bool = False) -> str:
            title = tr("debug.image_list")
            if role:
                title = (
                    f"{title}（{role}）"
                    if language_code().lower().startswith("zh")
                    else f"{title} ({role})"
                )
            if loading:
                title += _lite_text(" · 正在加载…", " · Loading...")
            return title

        def finish_list_refresh() -> None:
            self._lite_list_refresh_timer.stop()
            self._lite_list_refresh_active = False
            role = str(getattr(self, "_lite_list_refresh_role", "") or "")
            if hasattr(page, "lbl_images_section"):
                page.lbl_images_section.setText(list_title(role))
            page._update_sample_panel_widgets()
            pending_path = self._lite_pending_list_selection
            self._lite_pending_list_selection = ""
            if pending_path:
                original_select_path(pending_path)

        def append_list_batch() -> None:
            states = self._lite_list_refresh_state
            while states and int(states[0]["cursor"]) >= len(states[0]["files"]):
                completed = states.pop(0)
                selected_path = str(completed.get("selected_path", "") or "")
                selected_row = int(completed.get("selected_row", -1))
                if selected_path and selected_row >= 0:
                    list_widget = completed["widget"]
                    blocker = QtCore.QSignalBlocker(list_widget)
                    list_widget.setCurrentRow(selected_row)
                    del blocker
            if not states:
                finish_list_refresh()
                return

            state = states[0]
            list_widget = state["widget"]
            files = state["files"]
            cursor = int(state["cursor"])
            end = min(len(files), cursor + 128)
            deadline = time.perf_counter() + 0.010
            list_widget.setUpdatesEnabled(False)
            blocker = QtCore.QSignalBlocker(list_widget)
            try:
                while cursor < end:
                    path = str(files[cursor])
                    item = QtWidgets.QListWidgetItem(
                        page._sample_item_display_text(
                            path,
                            str(state["sample_kind"]),
                            str(state["role"]),
                        )
                    )
                    item.setToolTip(path)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                    list_widget.addItem(item)
                    if path == state["selected_path"]:
                        state["selected_row"] = cursor
                    cursor += 1
                    if cursor % 16 == 0 and time.perf_counter() >= deadline:
                        break
                state["cursor"] = cursor
            finally:
                del blocker
                list_widget.setUpdatesEnabled(True)

        self._lite_list_refresh_timer.timeout.connect(append_list_batch)

        def lite_refresh_lists(tool_page) -> None:
            self._lite_list_refresh_generation += 1
            self._lite_list_refresh_timer.stop()
            role = tool_page._selected_image_list_camera_role()
            self._lite_list_refresh_role = role
            train_paths = list(tool_page._sample_paths_for_kind("train", role))
            test_paths = list(tool_page._sample_paths_for_kind("test", role))
            ordered = (
                [(tool_page.ok_list, train_paths, "train"), (tool_page.test_list, test_paths, "test")]
                if tool_page.tabs.currentIndex() == 0
                else [(tool_page.test_list, test_paths, "test"), (tool_page.ok_list, train_paths, "train")]
            )
            states: list[dict[str, object]] = []
            for list_widget, files, sample_kind in ordered:
                selected_path = item_path(list_widget.currentItem())
                blocker = QtCore.QSignalBlocker(list_widget)
                list_widget.setUpdatesEnabled(False)
                try:
                    list_widget.clear()
                finally:
                    list_widget.setUpdatesEnabled(True)
                    del blocker
                states.append(
                    {
                        "widget": list_widget,
                        "files": files,
                        "sample_kind": sample_kind,
                        "role": role,
                        "cursor": 0,
                        "selected_path": selected_path,
                        "selected_row": -1,
                    }
                )
            blocker = QtCore.QSignalBlocker(tool_page.ng_list)
            tool_page.ng_list.clear()
            del blocker
            if hasattr(tool_page, "tabs"):
                tool_page.tabs.setTabText(0, f"{tr('debug.train_samples')} ({len(train_paths)})")
                tool_page.tabs.setTabText(1, f"{tr('debug.test_samples')} ({len(test_paths)})")
            if hasattr(tool_page, "lbl_images_section"):
                tool_page.lbl_images_section.setText(list_title(role, loading=True))
            self._lite_list_refresh_state = states
            self._lite_list_refresh_active = True
            self._lite_list_refresh_timer.start()

        original_localize_files = controller._localize_training_files

        def lite_localize_files(sample_controller, files: list[str]):
            if len(files) < 8:
                return original_localize_files(files)
            return self._run_lite_file_job(
                _lite_text("导入训练图片", "Importing Training Images"),
                _lite_text(
                    f"正在复制并检查 {len(files)} 张图片，请稍候…",
                    f"Copying and checking {len(files)} images. Please wait...",
                ),
                lambda: original_localize_files(files),
            )

        original_select_path = controller.select_path_in_current_tab

        def lite_select_path(sample_controller, path: str) -> None:
            if self._lite_list_refresh_active:
                self._lite_pending_list_selection = str(path or "")
                return
            original_select_path(path)

        original_show_path = controller.show_selected_image_path

        def lite_show_path(sample_controller, path: Optional[str]) -> None:
            if not path:
                self._lite_image_request_id += 1
                self._lite_image_loading_path = ""
                self._lite_image_pool.clear()
                original_show_path(path)
                return
            path = str(path)
            if page.canvas.image_path() == path:
                # Lite training is tool-oriented. Keep the selected tool when
                # operators review samples so calibration remains available.
                page._set_status_for_current_image(path)
                page._update_sample_panel_widgets()
                return
            self._lite_image_request_id += 1
            request_id = self._lite_image_request_id
            self._lite_image_loading_path = path
            self._lite_image_pool.clear()
            page.lbl_status.setText(
                _lite_text(
                    f"状态：正在加载图片 {os.path.basename(path)}…",
                    f"Status: loading image {os.path.basename(path)}...",
                )
            )
            self._lite_image_pool.start(
                _LiteImageLoadTask(self._lite_ui_signals, request_id, path)
            )

        page._refresh_lists = MethodType(lite_refresh_lists, page)
        controller._localize_training_files = MethodType(lite_localize_files, controller)
        controller.select_path_in_current_tab = MethodType(lite_select_path, controller)
        controller.show_selected_image_path = MethodType(lite_show_path, controller)

    @QtCore.Slot(int, str, QtGui.QImage, str)
    def _on_lite_image_loaded(
        self,
        request_id: int,
        path: str,
        image: QtGui.QImage,
        error_text: str,
    ) -> None:
        if request_id != self._lite_image_request_id:
            return
        page = self.tool_page
        selected_path = page.sample_list_controller.current_selected_path()
        if str(selected_path or "") != str(path):
            return
        self._lite_image_loading_path = ""
        if error_text or image.isNull():
            page.lbl_status.setText(
                _lite_text(
                    f"状态：图片加载失败：{error_text or os.path.basename(path)}",
                    f"Status: image load failed: {error_text or os.path.basename(path)}",
                )
            )
            page._update_sample_panel_widgets()
            return
        try:
            page.canvas.set_image(path, pixmap=QtGui.QPixmap.fromImage(image))
            page._load_shape_for_label(path, page._current_label())
            page._set_status_for_current_image(path)
        except Exception as exc:
            page.lbl_status.setText(
                _lite_text(f"状态：图片显示失败：{exc}", f"Status: image display failed: {exc}")
            )
        page._update_sample_panel_widgets()

    def _refresh_lite_text(self) -> None:
        import_label = getattr(self, "lbl_lite_import_mode", None)
        if import_label is not None:
            import_label.setText(_lite_text("导入方式", "Import mode"))
        import_mode = getattr(self, "cmb_lite_import_mode", None)
        if import_mode is not None:
            current_mode = str(import_mode.currentData() or "halcon")
            blocker = QtCore.QSignalBlocker(import_mode)
            import_mode.clear()
            import_mode.addItem(_lite_text("HALCON 小图（整图 ROI）", "HALCON crop (full-image ROI)"), "halcon")
            import_mode.addItem(_lite_text("普通外部图片", "Regular external image"), "external")
            selected_index = import_mode.findData(current_mode)
            import_mode.setCurrentIndex(max(0, selected_index))
            del blocker
            import_mode.setToolTip(
                _lite_text(
                    "HALCON 小图会选择检测项和 OK/NG；普通外部图片保留原始模板/ROI流程。",
                    "HALCON crops select a tool and OK/NG; regular images keep the original template/ROI workflow.",
                )
            )
        button = getattr(self, "btn_lite_import_execute", None)
        if button is not None:
            button.setText(_lite_text("选择图片并导入", "Choose and Import"))
            button.setToolTip(
                _lite_text(
                    "按左侧选择的方式导入训练图片",
                    "Import training images using the selected mode",
                )
            )
        analysis_button = getattr(self.tool_page, "btn_embedding_analysis", None)
        if analysis_button is not None:
            analysis_button.setText(_lite_text("特征分析图", "Feature Analysis"))
            analysis_button.setToolTip(
                _lite_text(
                    "查看当前检测项模型的特征分布、相似度和样本分析图",
                    "View feature distribution, similarity, and sample analysis charts for the current tool model",
                )
            )
        if hasattr(self, "lbl_lite_warmup_status"):
            self._set_lite_warmup_status(
                getattr(self, "_lite_warmup_visual_state", "checking"),
                getattr(self, "_lite_warmup_visual_detail", ""),
            )

    def _change_language(self, language: str) -> None:
        super()._change_language(language)
        self._refresh_lite_text()

    def _trainable_items_for_current_camera(self) -> list[object]:
        page = self.tool_page
        role = page.current_camera_role()
        return [
            item
            for item in list(getattr(page, "inspection_items", []) or [])
            if bool(getattr(item, "enabled", True))
            and normalize_camera_role(getattr(item, "camera_id", ""), default="cam1") == role
            and str(getattr(item, "roi_label", "")).strip()
            and not page.algo.is_measurement_tool(getattr(item, "algorithm_code", ""))
        ]

    def _show_halcon_crop_import(self) -> None:
        page = self.tool_page
        require_permission = getattr(self, "_require_permission", None)
        if callable(require_permission) and not require_permission("sample.manage", "导入 HALCON 小图"):
            return
        inspection_items = self._trainable_items_for_current_camera()

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(_lite_text("导入 HALCON 小图", "Import HALCON Crops"))
        dialog.setModal(True)
        dialog.resize(430, 180)
        layout = QtWidgets.QVBoxLayout(dialog)
        hint = QtWidgets.QLabel(
            _lite_text(
                "图片已经由 HALCON 裁切。导入后整张小图会作为所选检测项的 ROI，且不会执行模板匹配。",
                "The images were already cropped by HALCON. The full image becomes the selected tool's ROI; no template matching is run.",
            ),
            dialog,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QtWidgets.QFormLayout()
        tool_combo = QtWidgets.QComboBox(dialog)
        for item in inspection_items:
            display_name = str(getattr(item, "display_name", "") or getattr(item, "roi_label", ""))
            roi_label = str(getattr(item, "roi_label", ""))
            tool_combo.addItem(f"{display_name} ({roi_label})", item)
        if inspection_items:
            tool_combo.insertSeparator(tool_combo.count())
        tool_combo.addItem(_lite_text("新建检测项...", "Create inspection item..."), "__new__")
        new_item_name = QtWidgets.QLineEdit(dialog)
        new_item_name.setPlaceholderText(_lite_text("例如：Damage、Cover、Flash", "e.g. Damage, Cover, Flash"))
        status_combo = QtWidgets.QComboBox(dialog)
        status_combo.addItem("OK", "OK")
        status_combo.addItem("NG", "NG")
        form.addRow(_lite_text("检测项：", "Inspection item:"), tool_combo)
        form.addRow(_lite_text("新检测名称：", "New item name:"), new_item_name)
        form.addRow(_lite_text("样本类别：", "Sample class:"), status_combo)
        layout.addLayout(form)

        def sync_new_item_editor() -> None:
            creating = tool_combo.currentData() == "__new__"
            new_item_name.setEnabled(creating)
            if creating:
                new_item_name.setFocus()

        tool_combo.currentIndexChanged.connect(lambda _index: sync_new_item_editor())
        if not inspection_items:
            tool_combo.setCurrentIndex(tool_combo.count() - 1)
        sync_new_item_editor()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        def accept_import_options() -> None:
            if tool_combo.currentData() == "__new__" and not new_item_name.text().strip():
                QtWidgets.QMessageBox.warning(
                    dialog,
                    tr("common.info"),
                    _lite_text("请输入检测项名称。", "Enter an inspection item name."),
                )
                return
            dialog.accept()

        buttons.accepted.connect(accept_import_options)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selected_item = tool_combo.currentData()
        create_new_item = selected_item == "__new__"
        inspection_item = (
            self._make_lite_inspection_item(new_item_name.text().strip())
            if create_new_item
            else selected_item
        )
        if inspection_item is None:
            return
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            _lite_text("选择 HALCON 裁切小图", "Select HALCON Crop Images"),
            "",
            IMAGE_FILTER,
        )
        if not files:
            return

        role = page.current_camera_role()
        roi_label = str(getattr(inspection_item, "roi_label", ""))
        status = str(status_combo.currentData() or "OK")
        product_dir = str(page.session.product_dir)
        item_id = str(getattr(inspection_item, "item_id", ""))
        imported, errors = self._run_lite_file_job(
            _lite_text("导入 HALCON 小图", "Importing HALCON Crops"),
            _lite_text(
                f"正在复制并创建 {len(files)} 张图片的标注，请稍候…",
                f"Copying and annotating {len(files)} images. Please wait...",
            ),
            lambda: _import_halcon_crop_files(
                files,
                product_dir=product_dir,
                camera_role=role,
                item_id=item_id,
                roi_label=roi_label,
                status=status,
            ),
        )
        annotation_key = page.roi_annotations.roi_key(role, roi_label)
        for path in imported:
            normalized_path = os.path.normpath(path)
            annotations = dict(page._sample_roi_annotations_by_path.get(normalized_path, {}))
            annotations[annotation_key] = status
            page._sample_roi_annotations_by_path[normalized_path] = annotations
        if imported:
            page._save_sample_roi_annotations()

        if imported:
            if create_new_item:
                page.inspection_items.append(inspection_item)
                page._persist_inspection_items()
                page._refresh_inspection_items_table()
            page.train_files.extend(imported)
            page.train_files = sorted(list(dict.fromkeys(page.train_files)))
            page.tabs.setCurrentIndex(0)
            page._refresh_lists()
            page._clear_training_roi_review_state(role)
            page._save_session()
            page.sample_list_controller.select_path_in_current_tab(imported[-1])

        message = _lite_text(
            f"HALCON 小图导入完成：成功 {len(imported)} 张，失败 {len(errors)} 张。",
            f"HALCON crop import finished: {len(imported)} imported, {len(errors)} failed.",
        )
        if errors:
            message += "\n\n" + "\n".join(errors[:10])
            QtWidgets.QMessageBox.warning(self, tr("common.done"), message)
        else:
            QtWidgets.QMessageBox.information(self, tr("common.done"), message)

    def _make_lite_inspection_item(self, display_name: str) -> InspectionItem:
        page = self.tool_page
        existing_ids = {
            str(getattr(item, "item_id", "") or "").strip()
            for item in list(getattr(page, "inspection_items", []) or [])
        }
        existing_labels = {
            str(getattr(item, "roi_label", "") or "").strip()
            for item in list(getattr(page, "inspection_items", []) or [])
        }
        index = 1
        while f"roi{index}" in existing_ids or f"roi{index}" in existing_labels:
            index += 1
        roi_label = f"roi{index}"
        return InspectionItem(
            item_id=roi_label,
            display_name=str(display_name or roi_label).strip() or roi_label,
            camera_id=page.current_camera_role(),
            roi_label=roi_label,
            algorithm_code="shared_backbone_register",
            enabled=True,
            params={},
        )

    def _install_lite_training_filters(self) -> None:
        page = self.tool_page
        builder = page.training_controller.task_builder
        original_train_paths = builder.train_sample_paths_for_role
        original_ensure_reviewed = page._ensure_training_roi_reviewed

        def crop_applies(path: str, role: str, label: str) -> bool:
            metadata = self._lite_crop_context(path)
            if metadata is None:
                return True
            return (
                normalize_camera_role(metadata.get("camera_role"), default="cam1") == role
                and str(metadata.get("roi_label", "")).strip() == label
            )

        def lite_has_geometry(tool_page, path: str, label_name: str) -> bool:
            _json_exists, labels, _metadata = self._lite_annotation_snapshot(path)
            if str(label_name or "").strip() in labels:
                return True
            metadata = self._lite_crop_context(path)
            return bool(
                metadata is not None
                and str(metadata.get("roi_label", "")).strip() == str(label_name or "").strip()
            )

        page._path_has_roi_geometry = MethodType(lite_has_geometry, page)

        def lite_sample_groups(
            task_builder,
            camera_role: object = None,
            *,
            roi_label: object = None,
        ) -> tuple[list[str], list[str], list[str]]:
            role = normalize_camera_role(camera_role or page.current_camera_role(), default="cam1")
            label = str(roi_label or "").strip()
            candidate_paths = [
                path
                for path in original_train_paths(role)
                if not label or crop_applies(path, role, label)
            ]
            if not label:
                return [], [], candidate_paths
            ok_files: list[str] = []
            ng_files: list[str] = []
            for path in candidate_paths:
                if not page._path_has_roi_geometry(path, label):
                    continue
                sample_status = page._sample_roi_status_for_path(path, role, label)
                if sample_status == "OK":
                    ok_files.append(path)
                elif sample_status == "NG":
                    ng_files.append(path)
            return ok_files, ng_files, candidate_paths

        builder.sample_groups_for_role = MethodType(lite_sample_groups, builder)

        def lite_counts(
            tool_page,
            roi_label: str,
            camera_role: object = None,
            *,
            paths: Optional[list[str]] = None,
        ) -> tuple[int, int, int]:
            role = normalize_camera_role(camera_role or tool_page.current_camera_role(), default="cam1")
            source_paths = list(paths) if paths is not None else list(original_train_paths(role))
            target_paths = [
                path
                for path in source_paths
                if crop_applies(path, role, roi_label)
            ]
            ok_count = 0
            ng_count = 0
            unset_count = 0
            for path in target_paths:
                if not tool_page._path_has_roi_geometry(path, roi_label):
                    unset_count += 1
                    continue
                sample_status = tool_page._sample_roi_status_for_path(path, role, roi_label)
                if sample_status == "OK":
                    ok_count += 1
                elif sample_status == "NG":
                    ng_count += 1
                else:
                    unset_count += 1
            return ok_count, ng_count, unset_count

        page._sample_annotation_counts_for_roi = MethodType(lite_counts, page)
        original_state = page._sample_annotation_state_for_path
        original_progress = page._sample_annotation_progress_for_path

        def lite_state(tool_page, path: str, camera_role: object = None) -> str:
            metadata = self._lite_crop_context(path)
            if metadata is None:
                return original_state(path, camera_role)
            role = normalize_camera_role(camera_role or tool_page.current_camera_role(), default="cam1")
            label = str(metadata.get("roi_label", "")).strip()
            if not label or not tool_page._path_has_roi_geometry(path, label):
                return tr("sample.missing_roi")
            return (
                tr("sample.complete")
                if tool_page._sample_roi_status_for_path(path, role, label) in {"OK", "NG"}
                else tr("sample.unset")
            )

        def lite_progress(tool_page, path: str, camera_role: object = None) -> tuple[int, int]:
            metadata = self._lite_crop_context(path)
            if metadata is None:
                return original_progress(path, camera_role)
            role = normalize_camera_role(camera_role or tool_page.current_camera_role(), default="cam1")
            label = str(metadata.get("roi_label", "")).strip()
            complete = bool(
                label
                and tool_page._path_has_roi_geometry(path, label)
                and tool_page._sample_roi_status_for_path(path, role, label) in {"OK", "NG"}
            )
            return (1 if complete else 0, 1)

        page._sample_annotation_state_for_path = MethodType(lite_state, page)
        page._sample_annotation_progress_for_path = MethodType(lite_progress, page)

        def lite_ensure_reviewed(
            tool_page,
            camera_role: object,
            *,
            action_name: str,
            action_key: str,
            confirmation_token: str = "",
        ) -> bool:
            role = normalize_camera_role(camera_role or tool_page.current_camera_role(), default="cam1")
            normal_paths = [path for path in original_train_paths(role) if self._lite_crop_context(path) is None]
            if not normal_paths:
                return True

            previous_train_paths = builder.train_sample_paths_for_role

            def normal_train_paths(_builder, requested_role: object = None) -> list[str]:
                requested = normalize_camera_role(requested_role or role, default="cam1")
                return [
                    path
                    for path in original_train_paths(requested)
                    if self._lite_crop_context(path) is None
                ]

            builder.train_sample_paths_for_role = MethodType(normal_train_paths, builder)
            try:
                return original_ensure_reviewed(
                    role,
                    action_name=action_name,
                    action_key=action_key,
                    confirmation_token=confirmation_token,
                )
            finally:
                builder.train_sample_paths_for_role = previous_train_paths

        page._ensure_training_roi_reviewed = MethodType(lite_ensure_reviewed, page)

    def _install_lite_test_execution(self) -> None:
        """Test HALCON crops as full-image ROIs without changing normal-image runtime."""
        page = self.tool_page
        controller = page.test_execution_controller
        original_execute_image = controller.execute_image
        original_target_items = controller.target_inspection_items

        def lite_execute_image(test_controller, path: str) -> dict[str, object]:
            metadata = self._lite_crop_context(path)
            if metadata is None:
                return original_execute_image(path)

            self._wait_for_lite_model_warmup()

            crop_role = normalize_camera_role(metadata.get("camera_role"), default="cam1")
            crop_item_id = str(metadata.get("item_id", "")).strip()
            crop_label = str(metadata.get("roi_label", "")).strip()
            matching_items = [
                item
                for item in list(getattr(page, "inspection_items", []) or [])
                if bool(getattr(item, "enabled", True))
                and normalize_camera_role(getattr(item, "camera_id", ""), default="cam1") == crop_role
                and (
                    (crop_item_id and str(getattr(item, "item_id", "")).strip() == crop_item_id)
                    or (crop_label and str(getattr(item, "roi_label", "")).strip() == crop_label)
                )
            ]
            if not matching_items:
                raise RuntimeError(
                    _lite_text(
                        f"HALCON 小图对应的检测项不存在或未启用：{crop_item_id or crop_label}",
                        f"The HALCON crop inspection item is missing or disabled: {crop_item_id or crop_label}",
                    )
                )

            image_path = str(path)
            base_context = test_execution_module.ProductRuntimeContext(page.session, page.algo)

            class LiteCropRuntimeContext:
                def __getattr__(self, name: str):
                    return getattr(base_context, name)

                def predict_items_batch_from_frame(
                    self,
                    image_bgr,
                    *,
                    camera_role: str,
                    items: list[InspectionItem],
                    feat_net=None,
                ) -> RuntimeFrameBatchPrediction:
                    height, width = image_bgr.shape[:2]
                    rows: list[dict[str, object]] = []
                    for item in items:
                        label = str(getattr(item, "roi_label", "") or crop_label).strip() or "roi1"
                        resolved_algorithm = page.algo.resolve_tool_algorithm(
                            getattr(item, "algorithm_code", ""),
                            getattr(item, "camera_id", camera_role),
                        )
                        if page.algo.is_embedding_algorithm(resolved_algorithm):
                            page.load_embedding_model(
                                resolved_algorithm,
                                model_key=getattr(item, "model_key", ""),
                            )
                            if page.algo.model is None:
                                expected_model = page.algo.embedding_model_path(
                                    resolved_algorithm,
                                    page.session.product_dir,
                                    model_key=getattr(item, "model_key", ""),
                                )
                                raise RuntimeError(
                                    _lite_text(
                                        f"检测模型加载失败：{expected_model}",
                                        f"Failed to load inspection model: {expected_model}",
                                    )
                                )
                        result_box: dict[str, object] = {}
                        completed = threading.Event()

                        def run_prediction() -> None:
                            try:
                                result_box["prediction"] = page.algo.predict_image(
                                    image_path,
                                    labels=[label],
                                    feat_net=feat_net,
                                    roi=(0, 0, int(width), int(height)),
                                    match_ms=0.0,
                                    algorithm_override=resolved_algorithm,
                                    model_key_override=getattr(item, "model_key", ""),
                                    params_override=dict(getattr(item, "params", {}) or {}),
                                )
                            except BaseException as exc:
                                result_box["error"] = exc
                                result_box["traceback"] = traceback.format_exc()
                            finally:
                                completed.set()

                        progress = QtWidgets.QProgressDialog(
                            _lite_text(
                                "正在加载模型并检测，首次运行可能需要十几秒……",
                                "Loading the model and inspecting; the first run may take several seconds...",
                            ),
                            "",
                            0,
                            0,
                            page,
                        )
                        progress.setWindowTitle(_lite_text("AI 检测中", "AI Inspection"))
                        progress.setCancelButton(None)
                        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                        progress.setMinimumDuration(250)
                        page.lbl_status.setText(
                            _lite_text(
                                "Status: 正在加载模型并执行 AI 检测……",
                                "Status: loading model and running AI inspection...",
                            )
                        )

                        worker = threading.Thread(
                            target=run_prediction,
                            name="MainLiteInference",
                            daemon=True,
                        )
                        worker.start()
                        wait_loop = QtCore.QEventLoop(page)
                        poll_timer = QtCore.QTimer(page)
                        poll_timer.setInterval(40)
                        started_at = time.monotonic()
                        timed_out = {"value": False}

                        def poll_prediction() -> None:
                            if completed.is_set():
                                wait_loop.quit()
                            elif time.monotonic() - started_at >= 120.0:
                                timed_out["value"] = True
                                wait_loop.quit()

                        poll_timer.timeout.connect(poll_prediction)
                        poll_timer.start()
                        progress.setValue(0)
                        wait_loop.exec()
                        poll_timer.stop()
                        progress.close()
                        progress.deleteLater()

                        if timed_out["value"]:
                            raise RuntimeError(
                                _lite_text(
                                    "AI 检测超过 120 秒仍未完成，请关闭程序后重试；若持续出现，请重新生成 ORT 模型。",
                                    "AI inspection did not finish within 120 seconds. Restart and retry; regenerate the ORT model if it persists.",
                                )
                            )
                        error = result_box.get("error")
                        if isinstance(error, BaseException):
                            detail = str(result_box.get("traceback", "") or "")
                            raise RuntimeError(f"{error}\n{detail}".strip()) from error
                        prediction = result_box.get("prediction")
                        if prediction is None:
                            raise RuntimeError(_lite_text("AI 检测没有返回结果。", "AI inspection returned no result."))
                        to_dict = getattr(prediction, "to_dict", None)
                        rows.append(dict(to_dict() if callable(to_dict) else prediction))

                    runtime_label = crop_label or str(getattr(items[0], "roi_label", "") or "roi1")
                    full_image_roi = RuntimePreviewShape(
                        label=runtime_label,
                        shape_type="rectangle",
                        points=((0.0, 0.0), (float(width), float(height))),
                    )
                    return RuntimeFrameBatchPrediction(
                        rows=rows,
                        match_ms=0.0,
                        roi_shapes=(full_image_roi,),
                        timing_breakdown={"lite_halcon_crop": True},
                    )

            def crop_target_items(_test_controller) -> list[InspectionItem]:
                return list(matching_items)

            original_runtime_context_factory = test_execution_module.ProductRuntimeContext
            controller.target_inspection_items = MethodType(crop_target_items, controller)
            test_execution_module.ProductRuntimeContext = lambda _session, _algo: LiteCropRuntimeContext()
            try:
                return original_execute_image(image_path)
            finally:
                test_execution_module.ProductRuntimeContext = original_runtime_context_factory
                controller.target_inspection_items = original_target_items

        controller.execute_image = MethodType(lite_execute_image, controller)

    def _install_lite_calibration_model_name(self) -> None:
        """Persist a friendly model name and copy each trained NPZ under that name."""
        page = self.tool_page
        action_layout = page.btn_export_onnx.parentWidget().layout()
        if action_layout is None:
            return

        name_row = QtWidgets.QWidget(page.btn_export_onnx.parentWidget())
        name_layout = QtWidgets.QHBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(6)
        name_label = QtWidgets.QLabel(_lite_text("标定模型名称：", "Calibration model name:"), name_row)
        name_edit = QtWidgets.QLineEdit(name_row)
        name_edit.setPlaceholderText(_lite_text("例如：Cover外观模型", "e.g. Cover appearance model"))
        name_edit.setMinimumHeight(28)
        name_edit.setStyleSheet(
            "QLineEdit{background:#232323;color:#f2f2f2;border:1px solid #6a6a6a;"
            "border-radius:3px;padding:3px 7px;}"
            "QLineEdit:focus{border:1px solid #2d8cff;}"
            "QLineEdit:disabled{background:#303030;color:#888;border-color:#4a4a4a;}"
        )
        name_edit.setToolTip(
            _lite_text(
                "标定完成后会生成同名 NPZ 副本；内部运行模型名称保持不变。",
                "After calibration, a same-named NPZ copy is created; the internal runtime model remains unchanged.",
            )
        )
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit, 1)
        export_index = action_layout.indexOf(page.btn_export_onnx)
        action_layout.insertWidget(max(0, export_index), name_row)
        self.lbl_lite_calibration_model_name = name_label
        self.edit_lite_calibration_model_name = name_edit

        def names_path() -> str:
            return os.path.join(page.session.product_dir, LITE_MODEL_NAMES_FILENAME)

        def load_names() -> dict[str, str]:
            payload = load_json_with_backup(names_path(), default={})
            if not isinstance(payload, dict):
                return {}
            return {
                str(key): str(value or "").strip()
                for key, value in payload.items()
                if str(key).strip() and str(value or "").strip()
            }

        def selected_item():
            return page._selected_inspection_item()

        def selected_or_only_trainable_item():
            item = selected_item()
            if item is not None:
                return item
            role = page.current_camera_role()
            candidates = [
                (index, candidate)
                for index, candidate in enumerate(list(getattr(page, "inspection_items", []) or []))
                if bool(getattr(candidate, "enabled", True))
                and normalize_camera_role(getattr(candidate, "camera_id", ""), default="cam1") == role
                and page.algo.is_learning_tool(getattr(candidate, "algorithm_code", ""))
            ]
            if len(candidates) != 1:
                return None
            item_index, candidate = candidates[0]
            visible_indexes = list(getattr(page, "_visible_inspection_item_indexes", []) or [])
            try:
                row = visible_indexes.index(item_index)
            except ValueError:
                row = 0 if page.inspection_items_table.rowCount() == 1 else -1
            if row >= 0:
                page.inspection_items_table.setCurrentCell(row, 1)
                page.inspection_items_table.selectRow(row)
            return selected_item() or candidate

        def sync_name_editor(*_args) -> None:
            item = selected_item()
            name_edit.blockSignals(True)
            try:
                if item is None:
                    name_edit.clear()
                    name_edit.setEnabled(False)
                else:
                    configured = load_names().get(item.model_key, "")
                    default_name = str(getattr(item, "display_name", "") or item.roi_label or item.item_id)
                    name_edit.setText(configured or default_name)
                    name_edit.setEnabled(page.algo.is_learning_tool(item.algorithm_code))
            finally:
                name_edit.blockSignals(False)

        def save_current_name() -> None:
            item = selected_item()
            if item is None:
                return
            custom_name = _safe_export_stem(name_edit.text(), str(item.display_name or "model"))
            name_edit.setText(custom_name)
            names = load_names()
            names[item.model_key] = custom_name
            atomic_write_json(names_path(), names, ensure_ascii=False, indent=2)

        def ask_model_name(item) -> bool:
            names = load_names()
            default_name = names.get(
                item.model_key,
                str(getattr(item, "display_name", "") or item.roi_label or item.item_id or "model"),
            )
            entered, accepted = QtWidgets.QInputDialog.getText(
                page,
                _lite_text("设置标定模型名称", "Set Calibration Model Name"),
                _lite_text(
                    f"检测项：{item.display_name}\n请输入标定后生成的模型名称：",
                    f"Tool: {item.display_name}\nEnter the model name to create after calibration:",
                ),
                QtWidgets.QLineEdit.EchoMode.Normal,
                default_name,
            )
            if not accepted:
                return False
            custom_name = _safe_export_stem(entered, "")
            if not custom_name:
                QtWidgets.QMessageBox.warning(
                    page,
                    tr("common.info"),
                    _lite_text("模型名称不能为空。", "The model name cannot be empty."),
                )
                return False
            names[item.model_key] = custom_name
            atomic_write_json(names_path(), names, ensure_ascii=False, indent=2)
            if selected_item() is item:
                name_edit.setText(custom_name)
            return True

        name_edit.editingFinished.connect(save_current_name)
        selection_model = page.inspection_items_table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(sync_name_editor)

        original_refresh_table = page._refresh_inspection_items_table

        def lite_refresh_table(tool_page) -> None:
            original_refresh_table()
            table = getattr(tool_page, "inspection_items_table", None)
            if (
                table is not None
                and table.rowCount() == 1
                and not table.selectionModel().hasSelection()
            ):
                table.setCurrentCell(0, 1)
                table.selectRow(0)
            QtCore.QTimer.singleShot(0, sync_name_editor)

        page._refresh_inspection_items_table = MethodType(lite_refresh_table, page)

        original_train_current = page._train
        original_train_all = page._train_all_tools

        def lite_train_current(tool_page) -> None:
            item = selected_or_only_trainable_item()
            if item is None or not page.algo.is_learning_tool(getattr(item, "algorithm_code", "")):
                original_train_current()
                return
            if ask_model_name(item):
                original_train_current()

        def lite_train_all(tool_page) -> None:
            role = page.current_camera_role()
            items = [
                item
                for item in list(getattr(page, "inspection_items", []) or [])
                if bool(getattr(item, "enabled", True))
                and normalize_camera_role(getattr(item, "camera_id", ""), default="cam1") == role
                and page.algo.is_learning_tool(getattr(item, "algorithm_code", ""))
            ]
            for item in items:
                if not ask_model_name(item):
                    return
            original_train_all()

        page._train = MethodType(lite_train_current, page)
        page._train_all_tools = MethodType(lite_train_all, page)
        for target_button, handler in (
            (getattr(page, "btn_train_current", None), page._train),
            (getattr(page, "btn_train", None), page._train_all_tools),
        ):
            if target_button is None:
                continue
            try:
                target_button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            target_button.clicked.connect(handler)

        def lite_start_worker(training_controller, payload: dict) -> None:
            save_current_name()
            names = load_names()
            pending: list[dict[str, str]] = []
            product_dir = str(payload.get("product_dir", "") or page.session.product_dir)
            for task in list(payload.get("tasks", []) or []):
                task = dict(task or {})
                model_key = str(task.get("model_key", "") or "").strip()
                display_name = str(task.get("display_name", "") or "model").strip()
                custom_name = _safe_export_stem(names.get(model_key, display_name), "model")
                pending.append({
                    "display_name": display_name,
                    "model_key": model_key,
                    "algorithm": str(task.get("algorithm", "") or "").strip(),
                    "product_dir": product_dir,
                    "custom_name": custom_name,
                })

            destination_names = [
                os.path.normcase(os.path.abspath(os.path.join(entry["product_dir"], f"{entry['custom_name']}.npz")))
                for entry in pending
            ]
            if len(destination_names) != len(set(destination_names)):
                QtWidgets.QMessageBox.warning(
                    page,
                    tr("common.info"),
                    _lite_text(
                        "多个检测项使用了相同的标定模型名称，请分别设置不同名称。",
                        "Multiple tools use the same calibration model name. Assign a unique name to each tool.",
                    ),
                )
                return
            existing_names = [
                os.path.basename(destination)
                for destination in destination_names
                if os.path.exists(destination)
            ]
            if existing_names:
                answer = QtWidgets.QMessageBox.question(
                    page,
                    _lite_text("覆盖自定义模型", "Overwrite Custom Model"),
                    _lite_text(
                        "以下模型已存在，标定成功后是否覆盖？\n" + "\n".join(existing_names),
                        "These models already exist. Overwrite them after successful calibration?\n"
                        + "\n".join(existing_names),
                    ),
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            self._lite_pending_calibration_exports = pending

            # A MethodType callback patched onto ``page`` is seen by PySide as a
            # plain Python callable. Connecting the worker signal to it directly
            # can therefore execute UI code inside the training thread. Route
            # completion through a real QObject slot owned by this main window.
            if getattr(page, "_training_in_progress", False):
                QtWidgets.QMessageBox.information(
                    page,
                    tr("common.info"),
                    "Training is already running.",
                )
                return
            training_controller.set_running(True)
            training_controller.on_progress("training queued")

            thread = QtCore.QThread(page)
            worker = training_controller.worker_cls(page.algo, payload)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progressChanged.connect(page._on_training_progress)
            worker.finished.connect(
                self._dispatch_lite_training_finished,
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(
                lambda: page._forget_training_job(thread, worker)
            )
            page._training_thread = thread
            page._training_worker = worker
            thread.start()

        page.training_controller.start_worker = MethodType(
            lite_start_worker,
            page.training_controller,
        )

        original_training_finished = page._on_training_finished

        def lite_training_finished(tool_page, payload: object) -> None:
            result = dict(payload or {}) if isinstance(payload, dict) else {}
            success_names = {
                str(value).strip()
                for value in list(result.get("success_names", []) or [])
                if str(value).strip()
            }
            exported: list[str] = []
            failures: list[str] = []
            for entry in list(getattr(self, "_lite_pending_calibration_exports", []) or []):
                if entry["display_name"] not in success_names:
                    continue
                source = page.algo.embedding_model_path(
                    entry["algorithm"],
                    entry["product_dir"],
                    model_key=entry["model_key"],
                )
                destination = os.path.join(
                    entry["product_dir"],
                    f"{entry['custom_name']}.npz",
                )
                try:
                    if not os.path.exists(source):
                        raise FileNotFoundError(source)
                    if os.path.normcase(os.path.abspath(source)) != os.path.normcase(os.path.abspath(destination)):
                        shutil.copy2(source, destination)
                    exported.append(destination)
                except Exception as exc:
                    failures.append(f"{entry['display_name']}: {exc}")
            self._lite_pending_calibration_exports = []

            if exported:
                suffix = _lite_text(
                    "\n自定义模型：" + ", ".join(os.path.basename(path) for path in exported),
                    "\nCustom model: " + ", ".join(os.path.basename(path) for path in exported),
                )
                result["last_dialog_message"] = str(result.get("last_dialog_message", "") or "") + suffix
                result["last_status_message"] = str(result.get("last_status_message", "") or "") + suffix
            if failures:
                existing_failures = list(result.get("failure_messages", []) or [])
                result["failure_messages"] = existing_failures + failures
            original_training_finished(result)

        page._on_training_finished = MethodType(lite_training_finished, page)
        sync_name_editor()

    def _install_lite_embedding_analysis(self) -> None:
        """Expose the existing feature-analysis dialog in the Lite action panel."""
        page = self.tool_page
        export_button = getattr(page, "btn_export_onnx", None)
        if export_button is None:
            return
        action_layout = export_button.parentWidget().layout()
        if action_layout is None:
            return

        tools_row = QtWidgets.QWidget(export_button.parentWidget())
        tools_layout = QtWidgets.QHBoxLayout(tools_row)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(6)
        analysis_button = QtWidgets.QPushButton(tools_row)
        analysis_button.setIcon(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView
            )
        )
        analysis_button.setStyleSheet(export_button.styleSheet())
        analysis_button.clicked.connect(page.open_embedding_analysis_tool)
        export_index = action_layout.indexOf(export_button)
        action_layout.removeWidget(export_button)
        export_button.setParent(tools_row)
        tools_layout.addWidget(analysis_button, 1)
        tools_layout.addWidget(export_button, 1)
        action_layout.insertWidget(max(0, export_index), tools_row)
        self.lite_ai_tools_row = tools_row
        page.btn_embedding_analysis = analysis_button
        self._refresh_lite_text()
        page._update_runtime_widgets()

    def _install_lite_onnx_export(self) -> None:
        """Give the Lite exporter a save-as filename without changing ToolPage/Main."""
        page = self.tool_page
        button = getattr(page, "btn_export_onnx", None)
        if button is None:
            return

        def lite_export_onnx(tool_page) -> None:
            algorithm = tool_page.current_algorithm()
            if not tool_page._is_embedding_algorithm(algorithm):
                QtWidgets.QMessageBox.information(
                    tool_page,
                    tr("common.info"),
                    _lite_text("请选择学习工具后再导出 ONNX。", "Select a learning tool before exporting ONNX."),
                )
                return

            selected_item = tool_page._selected_inspection_item()
            camera_role = (
                selected_item.camera_id
                if selected_item is not None
                else tool_page.current_camera_role()
            )
            backbone = tool_page.algo.resolve_learning_algorithm(algorithm, camera_role)
            display_name = tool_page.algo.algorithm_display_name(backbone) or backbone
            product_name = _safe_export_stem(tool_page.session.current_product, "product")
            item_name = _safe_export_stem(
                getattr(selected_item, "display_name", "") if selected_item is not None else camera_role,
                "model",
            )
            configured_names = load_json_with_backup(
                os.path.join(tool_page.session.product_dir, LITE_MODEL_NAMES_FILENAME),
                default={},
            )
            if isinstance(configured_names, dict) and selected_item is not None:
                item_name = _safe_export_stem(
                    configured_names.get(selected_item.model_key, item_name),
                    item_name,
                )
            default_filename = f"{product_name}_{item_name}.onnx"
            default_path = os.path.join(tool_page.session.product_dir, default_filename)
            destination, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                tool_page,
                _lite_text("导出并命名 ONNX 模型", "Export and Name ONNX Model"),
                default_path,
                "ONNX Model (*.onnx)",
            )
            destination = str(destination or "").strip()
            if not destination:
                return
            if Path(destination).suffix.lower() != ".onnx":
                destination += ".onnx"

            export_button = getattr(tool_page, "btn_export_onnx", None)
            if export_button is not None:
                export_button.setEnabled(False)
            tool_page.lbl_training_validation.setText(f"Status: exporting ONNX {display_name}...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                info = dict(qr_core.export_backbone_onnx(backbone, device="cpu") or {})
                source_onnx = str(info.get("onnx_path", "") or "").strip()
                if not source_onnx or not os.path.exists(source_onnx):
                    raise FileNotFoundError(source_onnx or "ONNX export did not produce a file")
                destination_path = os.path.abspath(destination)
                if os.path.normcase(os.path.abspath(source_onnx)) != os.path.normcase(destination_path):
                    shutil.copy2(source_onnx, destination_path)

                source_runtime = str(info.get("runtime_path", "") or "").strip()
                runtime_destination = ""
                if source_runtime and os.path.exists(source_runtime):
                    runtime_destination = str(Path(destination_path).with_suffix(".ort"))
                    if os.path.normcase(os.path.abspath(source_runtime)) != os.path.normcase(runtime_destination):
                        shutil.copy2(source_runtime, runtime_destination)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    tool_page,
                    _lite_text("导出ONNX失败", "ONNX Export Failed"),
                    f"{exc}\n\n{traceback.format_exc()}",
                )
                return
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
                if export_button is not None:
                    export_button.setEnabled(True)

            opsets = ", ".join(
                f"{domain}:{version}"
                for domain, version in list(info.get("opsets", []) or [])
            ) or "-"
            input_shape = info.get("input_shape", []) or []
            input_text = "x".join(str(value) for value in input_shape) if input_shape else "-"
            tool_page.lbl_training_validation.setText(
                f"Status: ONNX exported {os.path.basename(destination_path)} "
                f"opset={opsets} input={input_text}"
            )
            message = f"ONNX: {destination_path}"
            if runtime_destination:
                message += f"\nORT: {runtime_destination}"
            message += f"\nopset: {opsets}\ninput: {input_text}"
            QtWidgets.QMessageBox.information(
                tool_page,
                _lite_text("导出ONNX完成", "ONNX Export Complete"),
                message,
            )

        page._export_current_backbone_onnx = MethodType(lite_export_onnx, page)
        try:
            button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        button.clicked.connect(page._export_current_backbone_onnx)


def _normalize_application_font(app: QtWidgets.QApplication) -> None:
    font = QtGui.QFont(app.font())
    if font.pointSizeF() > 0:
        return
    if font.pixelSize() > 0:
        font.setPointSize(max(1, int(round(font.pixelSize() * 0.75))))
    else:
        font.setPointSize(10)
    app.setFont(font)


def _resource_path(filename: str) -> Path:
    return packaged_embedding_test_root(__file__) / "res" / filename


def _app_icon() -> QtGui.QIcon:
    for name in ("logo.ico", "logo.png"):
        path = _resource_path(name)
        if path.exists():
            icon = QtGui.QIcon(str(path))
            if not icon.isNull():
                return icon
    return QtGui.QIcon()


def _set_windows_app_id(app_id: str) -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def main() -> None:
    _set_windows_app_id(WINDOWS_APP_ID)
    app = QtWidgets.QApplication(sys.argv)
    _normalize_application_font(app)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    icon = _app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = LiteDebugMainWindow()
    if not icon.isNull():
        window.setWindowIcon(icon)

    screen = app.primaryScreen()
    available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1366, 768)
    small_screen = available.width() <= 1366 or available.height() <= 800
    if small_screen:
        window.showMaximized()
    else:
        window.resize(
            min(1400, max(1200, available.width() - 80)),
            min(900, max(800, available.height() - 80)),
        )
        window.show()
    app.exec()


if __name__ == "__main__":
    main()
