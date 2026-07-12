from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from common import labelme_io
from common.app_logging import get_app_logger
from common.camera_roles import normalize_camera_role
from common.path_utils import product_relative_path, resolve_product_path
from common.safe_io import atomic_write_json, backup_path_for, load_json_with_backup
from ui.i18n import tr


LOGGER = get_app_logger(__name__)


def _normalize_camera_role(camera_id: object) -> str:
    return normalize_camera_role(camera_id)


class RoiAnnotationController:
    def __init__(self, owner) -> None:
        self.owner = owner

    def store_path(self) -> str:
        return os.path.join(self.owner.session.product_dir, "sample_annotations.json")

    def path_key(self, path: object) -> str:
        return product_relative_path(path, base_dir=self.owner.session.product_dir)

    @staticmethod
    def roi_key(camera_role: object, label_name: object) -> str:
        role = _normalize_camera_role(camera_role) or "cam1"
        label = str(label_name or "").strip()
        return f"{role}::{label}" if label else role

    def load(self) -> None:
        self.owner._sample_roi_annotations_by_path = {}
        store_path = self.store_path()
        if not store_path:
            return
        raw_payload = load_json_with_backup(store_path, default=None)
        if raw_payload is None:
            return
        image_payload = raw_payload.get("images", raw_payload) if isinstance(raw_payload, dict) else {}
        if not isinstance(image_payload, dict):
            return
        for stored_path, payload in image_payload.items():
            resolved_path = resolve_product_path(
                stored_path,
                base_dir=self.owner.session.product_dir,
                anchor_dir=self.owner.session.product_dir,
                prefer_existing=False,
            )
            if not resolved_path:
                continue
            raw_labels = payload.get("roi_status", payload) if isinstance(payload, dict) else {}
            if not isinstance(raw_labels, dict):
                continue
            normalized_labels: Dict[str, str] = {}
            for key, value in raw_labels.items():
                annotation_key = str(key or "").strip()
                annotation_value = str(value or "").strip().upper()
                if annotation_key and annotation_value in {"OK", "NG"}:
                    normalized_labels[annotation_key] = annotation_value
            if normalized_labels:
                self.owner._sample_roi_annotations_by_path[os.path.normpath(resolved_path)] = normalized_labels

    def save(self) -> None:
        store_path = self.store_path()
        if not store_path:
            return
        images_payload: Dict[str, Dict[str, Dict[str, str]]] = {}
        for path, labels in sorted(self.owner._sample_roi_annotations_by_path.items()):
            normalized_path = os.path.normpath(str(path or ""))
            if not normalized_path or not labels:
                continue
            key = self.path_key(normalized_path)
            if key:
                images_payload[key] = {"roi_status": dict(sorted(labels.items()))}
        if not images_payload:
            self.delete_store()
            return
        atomic_write_json(store_path, {"images": images_payload}, ensure_ascii=False, indent=2)

    def delete_store(self) -> None:
        store_path = self.store_path()
        try:
            if store_path and os.path.exists(store_path):
                os.remove(store_path)
            backup_path = backup_path_for(store_path) if store_path else None
            if backup_path is not None and backup_path.exists():
                backup_path.unlink()
        except Exception as exc:
            LOGGER.exception("Failed to delete sample annotation store %s: %s", store_path, exc)

    @staticmethod
    def has_geometry(path: str, label_name: str) -> bool:
        if not path or not label_name:
            return False
        json_path = labelme_io.labelme_json_of_image(path)
        if not os.path.exists(json_path):
            return False
        try:
            return labelme_io.read_shape_from_labelme(json_path, label_name) is not None
        except Exception as exc:
            LOGGER.exception("Failed to read ROI geometry %s label=%s: %s", json_path, label_name, exc)
            return False

    def status_for_path(self, path: str, camera_role: object, label_name: str) -> str:
        normalized_path = os.path.normpath(str(path or ""))
        if not normalized_path:
            return ""
        annotation_key = self.roi_key(camera_role, label_name)
        return str(
            self.owner._sample_roi_annotations_by_path.get(normalized_path, {}).get(annotation_key, "") or ""
        ).strip().upper()

    def set_status_for_path(self, path: str, camera_role: object, label_name: str, status: object) -> None:
        normalized_path = os.path.normpath(str(path or ""))
        if not normalized_path:
            return
        annotation_key = self.roi_key(camera_role, label_name)
        status_text = str(status or "").strip().upper()
        annotations = dict(self.owner._sample_roi_annotations_by_path.get(normalized_path, {}))
        if status_text in {"OK", "NG"}:
            annotations[annotation_key] = status_text
        else:
            annotations.pop(annotation_key, None)
        if annotations:
            self.owner._sample_roi_annotations_by_path[normalized_path] = annotations
        else:
            self.owner._sample_roi_annotations_by_path.pop(normalized_path, None)
        self.save()

    def mark_all_status(self, path: str, status: object, camera_role: object = None) -> None:
        role = _normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        status_text = str(status or "").strip().upper()
        if status_text not in {"OK", "NG"}:
            return
        for label in self.owner._inspection_label_names_for_role(role):
            if self.has_geometry(path, label):
                self.set_status_for_path(path, role, label, status_text)

    def mark_all_ok(self, path: str, camera_role: object = None) -> None:
        self.mark_all_status(path, "OK", camera_role)

    def mark_all_ng(self, path: str, camera_role: object = None) -> None:
        self.mark_all_status(path, "NG", camera_role)

    def clear_path(self, path: str, camera_role: object = None) -> None:
        role = _normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        for label in self.owner._inspection_label_names_for_role(role):
            self.set_status_for_path(path, role, label, "")

    def counts_for_roi(
        self,
        roi_label: str,
        camera_role: object = None,
        *,
        paths: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        role = _normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        target_paths = list(paths or self.owner._sample_paths_for_kind("train", role))
        ok_count = 0
        ng_count = 0
        unset_count = 0
        for path in target_paths:
            if not self.has_geometry(path, roi_label):
                unset_count += 1
                continue
            status = self.status_for_path(path, role, roi_label)
            if status == "OK":
                ok_count += 1
            elif status == "NG":
                ng_count += 1
            else:
                unset_count += 1
        return ok_count, ng_count, unset_count

    def progress_for_path(self, path: str, camera_role: object = None) -> Tuple[int, int]:
        labels = self.owner._inspection_label_names_for_role(camera_role)
        if not labels:
            return 0, 0
        role = _normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        present_count = sum(
            1
            for label in labels
            if self.has_geometry(path, label) and self.status_for_path(path, role, label) in {"OK", "NG"}
        )
        return present_count, len(labels)

    def state_for_path(self, path: str, camera_role: object = None) -> str:
        labels = self.owner._inspection_label_names_for_role(camera_role)
        if not labels:
            return tr("sample.unset")
        geometry_missing = sum(1 for label in labels if not self.has_geometry(path, label))
        if geometry_missing:
            return tr("sample.missing_roi")
        present_count, total_count = self.progress_for_path(path, camera_role)
        if total_count <= 0 or present_count <= 0:
            return tr("sample.unset")
        if present_count < total_count:
            return tr("sample.partial")
        return tr("sample.complete")
