from __future__ import annotations

import os
from typing import Dict, List, Optional, cast

import numpy as np
from PySide6 import QtCore

from shape.core.bootstrap import ensure_repo_root_on_path
from shape.core.template_core import (
    MaskRect,
    RoiRect,
    build_multi_backend_detector,
    pose_infos_from_ui_values,
)

ensure_repo_root_on_path()

from shape.like_matcher import TemplateLevel, save_detector_model  # noqa: E402


class _TemplateBuildWorker(QtCore.QObject):
    progressChanged = QtCore.Signal(int, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, payload: Dict[str, object], parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._payload = dict(payload)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.progressChanged.emit(10, "Validating template ROI...")
            pose = self._payload["pose"]
            assert isinstance(pose, dict)
            pose_infos = pose_infos_from_ui_values(
                float(pose["angle_start"]),
                float(pose["angle_end"]),
                float(pose["angle_step"]),
                float(pose["scale_start"]),
                float(pose["scale_end"]),
                float(pose["scale_step"]),
            )

            self.progressChanged.emit(25, "Building template features...")
            detector, kept, skipped = build_multi_backend_detector(
                class_id=str(self._payload["class_id"]),
                roi_img=cast(np.ndarray, self._payload["roi_img"]),
                roi_rect=cast(RoiRect, self._payload["roi_rect"]),
                mask_rects=cast(List[MaskRect], self._payload["mask_rects"]),
                pose_infos=pose_infos,
                pose_ui=cast(Dict[str, float], self._payload["pose_ui"]),
                levels=cast(List[int], self._payload["levels"]),
                num_features=int(self._payload["num_features"]),
                weak_threshold=float(self._payload["weak_threshold"]),
                strong_threshold=float(self._payload["strong_threshold"]),
                original_mode=str(self._payload["original_mode"]),
                original_editor_levels=cast(Optional[List[TemplateLevel]], self._payload["original_editor_levels"]),
                source_image_path=str(self._payload["source_image_path"]),
            )

            self.progressChanged.emit(85, "Writing model file...")
            os.makedirs(str(self._payload["role_dir"]), exist_ok=True)
            save_detector_model(detector, str(self._payload["model_path"]))
            self.progressChanged.emit(100, "Template saved.")
            self.finished.emit({"detector": detector, "kept": kept, "skipped": skipped})
        except Exception as exc:
            self.failed.emit(str(exc))


__all__ = ["_TemplateBuildWorker"]
