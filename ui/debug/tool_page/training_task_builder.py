from __future__ import annotations

import os
from typing import List

from common.app_logging import get_app_logger
from domain import InspectionItem
from ui.debug.tool_page.camera_roles import normalize_camera_role


LOGGER = get_app_logger(__name__)


class TrainingTaskBuilder:
    def __init__(self, owner) -> None:
        self.owner = owner

    @staticmethod
    def item_display_name(inspection_item: InspectionItem) -> str:
        return str(
            inspection_item.display_name
            or inspection_item.roi_label
            or inspection_item.item_id
            or "tool"
        ).strip() or "tool"

    def resolve_algorithm(self, inspection_item: InspectionItem) -> str:
        if self.owner.algo.is_learning_tool(inspection_item.algorithm_code):
            return self.owner.algo.current_learning_backbone()
        return self.owner.algo.resolve_tool_algorithm(inspection_item.algorithm_code)

    def train_sample_paths_for_role(self, camera_role: object = None) -> List[str]:
        role = normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        train_paths = list(getattr(self.owner, "train_files", []) or [])
        if not train_paths:
            train_paths = list(getattr(self.owner, "ok_files", []) or []) + list(getattr(self.owner, "ng_files", []) or [])
        return self.owner._filter_paths_for_camera(list(dict.fromkeys(train_paths)), role)

    def sample_groups_for_role(
        self,
        camera_role: object = None,
        *,
        roi_label: object = None,
    ) -> tuple[List[str], List[str], List[str]]:
        role = normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        candidate_paths = self.train_sample_paths_for_role(role)
        label = str(roi_label or "").strip()
        if not label:
            return [], [], candidate_paths
        training_ok_files: List[str] = []
        training_ng_files: List[str] = []
        for path in candidate_paths:
            if not self.owner._path_has_roi_geometry(path, label):
                continue
            status = self.owner._sample_roi_status_for_path(path, role, label)
            if status == "OK":
                training_ok_files.append(path)
            elif status == "NG":
                training_ng_files.append(path)
        return training_ok_files, training_ng_files, candidate_paths

    def ready_signature(self, camera_role: object = None) -> str:
        role = normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        candidate_paths = self.train_sample_paths_for_role(role)
        method_getter = getattr(self.owner, "loc_method_for_role", None)
        method = method_getter(role) if callable(method_getter) else self.owner.loc_method
        if method == "ncc":
            recipe_path = ""
            model_path = self.owner.ncc_model_path_for_role(role)
        else:
            recipe_path = self.owner.shape_recipe_path_for_role(role)
            model_path = self.owner.shape_model_path_for_role(role)
        recipe_mtime = os.path.getmtime(recipe_path) if recipe_path and os.path.exists(recipe_path) else -1.0
        model_mtime = os.path.getmtime(model_path) if model_path and os.path.exists(model_path) else -1.0
        return "|".join(
            [
                role,
                str(method),
                str(recipe_mtime),
                str(model_mtime),
                str(self.owner.ref_image or ""),
                *sorted(str(path) for path in candidate_paths),
            ]
        )

    def missing_training_roi_paths(
        self,
        roi_label: str,
        candidate_paths: List[str],
        *,
        camera_role: object = None,
    ) -> List[str]:
        role = normalize_camera_role(camera_role or self.owner.current_camera_role()) or "cam1"
        missing_paths: List[str] = []
        for path in candidate_paths:
            if not self.owner._path_has_roi_geometry(path, roi_label):
                missing_paths.append(path)
        method_getter = getattr(self.owner, "loc_method_for_role", None)
        method = method_getter(role) if callable(method_getter) else self.owner.loc_method
        if missing_paths and method in {"shape", "ncc"}:
            try:
                self.owner._autogen_roi_for_images(
                    missing_paths,
                    only_missing=False,
                    silent=True,
                    camera_role=role,
                )
                missing_paths = [
                    path for path in candidate_paths
                    if not self.owner._path_has_roi_geometry(path, roi_label)
                ]
            except Exception as exc:
                LOGGER.exception("Failed to auto-generate missing training ROI for label %s: %s", roi_label, exc)
        return missing_paths

    def build_task_for_item(self, inspection_item: InspectionItem) -> dict:
        if not inspection_item.enabled:
            raise RuntimeError("selected tool is disabled")

        algorithm = self.resolve_algorithm(inspection_item)
        if not algorithm:
            if self.owner.algo.is_learning_tool(inspection_item.algorithm_code):
                raise RuntimeError("please choose a learning tool subtype first")
            raise RuntimeError("please select an inspection tool")

        roi_label = str(inspection_item.roi_label or "").strip() or "roi"
        training_ok_files, training_ng_files, candidate_paths = self.sample_groups_for_role(
            inspection_item.camera_id,
            roi_label=roi_label,
        )
        if not candidate_paths:
            camera_id = normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing training images for {camera_id}")
        missing_groups: List[str] = []
        if not training_ok_files:
            missing_groups.append("OK")
        if not training_ng_files:
            missing_groups.append("NG")
        if missing_groups:
            camera_id = normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing {'/'.join(missing_groups)} images for {camera_id}")

        missing_paths = self.missing_training_roi_paths(
            roi_label,
            candidate_paths,
            camera_role=inspection_item.camera_id,
        )
        if missing_paths:
            missing = [os.path.basename(path) for path in missing_paths[:50]]
            raise RuntimeError(
                f"missing ROI label '{roi_label}' in some training sample jsons:\n" + "\n".join(missing)
            )

        embedding_cache_dir = ""
        embedding_checker = getattr(self.owner.algo, "is_embedding_algorithm", None)
        is_embedding = bool(embedding_checker(algorithm)) if callable(embedding_checker) else False
        if is_embedding:
            cache_dir_getter = getattr(self.owner.algo, "embedding_cache_dir", None)
            if callable(cache_dir_getter):
                embedding_cache_dir = str(cache_dir_getter(algorithm, self.owner.session.product_dir) or "")

        return {
            "item_id": str(inspection_item.item_id or ""),
            "display_name": self.item_display_name(inspection_item),
            "camera_id": normalize_camera_role(inspection_item.camera_id) or "cam1",
            "algorithm": algorithm,
            "ok_files": list(training_ok_files),
            "ng_files": list(training_ng_files),
            "label_names": [roi_label],
            "model_key": inspection_item.model_key,
            "embedding_cache_dir": embedding_cache_dir,
        }

    def payload(
        self,
        mode: str,
        tasks: List[dict],
        *,
        selected_item_id: str = "",
        failures: List[str] | None = None,
    ) -> dict:
        return {
            "mode": mode,
            "tasks": list(tasks),
            "product_dir": self.owner.session.product_dir,
            "score_mode": self.owner.cmb_mode.currentText(),
            "margin": float(self.owner.spin_margin.value()),
            "topk": int(self.owner.spin_topk.value()),
            "selected_item_id": str(selected_item_id or ""),
            "failure_messages": list(failures or []),
        }
