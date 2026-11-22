from __future__ import annotations

import os
from typing import List

from domain import InspectionItem
from ui.debug.tool_page.camera_roles import normalize_camera_role
from ui.i18n import tr


class TrainingTaskBuilder:
    def __init__(self, owner) -> None:
        self.owner = owner

    @staticmethod
    def item_display_name(inspection_item: InspectionItem) -> str:
        group = (
            str(getattr(inspection_item, "task_group", ""))
            if str(getattr(inspection_item, "model_source", "independent")) == "group"
            else ""
        )
        return str(
            group
            or inspection_item.display_name
            or inspection_item.roi_label
            or inspection_item.item_id
            or "tool"
        ).strip() or "tool"

    def resolve_algorithm(self, inspection_item: InspectionItem) -> str:
        resolver = getattr(self.owner.algo, "resolve_item_algorithm", None)
        if callable(resolver):
            return resolver(inspection_item)
        if self.owner.algo.is_learning_tool(inspection_item.algorithm_code):
            return self.owner.algo.current_learning_backbone(inspection_item.camera_id)
        return self.owner.algo.resolve_tool_algorithm(
            inspection_item.algorithm_code,
            inspection_item.camera_id,
        )

    @staticmethod
    def effective_model_key(inspection_item: InspectionItem) -> str:
        return str(getattr(inspection_item, "effective_model_key", inspection_item.model_key))

    def group_items_for_item(self, inspection_item: InspectionItem) -> List[InspectionItem]:
        """Resolve the training group without coupling it to ROI display names."""
        if str(getattr(inspection_item, "model_source", "independent")) != "group":
            return [inspection_item]
        if not str(getattr(inspection_item, "task_group", "")).strip():
            raise RuntimeError("please select a shared model group")
        key = self.effective_model_key(inspection_item)
        role = normalize_camera_role(inspection_item.camera_id) or "cam1"
        members = [
            item for item in list(getattr(self.owner, "inspection_items", [inspection_item]))
            if item.enabled
            and str(getattr(item, "model_source", "independent")) == "group"
            and (normalize_camera_role(item.camera_id) or "cam1") == role
            and self.effective_model_key(item) == key
        ]
        algorithm = self.resolve_algorithm(inspection_item)
        if any(self.resolve_algorithm(item) != algorithm for item in members):
            raise RuntimeError(tr("debug.model_source.group_incompatible"))
        return members or [inspection_item]

    def _samples_for_item(self, inspection_item: InspectionItem):
        if str(getattr(inspection_item, "model_source", "independent")) == "library":
            return [], [], []
        members = self.group_items_for_item(inspection_item)
        labels = list(dict.fromkeys(str(item.roi_label or "").strip() for item in members if item.roi_label))
        role = normalize_camera_role(inspection_item.camera_id) or "cam1"
        candidates = self.train_sample_paths_for_role(role)
        ok_samples: list[tuple[str, str]] = []
        ng_samples: list[tuple[str, str]] = []
        for path in candidates:
            for label in labels:
                if not self.owner._path_has_roi_geometry(path, label):
                    continue
                status = self.owner._sample_roi_status_for_path(path, role, label)
                if status == "OK":
                    ok_samples.append((path, label))
                elif status == "NG":
                    ng_samples.append((path, label))
        return ok_samples, ng_samples, candidates

    def sample_groups_for_item(self, inspection_item: InspectionItem) -> tuple[List[str], List[str], List[str]]:
        ok_samples, ng_samples, candidates = self._samples_for_item(inspection_item)
        return (
            list(dict.fromkeys(path for path, _label in ok_samples)),
            list(dict.fromkeys(path for path, _label in ng_samples)),
            candidates,
        )

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
        # Training must never mutate sample ROI geometry.  Template/NCC based
        # ROI generation is an explicit preparation action because generated
        # geometry still needs a human OK/NG annotation before model fitting.
        return [
            path
            for path in candidate_paths
            if not self.owner._path_has_roi_geometry(path, roi_label)
        ]

    def build_task_for_item(self, inspection_item: InspectionItem) -> dict:
        if not inspection_item.enabled:
            raise RuntimeError("selected tool is disabled")
        if str(getattr(inspection_item, "model_source", "independent")) == "library":
            raise RuntimeError("a published model reference does not need training")

        algorithm = self.resolve_algorithm(inspection_item)
        if not algorithm:
            if self.owner.algo.is_learning_tool(inspection_item.algorithm_code):
                raise RuntimeError("please choose a learning tool subtype first")
            raise RuntimeError("please select an inspection tool")

        members = self.group_items_for_item(inspection_item)
        label_names = list(dict.fromkeys(str(item.roi_label or "").strip() for item in members if item.roi_label))
        if not label_names:
            label_names = ["roi"]
        ok_samples, ng_samples, candidate_paths = self._samples_for_item(inspection_item)
        if not candidate_paths:
            camera_id = normalize_camera_role(inspection_item.camera_id) or "cam1"
            raise RuntimeError(f"missing training images for {camera_id}")

        training_ok_files = list(dict.fromkeys(path for path, _label in ok_samples))
        training_ng_files = list(dict.fromkeys(path for path, _label in ng_samples))
        missing_groups: List[str] = []
        if not training_ok_files:
            missing_groups.append("OK")
        if not training_ng_files:
            missing_groups.append("NG")
        if missing_groups:
            raise RuntimeError(
                tr(
                    "debug.training_missing_required_groups",
                    roi=(
                        str(getattr(inspection_item, "task_group", "") or "").strip()
                        if str(getattr(inspection_item, "model_source", "independent")) == "group"
                        else "/".join(label_names)
                    ),
                    groups="/".join(missing_groups),
                )
            )

        classifier = getattr(self.owner.algo, "is_yolo_algorithm", lambda _code: False)(algorithm)
        if classifier and (len(ok_samples) < 2 or len(ng_samples) < 2):
            raise RuntimeError(tr("debug.standard_learning_min_samples", ok=len(ok_samples), ng=len(ng_samples)))

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
            "label_names": label_names,
            "model_key": self.effective_model_key(inspection_item),
            "model_source": str(getattr(inspection_item, "model_source", "independent")),
            "ok_samples": ok_samples,
            "ng_samples": ng_samples,
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
