"""Resolve local and published models without tying them to an ROI position."""
from __future__ import annotations

import os
from pathlib import Path

from common.algorithm_codes import storage_code_backbone
from common.model_library import parse_library_model_key


class ModelBindingMixin:
    def resolve_item_algorithm(self, item) -> str:
        if getattr(item, "model_source", "independent") != "library":
            if getattr(item, "model_source", "independent") == "group":
                self._known_group_model_keys.add(item.effective_model_key)
            return self.resolve_tool_algorithm(item.algorithm_code, item.camera_id)
        manifest = self.model_library.resolve(item.library_model_id, item.library_version)
        algorithm = storage_code_backbone(manifest["algorithm"])
        if self.is_learning_tool(item.algorithm_code):
            if manifest["kind"] not in {"embedding", "yolo"} or not (self.is_embedding_algorithm(algorithm) or self.is_yolo_algorithm(algorithm)):
                raise ValueError("公共模型类型与当前学习工具不兼容")
        elif manifest["kind"] != "traditional" or algorithm != item.algorithm_code:
            raise ValueError("公共模型算法与当前检测工具不兼容")
        return algorithm

    def _library_model_for_key(self, model_key, algorithm=None):
        reference = parse_library_model_key(str(model_key or ""))
        if reference is None:
            return None
        return self.model_library.resolve(*reference, algorithm=algorithm)

    def _local_embedding_file(self, algorithm, product_dir, model_key):
        paths = self.embedding_model_storage_paths(algorithm, product_dir, model_key=model_key)
        found = next((path for path in paths if os.path.isfile(path)), "")
        # Legacy unkeyed models belong only to the old independent workflow.
        if not found and model_key and "__group__" not in model_key and model_key not in self._known_group_model_keys:
            found = next(
                (path for path in self.embedding_model_storage_paths(algorithm, product_dir)
                 if os.path.isfile(path)), ""
            )
        return found

    def describe_model_binding(self, item, product_dir) -> dict:
        source = getattr(item, "model_source", "independent")
        description = dict(ready=False, source=source, message="", path="", name="",
                           version="", algorithm="", parameters={})
        try:
            algorithm = self.resolve_item_algorithm(item)
            description["algorithm"] = algorithm
            if source == "library":
                manifest = self.model_library.resolve(
                    item.library_model_id, item.library_version, algorithm=algorithm
                )
                description.update(
                    ready=True, name=manifest["name"], version=manifest["version"],
                    model_id=manifest["model_id"], created_at=manifest.get("created_at", ""),
                    origin=dict(manifest.get("source") or {}),
                    path=manifest.get("model_file") or manifest["manifest_path"],
                    parameters=dict(manifest["parameters"]),
                    message=f"公共模型：{manifest['name']} / {manifest['version']}（固定版本）",
                )
                return description
            if source == "group" and not str(item.task_group or "").strip():
                raise ValueError("请选择或创建共享组")
            key = item.effective_model_key
            if self.is_yolo_algorithm(algorithm):
                paths = self.yolo_model_paths(product_dir, model_key=key)
                description.update(ready=paths.onnx.is_file() and paths.metadata.is_file(), path=str(paths.onnx))
            elif self.is_embedding_algorithm(algorithm):
                model_path = self._local_embedding_file(algorithm, product_dir, key)
                description.update(
                    ready=bool(model_path),
                    path=model_path or self.embedding_model_path(algorithm, product_dir, model_key=key),
                    legacy=bool(model_path and model_path != self.embedding_model_path(algorithm, product_dir, model_key=key)),
                )
            elif self.is_traditional_tool(item.algorithm_code):
                description["ready"] = bool(self.get_traditional_model_dict(algorithm, model_key=key))
            else:
                raise ValueError("当前工具不使用 OK/NG 模型")
            description["name"] = item.task_group if source == "group" else item.display_name
            prefix = f"共享组：{item.task_group}" if source == "group" else "独立模型"
            prefix += f"，{self.algorithm_display_name(algorithm) or algorithm}"
            description["message"] = prefix + ("，已训练" if description["ready"] else "，尚未训练")
        except Exception as exc:
            description["message"] = str(exc)
        return description

    def publish_model_to_library(self, item, product_dir, name, model_id=None) -> dict:
        if getattr(item, "model_source", "independent") == "library":
            raise ValueError("引用的公共模型为只读，请从独立模型或组模型发布新版本")
        algorithm = self.resolve_item_algorithm(item)
        description = self.describe_model_binding(item, product_dir)
        if not description["ready"]:
            raise ValueError(description["message"] or "请先完成模型训练")
        kwargs = {}
        if self.is_yolo_algorithm(algorithm):
            paths = self.yolo_model_paths(product_dir, model_key=item.effective_model_key)
            self._load_yolo_model_cached(product_dir, model_key=item.effective_model_key)
            kwargs.update(model_file=paths.onnx, classifier_metadata=paths.metadata)
        elif self.is_embedding_algorithm(algorithm):
            model = self._load_register_model_cached(description["path"])
            if not model.is_ready() or storage_code_backbone(model.backbone) != algorithm:
                raise ValueError("模型未就绪或模型算法与当前工具不一致")
            kwargs["model_file"] = description["path"]
        else:
            kwargs["traditional_model"] = self.get_traditional_model_dict(
                algorithm, model_key=item.effective_model_key
            )
        parameters = {"score_mode": self.product_params.score_mode,
                      "margin": float(self.product_params.margin), "topk": int(self.product_params.topk)}
        if self.is_yolo_algorithm(algorithm):
            parameters["yolo_ng_threshold"] = float(self.product_params.yolo_ng_threshold)
        return self.model_library.publish(
            name=name, algorithm=algorithm, model_id=model_id,
            parameters=parameters,
            source={"order": Path(product_dir).name, "roi": item.roi_label,
                    "camera_id": item.camera_id, "roi_display_name": item.display_name,
                    "group": getattr(item, "task_group", ""),
                    "model_key": item.effective_model_key},
            **kwargs,
        )
