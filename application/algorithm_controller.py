
from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from algorithms.traditional import (
    TRADITIONAL_ALGORITHMS,
    TraditionalThresholdModel,
    compute_roi_metrics,
    is_traditional_algorithm,
    metric_value,
    train_threshold_model,
)
from algorithms.measurement import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHM,
    FIND_LINE_ALGORITHMS,
    LINE_DISTANCE_ALGORITHMS,
    MEASUREMENT_ALGORITHMS,
    PIN_CENTER_DISTANCE_ALGORITHM,
    is_measurement_algorithm,
    judge_bright_block_y_distance,
    judge_edge_distance,
    judge_pin_center_distance,
    measure_bright_block_center,
    measure_bright_block_y_distance,
    measure_edge_distance,
    measure_find_line,
    measure_pin_center_distance,
)
from common.algorithm_codes import (
    DEFAULT_LEARNING_BACKBONE,
    LEARNING_BACKBONES,
    SHARED_BACKBONE_ALGORITHM_CODE,
    learning_backbone_storage_code,
    learning_backbone_storage_codes,
    normalize_tool_algorithm_code,
    storage_code_backbone,
)
from common.camera_roles import (
    CAMERA_ROLES,
    DEFAULT_CAMERA_ROLE,
    camera_role_from_text,
    normalize_camera_role,
)
from algorithms.registry import (
    get_tool_algorithm_spec,
    is_learning_tool_algorithm,
    is_measurement_tool_algorithm,
    is_traditional_tool_algorithm,
)
from infrastructure.product_params import (
    ProductRuntimeParams,
    load_product_params,
    save_product_params,
)
import algorithms.lazy_api as qr_core
from common import labelme_io
from ui.algorithm_labels import algorithm_display_name


SUPPORTED_EMBEDDING_ALGORITHMS = list(LEARNING_BACKBONES)
SUPPORTED_ALGORITHMS = SUPPORTED_EMBEDDING_ALGORITHMS + TRADITIONAL_ALGORITHMS + MEASUREMENT_ALGORITHMS
SUPPORTED_SCORE_MODES = ["proto", "topk"]


# ---------------------------------------------------------------------------
# 返回值数据类
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """_train 的纯数据返回值，不含任何 Widget 引用。"""
    algorithm: str
    is_embedding: bool
    status_message: str           # 写进 lbl_status 的文字
    dialog_message: str           # 弹出 information 的文字
    # embedding 算法：训练完的模型对象
    model: Any = None
    saved_model_path: str = ""
    # 传统算法：阈值模型字典（写进 product_params.traditional_models）
    traditional_model_dict: Optional[dict] = None
    # 传统算法：训练时生成的结果行（用于填结果表）
    result_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PredictResult:
    """_predict_image 的纯数据返回值。"""
    file_path: str
    file_name: str
    gt: str
    pred: str
    diff: float
    sim_ok: Optional[float]
    sim_ng: Optional[float]
    value: Optional[float]
    threshold: Optional[float]
    match_ms: Optional[float]
    total_ms: float
    json_name: str
    detail: str = ""
    measurement: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "gt": self.gt,
            "pred": self.pred,
            "diff": self.diff,
            "sim_ok": self.sim_ok,
            "sim_ng": self.sim_ng,
            "value": self.value,
            "threshold": self.threshold,
            "match_ms": self.match_ms,
            "total_ms": self.total_ms,
            "json_name": self.json_name,
            "detail": self.detail,
        }
        if self.measurement is not None:
            payload["measurement"] = dict(self.measurement)
        return payload


# ---------------------------------------------------------------------------
# AlgorithmController
# ---------------------------------------------------------------------------

class AlgorithmController:
    """
    算法参数与模型的非 UI 控制器。

    使用方式：
        ctrl = AlgorithmController()
        ctrl.load_params(session.product_params_path)
        model, msg = ctrl.load_model_for_algorithm(algorithm, session.product_dir)
    """

    def __init__(self) -> None:
        self.product_params: ProductRuntimeParams = ProductRuntimeParams()
        self.model: Optional[Any] = None          # qr_core.RegisterModel | None
        self._feat_net_cache: Dict[Tuple[str, str, str], Any] = {}
        self._embedding_model_cache: Dict[str, Tuple[Tuple[int, int], Any]] = {}
        self._embedding_model_cache_lock = threading.RLock()
        self._active_product_dir: str = ""
        self._loaded_embedding_model_key: Tuple[str, str] = ("", "")

    # ------------------------------------------------------------------
    # 参数持久化
    # ------------------------------------------------------------------

    def load_params(self, path: str) -> None:
        """从 product_params.json 加载；不存在则使用默认值。"""
        # Loading a product/session is an explicit cache boundary. Paths
        # already isolate products; clearing here also releases inactive ones.
        self.clear_embedding_model_cache()
        self.product_params = load_product_params(path)
        alg = storage_code_backbone(self.product_params.algorithm)
        learning_backbone = storage_code_backbone(self.product_params.learning_backbone)
        if learning_backbone not in SUPPORTED_EMBEDDING_ALGORITHMS:
            if alg in SUPPORTED_EMBEDDING_ALGORITHMS:
                learning_backbone = alg
            else:
                learning_backbone = ""
        learning_backbones = {
            role: normalized
            for camera_id, backbone in dict(
                getattr(self.product_params, "learning_backbones", {}) or {}
            ).items()
            if (role := normalize_camera_role(camera_id))
            and (normalized := storage_code_backbone(backbone)) in SUPPORTED_EMBEDDING_ALGORITHMS
        }
        if not learning_backbone and learning_backbones:
            learning_backbone = (
                learning_backbones.get(DEFAULT_CAMERA_ROLE)
                or next(iter(learning_backbones.values()))
            )
        if learning_backbone:
            for role in CAMERA_ROLES:
                learning_backbones.setdefault(role, learning_backbone)
        self.product_params.learning_backbones = learning_backbones
        self.product_params.learning_backbone = (
            learning_backbones.get(DEFAULT_CAMERA_ROLE, learning_backbone)
            if learning_backbones
            else learning_backbone
        )
        if alg and alg not in SUPPORTED_ALGORITHMS:
            self.product_params.algorithm = ""
        elif not alg:
            self.product_params.algorithm = ""
        else:
            self.product_params.algorithm = alg
        if str(self.product_params.score_mode or "") not in SUPPORTED_SCORE_MODES:
            self.product_params.score_mode = SUPPORTED_SCORE_MODES[0]
        self.product_params.topk = max(1, int(self.product_params.topk))
        self.product_params.margin = float(self.product_params.margin)

    def save_params(self, path: str) -> None:
        save_product_params(self.product_params, path)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def is_embedding_algorithm(self, algorithm: Optional[str] = None) -> bool:
        normalized = self.resolve_learning_algorithm(algorithm or str(self.product_params.algorithm or ""))
        if not normalized:
            return False
        return normalized in SUPPORTED_EMBEDDING_ALGORITHMS

    def is_measurement_algorithm(self, algorithm: Optional[str] = None) -> bool:
        normalized = str(algorithm or self.product_params.algorithm or "").strip()
        return is_measurement_algorithm(normalized)

    def current_learning_backbone(self, camera_role: object = None) -> str:
        role = normalize_camera_role(camera_role, default=DEFAULT_CAMERA_ROLE)
        backbone = storage_code_backbone(
            dict(getattr(self.product_params, "learning_backbones", {}) or {}).get(role, "")
        )
        if backbone in SUPPORTED_EMBEDDING_ALGORITHMS:
            return backbone
        backbone = storage_code_backbone(self.product_params.learning_backbone)
        if backbone in SUPPORTED_EMBEDDING_ALGORITHMS:
            return backbone
        algorithm = storage_code_backbone(self.product_params.algorithm)
        if algorithm in SUPPORTED_EMBEDDING_ALGORITHMS:
            return algorithm
        return ""

    def set_learning_backbone(self, backbone: str, camera_role: object = None) -> str:
        normalized = storage_code_backbone(backbone)
        if normalized not in SUPPORTED_EMBEDDING_ALGORITHMS:
            normalized = DEFAULT_LEARNING_BACKBONE
        role = normalize_camera_role(camera_role, default=DEFAULT_CAMERA_ROLE)
        previous = self.current_learning_backbone(role)
        fallback = previous or self.current_learning_backbone(DEFAULT_CAMERA_ROLE) or DEFAULT_LEARNING_BACKBONE
        learning_backbones = dict(getattr(self.product_params, "learning_backbones", {}) or {})
        for camera_id in CAMERA_ROLES:
            learning_backbones.setdefault(camera_id, fallback)
        learning_backbones[role] = normalized
        self.product_params.learning_backbones = learning_backbones
        self.product_params.learning_backbone = learning_backbones[DEFAULT_CAMERA_ROLE]
        if self.is_embedding_algorithm(self.product_params.algorithm):
            self.product_params.algorithm = learning_backbones[DEFAULT_CAMERA_ROLE]
        return normalized

    def resolve_learning_algorithm(self, algorithm: object, camera_role: object = None) -> str:
        normalized = storage_code_backbone(algorithm)
        if not normalized:
            return ""
        if normalized in SUPPORTED_EMBEDDING_ALGORITHMS:
            return normalized
        if normalize_tool_algorithm_code(normalized) == SHARED_BACKBONE_ALGORITHM_CODE:
            return self.current_learning_backbone(camera_role)
        return normalized

    def resolve_tool_algorithm(self, algorithm_code: object, camera_role: object = None) -> str:
        backbone = storage_code_backbone(algorithm_code)
        if backbone in SUPPORTED_EMBEDDING_ALGORITHMS:
            return backbone
        normalized = normalize_tool_algorithm_code(algorithm_code)
        if is_learning_tool_algorithm(normalized):
            return self.current_learning_backbone(camera_role)
        return normalized

    def tool_algorithm_spec(self, algorithm_code: object):
        return get_tool_algorithm_spec(algorithm_code)

    def algorithm_display_name(self, algorithm_code: object) -> str:
        return algorithm_display_name(algorithm_code)

    def is_learning_tool(self, algorithm_code: object) -> bool:
        return is_learning_tool_algorithm(algorithm_code)

    def is_traditional_tool(self, algorithm_code: object) -> bool:
        return is_traditional_tool_algorithm(algorithm_code)

    def is_measurement_tool(self, algorithm_code: object) -> bool:
        return is_measurement_tool_algorithm(algorithm_code)

    @staticmethod
    def _normalize_model_key(model_key: object) -> str:
        normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(model_key or "").strip()).strip("._-")
        return normalized

    def tool_model_key(self, model_key: object) -> str:
        return self._normalize_model_key(model_key)

    def embedding_model_path(self, algorithm: str, product_dir: str, *, model_key: object = "") -> str:
        normalized_key = self.tool_model_key(model_key)
        storage_code = learning_backbone_storage_code(algorithm)
        if normalized_key:
            return os.path.join(product_dir, f"{normalized_key}_register_model_{storage_code}.npz")
        return os.path.join(product_dir, f"register_model_{storage_code}.npz")

    def embedding_model_storage_paths(self, algorithm: str, product_dir: str, *, model_key: object = "") -> List[str]:
        normalized_key = self.tool_model_key(model_key)
        paths: List[str] = []
        for storage_code in learning_backbone_storage_codes(algorithm):
            if normalized_key:
                paths.append(os.path.join(product_dir, f"{normalized_key}_register_model_{storage_code}.npz"))
            else:
                paths.append(os.path.join(product_dir, f"register_model_{storage_code}.npz"))
        paths.append(self.embedding_model_legacy_path(algorithm, product_dir, model_key=normalized_key))
        return list(dict.fromkeys(paths))

    def embedding_model_legacy_path(self, algorithm: str, product_dir: str, *, model_key: object = "") -> str:
        normalized_key = self.tool_model_key(model_key)
        algorithm = storage_code_backbone(algorithm)
        if normalized_key:
            return os.path.join(product_dir, f"{normalized_key}_register_model_{algorithm}.npz")
        return os.path.join(product_dir, f"register_model_{algorithm}.npz")

    def embedding_cache_dir(self, algorithm: str, product_dir: str) -> str:
        storage_code = learning_backbone_storage_code(algorithm)
        return os.path.join(product_dir, "embedding_cache", storage_code)

    def traditional_model_storage_key(self, algorithm: str, *, model_key: object = "") -> str:
        normalized_key = self.tool_model_key(model_key)
        if normalized_key:
            return f"{algorithm}::{normalized_key}"
        return str(algorithm or "").strip()

    def get_traditional_model_dict(self, algorithm: str, *, model_key: object = "") -> Optional[dict]:
        storage_key = self.traditional_model_storage_key(algorithm, model_key=model_key)
        model_dict = self.product_params.traditional_models.get(storage_key)
        if isinstance(model_dict, dict):
            return model_dict
        legacy_model_dict = self.product_params.traditional_models.get(str(algorithm or "").strip())
        if isinstance(legacy_model_dict, dict):
            return legacy_model_dict
        return None

    def _loaded_embedding_matches(
        self,
        algorithm: str,
        *,
        labels: List[str],
        model_key: object = "",
    ) -> bool:
        if self.model is None:
            return False
        if str(getattr(self.model, "backbone", "") or "").strip() != str(algorithm or "").strip():
            return False
        normalized_key = self.tool_model_key(model_key)
        loaded_algorithm, loaded_key = self._loaded_embedding_model_key
        if str(loaded_algorithm or "").strip() and str(loaded_algorithm or "").strip() != str(algorithm or "").strip():
            return False
        if normalized_key and str(loaded_key or "").strip() and loaded_key != normalized_key:
            return False
        model_labels = []
        effective_labels = getattr(self.model, "effective_label_names", None)
        if callable(effective_labels):
            model_labels = [str(name).strip() for name in effective_labels() if str(name).strip()]
        else:
            label_name = str(getattr(self.model, "label_name", "")).strip()
            model_labels = [label_name] if label_name else []
        expected_labels = [str(name).strip() for name in labels if str(name).strip()]
        if expected_labels and model_labels and expected_labels != model_labels:
            return False
        return True

    @staticmethod
    def _feat_net_runtime_info(feat_net: Any) -> Dict[str, Any]:
        describer = getattr(qr_core, "describe_backbone_runner", None)
        if not callable(describer) or feat_net is None:
            return {}
        try:
            raw_info = dict(describer(feat_net) or {})
        except Exception:
            return {}
        backend = str(raw_info.get("backend", "") or "").strip().lower()
        model_format = str(raw_info.get("model_format", "") or "").strip().lower()
        model_path = str(raw_info.get("model_path", "") or "").strip()
        providers = tuple(
            str(provider).strip()
            for provider in tuple(raw_info.get("providers", ()) or ())
            if str(provider).strip()
        )
        cpu_inference_chunk_size = int(raw_info.get("cpu_inference_chunk_size", 0) or 0)
        return {
            "backend": backend,
            "backend_label": backend.upper() if backend else "",
            "model_format": model_format,
            "model_format_label": model_format.upper() if model_format else "",
            "model_path": model_path,
            "providers": providers,
            "cpu_inference_chunk_size": cpu_inference_chunk_size,
        }

    @classmethod
    def _print_feat_net_backend(cls, backbone: str, feat_net: Any) -> None:
        info = cls._feat_net_runtime_info(feat_net)
        backend_label = str(info.get("backend_label", "") or "").strip() or "TORCH"
        model_path = str(info.get("model_path", "") or "").strip()
        model_hint = f"  model={os.path.basename(model_path)}" if model_path else ""
        cpu_chunk_size = int(info.get("cpu_inference_chunk_size", 0) or 0)
        chunk_hint = f"  cpu_chunk={cpu_chunk_size}" if cpu_chunk_size > 0 else ""
        if not callable(getattr(getattr(sys, "stdout", None), "write", None)):
            return
        print(
            f"[EmbeddingTest] inference backbone={learning_backbone_storage_code(backbone)} "
            f"backend={backend_label}{model_hint}{chunk_hint}"
        )

    def get_feat_net(
        self,
        backbone: str,
        device: Optional[str] = None,
        *,
        preferred_backend: str = "auto",
    ) -> Any:
        normalized_backbone = storage_code_backbone(backbone)
        if not normalized_backbone:
            raise ValueError("backbone is required")
        normalized_device = str(device or qr_core.get_device()).strip() or "cpu"
        normalized_backend = str(preferred_backend or "auto").strip().lower() or "auto"
        cache_key = (normalized_backbone, normalized_device, normalized_backend)
        feat_net = self._feat_net_cache.get(cache_key)
        if feat_net is not None:
            self._print_feat_net_backend(normalized_backbone, feat_net)
            return feat_net
        feat_net, _ = qr_core.load_backbone(
            normalized_backbone,
            device=normalized_device,
            preferred_backend=normalized_backend,
        )
        self._feat_net_cache[cache_key] = feat_net
        self._print_feat_net_backend(normalized_backbone, feat_net)
        return feat_net

    def clear_feat_net_cache(self) -> None:
        self._feat_net_cache.clear()

    def clear_embedding_model_cache(self) -> None:
        with self._embedding_model_cache_lock:
            self._embedding_model_cache.clear()

    def _load_register_model_cached(self, model_file: str) -> Any:
        normalized_path = os.path.normcase(os.path.abspath(str(model_file)))
        stat = os.stat(normalized_path)
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        with self._embedding_model_cache_lock:
            cached = self._embedding_model_cache.get(normalized_path)
            if cached is not None and cached[0] == signature:
                return cached[1]
            model = qr_core.load_register_model_npz(normalized_path)
            self._embedding_model_cache[normalized_path] = (signature, model)
            return model

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_model_for_algorithm(
        self, algorithm: str, product_dir: str, *, model_key: object = ""
    ) -> Tuple[Any, str]:
        """
        加载指定算法的模型。
        返回 (model_or_None, status_message)。
        调用方负责把 status_message 写进 lbl_status。
        """
        algorithm = self.resolve_learning_algorithm(algorithm)
        display_name = self.algorithm_display_name(algorithm)
        normalized_model_key = self.tool_model_key(model_key)
        self._active_product_dir = str(product_dir or "").strip()
        if not algorithm:
            self.model = None
            self._loaded_embedding_model_key = ("", "")
            return None, "状态：请选择工具"
        if not self.is_embedding_algorithm(algorithm):
            self.model = None
            self._loaded_embedding_model_key = ("", "")
            return None, f"状态：当前工具={display_name or algorithm}，使用传统方法"

        model_file = self.embedding_model_path(algorithm, product_dir, model_key=normalized_model_key)
        source_model_key = normalized_model_key
        if not os.path.exists(model_file) and normalized_model_key:
            legacy_model_file = next(
                (path for path in self.embedding_model_storage_paths(algorithm, product_dir) if os.path.exists(path)),
                "",
            )
            legacy_raw_model_file = next(
                (
                    path
                    for path in self.embedding_model_storage_paths(
                        algorithm,
                        product_dir,
                        model_key=normalized_model_key,
                    )
                    if os.path.exists(path)
                ),
                "",
            )
            if legacy_model_file:
                model_file = legacy_model_file
                source_model_key = ""
            elif legacy_raw_model_file:
                model_file = legacy_raw_model_file
        if not os.path.exists(model_file):
            legacy_model_file = next(
                (
                    path
                    for path in self.embedding_model_storage_paths(
                        algorithm,
                        product_dir,
                        model_key=normalized_model_key,
                    )
                    if os.path.exists(path)
                ),
                "",
            )
            if legacy_model_file:
                model_file = legacy_model_file
        if not os.path.exists(model_file) and not normalized_model_key:
            legacy_shared_model = next(
                (path for path in self.embedding_model_storage_paths(algorithm, product_dir) if os.path.exists(path)),
                "",
            )
            if legacy_shared_model:
                model_file = legacy_shared_model
        if not os.path.exists(model_file):
            self.model = None
            self._loaded_embedding_model_key = ("", "")
            return None, f"状态：{display_name or algorithm} 未训练"

        model = self._load_register_model_cached(model_file)
        model.score_mode = self.product_params.score_mode
        model.margin = float(self.product_params.margin)
        model.topk = int(self.product_params.topk)
        self.model = model
        self._loaded_embedding_model_key = (algorithm, source_model_key)
        return model, (
            f"状态：已加载工具  {display_name or algorithm}  mode={model.score_mode}  "
            f"margin={model.margin:.4f}  topk={model.topk}"
        )

    def apply_params_to_model(self) -> None:
        """把当前 product_params 的 score_mode/margin/topk 同步到已加载的 model。"""
        if self.model is not None:
            self.model.score_mode = self.product_params.score_mode
            self.model.margin = float(self.product_params.margin)
            self.model.topk = int(self.product_params.topk)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(
        self,
        ok_files: List[str],
        ng_files: List[str],
        *,
        algorithm: str,
        product_dir: str,
        label_names: List[str],
        model_key: object = "",
        progress_callback: Optional[Callable[[str], None]] = None,
        embedding_cache_dir: Optional[str] = None,
    ) -> TrainResult:
        """
        训练 embedding 或传统阈值模型。
        成功返回 TrainResult；失败抛出异常（调用方负责捕获并弹 QMessageBox）。
        """
        algorithm = self.resolve_learning_algorithm(algorithm)
        display_name = self.algorithm_display_name(algorithm)
        normalized_model_key = self.tool_model_key(model_key)
        self._active_product_dir = str(product_dir or "").strip()
        mode = self.product_params.score_mode
        margin = float(self.product_params.margin)
        topk = int(self.product_params.topk)

        if self.is_embedding_algorithm(algorithm):
            device = qr_core.get_device()
            lite_runtime = str(os.environ.get("LC_SYSTEM_LITE", "")).strip().lower() in {"1", "true", "yes", "lite"}
            if lite_runtime and callable(progress_callback):
                progress_callback(f"preparing backbone {learning_backbone_storage_code(algorithm)} on {device}")
            feat_net = self.get_feat_net(algorithm, device)
            if lite_runtime and callable(progress_callback):
                info = self._feat_net_runtime_info(feat_net)
                backend = str(info.get("backend_label", "") or "TORCH").strip()
                model_path = str(info.get("model_path", "") or "").strip()
                model_hint = f" {os.path.basename(model_path)}" if model_path else ""
                progress_callback(f"backbone ready {backend}{model_hint}")
            model = qr_core.train_register_model(
                ok_files,
                ng_files,
                backbone=algorithm,
                score_mode=mode,
                margin=margin,
                topk=topk,
                label_name=label_names[0],
                label_names=label_names,
                progress_callback=progress_callback,
                cache_dir=embedding_cache_dir or self.embedding_cache_dir(algorithm, product_dir),
                device=device,
                feat_net=feat_net,
            )
            saved_path = self.embedding_model_path(algorithm, product_dir, model_key=normalized_model_key)
            qr_core.save_register_model_npz(model, saved_path)
            # Training replaced model data on disk. Rebuild the runtime model
            # snapshot on the next preparation instead of retaining old ROIs.
            self.clear_embedding_model_cache()
            self.model = model
            self._loaded_embedding_model_key = (algorithm, normalized_model_key)
            return TrainResult(
                algorithm=algorithm,
                is_embedding=True,
                status_message=(
                    f"状态：已训练  {display_name or algorithm}  mode={mode}  "
                    f"margin={margin:.4f}  topk={topk}"
                ),
                dialog_message="OK/NG 注册完成，可以开始测试。",
                model=model,
                saved_model_path=saved_path,
            )
        elif is_traditional_algorithm(algorithm):
            threshold_model, train_rows = train_threshold_model(
                ok_files,
                ng_files,
                algorithm,
                preferred_label=label_names[0],
            )
            result_rows: List[Dict[str, Any]] = []
            for sample in train_rows:
                pred, diff = threshold_model.predict(float(sample["value"]))
                result_rows.append({
                    "file_path": str(sample.get("file_path", "")),
                    "file_name": str(sample.get("file_name", "")),
                    "gt": str(sample.get("gt", "")),
                    "pred": pred,
                    "diff": float(diff),
                    "sim_ok": None,
                    "sim_ng": None,
                    "value": float(sample["value"]),
                    "threshold": float(threshold_model.threshold),
                    "match_ms": None,
                    "total_ms": None,
                    "json_name": os.path.basename(
                        labelme_io.labelme_json_of_image(str(sample.get("file_path", "")))
                    ),
                })
            storage_key = self.traditional_model_storage_key(algorithm, model_key=normalized_model_key)
            self.product_params.traditional_models[storage_key] = threshold_model.to_dict()
            self.model = None
            self._loaded_embedding_model_key = ("", "")
            return TrainResult(
                algorithm=algorithm,
                is_embedding=False,
                status_message=(
                    f"状态：已训练  {display_name or algorithm}  "
                    f"threshold={threshold_model.threshold:.4f}  "
                    f"ok_when={threshold_model.ok_when}  acc={threshold_model.accuracy:.4f}"
                ),
                dialog_message="传统算法阈值模型训练完成，可以开始测试。",
                traditional_model_dict=threshold_model.to_dict(),
                result_rows=result_rows,
            )
        else:
            raise RuntimeError(f"{display_name or algorithm} does not need OK/NG training")

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict_image(
        self,
        path: str,
        *,
        labels: List[str],
        feat_net: Any = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        match_ms: Optional[float] = None,
        algorithm_override: Optional[str] = None,
        model_key_override: Optional[str] = None,
        params_override: Optional[Dict[str, Any]] = None,
    ) -> PredictResult:
        """
        对单张已定位好 ROI 的图做推理。
        调用方负责：定位（shape）/ 获取 labels / 传入 roi。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        override_text = str(algorithm_override or "").strip()
        camera_role = camera_role_from_text(model_key_override)
        if override_text:
            algorithm = self.resolve_tool_algorithm(override_text, camera_role)
            if algorithm not in SUPPORTED_ALGORITHMS:
                raise RuntimeError(f"unsupported inspection algorithm: {override_text}")
        else:
            algorithm = self.resolve_learning_algorithm(
                self.product_params.algorithm,
                camera_role,
            )
        model_key = self.tool_model_key(model_key_override)
        if not algorithm.strip():
            raise RuntimeError("请先选择工具")
        total_t0 = time.perf_counter()
        measurement_payload: Optional[Dict[str, Any]] = None

        if self.is_embedding_algorithm(algorithm):
            if not self._loaded_embedding_matches(algorithm, labels=labels, model_key=model_key):
                if not self._active_product_dir:
                    raise RuntimeError(f"algorithm model not loaded: {learning_backbone_storage_code(algorithm)}")
                self.load_model_for_algorithm(algorithm, self._active_product_dir, model_key=model_key)
            if self.model is None:
                raise RuntimeError(f"algorithm model not loaded: {learning_backbone_storage_code(algorithm)}")
            self.apply_params_to_model()

            if len(labels) == 1 and roi is None:
                j = labelme_io.labelme_json_of_image(path)
                if not os.path.exists(j):
                    raise FileNotFoundError(f"缺少 labelme json: {j}")

            if feat_net is None:
                feat_net = self.get_feat_net(self.model.backbone, getattr(self.model, "device", None))
            if len(labels) > 1:
                e = qr_core.embed_many(path, feat_net, labels, device=self.model.device)
            else:
                e = qr_core.embed_one(
                    path, feat_net,
                    label_name=labels[0],
                    device=self.model.device,
                    roi_xywh=roi,
                )
            pred, diff, sim_ok, sim_ng = qr_core.predict_one_with_model(e, self.model)
            value: Optional[float] = None
            threshold: Optional[float] = None
        elif is_traditional_algorithm(algorithm):
            model_dict = self.get_traditional_model_dict(algorithm, model_key=model_key)
            if not isinstance(model_dict, dict):
                raise RuntimeError(f"{self.algorithm_display_name(algorithm) or algorithm} 尚未训练")
            threshold_model = TraditionalThresholdModel.from_dict(model_dict)
            metrics = compute_roi_metrics(path, preferred_label=threshold_model.roi_label or labels[0])
            value = metric_value(metrics, algorithm)
            pred, diff = threshold_model.predict(value)
            sim_ok = None
            sim_ng = None
            threshold = float(threshold_model.threshold)
        elif is_measurement_algorithm(algorithm):
            params = dict(params_override or {})
            if algorithm in FIND_LINE_ALGORITHMS:
                measurement = measure_find_line(
                    path,
                    preferred_label=labels[0] if labels else "roi1",
                    params=params,
                    algorithm=algorithm,
                )
                pred = "OK"
                diff = float(measurement.line.residual)
                detail = (
                    f"line_found pts={measurement.line.point_count}"
                    f" pos={measurement.position_px:.3f}px"
                    f" angle={measurement.angle_deg:.3f}deg"
                    f" residual={measurement.line.residual:.3f}"
                )
                measurement_payload = measurement.to_dict()
                value = None
                threshold = None
            elif algorithm in LINE_DISTANCE_ALGORITHMS:
                raise RuntimeError("Line Distance must be run with paired Find Line tools")
            elif algorithm == CENTER_DISTANCE_ALGORITHM:
                raise RuntimeError("Center Distance must be run with paired Bright Block Center tools")
            elif algorithm == BRIGHT_BLOCK_CENTER_ALGORITHM:
                diff = 0.0
                roi_label = labels[0] if labels else "roi1"
                try:
                    measurement = measure_bright_block_center(
                        path,
                        preferred_label=roi_label,
                        params=params,
                    )
                    pred = "OK"
                    detail = (
                        f"bright_block_center=({measurement.center_xy[0]:.3f},"
                        f"{measurement.center_xy[1]:.3f})px"
                        f" threshold={measurement.threshold:.1f}"
                    )
                    measurement_payload = measurement.to_dict()
                    measurement_payload.update({"pred": pred})
                except RuntimeError as exc:
                    pred = "NG"
                    detail = f"bright_block_center_missing: {exc}"
                    measurement_payload = {
                        "type": BRIGHT_BLOCK_CENTER_ALGORITHM,
                        "roi_label": str(roi_label or ""),
                        "center_points": [],
                        "candidates": [],
                        "pred": pred,
                        "detail": str(exc),
                    }
                value = None
                threshold = None
            elif algorithm == PIN_CENTER_DISTANCE_ALGORITHM:
                measurement = measure_pin_center_distance(
                    path,
                    preferred_label=labels[0] if labels else "roi1",
                    params=params,
                )
                pred, judged_value, lower, upper, unit = judge_pin_center_distance(measurement, params)
                diff = 0.0
                detail = (
                    f"pin_center_distance={judged_value:.3f}{unit}"
                    f" raw={measurement.distance_px:.3f}px"
                    f" threshold={measurement.threshold:.1f}"
                )
                measurement_payload = measurement.to_dict()
                measurement_payload.update(
                    {
                        "distance": float(judged_value),
                        "unit": unit,
                        "label": f"{judged_value:.3f}{unit}",
                        "pred": pred,
                    }
                )
                value = judged_value
                threshold = float(upper) if upper is not None else None
                if lower is not None or upper is not None:
                    detail += f" spec={lower if lower is not None else '-'}..{upper if upper is not None else '-'}{unit}"
            elif algorithm == BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM:
                measurement = measure_bright_block_y_distance(
                    path,
                    preferred_label=labels[0] if labels else "roi1",
                    params=params,
                )
                pred, judged_value, lower, upper, unit = judge_bright_block_y_distance(measurement, params)
                diff = 0.0
                detail = (
                    f"bright_block_y_distance={judged_value:.3f}{unit}"
                    f" raw={measurement.distance_px:.3f}px"
                    f" threshold={measurement.threshold:.1f}"
                )
                measurement_payload = measurement.to_dict()
                measurement_payload.update(
                    {
                        "distance": float(judged_value),
                        "unit": unit,
                        "label": f"{judged_value:.3f}{unit}",
                        "pred": pred,
                    }
                )
                value = judged_value
                threshold = float(upper) if upper is not None else None
                if lower is not None or upper is not None:
                    detail += f" spec={lower if lower is not None else '-'}..{upper if upper is not None else '-'}{unit}"
            else:
                measurement = measure_edge_distance(
                    path,
                    preferred_label=labels[0] if labels else "roi1",
                    params=params,
                )
                pred, judged_value, lower, upper, unit = judge_edge_distance(measurement, params)
                diff = float(measurement.line_a.residual + measurement.line_b.residual) * 0.5
                detail = f"distance={judged_value:.3f}{unit}"
                measurement_payload = measurement.to_dict()
                value = judged_value
                threshold = float(upper) if upper is not None else None
                if lower is not None or upper is not None:
                    detail += f" spec={lower if lower is not None else '-'}..{upper if upper is not None else '-'}{unit}"
            sim_ok = None
            sim_ng = None
        else:
            raise RuntimeError(f"unsupported inspection algorithm: {algorithm}")

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        return PredictResult(
            file_path=path,
            file_name=os.path.basename(path),
            gt="",
            pred=pred,
            diff=float(diff),
            sim_ok=float(sim_ok) if sim_ok is not None else None,
            sim_ng=float(sim_ng) if sim_ng is not None else None,
            value=float(value) if value is not None else None,
            threshold=float(threshold) if threshold is not None else None,
            match_ms=match_ms,
            total_ms=float(total_ms),
            json_name=os.path.basename(labelme_io.labelme_json_of_image(path)),
            detail=locals().get("detail", ""),
            measurement=measurement_payload,
        )
