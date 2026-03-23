"""
algorithm_controller.py

算法参数管理 + 模型加载 + 训练 + 单图预测，零 UI 依赖。

职责：
  - 持有 ProductRuntimeParams（算法名、score_mode、margin、topk、传统模型字典）
  - load_params / save_params
  - load_model_for_algorithm → 返回 (model | None, 状态文字)
  - train → 返回 TrainResult
  - predict_image → 返回预测结果 dict（供 MainWindow 填表 / 记 CSV）

不负责：
  - 任何 Qt Widget 操作
  - 定位（line2dup）逻辑（由调用方传入 labels / roi 等）
  - CSV 写入 / test_log（由调用方处理返回值）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from algorithms.traditional import (
    TRADITIONAL_ALGORITHMS,
    TraditionalThresholdModel,
    compute_roi_metrics,
    is_traditional_algorithm,
    metric_value,
    train_threshold_model,
)
from infrastructure.product_params import (
    ProductRuntimeParams,
    load_product_params,
    save_product_params,
)
import algorithms.proxy as qr_core


SUPPORTED_EMBEDDING_ALGORITHMS = ["efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large"]
SUPPORTED_ALGORITHMS = SUPPORTED_EMBEDDING_ALGORITHMS + TRADITIONAL_ALGORITHMS
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

    def to_dict(self) -> Dict[str, Any]:
        return {
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
        }


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
        self._feat_net_cache: Dict[Tuple[str, str], Any] = {}

    # ------------------------------------------------------------------
    # 参数持久化
    # ------------------------------------------------------------------

    def load_params(self, path: str) -> None:
        """从 product_params.json 加载；不存在则使用默认值。"""
        self.product_params = load_product_params(path)
        alg = str(self.product_params.algorithm or "").strip()
        if alg and alg not in SUPPORTED_ALGORITHMS:
            self.product_params.algorithm = SUPPORTED_EMBEDDING_ALGORITHMS[0]
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
        normalized = str(algorithm or str(self.product_params.algorithm or "")).strip()
        if not normalized:
            return False
        return not is_traditional_algorithm(normalized)

    def embedding_model_path(self, algorithm: str, product_dir: str) -> str:
        return os.path.join(product_dir, f"register_model_{algorithm}.npz")

    def get_feat_net(self, backbone: str, device: Optional[str] = None) -> Any:
        normalized_backbone = str(backbone or "").strip()
        if not normalized_backbone:
            raise ValueError("backbone is required")
        normalized_device = str(device or qr_core.get_device()).strip() or "cpu"
        cache_key = (normalized_backbone, normalized_device)
        feat_net = self._feat_net_cache.get(cache_key)
        if feat_net is not None:
            return feat_net
        feat_net, _ = qr_core.load_backbone(normalized_backbone, device=normalized_device)
        self._feat_net_cache[cache_key] = feat_net
        return feat_net

    def clear_feat_net_cache(self) -> None:
        self._feat_net_cache.clear()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_model_for_algorithm(
        self, algorithm: str, product_dir: str
    ) -> Tuple[Any, str]:
        """
        加载指定算法的模型。
        返回 (model_or_None, status_message)。
        调用方负责把 status_message 写进 lbl_status。
        """
        algorithm = str(algorithm or "").strip()
        if not algorithm:
            self.model = None
            return None, "状态：请选择工具"
        if not self.is_embedding_algorithm(algorithm):
            self.model = None
            return None, f"状态：当前算法={algorithm}，使用传统阈值方法"

        model_file = self.embedding_model_path(algorithm, product_dir)
        if not os.path.exists(model_file):
            self.model = None
            return None, f"状态：{algorithm} 模型未训练"

        model = qr_core.load_register_model_npz(model_file)
        model.score_mode = self.product_params.score_mode
        model.margin = float(self.product_params.margin)
        model.topk = int(self.product_params.topk)
        self.model = model
        return model, (
            f"状态：已加载模型  algorithm={algorithm}  mode={model.score_mode}  "
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
    ) -> TrainResult:
        """
        训练 embedding 或传统阈值模型。
        成功返回 TrainResult；失败抛出异常（调用方负责捕获并弹 QMessageBox）。
        """
        mode = self.product_params.score_mode
        margin = float(self.product_params.margin)
        topk = int(self.product_params.topk)

        if self.is_embedding_algorithm(algorithm):
            model = qr_core.train_register_model(
                ok_files,
                ng_files,
                backbone=algorithm,
                score_mode=mode,
                margin=margin,
                topk=topk,
                label_name=label_names[0],
                label_names=label_names,
            )
            saved_path = self.embedding_model_path(algorithm, product_dir)
            qr_core.save_register_model_npz(model, saved_path)
            self.model = model
            return TrainResult(
                algorithm=algorithm,
                is_embedding=True,
                status_message=(
                    f"状态：已训练  algorithm={algorithm}  mode={mode}  "
                    f"margin={margin:.4f}  topk={topk}"
                ),
                dialog_message="OK/NG 注册完成，可以开始测试。",
                model=model,
                saved_model_path=saved_path,
            )
        else:
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
                        qr_core.labelme_json_of_image(str(sample.get("file_path", "")))
                    ),
                })
            self.product_params.traditional_models[algorithm] = threshold_model.to_dict()
            self.model = None
            return TrainResult(
                algorithm=algorithm,
                is_embedding=False,
                status_message=(
                    f"状态：已训练传统算法  algorithm={algorithm}  "
                    f"threshold={threshold_model.threshold:.4f}  "
                    f"ok_when={threshold_model.ok_when}  acc={threshold_model.accuracy:.4f}"
                ),
                dialog_message="传统算法阈值模型训练完成，可以开始测试。",
                traditional_model_dict=threshold_model.to_dict(),
                result_rows=result_rows,
            )

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
    ) -> PredictResult:
        """
        对单张已定位好 ROI 的图做推理。
        调用方负责：定位（line2dup）/ 获取 labels / 传入 roi。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        algorithm = str(self.product_params.algorithm or "")
        if not algorithm.strip():
            raise RuntimeError("请先选择工具")
        total_t0 = time.perf_counter()

        if self.is_embedding_algorithm(algorithm):
            if self.model is None:
                raise RuntimeError(f"algorithm model not loaded: {algorithm}")
            self.apply_params_to_model()

            if len(labels) == 1 and roi is None:
                j = qr_core.labelme_json_of_image(path)
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
        else:
            model_dict = self.product_params.traditional_models.get(algorithm)
            if not isinstance(model_dict, dict):
                raise RuntimeError(f"传统算法 {algorithm} 尚未训练")
            threshold_model = TraditionalThresholdModel.from_dict(model_dict)
            metrics = compute_roi_metrics(path, preferred_label=threshold_model.roi_label or labels[0])
            value = metric_value(metrics, algorithm)
            pred, diff = threshold_model.predict(value)
            sim_ok = None
            sim_ng = None
            threshold = float(threshold_model.threshold)

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
            json_name=os.path.basename(qr_core.labelme_json_of_image(path)),
        )
