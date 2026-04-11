"""Per-item runtime parameter helpers for learning/anomaly tools."""

from __future__ import annotations

from typing import Any


SUPPORTED_SCORE_MODES = {"proto", "topk"}
DEFAULT_SCORE_MODE = "proto"
DEFAULT_MARGIN = 0.02
DEFAULT_TOPK = 3


def _coerce_margin(value: object, default: float = DEFAULT_MARGIN) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_topk(value: object, default: int = DEFAULT_TOPK) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(default))


def _coerce_score_mode(value: object, *, anomaly: bool, default: str = DEFAULT_SCORE_MODE) -> str:
    if anomaly:
        return "topk"
    normalized = str(value or "").strip()
    if normalized in SUPPORTED_SCORE_MODES:
        return normalized
    fallback = str(default or DEFAULT_SCORE_MODE).strip()
    if fallback in SUPPORTED_SCORE_MODES:
        return fallback
    return DEFAULT_SCORE_MODE


def _product_param_defaults(tool_page) -> dict[str, Any]:
    product_params = getattr(getattr(tool_page, "algo", None), "product_params", None)
    return {
        "score_mode": _coerce_score_mode(
            getattr(product_params, "score_mode", DEFAULT_SCORE_MODE),
            anomaly=False,
            default=DEFAULT_SCORE_MODE,
        ),
        "margin": _coerce_margin(getattr(product_params, "margin", DEFAULT_MARGIN), DEFAULT_MARGIN),
        "topk": _coerce_topk(getattr(product_params, "topk", DEFAULT_TOPK), DEFAULT_TOPK),
    }


def find_inspection_item_by_model_key(tool_page, model_key: object):
    normalized = str(model_key or "").strip()
    if not normalized:
        return None
    algo = getattr(tool_page, "algo", None)
    normalizer = getattr(algo, "tool_model_key", None)
    if callable(normalizer):
        normalized = str(normalizer(normalized) or "").strip()
    task_group_match = None
    for item in list(getattr(tool_page, "inspection_items", []) or []):
        item_key = str(getattr(item, "effective_model_key", getattr(item, "model_key", "")) or "").strip()
        if callable(normalizer):
            item_key = str(normalizer(item_key) or "").strip()
        if item_key == normalized:
            return item
        task_key = str(getattr(item, "task_model_key", "") or "").strip()
        if callable(normalizer):
            task_key = str(normalizer(task_key) or "").strip()
        if task_key == normalized and task_group_match is None:
            task_group_match = item
    return task_group_match


def _resolve_algorithm(tool_page, inspection_item=None, *, algorithm: object = None) -> str:
    algo = getattr(tool_page, "algo", None)
    if inspection_item is not None:
        learning_checker = getattr(algo, "is_learning_tool", None)
        if callable(learning_checker) and learning_checker(getattr(inspection_item, "algorithm_code", "")):
            current_backbone = getattr(algo, "current_learning_backbone", None)
            if callable(current_backbone):
                return str(current_backbone() or "").strip()
        resolver = getattr(algo, "resolve_tool_algorithm", None)
        if callable(resolver):
            return str(resolver(getattr(inspection_item, "algorithm_code", "")) or "").strip()
        return str(getattr(inspection_item, "algorithm_code", "") or "").strip()

    normalized = str(algorithm or "").strip()
    if not normalized:
        current_algorithm = getattr(tool_page, "current_algorithm", None)
        if callable(current_algorithm):
            normalized = str(current_algorithm() or "").strip()
    if not normalized:
        normalized = str(getattr(getattr(algo, "product_params", None), "algorithm", "") or "").strip()
    if not normalized:
        return ""

    learning_resolver = getattr(algo, "resolve_learning_algorithm", None)
    if callable(learning_resolver):
        resolved = str(learning_resolver(normalized) or "").strip()
        if resolved:
            return resolved
    tool_resolver = getattr(algo, "resolve_tool_algorithm", None)
    if callable(tool_resolver):
        return str(tool_resolver(normalized) or "").strip()
    return normalized


def _is_embedding_item(tool_page, inspection_item, resolved_algorithm: str) -> bool:
    algo = getattr(tool_page, "algo", None)
    if inspection_item is not None:
        learning_checker = getattr(algo, "is_learning_tool", None)
        anomaly_checker = getattr(algo, "is_anomaly_tool", None)
        algorithm_code = getattr(inspection_item, "algorithm_code", "")
        if callable(learning_checker) and learning_checker(algorithm_code):
            return True
        if callable(anomaly_checker) and (
            anomaly_checker(algorithm_code) or anomaly_checker(resolved_algorithm)
        ):
            return True
    embedding_checker = getattr(algo, "is_embedding_algorithm", None)
    if callable(embedding_checker):
        return bool(embedding_checker(resolved_algorithm))
    return False


def _is_anomaly_algorithm(tool_page, inspection_item, resolved_algorithm: str) -> bool:
    algo = getattr(tool_page, "algo", None)
    if inspection_item is not None:
        anomaly_tool_checker = getattr(algo, "is_anomaly_tool", None)
        algorithm_code = getattr(inspection_item, "algorithm_code", "")
        if callable(anomaly_tool_checker) and (
            anomaly_tool_checker(algorithm_code) or anomaly_tool_checker(resolved_algorithm)
        ):
            return True
    anomaly_checker = getattr(algo, "is_anomaly_algorithm", None)
    if callable(anomaly_checker):
        return bool(anomaly_checker(resolved_algorithm))
    anomaly_tool_checker = getattr(algo, "is_anomaly_tool", None)
    if callable(anomaly_tool_checker):
        return bool(anomaly_tool_checker(resolved_algorithm))
    return False


def effective_item_runtime_params(tool_page, inspection_item=None, *, algorithm: object = None) -> dict[str, Any]:
    defaults = _product_param_defaults(tool_page)
    resolved_algorithm = _resolve_algorithm(tool_page, inspection_item, algorithm=algorithm)
    embedding = _is_embedding_item(tool_page, inspection_item, resolved_algorithm)
    anomaly = _is_anomaly_algorithm(tool_page, inspection_item, resolved_algorithm)
    params = dict(getattr(inspection_item, "params", {}) or {}) if inspection_item is not None else {}
    return {
        "algorithm": resolved_algorithm,
        "embedding": embedding,
        "anomaly": anomaly,
        "score_mode": _coerce_score_mode(
            params.get("score_mode", defaults["score_mode"]),
            anomaly=anomaly,
            default=defaults["score_mode"],
        ),
        "margin": _coerce_margin(params.get("margin", defaults["margin"]), defaults["margin"]),
        "topk": _coerce_topk(params.get("topk", defaults["topk"]), defaults["topk"]),
    }


def current_item_runtime_params_from_ui(tool_page, inspection_item=None, *, algorithm: object = None) -> dict[str, Any]:
    current = effective_item_runtime_params(tool_page, inspection_item, algorithm=algorithm)
    margin_widget = getattr(tool_page, "spin_margin", None)
    topk_widget = getattr(tool_page, "spin_topk", None)
    score_mode_widget = getattr(tool_page, "cmb_mode", None)
    return {
        "algorithm": current["algorithm"],
        "embedding": current["embedding"],
        "anomaly": current["anomaly"],
        "score_mode": _coerce_score_mode(
            score_mode_widget.currentText() if score_mode_widget is not None else current["score_mode"],
            anomaly=bool(current["anomaly"]),
            default=str(current["score_mode"]),
        ),
        "margin": _coerce_margin(
            margin_widget.value() if margin_widget is not None else current["margin"],
            float(current["margin"]),
        ),
        "topk": _coerce_topk(
            topk_widget.value() if topk_widget is not None else current["topk"],
            int(current["topk"]),
        ),
    }


def sync_item_runtime_params_to_controller(tool_page, inspection_item=None, *, algorithm: object = None) -> dict[str, Any]:
    runtime_params = effective_item_runtime_params(tool_page, inspection_item, algorithm=algorithm)
    algo = getattr(tool_page, "algo", None)
    product_params = getattr(algo, "product_params", None)
    if product_params is not None:
        resolved_algorithm = str(runtime_params.get("algorithm", "") or "").strip()
        if resolved_algorithm:
            product_params.algorithm = resolved_algorithm
        if bool(runtime_params.get("embedding", False)):
            product_params.score_mode = str(runtime_params["score_mode"])
            product_params.margin = float(runtime_params["margin"])
            product_params.topk = int(runtime_params["topk"])
    return runtime_params


def store_item_runtime_params(
    tool_page,
    inspection_item,
    *,
    algorithm: object = None,
    score_mode: object = None,
    margin: object = None,
    topk: object = None,
) -> bool:
    if inspection_item is None:
        return False
    current = effective_item_runtime_params(tool_page, inspection_item, algorithm=algorithm)
    if not bool(current.get("embedding", False)):
        return False
    params = dict(getattr(inspection_item, "params", {}) or {})
    updated = dict(params)
    updated["score_mode"] = _coerce_score_mode(
        score_mode if score_mode is not None else current["score_mode"],
        anomaly=bool(current["anomaly"]),
        default=str(current["score_mode"]),
    )
    updated["margin"] = _coerce_margin(
        margin if margin is not None else current["margin"],
        float(current["margin"]),
    )
    updated["topk"] = _coerce_topk(
        topk if topk is not None else current["topk"],
        int(current["topk"]),
    )
    if updated == params:
        return False
    inspection_item.params = updated
    return True
