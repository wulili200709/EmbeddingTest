from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from algorithms.registry import learning_backbone_storage_code, storage_code_backbone
from safe_io import atomic_write_json, load_json_with_backup


@dataclass
class ProductRuntimeParams:
    algorithm: str = ""
    learning_backbone: str = ""
    score_mode: str = "proto"
    margin: float = 0.02
    topk: int = 3
    traditional_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductRuntimeParams":
        traditional_models = data.get("traditional_models", {})
        if not isinstance(traditional_models, dict):
            traditional_models = {}
        return cls(
            algorithm=storage_code_backbone(data.get("algorithm", "")),
            learning_backbone=storage_code_backbone(data.get("learning_backbone", "")),
            score_mode=str(data.get("score_mode", "proto")),
            margin=float(data.get("margin", 0.02)),
            topk=int(data.get("topk", 3)),
            traditional_models={
                str(key): value
                for key, value in traditional_models.items()
                if isinstance(value, dict)
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["algorithm"] = learning_backbone_storage_code(payload.get("algorithm", ""))
        payload["learning_backbone"] = learning_backbone_storage_code(payload.get("learning_backbone", ""))
        return payload


def load_product_params(path: str) -> ProductRuntimeParams:
    p = Path(path)
    data = load_json_with_backup(p, default=None)
    if data is None:
        return ProductRuntimeParams()
    if not isinstance(data, dict):
        raise ValueError(f"Invalid product params: {p}")
    return ProductRuntimeParams.from_dict(data)


def save_product_params(params: ProductRuntimeParams, path: str) -> None:
    atomic_write_json(path, params.to_dict(), ensure_ascii=False, indent=2)


__all__ = ["ProductRuntimeParams", "load_product_params", "save_product_params"]
