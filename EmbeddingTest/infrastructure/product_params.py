from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from algorithms.registry import normalize_learning_backbone
from infrastructure.json_backup import load_json_with_recovery, write_json_with_backup


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
        algorithm = normalize_learning_backbone(str(data.get("algorithm", "")).strip())
        learning_backbone = normalize_learning_backbone(str(data.get("learning_backbone", "")).strip())
        return cls(
            algorithm=algorithm,
            learning_backbone=learning_backbone,
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
        return asdict(self)


def load_product_params(path: str) -> ProductRuntimeParams:
    p = Path(path)
    if not p.exists():
        return ProductRuntimeParams()
    data = load_json_with_recovery(p, expected_type=dict)
    return ProductRuntimeParams.from_dict(data)


def save_product_params(params: ProductRuntimeParams, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_with_backup(p, params.to_dict(), expected_type=dict)


__all__ = ["ProductRuntimeParams", "load_product_params", "save_product_params"]
