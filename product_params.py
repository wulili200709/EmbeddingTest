from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ProductRuntimeParams:
    algorithm: str = "efficientnet_b0"
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
            algorithm=str(data.get("algorithm", "efficientnet_b0")),
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
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid product params: {p}")
    return ProductRuntimeParams.from_dict(data)


def save_product_params(params: ProductRuntimeParams, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["ProductRuntimeParams", "load_product_params", "save_product_params"]
