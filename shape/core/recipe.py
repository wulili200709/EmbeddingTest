from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from common.path_utils import product_relative_path, resolve_product_path
from common.safe_io import atomic_write_json, load_json_with_backup


def normalize_follow_mode(
    value: object,
    reference_regions: List[Dict[str, Any]] | None = None,
    reference_points: List[List[float]] | None = None,
) -> str:
    """Keep explicit reference ROIs on the affine-following path."""
    mode = str(value or "").strip().lower()
    if reference_regions or reference_points:
        return "affine_roi"
    return "match_bbox" if mode == "match_bbox" else "affine_roi"


@dataclass
class ShapeRecipe:
    schema_version: int = 1
    model_path: str = ""
    reference_image: str = ""
    class_id: str = ""
    backend: str = "original"
    threshold: float = 50.0
    auto_threshold_sweep: bool = False
    threshold_sweep_step: int = 10
    threshold_sweep_min: int = 20
    nms_iou: float = 0.3
    topk: int = 1
    crop_stride: int = 0
    use_scene_mask: bool = False
    follow_mode: str = "affine_roi"
    output_label: str = "roi"
    reference_label: str = "roi"
    reference_shape_type: str = ""
    reference_points: List[List[float]] | None = None
    reference_regions: List[Dict[str, Any]] | None = None
    search_shape_type: str = ""
    search_points: List[List[float]] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapeRecipe":
        raw_reference_regions = data.get("reference_regions") if "reference_regions" in data else None
        if isinstance(raw_reference_regions, list):
            reference_regions = [
                {
                    "reference_label": str(region.get("reference_label") or region.get("label") or ""),
                    "output_label": str(region.get("output_label") or region.get("reference_label") or region.get("label") or ""),
                    "display_name": str(
                        region.get("display_name")
                        or region.get("name")
                        or region.get("output_label")
                        or region.get("reference_label")
                        or region.get("label")
                        or ""
                    ),
                    "shape_type": str(region.get("shape_type", "rectangle")),
                    "points": [
                        [float(pt[0]), float(pt[1])]
                        for pt in region.get("points", []) or []
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2
                    ],
                }
                for region in raw_reference_regions
                if isinstance(region, dict)
            ]
        else:
            reference_regions = None
        reference_points = [
            [float(pt[0]), float(pt[1])]
            for pt in data.get("reference_points", []) or []
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ] or None
        return cls(
            schema_version=int(data.get("schema_version", 1) or 1),
            model_path=str(data.get("model_path", "")),
            reference_image=str(data.get("reference_image", "")),
            class_id=str(data.get("class_id", "")),
            backend=str(data.get("backend", "original")),
            threshold=float(data.get("threshold", 50.0)),
            auto_threshold_sweep=bool(data.get("auto_threshold_sweep", False)),
            threshold_sweep_step=int(data.get("threshold_sweep_step", 10)),
            threshold_sweep_min=int(data.get("threshold_sweep_min", 20)),
            nms_iou=float(data.get("nms_iou", 0.3)),
            topk=int(data.get("topk", 1)),
            crop_stride=int(data.get("crop_stride", 0)),
            use_scene_mask=bool(data.get("use_scene_mask", False)),
            follow_mode=normalize_follow_mode(
                data.get("follow_mode", "affine_roi"),
                reference_regions,
                reference_points,
            ),
            output_label=str(data.get("output_label", "roi")),
            reference_label=str(data.get("reference_label", "roi")),
            reference_shape_type=str(data.get("reference_shape_type", "")),
            reference_points=reference_points,
            reference_regions=reference_regions,
            search_shape_type=str(data.get("search_shape_type", "")),
            search_points=[
                [float(pt[0]), float(pt[1])]
                for pt in data.get("search_points", []) or []
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ] or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["follow_mode"] = normalize_follow_mode(
            self.follow_mode,
            self.reference_regions,
            self.reference_points,
        )
        return payload


def _product_dir_for_recipe_file(path: Path) -> str:
    if path.parent.parent.name.lower() in {"shape", "line2dup"}:
        return str(path.parent.parent.parent)
    return str(path.parent)


def load_recipe(path: str) -> ShapeRecipe:
    p = Path(path)
    data = load_json_with_backup(p, default=None)
    if data is None:
        return ShapeRecipe(model_path=str(p.with_name("model.json")))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid shape recipe: {p}")
    recipe = ShapeRecipe.from_dict(data)
    base_dir = str(p.parent)
    anchor_dir = _product_dir_for_recipe_file(p)
    recipe.model_path = resolve_product_path(recipe.model_path, base_dir=base_dir, anchor_dir=anchor_dir, prefer_existing=False)
    recipe.reference_image = resolve_product_path(
        recipe.reference_image,
        base_dir=base_dir,
        anchor_dir=anchor_dir,
        prefer_existing=True,
    )
    if not recipe.model_path:
        recipe.model_path = str(p.with_name("model.json"))
    return recipe


def save_recipe(recipe: ShapeRecipe, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = recipe.to_dict()
    base_dir = str(p.parent)
    payload["model_path"] = product_relative_path(payload.get("model_path", ""), base_dir=base_dir)
    payload["reference_image"] = product_relative_path(payload.get("reference_image", ""), base_dir=base_dir)
    atomic_write_json(p, payload, ensure_ascii=False, indent=2)


__all__ = ["ShapeRecipe", "load_recipe", "normalize_follow_mode", "save_recipe"]
