from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from path_utils import product_relative_path, resolve_product_path


def _levels_text_from_value(value: Any, default: str = "4,8") -> str:
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            try:
                parts.append(str(int(item)))
            except (TypeError, ValueError):
                text = str(item or "").strip()
                if text:
                    parts.append(text)
        return ",".join(parts) or default
    text = str(value or "").strip()
    return text or default


@dataclass
class Line2DupRecipe:
    model_path: str = ""
    reference_image: str = ""
    class_id: str = ""
    template_levels: str = "4,8"
    template_num_features: int = 128
    template_weak_threshold: float = 30.0
    template_strong_threshold: float = 60.0
    template_angle_start: float = -5.0
    template_angle_end: float = 10.0
    template_angle_step: float = 1.0
    template_scale_start: float = 1.0
    template_scale_end: float = 1.0
    template_scale_step: float = 0.05
    backend: str = "original"
    threshold: float = 50.0
    auto_threshold_sweep: bool = False
    threshold_sweep_step: int = 10
    threshold_sweep_min: int = 20
    nms_iou: float = 0.3
    topk: int = 1
    array_count: int = 1
    array_pitch_x: float = 0.0
    array_pitch_y: float = 0.0
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
    def from_dict(cls, data: Dict[str, Any]) -> "Line2DupRecipe":
        return cls(
            model_path=str(data.get("model_path", "")),
            reference_image=str(data.get("reference_image", "")),
            class_id=str(data.get("class_id", "")),
            template_levels=_levels_text_from_value(data.get("template_levels", data.get("T_levels", "4,8"))),
            template_num_features=int(data.get("template_num_features", data.get("num_features", 128))),
            template_weak_threshold=float(data.get("template_weak_threshold", data.get("weak_threshold", 30.0))),
            template_strong_threshold=float(data.get("template_strong_threshold", data.get("strong_threshold", 60.0))),
            template_angle_start=float(data.get("template_angle_start", -5.0)),
            template_angle_end=float(data.get("template_angle_end", 10.0)),
            template_angle_step=float(data.get("template_angle_step", 1.0)),
            template_scale_start=float(data.get("template_scale_start", 1.0)),
            template_scale_end=float(data.get("template_scale_end", 1.0)),
            template_scale_step=float(data.get("template_scale_step", 0.05)),
            backend=str(data.get("backend", "original")),
            threshold=float(data.get("threshold", 50.0)),
            auto_threshold_sweep=bool(data.get("auto_threshold_sweep", False)),
            threshold_sweep_step=int(data.get("threshold_sweep_step", 10)),
            threshold_sweep_min=int(data.get("threshold_sweep_min", 20)),
            nms_iou=float(data.get("nms_iou", 0.3)),
            topk=max(1, int(data.get("topk", data.get("array_count", 1)))),
            array_count=max(1, int(data.get("array_count", data.get("topk", 1)))),
            array_pitch_x=float(data.get("array_pitch_x", 0.0)),
            array_pitch_y=float(data.get("array_pitch_y", 0.0)),
            crop_stride=int(data.get("crop_stride", 0)),
            use_scene_mask=bool(data.get("use_scene_mask", False)),
            follow_mode=str(data.get("follow_mode", "affine_roi")),
            output_label=str(data.get("output_label", "roi")),
            reference_label=str(data.get("reference_label", "roi")),
            reference_shape_type=str(data.get("reference_shape_type", "")),
            reference_points=[
                [float(pt[0]), float(pt[1])]
                for pt in data.get("reference_points", []) or []
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ] or None,
            reference_regions=[
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
                for region in data.get("reference_regions", []) or []
                if isinstance(region, dict)
            ] or None,
            search_shape_type=str(data.get("search_shape_type", "")),
            search_points=[
                [float(pt[0]), float(pt[1])]
                for pt in data.get("search_points", []) or []
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ] or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recipe_array_count(recipe: Line2DupRecipe | None) -> int:
    # Array ROI generation is temporarily disabled. Always expose a single
    # logical instance even if old recipes still contain array metadata.
    return 1


def recipe_array_pitch(recipe: Line2DupRecipe | None) -> tuple[float, float]:
    return 0.0, 0.0


def normalized_reference_regions(recipe: Line2DupRecipe | None) -> List[Dict[str, Any]]:
    if recipe is None:
        return []

    normalized: List[Dict[str, Any]] = []
    for region in getattr(recipe, "reference_regions", []) or []:
        if not isinstance(region, dict):
            continue
        reference_label = str(region.get("reference_label") or region.get("label") or "").strip()
        output_label = str(region.get("output_label") or reference_label or region.get("label") or "").strip()
        display_name = str(
            region.get("display_name")
            or region.get("name")
            or output_label
            or reference_label
            or region.get("label")
            or ""
        ).strip() or output_label or reference_label
        shape_type = str(region.get("shape_type", "rectangle") or "rectangle").strip() or "rectangle"
        points = [
            [float(pt[0]), float(pt[1])]
            for pt in region.get("points", []) or []
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if not output_label or len(points) < 2:
            continue
        normalized.append(
            {
                "reference_label": reference_label or output_label,
                "output_label": output_label,
                "display_name": display_name or output_label,
                "shape_type": shape_type,
                "points": points,
            }
        )
    return normalized


def format_array_roi_label(base_label: str, instance_index: int, array_count: int) -> str:
    label = str(base_label or "roi").strip() or "roi"
    count = max(1, int(array_count))
    if count <= 1:
        return label
    width = max(2, len(str(count)))
    return f"{label}__{int(instance_index) + 1:0{width}d}"


def format_array_display_name(base_display_name: str, instance_index: int, array_count: int) -> str:
    name = str(base_display_name or "roi").strip() or "roi"
    count = max(1, int(array_count))
    if count <= 1:
        return name
    return f"{name} #{int(instance_index) + 1}"


def expanded_reference_region_specs(recipe: Line2DupRecipe | None) -> List[Dict[str, Any]]:
    base_regions = normalized_reference_regions(recipe)
    if not base_regions:
        default_label = "roi"
        default_name = "roi"
        if recipe is not None:
            default_label = str(getattr(recipe, "output_label", "") or getattr(recipe, "reference_label", "") or "roi").strip() or "roi"
            default_name = default_label
        return [
            {
                "roi_label": default_label,
                "display_name": default_name,
                "base_roi_label": default_label,
                "base_display_name": default_name,
                "instance_index": 0,
                "array_count": 1,
            }
        ]

    specs: List[Dict[str, Any]] = []
    for region in base_regions:
        base_label = str(region.get("output_label") or region.get("reference_label") or "roi").strip() or "roi"
        base_name = str(region.get("display_name") or base_label).strip() or base_label
        specs.append(
            {
                "roi_label": base_label,
                "display_name": base_name,
                "base_roi_label": base_label,
                "base_display_name": base_name,
                "instance_index": 0,
                "array_count": 1,
            }
        )
    return specs


def _product_dir_for_recipe_file(path: Path) -> str:
    if path.parent.parent.name.lower() == "line2dup":
        return str(path.parent.parent.parent)
    return str(path.parent)


def load_recipe(path: str) -> Line2DupRecipe:
    p = Path(path)
    if not p.exists():
        return Line2DupRecipe(model_path=str(p.with_name("line2dup_model.json")))
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid line2dup recipe: {p}")
    recipe = Line2DupRecipe.from_dict(data)
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
        recipe.model_path = str(p.with_name("line2dup_model.json"))
    return recipe


def save_recipe(recipe: Line2DupRecipe, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = recipe.to_dict()
    base_dir = str(p.parent)
    payload["model_path"] = product_relative_path(payload.get("model_path", ""), base_dir=base_dir)
    payload["reference_image"] = product_relative_path(payload.get("reference_image", ""), base_dir=base_dir)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "Line2DupRecipe",
    "expanded_reference_region_specs",
    "format_array_display_name",
    "format_array_roi_label",
    "load_recipe",
    "normalized_reference_regions",
    "recipe_array_count",
    "recipe_array_pitch",
    "save_recipe",
]
