from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List

from common.path_utils import product_relative_path, resolve_product_path
from common.safe_io import atomic_write_json, load_json_with_backup


def normalize_group_id(value: object) -> str:
    """The reserved global binding has one representation; other ids stay explicit."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("定位组 ID 必须为字符串。")
    group_id = value.strip()
    return "" if group_id.lower() == "global" else group_id


def _group_number(value: object, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"定位组 {field_name} 必须为有限数值。")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"定位组 {field_name} 必须为有限数值。") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"定位组 {field_name} 必须位于 {minimum:g} 到 {maximum:g} 之间。")
    return number


def _group_rectangle(value: object) -> List[List[float]]:
    """Keep an empty editor draft, or normalize an axis-aligned rectangle.

    Size and image-boundary checks belong to execution: an unfinished anchor
    must remain editable without silently changing its ROI bindings.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) not in (0, 2, 4):
        raise ValueError("定位锚点必须为空草稿、两个对角点或四个轴对齐矩形顶点。")
    if not value:
        return []
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("定位锚点的每个坐标必须包含 x、y 两个数值。")
        if any(isinstance(coordinate, bool) for coordinate in point):
            raise ValueError("定位锚点坐标必须为有限数值。")
        try:
            converted = [float(coordinate) for coordinate in point]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("定位锚点坐标必须为有限数值。") from exc
        if not all(math.isfinite(coordinate) for coordinate in converted):
            raise ValueError("定位锚点坐标必须为有限数值。")
        points.append(converted)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if len(points) == 4 and set(map(tuple, points)) != {
        (left, top), (right, top), (right, bottom), (left, bottom)
    }:
        raise ValueError("定位锚点的四个顶点必须组成轴对齐矩形。")
    return [[left, top], [right, bottom]]


def normalize_localization_groups(value: object) -> List[Dict[str, Any]]:
    """Normalize supported group settings, without activating or repairing drafts.

    Groups contain coordinates in the shared reference image, not arbitrary
    image/model paths. Unrecognized per-group fields are not executed or saved.
    Dangling bindings, empty anchors and needs_rebuild remain visible to the
    editor; the locator must fail explicitly if a bound group is unusable.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("定位组配置必须为列表。")
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("每个定位组必须是包含 ID 和锚点配置的对象。")
        group_id = normalize_group_id(item.get("id"))
        if not group_id:
            raise ValueError("局部定位组必须具有非空 ID，且不能使用保留名称 global。")
        if group_id in seen:
            raise ValueError(f"定位组 ID 重复：{group_id}")
        seen.add(group_id)
        radius = _group_number(item.get("search_radius", 64), "搜索半径", 1, 512)
        if radius != int(radius):
            raise ValueError("定位组搜索半径必须为整数像素。")
        needs_rebuild = item.get("needs_rebuild", False)
        if not isinstance(needs_rebuild, bool):
            raise ValueError("定位组 needs_rebuild 必须为布尔值。")
        name = item.get("name", group_id)
        if name is None:
            name = group_id
        if not isinstance(name, str):
            raise ValueError("定位组名称必须为字符串。")
        result.append({
            "id": group_id,
            "name": name.strip() or group_id,
            "points": _group_rectangle(item.get("points", [])),
            "search_radius": int(radius),
            "min_score": _group_number(item.get("min_score", 0.65), "最低匹配分数", 0, 1),
            "min_peak_margin": _group_number(item.get("min_peak_margin", 0.03), "候选分差", 0, 1),
            "needs_rebuild": needs_rebuild,
        })
    return result


def has_bound_localization_groups(recipe: "ShapeRecipe") -> bool:
    """Include missing/stale groups so consumers cannot bypass their failure."""
    return any(
        bool(normalize_group_id(region.get("group_id")))
        for region in (recipe.reference_regions or [])
        if isinstance(region, dict)
    )


def _schema_version(value: object) -> int:
    if value is None or value == "":
        return 1
    if isinstance(value, bool) or value not in (1, 2, "1", "2"):
        raise ValueError("不支持此形状配方版本；请使用支持该配方的程序。")
    return int(value)


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
    localization_groups: List[Dict[str, Any]] = field(default_factory=list)

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
                    "group_id": normalize_group_id(region.get("group_id")),
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
            schema_version=_schema_version(data.get("schema_version", 1)),
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
            localization_groups=normalize_localization_groups(data.get("localization_groups")),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["localization_groups"] = normalize_localization_groups(self.localization_groups)
        if payload["reference_regions"] is not None:
            for region in payload["reference_regions"]:
                if isinstance(region, dict):
                    region["group_id"] = normalize_group_id(region.get("group_id"))
        payload["schema_version"] = max(
            _schema_version(self.schema_version),
            2 if payload["localization_groups"] or has_bound_localization_groups(self) else 1,
        )
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


__all__ = [
    "ShapeRecipe", "has_bound_localization_groups", "load_recipe", "normalize_follow_mode",
    "normalize_group_id", "normalize_localization_groups", "save_recipe",
]
